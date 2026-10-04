# SingleParticleBasisFunctionScalarInd

| | |
|---|---|
| Source | `instructions/compute.py:1125-1299` (class at 1126), bases `TPEquivariantInstruction`, `LORAInstructionMixin` (`instructions/base.py:492`, `393`) |
| Family | embedding and single-particle basis |
| Used in | both 2L yamls: `A` (radial `R`) and `B0` (radial `R1`), both with `indicator Z`, `indicator_l_depend false`, `sum_neighbors true`, `avg_n_neigh 39.773345702648434`, `angular Y`, no `lmax` (omat) or `lmax null` (large_base) |
| Reads | the radial and angular entries, the indicator table; `ind_i`, `mu_j`, `atomic_mu_i` (`compute.py:1134-1138`) |
| Writes | `<name>` `[n_atoms, n_rad_max, (lmax + 1)**2]` (`[lm, n_atoms, n_rad_max]` with `lm_first`) |
| Variables | one `DenseLayer` weight when there is an indicator |

The ACE single-particle basis with a scalar indicator: the radial function, the spherical harmonic and a
learned per-element weight of the neighbour are multiplied bond by bond, summed over the neighbours of each atom
and divided by the average neighbour count.

## 1. Constructor arguments

`compute.py:1140-1214`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `name` | required | any unique `str` | Key of the output. |
| `radial` | required | an `MLPRadialFunction` or `_v2` instruction (an object, stored as a name) | Needs `n_rad_max`, `lmax` and `l_tile`. |
| `angular` | required | a `SphericalHarmonic` instruction | Needs `lmax`, `coupling_meta_data`, `coupling_origin`. |
| `indicator` | `None` | a `ScalarChemicalEmbedding` or `None` | `None`: no species weight. Per-atom indicators (`is_per_atom`) do not exist (see the sheet of the embedding). |
| `indicator_l_depend` | `False` | bool | `True`: the species weight depends on `l` as well (`n_out = n_rad_max * (lmax + 1)` instead of `n_rad_max`). Ignored without an indicator. |
| `sum_neighbors` | `True` | bool | `False` returns the per-bond tensor and applies **no** averaging (section 5). |
| `avg_n_neigh` | `1.0` | a `float`, or a `dict` element index `->` mean neighbour count | A **`float`** (an `int` raises `TypeError`, TEST6 finding 13, pinned at `tests/test_compute.py:2389`; `np.float64` is accepted because it is a `float`). A dict switches to a per-central-element factor; keys are element indices used to index a `[len(dict), 1]` array, so they must be `0 .. len(dict) - 1`; a value `<= 0` becomes `1.0` (`compute.py:1166-1176`). Both 2L yamls use one float. |
| `lora_config` | `None` | dict or `None` | Section 7. |
| `lmax` | `None` | `None` or integer `<= angular.lmax` | Highest `l` used; `None` means `angular.lmax`. A larger value fails an `assert`. The yaml stores `lmax: null` in large_base and no key in omat; both mean `angular.lmax = 4`. |
| `lm_first` | `False` | bool | Output layout (section 5). |

Further checks: `radial.lmax` has to equal `lmax` (`assert`, `compute.py:1202-1204`). Neither yaml passes any
keyword that the constructor does not read.

## 2. Derived tables

- `slice_angular = (lmax + 1)**2` when `angular.lmax > lmax`, else `None` (`compute.py:1198-1201`): the number of
  leading columns of `Y` that are used.
- `coupling_meta_data` (`compute.py:1206-1214`): the table of the angular entry, restricted to `l <= lmax` when
  sliced, with a column `symbol` added by `init_coupling_symbols` (`poly.py:166-173`). Columns `l, m, hist, parity,
  sum_of_ls`; for `Y` with `lmax = 4` it has 25 rows, `hist = ""`, `parity = (-1)**l`. This table is what the
  product instructions (`ProductFunction`, `FunctionReduceN`) read to build their own coupling tables; the twin
  produces the same table with numpy. `symbol` is bookkeeping of `simplify=True` and is not needed at inference.
  `coupling_origin` is copied from the angular entry (`None` for `SphericalHarmonic`).
- `inv_avg_n_neigh`: `1 / avg_n_neigh` for a float; for a dict an array `[len(dict), 1]` with `1 / v`, or `1` where
  `v <= 0`.
- `n_out = radial.n_rad_max` (`compute.py:1196`): the channel count downstream instructions read.

## 3. Parameters

Names, shapes and dtypes are **[I]**.

| Attribute | TF name (omat, `name = A`) | Shape | Notes |
|---|---|---|---|
| `lin_transform.w` (only with an indicator) | `A_ChemIndTransf/DenseLayer_A_ChemIndTransf__:0` | `[embedding_size, n_out]` with `n_out = n_rad_max`, or `n_rad_max * (lmax + 1)` if `indicator_l_depend`; here `[128, 32]` (omat, `A`) and `[128, 42]` (large_base, `A`) | `float_dtype`, trainable, `N(0, 1)`. No bias (`use_bias` is `False`, and `assert not lin_transform.use_bias` guards the LoRA path). The weight-decay flag is `_` because the layer is built with the default `no_weight_decay = False`. |

## 4. Runtime constants

- `inv_avg_n_neigh`, a tensor in `float_dtype` (`compute.py:1221-1224`), **not in the checkpoint**: the twin
  recomputes it from the yaml.
- `DenseLayer.norm = 1 / sqrt(embedding_size)`, a tensor in the weight dtype made in `build`
  (`functions/nn.py:295`), also not stored. The stored weight is the unscaled one.

## 5. Forward

`compute.py:1227-1287`. Let `R = data[radial]` `[n_bonds, n, lm]`, `Y = data[angular]` `[n_bonds, lm_Y]`, `n = n_rad_max`,
`lm = (lmax + 1)**2`.

```
Y   = Y[:, :slice_angular]                    (if slice_angular)
Y   = cast(Y, R.dtype)
a   = R * Y[:, None, :]                       # einsum "jnl,jl->jnl", [n_bonds, n, lm]
if indicator:
    z   = cast(data[indicator], R.dtype)       # [n_elements, emb]
    zt  = z @ (w / sqrt(emb))                  # DenseLayer, cast to the weight dtype; [n_elements, n_out]
    a   = cast(a, zt.dtype)                    # only if the dtypes differ
    if indicator_l_depend:
        zt = reshape(zt, [n_elements, n, lmax + 1]);  zt = gather(zt, radial.l_tile, axis=2)   # [n_el, n, lm]
        a  = a * gather(zt, mu_j, axis=0)                                                       # [n_bonds, n, lm]
    else:
        a  = a * gather(zt, mu_j, axis=0)[:, :, None]                                           # [n_bonds, n, 1]
if sum_neighbors:
    a = segment_sum(a, ind_i, num_segments = n_atoms)             # n_atoms = len(atomic_mu_i); [n_atoms, n, lm]
    float avg:  a = a * (1 / avg_n_neigh)
    dict avg:   a = a * gather(inv, atomic_mu_i)[:, :, None]       # factor by the element of the central atom
if lm_first:  a = transpose(a, [2, 0, 1])                         # [lm, n_atoms, n]
```

Notes:

- `n_atoms` is the length of `atomic_mu_i`, the **padded** number of atoms of the batch; padding atoms get 0.
- With `sum_neighbors = False` the result is the per-bond `a` of shape `[n_bonds, n, lm]`, **without** the
  `inv_avg_n_neigh` factor.
- With a dict `avg_n_neigh` and `local = True` TF raises `NotImplementedError`; the twin has no `local` mode
  (see the sheet of `MLPRadialFunction_v2`).
- All sums over bonds are `unsorted_segment_sum`: the order of the additions is unspecified, so two backends agree
  to rounding, not bit for bit.
- Written out for the shipped case: `A[i, n, lm] = (1/avg) * sum_{j in N(i)} R[j, n, lm] Y[j, lm] Zt[mu_j, n]`.

The formula was re-implemented in numpy (float, dict and `l`-dependent variants, no indicator, `sum_neighbors`
off, `lm_first`) and compared with TF: maximum absolute difference `8.9e-16` (float64).

## 6. Dtype and promotion

**Measured** on the two 2L yamls with float32 parameters and float64 data (energy and all instruction outputs):
`R`, `A`, `B0` are float32, `Y` is float64. The class follows the radial function: `Y` is cast **down** to the
dtype of `R`, `z` is cast to it, the weight is in the parameter dtype, and `a` is cast to the weight dtype
(`compute.py:1233-1242`). The result has the parameter dtype in a real model. TEST6 finding 1: the bond tensors
are cast to the parameter dtype, so float32 parameters with float64 data give float32 output.

The twin therefore rounds `Y` to float32 before the product in a float32 model; computing the product in float64
and rounding afterwards would differ at the float32 ulp level (a golden for the float32 variant has to be taken
from TF, not derived from the float64 one).

## 7. Options rejected

- `lora_config` not `None`: **Proposal**, rejected as for the radial MLPs; `enable_lora_adaptation` and
  `finalize_lora_update` (`compute.py:1289-1299`) are training-side.
- `indicator_is_per_atom = True` (`compute.py:1246-1247`): never true for a shipped embedding; the twin rejects it.
- `avg_n_neigh` of any type other than `float` or `dict`: `TypeError` in TF; the twin raises at load time. An
  `int` in a yaml (YAML `40` instead of `40.0`) is rejected by TF too, so no saved model has one.
- `local = True`: not applicable (no such mode).

## 8. Golden-fixture keys

Fixture keys `A` and `B0`: omat `[n_atoms, 32, 25]` both; large_base `[n_atoms, 42, 25]` and `[n_atoms, 32, 25]`. TF
oracle: an explicit loop over bonds with real harmonics from `scipy.special`, rotation covariance, translation and
permutation invariance (`tests/test_compute.py:2255-2832`). FIX2 produces the yaml fixtures; no achieved error
exists yet. The option pairs `indicator_l_depend`, dict `avg_n_neigh`, `lm_first` and `sum_neighbors = False`
need option-pair fixtures: no 2L yaml uses them.

## For the reviewer

None beyond the family-wide proposals of the README.
