# MLPRadialFunction

| | |
|---|---|
| Source | `instructions/compute.py:673-823` (class at 674); layers in `functions/nn.py:255-433` (`DenseLayer`, `FullyConnectedMLP`) |
| Family | geometry and radial |
| Bases | `TPInstruction`, `LORAInstructionMixin` (`instructions/base.py:327`, `393`) |
| Used in | the omat yaml: `R` and `R1`, both `n_rad_max 32, lmax 4, hidden_layers [64, 64], activation tanh, norm false`, no chemical embedding. Not in the large_base yaml (it uses `MLPRadialFunction_v2`). |
| Reads | the basis entry `[n_bonds, nfunc]`; with a chemical embedding also its table, `mu_i`, `mu_j` (or `ind_i`, `ind_j`) |
| Writes | `<name>` `[n_bonds, n_rad_max, (lmax + 1)**2]` |
| Variables | one weight matrix per layer; `gamma` if `norm` |

The radial function `R_nl(r)` as a bias-free MLP of the radial basis (optionally of the basis concatenated with
chemical embeddings of the two atoms), repeated across the `2l + 1` entries of each degree.

## 1. Constructor arguments

`compute.py:680-764`. The yaml stores every argument (G3); `basis` is stored as the name of the instruction.

| Argument | Default | Values | Effect |
|---|---|---|---|
| `n_rad_max` | required | integer `>= 1` | Number of radial functions per `l`. |
| `lmax` | required | integer `>= 0` | Highest `l`; sets `n_out = n_rad_max * (lmax + 1)`. |
| `basis` | `None` | a `RadialBasis`, another `TPInstruction`, a `str`, or `None` | Source of the input: instruction (`input_shape = basis.basis_function.nfunc`, or `basis.nfunc`), or a name (then `input_shape` is required, an `assert`), or `None` (then `input_shape` is required but `basis_name` is **never set** and the forward pass raises `AttributeError`; TEST6 finding 11, pinned at `tests/test_compute.py:1185`). Another type raises `ValueError`. |
| `input_shape` | `None` | integer | Width of the basis when `basis` is a name. |
| `hidden_layers` | `None` | list of ints | `None` means `[64, 64, 64]` (`compute.py:701-704`). The omat yaml stores `[64, 64]`. |
| `norm` | `False` | `False`, `True` | Output normalisation (section 5). |
| `name` | `"MLPRadialFunction"` | any unique `str` | Key of the output; the layers are named `<name>_MLP_layer<i>`. |
| `activation` | `None` | `None`, `"tanh"`, `"silu"`, `"sigmoid"` | `None`: `silu(x) * 1.6759` (`functions/nn.py:343-344`). A string: that plain function (`functions/nn.py:14`). A non-`str` raises `ValueError`; an unknown string raises `ValueError` in `FullyConnectedMLP` (`functions/nn.py:386-391`). |
| `no_weight_decay` | `True` | `True`, `False` | Only the weight names (G8). |
| `chemical_embedding_i` | `None` | a `ScalarChemicalEmbedding` or `None` | Adds `embedding_size` to the MLP input width (`compute.py:730-733`) and concatenates the embedding of the central atom. |
| `chemical_embedding_j` | `None` | the same | The same for the neighbouring atom. |
| `lora_config` | `None` | dict or `None` | Fine-tuning state; see section 7. |

Extra keywords are swallowed by `**kwargs` and stored; unlike `RadialBasis` nothing downstream reads them.

## 2. Derived tables

`l_tile`, an int32 vector of length `(lmax + 1)**2` built in `__init__` (`compute.py:735-741`):
`l_tile = [0, 1, 1, 1, 2, 2, 2, 2, 2, ...]`, that is `l` repeated `2 l + 1` times. In numpy:
`np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)`. It is a TF tensor, not a variable, so it is not in
the checkpoint; the twin builds it.

Layer widths `layers_config = [input_shape] + hidden_layers + [n_rad_max * (lmax + 1)]`
(`functions/nn.py:379`), where `input_shape` already includes the embedding widths.

## 3. Parameters

All created in `build` (`compute.py:766-780`; `functions/nn.py:281-300`, `413-419`), in `float_dtype`, trainable,
initialised from `N(0, 1)` (not zero). There is **no bias**: `use_bias` is `False` and `MLPRadialFunction` does
not pass it.

| Attribute | Shape | TF name (omat, `name = R`) |
|---|---|---|
| `mlp.layer<i>.w`, `i = 0 .. len(hidden_layers)` | `[layers_config[i], layers_config[i+1]]`, e.g. `[8, 64]`, `[64, 64]`, `[64, 160]` | `R_MLP_layer<i>/DenseLayer_R_MLP_layer<i>_no_decay:0` |
| `gamma` (only if `norm`) | `[1, n_out]`, `N(0, 1)` | unnamed variable |

All names, shapes and dtypes of this section are **[I]**. The names were **Measured** for
`name = "MLPRadialFunction"` and carry the instruction name twice (name scope, then the explicit variable name).

## 4. Runtime constants

(The dtypes of this section are **[I]**.)

- `DenseLayer.norm = 1 / sqrt(n_in)` per layer, a tensor in the weight dtype made in `build`
  (`functions/nn.py:295`). It is **not stored in the checkpoint**; the twin recomputes it from the layer's
  `n_in`. The stored weight is the unscaled `N(0, 1)` weight.
- `epsilon = 1e-5` in the weight dtype, only when `norm` (`compute.py:775`).
- The `1.6759` of the default activation (`functions/nn.py:344`), a Python float.

## 5. Forward

`compute.py:792-823`. Let `b` be the basis `[n_bonds, nfunc]`.

```
if chemical_embedding_i:   z_i = cast(data[emb_i.name], b.dtype)             # [n_elem, emb] (or [n_atoms, emb])
                           b = concat([b, gather(z_i, mu_i  or ind_i if per-atom)], axis=1)
if chemical_embedding_j:   b = concat([b, gather(z_j, mu_j  or ind_j if per-atom)], axis=1)
h_0 = b
for i in 0 .. L-1:         h_{i+1} = act( h_i @ (w_i / sqrt(n_in_i)) )       # L = len(hidden_layers); act on hidden layers only
y   = h_L @ (w_L / sqrt(n_in_L))                                              # no activation on the last layer
if norm:                   y = y * rsqrt( var(y, axis=-1) + 1e-5 ) * gamma    # variance of the row, no mean subtraction
y   = reshape(y, [-1, n_rad_max, lmax + 1])
out = gather(y, l_tile, axis=-1)                                              # [n_bonds, n_rad_max, (lmax + 1)**2]
```

- `DenseLayer.__call__` (`functions/nn.py:302-325`): `w * norm`, cast of the input to the weight dtype if they
  differ, `matmul`, activation if the layer has one. The hidden layers have the activation, the last one has
  none (`out_act = False`, `functions/nn.py:396-399`).
- The embedding of the central atom comes first, then that of the neighbour (columns `[nfunc, emb_i, emb_j]`).
- `reduce_variance` is the population variance of each row around its own mean; `y` is not centred before the
  rescaling.
- The reshape gives `y[b, n, l]` from column `n * (lmax + 1) + l` of the MLP output.

## 6. Dtype and promotion

- With float32 parameters and float64 data the first layer casts the basis down to float32
  (`functions/nn.py:310-311`) and the output is float32: **Measured** `(3, 32, 25)` float32 for float32
  parameters, float64 for float64. Nothing casts it back up. With the embedding, `z` is first cast **up** to
  the basis dtype (float64) for the concatenation and cast down again by the first layer.
- `epsilon` and `gamma` are created in the weight dtype and cast to the dtype of `y` where they are used
  (`compute.py:816-818`).
- `silu(x) * 1.6759`: the factor is applied after the silu in the activation dtype.

The twin follows the same chain of casts, because the downstream class (`SingleParticleBasisFunction*`) also
casts, and the sum over neighbours in float32 versus float64 changes the last digits of the energy.

## 7. Options rejected

- `lora_config` not `None`. The state belongs to a fine-tune in progress: the instruction then holds extra LoRA
  tensors and the main weights are not the effective weights (`compute.py:777-778`, `functions/nn.py:104-105`,
  `127-131`). A model is exported after `finalize_lora_update` has merged them and removed `lora_config` from the
  saved arguments (`instructions/base.py:409-415`). **Proposal**: the twin raises on a yaml that still has a
  `lora_config`, and says to merge it first. User fine-tunes are out of the scope of the rewrite.
- `chemical_embedding_i` or `chemical_embedding_j` with `is_per_atom = True` (gather by `ind_i` / `ind_j`):
  no shipped embedding sets the flag (the attribute is read with `getattr(..., False)`,
  `compute.py:727-728`), so the branch is reached only with a stub (`tests/test_compute.py:1112-1417`).
  **Proposal**: the twin raises on it.
- `basis = None`: TF builds but fails at the first forward pass; the twin raises at load time.
- `norm = True` and a chemical embedding are ported although no shipped model uses them (see the questions).

## 8. Golden-fixture keys

Fixture keys `R` and `R1`, `[n_bonds, 32, 25]` (omat). The oracle is a numpy re-computation of the forward pass
from seeded weights (`tests/test_compute.py:1112-1417`, helper `tests/seeded_weights.py`). FIX2 produces the
yaml fixtures; no achieved error exists yet.

## For the reviewer

1. **`norm` and the embeddings.** No 2L model uses `norm = True`, `chemical_embedding_i` or
   `chemical_embedding_j`. Rule R3 says all options of in-scope classes are ported. Proposal: port them (a few
   lines each, TF oracles exist) and test them with option-pair fixtures; alternative: reject until needed.
2. **Proposal (section 7)**: reject `lora_config` and `is_per_atom` embeddings.
