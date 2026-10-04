# SingleParticleBasisFunctionEquivariantInd

| | |
|---|---|
| Source | `instructions/compute.py:1302-1526` (class at 1303); helpers `_dense_reshape_einsum` (58-70), `_resolve_dense_nbr` (73-76), `_equiv_cg_couple` (79-123); tables from `functions/couplings.py:2285-2562` (`real_coupling_metainformation`) |
| Family | embedding and single-particle basis |
| Base | `TPEquivariantInstruction` (`instructions/base.py:492`) |
| Used in | both 2L yamls, name `YI`, `radial R1`, `angular Y`, `indicator I`, `lmax 4`, `sum_neighbors true`, `avg_n_neigh 39.773345702648434`, `normalize true`. omat: `Lmax 4`, natural-parity `keep_parity` up to `L = 6`; large_base: `Lmax 3`, both parities of every `L` up to 6. |
| Reads | the radial, angular and indicator entries; `ind_i`, `ind_j`, `atomic_mu_i`; `mu_j` and the chemical embedding when given |
| Writes | `YI` `[n_atoms, n_rad_max, n_func]` (`[n_func, n_atoms, n_rad_max]` with `lm_first`); omat `n_func = 65`, large_base `n_func = 190` |
| Variables | none in the shipped yamls (one `Linear` with a chemical embedding) |

The second-layer single-particle basis: the radial-angular bond product is multiplied with the **equivariant**
first-layer indicator `I` of the neighbour (not a scalar), summed over neighbours, and coupled with real
Clebsch-Gordan coefficients to functions of definite `(L, M, parity)`.

## 1. Constructor arguments

`compute.py:1311-1441`. All stored in the yaml (G3); `dense_nbr` is written into the saved arguments after
resolution (`compute.py:1441`).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `angular` | required | a `SphericalHarmonic` | `lmax` and meta table. |
| `indicator` | required | a `TPEquivariantInstruction` with a `coupling_meta_data` and `n_out` | The first-layer reduction `I`; `[n_atoms, n_ind, lm_I]`. `n_ind` has to equal `radial.n_rad_max` (both are the einsum axis `n`). |
| `name` | required | any unique `str` | Key of the output. |
| `lmax` | required | integer `<= angular.lmax` | Highest degree of the angular part **and of the indicator** (`l1 > lmax or l2 > lmax` skips a pair, `couplings.py:2390`; TEST6 finding 6). |
| `Lmax` | required | integer `>= 0` | Highest degree `L` of the output; stored as `self.lmax`. |
| `radial` | required | an `MLPRadialFunction(_v2)` | `radial.lmax == lmax` if `angular.lmax > lmax`, else `radial.lmax == angular.lmax` (`assert`s, `compute.py:1372-1383`). |
| `keep_parity` | `None` | list of `[L, parity]` | Output components kept; `None` means natural parity, `[[L, (-1)**L] for L in 0 .. Lmax]`. |
| `history_drop_list` | `None` | list of `[hist, L]` | Extra `(history, L)` pairs to drop from the coupling table. Both yamls: `null`. |
| `l_max_ind` | `None` | integer or `None` | Extra cap on the indicator's `l` (`lmax_B`). Both yamls: `null`. |
| `max_sum_l` | `None` | integer or `None` | Cap on the sum of `l` over the coupling history. Both yamls: `null`. |
| `sum_neighbors` | `True` | bool | `False` keeps the per-bond tensor, no averaging. |
| `avg_n_neigh` | `1.0` | a number | **No dict support** (`1.0 / avg_n_neigh`, a dict fails with `TypeError`); `per_specie_n_neigh` is always `False`. |
| `normalize` | `False` | bool | `True` multiplies each real CG block by `CG_normalization_dict[(max(l1, l2), min(l1, l2), L)]` (default factor 1.0 for a missing key). Both yamls: `true`. |
| `radial_basis`, `hidden_layers` | `None` | any | **Never read**: accepted and ignored. Both yamls store them as `null`. |
| `chemical_embedding` | `None` | a `ScalarChemicalEmbedding` or `None` | Adds a species-dependent `L = 0` term to the indicator of the neighbour (section 5). Both yamls: none. |
| `lm_first` | `False` | bool | Layout of the indicator input and of the output. |
| `dense_nbr` | `None` | `None`, `True`, `False` | Neighbour-sum mode; `None` takes the default of the active `InstructionManager` (`False` outside one) (`compute.py:73-76`, `1439`). The two shipped yamls predate the key; loaded without a manager they resolve to `False`. |
| `**kwargs` | | | Swallowed. The omat yaml stores two such keys, `radia_basis: null` (a misspelling of `radial_basis`) and `n_out: null`; **both have no effect**. |

Because the omat yaml carries keys that the constructor does not read, a loader that rejects unread keys cannot
load a shipped model: see "For the reviewer".

## 2. Derived tables

All built in `__init__` from the two meta tables (`compute.py:1393-1431`); the twin builds them with numpy and
checks them against TF dumps (decision D7: duplicated plan code with table-equality tests).

**`coupling_meta_data`** `[n_func, 10]`, from `real_coupling_metainformation(A=angular meta, B=indicator meta, lmax,
lmax_B=l_max_ind, Lmax, history_drop_list, max_sum_l, keep_parity, normalize, optimize_ms_comb=False)`
(`couplings.py:2285-2562`; the call is `compute.py:1393-1404`). Columns `l, m, hist, left_inds, right_inds, l1, l2,
parity, sum_of_ls, cg_list`. The algorithm, in the order of the code:

1. Group each input table by `(l, hist, parity, sum_of_ls)` and index its rows by `(l, m, hist, parity)`, which has to
   be unique (`ValueError` otherwise) (`couplings.py:2346-2362`).
2. For every pair of groups `(l1, hist1, p1, s1)` of `A` and `(l2, hist2, p2, s2)` of `B`: `parity = p1 * p2`; skip
   the pair if `s1 + s2 > max_sum_l`, if `l1 > lmax` or `l2 > lmax`, or if `l2 > l_max_ind`
   (`couplings.py:2385-2397`). The history-based limits and `is_A_B_equal` are not used by this class. The loop order
   does not matter: the final sort (step 6) is on a key that is unique (**Measured**: no duplicate
   `(l, parity, hist, m)` in the tables of `YI`, `AAA`, `BBB`, `I`, `B` of either yaml).
3. The new history is `f"({hist1}{l1}{sign1},{hist2}{l2}{sign2})"` with `sign = "+"` for parity `+1` and `"-"`
   otherwise (`couplings.py:2418-2428`).
4. For every `L` from `|l1 - l2|` to `l1 + l2` with `L <= Lmax` and `(L, parity)` in `keep_parity`, take the real CG
   tensor `RCG[2L+1, 2*l1+1, 2*l2+1]` (below), multiplied by `CG_normalization_dict[...]` when `normalize`, and for
   every `M = -L .. L` keep the pairs `(m1, m2)` with `|cg| > 1e-15`, in `np.argwhere` (row-major) order of the matrix
   `RCG[M + L]`. A row is emitted when it has at least one pair, with `left_inds = index of (l1, m1, hist1, p1) in A`,
   `right_inds = index of (l2, m2, hist2, p2) in B` and `cg_list = the values` (`couplings.py:2443-2508`).
5. Drop every row whose `(hist, L)` is in the union of the three exclusion tables
   `EXCLUSION_HIST_LISTS` (`natural_parity` 264 entries, `natural_parity_extra` 22, `cross_parity` 1772;
   `couplings.py:135`, `2240`, `418`, `2278-2283`) and of `history_drop_list` (`couplings.py:2510-2541`). The tables are
   **data** the twin copies verbatim (they remove couplings that are linearly dependent on others).
6. Sort by `["l", "parity", "hist", "m"]` and reset the index (`couplings.py:2560`; the `hist` strings sort as plain strings). The order of the rows is the
   order of the output channels `n_func`, so it is part of the contract.

The real CG tensor (`couplings.py:2582-2623`): the complex Clebsch-Gordan coefficients `<l1 m1 l2 m2 | L M>` (sympy
`CG`, `lru_cache`d per `(l1, l2, L)`), rotated to the real basis by
`RCG = real_if_close( einsum("PM,Mij,ik,jl->Pkl", C2R(L)^H, CG, C2R(l1), C2R(l2)) )`, with `C2R(l)` from
`c2r_harm_matrix` (`couplings.py:2565-2576`). `CG_normalization_dict` (120 entries, `couplings.py:11-134`) maps
`(l1, l2, L)` to a float; the code multiplies the real CG by it.

**Index tables** (`compute.py:1407-1431`):

- `lr_inds` `[n_cg, 2]`, int32: column 0 is `np.concatenate(left_inds)`, column 1 `np.concatenate(right_inds)`; the
  position in the flattened `(lm_Y, lm_I)` plane is `left * lm_I + right`.
- `m_sum_ind` `[n_cg]`, int32: the row `i` of `coupling_meta_data` repeated `len(cg_list[i])` times, so it maps each CG
  term to its output channel.
- `n_func = max(m_sum_ind) + 1` (equal to the number of rows).
- `cg` `[n_cg, 1, 1]`, `float_dtype` after `build`: `np.concatenate(cg_list)`.

Sizes (**Measured**): omat `n_func = 65`, `n_cg = 157`; large_base `n_func = 190`, `n_cg = 877`. The indicator tables
have 4 rows (omat, `L <= 1`) and 16 rows (large_base, `L <= 3`).

**Other attributes**: `slice_angular = (lmax + 1)**2` if `angular.lmax > lmax`, else `None`;
`coupling_origin = [angular.name, indicator.name]`; `n_out = radial.n_rad_max`; `inv_avg_n_neigh = 1 / avg_n_neigh`.
With a chemical embedding: `chem_l0_idx`, the row of the indicator table with `(l, m, parity) = (0, 0, +1)` (an `assert`
that it is unique, `compute.py:1356-1364`) and `n_lm_indicator`.

## 3. Parameters

None in the shipped yamls. With a `chemical_embedding` there is one `Linear` `chem_linear`
(`<name>_ChemProj`, `n_in = embedding_size`, `n_out = indicator.n_out`, no bias, `normalize = True`, so the factor
`1 / sqrt(embedding_size)` applies; `functions/nn.py:17-147`), whose names, shapes and dtypes are **[I]**.

## 4. Runtime constants

Created in `build` (`compute.py:1444-1459`), in `float_dtype`: `inv_avg_n_neigh`, `cg`, and with a chemical
embedding `chem_l0_mask = one_hot(chem_l0_idx, n_lm_indicator)`. `lr_inds` and `m_sum_ind` are int32 constants made in
`__init__`. None is in the checkpoint; the twin recomputes them. (dtypes **[I]**)

## 5. Forward

`compute.py:1461-1526`. `I = data[indicator]`, `Y = data[angular]`, `R = data[radial]` `[n_bonds, n, lm]`,
`lm = (lmax + 1)**2`, `lm_I` the indicator's `lm` count.

```
if lm_first:  I = transpose(I, [1, 2, 0])                       # to [n_atoms, n, lm_I]
bond_I = gather(I, ind_j, axis=0)                               # [n_bonds, n, lm_I]
if chemical_embedding:
    zp   = chem_linear(cast(data[emb], bond_I.dtype))           # [n_elements, n]
    bond_I = bond_I + gather(zp, mu_j)[:, :, None] * one_hot(chem_l0_idx)[None, None, :]
Y  = Y[:, :slice_angular];  Y = cast(Y, R.dtype)
RY = R * Y[:, None, :]                                          # [n_bonds, n, lm]
if sum_neighbors:
    prod[a, n, l, r] = sum_{j : ind_i[j] = a} RY[j, n, l] * bond_I[j, n, r]      # n_atoms = len(atomic_mu_i)
    prod = prod * (1 / avg_n_neigh)
else:
    prod[j, n, l, r] = RY[j, n, l] * bond_I[j, n, r]                             # per bond, no averaging
out[a, n, f] = sum_{k : m_sum_ind[k] = f} prod[a, n, lr_inds[k, 0], lr_inds[k, 1]] * cg[k]
if lm_first:  out = transpose(out, [2, 0, 1])                    # [n_func, n_atoms, n]
```

- **Coupling** (`_equiv_cg_couple`, `compute.py:79-123`): with `_USE_GEMM_COUPLE = True` (`compute.py:55`) it builds the
  dense matrix `W[lm_Y * lm_I, n_func]` by `scatter_nd` of `cg` at `(lr_inds[:, 0] * lm_I + lr_inds[:, 1], m_sum_ind)`
  (duplicates add) and computes `prod_flat @ W`. The fallback branch (`gather_nd`, multiply by `cg`, segment sum)
  is stated to be bit-identical in value; it is not executed. `cg` is cast to `prod.dtype`, `inv_avg_n_neigh` too.
- **`dense_nbr = True`** replaces the segment sum by `_dense_reshape_einsum` (`compute.py:58-70`): it reshapes the bond
  tensors to `[n_atoms, max_neigh, n, ...]` with `max_neigh = n_bonds // n_atoms` and contracts `"amnl,amnr->anlr"`.
  This needs a bond layout with exactly `max_neigh` consecutive bonds per atom (padding bonds at a distance beyond
  the cutoff, so the radial envelope zeroes them: `data/databuilder.py:690-707`). It adds no input and gives the same
  value as the segment sum for such a layout (**Measured**: difference 0.0 on a random uniform layout).
- Written out: `YI[i, n, f] = (1/avg) * sum_k cg_k * sum_{j in N(i)} R[j, n, l_k] Y[j, l_k] I[j_atom, n, r_k]`, with
  `(l_k, r_k) = lr_inds[k]` and `f = m_sum_ind[k]`.
- Equivariance: `YI[:, :, f]` with `(L, M)` from the row `f` of the table transforms as `D^L` under rotations of
  the bonds when `I` does. Fixtures can check it with Wigner matrices.

The formula, with the tables taken from the TF instruction, was re-implemented in numpy and compared with TF
(`sum_neighbors` on and off, `lm_first` on and off, dense and segment sum, and the chemical-embedding branch):
maximum absolute difference `8.9e-16` (float64). This validates the forward formula, not the tables.

## 6. Dtype and promotion

`R` has the parameter dtype in a real model, `Y` is cast to it, `bond_I` follows the indicator (parameter dtype), so
`prod` and the output have the parameter dtype: **Measured** `YI` float32 for float32 parameters, float64 for
float64. In general the class **follows the promoted dtype of its inputs** (TEST6 finding 1: with a float64 radial
tensor and float32 parameters the result is float64); `cg` and `inv_avg_n_neigh` are cast to `prod.dtype`, so they
never change it. The twin has to keep `Y` cast to the radial dtype and let the product follow its inputs. The CG
values enter float32 models as float32 numbers (`cg` is created in `float_dtype`).

## 7. Options rejected

- `chemical_embedding.is_per_atom` true (`compute.py:1477-1478`): never true for a shipped embedding; rejected.
- A `dict` `avg_n_neigh`: TF fails; rejected.
- The unread keys `radial_basis`, `hidden_layers`, `radia_basis`, `n_out`: **accepted and ignored** (allow-list),
  because the omat yaml has them. Any other unknown key is rejected (see the README, "Unread keywords").
- `history_drop_list`, `l_max_ind`, `max_sum_l`, `keep_parity` other than the shipped ones are **ported**: they only
  change the table, and the table generator is ported as a whole.
- `local = True`: not applicable (no such mode).

## 8. Golden-fixture keys

Fixture key `YI`: omat `[n_atoms, 32, 65]`, large_base `[n_atoms, 32, 190]`. The plan tables (`coupling_meta_data`
columns, `lr_inds`, `m_sum_ind`, `cg`) are dumped from TF as fixtures too (table-equality tests, decision D7). TF
oracle: an explicit loop over bonds with real harmonics from `scipy.special`, real CG from sympy rotated by an analytic
matrix, Wigner covariance, translation and permutation invariance (`tests/test_compute.py:2833-3346`). FIX2 produces the
yaml fixtures; no achieved error exists yet. `lm_first`, `dense_nbr` and `chemical_embedding` need option-pair
fixtures (no 2L yaml sets them).

## For the reviewer

1. **Unread keys.** The rule "reject keys that the class does not read" (README) would reject the shipped omat yaml,
   which has `radia_basis` and `n_out` on this class. Proposal: a per-class allow-list of keys ignored by TF, accepted
   with any value, everything else rejected. The list for this class is `radial_basis`, `hidden_layers`,
   `radia_basis`, `n_out`.
2. **`dense_nbr`.** Proposal: the twin accepts `true` and `false` and computes the same sum; the TF dense layout is a
   data-builder concern. The option then only changes reduction order (rounding), not the result.
