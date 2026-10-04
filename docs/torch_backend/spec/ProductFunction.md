# ProductFunction

| | |
|---|---|
| Source | `instructions/compute.py:2021-2169` (class at 2021); tables from `functions/couplings.py:2285-2562` (`real_coupling_metainformation`, called with `legacy_format=True`) |
| Family | product and reduce |
| Base | `TPEquivariantInstruction` (`instructions/base.py:493`) |
| Used in | both 2L yamls: `AA`, `AAA`, `AAAA` (products of the `A` basis) and `BB`, `BBB`, `BBBB` (of the `B` basis); `normalize true`, natural-parity `keep_parity` (large_base `BB` also keeps `[0, -1]`) |
| Reads | the `left` and `right` entries, each `[n_atoms, n_channels, lm]` (`[lm, n_atoms, n_channels]` with `lm_first`) |
| Writes | `<name>` `[n_atoms, n_channels, n_func]` |
| Variables | none |

The symmetric (ACE) product of two atomic basis tensors: the `(l1, m1)` and `(l2, m2)` entries of the two operands
are combined with real Clebsch-Gordan coefficients into entries of definite `(L, M)`, channel by channel (the channel
axis is not mixed). It builds the higher body orders `AA`, `AAA`, `AAAA` from `A` (and `A1`, `AA1`, ...).

## 1. Constructor arguments

`compute.py:2027-2093`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `left`, `right` | required | `TPEquivariantInstruction` objects (stored as names) | The operands; each needs `coupling_meta_data`, `name` and `n_out`. `left.n_out == right.n_out` (`assert`); `n_out` of the product is that value. |
| `name` | required | any unique `str` | Key of the output. |
| `lmax` | required | integer `>= 0` | Highest `l` of an operand entry that is coupled (`l1 > lmax or l2 > lmax` is skipped). The yamls use `4` (omat) and `4` / `3` (large_base `A`-side / `B`-side). |
| `Lmax` | required | integer `>= 0` | Highest `L` of the output; stored as `self.lmax`. Omat: `4, 1, 0, 4, 0, 0` for `AA, AAA, AAAA, BB, BBB, BBBB`. |
| `is_left_right_equal` | `None` | `None`, `True`, `False` | `None` means `left.name == right.name`. `True` selects the symmetric table (below). Both yamls give `None` or an explicit `True` for the squares (`AA`, `BB` of large_base; `AAAA`, `BBBB` are squares by name). |
| `lmax_left`, `lmax_right` | `None` | integer or `None` | Extra caps on `l1`, `l2` (`lmax_A`, `lmax_B`). |
| `lmax_hist`, `lmax_hist_left`, `lmax_hist_right` | `None` | integer or `None` | Caps on the largest `l` appearing in the coupling history of an operand entry. |
| `history_drop_list` | `None` | list of `[hist, L]` | Extra rows to drop; compared with the **legacy** history strings, which carry no parity signs. |
| `max_sum_l` | `None` | integer or `None` | Cap on `sum_of_ls1 + sum_of_ls2`. |
| `keep_parity` | `None` | list of `[L, parity]` | Kept output components; `None` means natural parity, `[[L, (-1)**L] for L in 0 .. Lmax]`. |
| `normalize` | `False` | bool | Multiply each real CG block by `CG_normalization_dict` (section 2). Both yamls: `true`. |
| `lm_first` | `False` | bool | Layout. |
| `**kwargs` | | | Swallowed. The omat yaml stores `n_out: null`, `chemical_embedding: null`, `downscale_embedding_size: 16`, all **ignored** (allow-list of this class). |

The constructor fails an `assert` when no coupling channel survives (`compute.py:2113-2115`).

## 2. Derived tables

Built in `init_coupling` (`compute.py:2095-2139`). The twin builds them with numpy and checks them against TF dumps
(decision D7).

**`coupling_meta_data`**, from `real_coupling_metainformation(A = left meta, B = right meta, lmax, lmax_A, lmax_B,
lmax_hist, lmax_hist_A, lmax_hist_B, Lmax, is_A_B_equal, history_drop_list, max_sum_l, keep_parity, normalize,
legacy_format = True)` with the default `optimize_ms_comb = True`. The algorithm is the one of the sheet of
`SingleParticleBasisFunctionEquivariantInd` (section 2, steps 1 to 6), with these differences:

1. **Legacy history format** (`couplings.py:2420-2427`): the new history is `f"({hist1}{l1},{hist2}{l2})"`, **without**
   the parity signs (that class uses `f"({hist1}{l1}{sign1},{hist2}{l2}{sign2})"`).
2. **History limits** (`couplings.py:2402-2416`) are applied when any of `lmax_hist`, `lmax_hist_A`, `lmax_hist_B` is
   set: with `lmax_from_hist(h)` the largest integer in `h` once `+` and `-` are removed (0 for `""`;
   `couplings.py:2631-2638`), the pair is skipped if `max(lmax_from_hist(hist1), lmax_from_hist(hist2)) > lmax_hist`,
   or the first one `> lmax_hist_A`, or the second `> lmax_hist_B`. None of the shipped models sets them.
3. **Symmetric operands** (`is_A_B_equal`, `couplings.py:2399`, `2428-2442`, `2470-2495`):
   a. a pair with `l2 > l1` is skipped;
   b. the pair is skipped if its history, or the swapped history `f"({hist2}{l2},{hist1}{l1})"`, was already produced
      (the code records both strings of every accepted pair);
   c. with `optimize_ms_comb`, for every `(L, M)` the `(index_A, index_B)` pairs are merged: the key is
      `(max(i, j), min(i, j))` and the CG values of pairs with the same key are **added**, in the order of first
      occurrence; sums with `|sum| <= 1e-15` are dropped, and the merged pair is stored with the larger index in
      `left_inds` and the smaller in `right_inds`. A row is emitted only if a pair is left.
4. **Pruning** (`couplings.py:2511-2541`): only the table `EXCLUSION_HIST_LIST` (264 entries, `couplings.py:135`) applies,
   with `+` and `-` removed from each history string; the other two exclusion tables are **not** used. The
   entries of `history_drop_list` are added as given.

Columns, ordering and the final sort are as in the other sheet (`l, m, hist, left_inds, right_inds, l1, l2, parity,
sum_of_ls, cg_list`, sorted by `["l", "parity", "hist", "m"]`).

**Index tables** (`compute.py:2117-2139`):

- `left_ind`, `right_ind` `[n_cg]`, int32: `np.concatenate(left_inds)`, `np.concatenate(right_inds)`.
- `m_sum_ind` `[n_cg]`, int32: row `i` repeated `len(cg_list[i])` times.
- `nfunc = max(m_sum_ind) + 1`, the number of rows.
- `cg`: `np.concatenate(cg_list)`, shape `[1, 1, n_cg]` (`[n_cg, 1, 1]` with `lm_first`), cast to `float_dtype` in `build`.

Sizes (**Measured**, `n_func / n_cg`): omat `AA 141/623`, `AAA 107/547`, `AAAA 69/335`, `BB 141/623`, `BBB 14/50`,
`BBBB 69/335`; large_base `AA 141/623`, `AAA 637/3161`, `AAAA 378/2138`, `BB 76/264`, `BBB 11/37`, `BBBB 38/156`.

An independent numpy implementation of the algorithm above (using only `gen_CG_matrix_REAL`, `CG_LM_to_sparse` and the
two data tables, with its own loops) reproduces `coupling_meta_data["hist"]`, `left_ind`, `right_ind` and `cg` of all
twelve products and of both `SingleParticleBasisFunctionEquivariantInd` instructions exactly (CG to `1e-13`). It
validates the description, not the data tables.

## 3. Parameters

None. `build` only converts `cg` (`compute.py:2141-2145`).

## 4. Runtime constants

`cg` in `float_dtype`, `left_ind`, `right_ind`, `m_sum_ind` and `nfunc` int32 constants. None is in the checkpoint. **[V]** against the
three probe files, both parameter dtypes (`lm_first = False` in every probed yaml): `cg` has shape `[1, 1, n_cg]` and the
parameter dtype, `left_ind`, `right_ind` and `m_sum_ind` are int32 vectors of length `n_cg` (156 to 3161 in the probed models),
`nfunc` is a scalar int32. The `lm_first = True` layout (`[n_cg, 1, 1]`) is **[I]**.

## 5. Forward

`compute.py:2147-2166`, for `lm_first = False`:

```
lft   = gather(data[left],  left_ind,  axis=2)           # [n_atoms, n, n_cg]
rght  = gather(data[right], right_ind, axis=2)
prod  = lft * rght * cg                                   # cg [1, 1, n_cg]
out   = segment_sum over m_sum_ind of prod along the last axis, num_segments = nfunc    # [n_atoms, n, n_func]
```

(TF transposes to `[n_cg, n_atoms, n]`, runs `unsorted_segment_sum`, and transposes back.) With `lm_first` the gather is on
axis 0 and the result is `[n_func, n_atoms, n]`. No averaging, no activation, no parameters; the channels never mix.
The sum over `k` is a segment sum, so its order is unspecified (rounding level only).

The formula was re-implemented in numpy from the instruction's own tables and compared with the TF output on a periodic
16-atom structure for all twelve products: maximum difference 0 (the relative size of the output is `1e-6`..`1e-14`
with the seeded weights used).

## 6. Dtype and promotion

No cast: `lft * rght * cg` follows the promotion of its three operands. In a real model the two operands are float32
(float32 parameters) or float64 and `cg` has the parameter dtype, so the output has the parameter dtype
(**Measured**: float32 for `AA` ... `BBBB` with float32 parameters). A mixed pair (float64 operand, float32 `cg`) would
promote to float64 under G2; the twin must do the same if it ever sees one.

## 7. Options rejected

- `local = True`: not applicable (no such mode).
- The unread keys `n_out`, `chemical_embedding`, `downscale_embedding_size`: accepted and ignored (allow-list). Any
  other unknown key is an error.
- `lmax_hist*`, `lmax_left`, `lmax_right`, `max_sum_l`, `history_drop_list`, `keep_parity`, `is_left_right_equal` other
  than the shipped values are ported: they only change the table, and the generator is ported whole.
- The legacy `ProductFunction` is the only product class in scope; `CropProductFunction` and `GeneralProductFunction`
  (same file) are not used by the 2L models and have no sheet.

## 8. Golden-fixture keys

Fixture keys `AA`, `AAA`, `AAAA`, `BB`, `BBB`, `BBBB` (shapes `[n_atoms, 32, n_func]` omat, `[n_atoms, 42, n_func]` for
the `A` side and `[n_atoms, 64, n_func]` for the `B` side in large_base) plus the dumped plan tables (`coupling_meta_data`
columns, `left_ind`, `right_ind`, `m_sum_ind`, `cg`). TF oracle: explicit loops with sympy real CG, the exchange symmetry
of a product of a function with itself, Wigner covariance (`tests/test_compute.py:3347-3687`). FIX2 produces the yaml
fixtures; no achieved error exists yet.

## For the reviewer

1. **Scope.** The sheet covers `ProductFunction` only. `CropProductFunction` and `GeneralProductFunction` are not in the
   17 classes and not used by either 2L yaml; they stay unsupported (an error naming the class) as rule R3 says.
