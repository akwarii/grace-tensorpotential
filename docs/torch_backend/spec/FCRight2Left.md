# FCRight2Left

| | |
|---|---|
| Source | `instructions/compute.py:3443-3772` (class at 3443), bases `TPEquivariantInstruction`, `ElementsReduceInstructionMixin`, `LORAInstructionMixin` (`instructions/base.py:493`, `418`, `394`) |
| Family | product and reduce |
| Used in | both 2L yamls: `A1`, `AA1`, `AA2`, `B1`, `BB1`, `BB2`; always `left_coefs true`, `init_vars random`, `norm_out true`, `is_central_atom_type_dependent` null or false (omat: `normalize` absent, so `true`; large_base: `normalize true`) |
| Reads | the `left` and `right` entries; `atomic_mu_i` |
| Writes | `<name>` `[n_atoms, n_out, lm_left]` (`[lm_left, n_atoms, n_out]` with `lm_first`) |
| Variables | `w_left`, `w_right` |

A learned channel mixing of the left operand, to which a learned channel mixing of the right operand is **added** on
the entries of equal `(l, parity)`. The output has the angular layout of `left`. In the models it prepares the
operands of the next product (`A -> A1`, `AA -> AA1`).

## 1. Constructor arguments

`compute.py:3451-3549`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `left`, `right` | required | `TPEquivariantInstruction` objects (stored as names) | Operands; need `coupling_meta_data`, `n_out`, `lmax`. |
| `name` | required | any unique `str` | Key of the output; the weights are named after it. |
| `n_out` | `None` | integer or `None` | Output channels; `None` means `left.n_out`. The yamls give `32` (omat) and `42`, `64` (large_base). |
| `left_coefs` | `True` | bool | `False` skips the left weight: the left operand is used as it is (needs `left.n_out == n_out`, `assert`). Both yamls: `true`. |
| `is_central_atom_type_dependent` | `None` | `None`, a bool, or a list of two bools `[left, right]` | Per-element weights for the left and/or right part; `None` means `[False, False]`, a bool applies to both. Needs `number_of_atom_types` when any is true (`assert`); another type raises `ValueError`. Both yamls: `None` or `false`. |
| `number_of_atom_types` | `None` | integer or `None` | Leading axis of per-element weights. |
| `init_vars` | `"random"` | `"random"`, `"uniform"`, `"zeros"` | Training only; any other value fails an `assert`. **Defect**: for `"zeros"` the right weight is still drawn from a normal (TEST6 finding 7, pinned at `tests/test_compute.py:4616`); irrelevant for a loaded model. |
| `normalize` | `True` | bool | Where the `1/sqrt(n_in)` lives: `True`: weights `N(0, 1)` and forward factors `1/sqrt(left.n_out)`, `1/sqrt(right.n_out)`; `False`: weights `N(0, 1/sqrt(n_in))` and factors `1`. The factors are runtime constants, so the twin needs this flag. |
| `norm_out` | `False` | bool | `True` multiplies the result by a per-entry factor (section 2). Both yamls: `true`. |
| `lora_config` | `None` | dict or `None` | Section 7. |
| `lm_first` | `False` | bool | Layout. |
| `**kwargs` | | | Swallowed; none in the shipped yamls. |

## 2. Derived tables

All in `__init__` (`compute.py:3515-3549`); numpy in the twin, checked against TF dumps (D7). Groups are taken with
`groupby(["l", "parity", "hist"]).indices`, i.e. in **ascending order of the key tuple** `(l, parity, hist)` with `hist`
compared as a string (**Measured**: this equals the order of the group keys for every table of both yamls); each group is
the array of row indices of the operand's table.

Let `L_groups` be the groups of `left.coupling_meta_data` and `R_groups` those of `right.coupling_meta_data`.

- `coupling_meta_data` = a copy of the left table: the output has the layout of `left`; `lmax = left.lmax`.
- `w_tile_left` `[lm_left]`, int: for every left row the number (position in `L_groups`) of its group;
  `w_shape_left = len(L_groups)` (`compute.py:3517-3525`).
- For each left group in order, and for each right group in order with the same `l` and the same `parity`
  (**any** history) (`compute.py:3527-3541`): append the left group's rows to `collect_to`, the right group's rows to
  `collect_from`, `count` repeated `len(right rows)` times to `w_tile_right`, then `count += 1` and add 1 to the
  left group's entry of `norms` (initialised to 1). Right groups with no left partner are not used. The right and left
  groups must have the same number of rows (`2 l + 1`); `tensor_scatter_nd_add` would fail otherwise.
- `collect_to`, `collect_from` int32, `np.concatenate` of the lists; `w_tile_right` int32;
  `w_shape_right = max(w_tile_right) + 1 = count`.
- `norm_map[row] = 1 / sqrt(norms[w_tile_left[row]])`, `[lm_left]` (`compute.py:3543-3544`). It is the number of terms added
  to a left entry (its own and one per matching right group), to the power `-1/2`. It is used only with `norm_out`.

These tables were re-derived with an independent numpy implementation (group order, matching, `norms`) and compared with
all twelve instances of both yamls: identical. Sizes (**Measured**, `w_shape_left / w_shape_right / n_collected`):
omat `A1 5/5/25`, `AA1 27/27/141`, `B1 5/5/25`; large_base `A1 5/5/25`, `AA1 27/27/141`, `B1 7/7/31`, `BB1 18/18/76`.

## 3. Parameters

Names, shapes and dtypes are **[I]**. Both are trainable, `float_dtype`, initialised `N(0, 1)` (`normalize`), not zero.

| Attribute | TF name | Shape |
|---|---|---|
| `w_left` (if `left_coefs`) | `<name>/w_left_FC:0` | `[n_out, left.n_out, w_shape_left]`, or `[n_types, n_out, left.n_out, w_shape_left]` if `is_central_atom_type_dependent[0]` |
| `w_right` | `<name>/w_right_FC:0` | `[n_out, right.n_out, w_shape_right]`, or with the leading `n_types` if `[1]` |

Examples (**Measured**): omat `A1` `[32, 32, 5]` both, `AA1` `[32, 32, 27]`; large_base `A1` `[42, 42, 5]`,
`B1` `[64, 64, 7]`, `BB1` `[64, 64, 18]`.

## 4. Runtime constants

`norm_left = 1/sqrt(left.n_out)` (or 1), `norm_right = 1/sqrt(right.n_out)` (or 1), scalars in `float_dtype`
(`compute.py:3579-3650`); with `norm_out` the tensor `norm_map` reshaped to `[lm, 1, 1]` in `float_dtype`
(`norm_out_factor`, `compute.py:3652-3655`). None is in the checkpoint. dtypes **[I]**.

## 5. Forward

`compute.py:3700-3764`, for `lm_first = False` (axis `-1` is the angular one):

```
if left_coefs:
    wl   = gather(w_left, w_tile_left, axis=-1)                         # [n_out, n_in, lm_left]
    left = einsum("knw,anw->wak", wl, data[left]) * norm_left           # [lm_left, n_atoms, n_out]
else:
    left = transpose(data[left], [2, 0, 1])                             # [lm_left, n_atoms, n_out]
right = gather(data[right], collect_from, axis=-1)                      # [n_atoms, n_in_r, n_collected]
wr    = gather(w_right, w_tile_right, axis=-1)                          # [n_out, n_in_r, n_collected]
right = einsum("knw,anw->wak", wr, right) * norm_right                  # [n_collected, n_atoms, n_out]
left  = scatter_nd_add(left, collect_to[:, None], right)                # add the right terms to the left rows
if norm_out:  left = left * norm_map[:, None, None]
out   = transpose(left, [1, 2, 0])                                      # [n_atoms, n_out, lm_left]
```

With `is_central_atom_type_dependent[i]` the corresponding weight is first gathered by `atomic_mu_i` (leading axis) and the
einsum is `"aknw,anw->wak"`. With `lm_first` the operands are `[lm, n_atoms, n]`, the gather axis is 0, the einsums are
`"knw,wan->wak"` / `"aknw,wan->wak"` and no transposes are made. `scatter_nd_add` accumulates duplicates in `collect_to`
(a left group with several matching right groups gets all of them).

Per entry `(l, m)` of the left layout: `out[a, k, (l, m)] = s * (sum_n w_left[k, n, g_l] x_left[a, n, (l, m)] / sqrt(n_l)
+ sum_{matching right groups r} sum_n w_right[k, n, r] x_right[a, n, (l, m)_r] / sqrt(n_r))`, `s` the `norm_map` entry.
The weight is shared by the `2 l + 1` components of a group, which keeps the output equivariant.

The formula was re-implemented in numpy from the instruction's variables and tables and compared with the TF output for all
twelve instances: maximum difference `1.5e-21` on outputs of size up to `3e-6` (float64, seeded weights).

## 6. Dtype and promotion

**No cast** (TEST6 finding 1): the einsums follow the promotion of the weight and the operand. With float32 parameters
and float32 operands (a real float32 model: **Measured** float32 outputs for all six instances) there is no issue; with a
float64 operand and float32 weights TF raises `InvalidArgumentError` (pinned at `tests/test_compute.py:4741`). The twin
does not need to reproduce the failure, but must not silently promote a mixed input either: it raises.

## 7. Options rejected

- `lora_config` not `None`: **Proposal**, rejected (`compute.py:3657-3658`, `3662-3692`).
- `is_central_atom_type_dependent` with any `true`: **ported** (a gather by `atomic_mu_i`); no 2L model uses it, so it needs
  an option-pair fixture. `prepare_variables_for_selected_elements` and `upd_init_args_new_elements` raise
  `NotImplementedError` for it in TF (`compute.py:3766-3772`); element restriction is out of scope (decision D19).
- `left_coefs = False`, `normalize = False`, `norm_out = False`: ported.
- `init_vars`: accepted and ignored.
- No common `(l, parity)` between the operands ends in a bare numpy `ValueError` in TF (finding 14, pinned at
  `tests/test_compute.py:4687`); the twin raises a typed error naming the instruction.
- `local = True`: not applicable.

## 8. Golden-fixture keys

Fixture keys `A1`, `AA1`, `AA2`, `B1`, `BB1`, `BB2` (`[n_atoms, n_out, lm]` with `lm = 25` for `A1`, `B1`, `141` for `AA1`, `AA2`, `BB1`,
`BB2` in omat) and the dumped tables (`w_tile_left`, `w_tile_right`, `collect_to`, `collect_from`, `norm_map`). TF oracle: a numpy
re-computation from the stored weights, equivariance, the effect of `norm_out` (`tests/test_compute.py:4472-4874`). FIX2
produces the yaml fixtures; no achieved error exists yet. Per-element weights and `left_coefs = False` need option-pair
fixtures.

## For the reviewer

None beyond the family-wide proposals of the README.
