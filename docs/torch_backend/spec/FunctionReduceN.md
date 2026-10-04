# FunctionReduceN

| | |
|---|---|
| Source | `instructions/compute.py:3044-3340` (class at 3045); `collect_functions` in `instructions/base.py:567-612`; helpers `_lmp_lookup`, `_lmp_matches` (`compute.py:126-153`) |
| Family | product and reduce |
| Bases | `TPEquivariantInstruction`, `SimplifyingReduceMixin`, `LORAInstructionMixin` (`instructions/base.py:492`, `441`, `393`) |
| Used in | both 2L yamls. omat: `I0`, `I`, `B`, `I_out`. large_base: `I1`, `I`, `rho`, `B`, `I2`. Per-element weights in `I0`, `I_out`, `I1`, `rho`, `I2` (89 element types). |
| Reads | the collected instructions (several), each `[n_atoms, n_in, lm]`; `atomic_mu_i` |
| Writes | `<name>` `[n_atoms, n_out, n_lm_out]` (`[n_lm_out, n_atoms, n_out]` with `lm_first`) |
| Variables | one weight tensor per collected instruction, `reducing_<instruction name>` |

The learned linear reduction of several basis tensors to one: from each source it selects the entries with
`l <= ls_max` and an allowed `(l, parity)`, mixes the channels with a weight that depends on the `(parity, l, history)` of
the entry (and optionally on the element of the central atom), and sums the contributions of all sources into one
tensor with a fresh angular layout. It builds the invariant energy features (`I_out`, `rho`, `I2`) and the
first-layer equivariant indicator `I` that feeds the second layer.

## 1. Constructor arguments

`compute.py:3052-3151`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `instructions` | required | list of `TPEquivariantInstruction` objects (stored as names) | Sources, in order; names must be unique (`assert`). |
| `name` | required | any unique `str` | Key of the output. |
| `ls_max` | required | an int, or a list with one int per source | Highest `l` taken from each source; an int is repeated. `lmax = max(ls_max)`. The yamls use lists (`[1, 1, 1, 0]`, `[3, 3]`, ...) and ints (`4`, `0`). |
| `n_out` | required | integer | Output channels. |
| `allowed_l_p` | required | list of `[l, parity]` | The `(l, parity)` entries kept, and the layout of the output (section 2). |
| `out_norm` | `False` | bool | Multiply the output by `norm_map` (section 2). Both yamls: `false`. |
| `is_central_atom_type_dependent` | `False` | bool | Per-element weights; needs `number_of_atom_types` (`assert`). Shipped: true for the invariant reductions of the first-layer products. |
| `number_of_atom_types` | `None` | integer or `None` | Leading axis of per-element weights (89 in both yamls). |
| `init_vars` | `"random"` | `"random"`, `"uniform"`, `"zeros"` | Training only; others fail an `assert`. |
| `normalize` | `True` | bool | `True`: weights `N(0, scale)`, forward factor `scale / sqrt(n_in)`; `False`: weights `N(0, 1/sqrt(n_in))`, factor `1`. The factor is a runtime constant, so the twin needs this flag (the omat yaml has no `normalize` key: `True`). |
| `init_target_value` | `"zeros"` | `"zeros"`, `"ones"` | Initial value of the accumulator in the forward pass: **it is not training-only**: `"ones"` makes the output start from 1 (section 5). Others fail an `assert`. Both yamls: `zeros`. |
| `simplify` | `False` | bool | `True` rewrites coupling histories at construction; with `ProductFunction` it always raises `TypeError` (TEST6 finding 8). Section 7. |
| `scale` | `1.0` | float | Multiplies the init and the forward factor when `normalize` (section 4). Both yamls: `1.0`. |
| `lora_config` | `None` | dict or `None` | Section 7. |
| `lm_first` | `False` | bool | Layout. |
| `**kwargs` | | | Swallowed. The omat yaml stores `n_in: null`, `chemical_embedding: null`, `downscale_embedding_size: 16`: all **ignored** (allow-list of this class). |

## 2. Derived tables

All in `__init__` (`compute.py:3105-3151`); numpy in the twin, checked against TF dumps (D7). Groupings use `groupby(...).indices`,
that is, groups in **ascending order of the key tuple** (**Measured**: this is the order of the keys for every table of both
yamls), `hist` compared as a string.

**Output layout** `coupling_meta_data` (`compute.py:3111-3123`): for `p` in `[-1, 1]`, for `l` in `0 .. lmax`, if
`[l, p]` is in `allowed_l_p`, for `m` in `-l .. l`, a row `(l, m, hist = "", parity = p, sum_of_ls = l)`; then sorted by
`["l", "parity", "hist", "m"]` (so by `l`, then parity `-1` before `+1`, then `m`). `n_lm_out` is the number of rows.
`lmax = max(ls_max)`. Examples: omat `I0`, `I`: `l = 0 (+)`, `l = 1 (-)`, 4 rows; large_base `I1`, `I`: `l <= 3` natural
parity, 16 rows; the scalar reductions: 1 row.

**Per source** `s` (`collect_functions(max_l = ls_max[s], l_p_list = allowed_l_p)`, `instructions/base.py:567-612`):

1. Group the source's table by `(parity, l, hist)`; keep the groups with `l <= max_l` and `[l, parity]` in `allowed_l_p`;
   `func_collect_ind` = the concatenated row indices of the kept groups, in group order (then within a group in table order).
   (`np.concatenate` of an empty list fails with a bare `ValueError` when nothing is kept; TF comment at
   `base.py:587-589`.)
2. `collect_meta_df` = the source table's rows `func_collect_ind`; group it again by `(parity, l, hist)`:
   `w_shape` = the number of groups; `w_l_tile[row]` = the position of the row's group among them (ascending key order).
3. `total_sum_ind[row]` = the index of the output row with the same `(l, m, parity)` (`_lmp_matches`; the output rows are
   unique in `(l, m, parity)`, so there is exactly one). This is the destination of each collected row.
4. `norms[output row]` counts how many collected rows (over all sources) land on it; `norms == 0` becomes 1;
   `norm_map = 1 / sqrt(norms)`.

These tables were re-derived with an independent numpy implementation and compared with all nine instances of both yamls:
identical. Sizes (**Measured**, `n_collected / w_shape` per source): omat `I0`: `A 4/2`, `AA 17/9`, `AAA 107/45`, `AAAA 69/69`;
`B`: `YI 65/13`, `B0 25/5`; `I_out` (one row): `A 1/1`, `AA 5/5`, `AAA 14/14`, `AAAA 69/69`, and the same four from `B`'s family.
large_base: `I1`: `A 16/4`, `AA 87/21`, `AAA 637/133`, `AAAA 378/172`; `B`: `YI 190/40`, `B0 16/4`; `rho`: `1/1, 5/5, 14/14, 69/69`.

## 3. Parameters

Names, shapes and dtypes are **[I]**. Trainable, `float_dtype`, `N(0, scale)` if `normalize` (not zero).

| Attribute | TF name | Shape |
|---|---|---|
| `reducing_<s>` for each source `s` | `<name>/reducing_<s>:0` | `[n_out, n_in(s), w_shape(s)]`, or `[n_types, n_out, n_in(s), w_shape(s)]` if `is_central_atom_type_dependent`; `n_in(s) = s.n_out` |

Examples (**Measured**): omat `I0/reducing_A` `[89, 16, 32, 2]`, `I0/reducing_AAAA` `[89, 16, 32, 69]`, `I/reducing_I0`
`[32, 16, 2]`, `B/reducing_YI` `[32, 32, 13]`, `I_out/reducing_A` `[89, 13, 32, 1]`; large_base `I1/reducing_AAA`
`[89, 12, 42, 133]`, `B/reducing_YI` `[64, 32, 40]`, `rho/reducing_AAAA` `[89, 17, 42, 69]`. The attribute names are
`reducing_<instruction name>`, so a source called `A` gives `reducing_A`; the extractor keys follow them.

## 4. Runtime constants

Created in `build` (`compute.py:3153-3228`), `float_dtype`, none in the checkpoint (dtypes **[I]**):

- `norm_<s> = scale / sqrt(n_in(s))` if `normalize`, else `1.0` (a scalar per source).
- `norm_map`: with `out_norm`, the table of section 2 reshaped to `[n_lm_out, 1, 1]`; without it the scalar `1`.

## 5. Forward

`compute.py:3258-3337`, `lm_first = False`:

```
acc = zeros (or ones if init_target_value = "ones") of shape [n_lm_out (1 if only_invar), n_atoms, n_out]   # dtype of the first source cast to its weight dtype
for each source s:
    A = gather(data[s], func_collect_ind[s], axis=2)                       # [n_atoms, n_in, n_collected]
    A = cast(A, w.dtype)
    w = gather(reducing_s, w_l_tile[s], axis=-1)                           # [(types,) n_out, n_in, n_collected]
    if is_central_atom_type_dependent:  w = gather(w, atomic_mu_i, axis=0);  A = A[: n_atoms]    # [atoms, n_out, n_in, n_collected]
    pr = einsum("knw,anw->wak", w, A) * norm_s                             # per element: "aknw,anw->wak"; [n_collected, n_atoms, n_out]
    only_invar:  acc += sum(pr, axis=0, keepdims=True)
    else:        acc = scatter_nd_add(acc, total_sum_ind[s], pr)
acc = acc * norm_map
out = transpose(acc, [1, 2, 0])                                            # [n_atoms, n_out, n_lm_out]
```

- `only_invar` is `True` when **all** `ls_max` are 0: the accumulator has one row and all collected entries (all
  `l = 0`) are summed into it, whatever the output table says.
- `n_atoms` is the length of `atomic_mu_i` (padded); with per-element weights the source tensor is cut to that length.
- `scatter_nd_add` accumulates duplicates, so several sources add to the same output row; the sum is not ordered
  (rounding level only).
- `init_target_value = "ones"` adds 1 to **every** output entry before the scale. It is stored in the yaml and has an
  effect at inference.
- With `lm_first` the gather axis is 0, the einsums read `"knw,wan->wak"` and `"aknw,wan->wak"`, and the accumulator is returned
  as `[n_lm_out, n_atoms, n_out]` without a transpose. Consumers that need a scalar then transpose themselves
  (`compute.py:3332-3336`).
- Written out for an invariant output: `out[a, k] = sum_s norm_s sum_{n, w} reducing_s[mu_a, k, n, g(w)] A_s[a, n, w]`.

The formula was re-implemented in numpy from the instruction's variables and tables and compared with the TF output for all
nine instances: maximum difference `5.3e-22` on outputs up to `3e-6` (float64, seeded weights).

## 6. Dtype and promotion

Each source is **cast to the weight dtype** (`tf.cast(A_r, w.dtype)`, `compute.py:3287`) and the accumulator takes the dtype of
the first cast source, so the output has the parameter dtype whatever the input dtype (TEST6 finding 1;
**Measured** float32 for all instances of a float32 model). `norm_<s>` and `norm_map` are in `float_dtype`. The twin casts each
source to the weight dtype before the contraction, as TF does.

## 7. Options rejected

- `lora_config` not `None`: **Proposal**, rejected (`compute.py:3225-3226`, `3230-3250`).
- `simplify = True`: **Proposal**, rejected. With a plain `ProductFunction` TF raises `TypeError` (the other class combination
  runs but leaves the tables unchanged: `drop_unused` is a no-op on them, `tests/test_compute.py:4119`, `4168`); no
  shipped model has it.
- `init_vars`: accepted, no effect. `init_target_value` is **ported** (see section 5). `is_central_atom_type_dependent`,
  `normalize = False`, `out_norm = True`, `scale != 1`, `lm_first`: ported; those not set by a 2L yaml need option-pair
  fixtures.
- `upd_init_args_new_elements` and the element-selection mixin (`compute.py:3339-3340`): element restriction is out of
  scope (D19).
- A source with no kept group (empty `func_collect_ind`): TF fails with a bare `ValueError`; the twin raises a typed error
  naming the instruction and the source.
- `local = True`: not applicable.

## 8. Golden-fixture keys

Fixture keys `I0`, `I`, `B`, `I_out` (omat) and `I1`, `I`, `rho`, `B`, `I2` (large_base): shapes `[n_atoms, n_out, n_lm_out]`
(`[n_atoms, 32, 4]` for `I`, `[n_atoms, 13, 1]` for `I_out` in omat; `[n_atoms, 17, 1]` for `rho`, `I2` in large_base), plus
the dumped tables (output layout, `func_collect_ind`, `w_l_tile`, `total_sum_ind`, `norm_map`). TF oracle: numpy
re-computation from the stored weights, invariance of the scalar outputs, the layout of the output table
(`tests/test_compute.py:3688-4471`). FIX2 produces the yaml fixtures; no achieved error exists yet.

## For the reviewer

1. **`init_target_value`.** It looks like a training knob but changes the inference result when `"ones"`. The twin ports it.
   Both yamls use `"zeros"`; say if it should be rejected instead.
