# ConstantScaleShiftTarget

| | |
|---|---|
| Source | `instructions/output.py:659-788` (class at 659), bases `TPOutputInstruction` (`output.py:50-75`), `ElementsReduceInstructionMixin` (`instructions/base.py:418`) |
| Family | norm and output |
| Used in | the omat yaml only: name `ConstantScaleShiftTarget`, `scale 1.892985414710868`, `shift 0` (an integer in the yaml), no `atomic_shift_map`, no chemical embedding, `l 0`, target `atomic_energy`. Not in the large_base yaml. |
| Reads | the target; with a shift also `atomic_mu_i`, `batch_tot_nat_real` |
| Writes | **`atomic_energy`** (the target's name), `[n_atoms, 1]` |
| Variables | none in the omat yaml (`embedding_shift` with a chemical embedding) |

Scales the energy by a constant and shifts it by a constant, by a per-element constant, or by a learned projection of
the chemical embedding. In omat it is a pure rescaling: the foundation-model scale.

## 1. Constructor arguments

`output.py:670-703`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `target` | required | a `CreateOutputTarget` (stored as a name) | The accumulator to rescale. |
| `scale` | `1.0` | float | Multiplies the target. |
| `shift` | `0.0` | number | A constant added to **every real atom**. The yaml stores whatever it was given (`0`, an int, in omat). |
| `atomic_shift_map` | `None` | `None` or dict `element index -> float` | A per-element shift. The map is turned into an array in the **sorted order of its keys** (`output.py:684-686`); the key is then not used as the index (below). Needs `shift == 0` (`assert`). |
| `chemical_embedding` | `None` | a `ScalarChemicalEmbedding` or `None` | Adds a shift `Z[mu_i] @ embedding_shift / sqrt(embedding_size)`. Needs `shift == 0` (`assert`). |
| `name` | `"ConstantScaleShiftTarget"` | any unique `str` | Key of the instruction (the output goes to the target's name). |
| `l` | `0` | integer | `l` of the target. |

`apply_shift` is `False` when `shift == 0` and there is no map and no embedding (`output.py:691-703`); that is the omat case.
Note that `self.l` is not compared with the target here (no `assert_l_compatibility`).

## 2. Derived tables

`atomic_shift_map`: with a map, the array of its values **in the sorted order of the keys**, shape `[n]`, then `[n, 1]` after
`build`. The row of an atom is `atomic_mu_i`, **not** the key: a map `{0: a, 2: b}` gives `b` to element 1 and fails in the
gather for element 2 (TEST6 finding 16, pinned at `tests/test_output.py:889`). The yaml of a real model has keys `0 .. n-1`,
for which position and key agree.

## 3. Parameters

Names, shapes and dtypes are **[I]**.

| Attribute | TF name | Shape | Present when |
|---|---|---|---|
| `embedding_shift` | `embedding_shift:0` | `[embedding_size, 1]`, `N(0, 1)`, trainable, `float_dtype` | `chemical_embedding` is given |

`scale`, `shift` and `atomic_shift_map` are **constants from the yaml**, not variables: they are not in the checkpoint.

## 4. Runtime constants

Created in `build` (`output.py:705-723`), in `float_dtype`: `scale` (`tf.constant`), `constant_shift` (only if non-zero),
`atomic_shift_map` as `[n, 1]`, and with an embedding `embedding_norm = rsqrt(embedding_size)`. **The scale is rounded to
`float_dtype`**: for float32 parameters `1.892985414710868` becomes `1.892985463142395` (**Measured**), a relative change
of `2.6e-8` of the energy. The twin has to round the yaml constants to the parameter dtype in the same way (and not use a
float64 scale on a float32 energy).

## 5. Forward

`output.py:725-771`. `target = data[target.name]`, `real[a] = (a < batch_tot_nat_real)`.

```
if apply_shift:
    total = zeros_like(target)
    if atomic_shift_map:      total += where(real, gather(map, atomic_mu_i), 0)                  # [n_atoms, 1]
    if chemical_embedding:    total += where(real, (gather(data[emb], atomic_mu_i) @ embedding_shift) * embedding_norm, 0)
    if constant_shift != 0:   total += where(real, ones_like(target) * constant_shift, 0)
    out = target * scale + total
else:
    out = target * scale
```

The shifts are **not** scaled: the shift is added after the scale. `where(real, ...)` is the identity in the twin (no padding).
With several shifts the code allows only one at a time by the `assert` on `shift`, but a map and an embedding can be given
together.

The formula was re-implemented in numpy and compared with TF: the constant, the per-element and the embedding shifts and
the pure scaling, on synthetic targets (exact, difference 0); and the omat chain (`LinMLPOut2ScalarTarget` followed by the
scale), `3.1e-17` on values up to `8e-2` (float64).

## 6. Dtype and promotion

The output is `target * cast(scale, target.dtype) + cast(total, total.dtype)`: the dtype of the target (float32 in a float32
model, **Measured**). A float32 target with an instruction **built in float64** raises `InvalidArgumentError`
(`tf.where` runs before the cast to the target dtype; TEST6 finding 3, pinned at `tests/test_output.py:968`): the
instruction must be built in the dtype of the model. The twin builds every constant in the parameter dtype, as TF does for
a consistent model.

## 7. Options rejected

- `atomic_shift_map` whose keys are not exactly `0 .. n-1`: **Proposal**, rejected with a message (TF's sorted-position
  semantics differ from the key semantics and fail for gaps; no shipped model has gaps).
- `atomic_shift_map` with `n` smaller than the number of elements of the model: the gather fails in TF for an element beyond
  `n`; the twin raises at load time.
- `chemical_embedding`, `atomic_shift_map` and a constant `shift`: **ported** (the formulas above); no 2L model uses them, so
  they need option fixtures.
- `prepare_variables_for_selected_elements`, `upd_init_args_new_elements` (`output.py:773-788`): element restriction is out
  of scope (decision D19).
- `local = True`: not applicable.

## 8. Golden-fixture keys

The final `atomic_energy` `[n_atoms, 1]` of the omat chain (this instruction is the last), and for tests of this class alone the
value before it (a snapshot before and after, because the key is overwritten in place). FIX2 produces the yaml fixtures; no
achieved error exists yet. TF oracle: `tests/test_output.py:838-1246` (numpy re-computation, per-element shifts, the embedding
shift, padding atoms).

## For the reviewer

1. **Proposal (section 7)**: reject `atomic_shift_map` keys other than `0 .. n-1`.
