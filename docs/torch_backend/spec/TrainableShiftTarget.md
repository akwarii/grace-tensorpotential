# TrainableShiftTarget

| | |
|---|---|
| Source | `instructions/output.py:863-916` (class at 863), bases `TPOutputInstruction` (`output.py:50-75`), `ElementsReduceInstructionMixin` (`instructions/base.py:418`) |
| Family | norm and output |
| Used in | the large_base yaml only: name `TrainableShiftTarget`, `number_of_atom_types 89`, `l 0`, target `atomic_energy`; the last instruction of the model. Not in the omat yaml. |
| Reads | the target, `atomic_mu_i`, `batch_tot_nat_real` |
| Writes | **`atomic_energy`**, `[n_atoms, 1]` |
| Variables | `at_shifts` |

A learned energy offset per chemical element (the "atomic reference energy"), added to the energy of every real atom.

## 1. Constructor arguments

`output.py:870-878`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `target` | required | a `CreateOutputTarget` (stored as a name) | The accumulator. |
| `number_of_atom_types` | required | integer | Number of elements (89 in large_base); first axis of `at_shifts`. |
| `name` | `"TrainableShiftTarget"` | any unique `str` | Key of the instruction. |
| `l` | `0` | integer | Stored; not compared with the target (no `assert_l_compatibility`). |

## 2. Derived tables

None.

## 3. Parameters

Name, shape, dtype and trainability are **[V]** against `probe_model_grace_2L_omat_large_base.json`, both parameter dtypes (name
`tr_atomic_shift:0`, `[89, 1]`, parameter dtype, trainable, checkpoint key `model/instructions/TrainableShiftTarget/at_shifts/...`).
The initial value (zeros) is read from the code, not from the probe, which uses random weights.

| Attribute | TF name | Shape | dtype | Init |
|---|---|---|---|---|
| `at_shifts` | `tr_atomic_shift:0` | `[number_of_atom_types, 1]`, `[89, 1]` | `float_dtype` | **zeros**, trainable |

It is a real weight of the model (it holds the fitted reference energies), so it is in the checkpoint and the twin reads it.

## 4. Runtime constants

None.

## 5. Forward

`output.py:889-906`:

```
shift         = gather(at_shifts, atomic_mu_i, axis=0)                    # [n_atoms, 1]
shift         = where(real, shift, 0)                                      # real[a] = a < batch_tot_nat_real
atomic_energy = target + cast(shift, target.dtype)
```

No scale, no averaging. `where(real, ...)` is the identity in the twin (no padding atoms).

The formula was re-implemented in numpy and compared with TF on synthetic targets (difference 0) and on the large_base chain,
`1.7e-18` on values up to `5e-2` (float64, seeded weights).

## 6. Dtype and promotion

The result has the dtype of the target (float32 in a float32 model, **Measured**). With a float32 target and the instruction
built in float64, `tf.where(..., tf.zeros_like(target))` raises `InvalidArgumentError` before the cast (TEST6 finding 3,
pinned at `tests/test_output.py:1323`). A consistent model does not meet it.

## 7. Options rejected

- `upd_init_args_new_elements` changes only `_init_args`, not `number_of_atom_types`, and
  `prepare_variables_for_selected_elements` (`output.py:908-916`) is element restriction: out of scope (decision D19; TEST6
  finding 18).
- `local = True`: not applicable.
- `TrainableShiftTarget_v2` (same file, a shift as a projection of the chemical embedding) is not used by the 2L models and has
  no sheet: an error naming the class if a yaml contains it.

## 8. Golden-fixture keys

The final `atomic_energy` `[n_atoms, 1]` of the large_base chain, and the weight `at_shifts` `[89, 1]`. FIX2 produces the
yaml fixtures; no achieved error exists yet. TF oracle: `tests/test_output.py:1247-end` (numpy re-computation, padding atoms).

## For the reviewer

None beyond the family-wide proposals.
