# CreateOutputTarget

| | |
|---|---|
| Source | `instructions/output.py:30-47` (class at 30), base `TPInstruction` (`instructions/base.py:328`) |
| Family | norm and output |
| Used in | both 2L yamls: `atomic_energy` with `initial_value 0.0`, `l 0`; always the first output instruction |
| Reads | nothing |
| Writes | `atomic_energy`, a scalar tensor `[]` |
| Variables | none |

The accumulator of the energy. The output instructions that follow (`LinMLPOut2ScalarTarget`, the shift and scale
targets) read it, add their term and write the result **back under the same name**.

## 1. Constructor arguments

`output.py:32-38`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `name` | required | any unique `str` | For the energy it has to be `atomic_energy`: the model reads `data["atomic_energy"]` (`constants.py:59`, `tpmodel.py:296`). |
| `initial_value` | `0.0` | a number or `None` | The starting value; `None` becomes `0.0`. Both yamls: `0.0`. |
| `l` | `0` | integer | Tensor rank of the target; compared by the dependants (`assert self.l == target.l`, `output.py:60-63`). Both yamls: `0`. |

## 2. Derived tables

None.

## 3. Parameters

None.

## 4. Runtime constants

`value`: `reshape(constant(initial_value, float_dtype), [])`, built in `build` (`output.py:40-44`). Not in the checkpoint
(it comes from the yaml). **[V]** against the three probe files: `value` is a scalar tensor (`[]`) in the parameter dtype.

## 5. Forward

`output.py:46-47`: `return self.value`, a **scalar** (shape `[]`), not a per-atom tensor. The first output instruction that
follows broadcasts it: `target + origin` with `origin` of shape `[n_atoms, 1]` gives `[n_atoms, 1]`. The model then reads
`atomic_energy` as `[n_atoms, 1]` (`tpmodel.py:296`, `reshape(..., [-1, 1])`) and sums it per structure. In the twin
`atomic_energy` is therefore `initial_value + sum of the output terms`, per atom.

Output chain of the two 2L models (the order of the yaml is the order of execution, and every output instruction
overwrites `atomic_energy`, so a dump of the dictionary shows only the last value):

| Model | Chain |
|---|---|
| omat | `atomic_energy = 0` -> `LinMLPOut2ScalarTarget` (name `MLPOut2ScalarTarget`) -> `ConstantScaleShiftTarget` |
| large_base | `atomic_energy = 0` -> `LinMLPOut2ScalarTarget` -> `TrainableShiftTarget` |

## 6. Dtype and promotion

`value` has the parameter dtype. Before `build` the forward pass returns a **Python float** (no tensor): pinned at
`tests/test_output.py:351` (TEST6 finding 18). A second `build` in another dtype is a silent no-op (`is_built`). The result
of the chain is float32 for float32 parameters (**Measured**).

## 7. Options rejected

- `l != 0` for the energy chain: the 2L models only have `l = 0`; `CreateOutputTarget` itself works for any `l` (it only
  stores it). **Proposal**: the twin supports `l = 0` and raises on a model whose output chain has another `l`
  (forces and stress are derived by autograd, not as separate targets).
- `initial_value != 0.0`: ported (a constant added to the energy of every atom).

## 8. Golden-fixture keys

None of its own: the key `atomic_energy` of the fixtures is the end of the whole chain (`[n_atoms, 1]`), recorded as the
final value of the dictionary entry. FIX2 produces the fixtures; no achieved error exists yet. TF oracle:
`tests/test_output.py:309-378`.

## For the reviewer

None beyond the family-wide proposals.
