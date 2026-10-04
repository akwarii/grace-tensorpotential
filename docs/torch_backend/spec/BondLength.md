# BondLength

| | |
|---|---|
| Source | `instructions/compute.py:156-193` (class at 157), base `TPInstruction` (`instructions/base.py:327`) |
| Family | geometry and radial |
| Used in | both 2L yamls, name `BondLength`, as `BondLength(instruction_with_bonds=None)` |
| Reads | `bond_vector` `[n_bonds, 3]` (`compute.py:163`) |
| Writes | `BondLength` `[n_bonds, 1]` |
| Variables | none |

The Euclidean norm of every bond vector, softened by `1e-10` under the square root, so that the square root has
a finite value and a zero gradient at the zero vector.

## 1. Constructor arguments

| Argument | Default | Values | Effect |
|---|---|---|---|
| `instruction_with_bonds` | `None` | `None`, a `TPInstruction`, or a `str` | Name of the data entry that holds the vectors: `bond_vector` when `None`, else the name of the instruction or the string (`compute.py:165-178`). Any other type raises `TypeError`. |
| `name` | `"BondLength"` | any unique `str` | Key under which the output is stored. Consumers refer to it by this name. |

Both shipped yamls store `instruction_with_bonds: null` and no `name`, so the pinned default name applies.

## 2. Derived tables

None.

## 3. Parameters

None. `build` only sets `is_built` (`compute.py:180-182`).

## 4. Runtime constants

`1e-10`, a Python float written into the forward pass (`compute.py:192`). It takes the dtype of the tensor it is
added to. It is not stored in the yaml and not configurable: a twin has to hard-code it.

## 5. Forward

```
d[b] = sqrt( sum_k v[b, k]**2 + 1e-10 )          v = data[<instruction_with_bonds or "bond_vector">]
```

`compute.py:186-193`, `reduce_sum(..., axis=1, keepdims=True)`. Output `[n_bonds, 1]`.

Consequences, all pinned by `tests/test_compute.py:251-376`:

- The length of the zero vector is `sqrt(1e-10) = 1e-5`, not 0, and the gradient at the zero vector is exactly 0
  (`d/dv sqrt(v.v + eps) = v / d`).
- For `|v| = 1 A` the softened length differs from the true one by about `5e-11` (relative): negligible for
  physics, but a twin that omits the `1e-10` is **not** bit-comparable, and the difference is above float64
  round-off.

## 6. Dtype and promotion

The output has the dtype of the input (float64 by G1). There is no parameter, so the parameter dtype plays no
role. The twin keeps the dtype of its input.

## 7. Options rejected

None are needed: the only branch is the `TypeError` for an `instruction_with_bonds` that is neither `None`, a
`str` nor a `TPInstruction`; the twin raises the same error type at load time.

## 8. Golden-fixture keys

Fixture key: the instruction name, `BondLength`, as in the dictionary dumped by `tools/oracle_snapshot.py`.
Fixtures are produced by FIX2; no achieved error exists yet. The TF reference values are the hand vectors and
the `numpy.linalg.norm` oracle of `tests/test_compute.py:251-376`.

## For the reviewer

None.
