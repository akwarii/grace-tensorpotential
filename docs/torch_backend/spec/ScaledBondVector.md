# ScaledBondVector

| | |
|---|---|
| Source | `instructions/compute.py:196-237` (class at 197), base `TPInstruction` (`instructions/base.py:327`) |
| Family | geometry and radial |
| Used in | both 2L yamls, name `ScaledBondVector`, `bond_length` = the `BondLength` instruction |
| Reads | `bond_vector` `[n_bonds, 3]` and the bond-length entry `[n_bonds, 1]` |
| Writes | `ScaledBondVector` `[n_bonds, 3]` |
| Variables | none |

The bond vector divided by the softened bond length: a vector of norm `r / sqrt(r**2 + 1e-10)`, which is 1 to
within `5e-11` for `r = 1 A`. It is the direction that `SphericalHarmonic` consumes.

## 1. Constructor arguments

| Argument | Default | Values | Effect |
|---|---|---|---|
| `bond_length` | required | a `TPInstruction` or a `str` | Name of the data entry holding the lengths (`compute.py:217-220`). |
| `bonds` | `None` | `None`, a `TPInstruction` or a `str` | Name of the vector entry: `bond_vector` when `None` (`compute.py:221-227`). |
| `name` | `"ScaledBondVector"` | any unique `str` | Key of the output. |

The omat yaml stores only `bond_length`; the large_base yaml also stores `bonds: null`. Both mean the default.

A `bond_length` or a `bonds` of another type is **silently ignored**: the attribute stays undefined and the
failure is an `AttributeError` at the first forward pass (TEST6 finding 10, pinned at `tests/test_compute.py:399`).
The twin raises `TypeError` at load time.

## 2. Derived tables

None.

## 3. Parameters

None. `build` only sets `is_built` (`compute.py:229-232`).

## 4. Runtime constants

None of its own. The `1e-10` it depends on is inside the bond length (see the sheet of `BondLength`).

## 5. Forward

```
vhat[b, :] = v[b, :] / d[b, 0]            v = data[bonds], d = data[bond_length]
```

`compute.py:234-237`: a broadcast division `[n_bonds, 3] / [n_bonds, 1]`. No `where`, no epsilon of its own.

A zero vector gives exactly `0 / 1e-5 = 0`, and the gradient there is finite, because `d` is never 0. The
output of a zero vector is the zero vector, which is not a unit vector; `SphericalHarmonic` then returns the
harmonics of the zero vector (see its sheet), not an error.

## 6. Dtype and promotion

The output follows the promotion of the two inputs; both are float64 in every shipped use (G1), so the
output is float64. TF performs no cast of its own.

## 7. Options rejected

None.

## 8. Golden-fixture keys

Fixture key `ScaledBondVector`. Fixtures come from FIX2; no achieved error exists yet. The TF reference values
are the hand vectors, the unit-norm check, the rotation covariance and the inversion test of
`tests/test_compute.py:377-500`.

## For the reviewer

None.
