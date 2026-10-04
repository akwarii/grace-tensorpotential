# SphericalHarmonic

| | |
|---|---|
| Source | `instructions/compute.py:241-277` (class at 241); the maths is `functions/spherical_harmonics.py:8-272` (`SphericalHarmonics`) |
| Family | geometry and radial |
| Base | `TPEquivariantInstruction` (`instructions/base.py:493`) |
| Used in | both 2L yamls, name `Y`, `vhat` = `ScaledBondVector`, `lmax: 4` |
| Reads | the entry named by `vhat`, `[n_bonds, 3]` |
| Writes | `Y` `[n_bonds, (lmax + 1)**2]` |
| Variables | none (the recursion tables are constants) |

Real spherical harmonics of the bond direction, up to degree `lmax`, in the layout of G7 and with the factor
`sqrt(4 pi)` already applied, so that `Y_00 = 1`.

## 1. Constructor arguments

| Argument | Default | Values | Effect |
|---|---|---|---|
| `vhat` | required | a `TPInstruction` or a `str` | Name of the entry with the unit directions (`compute.py:259-264`); anything else raises `ValueError`. |
| `name` | required | any unique `str` | Key of the output. |
| `lmax` | required | integer `>= 0` | Highest degree; the output has `(lmax + 1)**2` columns. Stored as `self.lmax`. |
| `norm` | `False` | `False`, `True` | Passed to `SphericalHarmonics` through `**kwargs` (`compute.py:266`). `False` multiplies by `sqrt(4 pi)` (`Y_00 = 1`); `True` leaves the orthonormal harmonics (`Y_00 = 1/sqrt(4 pi) = 0.28209`). The flag name is the reverse of what the docstring says (`spherical_harmonics.py:17-18`). |
| `type` | `"real"` | `"real"`, `"complex"` | `"complex"` constructs, but the forward pass fails (below). |

Any other keyword raises `TypeError` from `SphericalHarmonics.__init__` (`spherical_harmonics.py:26`). Because
`**kwargs` is captured, `norm` and `type` appear as flat keys of the instruction in `model.yaml` when given. Both
2L yamls give neither, so `norm = False`, `type = "real"`.

## 2. Derived tables

Metadata table `coupling_meta_data` (pandas, built in `__init__`, `compute.py:267`, by
`init_uncoupled_meta_data`, `instructions/base.py:516-528`): one row per output column with `l`, `m`, `hist = ""`,
`parity = (-1)**l`, `sum_of_ls = l`; shape `((lmax + 1)**2, 5)`. Downstream instructions (`SingleParticleBasisFunction*`)
read `l`, `m` and `parity` from it to build their own coupling tables. The twin builds the same table with numpy.
`coupling_origin` is `None`.

Recursion tables `alm`, `blm` (`spherical_harmonics.py:60-83`), built in `build`, indexed by
`lm1d(l, m) = m + l (l + 1) / 2` for `0 <= m <= l` (`spherical_harmonics.py:49-51`). For `l >= 1` and `m < l`

```
a[l, m] = sqrt( (4 l**2 - 1) / (l**2 - m**2) )
b[l, m] = -sqrt( ((l - 1)**2 - m**2) / (4 (l - 1)**2 - 1) )
a[l, l] = -sqrt( 1 + 1 / (2 l) )                    (the diagonal; b[l, l] is a placeholder 0)
```

float64 constants, row 0 is 0 (`alm = [0.0]`, `blm = [0.0]`). The twin generates them with numpy at load time; they
are not weights.

## 3. Parameters

None.

## 4. Runtime constants

(**[V]**: names, dtypes and shapes agree with the three probe files for both parameter dtypes; `PI`, `factor4pi`, `alm` and `blm` are
float64 even when the parameters are float32, `l_tile` is an int32 vector of length `(lmax + 1)**2`, `alm` and `blm` have
one entry per pair `0 <= m <= l`, that is `(lmax + 1)(lmax + 2)/2`, here 6 and 15 for `lmax` 2 and 4) `PI = float64(pi)`, `factor4pi = sqrt(4 pi)`, `l_tile` (int32), `alm`, `blm`, all created in `build`
(`spherical_harmonics.py:36-47`). **`build` ignores its `float_dtype` argument**: `float_dtype` is set to `tf.float64`
whatever the model's parameter dtype (`spherical_harmonics.py:37`; pinned at `tests/test_compute.py:558`).

## 5. Forward

`compute.py:275-277` calls `SphericalHarmonics.__call__` (`spherical_harmonics.py:265-272`), which for
`type = "real"` runs `_compute_rsh` (241-262).

1. The input is converted to float64 (`_compute_sph_harm`, 163-198) and `(x, y, z) = rhat`. The input has to be a
   unit vector: the code uses `sin(theta) exp(i phi) = x + i y` and `cos(theta) = z` and never normalises.
2. Normalised associated Legendre functions `P[l, m](z)`, `m >= 0` (`legendre`, 84-115): `P00 = 1/sqrt(4 pi)`,
   `P10 = sqrt(3/(4 pi)) z`, `P11 = -sqrt(3/(8 pi))`; for `l >= 2`: `P[l, l] = a[l, l] P[l-1, l-1]`,
   `P[l, l-1] = sqrt(2 (l-1) + 3) z P[l-1, l-1]`, otherwise `P[l, m] = a[l, m] (z P[l-1, m] + b[l, m] P[l-2, m])`.
3. Complex harmonics `Y[l, m] = (x + i y)**m P[l, m]` for `m >= 0`, with the powers formed by repeated complex
   multiplication (189-196). The list is ordered by `m`, then `l`, and addressed by
   `lmsh(l, m) = l + |m| lmax - |m| (|m| - 1) / 2` (52-53).
4. Real harmonics (241-262), for each `l` and `m = -l .. l`:

   ```
   m < 0 :  S[l, m] = sqrt(2) * (-1)**m * Im Y[l, |m|]
   m = 0 :  S[l, 0] = Re Y[l, 0]
   m > 0 :  S[l, m] = sqrt(2) * (-1)**m * Re Y[l, m]
   ```

   The `(-1)**m` cancels the Condon-Shortley sign that the recursion carries, so these are the real harmonics
   without the Condon-Shortley phase (`tests/test_compute.py:501-721` checks them against `scipy.special.sph_harm_y`).
5. Unless `norm`, every entry is multiplied by `sqrt(4 pi)` (259-260).

Hand values (**Measured** on the pinned commit, `lmax = 2`, `norm = False`; they follow from the formulas above):

| direction | `l=0` | `l=1`, `m=-1,0,1` | `l=2`, `m=-2..2` |
|---|---|---|---|
| `z` | 1 | `0, sqrt3, 0` | `0, 0, sqrt5, 0, 0` |
| `x` | 1 | `0, 0, sqrt3` | `0, 0, -sqrt5/2, 0, sqrt15/2` |
| `y` | 1 | `sqrt3, 0, 0` | `0, 0, -sqrt5/2, 0, -sqrt15/2` |
| zero vector | 1 | `0, 0, 0` | `0, 0, -sqrt5/2, 0, 0` |

So `l = 1` is `sqrt3 (y, z, x)`. The zero vector (which `ScaledBondVector` produces for a zero bond) gives the
harmonics of the point `z = 0`, not an error.

Cost note for the twin: the TF graph is Python-unrolled over `(l, m)`; there are no data-dependent shapes.

## 6. Dtype and promotion

Always float64: the input is converted with `convert_to_tensor(..., dtype=float64)` and every constant is
float64 (`spherical_harmonics.py:37`, `165`). A float32 model therefore gets float64 harmonics (TEST6 finding 2).
A float32 **tensor** input would raise at the conversion; this does not occur (G1). The twin computes the
harmonics in float64 whatever the parameter dtype, and downstream classes decide whether to cast them (see
`SingleParticleBasisFunctionScalarInd` and `...EquivariantInd`).

## 7. Options rejected

- `type = "complex"`. The constructor accepts it, and the forward pass raises `InvalidArgumentError`
  (`_compute_ylm` applies a rank-3 permutation to a rank-2 stack, `spherical_harmonics.py:230-235`; TEST6
  finding 9, pinned at `tests/test_compute.py:608`). No shipped model uses it and TF cannot run it. Proposal:
  the twin raises an actionable error naming the option when the yaml has `type: complex`.
- `norm = True` is supported (a factor), although no shipped model uses it.

## 8. Golden-fixture keys

Fixture key `Y` (omat and large_base), `[n_bonds, 25]`. FIX2 produces the fixtures; no achieved error exists
yet. TF oracles already in the suite: `scipy.special.sph_harm_y`, hand formulas for `l <= 2`, orthonormality by
Gauss-Legendre quadrature, parity `(-1)**l`, the addition theorem and rotation covariance with Wigner matrices
(`tests/test_compute.py:501-721`).

## For the reviewer

1. Decided (owner, 2026-10-04): `type: complex` is rejected at load (section 7), because TF itself fails on it.
