# RadialBasis

| | |
|---|---|
| Source | `instructions/compute.py:347-416` (class at 347); the maths is in `functions/radial.py:11-358` |
| Family | geometry and radial |
| Base | `TPInstruction` (`instructions/base.py:328`) |
| Used in | both 2L yamls, name `RadialBasis`, `bonds` = `BondLength`, `basis_type: Cheb` (omat: `nfunc 8, p 5, rcut 6.0`; large_base: `nfunc 10, p 16, rcut 6.0`) |
| Reads | the bond-length entry, `[n_bonds, 1]` |
| Writes | `RadialBasis` `[n_bonds, nfunc]` |
| Variables | `rc` (non-trainable scalar); `grid` and `scale` for the Gaussian basis only |

Selects one of four radial bases, evaluates it on the bond lengths and sets it to 0 beyond the cutoff.

## 1. Constructor arguments

`RadialBasis(bonds, basis_type, name="RadialBasis", **kwargs)` (`compute.py:367-394`). `kwargs` go to the basis
class and are captured **flat** in the yaml (`nfunc`, `p`, `rcut`, ...).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `bonds` | required | `TPInstruction` or `str` | Name of the length entry; anything else raises `ValueError`. |
| `basis_type` | required | `"Cheb"`, `"RadSinBessel"`, `"SBessel"`, `"Gaussian"` | Selects the class (`compute.py:384-393`); any other value raises `ValueError`. Both 2L yamls use `Cheb`. |
| `name` | `"RadialBasis"` | any unique `str` | Key of the output. |
| `nfunc` | required | integer `>= 1` | Number of basis functions (the width of the output); `< 1` raises `ValueError` (`radial.py:35-40`). |
| `rcut` | required | float `> 0` | Cutoff radius; `<= 0` raises `ValueError` (`radial.py:41-46`). |

Basis-specific keywords. Unknown keywords are **accepted and ignored** (every basis class ends in `**kwargs`; a
misspelt `normalised` is stored in the yaml and has no effect). Measured: `RadialBasis(..., bogus=1)` constructs.

| `basis_type` | keyword | Default | Values / effect |
|---|---|---|---|
| `Cheb` (`radial.py:133-155`) | `p` | `5` | Order of the envelope polynomial (`cutoff_func_p_order_poly`). |
| | `normalized` | `False` | `True` multiplies the basis by `sqrt(1/pi)`. |
| | `kind` | `1` | `1`: Chebyshev polynomials of the first kind; `2`: of the second kind. Any other value raises `ValueError("kind must be 1 or 2")` **at the first forward pass**, not at construction (`radial.py:322-324`). |
| | `reversed` | `False` | `True` flips the sign of the rescaled argument. |
| `RadSinBessel` (`radial.py:178-218`) | `p` | required | Envelope order. |
| | `normalized` | `False` | `True` replaces the plain basis by a standardised one (section 5). |
| `Gaussian` (`radial.py:73-101`) | `p` | required | Envelope order. |
| | `rmin` | `0.0` | Lower end of the centre grid. |
| | `init_gamma` | `1.0` | Width factor. |
| | `trainable` | `False` | Whether `grid` and `scale` are trainable (no effect at inference). |
| | `normalized` | `False` | Accepted and **ignored** (the code using it is commented out, `radial.py:99-101`). |
| `SBessel` (`radial.py:257-268`) | none | | No `p`: this basis has no envelope. |

`Gaussian` with `nfunc = 1` raises `IndexError` in the constructor (`grid[0, 1]`, `radial.py:95-96`; **Measured**).

## 2. Derived tables

None as tables. The Gaussian `grid` is `linspace(rmin, rcut, nfunc)` reshaped to `[1, nfunc]` and its `scale` is
`-0.5 / (init_gamma * (grid[1] - grid[0]))**2`, both computed in the constructor (`radial.py:94-97`) and stored as
variables (below). For `RadSinBessel` with `normalized`, `mu[n]` and `sigma[n]` for `n = 1 .. nfunc` are
computed in the constructor with `scipy.special.sici` (`radial.py:206-218`):

```
mu[n]     = sqrt(2) * Si(n pi)
sigma2[n] = 2 n pi * Si(2 n pi) + (cos(2 n pi) - 1) - 2**(3/2) * mu[n] * Si(n pi) + mu[n]**2
sigma[n]  = sqrt(sigma2[n])                   (urcut = 1.0)
```

`Si` is the sine integral. They are recomputed from `nfunc` at load time; they are not weights.

## 3. Parameters

All non-trainable; none is a learned weight of a shipped model. `rc` is **[V]** against the three probe files (SBessel in
`model_grace`, Cheb in the two 2L models; both parameter dtypes: `RadialBasis/cutoff:0`, shape `[]`, `float32` for a float32
model and `float64` for a float64 one, `trainable = False`, saved under `model/instructions/RadialBasis/rc/...`). `grid` and
`scale` are **[I]**: no probed yaml uses the Gaussian basis.

| Attribute | TF name | Shape | dtype | Notes |
|---|---|---|---|---|
| `basis_function.rc` (alias `RadialBasis.rc`, `compute.py:409`) | `RadialBasis/cutoff:0` | `[]` | `float_dtype` (the model's parameter dtype) | `Variable(rcut)` (`radial.py:52-54`, `225-227`). Present for all four bases. |
| `basis_function.grid` | `RadialBasis/Variable:0` | `[1, nfunc]` | `float_dtype` | Gaussian only; `trainable` as given. |
| `basis_function.scale` | `RadialBasis/Variable:0` | `[]` | `float_dtype` | Gaussian only; negative. |

The checkpoint therefore holds `rc` for every model: it is data, not a derived value, and the twin has to read it
(rule R6) rather than recompute it from `rcut`; see section 6 for why the two differ for a float32 model.

## 4. Runtime constants

Created in `build` (`radial.py:49-56`, `220-235`). `PI` and `epsilon` are **[V]** for the bases the probed yamls use (`basis_function/PI` and
`basis_function/epsilon`, scalar tensors of dtype float64 for both parameter dtypes; the value `1e-10` is read from the code). `urc`, `mu`, `sigma`
and `norm` are **[I]**: no probed yaml uses `RadSinBessel`, or `Cheb` with `normalized = True` (`norm` does not appear in
the probed attributes of the `normalized: false` Cheb):

| Constant | Value | dtype | Used by |
|---|---|---|---|
| `PI` | `pi` | float64 always | `RadSinBessel`, `SBessel` |
| `epsilon` | `1e-10` (base, used by `Cheb`, `Gaussian`, `SBessel`); **`1e-8` for `RadSinBessel`** (`radial.py:228`) | float64 always | the `r == 0` substitution below |
| `urc`, `mu`, `sigma` | `1.0`, the tables above | float64 | `RadSinBessel` with `normalized` |
| `norm` | `sqrt(1/pi)` | float64 | `Cheb` with `normalized` (`radial.py:154-161`) |

The envelope exponent `p` and `nfunc` are Python numbers fixed at construction.

## 5. Forward

Entry point (`radial.py:65-70`), for the length tensor `r` of shape `[n_bonds, 1]`:

```
r'    = where(r == 0.0, r + epsilon, r)               # exact float comparison with 0
basis = compute_basis(r')                              # [n_bonds, nfunc]
out   = where(r' > rcut, 0, basis)                     # strict >, rcut the Python float
```

The envelope, used by `Cheb`, `RadSinBessel` and `Gaussian` (`radial.py:302-308`):

```
E(x; p) = 1 - (p + 1)(p + 2)/2 * x**p + p (p + 2) * x**(p + 1) - p (p + 1)/2 * x**(p + 2)
```

`E(0) = 1`, `E(1) = 0`, and `E` has zero first and second derivatives at `x = 1`.

**Cheb** (`radial.py:164-175`), with `rc = cast(rc_variable, r.dtype)`, `x = r / rc`:

```
u     = 2 (1 - |1 - x|) - 1            (reversed: -u)             # in [-1, 1] for r <= 2 rc
T_k   = chebvander(u, nfunc + 1, kind)[:, 1:]                     # degrees 1 .. nfunc, [n_bonds, nfunc]
basis = T_k * E(x; p)         (normalized: * sqrt(1/pi))
```

`chebvander` (`radial.py:315-332`) is the recurrence `T_0 = 1`, `T_1 = u` (`2u` for `kind = 2`),
`T_k = 2u T_{k-1} - T_{k-2}`. Degree 0 is dropped. The exponent of `|1 - x|` is `lmbda = 1` (not an argument).

**RadSinBessel** (`radial.py:237-254`), `n = 1 .. nfunc` (in `r.dtype`), `rc` the float variable:

```
plain:       sqrt(2 / rc) * sin(n pi r / rc) / r * E(r / rc; p)
normalized:  s = r / rc;  ( sqrt(2) * sin(n pi s) / s - mu[n] ) / sigma[n] * E(s; p)
```

**Gaussian** (`radial.py:121-130`): `exp(scale * (r - grid)**2) * E(r / rc; p)`, `grid` of shape `[1, nfunc]`.

**SBessel** (`radial.py:271-299`), no envelope. With `fn(r, n) = (-1)**n sqrt(2) pi / rc**(3/2) * (n+1)(n+2) /
sqrt((n+1)**2 + (n+2)**2) * ( sinc((n+1) pi r / rc) + sinc((n+2) pi r / rc) )`, `sinc(x) = sin(x)/x` with
`sinc(0) = 1` (`radial.py:311-312`), and for `n = 0 .. nfunc - 1`:

```
d_0 = 1,  e_n = n**2 (n + 2)**2 / (4 (n + 1)**4 + 1),  d_n = 1 - e_n / d_{n-1}
B_0 = fn(r, 0)
B_n = ( fn(r, n) + sqrt(e_n / d_{n-1}) * B_{n-1} ) / sqrt(d_n)
```

The recursion runs over `nfunc` Python-unrolled terms, and the output is `[n_bonds, nfunc]` with `B_n` in
column `n`.

Behaviours that shape the twin (all **Measured** on the pinned commit unless a test is cited):

- `r == 0` is the only value replaced; a length of `1e-12` is not. `BondLength` never returns 0 (it returns at
  least `1e-5`), so the substitution is dormant in a model run and live in a direct call.
- At `r == rc` the value is 0 for `Cheb`, `RadSinBessel` and `Gaussian` (`E(1) = 0`), and for any `r > rcut`. For
  `SBessel`, which has no envelope, only the strict `r > rcut` test zeroes the value.
- `Cheb`: the second derivative with respect to `r` is `NaN` exactly at `r == rc` and finite just inside; the first
  derivative at `rc` is 0 (TEST6 finding 4, pinned at `tests/test_compute.py:890`).
- `Cheb` at `r = 0` (replaced by `1e-10`): `T_k(-1) = (-1)**k`, so the row is `(-1, 1, -1, ...)`.

## 6. Dtype and promotion

- The output is float64 for float64 data whatever the parameter dtype: the float64 constants (`epsilon`, `PI`,
  `norm`, `mu`, `sigma`) promote the distance (G2; TEST6 finding 2, pinned at `tests/test_compute.py:1093`).
- **The cutoff is stored in the parameter dtype and used in two precisions.** `rc` is a `float_dtype` variable.
  `Cheb` casts it to the data dtype (`radial.py:165`), so a float32 model evaluates the basis with
  `float32(rcut)` widened to float64; the final `where` and `get_cutoff` for an unbuilt object use the Python
  float `rcut`. For `rcut = 6.0` the two are equal. For `rcut = 5.6` and float32 parameters, **Measured**: `rc =
  5.599999904632568`; at `r = 5.6` the `Cheb` basis is `5.3e-15` (not 0), because the `where` test (`r > 5.6`)
  does not fire and the envelope is evaluated at `x = 1.0000000170`. `RadSinBessel`, `Gaussian` and `SBessel`
  use `self.rc` (float32) directly, which promotes to float64 in the same way. The twin has to keep `rc` in the
  parameter dtype, cast it to float64 at the use, and keep the Python-float `rcut` for the strict `>` test.
- `get_cutoff()` returns `float(rc)` once built and `float(rcut)` before (`compute.py:396-403`). For a float32
  model it returns `5.599999904632568` for `rcut = 5.6`. The neighbour-list cutoff of a calculator comes from it.
- A float32 **data** tensor raises `InvalidArgumentError` in the `r == 0` substitution (`r + epsilon` is float64
  against a float32 `r`; **Measured**, not pinned by a test). This does not occur with the default data dtype
  (G1).

## 7. Options rejected

- Unknown `basis_type`: `ValueError` already in TF; same in the twin.
- `Cheb` with `kind` not in `{1, 2}`: TF raises at the first forward pass; the twin raises at load time.
- Unknown extra keywords: TF ignores them. **Proposal**: the twin raises on a keyword that the chosen basis does
  not read, naming it, because a silent no-op in a saved model (a misspelt `normalized`) would otherwise
  produce a wrong energy without a message (rule R3). A keyword that is read but has no effect (`Gaussian`
  `normalized`) is accepted only at its default.
- The commented-out normalisation of the Gaussian basis is not part of the contract.

## 8. Golden-fixture keys

Fixture key `RadialBasis`, `[n_bonds, nfunc]` (`8` for omat, `10` for large_base). Both shipped yamls use `Cheb`,
so the other three bases have no yaml fixture: they are tested against the closed forms of
`tests/test_compute.py:722-1111` (sine Bessel and its normalisation by `scipy.integrate.quad`, Chebyshev
polynomials of both kinds, Gaussians, orthonormality of the simplified spherical Bessel functions against
`r**2 dr`, envelope values at 0 and at the cutoff). FIX2 produces the fixtures; no achieved error exists yet.

## For the reviewer

1. **Scope of the bases.** Rule R3 says every option of an in-scope class is ported. The shipped models use only
   `Cheb` with `kind = 1`, `reversed = False`, `normalized = False`. Proposal: port all four bases and both
   Chebyshev kinds (they are short and have TF oracles), and say in the sheet of the ticket which of them the
   2L gates exercise (`Cheb` only). Alternative: port `Cheb` first and reject the rest with an error until a model
   needs them.
2. **Proposal (section 7)**: reject unread keywords instead of ignoring them.
3. **`rc` in two precisions (section 6)** is a TF quirk the twin copies on purpose, so that float32 models agree
   with TF. Alternative: use one precision and document the `5e-15` tail as a tolerated difference. It only
   matters for a cutoff that is not exactly representable in float32.
