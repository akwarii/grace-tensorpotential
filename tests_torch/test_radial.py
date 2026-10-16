"""The radial bases (``torch_backend/ops/radial.py``) against numpy, scipy, quadrature and the TF goldens.

Logic: the envelope and its derivatives at the cutoff, the Chebyshev recurrence, the zero-length substitution, the zero
beyond ``rcut``, the cutoff in the parameter dtype, the errors, the dtype audit and ``gradcheck`` of every basis.
Physical values: each basis against an oracle that does not call it (numpy and scipy closed forms, Gauss-Legendre
quadrature for the normalisations and the orthonormality), and the Chebyshev basis of both 2L models against the
``RadialBasis`` outputs of the TF goldens (nfunc 8, p 5 and nfunc 10, p 16).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from scipy.special import eval_chebyt, eval_chebyu, sici

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
)
from tensorpotential.torch_backend.ops import radial
from tests_torch.dtype_audit import DtypeAudit
from tests_torch.golden import GOLDEN, TINY_F32, TINY_F64, cases, load
from tests_torch.golden.tolerances import FEATURE_F64, assert_close

RCUT = 6.0
F64 = torch.float64
F32 = torch.float32
GOLDEN_CONFIGS = {"omat": (8, 5), "large": (10, 16)}
"""(nfunc, p) of the Chebyshev basis of each 2L model, as the issue names them."""


def _rc(value: float = RCUT, dtype: torch.dtype = F64) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype)


def _lengths(n: int, rcut: float = RCUT) -> torch.Tensor:
    return torch.linspace(0.3, 0.98 * rcut, n, dtype=F64).reshape(-1, 1)


def _envelope(x: np.ndarray | float, p: int) -> np.ndarray:
    # E(x) = 1 - (p+1)(p+2)/2 x^p + p(p+2) x^(p+1) - p(p+1)/2 x^(p+2), written as a numpy polynomial
    coefficients = np.zeros(p + 3)
    coefficients[[0, p, p + 1, p + 2]] = [
        1.0,
        -(p + 1) * (p + 2) / 2,
        p * (p + 2),
        -p * (p + 1) / 2,
    ]
    return np.polynomial.polynomial.polyval(x, coefficients)


def _gauss_legendre(a: float, b: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1.0) / 2.0 * (b - a) + a, w * (b - a) / 2.0


def _bases() -> dict[str, tuple[object, tuple[torch.Tensor, ...]]]:
    """One instance of each basis with the extra weights its call needs."""
    grid = torch.linspace(0.0, RCUT, 6, dtype=F64).reshape(1, -1)
    scale = torch.tensor(-0.5 / (RCUT / 5) ** 2, dtype=F64)
    return {
        "Cheb": (radial.Chebyshev(nfunc=6, rcut=RCUT, p=5), ()),
        "Cheb_kind2": (radial.Chebyshev(nfunc=6, rcut=RCUT, p=5, kind=2), ()),
        "RadSinBessel": (radial.SineBessel(nfunc=6, rcut=RCUT, p=5), ()),
        "RadSinBessel_norm": (
            radial.SineBessel(nfunc=6, rcut=RCUT, p=5, normalized=True),
            (),
        ),
        "SBessel": (radial.SimplifiedBessel(nfunc=6, rcut=RCUT), ()),
        "Gaussian": (radial.Gaussian(nfunc=6, rcut=RCUT, p=5), (grid, scale)),
    }


BASES = _bases()
ENVELOPE_BASES = [k for k in BASES if k != "SBessel"]


def _call(key: str, r: torch.Tensor, rc: torch.Tensor | None = None) -> torch.Tensor:
    basis, extra = BASES[key]
    rc = _rc() if rc is None else rc
    weights = tuple(w.to(rc.dtype) for w in extra)
    return basis(r, rc, *weights)  # ty: ignore[call-non-callable]


# ------------------------------------------------------------------ the envelope


@pytest.mark.parametrize("p", [5, 16])
def test_the_envelope_is_the_polynomial_one_at_zero_and_vanishes_to_second_order_at_one(
    p,
):
    x = torch.linspace(0.0, 1.0, 41, dtype=F64, requires_grad=True)
    value = radial.cutoff_envelope(x, p)
    assert_close(value, _envelope(x.detach().numpy(), p), FEATURE_F64)
    (slope,) = torch.autograd.grad(value.sum(), x, create_graph=True)
    (curvature,) = torch.autograd.grad(slope.sum(), x)
    assert value[0].item() == 1.0
    assert_close(value[-1:], [0.0], FEATURE_F64)
    assert_close(slope[-1:], [0.0], FEATURE_F64)
    assert_close(curvature[-1:], [0.0], FEATURE_F64)
    assert torch.all(value[:-1] > 0)


@pytest.mark.parametrize("key", ENVELOPE_BASES)
def test_each_envelope_basis_and_its_first_derivative_vanish_at_the_cutoff(key):
    r = torch.full((1, 1), RCUT, dtype=F64, requires_grad=True)
    out = _call(key, r)
    assert_close(out, np.zeros((1, out.shape[1])), FEATURE_F64)
    for k in range(out.shape[1]):
        (slope,) = torch.autograd.grad(out[0, k], r, retain_graph=True)
        assert_close(slope, [[0.0]], FEATURE_F64, f"{key} function {k}")


def test_the_chebyshev_second_derivative_is_finite_at_the_cutoff_unlike_tf():
    # TF gives NaN exactly at r = rc (the power rule of |1 - r / rc| ** 1 takes 0 ** -1; TEST6 finding 4, pinned in
    # tests/test_compute.py); torch's abs has the derivative sign(0) = 0 there, so the curvature is finite
    for value in (RCUT, 0.99 * RCUT):
        r = torch.full((1, 1), value, dtype=F64, requires_grad=True)
        out = _call("Cheb", r)[0, 1]
        (slope,) = torch.autograd.grad(out, r, create_graph=True)
        (curvature,) = torch.autograd.grad(slope.sum(), r)
        assert torch.isfinite(curvature).all()


# ------------------------------------------------------------------ Chebyshev polynomials


@pytest.mark.parametrize("kind", [1, 2])
def test_the_chebyshev_recurrence_equals_scipy(kind):
    u = torch.linspace(-1.0, 1.0, 33, dtype=F64).reshape(-1, 1)
    out = radial.chebyshev_polynomials(u, 9, kind)
    degrees = np.arange(9)
    evaluate = eval_chebyt if kind == 1 else eval_chebyu
    assert out.shape == (33, 9)
    assert_close(out, evaluate(degrees, u.numpy()), FEATURE_F64)


def test_one_chebyshev_polynomial_is_the_constant():
    u = torch.zeros((2, 1), dtype=F64)
    assert radial.chebyshev_polynomials(u, 1, 2).tolist() == [[1.0], [1.0]]


@pytest.mark.parametrize(
    ("degree", "kind", "match"), [(3, 3, "kind must be 1 or 2"), (0, 1, "at least 1")]
)
def test_the_recurrence_refuses_an_unknown_kind_and_no_degree(degree, kind, match):
    with pytest.raises(OptionValueError, match=match):
        radial.chebyshev_polynomials(torch.zeros((1, 1), dtype=F64), degree, kind)


# ------------------------------------------------------------------ physical values: numpy references


@pytest.mark.parametrize(
    ("kind", "reversed_", "normalized"),
    [(1, False, False), (1, True, False), (2, False, False), (2, True, True)],
)
@pytest.mark.parametrize(("nfunc", "p"), GOLDEN_CONFIGS.values())
def test_the_chebyshev_basis_equals_numpy(kind, reversed_, normalized, nfunc, p):
    basis = radial.Chebyshev(
        nfunc=nfunc,
        rcut=RCUT,
        p=p,
        kind=kind,
        reversed=reversed_,
        normalized=normalized,
    )
    r = _lengths(40)
    x = r.numpy() / RCUT
    u = (1 - 2 * reversed_) * (2.0 * (1.0 - np.abs(1.0 - x)) - 1.0)
    degrees = np.arange(1, nfunc + 1)
    poly = (
        np.polynomial.chebyshev.chebvander(u[:, 0], nfunc)[:, 1:]
        if kind == 1
        else eval_chebyu(degrees, u)
    )
    expected = poly * _envelope(x, p) * (np.sqrt(1.0 / np.pi) if normalized else 1.0)
    assert_close(basis(r, _rc()), expected, FEATURE_F64)


def test_the_sine_bessel_basis_equals_its_closed_form():
    r = _lengths(40)
    n = np.arange(1, 7)
    x = r.numpy()
    expected = (
        np.sqrt(2.0 / RCUT) * np.sin(n * np.pi * x / RCUT) / x * _envelope(x / RCUT, 5)
    )
    assert_close(_call("RadSinBessel", r), expected, FEATURE_F64)


def test_the_moments_are_the_mean_and_deviation_of_the_sine_functions_by_quadrature():
    x, w = _gauss_legendre(0.0, 1.0)
    mu, sigma = radial.sine_bessel_moments(6)
    for n in range(1, 7):
        f = np.sqrt(2.0) * np.sin(n * np.pi * x) / x
        mean = w @ f
        assert_close([mu[n - 1]], [mean], FEATURE_F64, f"mu[{n}]")
        assert_close([sigma[n - 1]], [np.sqrt(w @ f**2 - mean**2)], FEATURE_F64)
    assert_close(mu, np.sqrt(2.0) * sici(np.arange(1, 7) * np.pi)[0], FEATURE_F64)


def test_the_normalised_sine_bessel_basis_is_the_standardised_closed_form():
    # (sqrt(2) sin(n pi s) / s - mean) / deviation * E(s), the mean and deviation over [0, 1] by quadrature
    x, w = _gauss_legendre(0.0, 1.0)
    n = np.arange(1, 7)
    raw = np.sqrt(2.0) * np.sin(n * np.pi * x[:, None]) / x[:, None]
    mean = w @ raw
    deviation = np.sqrt(w @ raw**2 - mean**2)
    r = _lengths(40)
    s = r.numpy() / RCUT
    expected = (
        (np.sqrt(2.0) * np.sin(n * np.pi * s) / s - mean) / deviation * _envelope(s, 5)
    )
    assert_close(_call("RadSinBessel_norm", r), expected, FEATURE_F64)


def _sbessel_raw(r: np.ndarray, nfunc: int) -> np.ndarray:
    """``(-1)^n sqrt(2) pi rc^-3/2 (n+1)(n+2) / sqrt((n+1)^2 + (n+2)^2) (j0 + j0)``; numpy's sinc is sin(pi x) / (pi x)."""
    columns = [
        (-1.0) ** n
        * np.sqrt(2.0)
        * np.pi
        / RCUT**1.5
        * (n + 1)
        * (n + 2)
        / np.sqrt((n + 1) ** 2 + (n + 2) ** 2)
        * (np.sinc(r * (n + 1) / RCUT) + np.sinc(r * (n + 2) / RCUT))
        for n in range(nfunc)
    ]
    return np.stack(columns, axis=1)


def test_the_simplified_bessel_basis_is_orthonormal_with_weight_r_squared():
    r, w = _gauss_legendre(0.0, RCUT)
    g = _call("SBessel", torch.as_tensor(r, dtype=F64).reshape(-1, 1)).numpy()
    assert_close(g.T @ ((w * r**2)[:, None] * g), np.eye(6), FEATURE_F64)


def test_the_simplified_bessel_basis_is_the_gram_schmidt_of_the_raw_functions():
    # the Cholesky factor of the Gram matrix (weight r^2) orthonormalises the raw functions with positive leading
    # coefficients, which is what the recursion computes
    r, w = _gauss_legendre(0.0, RCUT)
    raw = _sbessel_raw(r, 6)
    cholesky = np.linalg.cholesky(raw.T @ ((w * r**2)[:, None] * raw))
    probe = _lengths(25)
    expected = _sbessel_raw(probe.numpy()[:, 0], 6) @ np.linalg.inv(cholesky).T
    assert_close(_call("SBessel", probe), expected, FEATURE_F64)


def test_the_recursion_table_follows_its_definition():
    e, d = radial.simplified_bessel_recursion(4)
    assert d[0] == 1.0
    for n in range(1, 4):
        assert e[n] == n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        assert d[n] == 1 - e[n] / d[n - 1]


def test_the_gaussian_basis_equals_its_closed_form():
    r = _lengths(30)
    centres = np.linspace(0.0, RCUT, 6).reshape(1, -1)
    width = centres[0, 1] - centres[0, 0]
    x = r.numpy()
    expected = np.exp(-0.5 * (x - centres) ** 2 / width**2) * _envelope(x / RCUT, 5)
    assert_close(_call("Gaussian", r), expected, FEATURE_F64)


# ------------------------------------------------------------------ physical values: the TF goldens


def _radial_args(name: str) -> dict[str, Any]:
    manifest = json.loads((GOLDEN / f"{name}.json").read_text())
    return manifest["instructions"]["RadialBasis"]["init_args"]


@pytest.mark.golden
@pytest.mark.parametrize("name", TINY_F64 + TINY_F32)
def test_the_chebyshev_basis_of_both_models_equals_the_tf_golden(name):
    args = _radial_args(name)
    assert args["basis_type"] == "Cheb"
    assert (args["nfunc"], args["p"]) == GOLDEN_CONFIGS[name.split("_")[0]]
    basis = radial.Chebyshev(
        nfunc=args["nfunc"],
        rcut=args["rcut"],
        p=args["p"],
        normalized=args["normalized"],
    )
    arrays = load(name)
    rc = torch.as_tensor(load(f"{name}.weights")["RadialBasis/rc"])
    assert rc.dtype == (F32 if name.endswith("f32") else F64)
    for case in cases(arrays):
        lengths = torch.as_tensor(arrays[f"{case}/ins/BondLength"])
        out = basis(lengths, rc)
        assert out.dtype == F64
        assert_close(
            out, arrays[f"{case}/ins/RadialBasis"], FEATURE_F64, f"{name} {case}"
        )


# ------------------------------------------------------------------ the length and cutoff handling


@pytest.mark.parametrize("key", list(BASES))
def test_beyond_the_cutoff_every_basis_is_exactly_zero(key):
    r = torch.tensor([[RCUT + 1e-9], [RCUT + 1.0], [45.0]], dtype=F64)
    out = _call(key, r)
    assert out.shape == (3, 6)
    assert torch.all(out == 0)


@pytest.mark.parametrize("key", list(BASES))
def test_a_zero_length_is_replaced_by_epsilon_and_gives_finite_values(key):
    basis = BASES[key][0]
    zero = _call(key, torch.zeros((2, 1), dtype=F64))
    at_epsilon = _call(key, torch.full((1, 1), basis.epsilon, dtype=F64))  # ty: ignore[unresolved-attribute]
    assert torch.all(torch.isfinite(zero))
    assert torch.equal(zero[0], at_epsilon[0])


def test_the_epsilons_are_those_of_tf():
    assert radial.SineBessel.epsilon == 1e-8
    assert radial.Chebyshev.epsilon == radial.SimplifiedBessel.epsilon == 1e-10
    assert radial.Gaussian.epsilon == 1e-10


def test_the_chebyshev_basis_at_zero_length_is_its_value_at_epsilon_near_alternating_signs():
    # r = 0 becomes 1e-10, where the argument is -1 + 2e-10 / rc: T_k is (-1)**k up to k**2 times that
    out = _call("Cheb", torch.zeros((1, 1), dtype=F64))[0]
    u = 2.0 * (1.0 - abs(1.0 - radial.EPSILON / RCUT)) - 1.0
    degrees = np.arange(1, 7)
    assert_close(
        out, eval_chebyt(degrees, u) * _envelope(radial.EPSILON / RCUT, 5), FEATURE_F64
    )
    assert np.all(np.sign(out.numpy()) == (-1.0) ** degrees)


def test_a_float32_cutoff_is_used_rounded_and_a_length_at_rcut_is_not_zeroed():
    # TF measured (spec sheet, section 6): for rcut = 5.6 and float32 parameters the Cheb basis at r = 5.6 is 5.3e-15
    rcut = 5.6
    basis = radial.Chebyshev(nfunc=8, rcut=rcut, p=5)
    r = torch.tensor([[rcut]], dtype=F64)
    out32 = basis(r, _rc(rcut, F32))
    rc32 = float(np.float32(rcut))
    x = rcut / rc32
    expected = np.polynomial.chebyshev.chebvander(
        [2.0 * (1.0 - abs(1.0 - x)) - 1.0], 8
    )[:, 1:] * _envelope(np.array([[x]]), 5)
    assert out32.dtype == F64
    assert_close(out32, expected, FEATURE_F64)
    assert 1e-15 < out32.abs().max().item() < 1e-13
    assert torch.all(basis(r, _rc(rcut, F64)) == 0)


@pytest.mark.parametrize("key", ["Cheb", "Cheb_kind2", "RadSinBessel_norm"])
def test_a_float32_model_with_an_exact_cutoff_gives_the_float64_basis(key):
    # rc = 6 is exact in float32 and these bases widen it before any arithmetic
    r = _lengths(12)
    out = _call(key, r, _rc(RCUT, F32))
    assert out.dtype == F64
    assert torch.equal(out, _call(key, r))


@pytest.mark.parametrize(
    ("key", "factor"),
    [
        # TF evaluates sqrt(2 / rc) and rc ** 1.5 in the dtype of rc, then promotes: the basis is the float64 one times
        # the float32 rounding of that factor
        ("RadSinBessel", lambda rc: np.sqrt(2.0 / rc)),
        ("SBessel", lambda rc: rc**1.5),
    ],
)
def test_a_float32_model_rounds_the_sub_expressions_of_its_cutoff_as_tf(key, factor):
    r = _lengths(12)
    out = _call(key, r, _rc(RCUT, F32))
    rounded = float(factor(np.float32(RCUT)))
    exact = factor(RCUT)
    ratio = rounded / exact if key == "RadSinBessel" else exact / rounded
    assert ratio != 1.0
    assert out.dtype == F64
    assert_close(out, _call(key, r).numpy() * ratio, FEATURE_F64)


def test_a_float32_gaussian_uses_its_rounded_weights_widened():
    r = _lengths(12)
    grid, scale = BASES["Gaussian"][1]
    out = _call("Gaussian", r, _rc(RCUT, F32))
    grid32 = grid.numpy().astype(np.float32).astype(np.float64)
    scale32 = float(np.float32(scale.item()))
    x = r.numpy()
    expected = np.exp(scale32 * (x - grid32) ** 2) * _envelope(x / RCUT, 5)
    assert out.dtype == F64
    assert_close(out, expected, FEATURE_F64)


def test_float32_lengths_are_promoted_to_float64_as_in_tf():
    # TF: the float64 epsilon promotes a float32 distance (pinned in tests/test_compute.py)
    r = _lengths(5)
    out = _call("Cheb", r.to(F32))
    assert out.dtype == F64
    assert_close(out, _call("Cheb", r.to(F32).to(F64)), FEATURE_F64)


@pytest.mark.parametrize("rc_dtype", [F32, F64])
@pytest.mark.parametrize("key", list(BASES))
def test_no_basis_mixes_dtypes_or_reads_on_the_host(key, rc_dtype):
    r = _lengths(7).requires_grad_()
    with DtypeAudit() as audit:
        out = _call(key, r, _rc(RCUT, rc_dtype))
        out.sum().backward()
    audit.assert_clean()


@pytest.mark.parametrize("key", list(BASES))
def test_every_basis_passes_gradcheck(key):
    r = _lengths(5).requires_grad_()
    assert torch.autograd.gradcheck(lambda q: _call(key, q), (r,))


# ------------------------------------------------------------------ errors and the names of the bases


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: radial.Chebyshev(nfunc=0, rcut=RCUT), "at least one function"),
        (lambda: radial.SineBessel(nfunc=3, rcut=0.0, p=5), "larger than 0"),
        (lambda: radial.SimplifiedBessel(nfunc=0, rcut=RCUT), "at least one function"),
        (lambda: radial.Chebyshev(nfunc=3, rcut=RCUT, kind=3), "kind must be 1 or 2"),
        (lambda: radial.Gaussian(nfunc=1, rcut=RCUT, p=5), "at least two functions"),
    ],
)
def test_option_values_tf_refuses_are_refused_at_construction(make, match):
    with pytest.raises(OptionValueError, match=match):
        make()


@pytest.mark.parametrize("shape", [(3,), (3, 2), (1, 1, 1)])
def test_lengths_of_another_shape_are_refused(shape):
    with pytest.raises(ShapeError, match=r"\[n_bonds, 1\]"):
        _call("Cheb", torch.ones(shape, dtype=F64))


@pytest.mark.parametrize(
    ("grid_shape", "scale_shape"), [((6,), ()), ((1, 5), ()), ((1, 6), (1,))]
)
def test_gaussian_weights_of_another_shape_are_refused(grid_shape, scale_shape):
    basis = radial.Gaussian(nfunc=6, rcut=RCUT, p=5)
    with pytest.raises(ShapeError, match="grid"):
        basis(
            _lengths(3),
            _rc(),
            torch.zeros(grid_shape, dtype=F64),
            torch.zeros(scale_shape, dtype=F64),
        )


def test_a_cutoff_of_a_dtype_other_than_a_parameter_dtype_is_refused():
    with pytest.raises(DtypeError, match="float32 or float64"):
        _call("Cheb", _lengths(3), torch.tensor(6, dtype=torch.int64))


def test_each_basis_type_of_tf_has_its_class_and_an_unknown_one_is_named():
    assert radial.basis_class("Cheb") is radial.Chebyshev
    assert radial.basis_class("RadSinBessel") is radial.SineBessel
    assert radial.basis_class("SBessel") is radial.SimplifiedBessel
    assert radial.basis_class("Gaussian") is radial.Gaussian
    with pytest.raises(
        OptionValueError, match=r"'Fourier'.*Cheb, RadSinBessel, SBessel, Gaussian"
    ):
        radial.basis_class("Fourier")
