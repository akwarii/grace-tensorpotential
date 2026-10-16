"""Real spherical harmonics (``ops.spherical_harmonics``) and their plan against independent oracles and TF.

Logic: the column order, the plan's coefficients, shapes and dtypes, the hand values of the spec sheet, the gradient,
the errors. Physical values: sympy's ``Ynm`` (independent of the recursion), orthonormality by quadrature, parity, and
the outputs ``Y`` of TensorFlow's ``SphericalHarmonic`` stored in the tiny anchors.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import sympy
import torch

from tensorpotential.torch_backend.errors import DtypeError, ShapeError
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)
from tests_torch.dtype_audit import DtypeAudit
from tests_torch.golden import TINY_F64, cases, load
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261016
LMAX = 4
"""The degree of both shipped 2L models."""
SQ3, SQ5, SQ15 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(15.0)


def _directions(n: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    v = rng.normal(size=(n, 3))
    poles = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    return np.concatenate([poles, v / np.linalg.norm(v, axis=1, keepdims=True)])


def _sh(v: np.ndarray, lmax: int = LMAX) -> np.ndarray:
    return real_spherical_harmonics(
        torch.as_tensor(v, dtype=torch.float64), lmax
    ).numpy()


# ------------------------------------------------------------------ the plan


def test_columns_are_l_major_with_m_from_minus_l_to_l():
    plan = spherical_harmonics_plan(2)
    assert plan.n_columns == 9
    assert plan.l.tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert plan.m.tolist() == [0, -1, 0, 1, -2, -1, 0, 1, 2]
    assert np.array_equal(plan.l * plan.l + plan.l + plan.m, np.arange(9))


def test_the_plan_coefficients():
    plan = spherical_harmonics_plan(LMAX)
    assert plan.diagonal.tolist() == [1.0, 1.0, 3.0, 15.0, 105.0]
    assert plan.scale[:4].tolist() == pytest.approx(
        [1.0, SQ3, SQ3, SQ3], rel=1e-15, abs=0
    )
    assert plan.step_z[1, 0] == 1.0 and plan.step_z[4, 1] == 7.0 / 3.0
    assert plan.step_back[2, 0] == 0.5 and plan.step_back[3, 2] == 0.0
    assert np.all(np.triu(plan.step_z) == 0) and np.all(np.triu(plan.step_back) == 0)


@pytest.mark.parametrize("lmax", [-1, 2.0, True, "4"])
def test_an_lmax_that_is_not_a_degree_is_refused(lmax):
    with pytest.raises(ShapeError, match="lmax must be an integer of at least 0"):
        spherical_harmonics_plan(lmax)


# ------------------------------------------------------------------ the op: logic


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ((0.0, 0.0, 1.0), [1, 0, SQ3, 0, 0, 0, SQ5, 0, 0]),
        ((1.0, 0.0, 0.0), [1, 0, 0, SQ3, 0, 0, -SQ5 / 2, 0, SQ15 / 2]),
        ((0.0, 1.0, 0.0), [1, SQ3, 0, 0, 0, 0, -SQ5 / 2, 0, -SQ15 / 2]),
        ((0.0, 0.0, 0.0), [1, 0, 0, 0, 0, 0, -SQ5 / 2, 0, 0]),
    ],
    ids=["z", "x", "y", "zero_vector"],
)
def test_hand_values_of_the_spec_sheet(direction, expected):
    assert_close(
        _sh(np.array(direction), 2), np.array(expected, dtype=np.float64), FEATURE_F64
    )


def test_y00_is_one_and_l1_is_sqrt3_y_z_x():
    v = _directions(50)
    y = _sh(v)
    assert np.array_equal(y[:, 0], np.ones(len(v)))
    assert_close(y[:, 1:4], SQ3 * v[:, [1, 2, 0]], FEATURE_F64)


def test_shape_follows_the_leading_axes_and_lmax():
    v = torch.as_tensor(_directions(6), dtype=torch.float64)
    assert real_spherical_harmonics(v, LMAX).shape == (10, 25)
    assert real_spherical_harmonics(v.reshape(2, 5, 3), LMAX).shape == (2, 5, 25)
    assert real_spherical_harmonics(v, 0).tolist() == [[1.0]] * 10
    assert real_spherical_harmonics(v[0], 1).shape == (4,)


def test_no_bonds_give_an_empty_result():
    out = real_spherical_harmonics(torch.zeros((0, 3), dtype=torch.float64), LMAX)
    assert out.shape == (0, 25) and out.dtype == torch.float64


def test_float32_input_stays_float32_and_mixes_no_dtype():
    v = _directions(30)
    with DtypeAudit() as audit:
        out = real_spherical_harmonics(torch.as_tensor(v, dtype=torch.float32), LMAX)
    audit.assert_clean()
    assert out.dtype == torch.float32
    assert_close(out, _sh(v), FEATURE_F32)


def test_float64_mixes_no_dtype_and_reads_no_value_on_the_host_in_either_pass():
    v = torch.as_tensor(_directions(8), dtype=torch.float64).requires_grad_()
    with DtypeAudit() as audit:
        (grad,) = torch.autograd.grad(real_spherical_harmonics(v, LMAX).sum(), v)
    audit.assert_clean()
    assert grad.dtype == torch.float64


def test_the_gradient_passes_gradcheck():
    v = torch.as_tensor(_directions(5), dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda d: real_spherical_harmonics(d, LMAX), (v,))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex128], ids=str)
def test_a_tensor_that_is_not_floating_is_refused(dtype):
    with pytest.raises(DtypeError, match="floating"):
        real_spherical_harmonics(torch.zeros((2, 3), dtype=dtype), LMAX)


@pytest.mark.parametrize("shape", [(), (3, 2), (4,)])
def test_a_last_axis_other_than_xyz_is_refused(shape):
    with pytest.raises(ShapeError, match="last axis of size 3"):
        real_spherical_harmonics(torch.zeros(shape, dtype=torch.float64), LMAX)


# ------------------------------------------------------------------ physical values


@functools.cache
def _sympy_real_harmonics(lmax: int) -> tuple:
    """Numpy functions of (theta, phi) for every column, built from sympy's complex ``Ynm`` (Condon-Shortley phase).

    The real harmonics without that phase are ``sqrt(2) (-1)**m Re Y[l, m]`` for ``m > 0`` and ``sqrt(2) (-1)**m Im
    Y[l, |m|]`` for ``m < 0``; times ``sqrt(4 pi)`` for the scale of TF's default.
    """
    theta, phi = sympy.symbols("theta phi", real=True)
    functions = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sympy.Ynm(l, abs(m), theta, phi).expand(func=True)
            part = sympy.re(ylm) if m >= 0 else sympy.im(ylm)
            factor = sympy.sqrt(4 * sympy.pi) * (sympy.sqrt(2) * (-1) ** m if m else 1)
            functions.append(
                sympy.lambdify((theta, phi), sympy.simplify(factor * part), "numpy")
            )
    return tuple(functions)


@pytest.mark.parametrize("lmax", [LMAX, 6])
def test_matches_sympy(lmax):
    v = _directions(200)
    theta, phi = np.arccos(np.clip(v[:, 2], -1.0, 1.0)), np.arctan2(v[:, 1], v[:, 0])
    expected = np.stack(
        [
            np.broadcast_to(f(theta, phi), theta.shape)
            for f in _sympy_real_harmonics(lmax)
        ],
        axis=-1,
    )
    assert_close(_sh(v, lmax), expected, FEATURE_F64, f"lmax {lmax} against sympy")


def test_orthonormal_over_the_sphere():
    # Gauss-Legendre in z (exact to degree 19) times an even grid in phi (exact to frequency 19): exact for products of
    # two harmonics of degree 4 or less, which are polynomials of degree 8
    z, wz = np.polynomial.legendre.leggauss(10)
    phi = np.arange(20) * (2 * np.pi / 20)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    s = np.sqrt(1.0 - zz**2)
    v = np.stack([s * np.cos(pp), s * np.sin(pp), zz], axis=-1).reshape(-1, 3)
    weights = np.repeat(wz, len(phi)) * (2 * np.pi / len(phi)) / (4 * np.pi)
    y = _sh(v)
    gram = (y * weights[:, None]).T @ y
    assert_close(gram, np.eye(25), FEATURE_F64, "mean of Y Y over the sphere")


def test_parity_is_minus_one_to_the_l():
    v = _directions(50)
    plan = spherical_harmonics_plan(LMAX)
    assert_close(_sh(-v), _sh(v) * (-1.0) ** plan.l, FEATURE_F64)


@pytest.mark.golden
@pytest.mark.parametrize("name", TINY_F64)
def test_matches_the_tf_harmonics_of_the_anchors(name):
    arrays = load(name)
    for case in cases(arrays):
        rhat = arrays[f"{case}/ins/ScaledBondVector"]
        assert_close(_sh(rhat), arrays[f"{case}/ins/Y"], FEATURE_F64, f"{name} {case}")
