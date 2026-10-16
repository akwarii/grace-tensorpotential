"""The ``ProductFunction`` plan (``plans.product_function``): logic on hand-made frames, physics on a real coupling.

Logic: the flattening of the per-row lists, the dtypes, ``nfunc`` as TensorFlow computes it, the errors. Physical
values: the tables built from ``real_coupling_metainformation`` of two sets of real spherical harmonics give a
product whose ``L = 0`` functions are rotation invariant and whose ``(l1, l2) = (1, 1)`` scalar is proportional to
the dot product of the two directions (an oracle that does not use the plan). The equality with TensorFlow's tables
is in ``test_plans_against_tf.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.functions.couplings import real_coupling_metainformation
from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.product_function import (
    ProductFunctionPlan,
    product_function_plan,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)

RNG_SEED = 20261016
LMAX = 2


def _frame(left, right, cg) -> pd.DataFrame:
    return pd.DataFrame({"left_inds": left, "right_inds": right, "cg_list": cg})


def uncoupled(lmax: int) -> pd.DataFrame:
    """The metadata of the real harmonics up to ``lmax``, as ``SphericalHarmonic`` builds it (``l*l + l + m`` order)."""
    l = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
    m = np.concatenate([np.arange(-d, d + 1) for d in range(lmax + 1)])
    columns = {
        "l": l,
        "m": m,
        "hist": [""] * len(l),
        "parity": (-1) ** l,
        "sum_of_ls": l,
    }
    return pd.DataFrame(columns)


def coupled(lmax: int, Lmax: int) -> pd.DataFrame:
    """The metadata of the product of two sets of harmonics, as ``ProductFunction`` builds it."""
    harmonics = uncoupled(lmax)
    keep = [[l, 1 if l % 2 == 0 else -1] for l in range(Lmax + 1)]
    return real_coupling_metainformation(
        A=harmonics,
        B=harmonics,
        lmax=lmax,
        Lmax=Lmax,
        is_A_B_equal=False,
        keep_parity=keep,
        normalize=True,
        legacy_format=True,
    )


# ------------------------------------------------------------------ logic


def test_the_lists_are_flattened_in_row_order():
    plan = product_function_plan(
        _frame([[0], [1, 2], [3]], [[0], [2, 1], [3]], [[1.0], [0.5, -0.5], [2.0]])
    )
    assert plan.left_ind.tolist() == [0, 1, 2, 3]
    assert plan.right_ind.tolist() == [0, 2, 1, 3]
    assert plan.m_sum_ind.tolist() == [0, 1, 1, 2]
    assert plan.cg.tolist() == [1.0, 0.5, -0.5, 2.0]
    assert plan.nfunc == 3
    assert (
        plan.left_ind.dtype == plan.right_ind.dtype == plan.m_sum_ind.dtype == np.int32
    )
    assert plan.cg.dtype == np.float64
    assert isinstance(plan.nfunc, int)


def test_numpy_rows_and_float32_coefficients_are_accepted():
    frame = _frame(
        [np.array([4, 5])], [np.array([6, 7])], [np.array([0.25, 0.75], np.float32)]
    )
    plan = product_function_plan(frame)
    assert plan.left_ind.tolist() == [4, 5] and plan.right_ind.tolist() == [6, 7]
    assert plan.cg.tolist() == [0.25, 0.75] and plan.cg.dtype == np.float64


def test_an_empty_row_keeps_its_number_and_nfunc_is_the_largest_used_plus_one():
    # TF: m_sum_ind comes from enumerate over the rows, nfunc = max(m_sum_ind) + 1
    plan = product_function_plan(
        _frame([[0], [], [1], []], [[0], [], [1], []], [[1.0], [], [1.0], []])
    )
    assert plan.m_sum_ind.tolist() == [0, 2]
    assert plan.nfunc == 3


def test_the_plan_is_frozen():
    plan = product_function_plan(_frame([[0]], [[0]], [[1.0]]))
    assert isinstance(plan, ProductFunctionPlan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.nfunc = 2  # ty: ignore[invalid-assignment]


def test_a_frame_without_rows_is_refused():
    with pytest.raises(ShapeError, match="no row"):
        product_function_plan(_frame([], [], []))


def test_a_frame_without_pairs_is_refused():
    with pytest.raises(ShapeError, match="no Clebsch-Gordan pair"):
        product_function_plan(_frame([[]], [[]], [[]]))


@pytest.mark.parametrize(
    ("left", "right", "cg"),
    [
        ([[0, 1]], [[0]], [[1.0, 1.0]]),
        ([[0]], [[0, 1]], [[1.0]]),
        ([[0]], [[0]], [[1.0, 2.0]]),
    ],
)
def test_lists_of_unequal_length_in_a_row_are_refused(left, right, cg):
    with pytest.raises(ShapeError, match="row 0 of the coupling metadata"):
        product_function_plan(_frame(left, right, cg))


# ------------------------------------------------------------------ physical values


def _directions(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _product(plan: ProductFunctionPlan, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``out[f] = sum cg Y(u)[left] Y(v)[right]`` over the pairs of ``f``, without torch's scatter."""
    sh = spherical_harmonics_plan(LMAX)
    yu = real_spherical_harmonics(torch.as_tensor(u), sh).numpy()
    yv = real_spherical_harmonics(torch.as_tensor(v), sh).numpy()
    terms = plan.cg * yu[:, plan.left_ind] * yv[:, plan.right_ind]
    return terms @ np.eye(plan.nfunc)[plan.m_sum_ind]


def test_the_scalar_functions_of_a_real_coupling_are_rotation_invariant():
    meta = coupled(LMAX, LMAX)
    plan = product_function_plan(meta)
    rng = np.random.default_rng(RNG_SEED)
    u, v = _directions(32, rng), _directions(32, rng)
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    before = _product(plan, u, v)
    after = _product(plan, u @ rotation.T, v @ rotation.T)
    scalars = np.flatnonzero(meta["l"].to_numpy() == 0)
    assert len(scalars) == LMAX + 1  # one per (l1, l1) pair
    assert np.allclose(after[:, scalars], before[:, scalars], rtol=0, atol=1e-13)
    vectors = np.flatnonzero(meta["l"].to_numpy() == 1)
    assert not np.allclose(after[:, vectors], before[:, vectors], rtol=0, atol=1e-6)


def test_the_l1_l1_scalar_is_proportional_to_the_dot_product():
    meta = coupled(LMAX, 0)
    plan = product_function_plan(meta)
    (row,) = np.flatnonzero(meta["hist"].to_numpy() == "(1,1)")
    (pairs,) = np.nonzero(plan.m_sum_ind == row)
    assert len(pairs) == 3  # m = -1, 0, 1
    rng = np.random.default_rng(RNG_SEED + 1)
    u, v = _directions(64, rng), _directions(64, rng)
    ratio = _product(plan, u, v)[:, row] / np.einsum("ij,ij->i", u, v)
    assert np.allclose(ratio, ratio[0], rtol=1e-12, atol=0)
    assert abs(ratio[0]) > 0.1
