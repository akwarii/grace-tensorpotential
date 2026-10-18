"""The ``ProductFunction`` plan (``plans.product_function``): logic on hand-made frames, physics on a real coupling.

Logic: the flattening of the per-row lists, the dtypes, ``nfunc`` as TensorFlow computes it, the errors; the options
read from a yaml and passed to the coupling generator as the TF constructor passes them. Physical
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

from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.product_function import (
    ProductFunctionOptions,
    ProductFunctionPlan,
    product_coupling_meta_data,
    product_function_plan,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)
from tensorpotential.torch_backend.spec.model import InstructionRef
from tests_torch.coupling_frames import coupled, uncoupled

RNG_SEED = 20261016
LMAX = 2


def _frame(left, right, cg) -> pd.DataFrame:
    return pd.DataFrame({"left_inds": left, "right_inds": right, "cg_list": cg})


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


def _options(**given: object) -> ProductFunctionOptions:
    return ProductFunctionOptions.from_options({
        "left": InstructionRef("A"),
        "right": InstructionRef("B"),
        "lmax": 2,
        "Lmax": 1,
        **given,
    })


def test_missing_options_take_the_pinned_defaults():
    options = _options()
    assert options == ProductFunctionOptions(lmax=2, Lmax=1, is_left_right_equal=False)


@pytest.mark.parametrize(
    ("right", "given", "equal"),
    [("A", None, True), ("B", None, False), ("B", True, True), ("A", False, False)],
)
def test_a_null_is_left_right_equal_compares_the_operands(right, given, equal):
    options = _options(right=InstructionRef(right), is_left_right_equal=given)
    assert options.is_left_right_equal is equal


def test_lists_of_pairs_become_tuples_and_limits_pass_through():
    options = _options(
        keep_parity=[[0, 1], [1, -1]],
        history_drop_list=[["(1,1)", 0]],
        lmax_left=1,
        lmax_right=2,
        lmax_hist=3,
        lmax_hist_left=4,
        lmax_hist_right=5,
        max_sum_l=6,
        normalize=True,
    )
    assert options.keep_parity == ((0, 1), (1, -1))
    assert options.history_drop_list == (("(1,1)", 0),)
    limits = (
        options.lmax_left,
        options.lmax_right,
        options.lmax_hist,
        options.lmax_hist_left,
        options.lmax_hist_right,
        options.max_sum_l,
    )
    assert limits == (1, 2, 3, 4, 5, 6)
    assert options.normalize is True


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    columns = ["l", "m", "hist", "parity", "left_inds", "right_inds", "cg_list"]
    return a[columns].astype(str).equals(b[columns].astype(str))


def test_natural_parity_and_the_legacy_history_are_the_defaults():
    harmonics = uncoupled(LMAX)
    meta = product_coupling_meta_data(
        harmonics, harmonics, _options(lmax=LMAX, Lmax=LMAX, normalize=True)
    )
    assert _same(meta, coupled(LMAX, LMAX))
    assert "(1,1)" in set(meta["hist"])  # no parity signs


@pytest.mark.parametrize(
    ("given", "check"),
    [
        (
            {"keep_parity": [[1, 1]]},
            lambda m: set(zip(m["l"], m["parity"], strict=True)) == {(1, 1)},
        ),
        ({"is_left_right_equal": True}, lambda m: (m["l1"] >= m["l2"]).all()),
        ({"lmax_left": 1}, lambda m: m["l1"].max() == 1),
        ({"lmax_right": 0}, lambda m: m["l2"].max() == 0),
        ({"max_sum_l": 2}, lambda m: m["sum_of_ls"].max() <= 2),
        (
            {"history_drop_list": [["(1,1)", 0]]},
            lambda m: "(1,1)" not in set(m["hist"][m["l"] == 0]),
        ),
    ],
)
def test_each_option_reaches_the_generator(given, check):
    harmonics = uncoupled(LMAX)
    meta = product_coupling_meta_data(
        harmonics, harmonics, _options(lmax=LMAX, Lmax=LMAX, **given)
    )
    default = product_coupling_meta_data(
        harmonics, harmonics, _options(lmax=LMAX, Lmax=LMAX)
    )
    assert check(meta)
    assert len(meta) < len(default)


@pytest.mark.parametrize("key", ["lmax_hist", "lmax_hist_left", "lmax_hist_right"])
def test_the_history_limits_reach_the_generator(key):
    pairs = coupled(LMAX, LMAX)  # histories up to l = 2 on both sides
    full = product_coupling_meta_data(pairs, pairs, _options(lmax=LMAX, Lmax=1))
    capped = product_coupling_meta_data(
        pairs, pairs, _options(lmax=LMAX, Lmax=1, **{key: 1})
    )
    assert 0 < len(capped) < len(full)


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
