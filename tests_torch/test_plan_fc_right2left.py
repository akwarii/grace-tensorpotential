"""The ``FCRight2Left`` plan (``plans.fc_right2left``): logic on hand-made frames, physics on a real coupling.

Logic: the group order (pandas ``groupby``: sorted by ``(l, parity, hist)``, rows in order), the tiles, the collect
tables, ``norm_map``, the weight shapes and norms of every option, the errors. Physical values: on the metadata of a
real coupling every collected pair joins rows of the same ``(l, m, parity)``, which is what keeps the mixing
rotation equivariant. The equality with TensorFlow's tables is in ``test_plans_against_tf.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.plans.fc_right2left import (
    FCRight2LeftWidths,
    fc_right2left_plan,
)
from tests_torch.coupling_frames import coupled, uncoupled

WIDTHS = FCRight2LeftWidths(n_out=3, n_left=4, n_right=9)


def _frame(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    l, parity, hist = zip(*rows, strict=True)
    return pd.DataFrame({"l": l, "parity": parity, "hist": hist})


# rows 0-2: (1, -1, "b"), rows 3: (0, 1, "a"), rows 4-6: (1, -1, "a"); sorted groups: (0,1,a) (1,-1,a) (1,-1,b)
LEFT = _frame([(1, -1, "b")] * 3 + [(0, 1, "a")] + [(1, -1, "a")] * 3)
# groups: (0,1,"") rows 0; (1,-1,"") rows 1-3; (2,1,"") rows 4-8 (no left partner)
RIGHT = _frame([(0, 1, "")] + [(1, -1, "")] * 3 + [(2, 1, "")] * 5)


# ------------------------------------------------------------------ logic


def test_groups_are_sorted_by_l_parity_hist_and_tile_the_left_weight():
    plan = fc_right2left_plan(LEFT, RIGHT, WIDTHS)
    assert plan.w_tile_left.tolist() == [2, 2, 2, 0, 1, 1, 1]
    assert plan.w_shape_left == 3
    assert plan.w_tile_left.dtype == np.int32


def test_every_right_group_of_the_same_l_and_parity_is_collected_into_each_left_group():
    plan = fc_right2left_plan(LEFT, RIGHT, WIDTHS)
    assert plan.collect_to.tolist() == [3, 4, 5, 6, 0, 1, 2]
    assert plan.collect_from.tolist() == [0, 1, 2, 3, 1, 2, 3]
    assert plan.w_tile_right.tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert plan.w_shape_right == 3
    for table in (plan.collect_to, plan.collect_from, plan.w_tile_right):
        assert table.dtype == np.int32


def test_norm_map_counts_the_right_groups_mixed_into_each_left_group():
    left = _frame([(0, 1, "a"), (0, 1, "b"), (2, 1, "a")])
    right = _frame([(0, 1, "x"), (0, 1, "y"), (1, -1, "z"), (1, -1, "z"), (1, -1, "z")])
    widths = FCRight2LeftWidths(n_out=2, n_left=2, n_right=2)
    plan = fc_right2left_plan(left, right, widths)
    # each l = 0 group takes both right l = 0 groups, the l = 2 group none
    assert plan.norm_map.tolist() == pytest.approx(
        [3**-0.5, 3**-0.5, 1.0], rel=1e-15, abs=0
    )
    assert plan.norm_map.dtype == np.float64
    assert plan.collect_to.tolist() == [0, 0, 1, 1]
    assert plan.collect_from.tolist() == [0, 1, 0, 1]
    assert plan.w_tile_right.tolist() == [0, 1, 2, 3]


def test_default_weight_shapes_and_norms():
    plan = fc_right2left_plan(LEFT, RIGHT, WIDTHS)
    assert plan.w_left_shape == (3, 4, 3)
    assert plan.w_right_shape == (3, 9, 3)
    assert plan.norm_left == 0.5
    assert plan.norm_right == pytest.approx(1 / 3, rel=1e-15, abs=0)


@pytest.mark.parametrize(
    ("per_element", "left", "right"),
    [
        ((True, False), (5, 3, 4, 3), (3, 9, 3)),
        ((False, True), (3, 4, 3), (5, 3, 9, 3)),
        ((True, True), (5, 3, 4, 3), (5, 3, 9, 3)),
    ],
)
def test_a_per_element_side_gets_a_leading_element_axis(per_element, left, right):
    widths = dataclasses.replace(
        WIDTHS, per_element=per_element, number_of_atom_types=5
    )
    plan = fc_right2left_plan(LEFT, RIGHT, widths)
    assert (plan.w_left_shape, plan.w_right_shape) == (left, right)


def test_without_normalize_the_norms_are_one():
    plan = fc_right2left_plan(LEFT, RIGHT, dataclasses.replace(WIDTHS, normalize=False))
    assert (plan.norm_left, plan.norm_right) == (1.0, 1.0)


def test_without_left_coefs_there_is_no_left_weight():
    widths = FCRight2LeftWidths(n_out=4, n_left=4, n_right=9, left_coefs=False)
    plan = fc_right2left_plan(LEFT, RIGHT, widths)
    assert plan.w_left_shape is None and plan.norm_left is None
    assert plan.w_right_shape == (4, 9, 3)
    assert plan.w_tile_left.tolist() == [2, 2, 2, 0, 1, 1, 1]


def test_without_left_coefs_n_out_must_be_the_left_width():
    widths = FCRight2LeftWidths(n_out=3, n_left=4, n_right=9, left_coefs=False)
    with pytest.raises(OptionValueError, match="left_coefs false needs n_out"):
        fc_right2left_plan(LEFT, RIGHT, widths)


def test_a_per_element_side_needs_the_number_of_atom_types():
    widths = dataclasses.replace(WIDTHS, per_element=(False, True))
    with pytest.raises(OptionValueError, match="number_of_atom_types is required"):
        fc_right2left_plan(LEFT, RIGHT, widths)


def test_inputs_without_a_common_l_and_parity_are_refused():
    with pytest.raises(ShapeError, match="nothing to collect"):
        fc_right2left_plan(_frame([(3, -1, "")]), RIGHT, WIDTHS)


def test_matched_groups_of_different_size_are_refused():
    with pytest.raises(ShapeError, match="has 2 rows, the matching right group 3"):
        fc_right2left_plan(_frame([(1, -1, "a")] * 2), RIGHT, WIDTHS)


def test_the_plan_is_frozen():
    plan = fc_right2left_plan(LEFT, RIGHT, WIDTHS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.w_shape_left = 1  # ty: ignore[invalid-assignment]


# ------------------------------------------------------------------ physical values


def test_on_a_real_coupling_every_pair_joins_rows_of_the_same_l_m_parity():
    left = coupled(2, 2)
    right = uncoupled(2)
    plan = fc_right2left_plan(left, right, WIDTHS)
    columns = ["l", "m", "parity"]
    to = left[columns].to_numpy()[plan.collect_to]
    source = right[columns].to_numpy()[plan.collect_from]
    assert np.array_equal(to, source)
    # every left row with an (l, parity) of the harmonics receives exactly one right group (one hist there)
    assert plan.w_shape_right == plan.w_shape_left
    assert np.all(plan.norm_map == 1 / 2**0.5)
