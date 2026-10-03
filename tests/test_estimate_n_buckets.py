"""Characterization tests for ``estimate_n_buckets`` (``tensorpotential/data/databuilder.py``).

Logic layer: every branch (a single batch, no neighbours, the smallest bucket count that
meets the padding budget, the fall-through), the 32-bucket cap, input order and mutation.
Value layer: padded overhead worked out by hand on toy frames, and a plain-Python oracle
(sorted list, ``divmod`` bucket sizes) that does not call the unit or pandas.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from tensorpotential.data.databuilder import estimate_n_buckets
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

BUDGET = 0.3
MAX_BUCKETS = 32


def frame(neigh: list[int], n_atoms: list[int] | None = None) -> pd.DataFrame:
    n = len(neigh)
    return pd.DataFrame({
        "n_neighbours": neigh,
        "n_atoms": n_atoms if n_atoms is not None else [1] * n,
        "n_structures": [1] * n,
    })


def overhead_by_hand(neigh: list[int], n_buckets: int) -> float:
    """Padded neighbours over real neighbours minus 1, for sorted batches in n_buckets.

    ``divmod`` gives the bucket sizes (the first ``rem`` buckets get one extra batch);
    each bucket is padded to its largest member, the first one after sorting.
    """
    ordered = sorted(neigh, reverse=True)
    size, rem = divmod(len(ordered), n_buckets)
    padded, start = 0, 0
    for b in range(n_buckets):
        count = size + (1 if b < rem else 0)
        padded += ordered[start] * count
        start += count
    return padded / sum(ordered) - 1.0


def smallest_n_by_hand(neigh: list[int], budget: float) -> int:
    n_batches = len(neigh)
    if n_batches <= 1 or sum(neigh) == 0:
        return 1
    limit = min(n_batches, MAX_BUCKETS)
    for n in range(1, limit + 1):
        if overhead_by_hand(neigh, n) <= budget:
            return n
    return limit


# ----------------------------------------------------------------------------- logic


def test_empty_and_single_batch_need_one_bucket():
    assert estimate_n_buckets(frame([]), BUDGET) == 1
    assert estimate_n_buckets(frame([57]), BUDGET) == 1


def test_no_neighbours_at_all_needs_one_bucket():
    assert estimate_n_buckets(frame([0, 0, 0]), BUDGET) == 1


def test_identical_batches_need_one_bucket():
    assert estimate_n_buckets(frame([40, 40, 40, 40, 40]), 0.0) == 1


def test_default_budget_is_thirty_percent():
    # [100, 100, 10, 10]: one bucket pads to 400 of 220 real (+82%), two buckets are exact.
    df = frame([100, 100, 10, 10])
    assert estimate_n_buckets(df) == estimate_n_buckets(df, 0.3) == 2


def test_budget_below_zero_is_never_met_and_falls_through_to_the_batch_count():
    assert estimate_n_buckets(frame([9, 7, 5, 3, 1]), -0.5) == 5


def test_fall_through_is_capped_at_32_buckets():
    # 40 strictly decreasing batches: even 32 buckets leave some padding above a 0 budget.
    df = frame(list(range(400, 0, -10)))
    assert len(df) == 40
    assert estimate_n_buckets(df, 0.0) == MAX_BUCKETS


def test_cap_applies_below_the_batch_count_only():
    assert estimate_n_buckets(frame(list(range(1, 21))), -1.0) == 20


def test_input_order_does_not_matter_and_input_is_not_modified():
    neigh = [10, 100, 30, 60]
    shuffled = frame(neigh)
    before = shuffled.copy()
    assert estimate_n_buckets(shuffled, 0.25) == estimate_n_buckets(
        frame(sorted(neigh)), 0.25
    )
    pd.testing.assert_frame_equal(shuffled, before)


def test_other_columns_do_not_change_the_answer():
    neigh = [100, 60, 30, 10]
    plain = estimate_n_buckets(frame(neigh), 0.29)
    heavy = estimate_n_buckets(frame(neigh, n_atoms=[9, 1, 5, 2]), 0.29)
    assert plain == heavy


def test_non_default_index_is_accepted():
    df = frame([100, 100, 10, 10])
    df.index = [7, 3, 9, 1]
    assert estimate_n_buckets(df, BUDGET) == 2


# ----------------------------------------------------------------------------- values


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(0.31, 2), (0.25, 3), (0.15, 4)],
)
def test_hand_computed_toy_frame(budget, expected):
    """Neighbours [100, 60, 30, 10], 200 real.

    1 bucket: 4 * 100 = 400, overhead 1.0.
    2 buckets [100, 60], [30, 10]: 200 + 60 = 260, overhead 0.30 (0.31 admits it).
    3 buckets [100, 60], [30], [10]: 200 + 30 + 10 = 240, overhead 0.20.
    4 buckets: no padding.
    """
    assert estimate_n_buckets(frame([100, 60, 30, 10]), budget) == expected


def test_hand_computed_overheads_match_the_oracle():
    neigh = [100, 60, 30, 10]
    assert overhead_by_hand(neigh, 1) == pytest.approx(
        1.0, rel=ARITH.rtol, abs=ARITH.atol
    )
    assert overhead_by_hand(neigh, 2) == pytest.approx(
        0.3, rel=ARITH.rtol, abs=ARITH.atol
    )
    assert overhead_by_hand(neigh, 3) == pytest.approx(
        0.2, rel=ARITH.rtol, abs=ARITH.atol
    )
    assert overhead_by_hand(neigh, 4) == pytest.approx(0.0, abs=ARITH.atol)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("budget", [0.0, 0.05, 0.3, 1.0])
def test_agrees_with_an_independent_oracle_on_random_frames(seed, budget):
    rng = random.Random(seed)
    neigh = [rng.randint(0, 500) for _ in range(rng.randint(2, 45))]
    assert estimate_n_buckets(frame(neigh), budget) == smallest_n_by_hand(neigh, budget)


@pytest.mark.parametrize("seed", range(5))
def test_result_is_the_smallest_count_within_budget_and_monotone_in_the_budget(seed):
    rng = np.random.default_rng(seed)
    neigh = [int(v) for v in rng.integers(1, 400, size=24)]
    budgets = [1.0, 0.5, 0.3, 0.1, 0.02]
    counts = [estimate_n_buckets(frame(neigh), b) for b in budgets]
    assert counts == sorted(counts)  # a tighter budget never needs fewer buckets
    for b, n in zip(budgets, counts):
        assert overhead_by_hand(neigh, n) <= b or n == len(neigh)
        assert all(overhead_by_hand(neigh, m) > b for m in range(1, n))
