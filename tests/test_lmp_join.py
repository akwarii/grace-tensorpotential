"""Tests of the (l, m, parity) join used by the reducing instructions.

``_lmp_lookup`` and ``_lmp_matches`` replace a nested ``iterrows`` loop over the collected
functions and the coupling metadata. The reference below is that loop, written out; the join
must return the same labels in the same order.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pytest

from tensorpotential.instructions import (
    FunctionReduce,
    ProductFunction,
    SphericalHarmonic,
)
from tensorpotential.instructions.compute import _lmp_lookup, _lmp_matches


def reference_join(collect_df, meta_df):
    """The nested loop that the dictionary join replaces."""
    out = []
    for _, row in collect_df.iterrows():
        for idx, rw in meta_df.iterrows():
            if (
                (row["l"] == rw["l"])
                & (row["m"] == rw["m"])
                & (row["parity"] == rw["parity"])
            ):
                out.append(idx)
    return out


def frame(rows, index=None):
    return pd.DataFrame(rows, columns=["l", "m", "parity"], index=index)


def test_lookup_groups_labels_in_row_order():
    meta = frame(
        [[0, 0, 1], [1, -1, -1], [0, 0, 1], [1, -1, 1]], index=[10, 11, 12, 13]
    )
    assert _lmp_lookup(meta) == {
        (0, 0, 1): [10, 12],
        (1, -1, -1): [11],
        (1, -1, 1): [13],
    }


def test_matches_hand_computed():
    meta = frame([[0, 0, 1], [1, 0, -1], [1, 1, -1], [0, 0, 1]])
    collect = frame([[1, 1, -1], [0, 0, 1], [2, 0, 1]])
    # row 0 -> label 2; row 1 -> labels 0 and 3; row 2 has no match
    assert _lmp_matches(collect, _lmp_lookup(meta)) == [2, 0, 3]


def test_parity_distinguishes_otherwise_equal_rows():
    meta = frame([[1, 0, 1], [1, 0, -1]])
    assert _lmp_matches(frame([[1, 0, -1]]), _lmp_lookup(meta)) == [1]


def test_empty_inputs():
    meta = frame([[0, 0, 1]])
    assert _lmp_matches(frame([]), _lmp_lookup(meta)) == []
    assert _lmp_matches(frame([[0, 0, 1]]), _lmp_lookup(frame([]))) == []


def test_labels_are_index_labels_not_positions():
    meta = frame([[0, 0, 1], [1, 0, 1]], index=[7, 3])
    assert _lmp_matches(frame([[1, 0, 1]]), _lmp_lookup(meta)) == [3]


@pytest.mark.parametrize("seed", range(5))
def test_equals_nested_loop_on_random_tables(seed):
    rng = np.random.default_rng(seed)

    def table(n, index=None):
        return frame(
            np.column_stack([
                rng.integers(0, 4, n),
                rng.integers(-3, 4, n),
                rng.choice([-1, 1], n),
            ]),
            index=index,
        )

    meta = table(60, index=rng.permutation(60) + 5)  # duplicated keys, shuffled labels
    collect = table(40)
    assert _lmp_matches(collect, _lmp_lookup(meta)) == reference_join(collect, meta)


def test_equals_nested_loop_on_a_real_instruction():
    """The join inside ``FunctionReduce`` gives the labels of the nested loop."""
    ylm = SphericalHarmonic(vhat="v", lmax=2, name="Y")
    ylm.n_out = 3
    prod = ProductFunction(left=ylm, right=ylm, name="YY", lmax=2, Lmax=2)
    prod.n_out = 5
    red = FunctionReduce(
        instructions=[ylm, prod],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=[[0, 1], [1, -1], [2, 1]],
    )
    assert set(red.collector) == {"Y", "YY"}
    for collection in red.collector.values():
        expected = reference_join(collection["collect_meta_df"], red.coupling_meta_data)
        assert expected
        assert collection["total_sum_ind"].numpy().ravel().tolist() == expected
