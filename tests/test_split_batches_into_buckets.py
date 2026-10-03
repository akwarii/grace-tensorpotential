"""Row splitting of batch tables into buckets (`split_batches_into_buckets`, `estimate_n_buckets`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tensorpotential.data.databuilder import split_batches_into_buckets


def _table(n: int) -> pd.DataFrame:
    # non-default index so that label preservation is observable
    return pd.DataFrame({"n_neighbours": np.arange(n)[::-1] * 10}, index=np.arange(n) + 100)


@pytest.mark.parametrize(("n_rows", "n_buckets"), [(10, 3), (7, 7), (8, 1), (5, 4), (12, 5)])
def test_sizes_follow_array_split_rule(n_rows: int, n_buckets: int) -> None:
    # hand rule: the first n_rows % n_buckets chunks hold one extra row
    base, extra = divmod(n_rows, n_buckets)
    expected = [base + 1] * extra + [base] * (n_buckets - extra)
    buckets = split_batches_into_buckets(_table(n_rows), n_buckets)
    assert [len(b) for b in buckets] == expected


def test_chunks_are_contiguous_and_cover_every_row_once() -> None:
    df = _table(11)
    buckets = split_batches_into_buckets(df, 4)
    assert all(isinstance(b, pd.DataFrame) for b in buckets)
    pd.testing.assert_frame_equal(pd.concat(buckets), df)


def test_more_buckets_than_rows_gives_empty_tail() -> None:
    buckets = split_batches_into_buckets(_table(2), 4)
    assert [len(b) for b in buckets] == [1, 1, 0, 0]
