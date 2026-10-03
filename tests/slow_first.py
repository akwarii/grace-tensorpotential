"""Start the slow tests first under ``pytest-xdist --dist load``.

A test is slow when it takes 30 s or more and carries ``@pytest.mark.slow``.
``--dist load`` hands the collected list to the workers in blocks, the first block
to the first worker and so on, and a worker runs its block in order. A slow test
late in the list may therefore start when the other workers are nearly done, and
slow tests all at the front would all go to the first worker. :func:`deal_slow_first`
deals them over the first block of each worker instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def first_block_size(n_items: int, workers: int) -> int:
    """Size of the first block that ``pytest-xdist --dist load`` hands to each worker."""
    return max(n_items // 4 // workers, 2)


def deal_slow_first(
    items: Sequence[T], is_slow: Callable[[T], bool], workers: int
) -> list[T]:
    """Reorder ``items`` so that each worker's first block starts with its share of slow tests.

    The slow tests, in collection order, are dealt round-robin over the first
    ``workers`` blocks, each followed by ordinary tests up to the block size; the
    rest keeps its order. The block size is the formula of pytest-xdist; if it
    changes, the order is merely less effective.

    Parameters
    ----------
    items : sequence
        The collected tests.
    is_slow : callable
        True for a slow test.
    workers : int
        Number of xdist workers; at most 1 leaves the order unchanged.

    Returns
    -------
    list
        The same items, reordered.
    """
    if workers <= 1:
        return list(items)
    slow = [it for it in items if is_slow(it)]
    rest = iter([it for it in items if not is_slow(it)])
    chunk = first_block_size(len(items), workers)
    ordered: list[T] = []
    for worker in range(workers):
        block = slow[worker::workers]
        ordered.extend(block)
        for _ in range(max(chunk - len(block), 0)):
            nxt = next(rest, None)
            if nxt is None:
                break
            ordered.append(nxt)
    ordered.extend(rest)
    return ordered
