"""The ``slow`` tier: tests that take 30 s or more, listed in ``slow_tests.txt``.

The list lives in a file, not in decorators, so that the tests it names can be
edited or folded without touching the list's mechanics; a test that disappears
fails ``test_every_listed_slow_test_exists`` and is removed from the list in the
same change. Nothing is skipped by default: ``-m "not slow"`` is the fast
development loop, CI and the gate run everything.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

import pytest


class _HasNodeId(Protocol):
    nodeid: str


T = TypeVar("T", bound=_HasNodeId)


def _base_id(item: _HasNodeId) -> str:
    """The node id of ``item`` without a ``[...]`` parametrization suffix."""
    return item.nodeid.split("[", 1)[0]


SLOW_LIST = Path(__file__).with_name("slow_tests.txt")
"""One ``tests/<file>.py::<test>`` per line, longest first; ``#`` starts a comment."""


def load_slow_ids(path: Path = SLOW_LIST) -> tuple[str, ...]:
    """Read the node ids of the slow tests from ``path``.

    Parameters
    ----------
    path : Path
        The list file; blank lines and ``#`` comments are ignored.

    Returns
    -------
    tuple of str
        The node ids in file order.

    Raises
    ------
    ValueError
        If an id is listed twice.
    """
    ids: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"slow test listed twice in {path.name}: {duplicates}")
    return tuple(ids)


def apply_slow_marks(items: Iterable[Any], slow_ids: Iterable[str]) -> int:
    """Add ``pytest.mark.slow`` to each collected item whose id is listed.

    A parametrized item matches through its id without the ``[...]`` suffix.

    Parameters
    ----------
    items : iterable of pytest.Item
        The collected tests.
    slow_ids : iterable of str
        Node ids from :func:`load_slow_ids`.

    Returns
    -------
    int
        The number of items that were marked.
    """
    listed = set(slow_ids)
    marked = 0
    for item in items:
        if _base_id(item) in listed:
            item.add_marker(pytest.mark.slow)
            marked += 1
    return marked


def _defines(body: Sequence[ast.stmt], names: Sequence[str]) -> bool:
    """Whether ``names`` (``[class, method]`` or ``[function]``) is defined in ``body``."""
    head, rest = names[0], names[1:]
    for node in body:
        if isinstance(node, ast.ClassDef) and rest and node.name == head:
            return _defines(node.body, rest)
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and not rest
            and node.name == head
        ):
            return True
    return False


def missing_ids(slow_ids: Iterable[str], root: Path) -> list[str]:
    """The listed ids that do not name a test function below ``root``.

    Parameters
    ----------
    slow_ids : iterable of str
        Node ids as in the list file.
    root : Path
        The repository root that the file part of an id is relative to.

    Returns
    -------
    list of str
        The ids whose file or function is not found, in input order.
    """
    missing = []
    for node_id in slow_ids:
        file_part, _, name_part = node_id.partition("::")
        path = root / file_part
        names = name_part.split("::") if name_part else []
        if not names or not path.is_file():
            missing.append(node_id)
            continue
        tree = ast.parse(path.read_text())
        if not _defines(tree.body, names):
            missing.append(node_id)
    return missing


def _first_chunk_size(n_items: int, workers: int) -> int:
    """Size of the first block of tests that ``pytest-xdist --dist load`` hands to each worker."""
    return max(n_items // 4 // workers, 2)


def deal_slow_first(
    items: Sequence[T], slow_ids: Sequence[str], workers: int
) -> list[T]:
    """Reorder ``items`` so that each worker's first block starts with its share of slow tests.

    ``--dist load`` sends the collected list to the workers in blocks, the first
    block to the first worker and so on, and a worker runs its block in order. The
    slow tests are therefore dealt round-robin (longest first, as listed) over the
    first ``workers`` blocks, each followed by ordinary tests up to the block size;
    the rest keeps its order. Without this the longest test may only start when
    every other worker is nearly done. The block size is the formula of
    pytest-xdist; if it changes, the order is merely less effective.

    Parameters
    ----------
    items : sequence
        The collected tests (anything with a ``nodeid``).
    slow_ids : sequence of str
        Node ids from :func:`load_slow_ids`, longest first.
    workers : int
        Number of xdist workers; at most 1 leaves the order unchanged.

    Returns
    -------
    list
        The same items, reordered.
    """
    if workers <= 1:
        return list(items)
    rank = {node_id: i for i, node_id in enumerate(slow_ids)}
    slow = sorted(
        (it for it in items if _base_id(it) in rank), key=lambda it: rank[_base_id(it)]
    )
    rest = iter([it for it in items if _base_id(it) not in rank])
    chunk = _first_chunk_size(len(items), workers)
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
