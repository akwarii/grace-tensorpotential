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
from typing import Any

import pytest

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
        if item.nodeid.split("[", 1)[0] in listed:
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
