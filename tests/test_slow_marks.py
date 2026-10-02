from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .slow_marks import (
    SLOW_LIST,
    deal_slow_first,
    load_slow_ids,
    missing_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("# header\n\ntests/a.py::t1  # 40 s\n  tests/b.py::C::t2\n")
    assert load_slow_ids(path) == ("tests/a.py::t1", "tests/b.py::C::t2")


def test_load_rejects_a_test_listed_twice(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("tests/a.py::t1\ntests/b.py::t2\ntests/a.py::t1\n")
    with pytest.raises(ValueError, match=r"listed twice.*tests/a.py::t1"):
        load_slow_ids(path)


def _collected(project: Path, *args: str) -> list[str]:
    """Node ids that pytest collects in ``project`` with ``args`` (a real session)."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *args,
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if "::" in line]


def test_hook_marks_listed_tests_and_parametrized_ids_in_a_real_session(
    tmp_path: Path,
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = slow: slow tests\n")
    (tmp_path / "slow_list.txt").write_text(
        "test_x.py::test_p\ntest_x.py::TestC::test_m\n"
    )
    (tmp_path / "conftest.py").write_text(
        "from pathlib import Path\n"
        "from tests.slow_marks import apply_slow_marks, load_slow_ids\n"
        "def pytest_collection_modifyitems(items):\n"
        "    apply_slow_marks(items, load_slow_ids(Path(__file__).with_name('slow_list.txt')))\n"
    )
    (tmp_path / "test_x.py").write_text(
        "import pytest\n"
        "def test_fast(): ...\n"
        "@pytest.mark.parametrize('n', [1, 2])\n"
        "def test_p(n): ...\n"
        "class TestC:\n"
        "    def test_m(self): ...\n"
        "    def test_other(self): ...\n"
    )
    assert _collected(tmp_path, "-m", "slow") == [
        "test_x.py::test_p[1]",
        "test_x.py::test_p[2]",
        "test_x.py::TestC::test_m",
    ]
    assert _collected(tmp_path, "-m", "not slow") == [
        "test_x.py::test_fast",
        "test_x.py::TestC::test_other",
    ]


def test_missing_ids_finds_files_functions_methods_and_classes(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_a(): ...\n\nclass TestC:\n    def test_m(self): ...\n"
    )
    listed = [
        "tests/test_x.py::test_a",
        "tests/test_x.py::TestC::test_m",
        "tests/test_x.py::test_gone",
        "tests/test_x.py::TestC::test_gone",
        "tests/test_x.py::TestOther::test_m",
        "tests/test_x.py::test_a::nested",
        "tests/test_x.py::TestC",
        "tests/test_x.py",
        "tests/test_nofile.py::test_a",
    ]
    assert missing_ids(listed, tmp_path) == listed[2:]


def test_every_listed_slow_test_exists() -> None:
    assert missing_ids(load_slow_ids(SLOW_LIST), REPO_ROOT) == []


class _Item:
    """The one attribute of a collected test that the ordering reads."""

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid

    def __repr__(self) -> str:
        return self.nodeid


def _items(n_fast: int, slow: list[str]) -> list[_Item]:
    fast = [_Item(f"tests/f.py::fast{i}") for i in range(n_fast)]
    # slow tests sit at the end of the collection, as they do alphabetically in places
    return [*fast, *(_Item(s) for s in slow)]


SLOW = [
    "tests/s.py::a",
    "tests/s.py::b",
    "tests/s.py::c",
    "tests/s.py::d",
    "tests/s.py::e",
]


def test_dealing_puts_the_longest_tests_first_one_per_worker() -> None:
    items = _items(40, SLOW[::-1] + ["tests/s.py::p[1]", "tests/s.py::p[2]"])
    ordered = deal_slow_first(items, [*SLOW, "tests/s.py::p"], workers=3)
    chunk = max(len(items) // 4 // 3, 2)  # the xdist first-block size: 5
    heads = [ordered[k * chunk] for k in range(3)]
    assert [h.nodeid for h in heads] == [
        "tests/s.py::a",
        "tests/s.py::b",
        "tests/s.py::c",
    ]
    # worker 0 also gets the 4th slow test (d), worker 1 the 5th (e), right after its head
    assert ordered[1].nodeid == "tests/s.py::d"
    assert ordered[chunk + 1].nodeid == "tests/s.py::e"


def test_dealing_keeps_every_test_once_and_the_order_of_the_others() -> None:
    items = _items(40, SLOW)
    ordered = deal_slow_first(items, SLOW, workers=4)
    assert sorted(i.nodeid for i in ordered) == sorted(i.nodeid for i in items)
    others = [i.nodeid for i in ordered if "tests/s.py" not in i.nodeid]
    assert others == [i.nodeid for i in items if "tests/s.py" not in i.nodeid]


def test_dealing_is_the_identity_without_workers_or_without_slow_tests() -> None:
    items = _items(10, SLOW)
    assert deal_slow_first(items, SLOW, workers=1) == items
    assert deal_slow_first(items, SLOW, workers=0) == items
    plain = _items(10, [])
    assert deal_slow_first(plain, SLOW, workers=4) == plain


def test_dealing_a_block_of_slow_tests_longer_than_the_chunk_is_kept_whole() -> None:
    items = _items(
        3, SLOW
    )  # chunk = max(8 // 4 // 2, 2) = 2, one worker gets 3 slow tests
    ordered = deal_slow_first(items, SLOW, workers=2)
    assert [i.nodeid for i in ordered[:3]] == [
        "tests/s.py::a",
        "tests/s.py::c",
        "tests/s.py::e",
    ]
    assert sorted(i.nodeid for i in ordered) == sorted(i.nodeid for i in items)
