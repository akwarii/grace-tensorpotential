from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .slow_marks import (
    SLOW_LIST,
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
