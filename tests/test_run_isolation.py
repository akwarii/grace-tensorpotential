"""Tests of the scratch-directory helper that keeps the test runs out of the source tree."""

from pathlib import Path

import pytest

from .utils import isolated_run_dir

DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_relative_data_path_resolves_from_the_run_dir():
    with isolated_run_dir(DATA_DIR) as run_dir:
        # input files of the integration tests say "../data/<file>"
        resolved = (run_dir / ".." / "data" / "MoNbTaW_train50.pkl.gz").resolve()
        assert resolved == DATA_DIR / "MoNbTaW_train50.pkl.gz"
        assert resolved.is_file()


def test_run_dir_starts_empty_and_is_outside_the_source_tree():
    with isolated_run_dir(DATA_DIR) as run_dir:
        assert list(run_dir.iterdir()) == []
        assert Path(__file__).parent.resolve() not in run_dir.resolve().parents


def test_run_dir_is_removed_on_exit_and_data_is_untouched():
    with isolated_run_dir(DATA_DIR) as run_dir:
        (run_dir / "seed").mkdir()
        root = run_dir.parent
    assert not root.exists()
    assert (DATA_DIR / "MoNbTaW_train50.pkl.gz").is_file()


def test_run_dir_is_removed_when_the_body_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with isolated_run_dir(DATA_DIR) as run_dir:
            root = run_dir.parent
            raise RuntimeError("boom")
    assert not root.exists()


def test_run_dirs_are_distinct():
    with isolated_run_dir(DATA_DIR) as first, isolated_run_dir(DATA_DIR) as second:
        assert first != second
