"""Logic tests for tools/move_manifest.py: a moved unit that changed or an old name that vanished fails the check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import move_manifest as mm

OLD_DATA = '''\
import os

LOG = 1


def load(x, y=2):
    """Doc."""
    return x + y  # a comment


class Holder:
    size: int = 3
'''
REEXPORT = "from tensorpotential.data.datasets import LOG, Holder, load\n"


def _tree(root: Path, old: str, new: str) -> None:
    """Write a scratch tree holding ``old`` and ``new`` for every module of ``MOVES``."""
    for move in mm.MOVES:
        for rel, text in ((move.old, old), (move.new, new)):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def _scoped_moves(monkeypatch) -> None:
    """Reduce ``MOVES`` to two modules with the unit lists of the scratch tree."""
    moves = (
        mm.Move(mm.MOVES[0].old, mm.MOVES[0].new, None),
        mm.Move(mm.MOVES[3].old, mm.MOVES[3].new, ("load",)),
    )
    monkeypatch.setattr(mm, "MOVES", moves)


def test_units_are_hashed_per_name_and_comments_do_not_count():
    units = mm.top_level_units(OLD_DATA)
    assert set(units) == {"LOG", "load", "Holder"}
    assert mm.top_level_units(OLD_DATA.replace("# a comment", "# other")) == units
    assert (
        mm.top_level_units(OLD_DATA.replace("x + y", "x - y"))["load"] != units["load"]
    )


def test_a_faithful_move_with_a_reexport_module_passes(tmp_path, monkeypatch):
    _scoped_moves(monkeypatch)
    _tree(tmp_path, OLD_DATA, OLD_DATA)
    manifest = mm.record(None, tmp_path)
    for move in mm.MOVES:
        (tmp_path / move.old).write_text(REEXPORT)
    assert mm.check(manifest, tmp_path) == []


def test_a_function_changed_during_the_move_is_reported(tmp_path, monkeypatch):
    _scoped_moves(monkeypatch)
    _tree(tmp_path, OLD_DATA, OLD_DATA)
    manifest = mm.record(None, tmp_path)
    for move in mm.MOVES:
        (tmp_path / move.old).write_text(REEXPORT)
        (tmp_path / move.new).write_text(OLD_DATA.replace("x + y", "x - y"))
    problems = mm.check(manifest, tmp_path)
    assert any("load changed" in p for p in problems)


def test_an_old_name_dropped_from_the_reexport_module_is_reported(
    tmp_path, monkeypatch
):
    _scoped_moves(monkeypatch)
    _tree(tmp_path, OLD_DATA, OLD_DATA)
    manifest = mm.record(None, tmp_path)
    for move in mm.MOVES:
        (tmp_path / move.old).write_text(
            "from tensorpotential.data.datasets import LOG, load\n"
        )
    problems = mm.check(manifest, tmp_path)
    assert any("Holder is no longer importable" in p for p in problems)


def test_a_missing_unit_at_the_new_path_is_reported(tmp_path, monkeypatch):
    _scoped_moves(monkeypatch)
    _tree(tmp_path, OLD_DATA, OLD_DATA)
    manifest = mm.record(None, tmp_path)
    for move in mm.MOVES:
        (tmp_path / move.old).write_text(REEXPORT)
        (tmp_path / move.new).write_text("LOG = 1\n")
    assert any("load is missing" in p for p in mm.check(manifest, tmp_path))


def test_main_records_then_checks_with_the_exit_status(tmp_path, monkeypatch, capsys):
    _scoped_moves(monkeypatch)
    monkeypatch.setattr(mm, "ROOT", tmp_path)
    _tree(tmp_path, OLD_DATA, OLD_DATA)
    target = tmp_path / "moves.json"
    assert mm.main(["record", str(target)]) == 0
    assert set(json.loads(target.read_text())) == {m.old for m in mm.MOVES}
    for move in mm.MOVES:
        (tmp_path / move.old).write_text(REEXPORT)
    assert mm.main(["check", str(target)]) == 0
    (tmp_path / mm.MOVES[0].new).write_text("LOG = 1\n")
    assert mm.main(["check", str(target)]) == 1
    assert "load is missing" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    ["data/datasets.py", "export_kokkos.py", "data/preprocess.py", "model_ops.py"],
)
def test_the_new_paths_are_the_decided_names(name):
    assert f"tensorpotential/{name}" in {m.new for m in mm.MOVES}
