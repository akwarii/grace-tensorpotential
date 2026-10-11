"""Logic tests for tools/ratchet_moves.py (no physics in this tool)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ratchet_moves as rm


def test_parse_moves_maps_old_to_new_and_accepts_nothing() -> None:
    assert rm.parse_moves([["a.py", "b.py"], ["c.py", "d/c.py"]]) == {
        "a.py": "b.py",
        "c.py": "d/c.py",
    }
    assert rm.parse_moves([]) == {}
    assert rm.parse_moves(None) == {}


def test_parse_moves_refuses_a_move_onto_itself() -> None:
    with pytest.raises(ValueError, match="the same"):
        rm.parse_moves([["a.py", "a.py"]])


def test_parse_moves_refuses_one_file_moved_twice() -> None:
    with pytest.raises(ValueError, match="moved twice"):
        rm.parse_moves([["a.py", "b.py"], ["a.py", "c.py"]])


def test_rekey_moves_the_entry_and_keeps_its_value_and_the_others() -> None:
    table = {"a.py": {"F401": 2}, "z.py": {"E722": 1}}
    assert rm.rekey(table, {"a.py": "pkg/a.py"}) == {
        "pkg/a.py": {"F401": 2},
        "z.py": {"E722": 1},
    }


def test_rekey_ignores_a_move_of_a_file_without_an_entry() -> None:
    assert rm.rekey({"z.py": 1}, {"a.py": "b.py"}) == {"z.py": 1}


def test_rekey_does_not_change_its_argument() -> None:
    table = {"a.py": 1}
    rm.rekey(table, {"a.py": "b.py"})
    assert table == {"a.py": 1}


def test_rekey_refuses_two_entries_for_one_path() -> None:
    with pytest.raises(ValueError, match=r"b\.py would hold two"):
        rm.rekey({"a.py": 1, "b.py": 2}, {"a.py": "b.py"})
    with pytest.raises(ValueError, match=r"c\.py would hold two"):
        rm.rekey({"a.py": 1, "b.py": 2}, {"a.py": "c.py", "b.py": "c.py"})


def test_a_swap_of_two_files_exchanges_their_entries() -> None:
    assert rm.rekey({"a.py": 1, "b.py": 2}, {"a.py": "b.py", "b.py": "a.py"}) == {
        "b.py": 1,
        "a.py": 2,
    }
