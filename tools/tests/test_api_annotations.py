"""Logic tests for tools/api_annotations.py (no physics in this tool).

Sources are written to ``tmp_path`` and the counts expected are done by hand from the text below.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api_annotations as aa

SOURCE = '''\
def full(a) -> int:
    """Documented."""
    return a


def no_return(a):
    """Documented."""
    return a


def no_doc(a) -> int:
    return a


def neither(a):
    return a


def _private(a):
    return a


async def coro(a) -> int:
    """Documented."""
    return a


def outer(a) -> int:
    """Documented."""

    def nested(b):
        return b

    return nested(a)


class Shown:
    def method(self) -> int:
        return 1

    class Inner:
        def deep(self):
            """Documented."""

    def _hidden(self):
        pass


class _Hidden:
    def method(self):
        pass
'''
# public: full, no_return, no_doc, neither, coro, outer, Shown.method, Shown.Inner.deep = 8
# return annotation: full, no_doc, coro, outer, method = 5; docstring: full, no_return, coro, outer, deep = 5
# both: full, coro, outer = 3
COUNTS = aa.Counts(total=8, returns=5, docstrings=5, both=3)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(
        'def only() -> None:\n    """Doc."""\n', encoding="utf-8"
    )
    return tmp_path


SCOPE = ("pkg/a.py", "pkg/b.py")


def test_public_functions_are_reached_through_public_names_only(tree: Path) -> None:
    names = [f.name for f in aa.parse_public(tree / "pkg" / "a.py")]
    assert names == [
        "full",
        "no_return",
        "no_doc",
        "neither",
        "coro",
        "outer",
        "method",
        "deep",
    ]


def test_counts_are_those_counted_by_hand(tree: Path) -> None:
    per_file = aa.measure(tree, SCOPE)
    assert per_file["pkg/a.py"] == COUNTS
    assert per_file["pkg/b.py"] == aa.Counts(1, 1, 1, 1)


def test_an_empty_set_counts_as_complete() -> None:
    assert aa.Counts().share(0) == 1.0


def test_a_file_of_the_scope_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pkg/nope.py"):
        aa.measure(tmp_path, ("pkg/nope.py",))


def test_incomplete_lists_what_each_function_lacks(tree: Path) -> None:
    assert aa.incomplete(tree, SCOPE) == [
        "pkg/a.py:6 no_return (return annotation)",
        "pkg/a.py:11 no_doc (docstring)",
        "pkg/a.py:15 neither (return annotation, docstring)",
        "pkg/a.py:38 method (docstring)",
        "pkg/a.py:42 deep (return annotation)",
    ]


def test_shortfalls_name_the_share_under_the_bar(tree: Path) -> None:
    per_file = aa.measure(tree, ("pkg/b.py",))
    assert aa.shortfalls(per_file, 0.9) == []
    per_file = aa.measure(tree, SCOPE)  # 6 of 9 each
    assert len(aa.shortfalls(per_file, 0.9)) == 2
    assert aa.shortfalls(per_file, 0.5) == []


def test_the_table_has_a_line_per_file_and_a_total(tree: Path) -> None:
    lines = aa.render(aa.measure(tree, SCOPE)).splitlines()
    assert [line.split()[0] for line in lines[1:]] == ["pkg/a.py", "pkg/b.py", "total"]
    assert lines[-1].split()[1] == "9"


def test_main_check_exits_1_under_the_bar_and_0_over_it(
    tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(aa, "SCOPE", SCOPE)
    assert aa.main(["check", "--root", str(tree)]) == 1
    assert "return annotations" in capsys.readouterr().err
    assert aa.main(["check", "--root", str(tree), "--min", "0.5"]) == 0
    assert aa.main(["table", "--root", str(tree)]) == 0
    assert aa.main(["missing", "--root", str(tree)]) == 0
    assert "no_return" in capsys.readouterr().out
