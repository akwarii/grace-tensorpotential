"""Logic tests for tools/ast_manifest.py (no physics in this tool)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast_manifest as am

BASE = '''\
"""Module doc."""
import os


def f(x, y=1):
    """Doc."""
    return x + y  # trailing comment
'''


def test_comment_only_change_keeps_hash():
    edited = BASE.replace("# trailing comment", "# other\n# lines") + "\n# footer\n"
    assert am.hash_source(edited) == am.hash_source(BASE)


def test_code_change_alters_hash_in_both_modes():
    edited = BASE.replace("x + y", "x - y")
    for normalised in (False, True):
        assert am.hash_source(edited, normalised) != am.hash_source(BASE, normalised)


def test_docstring_change_only_ignored_when_normalised():
    edited = BASE.replace('"""Doc."""', '"""Changed."""').replace(
        '"""Module doc."""', '"""Changed module."""'
    )
    assert am.hash_source(edited) != am.hash_source(BASE)
    assert am.hash_source(edited, True) == am.hash_source(BASE, True)


def test_removing_docstring_ignored_when_normalised():
    edited = BASE.replace('    """Doc."""\n', "")
    assert am.hash_source(edited, True) == am.hash_source(BASE, True)


def test_annotation_change_only_ignored_when_normalised():
    typed = BASE.replace("def f(x, y=1):", "def f(x: int, y: int = 1) -> int:")
    retyped = typed.replace("x: int", "x: float")
    assert am.hash_source(typed) != am.hash_source(BASE)
    assert am.hash_source(typed, True) == am.hash_source(BASE, True)
    assert am.hash_source(retyped, True) == am.hash_source(typed, True)


def test_async_functions_are_normalised_like_plain_ones():
    plain = "async def g(x):\n    return x\n"
    typed = 'async def g(x: int) -> int:\n    """Doc."""\n    return x\n'
    assert am.hash_source(typed) != am.hash_source(plain)
    assert am.hash_source(typed, True) == am.hash_source(plain, True)
    assert am.hash_source(
        typed.replace("return x", "return -x"), True
    ) != am.hash_source(plain, True)


def test_annotated_field_is_not_dropped_when_normalised():
    with_field = "class C:\n    a: int\n    b: int = 2\n"
    without_field = "class C:\n    b: int = 2\n"
    assert am.hash_source(with_field, True) != am.hash_source(without_field, True)


def test_docstring_only_body_keeps_a_valid_body():
    assert am.hash_source('def g():\n    """Only a doc."""\n', True) == am.hash_source(
        "def g():\n    pass\n", True
    )


def test_non_docstring_string_statement_is_kept():
    # A string that is not the first statement is code as far as the AST is concerned.
    a = 'def g():\n    x = 1\n    "not a docstring"\n'
    b = "def g():\n    x = 1\n"
    assert am.hash_source(a, True) != am.hash_source(b, True)


def test_syntax_error_is_recorded_not_raised():
    assert am.hash_source("def (:\n") == am.SYNTAX_ERROR


def test_diff_manifests_reports_added_removed_changed():
    expected = {"a.py": "1", "b.py": "2", "c.py": "3"}
    actual = {"a.py": "1", "b.py": "changed", "d.py": "4"}
    assert am.diff_manifests(expected, actual) == {
        "added": ["d.py"],
        "removed": ["c.py"],
        "changed": ["b.py"],
    }


def _git(repo, *args):
    # git from PATH, arguments written in this file: not user input
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(BASE)
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("not python\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    _git(tmp_path, "tag", "base")
    return tmp_path


def test_tracked_files_are_python_only_and_sorted(repo):
    assert am.tracked_python_files(repo) == ["b.py", "pkg/a.py"]
    assert am.tracked_python_files(repo, "base") == ["b.py", "pkg/a.py"]


def test_file_listing_at_rev_ignores_later_commits(repo):
    (repo / "later.py").write_text("w = 1\n")
    _git(repo, "add", "later.py")
    _git(repo, "commit", "-q", "-m", "later")
    assert "later.py" in am.tracked_python_files(repo)
    assert "later.py" not in am.tracked_python_files(repo, "base")
    assert "later.py" not in am.build_manifest(repo, "base")


def test_untracked_python_file_is_not_in_manifest(repo):
    (repo / "scratch.py").write_text("z = 3\n")
    assert "scratch.py" not in am.build_manifest(repo)


def test_working_tree_and_rev_agree_until_edited(repo):
    assert am.build_manifest(repo) == am.build_manifest(repo, "base")
    (repo / "b.py").write_text("y = 3\n")
    assert am.build_manifest(repo) != am.build_manifest(repo, "base")


def test_cli_write_then_check_roundtrip(repo, capsys):
    out = repo / "m.json"
    assert am.main(["write", str(out), "--root", str(repo)]) == 0
    assert set(json.loads(out.read_text())) == {"b.py", "pkg/a.py"}
    assert am.main(["check", str(out), "--root", str(repo)]) == 0

    (repo / "pkg" / "a.py").write_text(BASE.replace("# trailing comment", "# new"))
    assert am.main(["check", str(out), "--root", str(repo)]) == 0

    (repo / "pkg" / "a.py").write_text(BASE.replace("x + y", "x * y"))
    assert am.main(["check", str(out), "--root", str(repo)]) == 1
    assert "changed: pkg/a.py" in capsys.readouterr().out


def test_cli_check_against_rev_sees_committed_change(repo):
    out = repo / "m.json"
    am.main(["write", str(out), "--root", str(repo), "--rev", "base"])
    (repo / "b.py").write_text("y = 5\n")
    _git(repo, "commit", "-qam", "edit")
    assert am.main(["check", str(out), "--root", str(repo), "--rev", "HEAD"]) == 1
    assert am.main(["check", str(out), "--root", str(repo), "--rev", "base"]) == 0
