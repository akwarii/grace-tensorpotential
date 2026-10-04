"""Tests for tools/divergence.py, run on real throw-away git repositories.

The fixture repository has an upstream history (``base``) with three files and a fork branch on top of it; each test changes
the fork the way a real change would (edit, delete, rename, add) and checks what the ledger check reports.
The last test runs the check on this repository against the committed ledger.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import divergence as dv

REPO = Path(__file__).resolve().parents[2]


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def commit(repo, files=None, remove=(), message="change"):
    for path, text in (files or {}).items():
        f = repo / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    for path in remove:
        (repo / path).unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def row(path, **kw):
    return {
        "path": path,
        "reason": "r",
        "milestone": ["X1"],
        "upstream": "pending",
    } | kw


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "master")
    commit(
        tmp_path,
        {"a.py": "a\n" * 20, "b.py": "b\n" * 20, "pkg/c.py": "c\n" * 20},
        message="upstream",
    )
    git(tmp_path, "tag", "base")
    git(tmp_path, "checkout", "-q", "-b", "fork")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def problems(rows, base="base", head="HEAD"):
    return dv.ledger_problems(rows, dv.modified_upstream_files(base, head))


def test_no_change_and_no_rows_is_exact(repo):
    assert dv.modified_upstream_files("base", "HEAD") == []
    assert problems([]) == []


def test_modified_file_needs_a_row(repo):
    commit(repo, {"a.py": "changed\n"})
    assert dv.modified_upstream_files("base", "HEAD") == ["a.py"]
    assert problems([row("a.py")]) == []
    assert problems([]) == ["a.py: modified from upstream, no row in the ledger"]


def test_deleted_file_needs_a_row(repo):
    commit(repo, remove=["pkg/c.py"])
    assert problems([]) == ["pkg/c.py: modified from upstream, no row in the ledger"]
    assert problems([row("pkg/c.py")]) == []


def test_added_file_needs_no_row_and_a_row_for_it_is_stale(repo):
    commit(repo, {"tools/new.py": "x\n"})
    assert problems([]) == []
    assert problems([row("tools/new.py")]) == [
        "tools/new.py: row is stale, the file no longer differs from upstream"
    ]


def test_rename_counts_as_deletion_of_the_upstream_path(repo):
    git(repo, "mv", "b.py", "b2.py")
    git(
        repo,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "rename",
    )
    assert dv.modified_upstream_files("base", "HEAD") == ["b.py"]
    assert problems([row("b.py")]) == []


def test_row_whose_file_is_back_to_upstream_is_stale(repo):
    commit(repo, {"a.py": "changed\n"})
    commit(repo, {"a.py": "a\n" * 20}, message="revert")
    assert problems([row("a.py")]) == [
        "a.py: row is stale, the file no longer differs from upstream"
    ]


def test_row_for_an_untouched_file_is_stale(repo):
    assert problems([row("b.py")]) == [
        "b.py: row is stale, the file no longer differs from upstream"
    ]


def test_changes_that_only_upstream_made_do_not_count(repo):
    git(repo, "checkout", "-q", "master")
    commit(repo, {"b.py": "upstream moved on\n"}, message="upstream moves")
    git(repo, "tag", "-f", "base")
    git(repo, "checkout", "-q", "fork")
    assert dv.modified_upstream_files("base", "HEAD") == []


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (
            {"path": "a.py", "milestone": ["X1"], "upstream": "pending"},
            "row 1 (a.py): missing reason",
        ),
        (
            {"reason": "r", "milestone": ["X1"], "upstream": "pending"},
            "row 1 (?): missing path",
        ),
        (row("a.py", reason="  "), "a.py: empty reason"),
        (row("a.py", milestone=[]), "a.py: empty milestone"),
        (
            row("a.py", upstream="soon"),
            "a.py: upstream is 'soon', expected one of pending, merged, never",
        ),
        ("a.py", "row 1: not a mapping"),
    ],
)
def test_malformed_rows(bad, message):
    assert dv.row_problems([bad]) == [message]


def test_duplicate_path_is_reported():
    assert dv.row_problems([row("a.py"), row("a.py")]) == ["a.py: two rows"]


@pytest.mark.parametrize("state", dv.UPSTREAM_STATES)
def test_every_upstream_state_is_accepted(state):
    assert dv.row_problems([row("a.py", upstream=state)]) == []


def test_load_ledger_rejects_a_mapping(tmp_path):
    f = tmp_path / "l.yaml"
    f.write_text("a: 1\n")
    with pytest.raises(SystemExit, match="must be a YAML list"):
        dv.load_ledger(f)


def test_git_failure_exits_with_the_message(repo):
    with pytest.raises(SystemExit, match="failed"):
        dv.modified_upstream_files("no-such-ref", "HEAD")


def run_main(repo, ledger_rows, *extra):
    ledger = repo.parent / "ledger.yaml"
    ledger.write_text(yaml.safe_dump(ledger_rows))
    dv.main(["check", "--base", "base", "--ledger", str(ledger), *extra])


def test_main_ok_names_the_counts(repo, capsys):
    commit(repo, {"a.py": "changed\n"})
    run_main(repo, [row("a.py")])
    assert (
        "ok: 1 modified upstream file(s), each with one row; 0 pr/U* branch(es) checked"
        in capsys.readouterr().out
    )


def test_main_exit_lists_every_problem(repo, capsys):
    commit(repo, {"a.py": "changed\n"})
    with pytest.raises(SystemExit, match="2 problem"):
        run_main(repo, [row("b.py")])
    out = capsys.readouterr().out
    assert "a.py: modified from upstream, no row" in out
    assert "b.py: row is stale" in out


def pr_branch(repo, name, files):
    git(repo, "checkout", "-q", "-b", name, "base")
    commit(repo, files, message=name)
    git(repo, "checkout", "-q", "fork")


def test_pr_branches_lists_only_the_pr_u_branches(repo):
    pr_branch(repo, "pr/U1-fix", {"a.py": "x\n"})
    git(repo, "branch", "pr/other")
    git(repo, "branch", "feature")
    assert dv.pr_branches() == ["pr/U1-fix"]


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        ".claude/skills/x/SKILL.md",
        "tools/board.py",
        "baselines/x.json",
        "uv.lock",
        ".github/ISSUE_TEMPLATE/m.md",
    ],
)
def test_pr_branch_with_a_fork_only_path_is_reported(repo, path):
    pr_branch(repo, "pr/U1-fix", {"a.py": "x\n", path: "x\n"})
    assert dv.pr_branch_problems("base", ["pr/U1-fix"]) == [
        f"pr/U1-fix: fork-only path {path}"
    ]


def test_pr_branch_check_runs_from_main(repo, capsys):
    pr_branch(repo, "pr/U1-fix", {"a.py": "x\n"})
    pr_branch(repo, "pr/U2-bad", {"CLAUDE.md": "x\n"})
    with pytest.raises(SystemExit, match="1 problem"):
        run_main(repo, [], "--pr-branches")
    assert "pr/U2-bad: fork-only path CLAUDE.md" in capsys.readouterr().out


def test_the_committed_ledger_matches_this_repository_against_upstream():
    refs = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "upstream/master"],
        cwd=REPO,
        capture_output=True,
    )
    if refs.returncode != 0:
        pytest.skip(
            "no upstream/master ref in this clone (git remote add upstream <url>; git fetch upstream)"
        )
    rows = dv.load_ledger(dv.LEDGER)
    modified = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=MD",
            "upstream/master...HEAD",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert dv.ledger_problems(rows, sorted(modified)) == []
