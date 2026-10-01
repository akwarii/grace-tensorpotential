"""Logic tests for tools/check_pr_branch.py, run on real throw-away git repositories."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_pr_branch as cpb


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "master")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    git(tmp_path, "tag", "base")
    git(tmp_path, "checkout", "-q", "-b", "pr/U1-example")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def commit(repo, *paths):
    for p in paths:
        f = repo / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")


def test_clean_branch_passes(repo, capsys, monkeypatch):
    commit(repo, "tensorpotential/utils.py", "tests/test_x.py")
    monkeypatch.setattr(sys, "argv", ["check_pr_branch.py", "base", "HEAD"])
    cpb.main()
    assert "ok: 2 changed file(s), none fork-only" in capsys.readouterr().out


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        ".claude/skills/x/SKILL.md",
        ".github/PULL_REQUEST_TEMPLATE/torch-backend.md",
        ".github/ISSUE_TEMPLATE/milestone.md",
        "tools/board.py",
        "baselines/README.md",
        "plan/x.md",
        "TORCH_BACKEND_PLAN.md",
        "uv.lock",
        "tests_torch/test_a.py",
        "tensorpotential/torch_backend/model.py",
        "tensorpotential/core/neighbors.py",
    ],
)
def test_every_fork_only_path_is_rejected(repo, capsys, monkeypatch, path):
    commit(repo, "tensorpotential/utils.py", path)
    monkeypatch.setattr(sys, "argv", ["check_pr_branch.py", "base", "HEAD"])
    with pytest.raises(SystemExit) as e:
        cpb.main()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert path in out and "tensorpotential/utils.py" not in out


def test_a_removed_file_no_longer_counts(repo, capsys, monkeypatch):
    commit(repo, "tools/a.py")
    git(repo, "rm", "-q", "-r", "tools")
    git(repo, "commit", "-q", "-m", "remove")
    monkeypatch.setattr(sys, "argv", ["check_pr_branch.py", "base", "HEAD"])
    cpb.main()
    assert "ok: 0 changed file(s)" in capsys.readouterr().out


def test_changed_files_with_a_bad_reference_exits(repo):
    with pytest.raises(SystemExit, match="git diff failed"):
        cpb.changed_files("no-such-ref", "HEAD")


def test_default_arguments_are_upstream_master_and_head(repo, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cpb, "changed_files", lambda base, head: seen.update(base=base, head=head) or []
    )
    monkeypatch.setattr(sys, "argv", ["check_pr_branch.py"])
    cpb.main()
    assert seen == {"base": "upstream/master", "head": "HEAD"}
