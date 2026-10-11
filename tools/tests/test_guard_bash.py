"""Tests for tools/guard_bash.py, the PreToolUse hook that refuses commands CLAUDE.md forbids.

Git behaviour is real (a scratch repository with a linked worktree); the process table and the load average are replaced by
fixed values, because they are what the machine happens to be doing (an external boundary).
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guard_bash as gb

REAL_MACHINE_STATE = gb.machine_state  # the autouse fixture below replaces gb.machine_state by a fixed idle machine

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "PATH": __import__("os").environ["PATH"], "HOME": "/nonexistent",
}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository on torch-backend with one commit, and a linked worktree on the branch feat."""
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "-q", "-b", "torch-backend")
    (main / "a.txt").write_text("x")
    git(main, "add", "a.txt")
    git(main, "commit", "-q", "-m", "first")
    wt = tmp_path / "wt"
    git(main, "worktree", "add", "-q", "-b", "feat", str(wt))
    return main, wt


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    """By default the machine is idle; a test that needs it busy sets its own state."""
    monkeypatch.setattr(gb, "machine_state", lambda: (0.1, []))
    monkeypatch.delenv(gb.OVERRIDE_VAR, raising=False)
    monkeypatch.delenv("GRACE_BUSY_LOAD", raising=False)


def rules(command, cwd):
    return sorted(v.rule for v in gb.evaluate(command, Path(cwd)))


# ----------------------------------------------------------------------------------------------------- parsing


def test_split_commands_separates_operators_and_newlines_outside_quotes():
    assert gb.split_commands("a x && b; c | d\ne 'f\ng'") == [["a", "x"], ["b"], ["c"], ["d"], ["e", "f\ng"]]


def test_split_commands_ignores_here_document_bodies():
    command = "cat <<'EOF' > f\ngit commit -m x\nEOF\necho done"
    found = gb.split_commands(command)
    assert [w for w in found if w[0] == "git"] == []
    assert found[-1] == ["echo", "done"]


def test_split_commands_dash_heredoc_allows_an_indented_terminator():
    found = gb.split_commands("cat <<-EOT\n\tgit push\n\tEOT\nls")
    assert [w for w in found if w[0] == "git"] == []
    assert found[-1] == ["ls"]


def test_a_here_string_is_not_a_here_document_and_hides_nothing(tmp_path):
    assert rules("cat <<< foo\ngh pr merge 5", tmp_path) == ["merge"]


def test_a_backslash_newline_joins_the_lines(tmp_path):
    assert rules("gh pr \\\nmerge 5", tmp_path) == ["merge"]


@pytest.mark.parametrize(
    "command",
    ["bash -c 'gh project item-list 1'", "bash -lc 'gh project item-list 1'", "zsh -ic \"gh project item-list 1\"", "eval gh project item-list 1",
     "env FOO=1 gh project item-list 1", "command gh project item-list 1", "timeout 5 gh project item-list 1", "sudo -u root gh project item-list 1",
     "nice -n 5 gh project item-list 1", "time gh project item-list 1", "nohup gh project item-list 1", "exec gh project item-list 1"],
)
def test_wrappers_and_inner_scripts_are_looked_through(command, tmp_path):
    assert rules(command, tmp_path) == ["graphql"]


def test_an_override_written_outside_an_inner_script_applies_inside_it(tmp_path):
    assert rules("GRACE_HOOK_OVERRIDE=merge bash -c 'gh pr merge 5'", tmp_path) == []
    assert rules("GRACE_HOOK_OVERRIDE=merge bash -c 'gh pr merge 5; gh project item-list 1'", tmp_path) == ["graphql"]
    assert rules("env GRACE_HOOK_OVERRIDE=all gh pr merge 5", tmp_path) == []


def test_bash_without_a_script_and_a_bare_wrapper_are_harmless(tmp_path):
    assert rules("bash script.sh", tmp_path) == []
    assert rules("bash -c", tmp_path) == []
    assert rules("env", tmp_path) == []


def test_tokenize_falls_back_to_a_plain_split_on_an_unbalanced_quote():
    assert gb.tokenize("echo 'oops") == ["echo", "'oops"]


def test_strip_assignments_and_overrides(monkeypatch):
    words, env = gb._strip_assignments(["A=1", "GRACE_HOOK_OVERRIDE=busy, merge", "gh", "x"])
    assert words == ["gh", "x"]
    assert gb.overrides(env) == {"busy", "merge"}
    monkeypatch.setenv(gb.OVERRIDE_VAR, "all")
    assert gb.overrides({}) == {"all"}


# ----------------------------------------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    "command",
    ["gh project item-list 2 --owner a", "gh project field-list 2", "gh -R a/b project item-list 2", "echo hi && gh project item-list 2"],
)
def test_graphql_expensive_calls_are_refused(command, tmp_path):
    assert rules(command, tmp_path) == ["graphql"]


@pytest.mark.parametrize("command", ["gh project list --owner a", "gh issue list --repo a/b", "gh pr view 3", "python tools/board.py next"])
def test_cheap_board_calls_are_allowed(command, tmp_path):
    assert rules(command, tmp_path) == []


@pytest.mark.parametrize(
    "command",
    ["gh pr merge 5 --squash", "gh pr merge --auto 5", "gh pr ready 5", "gh api -X PUT repos/a/b/pulls/5/merge",
     "gh api graphql -f query='mutation { enablePullRequestAutoMerge(input:{}) { x } }'"],
)
def test_an_agent_never_merges_or_marks_ready(command, tmp_path):
    assert rules(command, tmp_path) == ["merge"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("gh pr create --repo ICAMS/grace-tensorpotential --title t", ["upstream"]),
        ("gh issue comment 3 -R icams/x --body b", ["upstream"]),
        ("gh api repos/ICAMS/x/issues -f title=t", ["upstream"]),
        ("gh api -X DELETE repos/ICAMS/x/issues/3", ["upstream"]),
        ("gh issue view 3 --repo ICAMS/x", []),
        ("gh api repos/ICAMS/x/issues", []),
        ("gh pr create --repo akwarii/x --draft --title t", []),
    ],
)
def test_upstream_writes_are_refused_and_reads_are_allowed(command, expected, tmp_path):
    assert rules(command, tmp_path) == expected


def test_worktree_remove_without_force_is_allowed(repo):
    main, wt = repo
    assert rules(f"git worktree remove {wt}", main) == []
    assert rules(f"git worktree remove --force {wt}", main) == ["worktree-force"]
    assert rules(f"git worktree remove -f {wt}", main) == ["worktree-force"]


def test_integration_branch_is_protected_from_history_rewrites(repo):
    main, wt = repo
    for sub in ("commit -m x", "merge feat", "rebase feat", "cherry-pick abc", "reset --hard HEAD", "pull origin", "am p.patch", "revert HEAD"):
        assert rules(f"git {sub}", main) == ["integration-branch"], sub
    assert rules("git commit -m x", wt) == []  # a work branch is free
    assert rules("git status && git log && git diff", main) == []


def test_fast_forward_updates_and_soft_resets_stay_allowed_on_the_integration_branch(repo):
    main, _ = repo
    assert rules("git pull --ff-only", main) == []
    assert rules("git merge --ff-only origin/torch-backend", main) == []
    assert rules("git reset --soft HEAD~1", main) == []


def test_git_dash_c_and_cd_choose_the_directory_that_counts(repo):
    main, wt = repo
    assert rules(f"git -C {main} commit -m x", wt) == ["integration-branch"]
    assert rules(f"cd {main} && git commit -m x", wt) == ["integration-branch"]
    assert rules(f"cd {wt} && git commit -m x", main) == []
    assert rules("git -c user.name=x --no-pager commit -m x", main) == ["integration-branch"]
    assert rules("cd - && git status", main) == []


@pytest.mark.parametrize(
    ("command", "cwd_is_main", "expected"),
    [
        ("git push origin torch-backend", False, ["integration-branch"]),
        ("git push origin HEAD:master", False, ["integration-branch"]),
        ("git push -f origin +feat:refs/heads/torch-backend", False, ["integration-branch"]),
        ("git push", True, ["integration-branch"]),
        ("git push origin HEAD", True, ["integration-branch"]),
        ("git push upstream feat", False, ["upstream"]),
        ("git push https://github.com/ICAMS/grace-tensorpotential feat", False, ["upstream"]),
        ("git push -u origin feat", False, []),
        ("git push origin HEAD:docs-branch", False, []),
        ("git push origin HEAD", False, []),
    ],
)
def test_pushes(command, cwd_is_main, expected, repo):
    main, wt = repo
    assert rules(command, main if cwd_is_main else wt) == expected


def test_uv_sync_is_refused_in_a_linked_worktree_only(repo):
    main, wt = repo
    assert rules("uv sync --group dev", wt) == ["worktree-sync"]
    assert rules("uv sync --frozen --group dev", main) == []
    assert rules("UV_PROJECT_ENVIRONMENT=/x/.venv uv sync --no-install-project --inexact", wt) == []
    assert rules("uv run --frozen --no-sync pytest tests/test_x.py", wt) == []


def test_git_state_helpers_on_a_directory_that_is_not_a_repository(tmp_path):
    assert gb.current_branch(tmp_path / "missing") == ""
    assert gb.in_linked_worktree(tmp_path) is False
    assert gb.check_git(["git", "commit"], tmp_path) is None
    assert gb.check_git(["git"], tmp_path) is None


# ----------------------------------------------------------------------------------------------------- the busy rule


@pytest.mark.parametrize(
    "command",
    ["uv run --frozen --no-sync pytest tests -q -n 4 --dist load", "pytest -n4 tests", "python -m pytest tests --cov=tensorpotential",
     "pytest tests --splits 4 --group 1", "gracemaker input.yaml", "python tools/oracle_snapshot.py write new.npz", "python tools/make_golden.py write --tier tiny"],
)
def test_suite_commands_are_recognised(command):
    assert gb.is_suite_command(gb.split_commands(command)[0])


@pytest.mark.parametrize(
    "command",
    ["pytest tests/test_x.py -q", "uv run pytest tests/test_x.py::test_y -vv", "python tools/oracle_snapshot.py compare a b", "bash -c 'pytest -n 4'",
     "echo 'pytest -n 4'", "ls"],
)
def test_single_file_runs_and_other_commands_are_not_suites(command):
    assert not gb.is_suite_command(gb.split_commands(command)[0])


def test_a_suite_while_another_runs_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "machine_state", lambda: (0.2, ["123 400 /x/.venv/bin/python -m pytest tests -n 4"]))
    verdicts = gb.evaluate("uv run --frozen --no-sync pytest tests -n 4", tmp_path)
    assert [v.rule for v in verdicts] == ["busy"]
    assert "another test or training process is running" in verdicts[0].message
    assert rules("pytest tests/test_x.py", tmp_path) == []  # one file, serially, is allowed while busy


def test_a_suite_under_a_high_load_is_refused_and_the_limit_can_be_changed(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "machine_state", lambda: (6.5, []))
    assert rules("pytest tests -n 4", tmp_path) == ["busy"]
    monkeypatch.setenv("GRACE_BUSY_LOAD", "8")
    assert rules("pytest tests -n 4", tmp_path) == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("/x/.venv/bin/python -m pytest tests -n 4", True),
        ("/x/.venv/bin/python /x/.venv/bin/gracemaker input.yaml", True),
        ("pytest tests", True),
        ("uv run --frozen pytest tests", True),
        ("python tools/oracle_snapshot.py write n.npz", True),
        ("/bin/zsh -c 'pytest -n 4 tests'", False),
        ("grep pytest log.txt", False),
        ("python script.py", False),
        ("", False),
        ("python -c 'oops", False),
    ],
)
def test_is_suite_process_looks_at_the_executable_not_at_a_shell_that_names_it(args, expected):
    assert gb.is_suite_process(args) is expected


def test_busy_reason_prefers_a_process_over_the_load_and_ignores_malformed_lines():
    assert gb.busy_reason(0.1, ["7 12"]) is None
    assert "process" in str(gb.busy_reason(9.0, ["7 12 pytest tests"]))
    assert "load average is 9.0" in str(gb.busy_reason(9.0, []))
    assert gb.busy_reason(4.0, []) is None


def test_machine_state_reads_the_load_and_the_process_table():
    load1, lines = REAL_MACHINE_STATE()
    assert isinstance(load1, float)
    assert all(isinstance(x, str) for x in lines)


def test_machine_state_survives_a_missing_ps(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ps")

    monkeypatch.setattr(gb.subprocess, "run", boom)
    assert REAL_MACHINE_STATE()[1] == []


# ----------------------------------------------------------------------------------------------------- overrides and the hook


def test_an_explicit_override_lets_the_named_rule_through_and_only_that_one(tmp_path):
    assert rules("GRACE_HOOK_OVERRIDE=merge gh pr merge 5", tmp_path) == []
    assert rules("GRACE_HOOK_OVERRIDE=graphql gh pr merge 5", tmp_path) == ["merge"]
    assert rules("GRACE_HOOK_OVERRIDE=all gh project item-list 2", tmp_path) == []
    assert rules("GRACE_HOOK_OVERRIDE=graphql,merge gh pr merge 5", tmp_path) == []


def run_hook(monkeypatch, payload, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload if isinstance(payload, str) else json.dumps(payload)))
    code = gb.main()
    return code, capsys.readouterr().err


def test_hook_blocks_with_exit_status_2_and_names_the_override(monkeypatch, capsys, tmp_path):
    code, err = run_hook(monkeypatch, {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 5"}, "cwd": str(tmp_path)}, capsys)
    assert code == 2
    assert "BLOCKED by tools/guard_bash.py (merge)" in err
    assert "GRACE_HOOK_OVERRIDE=merge" in err


def test_hook_lets_an_allowed_command_other_tools_and_garbage_through(monkeypatch, capsys, tmp_path):
    assert run_hook(monkeypatch, {"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": str(tmp_path)}, capsys) == (0, "")
    assert run_hook(monkeypatch, {"tool_name": "Edit", "tool_input": {"command": "gh pr merge 5"}}, capsys) == (0, "")
    code, err = run_hook(monkeypatch, "not json", capsys)
    assert code == 0
    assert "ignored an internal error" in err


def test_the_committed_settings_run_this_script_for_bash():
    settings = json.loads((Path(__file__).resolve().parents[2] / ".claude" / "settings.json").read_text())
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "Bash"
    assert "tools/guard_bash.py" in entry["hooks"][0]["command"]


def test_script_runs_as_a_program(tmp_path):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "gh project item-list 2"}, "cwd": str(tmp_path)})
    r = subprocess.run([sys.executable, str(Path(gb.__file__)), ], input=payload, capture_output=True, text=True, check=False)
    assert r.returncode == 2
    assert "graphql" in r.stderr


def test_a_backslash_escapes_the_next_character_and_an_assignment_alone_is_not_a_command(tmp_path):
    assert gb.split_commands("echo a\\ b \\' c") == [["echo", "a b", "'", "c"]]
    assert rules("FOO=1", tmp_path) == []
    assert rules("FOO=1; gh project item-list 2", tmp_path) == ["graphql"]


def test_repo_given_as_an_equals_option_counts_as_upstream(tmp_path):
    assert rules("gh issue create --repo=ICAMS/x --title t", tmp_path) == ["upstream"]


def test_git_failures_read_as_no_state(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(gb.subprocess, "run", boom)
    assert gb.current_branch(tmp_path) == ""


# ----------------------------------------------------------------------------------------------------- rules added after review


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("gh pr create --title t", ["repo"]),
        ("gh issue create --title t", ["repo"]),
        ("gh pr edit 3 --body b", ["repo"]),
        ("gh pr view 3", []),
        ("gh pr create --repo akwarii/x --title t", ["draft"]),
        ("gh pr create --repo akwarii/x --draft --title t", []),
        ("gh pr create --repo akwarii/x -d --title t", []),
        ("gh issue create --repo akwarii/x --title t", []),
        ("GH_REPO=ICAMS/x gh pr create --draft --title t", ["upstream"]),
        ("GH_REPO=akwarii/x gh pr create --draft --title t", []),
    ],
)
def test_gh_writes_name_their_repository_and_pull_requests_are_drafts(command, expected, tmp_path):
    assert rules(command, tmp_path) == expected


def test_gh_repo_environment_variable_of_the_hook_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("GH_REPO", "ICAMS/x")
    assert rules("gh issue comment 3 --body b", tmp_path) == ["upstream"]


def test_push_all_and_mirror_are_refused_from_any_branch(repo):
    _, wt = repo
    assert rules("git push --all origin", wt) == ["integration-branch"]
    assert rules("git push --mirror origin", wt) == ["integration-branch"]


def test_aborting_or_previewing_on_the_integration_branch_is_harmless(repo):
    main, _ = repo
    for sub in ("rebase --abort", "merge --abort", "commit --dry-run", "am --quit"):
        assert rules(f"git {sub}", main) == [], sub


@pytest.mark.parametrize(
    "command",
    ["pkill -f pytest", "killall gracemaker", "pkill -9 -f 'python.*pytest'", "bash -c 'pkill -f gracemaker'"],
)
def test_killing_a_test_or_training_process_is_refused(command, tmp_path):
    assert rules(command, tmp_path) == ["kill"]


@pytest.mark.parametrize("command", ["kill 1234", "pkill -f my_server", "kill -9 $PID"])
def test_other_kills_are_not_this_rules_business(command, tmp_path):
    assert rules(command, tmp_path) == []


@pytest.mark.parametrize(
    ("command", "suite"),
    [
        ("pytest tests -q", True),
        ("pytest", True),
        ("uv run --frozen --no-sync pytest -q", True),
        ("pytest tests/ -x", True),
        ("pytest tests/test_x.py -n 2", True),
        ("pytest tests_torch -q", False),
        ("pytest tools/tests -q", True),
        ("pytest tools/tests/test_board.py -q", False),
        ("pytest tests/test_x.py::test_y", False),
        ("pytest -k name tests/test_x.py", False),
        ("gracemaker --help", False),
        ("gracemaker -h input.yaml", False),
        ("uv run --frozen gracemaker input.yaml", True),
        ("git log -- gracemaker", False),
        ("grep gracemaker README.md", False),
    ],
)
def test_which_commands_count_as_suite_runs(command, suite):
    assert gb.is_suite_command(gb.split_commands(command)[0]) is suite


def test_a_busy_machine_refuses_a_whole_suite_run_even_without_parallelism(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "machine_state", lambda: (9.0, []))
    assert rules("pytest tests -q", tmp_path) == ["busy"]
    assert rules("pytest tests_torch -q", tmp_path) == []
    assert rules("gracemaker --help", tmp_path) == []
    assert rules("git log -- gracemaker", tmp_path) == []
