#!/usr/bin/env python3
"""PreToolUse hook for the Bash tool: refuses the commands that `CLAUDE.md` says an agent never runs.

Claude Code starts it before every Bash command (`.claude/settings.json`) and passes a JSON object on stdin
(`tool_input.command`, `cwd`). Exit status 2 blocks the command and shows the message on stderr to the agent; any
other exit status lets it run. It is a guard rail against forgetting a rule, not a sandbox: it reads the command
text (following `cd`, `git -C`, `bash -c`, `eval`, `env`/`timeout`/`sudo`-style wrappers, line continuations and
here-documents) and cannot see variables, scripts or what a program does. An internal error lets the command run.

Rules (the identifier is what an override names):

``graphql``
    ``gh project item-list`` and ``gh project field-list`` cost 100 and 150 GraphQL points of the shared budget.
``worktree-sync``
    ``uv sync`` inside a linked worktree repoints the shared editable install to the worktree.
``integration-branch``
    No commit, merge, rebase, reset or pull without ``--ff-only`` on ``torch-backend`` or ``master``, and no push to them
    (also ``--all`` and ``--mirror``) from any branch.
``merge``
    An agent never merges a pull request, marks it ready or turns on auto-merge: the user does.
``draft``
    A pull request is opened as a draft (``gh pr create --draft``).
``repo``
    ``gh pr`` and ``gh issue`` writes name their repository (``--repo akwarii/grace-tensorpotential``): the default
    repository of ``gh`` here is the upstream one.
``upstream``
    Nothing is pushed to, or written on, the upstream ``ICAMS`` repositories without an explicit go.
``worktree-force``
    ``git worktree remove --force`` would throw away uncommitted work of another agent.
``kill``
    Never kill another agent's test or training process.
``busy``
    A parallel, coverage or whole-suite test run, a ``gracemaker`` training or another TensorFlow job while the machine
    is busy (another such process runs or the 1-minute load is above 4).

Override: when the user has told you to run exactly that command, put ``GRACE_HOOK_OVERRIDE=<rule>`` (or ``all``) in
front of it. The hook cannot tell whether the user said so: the override is a deliberate, visible step, not a lock.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

INTEGRATION_BRANCHES = ("torch-backend", "master")
OVERRIDE_VAR = "GRACE_HOOK_OVERRIDE"
DEFAULT_BUSY_LOAD = 4.0  # the 1-minute load average above which the machine is busy (14 cores, CLAUDE.md)
SHELLS = {"bash", "sh", "zsh", "dash"}
WRAPPERS = {"env", "command", "builtin", "time", "nice", "nohup", "exec", "sudo", "timeout"}
LAUNCHERS = {"uv", "uvx", "env", "time", "nice", "nohup", "timeout"}
HISTORY_REWRITES = {"commit", "rebase", "cherry-pick", "revert", "am", "reset"}
GUARDED_GIT = HISTORY_REWRITES | {"merge", "pull", "push"}
HARMLESS_FLAGS = {"--abort", "--dry-run", "--quit"}
GH_WRITE_VERBS = {"create", "edit", "comment", "close", "reopen", "review", "delete", "lock", "unlock", "transfer", "develop"}
GH_API_MERGE = re.compile(r"/pulls/\d+/merge|enablePullRequestAutoMerge|markPullRequestReadyForReview|mergePullRequest")
GRAPHQL_POINTS = {"item-list": 100, "field-list": 150}
SUITE_SCRIPTS = ("oracle_snapshot.py", "make_golden.py", "grace_probe.py")
PARALLEL_FLAG = re.compile(r"-n\d*|--numprocesses(=.*)?|--cov(=.*|-.*)?|--splits(=.*)?")
KILLERS = {"kill", "pkill", "killall"}
FAST_TARGETS = ("tests_torch", "tools/")  # TF-free test directories that run in seconds


@dataclass(frozen=True)
class Verdict:
    """A rule that refuses the command, and what the agent should do instead."""

    rule: str
    message: str


# ---------------------------------------------------------------------------------------------------------------- parsing


def _skip_heredoc_bodies(command: str) -> str:
    """The command without the bodies of its here-documents (their text is data, not commands); `<<<` is a here-string."""
    out: list[str] = []
    pending: list[tuple[str, bool]] = []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        out.append(lines[i])
        pending += [(m.group(2), m.group(1) == "-") for m in re.finditer(r"(?<!<)<<(?!<)(-?)\s*['\"]?([A-Za-z_]\w*)['\"]?", lines[i])]
        i += 1
        while pending and i < len(lines):
            delim, strip = pending[0]
            body = lines[i].strip() if strip else lines[i]
            i += 1
            if body == delim:
                pending.pop(0)
    return "\n".join(out)


def _newlines_to_semicolons(command: str) -> str:
    """Replace each newline outside quotes by `;`, so that a script of several lines splits into commands."""
    out: list[str] = []
    quote = ""
    escaped = False
    for ch in command:
        if escaped:
            escaped = False
        elif ch == "\\" and quote != "'":
            escaped = True
        elif quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "\n":
            ch = ";"
        out.append(ch)
    return "".join(out)


def tokenize(command: str) -> list[str]:
    """Shell words and operators of a command line; an unbalanced quote falls back to a plain split."""
    text = _newlines_to_semicolons(_skip_heredoc_bodies(command.replace("\\\n", "")))
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        return list(lex)
    except ValueError:
        return text.split()


def split_commands(command: str) -> list[list[str]]:
    """The simple commands of a command line, each a list of words."""
    commands: list[list[str]] = []
    current: list[str] = []
    for token in [*tokenize(command), ";"]:
        if re.fullmatch(r"[();&|]+", token):
            if current:
                commands.append(current)
            current = []
        else:
            current.append(token)
    return commands


def _strip_assignments(words: list[str]) -> tuple[list[str], dict[str, str]]:
    """The words after leading `VAR=value` assignments, and those assignments."""
    env: dict[str, str] = {}
    k = 0
    while k < len(words) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[k]):
        name, _, value = words[k].partition("=")
        env[name] = value
        k += 1
    return words[k:], env


def _unwrap(words: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """The command behind `env`, `timeout 5`, `sudo`, `nice -n 5`, ... (their own options and assignments dropped)."""
    while words and Path(words[0]).name in WRAPPERS:
        wrapper, words = Path(words[0]).name, words[1:]
        while words and words[0].startswith("-"):
            takes_value = wrapper in ("nice", "sudo", "timeout") and words[0] in ("-n", "-u", "-k", "-s")
            words = words[2:] if takes_value else words[1:]
        if wrapper == "timeout" and words:
            words = words[1:]  # the duration
        words, more = _strip_assignments(words)
        env = {**env, **more}
    return words, env


def _inner_script(words: list[str]) -> str | None:
    """The script of `bash -c "<script>"` (also `-lc`, sh, zsh) or `eval ...`, else None."""
    base = Path(words[0]).name
    if base == "eval":
        return " ".join(words[1:])
    if base in SHELLS:
        for k, w in enumerate(words[1:], 1):
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", w):
                return words[k + 1] if k + 1 < len(words) else None
    return None


def overrides(env: dict[str, str]) -> set[str]:
    """The rules a command (or the hook's own environment) overrides."""
    value = env.get(OVERRIDE_VAR, os.environ.get(OVERRIDE_VAR, ""))
    return {x.strip() for x in value.split(",") if x.strip()}


def _chdir(cwd: Path, target: str) -> Path:
    """The directory `target` names from `cwd` (`~` expanded)."""
    return (cwd / Path(target).expanduser()).resolve()


# ---------------------------------------------------------------------------------------------------------------- git state


def _git(cwd: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def current_branch(cwd: Path) -> str:
    """The branch checked out in `cwd` ("" when detached or not a repository)."""
    return _git(cwd, "symbolic-ref", "--short", "-q", "HEAD")


def in_linked_worktree(cwd: Path) -> bool:
    """Whether `cwd` is inside a linked worktree (its git directory differs from the common one)."""
    git_dir = _git(cwd, "rev-parse", "--path-format=absolute", "--git-dir")
    common = _git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return bool(git_dir) and bool(common) and git_dir != common


def _git_subcommand(words: list[str], cwd: Path) -> tuple[str, list[str], Path]:
    """(subcommand, its arguments, directory) of a `git ...` command, honouring `-C dir` and the other global options."""
    k = 1
    while k < len(words) and words[k].startswith("-"):
        if words[k] == "-C" and k + 1 < len(words):
            cwd = _chdir(cwd, words[k + 1])
            k += 1
        elif words[k] in ("-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"):
            k += 1
        k += 1
    return (words[k], words[k + 1 :], cwd) if k < len(words) else ("", [], cwd)


def _push_destinations(args: list[str], branch: str) -> tuple[str, list[str]]:
    """(remote, destination branches) of `git push` arguments; with no refspec the current branch is pushed."""
    positional = [a for a in args if not a.startswith("-")]
    remote = positional[0] if positional else ""
    dests = []
    for spec in positional[1:]:
        dest = spec.lstrip("+").split(":")[-1]
        dests.append(branch if dest == "HEAD" else dest.removeprefix("refs/heads/"))
    return remote, dests or [branch]


# ---------------------------------------------------------------------------------------------------------------- rules


def check_git(words: list[str], cwd: Path) -> Verdict | None:
    """The `integration-branch`, `upstream` and `worktree-force` rules for a `git` command."""
    sub, args, where = _git_subcommand(words, cwd)
    if sub == "worktree" and args[:1] == ["remove"] and any(a in ("--force", "-f") for a in args):
        return Verdict("worktree-force", "`git worktree remove --force` discards another agent's uncommitted work. Remove it without --force; if git refuses, report the uncommitted work.")
    if sub not in GUARDED_GIT:
        return None
    branch = current_branch(where)
    if sub == "push":
        remote, dests = _push_destinations(args, branch)
        if remote.lower() == "upstream" or "icams" in remote.lower():
            return Verdict("upstream", "Nothing is pushed to the upstream ICAMS repositories without an explicit go for that unit.")
        if any(a in ("--all", "--mirror") for a in args) or any(d in INTEGRATION_BRANCHES for d in dests):
            return Verdict("integration-branch", "Never push to torch-backend or master (also with --all or --mirror): push your work branch and open a draft PR; the user merges.")
    elif branch in INTEGRATION_BRANCHES and _rewrites_integration(sub, args):
        return Verdict("integration-branch", f"You are on {branch}: never commit, merge, rebase or reset there. Work in your own worktree on the branch of your issue; update with `git pull --ff-only`.")
    return None


def _rewrites_integration(sub: str, args: list[str]) -> bool:
    if HARMLESS_FLAGS & set(args):
        return False
    if sub in HISTORY_REWRITES:
        return sub != "reset" or "--hard" in args
    return "--ff-only" not in args  # merge and pull


def check_uv(words: list[str], cwd: Path) -> Verdict | None:
    """`uv sync` in a linked worktree (without `--no-install-project`)."""
    if "sync" in words[1:] and "--no-install-project" not in words and in_linked_worktree(cwd):
        return Verdict("worktree-sync", "`uv sync` in a worktree repoints the shared editable install to it. Link the main .venv instead (CLAUDE.md, Worktrees); repair with `uv sync --frozen --group dev` in the main checkout.")
    return None


def _gh_parts(words: list[str]) -> tuple[list[str], str]:
    """(positional words after `gh`, the value of --repo/-R)."""
    positional, repo = [], ""
    k = 1
    while k < len(words):
        w = words[k]
        if w in ("--repo", "-R") and k + 1 < len(words):
            repo, k = words[k + 1], k + 1
        elif w.startswith("--repo="):
            repo = w.split("=", 1)[1]
        elif not w.startswith("-"):
            positional.append(w)
        k += 1
    return positional, repo


def _api_writes(words: list[str]) -> bool:
    method = next((words[k + 1] for k, w in enumerate(words[:-1]) if w in ("-X", "--method")), "GET")
    fields = any(w in ("-f", "-F", "--field", "--raw-field", "--input") for w in words)
    return method.upper() != "GET" or fields


def check_gh(words: list[str], env: dict[str, str]) -> Verdict | None:
    """The `graphql`, `merge`, `draft`, `repo` and `upstream` rules for a `gh` command."""
    parts, repo = _gh_parts(words)
    repo = repo or env.get("GH_REPO", os.environ.get("GH_REPO", ""))
    group, verb = (parts + ["", ""])[:2]
    joined = " ".join(words)
    if group == "project" and verb in GRAPHQL_POINTS:
        return Verdict("graphql", f"`gh project {verb}` costs about {GRAPHQL_POINTS[verb]} GraphQL points of the 5,000 an hour that all agents share. Use `python tools/board.py` (it reads the table for about 3 points) and `board.py budget`.")
    if (group == "pr" and verb in ("merge", "ready")) or (group == "api" and GH_API_MERGE.search(joined)):
        return Verdict("merge", "An agent never merges a pull request, marks it ready for review or turns on auto-merge: the user does. Ask the user.")
    upstream = repo.lower().startswith("icams/") or "repos/icams/" in joined.lower()
    if upstream and (verb in GH_WRITE_VERBS or (group == "api" and _api_writes(words))):
        return Verdict("upstream", "Nothing is written on the upstream ICAMS repositories without an explicit go for that unit.")
    if group in ("pr", "issue") and verb in GH_WRITE_VERBS:
        if not repo:
            return Verdict("repo", f"Name the repository: add `--repo akwarii/grace-tensorpotential`; the default repository of gh here is the upstream one (`gh {group} {verb}`).")
        if group == "pr" and verb == "create" and not {"--draft", "-d"} & set(words):
            return Verdict("draft", "Open the pull request as a draft (`gh pr create --draft`); the user marks it ready and merges.")
    return None


def check_kill(words: list[str]) -> Verdict | None:
    """The `kill` rule: no `kill`, `pkill` or `killall` naming a test or training process."""
    if Path(words[0]).name in KILLERS and re.search(r"pytest|gracemaker", " ".join(words[1:])):
        return Verdict("kill", "Never kill another agent's test or training process (it may be another agent's run). Wait for it, or ask the user.")
    return None


def _pytest_is_suite(args: list[str]) -> bool:
    """Whether pytest arguments make a parallel or coverage run, or a whole-suite run (no test file or fast directory named)."""
    if any(PARALLEL_FLAG.fullmatch(w) for w in args):
        return True
    targets = [w for w in args if not w.startswith("-")]
    specific = any(".py" in t or "::" in t or t.startswith(FAST_TARGETS) for t in targets)
    return not specific


def is_suite_command(words: list[str]) -> bool:
    """Whether the command starts a parallel, coverage or whole-suite pytest run, a gracemaker training or a TensorFlow baseline job."""
    base = Path(words[0]).name
    if base in SHELLS:
        return False
    names = [Path(w).name for w in words]
    if base == "gracemaker" or ((base in LAUNCHERS or base.startswith("python")) and "gracemaker" in names[:4]):
        return not {"-h", "--help"} & set(words)
    if "write" in words and any(s in names for s in SUITE_SCRIPTS):
        return True
    for k, name in enumerate(names):
        if name in ("pytest", "py.test"):
            return _pytest_is_suite(words[k + 1 :])
    return False


def _ps_args(line: str) -> str:
    """The command-line part of a `ps -eo pid,etimes,args` line ("" for a malformed line)."""
    parts = line.split(None, 2)
    return parts[2] if len(parts) == 3 else ""


def is_suite_process(args: str) -> bool:
    """Whether a `ps` command line is a pytest run, a gracemaker training or a baseline job (not a shell that merely names them)."""
    try:
        words = shlex.split(args)
    except ValueError:
        words = args.split()
    if not words:
        return False
    base = Path(words[0]).name
    if not (base in ("pytest", "py.test", "gracemaker", "uv", "uvx") or base.startswith("python")):
        return False
    names = {Path(w).name for w in words[1:]} | {base}
    return bool(names & {"pytest", "gracemaker", *SUITE_SCRIPTS})


def busy_reason(load1: float, process_lines: list[str], threshold: float = DEFAULT_BUSY_LOAD) -> str | None:
    """Why the machine is busy (a suite or training process runs, or the load is above `threshold`), else None."""
    running = [ln.strip() for ln in process_lines if is_suite_process(_ps_args(ln))]
    if running:
        return f"another test or training process is running: {running[0][:120]}"
    if load1 > threshold:
        return f"the 1-minute load average is {load1:.1f} (limit {threshold:g})"
    return None


def machine_state() -> tuple[float, list[str]]:
    """(1-minute load average, `ps -eo pid,etimes,args` lines without this process and its parent)."""
    try:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        load1 = 0.0
    try:
        out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return load1, []
    mine = {str(os.getpid()), str(os.getppid())}
    return load1, [ln for ln in out.splitlines()[1:] if ln.split(None, 1)[:1] and ln.split(None, 1)[0] not in mine]


def check_busy(words: list[str]) -> Verdict | None:
    """The `busy` rule: no suite or training run while another one runs or the load is high."""
    if not is_suite_command(words):
        return None
    load1, procs = machine_state()
    reason = busy_reason(load1, procs, float(os.environ.get("GRACE_BUSY_LOAD", DEFAULT_BUSY_LOAD)))
    if reason is None:
        return None
    return Verdict("busy", f"The machine is busy: {reason}. Start no full, -n or coverage run now; do work that needs no CPU (edits, ruff, ty, one test file serially), look again later and tell the user if waiting blocks you (CLAUDE.md, idle machine).")


def _checks(base: str, words: list[str], env: dict[str, str], where: Path) -> list[Verdict | None]:
    """The verdicts of every rule that applies to one simple command."""
    by_tool: dict[str, Callable[[], Verdict | None]] = {
        "git": lambda: check_git(words, where),
        "gh": lambda: check_gh(words, env),
        "uv": lambda: check_uv(words, where),
    }
    return [by_tool.get(base, lambda: None)(), check_kill(words), check_busy(words)]


def evaluate(command: str, cwd: Path, inherited: frozenset[str] = frozenset()) -> list[Verdict]:
    """Every rule the command breaks, honouring `cd`, wrappers, inner scripts and `GRACE_HOOK_OVERRIDE`."""
    verdicts: list[Verdict] = []
    where = cwd
    for raw in split_commands(command):
        words, env = _unwrap(*_strip_assignments(raw))
        if not words:
            continue
        skip = inherited | overrides(env)
        base = Path(words[0]).name
        inner = _inner_script(words)
        if inner is not None:
            verdicts += evaluate(inner, where, frozenset(skip))
        elif base == "cd":
            where = _chdir(where, words[1]) if len(words) > 1 and words[1] != "-" else where
        elif "all" not in skip:
            verdicts += [v for v in _checks(base, words, env, where) if v is not None and v.rule not in skip]
    return verdicts


def message(verdicts: list[Verdict]) -> str:
    """The text shown to the agent when a command is blocked."""
    lines = [f"BLOCKED by tools/guard_bash.py ({v.rule}): {v.message}" for v in verdicts]
    rules = ",".join(sorted({v.rule for v in verdicts}))
    lines.append(f"These are rules of CLAUDE.md. Only if the user told you to run exactly this command, put {OVERRIDE_VAR}={rules} in front of it.")
    return "\n".join(lines)


def main() -> int:
    """Read the hook input from stdin; exit 2 with a message when the command breaks a rule."""
    try:
        data = json.load(sys.stdin)
        if data.get("tool_name") != "Bash":
            return 0
        command = str(data.get("tool_input", {}).get("command", ""))
        cwd = Path(data.get("cwd") or os.getcwd())
        verdicts = evaluate(command, cwd)
    except Exception as exc:  # noqa: BLE001 - a broken guard must never block the agent: fail open and say so
        print(f"guard_bash: ignored an internal error: {exc!r}", file=sys.stderr)
        return 0
    if verdicts:
        print(message(verdicts), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
