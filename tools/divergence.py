"""Divergence ledger of the fork (BOARD2, D17 b): every file of upstream the fork modifies or deletes has a row.

usage: python tools/divergence.py check [--base upstream/master] [--head HEAD] [--ledger tools/divergence.yaml] [--pr-branches]

The file list is recomputed from ``git diff --name-only --no-renames --diff-filter=MD BASE...HEAD`` (a rename counts as a deletion and an
addition). ``check`` exits 1 and names each problem when

* a modified or deleted upstream file has no row (an undeclared divergence),
* a row names a file that no longer differs from upstream (a stale row; the fork took the upstream version, or the file never differed),
* a row is malformed (missing key, unknown ``upstream`` value, empty reason, duplicate path).

With ``--pr-branches`` it also runs the fork-only path check of ``tools/check_pr_branch.py`` on every local ``pr/U*`` branch, so that
``CLAUDE.md``, ``.claude/``, ``tools/`` and ``baselines/`` never reach an upstream branch.
Fork-only additions (new files) need no row: the ledger is about the upstream files that diverge, which is where an upstream merge conflicts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import check_pr_branch

LEDGER = Path(__file__).resolve().with_name("divergence.yaml")
UPSTREAM_STATES = ("pending", "merged", "never")
REQUIRED_KEYS = ("path", "reason", "milestone", "upstream")


def git(*args: str) -> str:
    """Run ``git`` and return its standard output; exit with the message of git when it fails."""
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def modified_upstream_files(base: str, head: str) -> list[str]:
    """Files present in ``base`` that ``head`` changes or deletes, sorted."""
    out = git(
        "diff", "--name-only", "--no-renames", "--diff-filter=MD", f"{base}...{head}"
    )
    return sorted(line for line in out.splitlines() if line)


def load_ledger(path: Path) -> list[dict]:
    """Read the rows of the ledger; the file is a YAML list of mappings."""
    rows = yaml.safe_load(path.read_text())
    if not isinstance(rows, list):
        sys.exit(f"{path}: the ledger must be a YAML list of rows")
    return rows


def row_problems(rows: list[dict]) -> list[str]:
    """Malformed rows: missing keys, empty reason or milestone, unknown ``upstream`` value, duplicate path."""
    problems = []
    seen: set[str] = set()
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            problems.append(f"row {i}: not a mapping")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            problems.append(
                f"row {i} ({row.get('path', '?')}): missing {', '.join(missing)}"
            )
            continue
        path = row["path"]
        if path in seen:
            problems.append(f"{path}: two rows")
        seen.add(path)
        if not str(row["reason"]).strip():
            problems.append(f"{path}: empty reason")
        if not row["milestone"]:
            problems.append(f"{path}: empty milestone")
        if row["upstream"] not in UPSTREAM_STATES:
            problems.append(
                f"{path}: upstream is {row['upstream']!r}, expected one of {', '.join(UPSTREAM_STATES)}"
            )
    return problems


def ledger_problems(rows: list[dict], modified: list[str]) -> list[str]:
    """All problems of ``rows`` against the recomputed list ``modified``, empty when the ledger is exact."""
    problems = row_problems(rows)
    declared = {r["path"] for r in rows if isinstance(r, dict) and "path" in r}
    problems += [
        f"{p}: modified from upstream, no row in the ledger"
        for p in sorted(set(modified) - declared)
    ]
    problems += [
        f"{p}: row is stale, the file no longer differs from upstream"
        for p in sorted(declared - set(modified))
    ]
    return problems


def pr_branches() -> list[str]:
    """Local branches named ``pr/U*``."""
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads/pr/")
    return [b for b in out.splitlines() if b.startswith("pr/U")]


def pr_branch_problems(base: str, branches: list[str]) -> list[str]:
    """Fork-only paths (``check_pr_branch.FORBIDDEN_PREFIXES``) on each of ``branches``."""
    problems = []
    for branch in branches:
        problems += [
            f"{branch}: fork-only path {f}"
            for f in check_pr_branch.changed_files(base, branch)
            if f.startswith(check_pr_branch.FORBIDDEN_PREFIXES)
        ]
    return problems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "check", help="compare the ledger with the diff against upstream"
    )
    c.add_argument("--base", default="upstream/master")
    c.add_argument("--head", default="HEAD")
    c.add_argument("--ledger", type=Path, default=LEDGER)
    c.add_argument(
        "--pr-branches",
        action="store_true",
        help="also check every local pr/U* branch for fork-only paths",
    )
    args = ap.parse_args(argv)
    modified = modified_upstream_files(args.base, args.head)
    problems = ledger_problems(load_ledger(args.ledger), modified)
    branches = pr_branches() if args.pr_branches else []
    if args.pr_branches:
        problems += pr_branch_problems(args.base, branches)
    for p in problems:
        print(p)
    if problems:
        sys.exit(f"{len(problems)} problem(s) in the divergence ledger")
    print(
        f"ok: {len(modified)} modified upstream file(s), each with one row; {len(branches)} pr/U* branch(es) checked"
    )


if __name__ == "__main__":
    main()
