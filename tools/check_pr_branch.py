"""Fail if a branch meant for an upstream pull request contains fork-only files.

usage: python tools/check_pr_branch.py [BASE] [HEAD]      (defaults: upstream/master and HEAD)

Upstream-eligible units are cut from upstream/master as `pr/U<n>-<slug>`, so the fork-only tooling never enters them by construction; this check
makes the rule explicit and catches a branch that was cut from `torch-backend` by mistake or had a fork-only file added.
Exit status 1 and the offending paths when a forbidden path appears in `git diff --name-only BASE...HEAD`.
"""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN_PREFIXES = (
    "CLAUDE.md",
    ".claude/",
    ".github/PULL_REQUEST_TEMPLATE/",
    ".github/ISSUE_TEMPLATE/",
    "tools/",
    "baselines/",
    "plan/",
    "TORCH_BACKEND_PLAN.md",
    "uv.lock",
    "tests_torch/",
    "tensorpotential/torch_backend/",
    "tensorpotential/core/",
)


def changed_files(base: str, head: str) -> list[str]:
    r = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f"git diff failed: {r.stderr.strip()}")
    return [line for line in r.stdout.splitlines() if line]


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "upstream/master"
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    files = changed_files(base, head)
    bad = [f for f in files if f.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        print(
            f"{len(bad)} fork-only path(s) on a branch meant for upstream ({base}...{head}):"
        )
        for f in bad:
            print("  ", f)
        sys.exit(1)
    print(f"ok: {len(files)} changed file(s), none fork-only")


if __name__ == "__main__":
    main()
