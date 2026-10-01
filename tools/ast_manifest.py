"""Fingerprint every tracked ``.py`` file by the hash of its AST.

Comments are not part of the AST, so a comment-only commit leaves the manifest
unchanged. The ``--normalised`` mode also ignores docstrings and annotations, which is
what the coverage gate (``check_touched_coverage.py``) needs to exempt edits that
cannot change behaviour.

Usage::

    python tools/ast_manifest.py write baselines/ast_manifest.json [--rev REV]
    python tools/ast_manifest.py check baselines/ast_manifest.json \
        [--rev REV] [--normalised] [--exclude PREFIX ...]

Fork-only paths (``tools/``, ``baselines/``, ``tests_torch/``, ``tensorpotential/core/``
and the other prefixes of ``check_pr_branch.py``) are always ignored, so the manifest of
the untouched tree stays comparable as the fork adds files; ``--exclude`` adds prefixes.

``check`` exits with status 1 when a file was added, removed or changed. Without
``--rev`` the files are read from the working tree; with ``--rev`` they are read from
that git revision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from check_pr_branch import FORBIDDEN_PREFIXES

# Fork-only paths are never part of the comparison: a baseline of the untouched
# tree must not flag the files the fork adds. One list, shared with check_pr_branch.
DEFAULT_EXCLUDES = FORBIDDEN_PREFIXES

SYNTAX_ERROR = "SYNTAX_ERROR"


_Scope = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class _Normaliser(ast.NodeTransformer):
    """Drop docstrings and annotations, which never change behaviour."""

    def _strip_docstring(self, node: _Scope) -> _Scope:
        body = node.body
        first = body[0] if body else None
        is_doc = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_doc:
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self.generic_visit(self._strip_docstring(node))

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self.generic_visit(self._strip_docstring(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        return self.generic_visit(self._strip_docstring(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.returns = None
        return self.generic_visit(self._strip_docstring(node))

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        # Keep the statement: a dataclass or NamedTuple field exists through it.
        node.annotation = ast.Constant(value=None)
        return self.generic_visit(node)


def hash_source(source: str, normalised: bool = False) -> str:
    """Return the sha256 of the AST of ``source``, or ``SYNTAX_ERROR`` if unparsable."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return SYNTAX_ERROR
    if normalised:
        tree = ast.fix_missing_locations(_Normaliser().visit(tree))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    # git is resolved from PATH on purpose; the arguments are built here, not user input
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def tracked_python_files(
    root: Path, rev: str | None = None, exclude: tuple[str, ...] = DEFAULT_EXCLUDES
) -> list[str]:
    """Sorted tracked ``.py`` paths, in the working tree or at ``rev``.

    Paths starting with one of the ``exclude`` prefixes are left out.
    """
    if rev is None:
        listing = _git(["ls-files", "*.py"], root)
    else:
        listing = _git(["ls-tree", "-r", "--name-only", rev], root)
    return sorted(
        p
        for p in listing.splitlines()
        if p.endswith(".py") and not p.startswith(exclude)
    )


def build_manifest(
    root: Path,
    rev: str | None = None,
    normalised: bool = False,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> dict[str, str]:
    """Map every tracked ``.py`` path (minus ``exclude`` prefixes) to its AST hash."""
    manifest = {}
    for path in tracked_python_files(root, rev, exclude):
        if rev is None:
            source = (root / path).read_text(encoding="utf-8")
        else:
            source = _git(["show", f"{rev}:{path}"], root)
        manifest[path] = hash_source(source, normalised)
    return manifest


def diff_manifests(
    expected: dict[str, str], actual: dict[str, str]
) -> dict[str, list[str]]:
    """Files added, removed and changed between two manifests."""
    return {
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
        "changed": sorted(
            p for p in set(expected) & set(actual) if expected[p] != actual[p]
        ),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["write", "check"])
    parser.add_argument("manifest", type=Path, help="JSON manifest to write or read")
    parser.add_argument("--rev", help="git revision to read the files from")
    parser.add_argument(
        "--normalised",
        action="store_true",
        help="ignore docstrings and annotations as well as comments",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PREFIX",
        help="also ignore paths starting with PREFIX (repeatable); the fork-only "
        "prefixes of check_pr_branch.py are always ignored",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ``write`` or ``check`` command; return the process exit status."""
    args = _parse_args(argv)
    exclude = (*DEFAULT_EXCLUDES, *args.exclude)
    actual = build_manifest(args.root, args.rev, args.normalised, exclude)
    if args.command == "write":
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        broken = [p for p, h in actual.items() if h == SYNTAX_ERROR]
        sys.stdout.write(
            f"{len(actual)} files written to {args.manifest}; syntax errors: {broken}\n"
        )
        return 0
    expected = {
        path: digest
        for path, digest in json.loads(args.manifest.read_text()).items()
        if not path.startswith(exclude)
    }
    diff = diff_manifests(expected, actual)
    for kind, paths in diff.items():
        for path in paths:
            sys.stdout.write(f"{kind}: {path}\n")
    sys.stdout.write(
        f"{len(actual)} files checked against {len(expected)} in {args.manifest}\n"
    )
    return 1 if any(diff.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
