"""Fail when a function, method or class body touched by a diff is covered under 90%.

The owner's rule (2026-09-30): code that is modified must already be well tested. The tool
finds the *units* (a ``def``, a method, a class body) whose code differs between a base
revision and the working tree, reads a coverage.py JSON report made with branch coverage,
and prints one line per unit::

    ok    94.4%  tensorpotential/export.py::extract_const_shift_scale  (17/18, L32-L55)

Usage::

    python -m pytest tests --cov=tensorpotential --cov-branch --cov-report=json:cov.json
    python tools/check_touched_coverage.py --coverage cov.json [--base REV] [--threshold 90]
    python tools/check_touched_coverage.py --coverage cov.json --unit path.py::Class.method

Coverage of a unit is ``(executed statements + executed branches) / (statements + branches)``
over the lines the unit owns. A nested ``def`` or ``class`` is its own unit: its body is
not counted for the enclosing one, its header (decorators and ``def`` line) is. Module-level
statements are not units and are never gated.

A unit counts as touched when it is new, or when the AST of its own code differs from the
base, ignoring comments, docstrings and annotations (the normalised mode of
``ast_manifest.py``), so comment-only, docstring-only and annotation-only edits are exempt.
Only ``.py`` files outside the fork-only prefixes and outside ``tests/`` are examined.
``--unit`` checks the named units whatever the diff says (used to gate a fixed list).

Exit status 1 when a unit is under the threshold or a named unit does not exist.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ast_manifest import _git, hash_source
from check_pr_branch import FORBIDDEN_PREFIXES

DEFAULT_THRESHOLD = 90.0
DEFAULT_BASE = "origin/torch-backend"
# Tests are not units to gate (the rule exempts test code); tools/ etc. are fork-only.
EXCLUDED_PREFIXES = (*FORBIDDEN_PREFIXES, "tests/")

_DefNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Unit:
    """A def, method or class body: its qualified name, own lines and AST fingerprint."""

    qualname: str
    first: int
    last: int
    lines: frozenset[int]
    statements: int
    fingerprint: str


@dataclass(frozen=True)
class UnitCoverage:
    """Coverage of one unit: ``hit`` of ``total`` statements plus branches."""

    path: str
    unit: Unit
    hit: int
    total: int
    excluded: int = 0

    @property
    def percent(self) -> float:
        """Percentage covered; a unit without statements or branches counts as 100."""
        return 100.0 if self.total == 0 else 100.0 * self.hit / self.total


def _children(node: ast.AST) -> list[_DefNode]:
    """The defs nested directly in ``node`` (inside ifs, trys and loops included)."""
    found: list[_DefNode] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, _DEF_TYPES):
            found.append(child)
        else:
            stack.extend(ast.iter_child_nodes(child))
    return sorted(found, key=lambda n: n.lineno)


def _header_start(node: _DefNode) -> int:
    decorators = [d.lineno for d in node.decorator_list]
    return min([node.lineno, *decorators])


def _body_start(node: _DefNode) -> int:
    """First line after the ``def``/``class`` header (``node.body[0]`` may be a docstring)."""
    return node.body[0].lineno


def _own_lines(node: _DefNode, children: list[_DefNode]) -> frozenset[int]:
    lines = set(range(_header_start(node), (node.end_lineno or node.lineno) + 1))
    for child in children:
        lines -= set(range(_body_start(child), (child.end_lineno or child.lineno) + 1))
    return frozenset(lines)


def _fingerprint(node: _DefNode) -> str:
    """Hash of the node with the bodies of nested defs removed, normalised."""
    clone = copy.deepcopy(node)
    for child in _children(clone):
        child.body = [ast.Pass()]
    return hash_source(ast.unparse(clone), normalised=True)


def units_of(source: str) -> dict[str, Unit]:
    """Map qualified name to :class:`Unit` for every def and class of ``source``.

    A repeated name (property setters, conditional redefinitions) gets ``#2``, ``#3``
    in source order. Raises ``SyntaxError`` when ``source`` does not parse.
    """
    tree = ast.parse(source)
    stmt_lines = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt)}
    units: dict[str, Unit] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in _children(node):
            qualname = prefix + child.name
            key, n = qualname, 1
            while key in units:
                n += 1
                key = f"{qualname}#{n}"
            own = _own_lines(child, _children(child))
            units[key] = Unit(
                key,
                _header_start(child),
                child.end_lineno or child.lineno,
                own,
                len(own & stmt_lines),
                _fingerprint(child),
            )
            visit(child, qualname + ".")

    visit(tree, "")
    return units


def touched_units(old: dict[str, Unit] | None, new: dict[str, Unit]) -> list[Unit]:
    """Units of ``new`` that are absent from ``old`` or whose fingerprint differs.

    ``old=None`` (a file that is new, or ``--unit`` mode) makes every unit touched.
    """
    if old is None:
        return list(new.values())
    return [
        u
        for name, u in new.items()
        if name not in old or old[name].fingerprint != u.fingerprint
    ]


def _branch_from_lines(pairs: list[list[int]]) -> list[int]:
    return [p[0] for p in pairs]


def unit_coverage(path: str, unit: Unit, report: dict | None) -> UnitCoverage:
    """Coverage of ``unit`` from one file entry of a coverage.py JSON report.

    ``report`` is ``None`` for a file coverage never saw (not imported by any test):
    nothing ran, so the unit scores 0 over its statements (at least 1).
    """
    if report is None:
        return UnitCoverage(path, unit, 0, max(unit.statements, 1))
    executed = set(report.get("executed_lines", ()))
    missing = set(report.get("missing_lines", ()))
    excluded = set(report.get("excluded_lines", ()))
    done_br = _branch_from_lines(report.get("executed_branches", ()))
    todo_br = _branch_from_lines(report.get("missing_branches", ()))
    own = unit.lines
    hit = len(executed & own) + sum(1 for b in done_br if b in own)
    miss = len(missing & own) + sum(1 for b in todo_br if b in own)
    return UnitCoverage(path, unit, hit, hit + miss, len(excluded & own))


def format_line(result: UnitCoverage, threshold: float) -> str:
    """One output line for ``result``."""
    status = "ok  " if result.percent >= threshold else "FAIL"
    extra = f", {result.excluded} excluded" if result.excluded else ""
    return (
        f"{status} {result.percent:5.1f}%  {result.path}::{result.unit.qualname}  "
        f"({result.hit}/{result.total}, L{result.unit.first}-L{result.unit.last}{extra})"
    )


def _diff_files(root: Path, base: str) -> list[str]:
    """Python files that differ between ``base`` and the working tree (added or modified)."""
    out = _git(
        ["diff", "--name-only", "--no-renames", "--diff-filter=AM", base, "--", "*.py"],
        root,
    )
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", "*.py"], root)
    names = {*out.splitlines(), *untracked.splitlines()}
    return sorted(n for n in names if not n.startswith(EXCLUDED_PREFIXES))


def _base_source(root: Path, base: str, path: str) -> str | None:
    """Source of ``path`` at ``base``, or None when the file does not exist there."""
    try:
        return _git(["show", f"{base}:{path}"], root)
    except subprocess.CalledProcessError:
        return None


def _lookup(files: dict, root: Path, path: str) -> dict | None:
    """The coverage entry of ``path`` (relative or absolute keys), or None."""
    for key in (path, str(root / path)):
        if key in files:
            return files[key]
    return None


def collect(
    root: Path, base: str, report: dict, units: list[str] | None = None
) -> tuple[list[UnitCoverage], list[str]]:
    """Coverage of every touched unit (or of the named ``units``) and the problems found.

    ``units`` entries look like ``tensorpotential/export.py::extract_const_shift_scale``.
    """
    files = report.get("files", {})
    results: list[UnitCoverage] = []
    problems: list[str] = []
    wanted: dict[str, list[str]] = {}
    for spec in units or []:
        path, _, qualname = spec.partition("::")
        wanted.setdefault(path, []).append(qualname)
    paths = sorted(wanted) if units else _diff_files(root, base)
    for path in paths:
        try:
            new = units_of((root / path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as err:
            problems.append(f"{path}: cannot parse ({err})")
            continue
        if units:
            chosen = [new[q] for q in wanted[path] if q in new]
            problems += [
                f"{path}::{q}: no such unit" for q in wanted[path] if q not in new
            ]
        else:
            old_source = _base_source(root, base, path)
            old = units_of(old_source) if old_source is not None else None
            chosen = touched_units(old, new)
        cover = _lookup(files, root, path)
        results += [unit_coverage(path, u, cover) for u in chosen]
    return results, problems


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--coverage", type=Path, required=True, help="coverage.py JSON report"
    )
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="revision the diff starts from"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--unit",
        action="append",
        default=[],
        metavar="PATH::QUALNAME",
        help="check this unit whatever the diff says (repeatable)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print one line per unit; return 1 if any is under the threshold or missing."""
    args = _parse_args(argv)
    report = json.loads(args.coverage.read_text())
    results, problems = collect(args.root, args.base, report, args.unit)
    for result in results:
        sys.stdout.write(format_line(result, args.threshold) + "\n")
    for problem in problems:
        sys.stdout.write(f"ERROR {problem}\n")
    failing = [r for r in results if r.percent < args.threshold]
    sys.stdout.write(
        f"{len(results)} unit(s) checked, {len(failing)} under {args.threshold:g}%, "
        f"{len(problems)} problem(s)\n"
    )
    return 1 if failing or problems else 0


if __name__ == "__main__":
    sys.exit(main())
