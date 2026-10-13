"""Return annotations and docstrings of the public functions, per file and for the public surface (QUAL6).

A *public function* is a function or method reachable through public names only: no leading underscore, at module
level or in a public class, not nested. ``quality_report.py`` counts them over the whole library; this tool counts
them over the **public surface** (``SCOPE`` below: what the docs and the tests import) and fails when a share is
under the bar. A function counts as complete when it has a return annotation and a docstring.

Usage::

    python tools/api_annotations.py table [--root DIR]            counts per file of the scope, then in total
    python tools/api_annotations.py missing [--root DIR]          each function of the scope that lacks one of the two
    python tools/api_annotations.py check [--root DIR] [--min S]  exit 1 when a share is under S (default 0.9)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

SCOPE: tuple[str, ...] = (
    "tensorpotential/tensorpot.py",
    "tensorpotential/tpmodel.py",
    "tensorpotential/calculator/asecalculator.py",
    "tensorpotential/calculator/foundation_models.py",
    "tensorpotential/instructions/base.py",
    "tensorpotential/data/databuilder.py",
    "tensorpotential/data/tpatoms.py",
    "tensorpotential/loss.py",
)
MIN_SHARE = 0.9

Function = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Counts:
    """Public functions of one file (or the sum over files) with a return annotation and with a docstring."""

    total: int = 0
    returns: int = 0
    docstrings: int = 0
    both: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.total + other.total,
            self.returns + other.returns,
            self.docstrings + other.docstrings,
            self.both + other.both,
        )

    def share(self, have: int) -> float:
        """Return ``have / total``; an empty set counts as complete."""
        return 1.0 if self.total == 0 else have / self.total


def public_functions(tree: ast.Module | ast.ClassDef) -> list[Function]:
    """Return the functions and methods reachable through public names only (nested functions are not API)."""
    found: list[Function] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            found += public_functions(node)
        elif isinstance(node, Function) and not node.name.startswith("_"):
            found.append(node)
    return found


def has_return(func: Function) -> bool:
    """Whether the function declares its return type."""
    return func.returns is not None


def has_docstring(func: Function) -> bool:
    """Whether the function has a docstring."""
    return ast.get_docstring(func) is not None


def count(funcs: list[Function]) -> Counts:
    """Count annotations and docstrings over ``funcs``."""
    return Counts(
        len(funcs),
        sum(map(has_return, funcs)),
        sum(map(has_docstring, funcs)),
        sum(has_return(f) and has_docstring(f) for f in funcs),
    )


def parse_public(path: Path) -> list[Function]:
    """Parse ``path`` and return its public functions."""
    return public_functions(ast.parse(path.read_text(encoding="utf-8")))


def measure(root: Path, scope: tuple[str, ...] = SCOPE) -> dict[str, Counts]:
    """Count the public functions of each file of ``scope`` (paths relative to ``root``)."""
    missing = [name for name in scope if not (root / name).is_file()]
    if missing:
        msg = f"files of the scope not found under {root}: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    return {name: count(parse_public(root / name)) for name in scope}


def incomplete(root: Path, scope: tuple[str, ...] = SCOPE) -> list[str]:
    """List ``file:line name (what is missing)`` for each public function without return annotation or docstring."""
    rows = []
    for name in scope:
        for func in parse_public(root / name):
            lacks = [
                label
                for label, ok in (
                    ("return annotation", has_return(func)),
                    ("docstring", has_docstring(func)),
                )
                if not ok
            ]
            if lacks:
                rows.append(f"{name}:{func.lineno} {func.name} ({', '.join(lacks)})")
    return rows


def render(per_file: dict[str, Counts]) -> str:
    """Return the table of counts, one line per file and a total."""
    total = sum(per_file.values(), Counts())
    lines = [f"{'file':52} {'functions':>9} {'returns':>8} {'docstrings':>10}"]
    for name, c in [*per_file.items(), ("total", total)]:
        lines.append(
            f"{name:52} {c.total:9d} {c.returns:5d} {c.share(c.returns):.0%} "
            f"{c.docstrings:5d} {c.share(c.docstrings):.0%}"
        )
    return "\n".join(lines)


def shortfalls(per_file: dict[str, Counts], minimum: float) -> list[str]:
    """Name each share of the whole scope that is under ``minimum``."""
    total = sum(per_file.values(), Counts())
    return [
        f"{label}: {total.share(have):.1%} ({have}/{total.total}) is under {minimum:.0%}"
        for label, have in (
            ("return annotations", total.returns),
            ("docstrings", total.docstrings),
        )
        if total.share(have) < minimum
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the exit code (1 for a share under the bar)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["table", "missing", "check"])
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--min", type=float, default=MIN_SHARE, dest="minimum")
    args = parser.parse_args(argv)
    per_file = measure(args.root, SCOPE)
    if args.command == "missing":
        print("\n".join(incomplete(args.root, SCOPE)))
        return 0
    print(render(per_file))
    problems = shortfalls(per_file, args.minimum) if args.command == "check" else []
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
