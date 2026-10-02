"""No-increase ratchet on duplicated functions (QUAL2).

A *clone group* is a set of two or more functions of ``MIN_LINES`` lines or more whose bodies (the
docstring and the signature are ignored) have the same AST. A body that is only ``pass`` or ``...``
is a stub, not duplicated logic, and is never counted (a long signature would otherwise make two
abstract methods a clone). The group is **identical** when the
bodies are equal, and **renamed** when they become equal once every identifier is abstracted
(variable, attribute, argument, keyword and nested function names; in the test tree also the constant
values, which is what ``pytest.mark.parametrize`` abstracts). ``compat/pace`` is excluded from the
library tree. The check is function-level: a clone inside a large function is not seen.

Per tree and kind the tool counts groups, functions and redundant lines (the lines of the members
beyond the largest one). The totals of a tree (both kinds) may not rise above
``baselines/clone_baseline.json``; the kinds are kept in it for the report.

Usage::

    python tools/check_clones.py check [--baseline FILE] [--fail-on-drop]
    python tools/check_clones.py record [--baseline FILE] [--allow-rise]
    python tools/check_clones.py list [--tree library|tests]

``check`` exits 1 on a rise (or on a drop with ``--fail-on-drop``) and 2 when a source file cannot
be parsed: a file that cannot be read is an error, never zero clones.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MIN_LINES = 8
BASELINE = Path("baselines/clone_baseline.json")
KINDS = ("identical", "renamed")
METRICS = ("groups", "functions", "redundant_lines")

Counts = dict[str, dict[str, dict[str, int]]]
_Named = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class ToolError(RuntimeError):
    """A source file cannot be parsed or the baseline cannot be used."""


@dataclass(frozen=True)
class Tree:
    """A source tree to scan and how its renamed clones are defined."""

    name: str
    path: str
    exclude: tuple[str, ...] = ()
    abstract_constants: bool = False


@dataclass(frozen=True)
class Func:
    """A function long enough to count, with the keys that decide its group."""

    where: str
    lines: int
    exact: str
    shape: str


@dataclass(frozen=True)
class Group:
    """Functions with the same body (``identical``) or the same shape (``renamed``)."""

    kind: str
    members: tuple[Func, ...]

    @property
    def redundant_lines(self) -> int:
        """Lines of all members except the largest one."""
        sizes = [f.lines for f in self.members]
        return sum(sizes) - max(sizes)


TREES = (
    Tree("library", "tensorpotential", exclude=("compat/pace",)),
    Tree("tests", "tests", abstract_constants=True),
)


class _Abstract(ast.NodeTransformer):
    """Replace identifiers (and optionally constants) by a placeholder."""

    def __init__(self, *, constants: bool) -> None:
        self.constants = constants

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.Name(id="_", ctx=node.ctx)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        node.attr = "_"
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        self.generic_visit(node)
        node.arg = "_"
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        self.generic_visit(node)
        node.arg = None if node.arg is None else "_"
        return node

    def _definition(self, node: _Named) -> _Named:
        self.generic_visit(node)
        node.name = "_"
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _definition

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.Constant(value=type(node.value).__name__) if self.constants else node


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    first = body[0]
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if is_doc else body


def _is_stub(body: list[ast.stmt]) -> bool:
    """Whether ``body`` is only ``pass`` or ``...``."""
    return all(
        isinstance(s, ast.Pass)
        or (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and s.value.value is Ellipsis
        )
        for s in body
    )


def _dump(body: list[ast.stmt], abstract: _Abstract | None = None) -> str:
    return "\n".join(
        ast.dump(abstract.visit(copy.deepcopy(s)) if abstract else s) for s in body
    )


class _Collector(ast.NodeVisitor):
    """Collect the functions of a module with their qualified names."""

    def __init__(self, rel: str, *, constants: bool, min_lines: int) -> None:
        self.rel = rel
        self.abstract = _Abstract(constants=constants)
        self.min_lines = min_lines
        self.scope: list[str] = []
        self.found: list[Func] = []

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        body = _without_docstring(node.body)
        if lines >= self.min_lines and body and not _is_stub(body):
            where = f"{self.rel}:{node.lineno}:{'.'.join([*self.scope, node.name])}"
            self.found.append(
                Func(where, lines, _dump(body), _dump(body, self.abstract))
            )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _sources(root: Path, tree: Tree) -> list[Path]:
    base = root / tree.path
    if not base.is_dir():
        msg = f"{tree.name} tree {tree.path!r} does not exist below {root}"
        raise ToolError(msg)
    return sorted(
        p
        for p in base.rglob("*.py")
        if not any(ex in p.relative_to(root).as_posix() for ex in tree.exclude)
    )


def functions(root: Path, tree: Tree, min_lines: int = MIN_LINES) -> list[Func]:
    """Every function of ``tree`` with at least ``min_lines`` lines."""
    found: list[Func] = []
    for path in _sources(root, tree):
        rel = path.relative_to(root).as_posix()
        try:
            module = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            msg = f"cannot parse {rel}: {exc}"
            raise ToolError(msg) from exc
        collector = _Collector(
            rel, constants=tree.abstract_constants, min_lines=min_lines
        )
        collector.visit(module)
        found += collector.found
    return found


def groups(funcs: list[Func]) -> list[Group]:
    """Clone groups of ``funcs``, sorted; a group is ``identical`` or ``renamed``, never both."""
    by_shape: dict[str, list[Func]] = defaultdict(list)
    for func in funcs:
        by_shape[func.shape].append(func)
    found = []
    for members in by_shape.values():
        if len(members) < 2:
            continue
        kind = "identical" if len({f.exact for f in members}) == 1 else "renamed"
        found.append(Group(kind, tuple(sorted(members, key=lambda f: f.where))))
    return sorted(found, key=lambda g: g.members[0].where)


def count(found: list[Group]) -> dict[str, dict[str, int]]:
    """``{kind: {groups, functions, redundant_lines}}``."""
    out = {}
    for kind in KINDS:
        sel = [g for g in found if g.kind == kind]
        out[kind] = {
            "groups": len(sel),
            "functions": sum(len(g.members) for g in sel),
            "redundant_lines": sum(g.redundant_lines for g in sel),
        }
    return out


def measure(root: Path, trees: tuple[Tree, ...] = TREES) -> Counts:
    """Measure the counts of every tree below ``root``."""
    return {t.name: count(groups(functions(root, t))) for t in trees}


def _totals(counts: Counts, tree: str) -> dict[str, int]:
    """Metrics of ``tree`` summed over the kinds: a clone group that is renamed stays a clone group."""
    kinds = counts.get(tree, {})
    return {m: sum(kinds.get(k, {}).get(m, 0) for k in KINDS) for m in METRICS}


def compare(base: Counts, new: Counts) -> tuple[list[str], list[str]]:
    """Lines for the numbers that rose and for those that dropped, ``tree metric: old -> new``.

    The kinds are summed, so turning an identical group into a renamed one is neither.
    """
    rises, drops = [], []
    for tree in sorted(base.keys() | new.keys()):
        before, after = _totals(base, tree), _totals(new, tree)
        for metric in METRICS:
            line = f"{tree} {metric}: {before[metric]} -> {after[metric]}"
            if after[metric] > before[metric]:
                rises.append(line)
            elif after[metric] < before[metric]:
                drops.append(line)
    return rises, drops


def _load(path: Path) -> dict:
    if not path.is_file():
        msg = f"no baseline at {path}; record one with `python tools/check_clones.py record`"
        raise ToolError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("min_lines") != MIN_LINES:
        msg = f"{path} was recorded with min_lines={data.get('min_lines')}, the tool uses {MIN_LINES}"
        raise ToolError(msg)
    return data


def check(root: Path, baseline: Path, *, fail_on_drop: bool = False) -> int:
    """Print the numbers that broke the ratchet and return the exit status."""
    rises, drops = compare(_load(baseline)["counts"], measure(root))
    report = [f"rose: {r}" for r in rises]
    if drops:
        report += [f"dropped: {d}" for d in drops]
        report.append(
            "record the lower baseline with: python tools/check_clones.py record"
        )
    sys.stdout.write("\n".join(report) + ("\n" if report else "clone ratchet: ok\n"))
    return 1 if rises or (fail_on_drop and drops) else 0


def record(root: Path, baseline: Path, *, allow_rise: bool = False) -> int:
    """Write the current counts as the baseline; refuse to record a rise unless allowed."""
    now = measure(root)
    if baseline.is_file() and not allow_rise:
        rises = compare(_load(baseline)["counts"], now)[0]
        if rises:
            sys.stdout.write(
                "refusing to record a rise (use --allow-rise):\n"
                + "\n".join(rises)
                + "\n"
            )
            return 1
    data = {"min_lines": MIN_LINES, "counts": now}
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = sum(m["groups"] for t in now.values() for m in t.values())
    sys.stdout.write(f"recorded {total} clone groups in {baseline}\n")
    return 0


def listing(root: Path, only: str | None = None) -> int:
    """Print every group with its members and line counts."""
    for tree in TREES:
        if only not in (None, tree.name):
            continue
        for group in groups(functions(root, tree)):
            members = ", ".join(f"{f.where} ({f.lines})" for f in group.members)
            sys.stdout.write(f"{tree.name} {group.kind}: {members}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "record", "list"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=Path())
        if name != "list":
            cmd.add_argument("--baseline", type=Path, default=BASELINE)
    sub.choices["check"].add_argument("--fail-on-drop", action="store_true")
    sub.choices["record"].add_argument("--allow-rise", action="store_true")
    sub.choices["list"].add_argument("--tree", choices=[t.name for t in TREES])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "list":
            return listing(root, args.tree)
        baseline = (
            args.baseline if args.baseline.is_absolute() else root / args.baseline
        )
        if args.command == "check":
            return check(root, baseline, fail_on_drop=args.fail_on_drop)
        return record(root, baseline, allow_rise=args.allow_rise)
    except ToolError as exc:
        sys.stderr.write(f"check_clones: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
