"""One quality report of the library, with a baseline that may not get worse (TOOL4).

The metrics that say whether the code gets better or worse come from six tools and three baseline files; this
tool collects them in one table (JSON and markdown) and ``check`` fails when a ratcheted one is worse than in
``baselines/quality.json``. Library means the tree ``check_clones.py`` calls ``library`` (``tensorpotential/`` without
``compat/pace/``, out of scope for every issue); the ``coverage``, ``clones``, ``api`` and ``todo`` groups are scoped to it,
the ``lint`` and ``divergence`` groups are those of the whole repository, as their tools measure it.

Metrics, by group (the name of a metric is ``group.detail``):

* ``lint``: ruff findings per rule and ty findings per rule of the legacy files, and the findings of strict files
  (``lint_ratchet.py``); counts, lower is better.
* ``coverage``: statements plus branches executed per package and in total, from a coverage.py JSON report made with
  ``--cov-branch``; ratios. A package is worse when its share fell **and** its uncovered count rose (the rule of
  ``coverage_ratchet.py``: removing covered code is not a regression).
* ``clones``: groups, functions and redundant lines of duplicated functions per tree (``check_clones.py``); counts.
* ``api``: share of the public functions (no leading underscore, module level or in a public class) with a return
  annotation and with a docstring; ratios, same rule as coverage.
* ``todo``: ``TODO``, ``FIXME`` and ``XXX`` markers in the library (a marker links an issue; the count may only fall).
* ``divergence``: upstream files the fork modifies, the rows of ``tools/divergence.yaml``; count.
* ``mutation``: the score per module of ``baselines/mutation_score.json`` when that file exists (``{"modules": {name:
  score}}``, a score being a number or ``{"score": number}``); scores, higher is better.
* ``tests``: test count and summed duration of the junit files given with ``--junit``; reported, never ratcheted
  (a duration depends on the runner and a count falls when a test is merged or deleted on purpose).

Usage::

    python tools/quality_report.py report [--coverage COV_JSON | --ratchet-coverage | --no-coverage] [--junit XML ...] [--json FILE] [--markdown FILE]
    python tools/quality_report.py check  [same options] [--baseline FILE]
    python tools/quality_report.py record [same options] [--baseline FILE] [--allow-worse]

``check`` exits 1 and names each worse metric with its old and new value, 2 when an input cannot be used. Coverage is
required: a report with ``--coverage``, the counts of ``baselines/coverage_baseline.json`` with ``--ratchet-coverage`` (how the
first baseline was recorded, without a full-suite run), or ``--no-coverage``; a group that is not measured keeps its baseline entries at ``record``.
``--markdown FILE`` appends the table to FILE (the job summary of the nightly run); without ``--json`` and
``--markdown`` ``report`` prints the table.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import check_clones
import coverage_ratchet
import divergence
import lint_ratchet
import yaml

BASELINE = Path("baselines/quality.json")
MUTATION_FILE = Path("baselines/mutation_score.json")
_LIBRARY_TREE = next(t for t in check_clones.TREES if t.name == "library")
LIBRARY = _LIBRARY_TREE.path
EXCLUDED = tuple(f"{LIBRARY}/{name}/" for name in _LIBRARY_TREE.exclude)
MARKER = re.compile(r"\b(TODO|FIXME|XXX)\b")
EPS = 1e-9


class ReportError(RuntimeError):
    """An input of the report cannot be used."""


@dataclass(frozen=True)
class Metric:
    """One number: ``count`` (lower is better), ``ratio`` (``have`` of ``total``), ``score`` or ``info``."""

    kind: Literal["count", "ratio", "score", "info"]
    value: float = 0.0
    have: int = 0
    total: int = 0

    def shown(self) -> str:
        """Return the value as it appears in a table."""
        if self.kind == "ratio":
            return f"{self.share:.2%} ({self.have}/{self.total})"
        return f"{self.value:g}"

    @property
    def share(self) -> float:
        """``have / total`` of a ratio; an empty one counts as complete."""
        return 1.0 if self.total == 0 else self.have / self.total

    def to_json(self) -> dict:
        """Return the JSON form (the value of a ratio is derived, not stored)."""
        if self.kind == "ratio":
            return {"kind": "ratio", "have": self.have, "total": self.total}
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_json(cls, data: dict) -> Metric:
        """Read the JSON form."""
        return cls(
            data["kind"],
            data.get("value", 0.0),
            data.get("have", 0),
            data.get("total", 0),
        )


Metrics = dict[str, Metric]


def count(value: int) -> Metric:
    """Make a count: lower is better."""
    return Metric("count", float(value))


def ratio(have: int, total: int) -> Metric:
    """Make a share: higher is better, unless the missing part did not grow."""
    return Metric("ratio", have=have, total=total)


def in_library(path: str) -> bool:
    """Whether a repository-relative path lies in the library and outside the excluded packages."""
    return path.startswith(f"{LIBRARY}/") and not path.startswith(EXCLUDED)


def library_files(root: Path) -> list[Path]:
    """List the ``.py`` files of the library, sorted, without the excluded packages."""
    return sorted(
        path
        for path in (root / LIBRARY).rglob("*.py")
        if in_library(path.relative_to(root).as_posix())
    )


def lint_metrics(root: Path) -> Metrics:
    """Sum the findings per rule over all files, from the measurement of ``lint_ratchet.py``."""
    legacy = lint_ratchet.read_legacy(root / lint_ratchet.LEGACY_LIST)
    found = lint_ratchet.measure(root, legacy)
    out: Metrics = {"lint.strict": count(len(found["strict"]))}
    for tool in ("ruff", "ty"):
        totals: Counter[str] = Counter()
        for rules in found[tool].values():
            totals.update(rules)
        out.update({f"lint.{tool}.{rule}": count(n) for rule, n in totals.items()})
    return out


def package_of(path: str) -> str:
    """``tensorpotential/data/x.py`` -> ``data``; a module directly in the library -> ``toplevel``."""
    parts = path.split("/")
    return parts[1] if len(parts) > 2 else "toplevel"


def coverage_metrics(counts: coverage_ratchet.Counts) -> Metrics:
    """Sum executed over total statements plus branches per package and in total."""
    covered: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for path, entry in counts.items():
        if in_library(path):
            for key in (package_of(path), "total"):
                covered[key] += entry["covered"]
                total[key] += entry["total"]
    return {f"coverage.{k}": ratio(covered[k], total[k]) for k in sorted(total)}


def clone_metrics(root: Path) -> Metrics:
    """Sum groups, functions and redundant lines per tree over the kinds (as ``check_clones.py`` does)."""
    out: Metrics = {}
    for tree, kinds in check_clones.measure(root).items():
        for metric in check_clones.METRICS:
            out[f"clones.{tree}.{metric}"] = count(
                sum(k[metric] for k in kinds.values())
            )
    return out


_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _public_functions(tree: ast.Module | ast.ClassDef) -> list[_Function]:
    """Functions and methods reachable through public names only (nested functions are not API)."""
    found: list[_Function] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            found += _public_functions(node)
        elif isinstance(node, _Function) and not node.name.startswith("_"):
            found.append(node)
    return found


def api_metrics(root: Path) -> Metrics:
    """Share of the public functions of the library with a return annotation and with a docstring."""
    funcs = []
    for path in library_files(root):
        funcs += _public_functions(ast.parse(path.read_text(encoding="utf-8")))
    return {
        "api.return_annotations": ratio(
            sum(f.returns is not None for f in funcs), len(funcs)
        ),
        "api.docstrings": ratio(
            sum(ast.get_docstring(f) is not None for f in funcs), len(funcs)
        ),
    }


def todo_metrics(root: Path) -> Metrics:
    """Count the lines of the library with a ``TODO``, ``FIXME`` or ``XXX`` marker (two on a line count once)."""
    lines = (
        line
        for path in library_files(root)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    return {"todo.markers": count(sum(bool(MARKER.search(line)) for line in lines))}


def divergence_metrics(root: Path) -> Metrics:
    """Rows of the divergence ledger: one per upstream file the fork modifies."""
    try:
        rows = divergence.load_ledger(root / "tools" / "divergence.yaml")
    except (SystemExit, yaml.YAMLError) as err:  # load_ledger exits with a message
        msg = f"the divergence ledger cannot be read: {err}"
        raise ReportError(msg) from err
    return {"divergence.upstream_files": count(len(rows))}


def junit_metrics(junit_files: list[Path]) -> Metrics:
    """Count the ``testcase`` entries and sum their seconds over junit files, reported only."""
    cases = []
    for path in junit_files:
        root = ET.parse(path).getroot()  # noqa: S314 - our own pytest output
        cases += list(root.iter("testcase"))
    seconds = sum(float(c.get("time", 0.0)) for c in cases)
    return {
        "tests.count": Metric("info", len(cases)),
        "tests.seconds": Metric("info", round(seconds, 1)),
    }


def mutation_metrics(path: Path) -> Metrics:
    """Score per module of the mutation baseline (``{"modules": {name: score | {"score": score}}}``)."""
    try:
        modules = json.loads(path.read_text(encoding="utf-8"))["modules"]
        return {
            f"mutation.{name}": Metric(
                "score", float(s["score"] if isinstance(s, dict) else s)
            )
            for name, s in sorted(modules.items())
        }
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        msg = f"{path} does not hold a mutation score per module: {err!r}"
        raise ReportError(msg) from err


def coverage_counts(
    root: Path, args: argparse.Namespace
) -> coverage_ratchet.Counts | None:
    """Read the per-file counts the command line names, or ``None`` for ``--no-coverage``."""
    if args.coverage is not None:
        return coverage_ratchet.counts_from_report(_read_json(args.coverage))
    if args.ratchet_coverage:
        return _read_json(root / coverage_ratchet.BASELINE)["files"]
    return None


def collect(root: Path, args: argparse.Namespace) -> tuple[Metrics, set[str]]:
    """All metrics of the tree and the groups that were measured."""
    metrics: Metrics = {
        **lint_metrics(root),
        **clone_metrics(root),
        **api_metrics(root),
        **todo_metrics(root),
        **divergence_metrics(root),
    }
    groups = {"lint", "clones", "api", "todo", "divergence"}
    counts = coverage_counts(root, args)
    if counts is not None:
        metrics.update(coverage_metrics(counts))
        groups.add("coverage")
    mutation = root / MUTATION_FILE
    if mutation.is_file():
        metrics.update(mutation_metrics(mutation))
        groups.add("mutation")
    if args.junit:
        metrics.update(junit_metrics(args.junit))
    return metrics, groups


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        msg = f"cannot read {path}: {err}"
        raise ReportError(msg) from err


def group_of(name: str) -> str:
    """``lint.ruff.C901`` -> ``lint``."""
    return name.split(".", 1)[0]


def worse(old: Metric | None, new: Metric) -> bool:
    """Whether ``new`` is worse than ``old``; an absent count was zero, another absent metric never fails."""
    if new.kind == "count":
        return new.value > (old.value if old else 0.0) + EPS
    if old is None:
        return False
    if new.kind == "score":
        return new.value < old.value - EPS
    if new.kind == "ratio":
        return (
            new.share < old.share - EPS and new.total - new.have > old.total - old.have
        )
    return False  # info is reported, never ratcheted


def regressions(base: Metrics, now: Metrics, groups: set[str]) -> list[str]:
    """``metric: old -> new`` for each metric of a measured group that got worse; a gone metric is not."""
    lines = []
    for name, new in sorted(now.items()):
        old = base.get(name)
        if group_of(name) in groups and worse(old, new):
            lines.append(f"{name}: {old.shown() if old else '0'} -> {new.shown()}")
    return lines


def table(now: Metrics, base: Metrics | None = None) -> str:
    """Format the metrics as a markdown table, one section per group, with the baseline value when there is one."""
    rows = ["| metric | now | baseline |", "|---|---|---|"]
    previous = None
    for name, metric in sorted(now.items()):
        if group_of(name) != previous:
            previous = group_of(name)
            rows.append(f"| **{previous}** | | |")
        old = (base or {}).get(name)
        rows.append(f"| {name} | {metric.shown()} | {old.shown() if old else ''} |")
    return "\n".join(rows) + "\n"


def load_baseline(path: Path) -> Metrics:
    """Read the ratcheted metrics of a baseline file."""
    if not path.is_file():
        msg = f"no baseline at {path}; record one with `python tools/quality_report.py record`"
        raise ReportError(msg)
    return {n: Metric.from_json(m) for n, m in _read_json(path)["metrics"].items()}


def _emit(now: Metrics, base: Metrics | None, args: argparse.Namespace) -> None:
    text = table(now, base)
    if args.json:
        data = {"metrics": {n: m.to_json() for n, m in sorted(now.items())}}
        args.json.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    if args.markdown:
        with args.markdown.open("a", encoding="utf-8") as out:
            out.write(f"## Quality report\n\n{text}\n")
    if args.command == "report" and not (args.json or args.markdown):
        sys.stdout.write(text)


def cmd_check(now: Metrics, groups: set[str], args: argparse.Namespace) -> int:
    """Print the worse metrics; 1 when there are any."""
    base = load_baseline(args.baseline)
    _emit(now, base, args)
    lines = regressions(base, now, groups)
    sys.stdout.write(
        "".join(f"worse: {line}\n" for line in lines) or "quality report: ok\n"
    )
    return 1 if lines else 0


def cmd_record(now: Metrics, groups: set[str], args: argparse.Namespace) -> int:
    """Write the baseline: the ratcheted metrics measured now, plus the entries of the groups not measured."""
    old = load_baseline(args.baseline) if args.baseline.is_file() else {}
    # the first record has nothing to be worse than; a group the baseline lacks is new, not worse
    lines = regressions(old, now, groups & {group_of(n) for n in old})
    if lines and not args.allow_worse:
        sys.stdout.write(
            "refusing to record a worse value (use --allow-worse):\n"
            + "\n".join(lines)
            + "\n"
        )
        return 1
    kept = {n: m for n, m in old.items() if group_of(n) not in groups}
    kept.update({n: m for n, m in now.items() if m.kind != "info"})
    data = {"metrics": {n: m.to_json() for n, m in sorted(kept.items())}}
    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(f"recorded {len(kept)} metrics in {args.baseline}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["report", "check", "record"])
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    cov = parser.add_mutually_exclusive_group(required=True)
    cov.add_argument(
        "--coverage", type=Path, help="coverage.py JSON report with branch coverage"
    )
    cov.add_argument(
        "--ratchet-coverage",
        action="store_true",
        help="use the per-file counts of baselines/coverage_baseline.json (what the coverage ratchet accepted last)",
    )
    cov.add_argument(
        "--no-coverage", action="store_true", help="leave the coverage group out"
    )
    parser.add_argument(
        "--junit", type=Path, nargs="*", default=[], help="junit files of the test run"
    )
    parser.add_argument("--json", type=Path, help="write the metrics as JSON")
    parser.add_argument("--markdown", type=Path, help="append the table to this file")
    parser.add_argument(
        "--allow-worse", action="store_true", help="record: accept a worse value"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``report``, ``check`` or ``record``; return the process exit status."""
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    args.baseline = root / args.baseline
    try:
        now, groups = collect(root, args)
        if args.command == "check":
            return cmd_check(now, groups, args)
        if args.command == "record":
            return cmd_record(now, groups, args)
        _emit(now, None, args)
    except (
        ReportError,
        coverage_ratchet.RatchetError,
        lint_ratchet.ToolError,
        check_clones.ToolError,
        OSError,
        KeyError,
        TypeError,
    ) as err:
        sys.stderr.write(f"quality_report: {err}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
