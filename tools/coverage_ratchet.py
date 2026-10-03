"""No-fall ratchet on the per-file coverage of the library (TEST3).

``baselines/coverage_baseline.json`` holds, for every file of a full-suite run with branch coverage,
the executed and the total number of statements plus branches. A file breaks the ratchet when its
covered share falls **and** its number of uncovered statements and branches (``total - covered``)
rises; ``check`` exits 1 when one does, ``record`` refuses to write a baseline that lowers a file
that way unless ``--allow-fall`` says so. A file that disappeared from the report (deleted or
renamed) is reported but is not a failure; a file that is new is recorded at the next ``record``.

Why both conditions: a refactor that removes fully covered code lowers the share even though it
leaves less untested code than before (``instructions/compute.py`` went from 2596 of 3218 covered,
80.67%, to 2581 of 3202, 80.61%, while the uncovered count fell from 622 to 621). The share alone
would fail that change; the uncovered count alone would accept a file that grows by untested code
while a lot of covered code is added next to it. Together they fail exactly the changes that add
untested code without improving the share. The lower share of an accepted file becomes the new
baseline at the next ``record``, so the uncovered count can only go down from there.

Usage::

    python tools/coverage_ratchet.py record COVERAGE_JSON [--baseline FILE] [--allow-fall]
    python tools/coverage_ratchet.py check COVERAGE_JSON [--baseline FILE] [--tolerance PP]

``COVERAGE_JSON`` is a coverage.py report (``--cov-report=json``) made with ``--cov-branch`` over
the whole suite. ``--tolerance`` is in percentage points (default 0: coverage of lines does not
depend on the numerics, and a parallel run gave the same per-file numbers as a serial one); it widens
the share condition only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE = Path("baselines/coverage_baseline.json")

Counts = dict[str, dict[str, int]]


class RatchetError(RuntimeError):
    """The report or the baseline cannot be used."""


def counts_from_report(report: dict) -> Counts:
    """Executed and total statements plus branches of each file of a coverage.py report."""
    if not report.get("meta", {}).get("branch_coverage"):
        msg = (
            "the report was made without branch coverage; run pytest with --cov-branch"
        )
        raise RatchetError(msg)
    counts: Counts = {}
    for path, entry in report["files"].items():
        s = entry["summary"]
        counts[path] = {
            "covered": s["covered_lines"] + s["covered_branches"],
            "total": s["num_statements"] + s["num_branches"],
        }
    return counts


def percent(entry: dict[str, int]) -> float:
    """Share covered, in percent; an empty file counts as fully covered."""
    return 100.0 if entry["total"] == 0 else 100.0 * entry["covered"] / entry["total"]


def uncovered(entry: dict[str, int]) -> int:
    """Statements plus branches that no test executed."""
    return entry["total"] - entry["covered"]


def falls(base: Counts, now: Counts, tolerance: float = 0.0) -> list[str]:
    """One line for each file whose share fell and whose uncovered count rose.

    The share must lose more than ``tolerance`` percentage points; the uncovered count must be
    strictly higher than in the baseline.
    """
    lines = []
    for path in sorted(set(base) & set(now)):
        before, after = percent(base[path]), percent(now[path])
        unc_before, unc_after = uncovered(base[path]), uncovered(now[path])
        if after < before - tolerance - 1e-9 and unc_after > unc_before:
            lines.append(
                f"fell: {path} {before:.2f}% -> {after:.2f}%, "
                f"uncovered {unc_before} -> {unc_after}"
            )
    return lines


def _load(path: Path) -> dict:
    if not path.is_file():
        msg = f"no baseline at {path}; record one with `python tools/coverage_ratchet.py record`"
        raise RatchetError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def check(report: dict, baseline: Path, tolerance: float = 0.0) -> int:
    """Print the files that broke the ratchet; return 1 if there are any."""
    base = _load(baseline)["files"]
    now = counts_from_report(report)
    problems = falls(base, now, tolerance)
    gone = sorted(set(base) - set(now))
    sys.stdout.write("\n".join(problems) + ("\n" if problems else ""))
    sys.stdout.write("".join(f"gone: {path}\n" for path in gone))
    sys.stdout.write(
        f"coverage ratchet: {len(base)} files in the baseline, {len(problems)} fell, "
        f"{len(gone)} gone, {len(set(now) - set(base))} new\n"
    )
    return 1 if problems else 0


def record(report: dict, baseline: Path, *, allow_fall: bool = False) -> int:
    """Write the baseline from ``report``; refuse a fall unless ``allow_fall``."""
    now = counts_from_report(report)
    if baseline.is_file() and not allow_fall:
        problems = falls(_load(baseline)["files"], now)
        if problems:
            sys.stdout.write(
                "refusing to record a fall (use --allow-fall):\n"
                + "\n".join(problems)
                + "\n"
            )
            return 1
    data = {"tool": f"coverage.py {report['meta']['version']}", "files": now}
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = {k: sum(v[k] for v in now.values()) for k in ("covered", "total")}
    sys.stdout.write(
        f"recorded {len(now)} files in {baseline}: {percent(total):.2f}% "
        f"({total['covered']}/{total['total']})\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run ``record`` or ``check``; return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["record", "check"])
    parser.add_argument("coverage", type=Path, help="coverage.py JSON report")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--tolerance", type=float, default=0.0, help="percentage points"
    )
    parser.add_argument("--allow-fall", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.coverage.read_text(encoding="utf-8"))
        if args.command == "record":
            return record(report, args.baseline, allow_fall=args.allow_fall)
        return check(report, args.baseline, args.tolerance)
    except (RatchetError, OSError, json.JSONDecodeError, KeyError) as err:
        sys.stderr.write(f"coverage_ratchet: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
