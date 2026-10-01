"""Per-test outcomes from a pytest junit file, and a diff of two of them.

Junit records an XFAIL as a skip with type ``pytest.xfail`` and an XPASS as a plain
pass, so XPASS ids are read from the ``-rX`` lines of the pytest log when given.

Usage::

    python tools/junit_outcomes.py summarize junit.xml out.json [--log pytest.log]
    python tools/junit_outcomes.py compare baseline.json new.json

``compare`` exits with status 1 when a test was added, removed or changed outcome.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_XPASS_LINE = re.compile(r"^XPASS (\S+)")


def _case_outcome(case: ET.Element) -> str:
    """Outcome of one ``<testcase>``; a failure or error outranks a skip or pass."""
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    skipped = case.find("skipped")
    if skipped is not None:
        return "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def _case_id(case: ET.Element) -> str:
    return f"{case.get('classname', '')}::{case.get('name', '')}"


def xpass_ids(log_text: str) -> set[str]:
    """Ids (``file.py::test``) on the ``XPASS`` lines of a ``-rX`` pytest summary."""
    return {
        m.group(1) for line in log_text.splitlines() if (m := _XPASS_LINE.match(line))
    }


def _junit_id(nodeid: str) -> str:
    """Convert ``dir/test_a.py::test_b`` to the junit form ``test_a::test_b``."""
    path, _, name = nodeid.partition("::")
    return f"{path.removesuffix('.py').replace('/', '.')}::{name}"


def _matching_ids(tests: dict[str, str], junit_id: str) -> list[str]:
    """Junit ids equal to ``junit_id`` or ending with it.

    Log ids are relative to the directory pytest ran in, junit class names to its
    rootdir (``tests.test_a::test_b`` for a run inside ``tests/``).
    """
    return [t for t in tests if t == junit_id or t.endswith("." + junit_id)]


def summarize(junit_path: Path, log_path: Path | None = None) -> dict:
    """Return ``{"counts": {...}, "tests": {id: outcome}}`` for a junit file."""
    tests: dict[str, str] = {}
    root = ET.parse(junit_path).getroot()  # noqa: S314 - our own pytest output
    for case in root.iter("testcase"):
        test_id = _case_id(case)
        outcome = _case_outcome(case)
        # a failing setup or teardown is a second testcase entry for the same id
        if tests.get(test_id) in ("failed", "error"):
            continue
        tests[test_id] = outcome
    if log_path is not None:
        for nodeid in xpass_ids(log_path.read_text(errors="replace")):
            matches = _matching_ids(tests, _junit_id(nodeid))
            if len(matches) != 1:
                raise ValueError(  # noqa: TRY003
                    f"XPASS line {nodeid!r} matches {len(matches)} junit tests "
                    f"({matches}); is the log from the same run as the junit file?"
                )
            if tests[matches[0]] == "passed":
                tests[matches[0]] = "xpassed"
    counts = Counter(tests.values())
    return {
        "counts": {k: counts.get(k, 0) for k in OUTCOMES},
        "tests": dict(sorted(tests.items())),
    }


def compare(baseline: dict, new: dict) -> dict[str, list]:
    """Return the tests added, removed or changed in outcome (``[id, old, new]``)."""
    old_tests, new_tests = baseline["tests"], new["tests"]
    return {
        "added": sorted(set(new_tests) - set(old_tests)),
        "removed": sorted(set(old_tests) - set(new_tests)),
        "changed": [
            [t, old_tests[t], new_tests[t]]
            for t in sorted(set(old_tests) & set(new_tests))
            if old_tests[t] != new_tests[t]
        ],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    summ = sub.add_parser("summarize", help="junit xml -> outcome json")
    summ.add_argument("junit", type=Path)
    summ.add_argument("out", type=Path)
    summ.add_argument("--log", type=Path, help="pytest log with -rX lines (XPASS)")
    cmp_ = sub.add_parser("compare", help="diff two outcome json files")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("new", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ``summarize`` or ``compare`` command; return the exit status."""
    args = _parse_args(argv)
    if args.command == "summarize":
        result = summarize(args.junit, args.log)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        sys.stdout.write(f"{len(result['tests'])} tests: {result['counts']}\n")
        return 0
    diff = compare(
        json.loads(args.baseline.read_text()), json.loads(args.new.read_text())
    )
    for kind, items in diff.items():
        for item in items:
            sys.stdout.write(f"{kind}: {item}\n")
    return 1 if any(diff.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
