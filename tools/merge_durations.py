r"""Merge the per-shard test durations of the ``Durations`` workflow, and warn when the committed file is stale.

The four ``test-tf`` shards are balanced by ``tests/.test_durations`` (``pytest-split``). The ``Durations`` workflow
runs each shard with ``--store-durations --clean-durations`` on the hosted runner, so a shard file holds exactly the
tests of that shard; ``merge`` joins them into one file in the format ``pytest-split`` writes (sorted keys, four-space
indent, trailing newline) and drops the committed entries of tests that no shard ran any more. ``stale`` counts the
collected tests the committed file does not know and prints a GitHub ``::warning::`` above a threshold.

Usage::

    python tools/merge_durations.py merge --committed tests/.test_durations --out merged.json \\
        [--summary summary.md] shard-1.json shard-2.json shard-3.json shard-4.json
    python tools/merge_durations.py stale --committed tests/.test_durations --collected collected.txt [--threshold 0.10]

Standard library only. A malformed file, a test in two shards or an empty set of shards is an error (status 2).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CHANGE_THRESHOLD = 0.20  # a duration that moved by more than this share of the committed one counts as changed
STALE_THRESHOLD = 0.10  # share of unknown collected tests above which `stale` warns
_NODE_ID = re.compile(r"^[\w./-]+\.py::\S.*$")


class DurationsError(ValueError):
    """A durations file or a set of shard files that cannot be used."""


@dataclass(frozen=True)
class MergeResult:
    """The merged durations and what changed against the committed file.

    Attributes
    ----------
    durations
        Test id to seconds, the content of the new file.
    new
        Ids of tests the committed file did not know.
    dropped
        Ids of committed tests that no shard ran.
    changed_share
        Share of the total merged duration held by tests known to both files whose duration moved by more than
        ``CHANGE_THRESHOLD`` of the committed one; ``0.0`` when the total is zero.
    """

    durations: dict[str, float]
    new: frozenset[str]
    dropped: frozenset[str]
    changed_share: float


def load_durations(path: Path) -> dict[str, float]:
    """Read a durations file; anything but an object of ids to finite non-negative seconds is an error.

    Parameters
    ----------
    path
        File written by ``pytest-split`` (a JSON object).

    Returns
    -------
    dict[str, float]
        Test id to seconds.

    Raises
    ------
    DurationsError
        The file is missing, not JSON, not an object, or holds a value that is not a finite non-negative number.
    """
    try:
        data = json.loads(path.read_text())
    except OSError as err:
        msg = f"{path}: cannot be read ({err.strerror or err})"
        raise DurationsError(msg) from err
    except json.JSONDecodeError as err:
        msg = f"{path}: not valid JSON ({err})"
        raise DurationsError(msg) from err
    if not isinstance(data, dict):
        msg = f"{path}: expected a JSON object of test ids to seconds, got {type(data).__name__}"
        raise DurationsError(msg)
    for key, value in data.items():
        valid = isinstance(value, int | float) and not isinstance(value, bool)
        if not valid or not math.isfinite(value) or value < 0:
            msg = f"{path}: the duration of {key} is {value!r}, expected a finite non-negative number"
            raise DurationsError(msg)
    return {str(key): float(value) for key, value in data.items()}


def format_durations(durations: dict[str, float]) -> str:
    """Return the text of a durations file: sorted keys, four-space indent, trailing newline."""
    return json.dumps(durations, sort_keys=True, indent=4) + "\n"


def _union_of_shards(shards: list[dict[str, float]]) -> dict[str, float]:
    """Join disjoint shard files; a test in two of them means the shards were split from different files."""
    if not shards:
        msg = "no shard file given"
        raise DurationsError(msg)
    owner: dict[str, int] = {}
    union: dict[str, float] = {}
    for index, shard in enumerate(shards, start=1):
        if not shard:
            msg = f"shard file {index} holds no test"
            raise DurationsError(msg)
        for test_id, seconds in shard.items():
            if test_id in owner:
                msg = f"{test_id} is in shard files {owner[test_id]} and {index}: the shards were not split from one durations file"
                raise DurationsError(msg)
            owner[test_id] = index
            union[test_id] = seconds
    return union


def _changed_share(committed: dict[str, float], merged: dict[str, float]) -> float:
    """Share of the merged total held by the tests of both files whose duration changed by more than 20%."""
    total = sum(merged.values())
    if total == 0:
        return 0.0
    changed = sum(
        seconds
        for test_id, seconds in merged.items()
        if test_id in committed
        and abs(seconds - committed[test_id]) > CHANGE_THRESHOLD * committed[test_id]
    )
    return changed / total


def merge(committed: dict[str, float], shards: list[dict[str, float]]) -> MergeResult:
    """Merge the shard files; the shard value of a test replaces the committed one.

    Parameters
    ----------
    committed
        The durations file in the repository.
    shards
        One mapping per shard, each holding only the tests of that shard.

    Returns
    -------
    MergeResult
        The new durations and the differences against ``committed``.

    Raises
    ------
    DurationsError
        There is no shard, a shard is empty, or a test is in two shards.
    """
    durations = _union_of_shards(shards)
    return MergeResult(
        durations=durations,
        new=frozenset(durations.keys() - committed.keys()),
        dropped=frozenset(committed.keys() - durations.keys()),
        changed_share=_changed_share(committed, durations),
    )


def summary_markdown(result: MergeResult) -> str:
    """Return the Markdown summary of a merge, for ``$GITHUB_STEP_SUMMARY``."""
    total = sum(result.durations.values())
    return (
        "### Test durations\n\n"
        "| | |\n|---|---|\n"
        f"| tests in the file | {len(result.durations)} |\n"
        f"| new (not in the committed file) | {len(result.new)} |\n"
        f"| dropped (no longer run) | {len(result.dropped)} |\n"
        f"| total duration | {total:.0f} s |\n"
        f"| share of the total that changed by more than {CHANGE_THRESHOLD:.0%} | {result.changed_share:.1%} |\n"
    )


def parse_collected(text: str) -> set[str]:
    """Test ids of a ``pytest --collect-only -q`` output; the summary and warning lines are not ids."""
    return {line.strip() for line in text.splitlines() if _NODE_ID.match(line.strip())}


def staleness_message(
    committed: dict[str, float], collected: set[str], threshold: float = STALE_THRESHOLD
) -> str | None:
    """Return a GitHub ``::warning::`` line when more than ``threshold`` of the collected tests are unknown, else ``None``.

    Raises
    ------
    DurationsError
        Nothing was collected (an empty list would make every file look fresh).
    """
    if not collected:
        msg = "no collected test id given"
        raise DurationsError(msg)
    unknown = collected - committed.keys()
    share = len(unknown) / len(collected)
    if share <= threshold:
        return None
    return (
        f"::warning title=Stale test durations::{len(unknown)} of {len(collected)} collected tests ({share:.0%}) "
        f"have no entry in tests/.test_durations, so the shards drift apart; regenerate it with the Durations "
        f"workflow (.github/CI.md)"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    merge_cmd = sub.add_parser(
        "merge", help="merge the shard files into one durations file"
    )
    merge_cmd.add_argument("--committed", type=Path, required=True)
    merge_cmd.add_argument("--out", type=Path, required=True)
    merge_cmd.add_argument(
        "--summary", type=Path, help="write the Markdown summary here"
    )
    merge_cmd.add_argument("shards", type=Path, nargs="+")
    stale_cmd = sub.add_parser(
        "stale", help="warn when the committed file does not know many collected tests"
    )
    stale_cmd.add_argument("--committed", type=Path, required=True)
    stale_cmd.add_argument(
        "--collected",
        type=Path,
        required=True,
        help="output of pytest --collect-only -q",
    )
    stale_cmd.add_argument("--threshold", type=float, default=STALE_THRESHOLD)
    return parser.parse_args(argv)


def _run_merge(args: argparse.Namespace) -> None:
    result = merge(
        load_durations(args.committed), [load_durations(path) for path in args.shards]
    )
    args.out.write_text(format_durations(result.durations))
    summary = summary_markdown(result)
    if args.summary is not None:
        args.summary.write_text(summary)
    sys.stdout.write(summary)


def _run_stale(args: argparse.Namespace) -> None:
    try:
        text = args.collected.read_text()
    except OSError as err:
        msg = f"{args.collected}: cannot be read ({err.strerror or err})"
        raise DurationsError(msg) from err
    message = staleness_message(
        load_durations(args.committed), parse_collected(text), args.threshold
    )
    sys.stdout.write((message or "test durations are up to date") + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run ``merge`` or ``stale``; return the exit status (0, or 2 for an unusable input)."""
    args = _parse_args(argv)
    try:
        {"merge": _run_merge, "stale": _run_stale}[args.command](args)
    except DurationsError as err:
        sys.stderr.write(f"merge_durations: {err}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
