"""No-increase ratchet on the dead code that ``vulture`` finds in the library (DEPS4).

``baselines/dead_code.json`` holds, for every file of ``tensorpotential/`` with at least one finding, the number of findings at the
recorded confidence (80 by default: unused imports, unreachable code and unused arguments; the 60% findings of vulture are mostly
public API, presets and loss classes that users call by name, which a tool cannot tell from dead code). ``check`` exits 1 when a
file has more findings than in the baseline, or a file without an entry has any; ``record`` refuses to write a baseline that
raises a file unless ``--allow-rise`` says so. A count that fell is reported, and is lowered at the next ``record``.

What is alive for a reason the tool cannot see goes in ``tools/dead_code_whitelist.txt`` (one name or glob pattern per line with
the reason after ``#``; a line that starts with ``@`` is a decorator pattern whose functions are alive, for example a web route).
``compat/pace/`` is dead by the owner's decision of 2026-10-01 and is not scanned.

``--moved OLD NEW`` (repeatable) says that a file moved: the baseline entry of ``OLD`` is read as the entry of ``NEW``.

Usage::

    python tools/dead_code.py check [--baseline FILE] [--whitelist FILE] [--moved OLD NEW]
    python tools/dead_code.py record [--baseline FILE] [--whitelist FILE] [--min-confidence N] [--allow-rise] [--moved OLD NEW]
    python tools/dead_code.py list [--whitelist FILE] [--min-confidence N]

Run it from the repository root; ``list`` prints every finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import vulture
from ratchet_moves import parse_moves, rekey

BASELINE = Path("baselines/dead_code.json")
WHITELIST = Path("tools/dead_code_whitelist.txt")
PACKAGE = "tensorpotential"
EXCLUDE = ("*/tensorpotential/compat/pace/*",)
MIN_CONFIDENCE = 80

Counts = dict[str, int]


class DeadCodeError(RuntimeError):
    """The baseline or the whitelist cannot be used."""


@dataclass(frozen=True)
class Finding:
    """One piece of code that vulture reports as unused."""

    path: str
    line: int
    confidence: int
    kind: str
    name: str

    def __str__(self) -> str:
        """Render the finding the way vulture prints it."""
        return f"{self.path}:{self.line}: unused {self.kind} '{self.name}' ({self.confidence}% confidence)"


def load_whitelist(path: Path) -> tuple[list[str], list[str]]:
    """Names and decorator patterns of the whitelist; the reason after ``#`` is for the reader."""
    if not path.is_file():
        msg = f"no whitelist at {path}"
        raise DeadCodeError(msg)
    names: list[str] = []
    decorators: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        (decorators if entry.startswith("@") else names).append(entry)
    return names, decorators


def scan(
    root: Path,
    whitelist: Path,
    min_confidence: int = MIN_CONFIDENCE,
    package: str = PACKAGE,
) -> list[Finding]:
    """Every finding of vulture over ``root/package`` at ``min_confidence``, sorted by file and line."""
    names, decorators = load_whitelist(whitelist)
    scavenger = vulture.Vulture(ignore_names=names, ignore_decorators=decorators)
    scavenger.scavenge([str(root / package)], exclude=list(EXCLUDE))
    base = root.resolve()
    found = [
        Finding(
            Path(item.filename).resolve().relative_to(base).as_posix(),
            item.first_lineno,
            item.confidence,
            item.typ,
            item.name,
        )
        for item in scavenger.get_unused_code(min_confidence=min_confidence)
    ]
    return sorted(found, key=lambda f: (f.path, f.line, f.name))


def counts_of(findings: list[Finding]) -> Counts:
    """Count the findings of each file."""
    counts: Counts = {}
    for finding in findings:
        counts[finding.path] = counts.get(finding.path, 0) + 1
    return counts


def rises(base: Counts, now: Counts) -> list[str]:
    """One line for each file that has more findings than in the baseline (a file without an entry has zero)."""
    return [
        f"rose: {path} {base.get(path, 0)} -> {count}"
        for path, count in sorted(now.items())
        if count > base.get(path, 0)
    ]


def _load(path: Path) -> dict:
    if not path.is_file():
        msg = (
            f"no baseline at {path}; record one with `python tools/dead_code.py record`"
        )
        raise DeadCodeError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_counts(data: dict, moves: dict[str, str]) -> Counts:
    return rekey(data["files"], moves)


def check(
    root: Path, baseline: Path, whitelist: Path, moves: dict[str, str] | None = None
) -> int:
    """Print the files that gained findings and the findings in them; return 1 if there are any."""
    data = _load(baseline)
    base = _baseline_counts(data, moves or {})
    findings = scan(root, whitelist, data["min_confidence"])
    now = counts_of(findings)
    problems = rises(base, now)
    sys.stdout.write("\n".join(problems) + ("\n" if problems else ""))
    risen = {line.split()[1] for line in problems}
    sys.stdout.write("".join(f"  {f}\n" for f in findings if f.path in risen))
    fallen = sorted(p for p in base if now.get(p, 0) < base[p])
    sys.stdout.write(
        f"dead code ratchet at {data['min_confidence']}% confidence: {sum(now.values())} findings "
        f"in {len(now)} files (baseline {sum(base.values())}), {len(problems)} files rose, "
        f"{len(fallen)} fell (record to lower the baseline)\n"
    )
    return 1 if problems else 0


def record(
    root: Path,
    baseline: Path,
    whitelist: Path,
    min_confidence: int = MIN_CONFIDENCE,
    moves: dict[str, str] | None = None,
    *,
    allow_rise: bool = False,
) -> int:
    """Write the baseline from the tree; refuse a rise unless ``allow_rise``."""
    now = counts_of(scan(root, whitelist, min_confidence))
    if baseline.is_file() and not allow_rise:
        data = _load(baseline)
        problems = rises(_baseline_counts(data, moves or {}), now)
        if data["min_confidence"] != min_confidence:
            problems.append(f"confidence {data['min_confidence']} -> {min_confidence}")
        if problems:
            sys.stdout.write(
                "refusing to record a rise or a change of confidence (use --allow-rise):\n"
                + "\n".join(problems)
                + "\n"
            )
            return 1
    data = {
        "tool": f"vulture {version('vulture')}",
        "min_confidence": min_confidence,
        "files": now,
    }
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"recorded {sum(now.values())} findings in {len(now)} files in {baseline}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run ``check``, ``record`` or ``list``; return the process exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("command", choices=["check", "record", "list"])
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--whitelist", type=Path, default=WHITELIST)
    parser.add_argument("--min-confidence", type=int, default=MIN_CONFIDENCE)
    parser.add_argument("--allow-rise", action="store_true")
    parser.add_argument(
        "--moved", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
        help="a file moved from OLD to NEW; its baseline entry follows it",
    )  # fmt: skip
    args = parser.parse_args(argv)
    root = Path.cwd()
    try:
        moves = parse_moves(args.moved)
        if args.command == "list":
            findings = scan(root, args.whitelist, args.min_confidence)
            sys.stdout.write("".join(f"{f}\n" for f in findings))
            return 0
        if args.command == "record":
            return record(
                root,
                args.baseline,
                args.whitelist,
                args.min_confidence,
                moves,
                allow_rise=args.allow_rise,
            )
        return check(root, args.baseline, args.whitelist, moves)
    except (DeadCodeError, OSError, json.JSONDecodeError, KeyError, ValueError) as err:
        sys.stderr.write(f"dead_code: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
