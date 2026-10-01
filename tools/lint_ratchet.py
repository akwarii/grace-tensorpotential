"""No-increase ratchet on the ruff and ty findings of legacy code (D13).

Legacy code keeps the root ruff config (E, F) and the ty overrides of ``pyproject.toml``; the findings
that remain, plus ERA001, F401, F841 and F811, are counted per (file, rule) and may not rise. The
strict sets of the new packages have no baseline: any finding there fails. Counts are keyed by file and
rule, never by line or message, because Stage 0 moves lines; ty's ``unresolved-import`` is dropped
because it depends on the installed environment.

Usage::

    python tools/lint_ratchet.py check [--baseline FILE] [--fail-on-drop]
    python tools/lint_ratchet.py record [--baseline FILE] [--allow-rise]
    python tools/lint_ratchet.py versions

``check`` exits 1 on a rise, a finding in a new package, or a version mismatch, and 2 when a tool
stopped early (a run that cannot be read is an error, never zero findings).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

NEW_PACKAGES = ("tensorpotential/torch_backend", "tensorpotential/core", "tests_torch")
RATCHET_RULES = ("ERA001", "F401", "F841", "F811")
IGNORED_TY_RULES = ("unresolved-import",)
BASELINE = Path("baselines/lint_ratchet.json")
VERSION_FILES = (
    "pyproject.toml",
    ".pre-commit-config.yaml",
    "CLAUDE.md",
    ".claude/skills",
)
_VERSION_PATTERN = {
    "ruff": re.compile(r"\bruff(?:@|==|\s)(\d+\.\d+\.\d+)"),
    "ty": re.compile(r"\bty(?:@|==)(\d+\.\d+\.\d+)"),
}

Counts = dict[str, dict[str, int]]


class ToolError(RuntimeError):
    """A linter stopped early or printed something unreadable."""


def _exe(name: str) -> str:
    """Path of ``name`` in the interpreter's environment, else on ``PATH``."""
    local = Path(sys.executable).parent / name
    found = str(local) if local.exists() else shutil.which(name)
    if found is None:
        msg = f"{name} is not installed; run `uv sync --group dev`"
        raise ToolError(msg)
    return found


def _run(args: list[str], cwd: Path, ok_codes: tuple[int, ...]) -> str:
    done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode not in ok_codes:
        msg = f"{Path(args[0]).name} exited with {done.returncode}: {done.stderr.strip()[:500]}"
        raise ToolError(msg)
    return done.stdout


def tool_version(name: str, cwd: Path) -> str:
    """Installed version of ``ruff`` or ``ty``."""
    text = _run([_exe(name), "--version"], cwd, (0,))
    match = re.search(r"\d+\.\d+\.\d+", text)
    if match is None:
        msg = f"cannot read the version of {name} from {text!r}"
        raise ToolError(msg)
    return match.group(0)


def _json_list(text: str, source: str) -> list[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"{source} printed unreadable JSON: {exc}"
        raise ToolError(msg) from exc
    if not isinstance(data, list):
        msg = f"{source} printed {type(data).__name__}, expected a list"
        raise ToolError(msg)
    return data


def _relative(path: str, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def run_ruff(root: Path) -> list[tuple[str, str]]:
    """``(file, code)`` of every ruff finding: the project config plus the ratchet rules."""
    out = _run(
        [_exe("ruff"), "check", "--no-cache", "--output-format", "json",
         "--extend-select", ",".join(RATCHET_RULES), "."],
        root, (0, 1),
    )  # fmt: skip
    return [
        (_relative(d["filename"], root), d["code"] or "syntax-error")
        for d in _json_list(out, "ruff")
    ]


def run_ty(root: Path) -> list[tuple[str, str]]:
    """``(file, rule)`` of every ty finding except the environment-dependent ones."""
    out = _run([_exe("ty"), "check", "--output-format", "gitlab"], root, (0, 1))
    found = [(d["location"]["path"], d["check_name"]) for d in _json_list(out, "ty")]
    return [(path, rule) for path, rule in found if rule not in IGNORED_TY_RULES]


def in_new_package(path: str, packages: tuple[str, ...] = NEW_PACKAGES) -> bool:
    """Whether ``path`` lies below one of the strict packages."""
    return any(path == p or path.startswith(p + "/") for p in packages)


def tally(
    findings: list[tuple[str, str]], packages: tuple[str, ...]
) -> tuple[Counts, list[str]]:
    """Legacy counts ``{file: {rule: n}}`` and ``file: rule`` lines for the new packages."""
    legacy: dict[str, Counter] = {}
    strict: list[str] = []
    for path, rule in findings:
        if in_new_package(path, packages):
            strict.append(f"{path}: {rule}")
        else:
            legacy.setdefault(path, Counter())[rule] += 1
    return {f: dict(sorted(c.items())) for f, c in sorted(legacy.items())}, sorted(
        strict
    )


def measure(root: Path, packages: tuple[str, ...] = NEW_PACKAGES) -> dict:
    """Measure ``{"ruff": counts, "ty": counts, "strict": [...]}`` for the project at ``root``."""
    ruff, strict_ruff = tally(run_ruff(root), packages)
    ty, strict_ty = tally(run_ty(root), packages)
    return {"ruff": ruff, "ty": ty, "strict": sorted(strict_ruff + strict_ty)}


def _flat(counts: Counts) -> dict[tuple[str, str], int]:
    return {(f, r): n for f, rules in counts.items() for r, n in rules.items()}


def compare(base: Counts, new: Counts) -> tuple[list[str], list[str]]:
    """Lines for the counts that rose and for those that dropped, ``file rule: old -> new``."""
    old, cur = _flat(base), _flat(new)
    rises, drops = [], []
    for key in sorted(old.keys() | cur.keys()):
        before, after = old.get(key, 0), cur.get(key, 0)
        line = f"{key[0]} {key[1]}: {before} -> {after}"
        if after > before:
            rises.append(line)
        elif after < before:
            drops.append(line)
    return rises, drops


def declared_versions(root: Path) -> dict[str, set[str]]:
    """Versions of ruff and ty named in the pins, the hook config and the docs."""
    files: list[Path] = []
    for name in VERSION_FILES:
        path = root / name
        files += sorted(path.rglob("*.md")) if path.is_dir() else [path]
    found: dict[str, set[str]] = {"ruff": set(), "ty": set()}
    for path in files:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for tool, pattern in _VERSION_PATTERN.items():
                found[tool].update(pattern.findall(text))
    return found


def version_problems(root: Path, recorded: dict[str, str] | None = None) -> list[str]:
    """Mismatches between installed, declared and recorded versions of ruff and ty."""
    declared = declared_versions(root)
    problems = []
    for tool in ("ruff", "ty"):
        installed = tool_version(tool, root)
        versions = (
            declared[tool] | {installed} | ({recorded[tool]} if recorded else set())
        )
        if len(versions) > 1:
            problems.append(
                f"{tool}: installed {installed}, named as {sorted(versions - {installed})}"
            )
    return problems


def _load(path: Path) -> dict:
    if not path.is_file():
        msg = f"no baseline at {path}; record one with `python tools/lint_ratchet.py record`"
        raise ToolError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def check(root: Path, baseline: Path, *, fail_on_drop: bool = False) -> int:
    """Print the findings that broke the ratchet and return the exit status."""
    base = _load(baseline)
    now = measure(root)
    problems = version_problems(root, base["versions"])
    rises: list[str] = []
    drops: list[str] = []
    for tool in ("ruff", "ty"):
        up, down = compare(base[tool], now[tool])
        rises += [f"{tool} {line}" for line in up]
        drops += [f"{tool} {line}" for line in down]
    report = [
        *problems,
        *(f"rose: {r}" for r in rises),
        *(f"strict: {s}" for s in now["strict"]),
    ]
    if drops:
        report += [f"dropped: {d}" for d in drops]
        report.append(
            f"record the lower baseline with: python tools/lint_ratchet.py record --baseline {baseline}"
        )
    sys.stdout.write("\n".join(report) + ("\n" if report else "lint ratchet: ok\n"))
    failed = problems or rises or now["strict"] or (fail_on_drop and drops)
    return 1 if failed else 0


def record(root: Path, baseline: Path, *, allow_rise: bool = False) -> int:
    """Write the current counts as the baseline; refuse to record a rise unless allowed."""
    now = measure(root)
    versions = {tool: tool_version(tool, root) for tool in ("ruff", "ty")}
    if baseline.is_file() and not allow_rise:
        old = _load(baseline)
        rises = [r for tool in ("ruff", "ty") for r in compare(old[tool], now[tool])[0]]
        if rises:
            sys.stdout.write(
                "refusing to record a rise (use --allow-rise):\n"
                + "\n".join(rises)
                + "\n"
            )
            return 1
    if now["strict"]:
        sys.stdout.write(
            "findings in the new packages cannot be baselined:\n"
            + "\n".join(now["strict"])
            + "\n"
        )
        return 1
    data = {"versions": versions, "ruff": now["ruff"], "ty": now["ty"]}
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = sum(sum(c.values()) for tool in ("ruff", "ty") for c in now[tool].values())
    sys.stdout.write(f"recorded {total} findings in {baseline}\n")
    return 0


def _dispatch(args: argparse.Namespace, root: Path, baseline: Path) -> int:
    if args.command == "check":
        return check(root, baseline, fail_on_drop=args.fail_on_drop)
    if args.command == "record":
        return record(root, baseline, allow_rise=args.allow_rise)
    problems = version_problems(root)
    sys.stdout.write("\n".join(problems) + "\n" if problems else "versions: ok\n")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "record", "versions"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--baseline", type=Path, default=BASELINE)
        cmd.add_argument("--root", type=Path, default=Path())
    sub.choices["check"].add_argument("--fail-on-drop", action="store_true")
    sub.choices["record"].add_argument("--allow-rise", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    baseline = args.baseline if args.baseline.is_absolute() else root / args.baseline
    try:
        status = _dispatch(args, root, baseline)
    except ToolError as exc:
        sys.stderr.write(f"lint_ratchet: {exc}\n")
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
