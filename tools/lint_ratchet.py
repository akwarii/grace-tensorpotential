"""No-increase ratchet on the ruff and ty findings of legacy code (D13).

Ruff has one configuration, in ``pyproject.toml``. Its legacy ``per-file-ignores`` keep a plain
``ruff check`` quiet on old code (E, F and ERA001 only); the ratchet runs ruff with that one entry removed,
so every file is checked with the whole strict set, and then sorts the findings by the status of the file:

* a file in a **new package** (``NEW_PACKAGES``) or in **neither** the legacy list nor a new package
  (``baselines/legacy_files.txt``, one path per line) must have no finding at all, from ruff or from ty
  (for ty the ``[[tool.ty.overrides]]`` that relax the legacy rules are switched off for such a file);
* a **legacy** file keeps its findings of the counted rules, per (file, rule), and they may not rise.
  ty findings of legacy files, under the overrides, are counted too.

Counted ruff rules: E, F, ERA001, C901, PLR0912, PLR0913, PLR0915, BLE001, B, RUF012, NPY002, SIM, RET, PLW,
PTH and S without S101. Left out on purpose: the D family (a docstring campaign), S101 (asserts in tests),
T201 (user-facing output of the CLI layers), and the other families (import order, pyupgrade and the like:
fixing them means a repository-wide reformat). E501 is off for the whole repository. Counts are keyed by
file and rule, never by line or message, because Stage 0 moves lines; ty's ``unresolved-import`` is dropped
because it depends on the installed environment.

The legacy list is written once by ``record``; later ``record`` runs only remove files that are gone or
follow ``--moved``, never add one, and the baseline stores the count and digest of the list, so an entry
added by hand fails ``check`` (a new file must pass the strict set instead).

Usage::

    python tools/lint_ratchet.py check [--baseline FILE] [--legacy FILE] [--fail-on-drop] [--moved OLD NEW] [--split OLD NEW]
    python tools/lint_ratchet.py record [--baseline FILE] [--legacy FILE] [--allow-rise] [--moved OLD NEW] [--split OLD NEW]
    python tools/lint_ratchet.py versions

``--moved OLD NEW`` (repeatable) re-keys the counts and the legacy status of a moved file before the
comparison. ``--split OLD NEW`` (repeatable) is for a file whose code was partly moved into the new file ``NEW``:
``NEW`` becomes a legacy file too (``record`` adds it to the list) and the counts of ``OLD`` and ``NEW`` together may
not exceed the recorded counts of ``OLD``; ``check`` exits 1 on a rise, a finding in a strict file, a legacy list that differs from the
recorded one, or a version mismatch, and 2 when a tool stopped early (a run that cannot be read is an error,
never zero findings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Callable, Collection
from pathlib import Path

from ratchet_moves import fold_splits, parse_moves, parse_splits, rekey

NEW_PACKAGES = ("tensorpotential/torch_backend", "tensorpotential/core", "tests_torch")
IGNORED_TY_RULES = ("unresolved-import",)
BASELINE = Path("baselines/lint_ratchet.json")
LEGACY_LIST = Path("baselines/legacy_files.txt")
COUNTED_FAMILIES = frozenset({"E", "F", "B", "SIM", "RET", "PLW", "PTH", "S"})
COUNTED_CODES = frozenset({
    "ERA001", "C901", "PLR0912", "PLR0913", "PLR0915", "BLE001", "RUF012", "NPY002",
    "invalid-syntax", "syntax-error",
})  # fmt: skip
UNCOUNTED_CODES = frozenset({"S101"})
LEGACY_HEADER = (
    "# Legacy files: checked by the counts of tools/lint_ratchet.py, not by the strict set.\n"
    "# Written by `python tools/lint_ratchet.py record`; it shrinks and never grows. A file in neither\n"
    "# this list nor a new package must have no finding.\n"
)
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


def _strict_per_file_ignores(root: Path) -> str:
    """TOML inline table of ``per-file-ignores`` in ``pyproject.toml`` without the legacy entry.

    The legacy entry is the negated glob (``!{new packages}/**``) that keeps a plain ``ruff check``
    quiet on old code; the ratchet decides by the legacy list instead.
    """
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = config.get("tool", {}).get("ruff", {}).get("lint", {}).get("per-file-ignores", {})
    kept = {glob: rules for glob, rules in ignores.items() if not glob.startswith("!")}
    return "{" + ", ".join(f"{json.dumps(g)} = {json.dumps(r)}" for g, r in kept.items()) + "}"


def run_ruff(root: Path) -> list[tuple[str, str]]:
    """``(file, code)`` of every ruff finding under ``pyproject.toml`` without its legacy ignores."""
    out = _run(
        [
            _exe("ruff"), "check", "--no-cache", "--output-format", "json",
            "--config", f"lint.per-file-ignores = {_strict_per_file_ignores(root)}", ".",
        ],
        root, (0, 1),
    )  # fmt: skip
    return parse_ruff(out, root)


def parse_ruff(text: str, root: Path) -> list[tuple[str, str]]:
    """``(file, code)`` from ruff's JSON; a finding without a code is a ``syntax-error``."""
    return [
        (_relative(d["filename"], root), d["code"] or "syntax-error")
        for d in _json_list(text, "ruff")
    ]


def lintable_files(root: Path) -> list[str]:
    """Python files and notebooks that ruff checks under ``root``, as sorted relative paths."""
    out = _run([_exe("ruff"), "check", "--no-cache", "--show-files", "."], root, (0, 1))
    paths = (_relative(line, root) for line in out.splitlines() if line.strip())
    return sorted(p for p in paths if p.endswith((".py", ".pyi", ".ipynb")))


def _ty_findings(out: str) -> list[tuple[str, str]]:
    found = [(d["location"]["path"], d["check_name"]) for d in _json_list(out, "ty")]
    return [(path, rule) for path, rule in found if rule not in IGNORED_TY_RULES]


def _strict_ty_config(root: Path) -> str:
    """Text of a ``ty.toml``: the ``[tool.ty]`` tables of ``pyproject.toml`` without ``overrides``.

    ``ty check --config`` cannot clear the overrides of ``pyproject.toml`` (measured with ty 0.0.84:
    the relaxed rules stay off), a configuration file can. A relative interpreter path is made absolute
    because the file is written elsewhere.
    """
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lines: list[str] = []
    for name, table in config.get("tool", {}).get("ty", {}).items():
        if name == "overrides" or not isinstance(table, dict):
            continue
        lines.append(f"[{name}]")
        for key, value in table.items():
            if name == "environment" and key in ("python", "root"):
                lines.append(f"{key} = {json.dumps(str((root / value).resolve()))}")
            else:
                lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def run_ty(root: Path, strict_files: Collection[str] = ()) -> list[tuple[str, str]]:
    """``(file, rule)`` of every ty finding except the environment-dependent ones.

    The ``[[tool.ty.overrides]]`` of ``pyproject.toml`` apply to the whole tree; a file of
    ``strict_files`` is checked again without them, and only that run counts for it.
    """
    out = _run([_exe("ty"), "check", "--output-format", "gitlab"], root, (0, 1))
    strict = set(strict_files)
    found = [f for f in _ty_findings(out) if f[0] not in strict]
    if strict:
        with tempfile.TemporaryDirectory() as scratch:
            config = Path(scratch) / "ty.toml"
            config.write_text(_strict_ty_config(root), encoding="utf-8")
            args = ["check", "--output-format", "gitlab", "--config-file", str(config)]
            found += _ty_findings(_run([_exe("ty"), *args, *sorted(strict)], root, (0, 1)))
    return found


def in_new_package(path: str, packages: tuple[str, ...] = NEW_PACKAGES) -> bool:
    """Whether ``path`` lies below one of the strict packages."""
    return any(path == p or path.startswith(p + "/") for p in packages)


def is_counted(rule: str) -> bool:
    """Whether a ruff rule is one of the counted families for a legacy file."""
    if rule in UNCOUNTED_CODES:
        return False
    family = re.match(r"[A-Z]+", rule)
    return rule in COUNTED_CODES or (
        family is not None and family.group() in COUNTED_FAMILIES
    )


def tally(
    findings: list[tuple[str, str]],
    packages: tuple[str, ...],
    legacy: Collection[str],
    keep: Callable[[str], bool],
) -> tuple[Counts, list[str]]:
    """Legacy counts ``{file: {rule: n}}`` and ``file: rule`` lines for the strict files.

    A strict file is one in a new package or in neither ``packages`` nor ``legacy``; a legacy file
    keeps the findings whose rule ``keep`` accepts.
    """
    counts: dict[str, Counter] = {}
    strict: list[str] = []
    for path, rule in findings:
        if in_new_package(path, packages):
            strict.append(f"{path}: {rule}")
        elif path not in legacy:
            strict.append(f"{path}: {rule} (in no new package and not in {LEGACY_LIST})")
        elif keep(rule):
            counts.setdefault(path, Counter())[rule] += 1
    return {f: dict(sorted(c.items())) for f, c in sorted(counts.items())}, sorted(
        strict
    )


def measure(
    root: Path, legacy: Collection[str], packages: tuple[str, ...] = NEW_PACKAGES
) -> dict:
    """Measure ``{"ruff": counts, "ty": counts, "strict": [...]}`` for the project at ``root``."""
    unlisted = [
        f
        for f in lintable_files(root)
        if f.endswith(".py") and f not in legacy and not in_new_package(f, packages)
    ]
    ruff, strict_ruff = tally(run_ruff(root), packages, legacy, is_counted)
    ty, strict_ty = tally(run_ty(root, unlisted), packages, legacy, lambda _: True)
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


def read_legacy(path: Path) -> list[str]:
    """Paths of the legacy list, without blank and comment lines."""
    if not path.is_file():
        msg = f"no legacy list at {path}; record one with `python tools/lint_ratchet.py record`"
        raise ToolError(msg)
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def legacy_digest(entries: Collection[str]) -> dict[str, int | str]:
    """Count and sha256 of a legacy list, as ``record`` stores them in the baseline."""
    text = "".join(f"{entry}\n" for entry in sorted(entries))
    return {"count": len(entries), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _committed_entries(root: Path, path: Path) -> set[str]:
    """Entries of the legacy list at ``HEAD``; empty when git cannot say."""
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    done = subprocess.run(  # noqa: S603
        ["git", "show", f"HEAD:{rel.as_posix()}"],  # noqa: S607 - git on PATH
        cwd=root, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        return set()
    lines = (line.strip() for line in done.stdout.splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def legacy_problems(
    root: Path, raw: list[str], listed: list[str], recorded: dict | None, path: Path
) -> list[str]:
    """Why the legacy list at ``path`` cannot be trusted: stale, duplicated or grown entries."""
    problems = [
        f"legacy list: {entry} is listed twice"
        for entry, n in sorted(Counter(raw).items())
        if n > 1
    ]
    for entry in sorted(listed):
        if in_new_package(entry):
            problems.append(f"legacy list: {entry} is in a new package")
        elif not (root / entry).is_file():
            problems.append(f"legacy list: {entry} does not exist; run `record` to drop it")
    if recorded is None:
        problems.append("the baseline has no legacy digest; run `record`")
    elif legacy_digest(raw) != recorded:
        added = sorted(set(raw) - _committed_entries(root, path))
        problems.append(
            f"legacy list: {path.name} has {len(raw)} entries against {recorded['count']} recorded "
            "or another digest; only `record` may change it, a new file must pass the strict set"
            + (f" (not in the committed list: {', '.join(added)})" if added else "")
        )
    return problems


def check(
    root: Path,
    baseline: Path,
    legacy_path: Path,
    moves: dict[str, str] | None = None,
    splits: dict[str, str] | None = None,
    *,
    fail_on_drop: bool = False,
) -> int:
    """Print the findings that broke the ratchet and return the exit status."""
    moves = moves or {}
    splits = splits or {}
    base = _load(baseline)
    raw = read_legacy(legacy_path)
    listed = [moves.get(path, path) for path in raw]
    split_problems = _split_problems(listed, splits)
    listed += list(splits)
    now = measure(root, listed)
    problems = [
        *version_problems(root, base["versions"]),
        *legacy_problems(root, raw, listed, base.get("legacy"), legacy_path),
        *split_problems,
    ]
    rises: list[str] = []
    drops: list[str] = []
    for tool in ("ruff", "ty"):
        up, down = compare(rekey(base[tool], moves), fold_splits(now[tool], splits))
        rises += [f"{tool} {line}" for line in up]
        drops += [f"{tool} {line}" for line in down]
    report = [
        *problems,
        *(f"rose: {r}" for r in rises),
        *(f"strict: {s}" for s in now["strict"]),
    ]
    if drops:
        report += [f"dropped: {d}" for d in drops]
        shown = (
            baseline.relative_to(root) if baseline.is_relative_to(root) else baseline
        )
        report.append(
            f"record the lower baseline with: python tools/lint_ratchet.py record --baseline {shown}"
        )
    sys.stdout.write("\n".join(report) + ("\n" if report else "lint ratchet: ok\n"))
    failed = problems or rises or now["strict"] or (fail_on_drop and drops)
    return 1 if failed else 0


def _split_problems(listed: list[str], splits: dict[str, str]) -> list[str]:
    """Why a ``--split`` cannot be taken: its source is not a legacy file, or its target already is one."""
    return [
        f"--split {old} {new}: {old} is not a legacy file" if old not in listed
        else f"--split {old} {new}: {new} is a legacy file already"
        for new, old in sorted(splits.items())
        if old not in listed or new in listed
    ]


def _recorded_list(
    root: Path, legacy_path: Path, moves: dict[str, str], splits: dict[str, str]
) -> list[str]:
    """List the legacy files to record: all files outside the new packages the first time.

    Later the old list minus the files that are gone, with the moves applied; never a file that
    was not on it.
    """
    if not legacy_path.is_file():
        return [f for f in lintable_files(root) if not in_new_package(f)]
    moved = {moves.get(path, path) for path in read_legacy(legacy_path)}
    moved |= set(splits)
    return sorted(path for path in moved if (root / path).is_file())


def _rises(old: dict, now: dict, moves: dict[str, str], splits: dict[str, str]) -> list[str]:
    return [
        r
        for tool in ("ruff", "ty")
        for r in compare(rekey(old[tool], moves), fold_splits(now[tool], splits))[0]
    ]


def record(
    root: Path,
    baseline: Path,
    legacy_path: Path,
    moves: dict[str, str] | None = None,
    splits: dict[str, str] | None = None,
    *,
    allow_rise: bool = False,
) -> int:
    """Write the current counts and the legacy list; refuse to record a rise unless allowed."""
    moves = moves or {}
    splits = splits or {}
    problems = _split_problems(
        sorted({moves.get(p, p) for p in read_legacy(legacy_path)}) if legacy_path.is_file() else [], splits
    )
    if problems:
        sys.stdout.write("\n".join(problems) + "\n")
        return 1
    listed = _recorded_list(root, legacy_path, moves, splits)
    now = measure(root, listed)
    versions = {tool: tool_version(tool, root) for tool in ("ruff", "ty")}
    if baseline.is_file() and not allow_rise:
        rises = _rises(_load(baseline), now, moves, splits)
        if rises:
            sys.stdout.write(
                "refusing to record a rise (use --allow-rise):\n"
                + "\n".join(rises)
                + "\n"
            )
            return 1
    if now["strict"]:
        sys.stdout.write(
            "findings in the new packages and in files outside the legacy list cannot be baselined:\n"
            + "\n".join(now["strict"])
            + "\n"
        )
        return 1
    data = {
        "versions": versions,
        "ruff": now["ruff"],
        "ty": now["ty"],
        "legacy": legacy_digest(listed),
    }
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        LEGACY_HEADER + "".join(f"{path}\n" for path in listed), encoding="utf-8"
    )
    total = sum(sum(c.values()) for tool in ("ruff", "ty") for c in now[tool].values())
    sys.stdout.write(
        f"recorded {total} findings in {baseline} and {len(listed)} legacy files in {legacy_path}\n"
    )
    return 0


def _dispatch(
    args: argparse.Namespace, root: Path, baseline: Path, legacy_path: Path
) -> int:
    if args.command in ("check", "record"):
        try:
            moves = parse_moves(args.moved)
            splits = parse_splits(args.split)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if args.command == "check":
            return check(
                root, baseline, legacy_path, moves, splits, fail_on_drop=args.fail_on_drop
            )
        return record(
            root, baseline, legacy_path, moves, splits, allow_rise=args.allow_rise
        )
    problems = version_problems(root)
    sys.stdout.write("\n".join(problems) + "\n" if problems else "versions: ok\n")
    return 1 if problems else 0


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "record", "versions"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--baseline", type=Path, default=BASELINE)
        cmd.add_argument("--root", type=Path, default=Path())
    for name in ("check", "record"):
        sub.choices[name].add_argument("--legacy", type=Path, default=LEGACY_LIST)
        sub.choices[name].add_argument(
            "--moved", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
            help="a file moved from OLD to NEW; its counts and legacy status follow it",
        )  # fmt: skip
        sub.choices[name].add_argument(
            "--split", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
            help="part of OLD moved into the new file NEW; both are legacy, their counts together may not rise",
        )  # fmt: skip
    sub.choices["check"].add_argument("--fail-on-drop", action="store_true")
    sub.choices["record"].add_argument("--allow-rise", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    legacy = _resolve(getattr(args, "legacy", LEGACY_LIST), root)
    try:
        status = _dispatch(args, root, _resolve(args.baseline, root), legacy)
    except ToolError as exc:
        sys.stderr.write(f"lint_ratchet: {exc}\n")
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
