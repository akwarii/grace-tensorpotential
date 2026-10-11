"""Resolve-only check of the optional dependencies (extras) of ``pyproject.toml``.

For each extra, ``uv pip compile`` resolves the project with that extra for a Python version and the check
fails when

* the extra lists nothing (an empty list, or one that only names the project itself), or
* a requirement the extra names is missing from the resolved set (a typo, a marker that never matches).

Nothing is installed or downloaded beyond the package metadata that resolution reads::

    python tools/resolve_extras.py [--python-version 3.11] [--python-platform linux] [EXTRA ...]

The extras to check default to all of them. The names are the project names after normalisation
(``torch_sim`` and ``torch-sim`` are one name); ``tensorpotential[tf]`` style references to the project itself are
expanded to the extras they name, so ``all`` is checked against the requirements of ``tf``, ``torch`` and
``torch-sim``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 600


def normalise(name: str) -> str:
    """The canonical form of a project name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The normalised project name of a requirement string (extras, versions and markers dropped)."""
    return normalise(re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0])


def extra_requirements(
    pyproject: dict, extra: str, _seen: frozenset[str] = frozenset()
) -> set[str]:
    """The project names an extra asks for, with references to the project itself expanded.

    Parameters
    ----------
    pyproject
        The parsed ``pyproject.toml``.
    extra
        The name of an extra of ``[project.optional-dependencies]``.

    Raises
    ------
    KeyError
        If the extra is not declared (or a self-reference names one that is not).
    """
    project = pyproject["project"]
    own = normalise(project["name"])
    declared = project.get("optional-dependencies", {})
    if extra not in declared:
        msg = f"extra {extra!r} is not declared; the extras are: {sorted(declared)}"
        raise KeyError(msg)
    names: set[str] = set()
    for requirement in declared[extra]:
        name = requirement_name(requirement)
        if name != own:
            names.add(name)
            continue
        for inner in re.findall(r"\[(.*?)\]", requirement):
            for sub in filter(None, (part.strip() for part in inner.split(","))):
                if sub not in _seen:
                    names |= extra_requirements(pyproject, sub, _seen | {extra})
    return names


def resolution_problems(extra: str, wanted: set[str], resolved: set[str]) -> list[str]:
    """Problems of one extra: nothing listed, or listed names that are not in the resolution."""
    if not wanted:
        return [f"extra {extra!r} lists no requirement"]
    missing = sorted(wanted - resolved)
    return (
        [f"extra {extra!r}: {', '.join(missing)} missing from the resolved set"]
        if missing
        else []
    )


def resolve(
    extra: str, python_version: str, python_platform: str, pyproject: Path
) -> set[str]:
    """Resolve the project with ``extra`` through ``uv pip compile`` and return the names it pins."""
    command = [
        "uv", "pip", "compile", str(pyproject), "--extra", extra, "--python-version", python_version,
        "--python-platform", python_platform, "--no-header", "--no-annotate", "--quiet",
    ]  # fmt: skip
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
    )
    if done.returncode != 0:
        msg = f"uv could not resolve extra {extra!r}:\n{done.stderr.strip()}"
        raise RuntimeError(msg)
    return {requirement_name(line) for line in done.stdout.splitlines() if "==" in line}


def main(argv: list[str] | None = None) -> int:
    """Check the extras named on the command line (default: all); return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("extras", nargs="*", help="extras to check (default: all)")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--python-platform", default="linux")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args(argv)

    pyproject = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    extras = args.extras or sorted(
        pyproject["project"].get("optional-dependencies", {})
    )
    if not extras:
        print("the project declares no extras", file=sys.stderr)
        return 1
    problems: list[str] = []
    for extra in extras:
        wanted = extra_requirements(pyproject, extra)
        if not wanted:
            problems += resolution_problems(extra, wanted, set())
            continue
        resolved = resolve(
            extra, args.python_version, args.python_platform, args.pyproject
        )
        found = resolution_problems(extra, wanted, resolved)
        print(
            f"{'ok  ' if not found else 'FAIL'} {extra}: {len(wanted)} requirement(s), {len(resolved)} packages resolved"
        )
        problems += found
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
