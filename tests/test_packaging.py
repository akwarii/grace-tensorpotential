"""Tests for the packaging of ``tensorpotential``: ``pyproject.toml`` and the wheel built from this tree.

The wheel is built once per session from a copy of the sources in a scratch directory (nothing is written
into the tree), or taken from the file named by ``TENSORPOTENTIAL_WHEEL`` when CI built it already.

Logic layer (``structure_problems`` on a real wheel): the top-level names are the package and its
``dist-info``, no ``tests`` package, ``resources/input_template.yaml`` is present, every tracked module of
the package is in the wheel, and the file list differs from the wheel built from upstream only by the
files this work removed. Metadata layer (``METADATA`` of the wheel): the base requirements have no
TensorFlow, the extras ``tf``, ``torch``, ``torch-sim`` and ``all`` exist, Python 3.11 or newer. A planted
mutant (package discovery without the ``tensorpotential*`` scope, which also returns the tracked ``tests``
package) must be reported by ``structure_problems``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WHEEL_ENV = "TENSORPOTENTIAL_WHEEL"
UPSTREAM_FILES = REPO_ROOT / "baselines" / "wheel_files_upstream.txt"
# Files of the upstream wheel that this tree no longer has (name the issue that removed each).
REMOVED_SINCE_UPSTREAM = {
    "tensorpotential/data/symbols.py",  # CLEAN3: unused module removed
}
BUILD_TIMEOUT_S = 600


def copy_sources(destination: Path) -> None:
    """Copy what a wheel is built from (package, ``pyproject.toml``, readme, licence) into ``destination``."""
    shutil.copytree(
        REPO_ROOT / "tensorpotential",
        destination / "tensorpotential",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for pattern in ("pyproject.toml", "README*", "LICENSE*"):
        for path in REPO_ROOT.glob(pattern):
            shutil.copy(path, destination / path.name)


def build_wheel(source: Path, out_dir: Path) -> Path:
    """Build the wheel of ``source`` with ``uv build`` (or ``python -m build``) and return its path."""
    if shutil.which("uv"):
        command = ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(source)]
    else:
        command = [
            "python",
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(out_dir),
            str(source),
        ]
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S, check=False
    )
    if done.returncode != 0:
        pytest.fail(
            f"building the wheel failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        )
    (wheel,) = out_dir.glob("*.whl")
    return wheel


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The wheel of this tree: the file named by ``TENSORPOTENTIAL_WHEEL``, else one built here."""
    given = os.environ.get(WHEEL_ENV)
    if given:
        return Path(given)
    work = tmp_path_factory.mktemp("wheel")
    copy_sources(work / "src")
    return build_wheel(work / "src", work / "dist")


def wheel_names(path: Path) -> list[str]:
    """The names of the files in the wheel."""
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def wheel_metadata(path: Path) -> object:
    """The parsed ``METADATA`` of the wheel."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return Parser().parsestr(archive.read(name).decode())


def tracked_modules() -> set[str]:
    """The ``.py`` files of ``tensorpotential/`` that git tracks (or the directory holds, outside a checkout)."""
    done = subprocess.run(
        ["git", "ls-files", "tensorpotential"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    names = done.stdout.split() if done.returncode == 0 else []
    if not names:
        names = [
            str(p.relative_to(REPO_ROOT))
            for p in (REPO_ROOT / "tensorpotential").rglob("*.py")
        ]
    return {n for n in names if n.endswith(".py")}


def structure_problems(path: Path) -> list[str]:
    """What is wrong with the file list of a wheel of this tree, as messages (empty when it is right)."""
    names = wheel_names(path)
    problems = []
    top = {n.split("/")[0] for n in names}
    expected_top = {"tensorpotential"} | {
        n for n in top if re.fullmatch(r"tensorpotential-.*\.dist-info", n)
    }
    if top != expected_top or len(top) != 2:
        problems.append(
            f"top-level names {sorted(top)}, expected the package and its dist-info only"
        )
    if "tensorpotential/resources/input_template.yaml" not in names:
        problems.append("resources/input_template.yaml is not in the wheel")
    in_package = {
        n for n in names if n.startswith("tensorpotential/") and n.endswith(".py")
    }
    if missing := tracked_modules() - in_package:
        problems.append(
            f"tracked modules missing from the wheel: {sorted(missing)[:5]}"
        )
    upstream = {n for n in UPSTREAM_FILES.read_text().split() if "dist-info" not in n}
    ours = {n for n in names if "dist-info" not in n}
    if removed := upstream - ours - REMOVED_SINCE_UPSTREAM:
        problems.append(f"files of the upstream wheel that are gone: {sorted(removed)}")
    if stray := {n for n in ours - upstream if not n.startswith("tensorpotential/")}:
        problems.append(
            f"files outside tensorpotential/ that upstream does not ship: {sorted(stray)[:5]}"
        )
    return problems


def test_the_wheel_structure_is_right(wheel: Path) -> None:
    assert structure_problems(wheel) == []


def test_the_wheel_has_no_tests_package(wheel: Path) -> None:
    assert not [
        n
        for n in wheel_names(wheel)
        if n.split("/")[0] in {"tests", "tests_torch", "tools"}
    ]


def test_a_package_discovery_without_the_scope_is_reported(tmp_path: Path) -> None:
    copy_sources(tmp_path / "src")
    config = tmp_path / "src" / "pyproject.toml"
    scoped = config.read_text()
    unscoped = scoped.replace('include = ["tensorpotential*"]', 'include = ["*"]')
    assert unscoped != scoped
    config.write_text(unscoped)
    (
        tmp_path / "src" / "tests"
    ).mkdir()  # a tracked package outside tensorpotential, as in the repository
    (tmp_path / "src" / "tests" / "__init__.py").write_text("")
    mutant = build_wheel(tmp_path / "src", tmp_path / "dist")
    assert any("top-level names" in p for p in structure_problems(mutant))


def requirement_name(requirement: str) -> str:
    """The normalised project name of a ``Requires-Dist`` line."""
    return re.split(r"[<>=!;\[ ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def requirements_of(requires: list[str], extra: str | None) -> set[str]:
    """The project names required by the base installation (``extra=None``) or by one extra."""
    marker = re.compile(r"""extra == ["'](.+?)["']""")
    names = set()
    for requirement in requires:
        found = marker.search(requirement)
        if (found.group(1) if found else None) == extra:
            names.add(requirement_name(requirement))
    return names


def test_the_base_requirements_have_no_tensorflow_and_the_extras_are_declared(
    wheel: Path,
) -> None:
    metadata = wheel_metadata(wheel)
    requires = metadata.get_all("Requires-Dist")
    base = requirements_of(requires, None)
    assert {"scipy", "numpy", "pandas", "ase", "matscipy", "sympy"} <= base
    assert not base & {
        "tensorflow",
        "tf-keras",
        "torch",
        "torch-sim-atomistic",
        "vesin",
    }
    assert set(metadata.get_all("Provides-Extra")) == {
        "tf",
        "torch",
        "torch-sim",
        "all",
    }
    assert requirements_of(requires, "tf") == {"tensorflow", "tf-keras"}
    assert requirements_of(requires, "torch") == {"torch"}
    assert requirements_of(requires, "torch-sim") == {
        "tensorpotential",
        "torch-sim-atomistic",
        "vesin",
    }
    assert requirements_of(requires, "all") == {"tensorpotential"}
    assert metadata["Requires-Python"] == ">=3.11"
