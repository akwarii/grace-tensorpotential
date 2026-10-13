"""Library code never imports the command layers: ``tensorpotential.cli`` and ``tensorpotential.scripts``.

The contract is in ``[tool.importlinter]`` of ``pyproject.toml`` (``lint-imports`` runs it in CI). These tests check that
it names every library module of the package, and that it fails on a library module that imports ``cli``.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "tensorpotential"
CONTRACT_NAME = "Library code does not import the command layers (cli, scripts)"
# not library: the command layers, the legacy converter kept as it is, data files and the package marker
NOT_SOURCES = {"cli", "scripts", "compat", "resources", "__init__", "_tf_options", "__pycache__"}


def _contract() -> dict:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contracts = config["tool"]["importlinter"]["contracts"]
    return next(c for c in contracts if c["name"] == CONTRACT_NAME)


def test_the_contract_names_every_library_module_of_the_package():
    top_level = {p.stem for p in PACKAGE.iterdir() if p.suffix == ".py" or p.is_dir()}
    expected = {f"tensorpotential.{name}" for name in top_level - NOT_SOURCES}
    assert set(_contract()["source_modules"]) == expected


def test_the_contract_forbids_both_command_layers():
    assert set(_contract()["forbidden_modules"]) == {"tensorpotential.cli", "tensorpotential.scripts"}


def _lint(project: Path) -> subprocess.CompletedProcess:
    code = "import sys; from importlinter.cli import lint_imports_command; sys.exit(lint_imports_command())"
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], cwd=project, capture_output=True, text=True, check=False
    )


def _scratch_project(tmp_path: Path, library_source: str) -> Path:
    """A package ``pkg`` with the contract of the repository, one library module and a ``cli`` module."""
    package = tmp_path / "pkg"
    (package / "cli").mkdir(parents=True)
    (package / "scripts").mkdir()
    for init in (package / "__init__.py", package / "cli" / "__init__.py", package / "scripts" / "__init__.py"):
        init.write_text("")
    (package / "cli" / "tool.py").write_text("VALUE = 1\n")
    (package / "lib.py").write_text(library_source)
    contract = _contract()
    (tmp_path / ".importlinter").write_text(
        "[importlinter]\nroot_package = pkg\n\n[importlinter:contract:direction]\n"
        f"name = {contract['name']}\ntype = forbidden\nsource_modules = pkg.lib\n"
        "forbidden_modules =\n    pkg.cli\n    pkg.scripts\n"
    )
    return tmp_path


def test_a_library_module_that_imports_cli_breaks_the_contract(tmp_path):
    project = _scratch_project(tmp_path, "from pkg.cli import tool\n")
    result = _lint(project)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout


def test_a_library_module_that_imports_cli_inside_a_function_breaks_the_contract(tmp_path):
    project = _scratch_project(tmp_path, "def f():\n    from pkg.scripts import grace\n    return grace\n")
    (project / "pkg" / "scripts" / "grace.py").write_text("VALUE = 1\n")
    assert _lint(project).returncode != 0


def test_a_library_module_that_imports_nothing_from_the_command_layers_keeps_it(tmp_path):
    project = _scratch_project(tmp_path, "VALUE = 2\n")
    result = _lint(project)
    assert result.returncode == 0, result.stdout


LOWER_LAYERS_CONTRACT = "Instructions and data do not import the model layers"
FOUNDATIONS_CONTRACT = "Foundation modules import nothing above them"


def _named_contract(name: str) -> dict:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return next(c for c in config["tool"]["importlinter"]["contracts"] if c["name"] == name)


def _module_exists(dotted: str) -> bool:
    relative = Path(*dotted.split(".")[1:])
    return (PACKAGE / relative).is_dir() or (PACKAGE / relative).with_suffix(".py").is_file()


def _layer_scratch_project(tmp_path: Path, contract_name: str, source: str, importer: str, target: str) -> Path:
    """A package ``pkg`` with the modules the contract names; ``importer`` imports ``target`` (both dotted below ``pkg``)."""
    contract = _named_contract(contract_name)
    package = tmp_path / "pkg"
    names = {m.removeprefix("tensorpotential.") for m in contract["source_modules"] + contract["forbidden_modules"]}
    names.add(target)
    for name in names:
        module = package / f"{name}.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("VALUE = 1\n")
    (package / "__init__.py").write_text("")
    (package / f"{importer}.py").write_text(f"from pkg import {target}\n" if source == "from" else f"import pkg.{target}\n")
    config = (
        f"[importlinter]\nroot_package = pkg\n\n[importlinter:contract:layers]\nname = {contract_name}\n"
        f"type = forbidden\nallow_indirect_imports = {str(contract.get('allow_indirect_imports', False)).lower()}\n"
        "source_modules =\n"
        + "".join(f"    {m.replace('tensorpotential', 'pkg')}\n" for m in contract["source_modules"])
        + "forbidden_modules =\n"
        + "".join(f"    {m.replace('tensorpotential', 'pkg')}\n" for m in contract["forbidden_modules"])
    )
    (tmp_path / ".importlinter").write_text(config)
    return tmp_path


def test_the_layer_contracts_name_only_modules_that_exist():
    for name in (LOWER_LAYERS_CONTRACT, FOUNDATIONS_CONTRACT):
        contract = _named_contract(name)
        missing = [m for m in contract["source_modules"] + contract["forbidden_modules"] if not _module_exists(m)]
        assert not missing, f"{name}: {missing}"


def test_data_importing_calculator_breaks_the_lower_layers_contract(tmp_path):
    project = _layer_scratch_project(tmp_path, LOWER_LAYERS_CONTRACT, "from", "data", "calculator")
    result = _lint(project)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout


def test_core_importing_instructions_breaks_the_foundations_contract(tmp_path):
    project = _layer_scratch_project(tmp_path, FOUNDATIONS_CONTRACT, "import", "core", "instructions")
    assert _lint(project).returncode != 0


def test_data_importing_a_foundation_keeps_the_lower_layers_contract(tmp_path):
    project = _layer_scratch_project(tmp_path, LOWER_LAYERS_CONTRACT, "from", "data", "constants")
    result = _lint(project)
    assert result.returncode == 0, result.stdout
