"""Importing a library module must leave the root logger of the caller as it was; a command configures it in ``main``.

``logging.basicConfig`` at import time reconfigured the logger of every program that merely imported
``tensorpotential.data.datasets`` (or a module that imports it, ``uq`` for one). The commands now call
``configure_logging`` themselves when they run.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.fresh_python import run_fresh_python

MODULES = [
    "tensorpotential.data.datasets",
    "tensorpotential.data.preprocess",
    "tensorpotential.data.weighting",
    "tensorpotential.data.dataset_plotter",
    "tensorpotential.export_kokkos",
    "tensorpotential.model_ops",
    "tensorpotential.cli.data",
    "tensorpotential.cli.train",
    "tensorpotential.cli.prepare",
    "tensorpotential.cli.gracemaker",
    "tensorpotential.scripts._kokkos_export",
    "tensorpotential.scripts.grace_preprocess",
    "tensorpotential.scripts.grace_utils",
    "tensorpotential.scripts.grace_collect",
    "tensorpotential.scripts.grace_predict",
    "tensorpotential.scripts.df2extxyz",
    "tensorpotential.scripts.extxyz2df",
]

COMMANDS = [
    "grace_utils",
    "grace_collect",
    "grace_predict",
    "df2extxyz",
    "extxyz2df",
    "gracemaker",
]

# imports each module in turn and reports the ones after which the root logger differs from before
PROBE = """
import importlib, json, logging, sys
root = logging.getLogger()
before = (list(root.handlers), root.level)
changed = []
for name in sys.argv[1:]:
    importlib.import_module(name)
    if (list(root.handlers), root.level) != before:
        changed.append(name)
        before = (list(root.handlers), root.level)
print(json.dumps(changed))
"""


def _changed_by_import(modules: list[str], cwd: Path, pythonpath=()) -> list[str]:
    result = run_fresh_python(PROBE, cwd, args=modules, pythonpath=pythonpath)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_a_library_module_leaves_the_root_logger_unchanged(tmp_path):
    assert _changed_by_import(MODULES, tmp_path) == []


def test_a_module_that_configures_logging_at_import_is_caught_by_the_probe(tmp_path):
    (tmp_path / "noisy.py").write_text("import logging\nlogging.basicConfig(level=logging.INFO)\n")
    (tmp_path / "quiet.py").write_text("import logging\nlog = logging.getLogger(__name__)\n")
    changed = _changed_by_import(["quiet", "noisy"], tmp_path, pythonpath=[tmp_path])
    assert changed == ["noisy"]


@pytest.mark.parametrize("command", COMMANDS)
def test_a_command_configures_logging_when_its_main_runs(command, monkeypatch):
    module = importlib.import_module(f"tensorpotential.scripts.{command}")
    calls = []
    monkeypatch.setattr(module, "configure_logging", lambda: calls.append(1))
    with pytest.raises(SystemExit):
        module.main(["--help"])
    assert calls == [1]
