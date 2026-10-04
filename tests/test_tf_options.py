"""Tests for ``tensorpotential/_tf_options.py``, the hook that applies the TensorFlow options.

Importing the hook applies numpy-style promotion and turns TensorFloat-32 off, once per process.
Every TensorFlow-side module imports it before its own first ``import tensorflow``, so the options
are in place however a user enters the package.

Logic layer: the hook runs once however many modules import it, and a scan of the source (the
hook is imported before the first TensorFlow import of each module). Behaviour layer: entering the
package through a TensorFlow-side module other than the model classes still gives a float32
matrix product that agrees with the float64 one to float32 rounding, which TensorFloat-32 would not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fresh_python import REPO_ROOT, run_fresh_python

PACKAGE = REPO_ROOT / "tensorpotential"
TF_ROOTS = {"tensorflow", "tf_keras"}
# compat/pace is legacy code that this work may not change (owner decision, 2026-10-01).
OUT_OF_SCOPE = (PACKAGE / "compat" / "pace",)


def imports_a_tf_root(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in TF_ROOTS for alias in node.names)
    return (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] in TF_ROOTS
    )


def imports_the_hook(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.ImportFrom)
        and node.module == "tensorpotential"
        and any(alias.name == "_tf_options" for alias in node.names)
    )


def modules_with_a_toplevel_tf_import() -> list[Path]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if (
            any(path.is_relative_to(skipped) for skipped in OUT_OF_SCOPE)
            or path.name == "__init__.py"
        ):
            continue
        if any(
            imports_a_tf_root(node)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
        ):
            found.append(path)
    return found


MODULES = modules_with_a_toplevel_tf_import()


def test_the_scan_finds_the_tensorflow_side_modules() -> None:
    names = {p.relative_to(PACKAGE).as_posix() for p in MODULES}
    assert {
        "tpmodel.py",
        "tensorpot.py",
        "utils.py",
        "instructions/base.py",
        "functions/nn.py",
    } <= names
    assert "constants.py" not in names
    assert "core/cutoffs.py" not in names


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda p: p.relative_to(PACKAGE).as_posix()
)
def test_the_hook_is_imported_before_the_first_tensorflow_import(path: Path) -> None:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    first_tf = next(i for i, node in enumerate(body) if imports_a_tf_root(node))
    assert any(imports_the_hook(node) for node in body[:first_tf]), (
        f"{path.relative_to(REPO_ROOT)}: add `from tensorpotential import _tf_options  # noqa: F401` "
        "before the first TensorFlow import"
    )


def test_the_hook_runs_once_however_many_modules_import_it(tmp_path: Path) -> None:
    code = "import tensorpotential.tpmodel, tensorpotential.utils, tensorpotential.functions.nn"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("tf.experimental.numpy behavior enabled") == 1
    assert result.stdout.count("TensorFloat-32 execution disabled") == 1


@pytest.mark.parametrize(
    "module",
    [
        "tensorpotential.functions.nn",
        "tensorpotential.metrics",
        "tensorpotential.data.streaming",
    ],
)
def test_entering_through_any_tensorflow_side_module_applies_the_options(
    module: str, tmp_path: Path
) -> None:
    """A float32 product with TensorFloat-32 on is off by about 1e-3 (10 mantissa bits); the bound
    1e-4 is two orders above float32 rounding of a 64-term product and an order below TensorFloat-32."""
    code = f"""
import {module}
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
a = rng.standard_normal((64, 64)).astype(np.float32)
b = rng.standard_normal((64, 64)).astype(np.float32)
err = np.abs(tf.matmul(a, b).numpy().astype(np.float64) - a.astype(np.float64) @ b.astype(np.float64)).max()
print(tf.config.experimental.tensor_float_32_execution_enabled(), hasattr(tf.constant(1.0), "ndim"), err < 1e-4)
"""
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False True True"
