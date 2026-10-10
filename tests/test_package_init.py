"""Characterization tests for ``tensorpotential/__init__.py``.

Importing the package neither sets nor reads the legacy-Keras variable (it warns a user who still exports
it), a TensorFlow-side import switches TensorFlow to numpy-style type promotion and turns TensorFloat-32
off, and the package offers
``TensorPotential``, ``TPModel``, ``LossFunction`` and ``L2Loss``.

Logic layer: the variable in every state of the environment, the public names. Physical-value
layer: TensorFloat-32 off means a float32 matrix product is evaluated to full float32
precision, checked against the same product evaluated in float64 (an oracle that does not use
the package); numpy-style promotion is what the model code relies on (``Tensor.ndim`` exists).
Import-time behaviour is checked in a fresh interpreter, because a second import is a no-op.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

import tensorpotential
from tests.fresh_python import last_stdout_line, run_fresh_python

FLAG = "TF_USE_LEGACY_KERAS"


# ---- the legacy-Keras variable: neither set nor read, a warning when the user still exports it ----


FLAG_PROBE = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import tensorpotential
import os
print(repr(os.environ.get("TF_USE_LEGACY_KERAS")), [str(w.message) for w in caught])
"""


@pytest.mark.parametrize("before", [None, "", "0", "false"])
def test_the_package_leaves_the_variable_alone_and_is_silent_when_it_does_not_ask_for_legacy_keras(
    before: str | None, tmp_path: Path
) -> None:
    assert last_stdout_line(FLAG_PROBE, tmp_path, {FLAG: before}) == f"{before!r} []"


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_a_user_who_still_exports_the_variable_gets_a_warning_and_keeps_the_value(
    value: str, tmp_path: Path
) -> None:
    line = last_stdout_line(FLAG_PROBE, tmp_path, {FLAG: value})
    assert line.startswith(f"{value!r} [")
    assert f"{FLAG}='{value}' is set" in line
    assert f"unset {FLAG}" in line


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_the_warning_is_a_runtime_warning(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, value)
    with pytest.warns(RuntimeWarning, match="Keras 3"):
        tensorpotential._warn_if_legacy_keras_is_requested()
    assert os.environ[FLAG] == value


@pytest.mark.parametrize("value", [None, "", "0", "false", "False"])
def test_no_warning_when_the_variable_does_not_ask_for_legacy_keras(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensorpotential._warn_if_legacy_keras_is_requested()


ENVIRONMENT_PROBE = """
import os
before = dict(os.environ)
import tensorpotential
import tensorpotential.tpmodel  # a TensorFlow-side import too: it configures TensorFlow without touching os.environ
print(before == dict(os.environ))
"""


def test_importing_the_package_leaves_the_environment_untouched(tmp_path: Path) -> None:
    assert last_stdout_line(ENVIRONMENT_PROBE, tmp_path, {FLAG: None}) == "True"
    assert last_stdout_line(ENVIRONMENT_PROBE, tmp_path, {FLAG: "0"}) == "True"


KERAS_SIDE_CHECK = """
import sys
import tensorpotential.tpmodel
import tensorflow as tf
print(tf.keras.optimizers.Adam.__module__.split(".")[0], sys.modules["keras"].__version__.split(".")[0])
"""


def test_with_the_variable_unset_tensorflow_uses_keras_3(
    tmp_path: Path,
) -> None:
    assert last_stdout_line(KERAS_SIDE_CHECK, tmp_path, {FLAG: None}) == "keras 3"


# ---- the TensorFlow options and the public names ----

TF_SIDE_CHECK = """
import tensorpotential.tpmodel
import numpy as np
import tensorflow as tf

tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
has_ndim = hasattr(tf.constant(1.0), "ndim")
rng = np.random.default_rng(0)
a = rng.standard_normal((64, 64)).astype(np.float32)
b = rng.standard_normal((64, 64)).astype(np.float32)
err = np.abs(tf.matmul(a, b).numpy().astype(np.float64) - a.astype(np.float64) @ b.astype(np.float64)).max()
print(tf32, has_ndim, err < 1e-4)
"""


def test_a_tensorflow_side_import_turns_tf32_off_and_numpy_behaviour_on(
    tmp_path: Path,
) -> None:
    """The error bound 1e-4 is about 2e3 times the float32 rounding of a 64-term dot product of
    unit-variance numbers (6e-8 * 8) and far below TF32's 1e-3 relative precision."""
    result = run_fresh_python(TF_SIDE_CHECK, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False True True"


@pytest.mark.parametrize(
    ("package", "owners"),
    [
        (
            "tensorpotential",
            {
                "TensorPotential": "tensorpotential.tensorpot",
                "TPModel": "tensorpotential.tpmodel",
                "LossFunction": "tensorpotential.loss",
                "L2Loss": "tensorpotential.loss",
            },
        ),
        (
            "tensorpotential.calculator",
            {
                "TPCalculator": "tensorpotential.calculator.asecalculator",
                "predict_structures": "tensorpotential.calculator.bulk",
                "grace_fm": "tensorpotential.calculator.foundation_models",
            },
        ),
    ],
)
def test_public_names_are_the_objects_of_their_modules(
    package: str, owners: dict[str, str], tmp_path: Path
) -> None:
    """A package ``__init__`` re-exports each name: the package attribute is the module's object."""
    checks = [
        f"getattr(importlib.import_module({package!r}), {name!r}) is "
        f"getattr(importlib.import_module({module!r}), {name!r})"
        for name, module in owners.items()
    ]
    names = sorted(owners)
    code = f"import importlib\nprint(all([{', '.join(checks)}]), sorted(importlib.import_module({package!r}).__all__))"
    assert last_stdout_line(code, tmp_path) == f"True {names}"


def test_star_import_offers_the_public_names(tmp_path: Path) -> None:
    result = run_fresh_python(
        "from tensorpotential import *\nprint(sorted(n for n in dir() if n[0].isupper()))\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.strip().splitlines()[-1]
        == "['L2Loss', 'LossFunction', 'TPModel', 'TensorPotential']"
    )


# ---- the package no longer imports TensorFlow ----

LOADED = (
    "sorted(m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras'))"
)


def test_importing_the_package_does_not_load_tensorflow(tmp_path: Path) -> None:
    assert (
        last_stdout_line(f"import sys, tensorpotential\nprint({LOADED})", tmp_path)
        == "[]"
    )


def test_a_public_name_loads_tensorflow_with_the_options_applied(
    tmp_path: Path,
) -> None:
    code = (
        "import sys, tensorpotential\n"
        "model_class = tensorpotential.TPModel\n"
        "import tensorflow as tf\n"
        "print(model_class.__name__, tf.config.experimental.tensor_float_32_execution_enabled(),"
        " hasattr(tf.constant(1.0), 'ndim'))\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "TPModel False True"


def test_dir_of_the_package_lists_the_public_names_without_importing_tensorflow(
    tmp_path: Path,
) -> None:
    code = f"import sys, tensorpotential\nprint('TPModel' in dir(tensorpotential), {LOADED})"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True []"


def test_an_unknown_name_is_an_attribute_error(tmp_path: Path) -> None:
    code = "import tensorpotential\nprint(hasattr(tensorpotential, 'NoSuchName'))"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_a_missing_tensorflow_is_explained_when_a_public_name_is_used(
    tmp_path: Path,
) -> None:
    code = (
        "import sys\n"
        "class Refuse:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('tensorflow', 'keras'):\n"
        "            raise ImportError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Refuse())\n"
        "import tensorpotential\n"
        "try:\n"
        "    tensorpotential.TensorPotential\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "'TensorPotential' needs TensorFlow" in result.stdout
    assert "tensorpotential[tf]" in result.stdout
