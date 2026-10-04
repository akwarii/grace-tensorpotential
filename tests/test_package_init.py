"""Characterization tests for ``tensorpotential/__init__.py``.

Importing the package sets the legacy-Keras flag before TensorFlow is imported, switches
TensorFlow to numpy-style type promotion and turns TensorFloat-32 off, and offers
``TensorPotential``, ``TPModel``, ``LossFunction`` and ``L2Loss``.

Logic layer: the flag in every state of the environment, the public names. Physical-value
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
from tests.fresh_python import run_fresh_python

FLAG = "TF_USE_LEGACY_KERAS"


# ---- the legacy-Keras flag ----


FLAG_PROBE = (
    "import tensorpotential, os; print(repr(os.environ.get('TF_USE_LEGACY_KERAS')))"
)


@pytest.mark.parametrize("before", [None, ""])
def test_flag_is_set_when_missing_or_empty_and_the_user_is_told(
    before: str | None, tmp_path: Path
) -> None:
    result = run_fresh_python(FLAG_PROBE, tmp_path, **{FLAG: before})
    assert result.returncode == 0, result.stderr
    assert "automatically set" in result.stdout
    assert result.stdout.strip().endswith("'1'")


@pytest.mark.parametrize("value", ["1", "true"])
def test_flag_already_right_is_left_alone_and_silent(
    value: str, tmp_path: Path
) -> None:
    result = run_fresh_python(FLAG_PROBE, tmp_path, **{FLAG: value})
    assert result.returncode == 0, result.stderr
    assert f"'{value}'" in result.stdout
    assert "automatically set" not in result.stdout
    assert "requires" not in result.stdout


def test_not_verbose_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(FLAG, "1")
    tensorpotential._configure_keras_backend(verbose=False)
    assert capsys.readouterr().out == ""


def test_flag_set_to_something_else_in_a_fresh_interpreter_is_kept_with_a_warning(
    tmp_path: Path,
) -> None:
    result = run_fresh_python(
        "import tensorpotential, os; print(os.environ['TF_USE_LEGACY_KERAS'])",
        tmp_path,
        **{FLAG: "0"},
    )
    assert result.returncode == 0, result.stderr
    assert "requires '1'" in result.stdout
    assert result.stdout.strip().endswith("0")


def test_flag_set_to_something_else_after_tensorflow_is_loaded_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow  # noqa: F401

    monkeypatch.setenv(FLAG, "0")
    with pytest.warns(RuntimeWarning, match="imported before"):
        tensorpotential._configure_keras_backend(verbose=False)
    assert os.environ[FLAG] == "0"


def test_flag_right_after_tensorflow_is_loaded_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow  # noqa: F401

    monkeypatch.setenv(FLAG, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensorpotential._configure_keras_backend(verbose=False)


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


def test_public_names_are_the_classes_of_their_modules(tmp_path: Path) -> None:
    code = (
        "import tensorpotential as t\n"
        "from tensorpotential.tensorpot import TensorPotential\n"
        "from tensorpotential.tpmodel import TPModel\n"
        "from tensorpotential.loss import LossFunction, L2Loss\n"
        "print(t.TensorPotential is TensorPotential, t.TPModel is TPModel,"
        " t.LossFunction is LossFunction, t.L2Loss is L2Loss, sorted(t.__all__))\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "True True True True ['L2Loss', 'LossFunction', 'TPModel', 'TensorPotential']"
    )


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


def test_calculator_subpackage_offers_its_three_names(tmp_path: Path) -> None:
    """``tensorpotential.calculator`` is the other package ``__init__`` that imports eagerly."""
    code = (
        "import tensorpotential.calculator as c\n"
        "from tensorpotential.calculator.asecalculator import TPCalculator\n"
        "from tensorpotential.calculator.bulk import predict_structures\n"
        "from tensorpotential.calculator.foundation_models import grace_fm\n"
        "print(c.TPCalculator is TPCalculator, c.predict_structures is predict_structures,"
        " c.grace_fm is grace_fm, sorted(c.__all__))\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "True True True ['TPCalculator', 'grace_fm', 'predict_structures']"
    )


# ---- the package no longer imports TensorFlow ----

LOADED = (
    "sorted(m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'tf_keras'))"
)


def test_importing_the_package_does_not_load_tensorflow(tmp_path: Path) -> None:
    result = run_fresh_python(f"import sys, tensorpotential\nprint({LOADED})", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


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
        "        if name.split('.')[0] in ('tensorflow', 'tf_keras'):\n"
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
