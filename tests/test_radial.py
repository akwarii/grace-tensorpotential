"""Characterization and behaviour tests of ``tensorpotential/functions/radial.py``.

The Gaussian basis accepts ``normalized`` and ignores it. ``RadialBasis(basis_type="Gaussian", ...)``
forwards its keywords to the class, and ``capture_init_args`` writes a passed ``normalized: false``
into a saved ``model.yaml``, so such a file must keep loading (and silently) and give the same model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.functions.radial import GaussianRadialBasisFunction
from tensorpotential.instructions.base import load_instructions, save_instructions_dict
from tensorpotential.instructions.compute import RadialBasis
from tests.tolerances import FLOAT64_ARITHMETIC

_RCUT = 5.5
_NFUNC = 6


def _gaussian(**extra) -> RadialBasis:
    rb = RadialBasis(
        bonds="r", basis_type="Gaussian", nfunc=_NFUNC, rcut=_RCUT, p=5, **extra
    )
    rb.build(tf.float64)
    return rb


def _basis_values(rb: RadialBasis) -> np.ndarray:
    r = np.linspace(0.2, 0.97 * _RCUT, 25).reshape(-1, 1)
    return rb.frwrd({"r": r}).numpy()


def _variables(rb: RadialBasis) -> dict[str, np.ndarray]:
    return {v.name: v.numpy() for v in rb.variables}


def _assert_same_model(first: RadialBasis, second: RadialBasis) -> None:
    assert _variables(first).keys() == _variables(second).keys()
    for name, value in _variables(first).items():
        np.testing.assert_array_equal(value, _variables(second)[name])
    np.testing.assert_allclose(
        _basis_values(first), _basis_values(second), **FLOAT64_ARITHMETIC._asdict()
    )


def test_gaussian_saved_with_normalized_false_loads_into_the_same_model(tmp_path):
    original = _gaussian(normalized=False)
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), {original.name: original}, param_dtype=tf.float64)
    assert "normalized: false" in path.read_text()

    loaded = load_instructions(str(path))[original.name]
    loaded.build(tf.float64)

    assert isinstance(loaded.basis_function, GaussianRadialBasisFunction)
    assert loaded.kwargs == original.kwargs
    _assert_same_model(original, loaded)


def test_gaussian_normalized_is_ignored_when_true():
    # The three blocks that used the argument were commented out: it has no effect.
    with pytest.warns(DeprecationWarning):
        flagged = _gaussian(normalized=True)
    _assert_same_model(flagged, _gaussian(normalized=False))


def test_gaussian_without_the_argument_equals_the_explicit_default():
    _assert_same_model(_gaussian(), _gaussian(normalized=False))


@pytest.mark.parametrize("trainable", [False, True])
def test_gaussian_variables_do_not_depend_on_normalized(trainable):
    plain = _gaussian(trainable=trainable)
    with pytest.warns(DeprecationWarning):
        flagged = _gaussian(trainable=trainable, normalized=True)
    assert [v.shape for v in plain.variables] == [v.shape for v in flagged.variables]
    assert [v.trainable for v in plain.variables] == [
        v.trainable for v in flagged.variables
    ]


def _gaussian_warnings(**extra) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _gaussian(**extra)
    return caught


@pytest.mark.parametrize("extra", [{}, {"normalized": False}])
def test_gaussian_default_normalized_is_silent(extra):
    # capture_init_args writes the default into saved yamls: loading one must warn about nothing.
    assert _gaussian_warnings(**extra) == []


@pytest.mark.parametrize("value", [True, None, 1])
def test_gaussian_non_default_normalized_warns_once_naming_class_and_argument(value):
    caught = _gaussian_warnings(normalized=value)
    assert [w.category for w in caught] == [DeprecationWarning]
    message = str(caught[0].message)
    assert "GaussianRadialBasisFunction" in message
    assert "normalized" in message
    assert "no effect" in message


def test_gaussian_warning_points_at_the_caller_of_the_class():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GaussianRadialBasisFunction(nfunc=_NFUNC, rcut=_RCUT, p=5, normalized=True)
    assert caught[0].filename == __file__


def test_gaussian_yaml_with_normalized_true_still_loads_with_the_warning(tmp_path):
    with pytest.warns(DeprecationWarning):
        original = _gaussian(normalized=True)
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), {original.name: original}, param_dtype=tf.float64)
    assert "normalized: true" in path.read_text()

    with pytest.warns(DeprecationWarning):
        loaded = load_instructions(str(path))[original.name]
    loaded.build(tf.float64)
    _assert_same_model(original, loaded)


def test_gaussian_yaml_with_normalized_false_loads_without_any_warning(tmp_path):
    original = _gaussian(normalized=False)
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), {original.name: original}, param_dtype=tf.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_instructions(str(path))
