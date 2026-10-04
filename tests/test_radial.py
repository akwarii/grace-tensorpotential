"""Characterization and behaviour tests of ``tensorpotential/functions/radial.py``.

The Gaussian basis accepts ``normalized`` and ignores it. ``RadialBasis(basis_type="Gaussian", ...)``
forwards its keywords to the class, and ``capture_init_args`` writes a passed ``normalized: false``
into a saved ``model.yaml``, so such a file must keep loading (and silently) and give the same model.
"""

from __future__ import annotations

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
    _assert_same_model(_gaussian(normalized=True), _gaussian(normalized=False))


def test_gaussian_without_the_argument_equals_the_explicit_default():
    _assert_same_model(_gaussian(), _gaussian(normalized=False))


@pytest.mark.parametrize("trainable", [False, True])
def test_gaussian_variables_do_not_depend_on_normalized(trainable):
    plain = _gaussian(trainable=trainable)
    flagged = _gaussian(trainable=trainable, normalized=True)
    assert [v.shape for v in plain.variables] == [v.shape for v in flagged.variables]
    assert [v.trainable for v in plain.variables] == [
        v.trainable for v in flagged.variables
    ]
