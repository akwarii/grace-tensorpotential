from __future__ import annotations

import numpy as np
import tensorflow as tf

from .shared_models import CuTwoLayerModels, weights_fingerprint


class _Tiny(tf.Module):
    """A module with a float, an int and a string variable, like a built ``TPModel``."""

    def __init__(self) -> None:
        super().__init__()
        self.w = tf.Variable(np.arange(6.0).reshape(2, 3), name="w")
        self.index = tf.Variable([0, 1], dtype=tf.int32, name="index")
        self.symbols = tf.Variable(["Al", "Li"], name="symbols")


def test_fingerprint_is_stable_for_unchanged_weights_including_strings() -> None:
    model = _Tiny()
    assert weights_fingerprint(model) == weights_fingerprint(model)
    assert weights_fingerprint(model) == weights_fingerprint(_Tiny())


def test_fingerprint_changes_with_any_float_int_or_string_value() -> None:
    reference = weights_fingerprint(_Tiny())
    floats, ints, strings = _Tiny(), _Tiny(), _Tiny()
    floats.w.assign(floats.w + 1e-12)
    ints.index.assign([1, 1])
    strings.symbols.assign(["Al", "Na"])
    for changed in (floats, ints, strings):
        assert weights_fingerprint(changed) != reference


def test_fingerprint_distinguishes_dtype_and_shape_of_equal_bytes() -> None:
    a, b = _Tiny(), _Tiny()
    b.w = tf.Variable(np.arange(6.0).reshape(3, 2), name="w")
    assert weights_fingerprint(a) != weights_fingerprint(b)


def test_shared_cu_models_are_built_once_and_a_change_is_reported() -> None:
    shared = CuTwoLayerModels()
    model = shared.get(False)
    assert shared.get(False) is model
    assert shared.get(True) is not model
    assert shared.changed() == []
    variable = next(v for v in model.variables if v.dtype.is_floating)
    variable.assign(variable + 1e-9)
    assert shared.changed() == [False]
