"""Tests for ``tensorpotential/keras_adapter.py``.

Logic layer: ``keras.Variable == <a Python type>`` answers ``False`` (it raised), every other comparison is
unchanged, the fix is installed once, and a Keras without the patched method raises an error that names it.
Value layer: with the package's numpy-style type promotion, a Keras 3 Adam step fails without the fix and
moves the weights, with it, by the amount that the Adam rule gives by hand. Concurrency layer: an optimizer
step inside the replica threads of a two-replica ``MirroredStrategy`` whose model needs the type promotion
works. An earlier design switched the promotion mode off around the step; the mode is process-wide, and a
replica tracing the model in another thread failed.
Import-time behaviour is checked in fresh interpreters, because the promotion mode and the fix are
process-wide and a second import is a no-op.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path

import numpy as np
import pytest

from tensorpotential import keras_adapter
from tensorpotential.keras_adapter import (
    KerasVariableUnavailableError,
    install_variable_comparison_fix,
)

import keras

from tests.fresh_python import last_stdout_line
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

LEARNING_RATE = (
    0.5  # exactly representable in float32, the dtype of the optimizer's learning rate
)
EPSILON = 1e-7

ADAM_STEP = """
import {first}
from tensorpotential import _tf_options  # numpy-style type promotion, as in every TensorFlow-side module
import json
import tensorflow as tf

weight = tf.Variable([1.0, 2.0], dtype=tf.float64)
optimizer = tf.keras.optimizers.Adam({lr})
gradient = tf.constant([0.25, -4.0], dtype=tf.float64)
try:
    optimizer.apply_gradients([(gradient, weight)])
    print(json.dumps(weight.numpy().tolist()))
except ValueError as error:
    print(json.dumps("ValueError: " + str(error)))
"""


def _adam_step(fresh_dir: Path, *, with_fix: bool):
    code = ADAM_STEP.format(
        first="tensorpotential.keras_adapter" if with_fix else "tensorpotential",
        lr=LEARNING_RATE,
    )
    return json.loads(last_stdout_line(code, fresh_dir))


def test_a_keras_3_adam_step_fails_under_the_numpy_style_promotion_without_the_fix(
    tmp_path,
):
    outcome = _adam_step(tmp_path, with_fix=False)

    assert isinstance(outcome, str)
    assert "unsupported type" in outcome


def test_a_keras_3_adam_step_with_the_fix_follows_the_adam_rule(tmp_path):
    # first step: m = (1 - b1) g, v = (1 - b2) g^2, alpha = lr sqrt(1 - b2) / (1 - b1), so the weight moves
    # by alpha m / (sqrt(v) + eps). The betas enter alpha rounded to float32 (see tests/test_tensorpot.py).
    gradient = np.array([0.25, -4.0])
    beta_1, beta_2 = float(np.float32(0.9)), float(np.float32(0.999))
    alpha = LEARNING_RATE * np.sqrt(1 - beta_2) / (1 - beta_1)
    expected = np.array([1.0, 2.0]) - alpha * (0.1 * gradient) / (
        np.sqrt(0.001 * gradient**2) + EPSILON
    )

    found = _adam_step(tmp_path, with_fix=True)

    np.testing.assert_allclose(found, expected, rtol=ARITH.rtol, atol=ARITH.atol)


MIRRORED_STEP = """
import tensorpotential.keras_adapter
from tensorpotential import _tf_options
import tensorflow as tf

cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(cpu, [tf.config.LogicalDeviceConfiguration()] * 2)
strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
with strategy.scope():
    weight = tf.Variable([1.0, 2.0], dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(0.5)
    optimizer.build([weight])  # Keras 3 cannot create its variables inside a step traced for replicas

@tf.function
def step():
    def replica():
        data = tf.constant([3.0, 4.0], dtype=tf.float64)  # float64 data, float32 weight: needs the promotion
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(weight * data)
        optimizer.apply_gradients([(tape.gradient(loss, weight), weight)])
        return loss

    return strategy.run(replica)

step()
print(strategy.num_replicas_in_sync, [round(float(w), 3) for w in weight.numpy()])
"""


def test_an_optimizer_step_works_in_the_replica_threads_of_a_mirrored_strategy(
    tmp_path,
):
    # Adam moves each weight by about lr in the direction of -sign(gradient): 1 - 0.5 and 2 - 0.5
    assert last_stdout_line(MIRRORED_STEP, tmp_path) == "2 [0.5, 1.5]"


def test_comparing_a_keras_variable_with_a_python_type_answers_false():
    variable = keras.Variable([1.0, 2.0])

    # operator.eq: the comparison with a type is the point of the test, not a type check
    for python_type in (bool, int, float):
        assert operator.eq(variable, python_type) is False


def test_other_comparisons_of_a_keras_variable_are_unchanged():
    variable = keras.Variable([1.0, 2.0])

    np.testing.assert_array_equal(np.asarray(variable == 2.0), [False, True])
    np.testing.assert_array_equal(
        np.asarray(variable == np.array([1.0, 3.0])), [True, False]
    )
    np.testing.assert_array_equal(np.asarray(variable != 2.0), [True, False])


def test_the_fix_is_installed_once_however_often_it_is_asked_for():
    before = keras.Variable.__eq__

    install_variable_comparison_fix()
    install_variable_comparison_fix()

    assert keras.Variable.__eq__ is before
    assert getattr(before, keras_adapter.PATCH_MARK) is True


@pytest.mark.parametrize("attribute", ["Variable", "__eq__"])
def test_a_keras_without_the_patched_method_is_refused_by_name(attribute, monkeypatch):
    # monkeypatch stands in for a Keras release that removed what the fix patches
    if attribute == "Variable":
        monkeypatch.delattr(keras, "Variable")
    else:
        monkeypatch.setattr(keras.Variable, "__eq__", None)

    with pytest.raises(KerasVariableUnavailableError, match=r"keras\.Variable\.__eq__"):
        install_variable_comparison_fix()
