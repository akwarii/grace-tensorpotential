"""Tests of ``tests/seeded_weights.py``: the helper that makes zero-initialised layers non-trivial.

Logic layer: what is and is not assigned, determinism, independence of the order, the key of equal names.
Value layer: the statistics of the values against their stated distribution, and the effect on a
real zero-initialised instruction (``InvariantLayerRMSNorm``), whose output is exactly zero before seeding.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.instructions import InvariantLayerRMSNorm
from tests.seeded_weights import seed_trainable_variables, seeded_values
from tests.tolerances import FLOAT64_ARITHMETIC, sample_std_rtol


class _Module(tf.Module):
    """Two trainable floats, one frozen float, one integer table, in a fixed order."""

    def __init__(self, order: tuple[str, ...] = ("w", "b")) -> None:
        super().__init__(name="m")
        makers = {
            "w": lambda: tf.Variable(tf.zeros([40, 30], tf.float64), name="w"),
            "b": lambda: tf.Variable(tf.zeros([30], tf.float64), name="b"),
        }
        for key in order:
            setattr(self, key, makers[key]())
        self.cutoff = tf.Variable(5.0, trainable=False, name="cutoff", dtype=tf.float64)
        self.index = tf.Variable(
            [0, 1, 2], name="index"
        )  # int32, trainable flag irrelevant


def test_trainable_floats_become_non_zero_and_constants_stay():
    m = _Module()
    seed_trainable_variables(m)
    assert np.all(m.w.numpy() != 0.0)
    assert np.all(m.b.numpy() != 0.0)
    assert m.cutoff.numpy() == 5.0
    assert m.index.numpy().tolist() == [0, 1, 2]


def test_returns_the_assigned_values_keyed_by_name():
    m = _Module()
    assigned = seed_trainable_variables(m)
    assert set(assigned) == {"w:0#0", "b:0#0"}
    np.testing.assert_array_equal(assigned["w:0#0"], m.w.numpy())


def test_same_seed_same_values_other_seed_other_values():
    a, b, c = _Module(), _Module(), _Module()
    seed_trainable_variables(a, seed=3)
    seed_trainable_variables(b, seed=3)
    seed_trainable_variables(c, seed=4)
    np.testing.assert_array_equal(a.w.numpy(), b.w.numpy())
    assert not np.array_equal(a.w.numpy(), c.w.numpy())


def test_values_do_not_depend_on_the_creation_order():
    wb, bw = _Module(("w", "b")), _Module(("b", "w"))
    seed_trainable_variables(wb)
    seed_trainable_variables(bw)
    np.testing.assert_array_equal(wb.w.numpy(), bw.w.numpy())
    np.testing.assert_array_equal(wb.b.numpy(), bw.b.numpy())


def test_equal_names_get_different_values():
    class Twin(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="twin")
            self.first = tf.Variable(tf.zeros([6], tf.float64), name="scale")
            self.second = tf.Variable(tf.zeros([6], tf.float64), name="scale")

    t = Twin()
    assigned = seed_trainable_variables(t)
    assert len(assigned) == 2
    assert not np.array_equal(t.first.numpy(), t.second.numpy())


def test_dtype_and_shape_of_the_variable_are_kept():
    class F32(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="f32")
            self.v = tf.Variable(tf.zeros([3, 4], tf.float32), name="v")

    f = F32()
    seed_trainable_variables(f)
    assert f.v.dtype == tf.float32
    assert f.v.shape.as_list() == [3, 4]
    assert np.all(f.v.numpy() != 0.0)


def test_a_module_without_trainable_variables_gives_an_empty_result():
    class Empty(tf.Module):
        pass

    assert seed_trainable_variables(Empty(name="e")) == {}


def test_matrix_values_have_the_stated_standard_deviation():
    values = seeded_values("w", (400, 300))
    expected = 1.0 / np.sqrt(400)
    assert abs(values.mean()) < 5 * expected / np.sqrt(values.size)
    assert values.std() == pytest.approx(expected, rel=sample_std_rtol(values.size))


def test_vector_values_are_centred_on_one():
    values = seeded_values("b", (4000,))
    assert values.mean() == pytest.approx(1.0, abs=5 * 0.1 / np.sqrt(values.size))
    assert values.std() == pytest.approx(0.1, rel=sample_std_rtol(values.size))


def test_scalar_shape_is_supported():
    assert seeded_values("s", ()).shape == ()


def test_zero_initialised_layer_gives_a_non_trivial_output_once_seeded():
    x = tf.constant(np.random.default_rng(0).normal(size=(3, 4, 1)), tf.float64)
    data = {
        "src": x,
        constants.N_ATOMS_BATCH_REAL: tf.constant(3),
        constants.ATOMIC_MU_I: tf.constant([0, 0, 0]),
    }

    class Source:
        name = "src"
        n_out = 4

    layer = InvariantLayerRMSNorm(Source(), name="norm", init="zeros")
    layer.build(tf.float64)
    zero_init = layer.frwrd(dict(data)).numpy()
    assert np.all(
        zero_init[:, 1:, :] == 0.0
    )  # the normalised channels vanish with scale 0

    seed_trainable_variables(layer)
    out = layer.frwrd(dict(data)).numpy()
    # only_nonlin: first channel passes through, the others are RMS-normalised with the seeded scale
    x_np = np.asarray(x.numpy())
    np.testing.assert_allclose(
        out[:, 0, :],
        x_np[:, 0, :],
        rtol=FLOAT64_ARITHMETIC.rtol,
        atol=FLOAT64_ARITHMETIC.atol,
    )
    assert np.all(out[:, 1:, :] != 0.0)
