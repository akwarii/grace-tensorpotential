"""Tests for ``_expand_l_to_lm``, the l-to-lm repeat of the radial functions (QUAL10).

The radial functions compute one value per ``l`` and repeat it for the ``2l+1`` values of ``m``;
``_expand_l_to_lm`` builds that repeat by broadcasting instead of a gather along the minor axis.

Logic layer: shapes for every ``lmax``, leading axes of any rank.
On a GPU the helper is the gather itself (QUAL10 GPU checkpoint).
Value layer: the oracle is the gather it replaces (``np.take`` with the ``l`` of every ``lm``
index), for the values and, through the chain rule, for the gradient: the gradient of the
repeat sums the ``2l+1`` upstream entries of each ``l``.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.instructions import compute
from tensorpotential.instructions.compute import _expand_l_to_lm

RNG = np.random.default_rng(10)


def _l_of_lm(lmax):
    return np.concatenate([[l] * (2 * l + 1) for l in range(lmax + 1)])


@pytest.mark.parametrize("lmax", [0, 1, 4])
def test_the_values_are_those_of_the_gather_by_l(lmax):
    x = RNG.standard_normal((7, 3, lmax + 1))

    got = _expand_l_to_lm(tf.constant(x), lmax, _l_of_lm(lmax))

    assert got.shape == (7, 3, (lmax + 1) ** 2)
    np.testing.assert_array_equal(got.numpy(), np.take(x, _l_of_lm(lmax), axis=-1))


def test_any_number_of_leading_axes():
    x = RNG.standard_normal((2, 3, 4, 3))

    got = _expand_l_to_lm(tf.constant(x), 2, _l_of_lm(2))

    np.testing.assert_array_equal(got.numpy(), np.take(x, _l_of_lm(2), axis=-1))


def test_the_gradient_sums_the_upstream_entries_of_each_l():
    lmax = 3
    x_np = RNG.standard_normal((5, 2, lmax + 1))
    x = tf.constant(x_np)
    seed = RNG.standard_normal((5, 2, (lmax + 1) ** 2))

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(_expand_l_to_lm(x, lmax, _l_of_lm(lmax)) * seed)
    grad = tf.convert_to_tensor(tape.gradient(loss, x)).numpy()

    want = np.zeros_like(x_np)
    for lm, l in enumerate(_l_of_lm(lmax)):
        want[..., l] += seed[..., lm]
    np.testing.assert_allclose(grad, want, rtol=1e-14, atol=1e-14)


def test_it_compiles_under_xla_with_an_unknown_bond_count():
    expand = tf.function(
        lambda x: _expand_l_to_lm(x, 2, _l_of_lm(2)),
        jit_compile=True,
        input_signature=[tf.TensorSpec([None, 3, 3], tf.float64)],
    )
    x = RNG.standard_normal((11, 3, 3))

    np.testing.assert_array_equal(
        expand(tf.constant(x)).numpy(), np.take(x, _l_of_lm(2), axis=-1)
    )


def test_on_a_gpu_the_repeat_is_the_original_gather(monkeypatch):
    monkeypatch.setattr(compute, "_cpu_formulations", lambda: False)
    x = RNG.standard_normal((4, 2, 3))
    original = tf.gather
    gathered = []

    def spy(*args, **kwargs):
        gathered.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute.tf, "gather", spy)

    got = _expand_l_to_lm(tf.constant(x), 2, _l_of_lm(2))

    assert gathered == [{"axis": -1}]
    np.testing.assert_array_equal(got.numpy(), np.take(x, _l_of_lm(2), axis=-1))
