"""Tests for the per-block weight contraction of the reducing instructions (QUAL10).

``_tile_runs`` splits the weight tile of a collection (channel -> weight index) into runs of
consecutive channels that share a weight; ``_contract_weight_runs`` contracts every run with its
weight without expanding the weights to every channel.

Logic layer: the runs of hand-written tiles, both layouts, one run and many.
Value layer: a numpy oracle that expands the weights to every channel first (the formulation the
contraction replaces) and contracts with ``np.einsum``, with and without the per-atom (central
type) weights; and the tile of a real ``FunctionReduceN`` collection, whose runs must cover its
channels in order.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.instructions import FunctionReduceN, SphericalHarmonic
from tensorpotential.instructions.compute import _contract_weight_runs, _tile_runs

RNG = np.random.default_rng(208)


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(
    ("tile", "runs"),
    [
        ([0], ((0, 1, 0),)),
        ([0, 1, 1, 1, 2, 2, 2, 2, 2], ((0, 1, 0), (1, 4, 1), (4, 9, 2))),
        ([3, 3, 0, 0, 3], ((0, 2, 3), (2, 4, 0), (4, 5, 3))),  # a weight may come back
        (np.array([[0.0], [0.0], [1.0]]), ((0, 2, 0), (2, 3, 1))),  # any int-like array
    ],
)
def test_runs_are_maximal_and_cover_the_tile_in_order(tile, runs):
    assert _tile_runs(tile) == runs


def test_the_runs_of_a_tensor_tile_are_those_of_its_values():
    assert _tile_runs(tf.constant([1, 1, 0], dtype=tf.int32)) == ((0, 2, 1), (2, 3, 0))


def test_an_empty_tile_has_no_runs():
    assert _tile_runs([]) == ()


# ------------------------------------------------------------------ values


def _oracle(w, x, tile, lm_first, mu=None):
    """Expand the weights to every channel, then contract: the replaced formulation."""
    w_exp = w[..., np.asarray(tile)]
    if lm_first:
        x = np.transpose(x, (1, 2, 0))  # [c, a, n] -> [a, n, c]
    if mu is None:
        return np.einsum("knw,anw->wak", w_exp, x)
    return np.einsum("aknw,anw->wak", w_exp[mu], x)


@pytest.mark.parametrize("lm_first", [False, True])
@pytest.mark.parametrize("per_type", [False, True])
def test_the_contraction_equals_the_expanded_weights_oracle(lm_first, per_type):
    tile = [0, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1]
    n_atoms, n_in, n_out, n_w, n_types = 5, 4, 3, 3, 2
    shape = [n_types, n_out, n_in, n_w] if per_type else [n_out, n_in, n_w]
    w = RNG.standard_normal(shape)
    x = RNG.standard_normal((n_atoms, n_in, len(tile)))
    mu = RNG.integers(0, n_types, n_atoms) if per_type else None
    x_in = np.transpose(x, (2, 0, 1)) if lm_first else x

    got = _contract_weight_runs(
        tf.constant(w),
        tf.constant(x_in),
        _tile_runs(tile),
        lm_first,
        None if mu is None else tf.constant(mu),
    )

    want = _oracle(w, x_in, tile, lm_first, mu)
    assert got.shape == (len(tile), n_atoms, n_out)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-13, atol=1e-13)


def test_one_run_returns_the_single_contraction():
    w = RNG.standard_normal((2, 3, 1))
    x = RNG.standard_normal((4, 3, 5))

    got = _contract_weight_runs(
        tf.constant(w), tf.constant(x), _tile_runs([0] * 5), False
    )

    np.testing.assert_allclose(got.numpy(), _oracle(w, x, [0] * 5, False), rtol=1e-13)


def test_the_gradient_equals_the_gradient_of_the_oracle():
    tile = [0, 1, 1, 1]
    w = tf.constant(RNG.standard_normal((2, 3, 2)))
    x = tf.constant(RNG.standard_normal((4, 3, 4)))
    seed = RNG.standard_normal((4, 4, 2))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([w, x])
        got = _contract_weight_runs(w, x, _tile_runs(tile), False)
        expanded = tf.einsum("knw,anw->wak", tf.gather(w, tile, axis=-1), x)
        got_loss = tf.reduce_sum(got * seed)
        want_loss = tf.reduce_sum(expanded * seed)

    for got_g, want_g in zip(
        tape.gradient(got_loss, [w, x]), tape.gradient(want_loss, [w, x]), strict=True
    ):
        np.testing.assert_allclose(
            got_g.numpy(), want_g.numpy(), rtol=1e-13, atol=1e-13
        )


def test_the_tile_of_a_real_collection_is_covered_by_its_runs():
    y = SphericalHarmonic(vhat="ScaledBondVector", lmax=3, name="Y")
    reduce = FunctionReduceN(
        instructions=[y],
        name="E",
        ls_max=3,
        n_out=2,
        allowed_l_p=[[0, 1], [1, -1], [2, 1], [3, -1]],
        is_central_atom_type_dependent=True,
        number_of_atom_types=2,
    )
    tile = reduce.collector["Y"]["w_l_tile"].numpy()

    runs = _tile_runs(tile)

    assert [r[0] for r in runs] == [0] + [r[1] for r in runs[:-1]]  # contiguous
    assert runs[-1][1] == len(tile)
    assert [r[1] - r[0] for r in runs] == [3, 7, 1, 5]  # 2l+1 channels, by parity then l
