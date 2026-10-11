"""Tests for ``tensorpotential/optimizer_layout.py``.

Logic layer: the layout stored in a checkpoint (``None`` without optimizer state, slot shapes by position),
the layout that a fresh optimizer would save, and the comparison of the two for a matching checkpoint, for
other optimizer options, for a model without optimizer state and for a checkpoint written by legacy Keras 2
(``tests/legacy_keras_checkpoint``). Value layer: the number and the shapes of the slots follow from the Adam
rule (one first moment and one second moment per trainable variable, a third slot with amsgrad, a fourth with
the moving average), counted from the model, not from the optimizer under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorpotential.optimizer_layout import (
    expected_optimizer_layout,
    optimizer_state_fits,
    saved_optimizer_layout,
)

import tensorflow as tf  # after tensorpotential

from tests.legacy_keras_checkpoint import LEGACY_CHECKPOINT, small_tp

PLAIN = {"learning_rate": 0.5, "amsgrad": False}
AMSGRAD = {"learning_rate": 0.5, "amsgrad": True}
WITH_EMA = {"learning_rate": 0.5, "amsgrad": True, "use_ema": True}


def apply_step(tp, seed: int = 1) -> None:
    """Apply seeded gradients once."""
    variables = tp.model.variables_to_train
    grads = [
        tf.constant(
            np.random.default_rng([seed, i]).standard_normal(v.shape), dtype=v.dtype
        )
        for i, v in enumerate(variables)
    ]
    tp.optimizer.apply_gradients(zip(grads, variables))


def _saved(tmp_path, options: dict) -> str:
    """A checkpoint written by this Keras line after one optimizer step."""
    tp = small_tp(options)
    apply_step(tp)
    path = str(tmp_path / "checkpoint")
    tp.save_checkpoint(checkpoint_name=path)
    return path


def _slot_shapes(tp, slots_per_variable: int) -> list[tuple[int, ...]]:
    return sorted(
        tuple(int(d) for d in v.shape)
        for v in tp.model.variables_to_train
        for _ in range(slots_per_variable)
    )


@pytest.mark.parametrize(
    ("options", "slots"), [(PLAIN, 2), (AMSGRAD, 3), (WITH_EMA, 4)]
)
def test_the_saved_layout_holds_the_adam_slots_of_every_trainable_variable(
    tmp_path, options, slots
):
    layout = saved_optimizer_layout(_saved(tmp_path, options))

    assert layout is not None
    assert sorted(layout.values()) == _slot_shapes(small_tp(), slots)
    assert len(layout) == slots * len(small_tp().model.variables_to_train)


def test_a_checkpoint_without_optimizer_state_has_no_layout(tmp_path):
    tp = small_tp()
    tp.optimizer = None
    tp.setup_checkpoint()
    path = str(tmp_path / "checkpoint")
    tp.save_checkpoint(checkpoint_name=path)

    assert saved_optimizer_layout(path) is None
    assert optimizer_state_fits(
        path, small_tp().optimizer, small_tp().model.variables_to_train
    )


@pytest.mark.parametrize("options", [PLAIN, AMSGRAD, WITH_EMA])
def test_a_fresh_optimizer_expects_the_layout_that_it_saves(tmp_path, options):
    path = _saved(tmp_path, options)
    tp = small_tp(options)

    assert expected_optimizer_layout(
        tp.optimizer, tp.model.variables_to_train
    ) == saved_optimizer_layout(path)
    assert optimizer_state_fits(path, tp.optimizer, tp.model.variables_to_train)


@pytest.mark.parametrize(
    ("written", "reading"), [(AMSGRAD, PLAIN), (PLAIN, AMSGRAD), (AMSGRAD, WITH_EMA)]
)
def test_other_optimizer_options_do_not_fit(tmp_path, written, reading):
    path = _saved(tmp_path, written)
    tp = small_tp(reading)

    assert not optimizer_state_fits(path, tp.optimizer, tp.model.variables_to_train)


def test_a_legacy_keras_checkpoint_does_not_fit_the_same_options():
    tp = small_tp()  # the options that the fixture was written with

    assert saved_optimizer_layout(LEGACY_CHECKPOINT) is not None
    assert not optimizer_state_fits(
        LEGACY_CHECKPOINT, tp.optimizer, tp.model.variables_to_train
    )


def test_the_probe_leaves_the_optimizer_untouched():
    tp = small_tp()

    expected_optimizer_layout(tp.optimizer, tp.model.variables_to_train)

    assert not tp.optimizer.built
    assert int(tp.optimizer.iterations.numpy()) == 0
    assert [id(v) for v in tp.optimizer.variables] == [
        id(tp.optimizer.iterations),
        id(tp.optimizer.learning_rate),
    ]


def test_the_expected_layout_leaves_out_the_counter_and_the_learning_rate():
    tp = small_tp(AMSGRAD)

    layout = expected_optimizer_layout(tp.optimizer, tp.model.variables_to_train)

    assert min(layout) == 2  # positions 0 and 1 hold them in a Keras 3 optimizer
    assert () not in layout.values()
    assert tf.TensorShape(layout[2]).rank >= 1
