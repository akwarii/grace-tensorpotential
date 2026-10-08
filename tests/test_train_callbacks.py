"""Characterization tests for ``CustomReduceLROnPlateau`` (``tensorpotential/cli/train_callbacks.py``).

Logic layer: when the learning rate is reduced (patience, cooldown, minimum, minimum improvement), what
happens at the minimum (stop or keep going), a missing monitored metric, the direction chosen by
``mode="auto"``, and ``reset_best``. Value layer: the learning rate after each epoch is derived by hand from
the rule ``lr -> max(lr * factor, min_lr)`` with exactly representable values, never from the callback.

The callback only needs ``model.optimizer.learning_rate`` and ``model.stop_training``: a real optimizer on a
small holder object stands in for ``TensorPotential``, which would cost seconds per test without changing
what is tested.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tensorpotential.cli.train_callbacks import CustomReduceLROnPlateau

import tensorflow as tf  # after tensorpotential, which selects the Keras flavour first

START_LR = 0.5  # exactly representable in float32: the optimizer keeps its learning rate in a float32 variable
FACTOR = 0.5


def _holder(lr: float = START_LR) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer=tf.keras.optimizers.Adam(lr), stop_training=False, step=0
    )


def _callback(holder=None, **kwargs) -> CustomReduceLROnPlateau:
    holder = holder or _holder()
    options = {"factor": FACTOR, "patience": 2, "mode": "min", **kwargs}
    callback = CustomReduceLROnPlateau(holder, **options)
    callback.on_train_begin()  # the training loop calls it; Keras 3 initialises its state there
    return callback


def _lr(callback: CustomReduceLROnPlateau) -> float:
    return float(callback.model.optimizer.learning_rate.numpy())


def _run(
    callback: CustomReduceLROnPlateau, values: list[float], key: str = "train_loss"
) -> list[float]:
    """The learning rate after each epoch end, for the given values of the monitored metric."""
    rates = []
    for epoch, value in enumerate(values):
        callback.on_epoch_end(epoch, {key: value})
        rates.append(_lr(callback))
    return rates


def test_the_rate_is_reduced_after_patience_epochs_without_improvement():
    callback = _callback(patience=2)

    rates = _run(callback, [1.0, 1.0, 1.0, 1.0, 1.0])

    # epoch 0 sets the best value; epochs 1 and 2 wait; the second wait reaches the patience
    assert rates == [0.5, 0.5, 0.25, 0.25, 0.125]


def test_an_improvement_resets_the_wait():
    callback = _callback(patience=2)

    rates = _run(callback, [1.0, 1.0, 0.9, 0.9, 0.8, 0.8])

    assert rates == [START_LR] * 6
    assert callback.wait == 1


def test_a_smaller_improvement_than_min_delta_does_not_count():
    callback = _callback(patience=1, min_delta=0.1)

    rates = _run(callback, [1.0, 0.95])

    assert rates == [0.5, 0.25]


def test_the_rate_is_never_reduced_below_the_minimum():
    callback = _callback(patience=1, min_lr=0.375)

    rates = _run(callback, [1.0, 1.0, 1.0])

    # 0.5 -> max(0.25, 0.375) = 0.375, then the minimum is reached and the rate stays
    assert rates == [0.5, 0.375, 0.375]


@pytest.mark.parametrize("stop_on_min_lr", [False, True])
def test_at_the_minimum_the_run_stops_only_when_asked_to(stop_on_min_lr, caplog):
    callback = _callback(patience=1, min_lr=0.5, stop_on_min_lr=stop_on_min_lr)

    with caplog.at_level(logging.INFO):
        _run(callback, [1.0, 1.0])

    assert _lr(callback) == START_LR
    assert callback.model.stop_training is stop_on_min_lr
    assert any("is achieved" in r.message for r in caplog.records)


def test_a_cooldown_of_two_epochs_pauses_the_counting_for_one_epoch():
    # PINNED, not endorsed: the override decrements the cooldown counter before it tests it, so a cooldown of
    # n epochs pauses the counting for n - 1 epochs
    callback = _callback(patience=1, cooldown=2)

    rates = _run(callback, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # epoch 1 reduces and starts the cooldown; epoch 2 is in it; epoch 3 leaves it and reduces again; same at 5
    assert rates == [0.5, 0.25, 0.25, 0.125, 0.125, 0.0625]


def test_a_missing_metric_is_skipped_with_a_warning(caplog):
    callback = _callback(patience=1)

    with caplog.at_level(logging.WARNING):
        callback.on_epoch_end(0, {"other": 1.0})
        callback.on_epoch_end(1, None)

    assert _lr(callback) == START_LR
    assert sum("not found in logs" in r.message for r in caplog.records) == 2


def test_auto_mode_minimises_a_loss_and_maximises_an_accuracy():
    loss = _callback(patience=1, mode="auto", monitor="train_loss")
    accuracy = _callback(patience=1, mode="auto", monitor="val_acc")

    # a falling loss and a rising accuracy keep improving: no reduction
    assert _run(loss, [1.0, 0.5, 0.25], "train_loss") == [START_LR] * 3
    assert _run(accuracy, [0.25, 0.5, 0.75], "val_acc") == [START_LR] * 3


def test_auto_mode_reduces_when_a_loss_rises_and_when_an_accuracy_falls():
    loss = _callback(patience=1, mode="auto", monitor="train_loss")
    accuracy = _callback(patience=1, mode="auto", monitor="val_acc")

    loss.on_epoch_end(0, {"train_loss": 1.0})
    loss.on_epoch_end(1, {"train_loss": 2.0})
    accuracy.on_epoch_end(0, {"val_acc": 0.5})
    accuracy.on_epoch_end(1, {"val_acc": 0.25})

    assert _lr(loss) == 0.25
    assert _lr(accuracy) == 0.25


def test_reset_best_makes_the_next_value_an_improvement():
    callback = _callback(patience=1)
    _run(callback, [0.001])

    callback.reset_best()
    rates = _run(callback, [5.0])

    assert callback.best == 5.0
    assert rates == [START_LR]


def test_a_new_callback_has_a_best_value_that_any_loss_improves_on():
    callback = _callback(patience=1)

    assert callback.best >= 1e99
    assert _run(callback, [1e6]) == [START_LR]


def test_a_best_value_of_zero_after_construction_is_replaced_by_a_huge_one():
    class StartsAtZero(CustomReduceLROnPlateau):
        """The parent state at the end of its constructor, set to the one value the override rewrites."""

        def _reset(self):
            super()._reset()
            self.best = 0.0

    callback = StartsAtZero(_holder(), factor=FACTOR, patience=2, mode="min")

    assert callback.best == 1e99
