"""Tests of the learning-rate schedules of ``tensorpotential/cli/train_callbacks.py``.

Logic layer: the factory (``LRSchedulerFactory``) maps an ``input.yaml`` ``fit`` section to the
right callback with the right step counts; the callbacks write the learning rate they compute
into the optimizer and into the log file; the plateau callback reduces, waits and stops on
the epochs it is told about.
Value layer: every expected learning rate is computed in the test, step by step, from the closed
form of the schedule (a straight line, a half cosine, a geometric progression, a hand-traced
plateau), never by the callback. The optimizer is a real Keras ``Adam``; only the object that
holds it (``TensorPotential``, which the callbacks read for ``optimizer``, ``step`` and
``stop_training``) is replaced by a small holder.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

# tensorpotential first: it applies the TensorFlow options before TensorFlow is imported
from tensorpotential.cli.train_callbacks import (
    CosineDecay,
    CustomReduceLROnPlateau,
    ExponentialDecay,
    LinearDecay,
    LRSchedulerFactory,
)

import tensorflow as tf  # isort: skip

from tests.tolerances import (
    FLOAT32_ELEMENTWISE as F32,
)  # the rate is a float32 variable
from tests.tolerances import FLOAT32_NETWORK as NET

N_BATCHES = 3
MAXITER = 5
WARMUP_EPOCHS = 2
PEAK_LR = 2.0
COLD_LR = 0.2
MIN_LR = 0.1
WARMUP_STEPS = WARMUP_EPOCHS * N_BATCHES  # 6
DECAY_STEPS = MAXITER * N_BATCHES - WARMUP_STEPS  # 9
TOTAL_STEPS = MAXITER * N_BATCHES  # 15
DECAY_KINDS = ("linear_decay", "cosine_decay", "exponential_decay")


class Host:
    """Stands in for ``TensorPotential``: a real optimizer, a step counter, a stop flag."""

    def __init__(self, learning_rate: float, step: int = 0) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.step = step
        self.stop_training = False

    @property
    def lr(self) -> float:
        return float(self.optimizer.learning_rate.numpy())


def fit_section(kind: str, *, warmup_epochs: float = WARMUP_EPOCHS, **extra) -> dict:
    params = {"minimum_learning_rate": MIN_LR, "warmup_epochs": warmup_epochs}
    if warmup_epochs:
        params["cold_learning_rate"] = COLD_LR
    return {
        "opt_params": {"learning_rate": PEAK_LR},
        "scheduler": kind,
        "scheduler_params": params,
        "maxiter": MAXITER,
        **extra,
    }


def make(kind: str, host: Host, *, n_batches: int = N_BATCHES, **kwargs):
    fit = fit_section(kind, **kwargs)
    return LRSchedulerFactory.create_lr_scheduler(
        host, n_batches, fit, SimpleNamespace(num_replicas_in_sync=1)
    )


def expected_lr(kind: str, step: int, *, peak, cold, low, warm, decay) -> float:
    """Learning rate after ``step`` batches, from the closed form of each schedule.

    Warm-up is a straight line from ``cold`` to ``peak`` over ``warm`` steps; the decay then goes
    from ``peak`` to ``low`` over ``decay`` steps with the fraction ``p = k / (decay - 1)`` of
    the decay already done at its k-th step; after ``warm + decay`` steps the rate stays at
    ``low``.
    """
    if warm and step < warm:
        return cold + (peak - cold) * step / warm
    if step >= warm + decay:
        return low
    p = (step - warm) / max(decay - 1, 1)
    if kind == "linear_decay":
        return (1 - p) * peak + p * low
    if kind == "cosine_decay":
        alpha = low / peak
        return peak * ((1 - alpha) * 0.5 * (1 + math.cos(math.pi * p)) + alpha)
    if kind == "exponential_decay":
        return peak * (low / peak) ** p
    raise AssertionError(kind)


def run_steps(callback, host: Host, n: int) -> list[float]:
    """Learning rate seen after each of the steps 1..n, as the training loop calls the callback."""
    seen = []
    for step in range(1, n + 1):
        host.step = step
        callback.on_batch_end(step, {})
        seen.append(host.lr)
    return seen


# -------------------------------------------------------------------- closed-form sequences
@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_warmup_then_decay_follows_the_closed_form(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host)
    seen = run_steps(cb, host, TOTAL_STEPS + 3)
    want = [
        expected_lr(
            kind,
            s,
            peak=PEAK_LR,
            cold=COLD_LR,
            low=MIN_LR,
            warm=WARMUP_STEPS,
            decay=DECAY_STEPS,
        )
        for s in range(1, TOTAL_STEPS + 4)
    ]
    np.testing.assert_allclose(seen, want, rtol=F32.rtol, atol=F32.atol)


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_decay_without_warmup_starts_at_the_peak_and_ends_at_the_minimum(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host, warmup_epochs=0, maxiter=MAXITER)
    assert cb.warmup_steps == 0
    assert cb.decay_steps == TOTAL_STEPS
    seen = run_steps(cb, host, TOTAL_STEPS + 2)
    want = [
        expected_lr(
            kind, s, peak=PEAK_LR, cold=0.0, low=MIN_LR, warm=0, decay=TOTAL_STEPS
        )
        for s in range(1, TOTAL_STEPS + 3)
    ]
    np.testing.assert_allclose(seen, want, rtol=F32.rtol, atol=F32.atol)
    # step 1 is already one decay step in: below the peak, above the minimum
    assert MIN_LR < seen[0] < PEAK_LR
    # the last decay step reaches the minimum exactly and the rate stays there afterwards
    assert seen[TOTAL_STEPS - 1] == pytest.approx(MIN_LR, rel=F32.rtol)
    assert seen[-2:] == [pytest.approx(MIN_LR, rel=F32.rtol)] * 2


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_warmup_is_a_straight_line_that_reaches_the_peak(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host)
    seen = run_steps(cb, host, WARMUP_STEPS)
    slope = (PEAK_LR - COLD_LR) / WARMUP_STEPS
    np.testing.assert_allclose(
        np.diff([COLD_LR, *seen]), slope, rtol=NET.rtol, atol=NET.atol
    )
    assert seen[-1] == pytest.approx(PEAK_LR, rel=F32.rtol)


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_decay_never_rises_and_reaches_the_minimum(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host)
    seen = run_steps(cb, host, TOTAL_STEPS + 4)
    decay = seen[WARMUP_STEPS - 1 :]
    assert all(a >= b - F32.atol for a, b in zip(decay, decay[1:], strict=False))
    assert seen[WARMUP_STEPS + DECAY_STEPS - 2] == pytest.approx(MIN_LR, rel=F32.rtol)
    assert seen[-1] == pytest.approx(MIN_LR, rel=F32.rtol)


def test_the_three_decays_have_their_own_midpoints():
    # same end points, different shapes: the middle of the cosine and of the straight line is the
    # mean of the two ends, the middle of the geometric progression is their geometric mean
    mid = WARMUP_STEPS + (DECAY_STEPS - 1) // 2
    values = {}
    for kind in DECAY_KINDS:
        host = Host(PEAK_LR)
        cb = make(kind, host)
        values[kind] = run_steps(cb, host, mid)[-1]
    assert values["cosine_decay"] == pytest.approx((PEAK_LR + MIN_LR) / 2, rel=NET.rtol)
    assert values["linear_decay"] == pytest.approx((PEAK_LR + MIN_LR) / 2, rel=NET.rtol)
    assert values["exponential_decay"] == pytest.approx(
        math.sqrt(PEAK_LR * MIN_LR), rel=NET.rtol
    )


# --------------------------------------------------------------------------- on_train_begin
@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_training_begins_at_the_cold_rate_when_there_is_a_warmup(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host)
    cb.on_train_begin()
    assert host.lr == pytest.approx(COLD_LR, rel=F32.rtol)


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_training_without_warmup_keeps_the_optimizer_rate_until_the_first_batch(kind):
    host = Host(PEAK_LR)
    cb = make(kind, host, warmup_epochs=0)
    cb.on_train_begin()
    assert host.lr == pytest.approx(PEAK_LR, rel=F32.rtol)
    run_steps(cb, host, 1)
    assert MIN_LR < host.lr < PEAK_LR


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_resuming_after_the_warmup_leaves_the_optimizer_rate_alone(kind):
    host = Host(0.77, step=WARMUP_STEPS + 1)
    cb = make(kind, host)
    cb.on_train_begin()
    assert host.lr == pytest.approx(0.77, rel=F32.rtol)


@pytest.mark.parametrize("kind", DECAY_KINDS)
def test_resuming_inside_the_warmup_restarts_from_the_cold_rate(kind):
    host = Host(0.77, step=WARMUP_STEPS - 1)
    cb = make(kind, host)
    cb.on_train_begin()
    assert host.lr == pytest.approx(COLD_LR, rel=F32.rtol)


# --------------------------------------------------------------------------------- log file
def test_log_file_has_one_line_per_step_and_marks_the_end(tmp_path):
    host = Host(PEAK_LR)
    log_file = tmp_path / "lr.log"
    cb = LinearDecay(
        host,
        initial_lr=COLD_LR,
        decay_steps=DECAY_STEPS,
        min_lr=MIN_LR,
        warmup_target=PEAK_LR,
        warmup_steps=WARMUP_STEPS,
        logfile=str(log_file),
    )
    run_steps(cb, host, TOTAL_STEPS + 1)
    lines = log_file.read_text().splitlines()
    # step 0 (the construction) plus the steps 1..16
    assert len(lines) == TOTAL_STEPS + 2
    assert lines[0] == "LinearDecay: lr = 2.000e-01 for step     0"
    assert lines[WARMUP_STEPS].startswith("LinearDecay: lr = 2.000e+00 for step     6")
    assert "# training completed" not in lines[TOTAL_STEPS - 1]
    assert lines[TOTAL_STEPS].endswith("# training completed")
    assert lines[TOTAL_STEPS + 1].endswith("# training completed")


def test_no_log_file_is_written_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host = Host(PEAK_LR)
    run_steps(make("cosine_decay", host), host, 3)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------------ the factory
@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("linear_decay", LinearDecay),
        ("cosine_decay", CosineDecay),
        ("exponential_decay", ExponentialDecay),
        ("reduce_on_plateau", CustomReduceLROnPlateau),
    ],
)
def test_factory_builds_the_class_named_in_the_input(kind, cls):
    host = Host(PEAK_LR)
    fit = fit_section(kind)
    cb = LRSchedulerFactory.create_lr_scheduler(
        host, N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=1)
    )
    assert type(cb) is cls
    assert cb.model is host


def test_factory_converts_epochs_to_steps_and_reads_the_rates():
    cb = make("cosine_decay", Host(PEAK_LR))
    assert cb.warmup_steps == WARMUP_STEPS
    assert cb.decay_steps == DECAY_STEPS
    assert cb.total_steps == TOTAL_STEPS
    assert cb.initial_lr == COLD_LR
    assert cb.warmup_target == PEAK_LR
    assert cb.min_lr == MIN_LR


def test_factory_counts_the_batches_of_one_replica():
    # 6 batches over 2 replicas are 3 steps per epoch, so the same step counts as above
    fit = fit_section("linear_decay")
    cb = LRSchedulerFactory.create_lr_scheduler(
        Host(PEAK_LR), 2 * N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=2)
    )
    assert (cb.warmup_steps, cb.decay_steps) == (WARMUP_STEPS, DECAY_STEPS)


def test_factory_rounds_a_fractional_warmup_down_to_whole_steps():
    cb = make("linear_decay", Host(PEAK_LR), warmup_epochs=1.5)
    assert cb.warmup_steps == 4  # int(1.5 * 3)
    assert cb.decay_steps == MAXITER * N_BATCHES - 4


def test_exponential_decay_rate_is_the_ratio_of_the_two_end_rates():
    cb = make("exponential_decay", Host(PEAK_LR))
    assert cb.decay_rate == pytest.approx(MIN_LR / PEAK_LR)


def test_factory_defaults_for_the_decay_parameters():
    fit = {
        "opt_params": {"learning_rate": 0.5},
        "scheduler": "linear_decay",
        "scheduler_params": {},
    }
    cb = LRSchedulerFactory.create_lr_scheduler(
        Host(0.5), 4, fit, SimpleNamespace(num_replicas_in_sync=1)
    )
    assert cb.min_lr == 1e-4
    assert cb.initial_lr == 1e-7
    assert cb.warmup_steps == 0
    assert cb.decay_steps == 500 * 4  # default maxiter


def test_factory_rejects_a_negative_warmup():
    with pytest.raises(AssertionError, match="warmup_epochs"):
        make("linear_decay", Host(PEAK_LR), warmup_epochs=-1)


def test_factory_without_scheduler_returns_none():
    fit = {"opt_params": {"learning_rate": 0.5}}
    assert not LRSchedulerFactory.if_lr_scheduler(fit)
    assert (
        LRSchedulerFactory.create_lr_scheduler(
            Host(0.5), 4, fit, SimpleNamespace(num_replicas_in_sync=1)
        )
        is None
    )


def test_factory_refuses_an_unknown_scheduler():
    with pytest.raises(
        ValueError, match="Unsupported learning rate scheduler: step_decay"
    ):
        make("step_decay", Host(PEAK_LR))


def test_factory_refuses_old_and_new_scheduler_together():
    fit = fit_section("linear_decay", learning_rate_reduction={"patience": 1})
    with pytest.raises(ValueError, match="both old and new scheduler"):
        LRSchedulerFactory.create_lr_scheduler(
            Host(PEAK_LR), N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=1)
        )


def test_legacy_reduction_section_builds_a_plateau_callback_with_a_warning(caplog):
    fit = {
        "opt_params": {"learning_rate": 1.0},
        "learning_rate_reduction": {
            "patience": 1,
            "factor": 0.1,
            "min": 1.0e-3,
            "stop_at_min": True,
        },
    }
    with caplog.at_level("WARNING"):
        cb = LRSchedulerFactory.create_lr_scheduler(
            Host(1.0), N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=1)
        )
    assert type(cb) is CustomReduceLROnPlateau
    assert any(
        "learning_rate_reduction is now deprecated" in r.message for r in caplog.records
    )
    assert cb.patience == 2  # the factory adds one to the configured patience
    assert cb.factor == 0.1
    assert cb.min_lr == 1e-3
    assert cb.monitor == "test_loss"
    assert cb.stop_on_min_lr is True


def test_plateau_section_defaults_and_adjusted_patience():
    fit = {
        "opt_params": {"learning_rate": 1.0},
        "scheduler": "reduce_on_plateau",
        "scheduler_params": {
            "patience": 3,
            "reduction_factor": 0.5,
            "monitor": "train_loss",
        },
    }
    cb = LRSchedulerFactory.create_lr_scheduler(
        Host(1.0), N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=1)
    )
    assert cb.patience == 4
    assert cb.factor == 0.5
    assert cb.monitor == "train_loss"
    assert cb.min_lr == 1e-4  # default minimum_learning_rate
    assert cb.cooldown == 0
    assert cb.stop_on_min_lr is False


# ------------------------------------------------------------------------------------ plateau
def plateau(host: Host, **kwargs) -> CustomReduceLROnPlateau:
    kwargs.setdefault("factor", 0.1)
    kwargs.setdefault("patience", 2)
    kwargs.setdefault("min_lr", 1e-3)
    kwargs.setdefault("monitor", "test_loss")
    return CustomReduceLROnPlateau(host, **kwargs)


def feed(cb, host: Host, losses: list[float | None]) -> list[float]:
    seen = []
    for epoch, loss in enumerate(losses, start=1):
        cb.on_epoch_end(epoch, {} if loss is None else {"test_loss": loss})
        seen.append(host.lr)
    return seen


def test_plateau_reduces_after_patience_epochs_without_improvement():
    host = Host(1.0)
    cb = plateau(host)
    # 10 and 9 improve; 9.5 is the first epoch without improvement, 9.6 the second: reduce by
    # 0.1 (1.0 -> 0.1); the counter restarts, so 9.7 and 9.8 reduce again (0.1 -> 0.01)
    seen = feed(cb, host, [10.0, 9.0, 9.5, 9.6, 9.7, 9.8])
    np.testing.assert_allclose(
        seen, [1.0, 1.0, 1.0, 0.1, 0.1, 0.01], rtol=F32.rtol, atol=F32.atol
    )


def test_plateau_improvement_resets_the_wait():
    host = Host(1.0)
    cb = plateau(host)
    seen = feed(cb, host, [10.0, 11.0, 9.0, 11.0, 8.0, 11.0])
    np.testing.assert_allclose(seen, [1.0] * 6, rtol=F32.rtol)


def test_plateau_never_goes_below_the_minimum_and_keeps_training_by_default():
    host = Host(1.0)
    cb = plateau(host, patience=1, factor=0.1, min_lr=0.05)
    seen = feed(cb, host, [1.0] + [2.0] * 6)
    # 1.0 -> 0.1 -> max(0.01, 0.05) = 0.05, then it stays
    np.testing.assert_allclose(
        seen, [1.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05], rtol=F32.rtol
    )
    assert host.stop_training is False


def test_plateau_stops_when_the_minimum_was_reached_and_stalls_again():
    host = Host(1.0)
    cb = plateau(host, patience=1, factor=0.1, min_lr=0.05, stop_on_min_lr=True)
    feed(cb, host, [1.0, 2.0, 2.0])
    assert host.lr == pytest.approx(0.05, rel=F32.rtol)
    assert host.stop_training is False
    feed(cb, host, [2.0])
    assert host.stop_training is True


def test_plateau_cooldown_pauses_the_counting_after_a_reduction():
    host = Host(1.0)
    cb = plateau(host, patience=1, factor=0.1, cooldown=3)
    seen = feed(cb, host, [1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # reduce at epoch 2 (1.0 -> 0.1); the counter of 3 is decremented before it is tested, so
    # epochs 3 and 4 are skipped; epoch 5 counts again and reduces (0.1 -> 0.01), epoch 6 is
    # the first of the next cooldown. Without a cooldown it would reduce at every epoch.
    np.testing.assert_allclose(seen, [1.0, 0.1, 0.1, 0.1, 0.01, 0.01], rtol=F32.rtol)
    no_cooldown = Host(1.0)
    seen = feed(
        plateau(no_cooldown, patience=1, factor=0.1), no_cooldown, [1.0, 2.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(seen, [1.0, 0.1, 0.01, 0.001], rtol=F32.rtol)


def test_plateau_skips_an_epoch_whose_log_lacks_the_monitored_quantity(caplog):
    host = Host(1.0)
    cb = plateau(host)
    with caplog.at_level("WARNING"):
        seen = feed(cb, host, [None, None, None, None])
    assert seen == [1.0] * 4
    assert any("test_loss not found in logs" in r.message for r in caplog.records)


def test_plateau_reset_best_forgets_the_best_value():
    host = Host(1.0)
    cb = plateau(host)
    feed(cb, host, [1.0])
    assert cb.best == 1.0
    cb.reset_best()
    assert cb.best == 1e99
    cb.on_epoch_end(2, {"test_loss": 5.0})
    assert cb.best == 5.0


def test_plateau_follows_the_factory_trace_of_the_integration_input():
    # the section of tests/MoNbTaW-LINEAR/input_lr_reduce_on_plateau_new_api.yaml, patience 1
    fit = {
        "opt_params": {"learning_rate": 1.0},
        "scheduler": "reduce_on_plateau",
        "scheduler_params": {
            "patience": 1.0,
            "reduction_factor": 0.1,
            "minimum_learning_rate": 1.0e-3,
            "resume_lr": True,
        },
    }
    host = Host(1.0)
    cb = LRSchedulerFactory.create_lr_scheduler(
        host, N_BATCHES, fit, SimpleNamespace(num_replicas_in_sync=1)
    )
    seen = feed(cb, host, [5.0, 4.0, 4.5, 4.6, 4.7, 4.8])
    np.testing.assert_allclose(seen, [1.0, 1.0, 1.0, 0.1, 0.1, 0.01], rtol=F32.rtol)


# ---- the plateau callback under Keras 3 (DEPS3): the direction of ``mode="auto"``, ``min_delta``, the first best value
#
# Keras 3 sets the monitor operation lazily inside its own ``on_epoch_end`` and resets ``best`` to ``None`` in
# ``on_train_begin``; the callback keeps its own direction and initial best value. Rates are exactly representable
# in float32 (the optimizer keeps its learning rate in a float32 variable).

PLATEAU_START_LR = 0.5
PLATEAU_FACTOR = 0.5


def keras3_plateau(**kwargs) -> tuple[CustomReduceLROnPlateau, Host]:
    host = Host(PLATEAU_START_LR)
    options = {"factor": PLATEAU_FACTOR, "patience": 1, "mode": "min", **kwargs}
    callback = CustomReduceLROnPlateau(host, **options)
    callback.on_train_begin()  # the training loop calls it; Keras 3 resets its state there
    return callback, host


def monitored_rates(callback, host: Host, values: list[float], key: str) -> list[float]:
    rates = []
    for epoch, value in enumerate(values):
        callback.on_epoch_end(epoch, {key: value})
        rates.append(host.lr)
    return rates


def test_plateau_a_smaller_improvement_than_min_delta_does_not_count():
    callback, host = keras3_plateau(min_delta=0.1)

    assert monitored_rates(callback, host, [1.0, 0.95], "train_loss") == [0.5, 0.25]


def test_plateau_auto_mode_minimises_a_loss_and_maximises_an_accuracy():
    loss, loss_host = keras3_plateau(mode="auto", monitor="train_loss")
    accuracy, accuracy_host = keras3_plateau(mode="auto", monitor="val_acc")

    # a falling loss and a rising accuracy keep improving: no reduction
    assert monitored_rates(loss, loss_host, [1.0, 0.5, 0.25], "train_loss") == [PLATEAU_START_LR] * 3
    assert monitored_rates(accuracy, accuracy_host, [0.25, 0.5, 0.75], "val_acc") == [PLATEAU_START_LR] * 3


def test_plateau_auto_mode_reduces_when_a_loss_rises_and_when_an_accuracy_falls():
    loss, loss_host = keras3_plateau(mode="auto", monitor="train_loss")
    accuracy, accuracy_host = keras3_plateau(mode="auto", monitor="val_acc")

    assert monitored_rates(loss, loss_host, [1.0, 2.0], "train_loss")[-1] == 0.25
    assert monitored_rates(accuracy, accuracy_host, [0.5, 0.25], "val_acc")[-1] == 0.25


def test_plateau_a_new_callback_has_a_best_value_that_any_loss_improves_on():
    callback, host = keras3_plateau()

    assert callback.best >= 1e99
    assert monitored_rates(callback, host, [1e6], "train_loss") == [PLATEAU_START_LR]


def test_plateau_a_best_value_of_zero_after_construction_is_replaced_by_a_huge_one():
    class StartsAtZero(CustomReduceLROnPlateau):
        """The parent state at the end of its constructor, set to the one value the override rewrites."""

        def _reset(self):
            super()._reset()
            self.best = 0.0

    callback = StartsAtZero(Host(PLATEAU_START_LR), factor=PLATEAU_FACTOR, patience=2, mode="min")

    assert callback.best == 1e99
