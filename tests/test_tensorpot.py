"""Characterization tests for ``TensorPotential.__init__`` (``tensorpotential/tensorpot.py``).

Logic layer: the defaults of an object built without a fit configuration, the Adam branch with
and without ``use_ema``, the batch-size fallback, the output paths, and the eager-mode option
that overrides ``jit_compile``. Value layer: with zero gradients Adam leaves a weight unchanged
and decoupled weight decay multiplies it by ``1 - lr * wd``; only the variables named in
``TensorPotential.WD_ALLOW_PATTERNS`` may shrink. That hand-derived factor does not call the
unit under test, which only decides which variables are excluded.
"""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np
import pytest

from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential

import tensorflow as tf  # after tensorpotential, which must set TF_USE_LEGACY_KERAS first

from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LEARNING_RATE = 1.0
WEIGHT_DECAY = 0.25


def _instructions():
    """A fresh, cheap instruction list (instructions are built once and cannot be shared)."""
    preset = get_preset("GRACE_2LAYER_v1_24")
    return preset(element_map={"Be": 0, "Li": 1}, lmax=0).get_instructions()


def _tp(**kwargs) -> TensorPotential:
    return TensorPotential(_instructions(), param_dtype=tf.float64, **kwargs)


def _is_traced(function) -> bool:
    """``True`` for a ``tf.function``-wrapped callable, ``False`` for a plain method."""
    return hasattr(function, "get_concrete_function")


def test_without_fit_config_there_is_no_optimizer():
    tp = _tp()

    assert tp.fit_config == {}
    assert tp.optimizer is None
    assert not hasattr(tp, "use_ema")
    assert not hasattr(tp.checkpoint, "optimizer")
    assert tp.stop_training is False
    assert tp.float_dtype == tf.float64
    assert tp.param_dtype == tf.float64


def test_default_model_functions_are_the_standard_ones():
    from tensorpotential.tpmodel import (
        ComputeBatchEnergyAndForces,
        ComputeStructureEnergyAndForcesAndVirial,
    )

    tp = _tp()

    assert isinstance(
        tp.model.compute_function, ComputeStructureEnergyAndForcesAndVirial
    )
    assert isinstance(tp.model.train_function, ComputeBatchEnergyAndForces)


def test_test_batch_size_falls_back_to_the_batch_size():
    assert _tp(global_batch_size=8).global_test_batch_size == 8
    assert (
        _tp(global_batch_size=8, global_test_batch_size=4).global_test_batch_size == 4
    )
    assert _tp().global_test_batch_size is None


@pytest.mark.parametrize("seed", [42, 7])
def test_output_paths_follow_the_seed(seed):
    tp = _tp(seed=seed)

    assert tp.seed == seed
    assert tp.output_dir == os.path.join("seed", str(seed))
    assert tp.checkpoint_dir == os.path.join("seed", str(seed), "checkpoints")
    assert tp.checkpoint_prefix == os.path.join(tp.checkpoint_dir, "checkpoint")


def test_adam_without_options_uses_the_default_parameters():
    tp = _tp(fit_config={"optimizer": "Adam"})

    assert isinstance(tp.optimizer, tf.keras.optimizers.Adam)
    assert tp.optimizer.learning_rate.numpy() == np.float32(0.01)
    assert tp.optimizer.amsgrad is True
    assert tp.use_ema is False
    assert tp.swap_on_epoch is False
    assert not hasattr(tp, "_ema_weights_in_model")
    assert tp.checkpoint.optimizer is tp.optimizer


def test_adam_with_ema_swaps_weights_on_epoch():
    tp = _tp(
        fit_config={
            "optimizer": "Adam",
            "opt_params": {"learning_rate": 1e-3, "use_ema": True},
        }
    )

    assert tp.optimizer.learning_rate.numpy() == np.float32(1e-3)
    assert tp.use_ema is True
    assert tp.swap_on_epoch is True
    assert tp._ema_weights_in_model is False


def test_adam_options_without_use_ema_mean_no_ema():
    tp = _tp(fit_config={"optimizer": "Adam", "opt_params": {"learning_rate": 0.5}})

    assert tp.optimizer.learning_rate.numpy() == np.float32(0.5)
    assert tp.use_ema is False
    assert tp.swap_on_epoch is False


def test_an_unknown_optimizer_name_builds_no_optimizer():
    tp = _tp(fit_config={"optimizer": "BFGS"})

    assert tp.optimizer is None
    assert not hasattr(tp.checkpoint, "optimizer")


def test_weight_decay_shrinks_only_the_allowed_variables():
    tp = _tp(
        fit_config={
            "optimizer": "Adam",
            "opt_params": {
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "amsgrad": True,
            },
        }
    )
    variables = tp.model.variables_to_train
    before = {v.name: v.numpy().copy() for v in variables}
    decayed = [v for v in variables if "reducing_" in v.name]
    kept = [v for v in variables if "reducing_" not in v.name]
    assert decayed
    assert kept

    # zero gradient: Adam moves nothing, only the decoupled decay acts: w -> w (1 - lr wd)
    tp.optimizer.apply_gradients([(tf.zeros_like(v), v) for v in variables])

    factor = 1.0 - LEARNING_RATE * WEIGHT_DECAY
    for v in decayed:
        np.testing.assert_allclose(
            v.numpy(), before[v.name] * factor, rtol=ARITH.rtol, atol=ARITH.atol
        )
    for v in kept:
        np.testing.assert_array_equal(v.numpy(), before[v.name])


def test_eager_mode_overrides_jit_compile_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        tp = _tp(eager_mode=True, jit_compile=True)

    assert any("overwrite `jit_compile`" in record.message for record in caplog.records)
    assert not _is_traced(tp.predict)
    assert not _is_traced(tp.model_grad)


def test_eager_mode_without_jit_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        tp = _tp(eager_mode=True, jit_compile=False)

    assert not [r for r in caplog.records if "jit_compile" in r.message]
    assert not _is_traced(tp.predict)


def test_graph_mode_decorates_the_step_functions():
    tp = _tp()

    assert _is_traced(tp.predict)
    assert _is_traced(tp.model_grad)
    assert _is_traced(tp.distributed_train_step)


# ------------------------------------------------- default functions are not shared


def test_each_tensorpotential_gets_its_own_default_functions():
    first, second = _tp(), _tp()

    assert first.model.compute_function is not second.model.compute_function
    assert first.model.train_function is not second.model.train_function


def test_explicit_functions_are_kept_by_identity():
    from tensorpotential.tpmodel import (
        ComputeBatchEnergyForcesVirials,
        ComputeStructureEnergyAndForcesAndVirial,
    )

    compute = ComputeStructureEnergyAndForcesAndVirial()
    train = ComputeBatchEnergyForcesVirials()
    tp = _tp(model_compute_function=compute, model_train_function=train)

    assert tp.model.compute_function is compute
    assert tp.model.train_function is train


def test_function_arguments_default_to_none():
    parameters = inspect.signature(TensorPotential.__init__).parameters

    names = ["model_compute_function", "model_train_function"]
    assert [parameters[name].default for name in names] == [None, None]
