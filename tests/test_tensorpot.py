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
import warnings

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk

from tensorpotential.calculator import TPCalculator
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.optimizer_layout import LegacyOptimizerStateWarning
from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential

import tensorflow as tf  # after tensorpotential, which applies the TensorFlow options first

from tests.tolerances import FINITE_DIFFERENCE_F64 as FD
from tests.legacy_keras_checkpoint import (
    LEGACY_CHECKPOINT,
    LEGACY_EPOCH,
    LEGACY_OPTIONS,
    LEGACY_STEPS,
    legacy_weights,
    small_tp,
)
from tests.tolerances import FLOAT32_ELEMENTWISE as F32
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FD_STEP = 1e-4  # A; the step assumed by FINITE_DIFFERENCE_F64 in tests/tolerances.py
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


# ---------------------------------------------------- param_dtype default (a code default of param_dtype)


def test_param_dtype_default_is_float32():
    # PINNED, not endorsed (see test_metadata_utils.py): float32 here, float64 for a yaml without the key
    assert inspect.signature(TensorPotential.__init__).parameters["param_dtype"].default is tf.float32


def test_default_param_dtype_gives_float32_weights_and_a_float64_input_dtype():
    tp = TensorPotential(_instructions())

    assert tp.param_dtype is tf.float32
    assert tp.float_dtype is tf.float64
    trainable = tp.model.trainable_variables
    assert trainable
    assert {v.dtype for v in trainable} == {tf.float32}


# ---------------------------------------------- LoRA at the level of TensorPotential


def test_is_lora_enabled_follows_the_model():
    tp = _tp()
    assert not tp.is_lora_enabled()

    tp.model.enable_lora_adaptation({"Z": {"rank": 2, "alpha": 1}})

    assert tp.is_lora_enabled()


LORA_CONFIG = {"all": {"rank": 2, "alpha": 1}}


def _lora_tp(**kwargs) -> TensorPotential:
    """A four-element float64 model whose LoRA tensors can be perturbed (B starts at zero)."""
    instructions = get_preset("GRACE_2LAYER_v1_24")(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0
    ).get_instructions()
    return TensorPotential(instructions, param_dtype=tf.float64, **kwargs)


def _perturb_lora_tensors(tp: TensorPotential, seed: int = 0) -> None:
    """Give every update tensor a small nonzero value (the zero-initialised ones included)."""
    rng = np.random.default_rng(seed)
    for variable in tp.model.trainable_variables:
        variable.assign(variable + 0.1 * rng.standard_normal(variable.shape))


def _structure() -> Atoms:
    atoms = bulk("Mo", cubic=True) * (2, 1, 1)
    atoms.set_chemical_symbols(["Mo", "Nb", "Ta", "W"])
    atoms.rattle(stdev=0.05, seed=3)
    return atoms


def _energy_forces(tp: TensorPotential, atoms: Atoms | None = None):
    atoms = (atoms or _structure()).copy()
    atoms.calc = TPCalculator(model=tp.model)
    return atoms.get_potential_energy(), atoms.get_forces()


def _variables_by_name(tp: TensorPotential) -> dict[str, np.ndarray]:
    return {v.name: v.numpy().copy() for v in tp.model.variables if v.dtype == tf.float64}


def test_enabling_lora_makes_the_checkpoint_save_the_update_tensors(tmp_path):
    tp = _lora_tp(fit_config={"optimizer": "Adam"})
    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    prefix = str(tmp_path / "checkpoint")

    tp.save_checkpoint(checkpoint_name=prefix)

    saved = {name for name, _ in tf.train.list_variables(prefix)}
    assert any("lora_tensors" in name for name in saved)


def test_enabling_and_reducing_lora_keep_the_step_and_epoch_counters():
    tp = _lora_tp(fit_config={"optimizer": "Adam"})  # an optimizer, so that the reduction rebuilds the checkpoint
    tp.step, tp.epoch = 7, 3

    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    assert (tp.step, tp.epoch) == (7, 3)

    tp.finalize_lora_update()
    assert (tp.step, tp.epoch) == (7, 3)


def test_reducing_lora_resets_the_optimizer_and_the_mid_epoch_flag():
    tp = _lora_tp(fit_config={"optimizer": "Adam"})
    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    tp.intra_epoch_save = True
    optimizer = tp.optimizer

    tp.finalize_lora_update()

    assert tp.optimizer is not optimizer
    assert not tp.intra_epoch_save


def test_a_freshly_activated_lora_model_equals_the_base_model():
    # physical value: the B matrices start at zero, so the update tensors add exactly nothing
    tp = _lora_tp()
    e_base, f_base = _energy_forces(tp)

    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    e_lora, f_lora = _energy_forces(tp)

    assert e_lora == pytest.approx(e_base, rel=ARITH.rtol, abs=ARITH.atol)
    np.testing.assert_allclose(f_lora, f_base, rtol=ARITH.rtol, atol=ARITH.atol)


@pytest.mark.parametrize("config", [LORA_CONFIG, {"all": {"mode": "full_additive"}}], ids=["lora", "additive"])
def test_a_reduced_lora_model_reproduces_the_activated_model(config):
    # physical value: merging W + delta_W into W is the same function as keeping delta_W apart
    tp = _lora_tp()
    tp.enable_lora_adaptation(dict(config))
    _perturb_lora_tensors(tp)
    e_active, f_active = _energy_forces(tp)

    tp.finalize_lora_update()
    e_reduced, f_reduced = _energy_forces(tp)

    assert not tp.is_lora_enabled()
    assert e_reduced == pytest.approx(e_active, rel=ARITH.rtol, abs=ARITH.atol)
    np.testing.assert_allclose(f_reduced, f_active, rtol=ARITH.rtol, atol=ARITH.atol)


def test_perturbing_the_update_tensors_changes_the_energy():
    # guards the two tests above against comparing two unchanged models
    tp = _lora_tp()
    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    e_zero, _ = _energy_forces(tp)

    _perturb_lora_tensors(tp)

    assert _energy_forces(tp)[0] != pytest.approx(e_zero, rel=1e-3)


def test_forces_of_an_activated_lora_model_match_finite_differences_of_its_energy():
    tp = _lora_tp()
    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    _perturb_lora_tensors(tp)
    atoms = _structure()
    _, forces = _energy_forces(tp, atoms)

    atom, axis, step = 1, 0, FD_STEP
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[atom, axis] += step
    minus.positions[atom, axis] -= step
    derivative = -(_energy_forces(tp, plus)[0] - _energy_forces(tp, minus)[0]) / (2 * step)

    assert forces[atom, axis] == pytest.approx(derivative, rel=FD.rtol, abs=FD.atol)


def test_the_lora_variables_survive_a_checkpoint_round_trip(tmp_path):
    tp = _lora_tp()
    tp.enable_lora_adaptation(dict(LORA_CONFIG))
    _perturb_lora_tensors(tp)
    tp.save_checkpoint(checkpoint_name=str(tmp_path / "checkpoint"))
    yaml_path = str(tmp_path / "model.yaml")
    save_instructions_dict(yaml_path, tp.model.instructions, param_dtype=tf.float64)

    restored = TensorPotential(load_instructions(yaml_path), param_dtype=tf.float64)
    restored.load_checkpoint(checkpoint_name=str(tmp_path / "checkpoint"), assert_consumed=False)

    assert restored.is_lora_enabled()
    expected, found = _variables_by_name(tp), _variables_by_name(restored)
    assert expected.keys() == found.keys()
    assert any("LORA" in name for name in found)
    assert all(np.array_equal(expected[k], found[k]) for k in expected)
    assert _energy_forces(restored)[0] == pytest.approx(_energy_forces(tp)[0], rel=ARITH.rtol)



# ------------------------------------------------------ the optimizer, against a numpy oracle
#
# Characterization of what the training loop relies on: the Adam update (with and without amsgrad), the
# decoupled weight decay on the variables of ``WD_ALLOW_PATTERNS``, the exponential moving average (EMA) and
# its swap with the weights, and the optimizer state in a checkpoint. The expected values are computed by
# hand in numpy from the documented update rules, never by the optimizer under test.

ADAM_LR = 0.5  # exactly representable: the optimizer keeps its learning rate in a float32 variable
BETA_1, BETA_2, EPSILON = 0.9, 0.999, 1e-7  # the optimizer defaults


def _through_float32(x: float) -> float:
    """``x`` rounded to float32: what ``cast(python float, float64)`` gives in the bias correction.

    The optimizer casts a Python float to the variable dtype through a float32 tensor, so its bias-correction
    powers use ``float32(0.9)`` and ``float32(0.999)`` (relative error 6e-6 on ``sqrt(1 - beta_2)``), while
    the moment updates use the exact Python floats. Both Keras versions do this; the oracle reproduces it.
    """
    return float(np.float32(x))
EMA_MOMENTUM = 0.9
# the gradient vanishes after the first step: the second moment then decays (by BETA_2 per step) while amsgrad
# keeps the largest one seen, so the two variants differ (a growing gradient would make them equal)
GRADIENT_SCALES = (1.0, 0.0, 0.0)


def _gradients(variables, step: int) -> list[np.ndarray]:
    """Reproducible gradients of one step: they depend on the step and the variable position only."""
    scale = GRADIENT_SCALES[step]
    return [
        scale * np.random.default_rng([11, step, i]).standard_normal(v.shape)
        for i, v in enumerate(variables)
    ]


def _apply_step(tp: TensorPotential, step: int) -> None:
    variables = tp.model.variables_to_train
    grads = [tf.constant(g, dtype=v.dtype) for g, v in zip(_gradients(variables, step), variables)]
    tp.optimizer.apply_gradients(zip(grads, variables))


def _adam_oracle(w0, grads, *, amsgrad: bool, decay: float = 0.0) -> list[np.ndarray]:
    """Weights after each step of Adam: bias-corrected moments, optional amsgrad and decoupled decay.

    The decay acts on the weights before the step, scaled by the learning rate: ``w -> w (1 - lr wd)``.
    """
    w = np.array(w0, dtype=np.float64)
    m, v, v_max = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    out = []
    for t, g in enumerate(grads, start=1):
        w = w - w * ADAM_LR * decay
        m = BETA_1 * m + (1 - BETA_1) * g
        v = BETA_2 * v + (1 - BETA_2) * g * g
        v_max = np.maximum(v_max, v)
        alpha = ADAM_LR * np.sqrt(1 - _through_float32(BETA_2) ** t) / (1 - _through_float32(BETA_1) ** t)
        w = w - alpha * m / (np.sqrt(v_max if amsgrad else v) + EPSILON)
        out.append(w)
    return out


def _adam_tp(**opt_params) -> TensorPotential:
    params = {"learning_rate": ADAM_LR, "amsgrad": True, **opt_params}
    return _tp(fit_config={"optimizer": "Adam", "opt_params": params})


def _ema_tp(**opt_params) -> TensorPotential:
    return _adam_tp(use_ema=True, ema_momentum=EMA_MOMENTUM, **opt_params)


def _flat(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrays])


def _weights(tp: TensorPotential) -> np.ndarray:
    return _flat(v.numpy() for v in tp.model.variables_to_train)


def _averages(tp: TensorPotential) -> np.ndarray:
    return _flat(a.numpy() for a in tp._get_ema_optimizer()._model_variables_moving_average)


def _assert_close(found, expected) -> None:
    np.testing.assert_allclose(found, expected, rtol=ARITH.rtol, atol=ARITH.atol)


def test_the_gradient_sequence_tells_amsgrad_from_plain_adam():
    w0 = np.ones(5)
    grads = [np.full(5, s) for s in GRADIENT_SCALES]

    plain = _adam_oracle(w0, grads, amsgrad=False)[-1]
    amsgrad = _adam_oracle(w0, grads, amsgrad=True)[-1]

    assert not np.allclose(plain, amsgrad, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_adam_steps_follow_the_numpy_oracle(amsgrad):
    tp = _adam_tp(amsgrad=amsgrad)
    variables = tp.model.variables_to_train
    start = [v.numpy().copy() for v in variables]
    steps = range(len(GRADIENT_SCALES))
    grads = [_gradients(variables, s) for s in steps]
    expected = [_adam_oracle(w, [g[i] for g in grads], amsgrad=amsgrad) for i, w in enumerate(start)]

    for s in steps:
        _apply_step(tp, s)
        _assert_close(_weights(tp), _flat(e[s] for e in expected))
    assert int(tp.optimizer.iterations.numpy()) == len(GRADIENT_SCALES)


def test_weight_decay_acts_before_the_step_and_only_on_the_allowed_variables():
    wd = 0.25
    tp = _adam_tp(weight_decay=wd)
    variables = tp.model.variables_to_train
    start = [v.numpy().copy() for v in variables]
    grads = [_gradients(variables, s) for s in range(len(GRADIENT_SCALES))]
    decayed = ["reducing_" in v.name for v in variables]
    assert any(decayed)
    assert not all(decayed)

    for s in range(len(GRADIENT_SCALES)):
        _apply_step(tp, s)
    expected = [
        _adam_oracle(w, [g[i] for g in grads], amsgrad=True, decay=wd if is_decayed else 0.0)[-1]
        for i, (w, is_decayed) in enumerate(zip(start, decayed))
    ]

    _assert_close(_weights(tp), _flat(expected))


def _ema_after_each_step(tp: TensorPotential) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """The weights before the first step, the average after the last step, and the weights after each step."""
    start = _weights(tp)
    after = []
    for s in range(len(GRADIENT_SCALES)):
        _apply_step(tp, s)
        after.append(_weights(tp))
    return start, _averages(tp), after


def test_the_average_follows_the_keras_3_recursion_that_starts_at_the_first_updated_weights():
    # Keras 3: the first update uses momentum 0 (average = weights after step 1), then
    # average -> momentum * average + (1 - momentum) * weights. DOCUMENTED CHANGE from legacy Keras 2, see below.
    start, average, after = _ema_after_each_step(_ema_tp())

    expected = after[0]
    for weights in after[1:]:
        expected = EMA_MOMENTUM * expected + (1 - EMA_MOMENTUM) * weights

    _assert_close(average, expected)


def test_the_average_differs_from_the_legacy_keras_one_by_a_closed_form_that_decays_with_the_momentum():
    # legacy Keras 2 started the recursion from the initial weights: avg_1 = m w_0 + (1 - m) w_1, so after n steps
    # the legacy and the Keras 3 averages differ by exactly m**n (w_0 - w_1) (release notes of DEPS3).
    start, average, after = _ema_after_each_step(_ema_tp())

    legacy = start
    for weights in after:
        legacy = EMA_MOMENTUM * legacy + (1 - EMA_MOMENTUM) * weights

    n = len(GRADIENT_SCALES)
    _assert_close(legacy - average, EMA_MOMENTUM**n * (start - after[0]))


# ---- the swap of weights and average


def _two_steps(tp: TensorPotential) -> None:
    for s in range(2):
        _apply_step(tp, s)


def test_swap_variables_exchanges_weights_and_average_and_swapping_twice_restores_them():
    tp = _ema_tp()
    _two_steps(tp)
    weights, average = _weights(tp), _averages(tp)
    assert not np.allclose(weights, average)

    tp.swap_variables()
    np.testing.assert_array_equal(_weights(tp), average)
    np.testing.assert_array_equal(_averages(tp), weights)

    tp.swap_variables()
    np.testing.assert_array_equal(_weights(tp), weights)
    np.testing.assert_array_equal(_averages(tp), average)


def test_the_on_device_swap_gives_the_same_exchange():
    tp = _ema_tp()
    _two_steps(tp)
    weights, average = _weights(tp), _averages(tp)

    tp._swap_variables_on_device(tp._get_ema_optimizer())

    np.testing.assert_array_equal(_weights(tp), average)
    np.testing.assert_array_equal(_averages(tp), weights)


@pytest.mark.parametrize("built", [False, True])
def test_swap_variables_without_ema_is_refused(built):
    tp = _adam_tp()
    if built:
        _two_steps(tp)

    with pytest.raises(ValueError, match="requires use_ema=True"):
        tp.swap_variables()


def test_swap_variables_before_the_first_step_is_refused_even_with_ema():
    tp = _ema_tp()

    with pytest.raises(ValueError, match="requires use_ema=True"):
        tp.swap_variables()


def test_the_ema_optimizer_is_the_optimizer_or_the_one_it_wraps():
    tp = _ema_tp()
    assert tp._get_ema_optimizer() is tp.optimizer

    class Wrapper:  # stands in for a loss-scale wrapper: only the attribute name matters
        inner_optimizer = tp.optimizer

    tp.optimizer = Wrapper()
    assert tp._get_ema_optimizer() is Wrapper.inner_optimizer


def test_ema_scope_shows_the_average_inside_and_restores_the_weights_after():
    tp = _ema_tp()
    _two_steps(tp)
    weights, average = _weights(tp), _averages(tp)

    with tp.ema_scope():
        np.testing.assert_array_equal(_weights(tp), average)
        assert tp._ema_weights_in_model is True

        with tp.ema_scope():  # nested: the inner scope does nothing
            np.testing.assert_array_equal(_weights(tp), average)
        np.testing.assert_array_equal(_weights(tp), average)

    np.testing.assert_array_equal(_weights(tp), weights)
    np.testing.assert_array_equal(_averages(tp), average)
    assert tp._ema_weights_in_model is False


def test_ema_scope_restores_the_weights_when_its_body_raises():
    tp = _ema_tp()
    _two_steps(tp)
    weights = _weights(tp)

    with pytest.raises(RuntimeError, match="boom"), tp.ema_scope():
        raise RuntimeError("boom")

    np.testing.assert_array_equal(_weights(tp), weights)
    assert tp._ema_weights_in_model is False


def test_ema_scope_does_nothing_without_ema_or_before_the_average_exists():
    for tp, steps in ((_adam_tp(), 2), (_ema_tp(), 0)):
        for s in range(steps):
            _apply_step(tp, s)
        weights = _weights(tp)

        with tp.ema_scope():
            np.testing.assert_array_equal(_weights(tp), weights)

        np.testing.assert_array_equal(_weights(tp), weights)


def test_prepare_for_training_swaps_the_average_out_of_the_model():
    tp = _ema_tp()
    _two_steps(tp)
    weights, average = _weights(tp), _averages(tp)
    tp._ema_weights_in_model = True  # a checkpoint stores the model holding the average

    tp.prepare_for_training()

    np.testing.assert_array_equal(_weights(tp), average)
    np.testing.assert_array_equal(_averages(tp), weights)
    assert tp._ema_weights_in_model is False


def test_prepare_for_training_does_nothing_without_ema_or_before_the_average_exists():
    for tp, steps in ((_adam_tp(), 2), (_ema_tp(), 0)):
        for s in range(steps):
            _apply_step(tp, s)
        weights = _weights(tp)

        tp.prepare_for_training()

        np.testing.assert_array_equal(_weights(tp), weights)


# ---- the optimizer is built before training, and an average is only swapped in after a first step


def test_prepare_for_training_builds_the_optimizer_variables():
    for tp in (_adam_tp(), _ema_tp()):
        assert not tp.optimizer.built

        tp.prepare_for_training() if tp.use_ema else tp._build_optimizer()

        assert tp.optimizer.built
        assert len(tp.optimizer.variables) > 2  # the slots exist before the first traced step


def test_a_built_optimizer_with_zero_averages_is_never_swapped_into_the_model():
    # Keras 3 creates the averages filled with zeros and sets them at the first update
    tp = _ema_tp()
    weights = _weights(tp)

    tp.prepare_for_training()

    assert not np.any(_averages(tp))  # the averages are still zero
    assert not tp._ema_ready(tp._get_ema_optimizer())
    np.testing.assert_array_equal(_weights(tp), weights)
    with tp.ema_scope():
        np.testing.assert_array_equal(_weights(tp), weights)
    with pytest.raises(ValueError, match="at least one optimizer step"):
        tp.swap_variables()


def test_the_average_is_ready_after_the_first_step():
    tp = _ema_tp()
    tp.prepare_for_training()

    _apply_step(tp, 0)

    assert tp._ema_ready(tp._get_ema_optimizer())


def test_building_an_optimizer_that_is_already_built_changes_nothing():
    tp = _adam_tp()
    _two_steps(tp)
    weights, iterations = _weights(tp), int(tp.optimizer.iterations.numpy())

    tp._build_optimizer()

    np.testing.assert_array_equal(_weights(tp), weights)
    assert int(tp.optimizer.iterations.numpy()) == iterations


def test_a_model_without_optimizer_has_nothing_to_build():
    tp = _tp()

    tp._build_optimizer()

    assert tp.optimizer is None


# ---- the optimizer state in a checkpoint


def _checkpoint_path(tmp_path) -> str:
    return str(tmp_path / "ckpt" / "checkpoint")


def test_a_checkpoint_restores_weights_counters_and_the_optimizer_state(tmp_path):
    saved = _ema_tp(weight_decay=0.25)
    _two_steps(saved)
    saved.step, saved.epoch = 2, 1
    saved.save_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))

    restored = _ema_tp(weight_decay=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyOptimizerStateWarning)  # a matching layout is restored silently
        restored.load_checkpoint(checkpoint_name=_checkpoint_path(tmp_path), expect_partial=True)

    np.testing.assert_array_equal(_weights(restored), _weights(saved))
    assert (restored.step, restored.epoch) == (2, 1)
    assert int(restored.optimizer.iterations.numpy()) == 2
    # the moments came back: the next step of both objects is the same, bit for bit
    _apply_step(saved, 2)
    _apply_step(restored, 2)
    np.testing.assert_array_equal(_weights(restored), _weights(saved))
    np.testing.assert_array_equal(_averages(restored), _averages(saved))


def test_a_model_only_load_leaves_the_counters_and_the_optimizer_alone(tmp_path):
    saved = _adam_tp()
    _two_steps(saved)
    saved.step = 2
    saved.save_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))

    restored = _adam_tp()
    restored.load_checkpoint(checkpoint_name=_checkpoint_path(tmp_path), model_only=True)

    np.testing.assert_array_equal(_weights(restored), _weights(saved))
    assert restored.step == 0
    assert int(restored.optimizer.iterations.numpy()) == 0


def test_loading_a_checkpoint_that_does_not_exist(tmp_path, caplog):
    tp = _adam_tp()
    path = _checkpoint_path(tmp_path)
    before = _weights(tp)

    with pytest.raises(ValueError, match="No checkpoint index found"):
        tp.load_checkpoint(checkpoint_name=path)

    with caplog.at_level(logging.INFO):
        tp.load_checkpoint(checkpoint_name=path, raise_errors=False, verbose=True)
    assert any("FAILED" in r.message for r in caplog.records)
    np.testing.assert_array_equal(_weights(tp), before)


def test_strict_load_options_accept_a_matching_checkpoint(tmp_path):
    saved = _adam_tp()
    _two_steps(saved)
    saved.save_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))
    restored = _adam_tp()
    restored.optimizer.build(restored.model.variables_to_train)

    restored.load_checkpoint(
        checkpoint_name=_checkpoint_path(tmp_path), assert_existing_objects_matched=True, verbose=True
    )

    np.testing.assert_array_equal(_weights(restored), _weights(saved))


def test_assert_consumed_accepts_a_checkpoint_whose_objects_all_match(tmp_path):
    saved = _tp()
    saved.step = 4
    saved.save_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))
    restored = _tp()

    restored.load_checkpoint(checkpoint_name=_checkpoint_path(tmp_path), assert_consumed=True)

    assert restored.step == 4
    np.testing.assert_array_equal(_weights(restored), _weights(saved))


# ---- a checkpoint written with legacy Keras 2 by the tree before DEPS3
#
# The optimizer slots of such a checkpoint sit at other positions than the ones of a Keras 3 optimizer
# (tests/legacy_keras_checkpoint/make_fixture.py): they must never be assigned to the wrong variables.


def _restore_legacy(tp: TensorPotential, **options) -> None:
    with pytest.warns(LegacyOptimizerStateWarning, match="not restored"):
        tp.load_checkpoint(checkpoint_name=LEGACY_CHECKPOINT, **options)


def test_a_legacy_keras_checkpoint_restores_the_model_and_resets_the_optimizer_with_a_warning():
    tp = small_tp()
    stored = legacy_weights()
    assert not np.array_equal(_flat(stored), _weights(tp))

    _restore_legacy(tp)

    for variable, expected in zip(tp.model.variables_to_train, stored):
        np.testing.assert_array_equal(variable.numpy(), expected)
    assert (tp.step, tp.epoch) == (LEGACY_STEPS, LEGACY_EPOCH)
    assert int(tp.optimizer.iterations.numpy()) == 0
    assert not hasattr(tp._get_ema_optimizer(), "_model_variables_moving_average")


def test_the_warning_for_a_legacy_checkpoint_names_the_checkpoint_and_what_is_lost(caplog):
    tp = small_tp()

    with caplog.at_level(logging.WARNING), pytest.warns(LegacyOptimizerStateWarning) as caught:
        tp.load_checkpoint(checkpoint_name=LEGACY_CHECKPOINT)

    message = str(caught[0].message)
    assert LEGACY_CHECKPOINT in message
    assert "start from scratch" in message
    assert any("not restored" in r.message for r in caplog.records)  # also in the training log


def test_the_mid_epoch_flag_of_a_legacy_checkpoint_is_cleared_with_the_optimizer_state():
    tp = small_tp()

    _restore_legacy(tp)

    # the fixture was saved mid-epoch; a rewind would complete updates that lived in the discarded moments
    assert tp.intra_epoch_save is False
    assert tp.checkpoint.optimizer is tp.optimizer


def test_the_first_step_after_a_legacy_restore_is_a_fresh_adam_step_from_the_stored_weights():
    tp = small_tp()
    _restore_legacy(tp)
    variables = tp.model.variables_to_train
    gradients = [np.random.default_rng([3, i]).standard_normal(v.shape) for i, v in enumerate(variables)]

    tp.optimizer.apply_gradients(
        [(tf.constant(g, dtype=v.dtype), v) for g, v in zip(gradients, variables)]
    )

    # moments, step count and averages restart from zero (learning rate 0.5, weight decay 0.1, amsgrad)
    decay = LEGACY_OPTIONS["weight_decay"]
    for variable, w0, g in zip(variables, legacy_weights(), gradients):
        expected = _adam_oracle(w0, [g], amsgrad=True, decay=decay if "reducing_" in variable.name else 0.0)[-1]
        np.testing.assert_allclose(variable.numpy(), expected, rtol=F32.rtol, atol=F32.atol)
    assert int(tp.optimizer.iterations.numpy()) == 1


def test_the_weight_decay_exclusions_survive_the_replacement_of_the_optimizer():
    tp = small_tp()
    _restore_legacy(tp)
    variables = tp.model.variables_to_train
    before = [v.numpy().copy() for v in variables]

    # a zero gradient leaves only the decoupled decay
    tp.optimizer.apply_gradients([(tf.zeros_like(v), v) for v in variables])

    decay = 1.0 - LEGACY_OPTIONS["learning_rate"] * LEGACY_OPTIONS["weight_decay"]
    for variable, w0 in zip(variables, before):
        expected = w0 * decay if "reducing_" in variable.name else w0
        np.testing.assert_allclose(variable.numpy(), expected, rtol=F32.rtol, atol=F32.atol)


def test_strict_consumption_is_not_demanded_of_a_checkpoint_whose_optimizer_is_discarded():
    _restore_legacy(small_tp(), assert_consumed=True)


def test_a_model_only_load_of_a_legacy_checkpoint_is_silent():
    tp = small_tp()

    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyOptimizerStateWarning)
        tp.load_checkpoint(checkpoint_name=LEGACY_CHECKPOINT, model_only=True)

    np.testing.assert_array_equal(_weights(tp), _flat(legacy_weights()))


def test_a_model_without_optimizer_loads_a_legacy_checkpoint_silently():
    # the way foundation-model and fine-tuning tools read a checkpoint: no optimizer, partial read
    tp = small_tp()
    no_optimizer = TensorPotential(tp.model.instructions, param_dtype=tf.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyOptimizerStateWarning)
        no_optimizer.load_checkpoint(checkpoint_name=LEGACY_CHECKPOINT, expect_partial=True)

    np.testing.assert_array_equal(_weights(no_optimizer), _flat(legacy_weights()))
    assert no_optimizer.step == LEGACY_STEPS


def test_a_checkpoint_with_other_optimizer_options_is_treated_the_same_way(tmp_path):
    saved = _adam_tp(amsgrad=True)
    _two_steps(saved)
    saved.step = 2
    saved.save_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))
    restored = _adam_tp(amsgrad=False)  # no velocity-hat slots: another layout

    with pytest.warns(LegacyOptimizerStateWarning, match="options differ"):
        restored.load_checkpoint(checkpoint_name=_checkpoint_path(tmp_path))

    np.testing.assert_array_equal(_weights(restored), _weights(saved))
    assert restored.step == 2
    assert int(restored.optimizer.iterations.numpy()) == 0
