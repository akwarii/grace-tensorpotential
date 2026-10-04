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

from ase import Atoms
from ase.build import bulk

from tensorpotential.calculator import TPCalculator
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential

import tensorflow as tf  # after tensorpotential, which must set TF_USE_LEGACY_KERAS first

from tests.tolerances import FINITE_DIFFERENCE_F64 as FD
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

