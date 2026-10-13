"""Tests of ``tensorpotential/loss.py``: the loss components and ``LossFunction``.

Logic layer: constructor options and their errors, the weight and parameter set/get, ``__repr__``,
the abstract bases, the input specification of every component (a component must run with only the
tensors it names), and ``LossFunction`` bookkeeping.

Physical-value layer: every component against a numpy oracle written from the formula of the
loss (``_Oracle`` below) on a three-structure batch of known atom counts, with and without the
sample normalisation, and the directional derivative of every component against a central finite
difference of its own value, with respect to every floating-point tensor it reads. The oracles do
not call the unit under test; the stress invariants are taken from the 3x3 tensor (trace,
half of trace squared minus trace of the square, determinant) and not from the Voigt formulas of
the unit. ``tests/test_loss_piecewise_linear.py`` covers the piecewise-linear components.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf  # after tensorpotential, which applies the TensorFlow options first

from tensorpotential import constants
from tensorpotential.loss import (
    HuberLoss,
    L2Loss,
    LossComponent,
    LossFunction,
    RegularizationLossComponent,
    WeightedHuberEnergyPerAtomLoss,
    WeightedHuberForceLoss,
    WeightedHuberStressLoss,
    WeightedHuberVirialLoss,
    WeightedMAEEPALoss,
    WeightedMAEStressLoss,
    WeightedOffsetEnergyHuberLoss,
    WeightedOffsetEnergyLoss,
    WeightedSSEEnergyLoss,
    WeightedSSEEnergyPerAtomLoss,
    WeightedSSEForceLoss,
    WeightedSSEStressLoss,
    WeightedSSEVirialLoss,
    WHuberForceCompNormLoss,
    WSMAEForceLoss,
    compute_energy_per_atom_error,
    compute_nat_per_structure,
    energy_offset_error,
    huber,
)
from tests.shared_models import l2_regularised_variables
from tests.tolerances import FINITE_DIFFERENCE_F64 as FD
from tests.tolerances import FLOAT32_ELEMENTWISE
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

DELTA = (
    0.5  # the Huber transition, inside the spread of the residuals of the batch below
)
NORM_THRESHOLD = (
    1.5  # A force norm that splits the reference forces of the batch in two
)
PERFECT_PREDICTION_LOSS = 1e-8  # the largest loss of a perfect prediction: the norm guard EPSILON gives its square root
EPSILON = 1e-10  # LossComponent.build: the guard of a norm or an absolute value at zero
FD_STEP = 1e-6  # the step assumed by FINITE_DIFFERENCE_F64 in tests/tolerances.py, for a loss of order 1
ATOM_COUNTS = (2, 1, 3)  # atoms of the three structures of the batch


def _value(variable) -> float:
    """The Python float held by a variable or tensor."""
    return float(np.asarray(variable))


def _const(value, dtype=None) -> tf.Tensor:
    """A constant tensor (``tf.constant`` is typed as possibly an operation)."""
    return cast(tf.Tensor, tf.constant(value, dtype=dtype))


def _assert_close(found, expected, tolerance=ARITH) -> None:
    np.testing.assert_allclose(
        found, expected, rtol=tolerance.rtol, atol=tolerance.atol
    )


@dataclass(frozen=True)
class _Batch:
    """The numpy arrays of a three-structure batch (energies, forces, virials, weights)."""

    e_true: np.ndarray
    e_pred: np.ndarray
    e_weight: np.ndarray
    n_atoms: np.ndarray
    f_true: np.ndarray
    f_pred: np.ndarray
    f_weight: np.ndarray
    v_true: np.ndarray
    v_pred: np.ndarray
    v_weight: np.ndarray
    volume: np.ndarray


def _make_batch(seed: int = 0) -> _Batch:
    rng = np.random.default_rng(seed)
    n_atoms = np.array(ATOM_COUNTS, dtype=float).reshape(-1, 1)
    total = int(n_atoms.sum())
    f_true = rng.normal(scale=1.2, size=(total, 3))
    return _Batch(
        e_true=rng.normal(scale=3.0, size=(3, 1)),
        e_pred=rng.normal(scale=3.0, size=(3, 1)),
        e_weight=rng.uniform(0.5, 2.0, size=(3, 1)),
        n_atoms=n_atoms,
        f_true=f_true,
        f_pred=f_true + rng.normal(scale=0.7, size=(total, 3)),
        f_weight=rng.uniform(0.5, 2.0, size=(total, 1)),
        v_true=rng.normal(scale=2.0, size=(3, 6)),
        v_pred=rng.normal(scale=2.0, size=(3, 6)),
        v_weight=rng.uniform(0.5, 2.0, size=(3, 1)),
        volume=rng.uniform(6.0, 14.0, size=(3, 1)),
    )


def _tensors(batch: _Batch) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """``(input_data, predictions)`` of the batch as float64 tensors, with the structure bookkeeping."""
    f64 = tf.float64
    atom_map = np.repeat(np.arange(3), ATOM_COUNTS).astype(np.int32)
    input_data = {
        constants.DATA_REFERENCE_ENERGY: _const(batch.e_true, f64),
        constants.DATA_ENERGY_WEIGHTS: _const(batch.e_weight, f64),
        constants.DATA_REFERENCE_FORCES: _const(batch.f_true, f64),
        constants.DATA_FORCE_WEIGHTS: _const(batch.f_weight, f64),
        constants.DATA_REFERENCE_VIRIAL: _const(batch.v_true, f64),
        constants.DATA_VIRIAL_WEIGHTS: _const(batch.v_weight, f64),
        constants.DATA_VOLUME: _const(batch.volume, f64),
        constants.ATOMS_TO_STRUCTURE_MAP: _const(atom_map, tf.int32),
        constants.N_STRUCTURES_BATCH_TOTAL: _const(3, tf.int32),
    }
    predictions = {
        constants.PREDICT_TOTAL_ENERGY: _const(batch.e_pred, f64),
        constants.PREDICT_FORCES: _const(batch.f_pred, f64),
        constants.PREDICT_VIRIAL: _const(batch.v_pred, f64),
    }
    return input_data, predictions


# --- numpy oracles: (numerator, denominator) of every loss, from the formulas of the losses ---


def _huber(x: np.ndarray, delta: float) -> np.ndarray:
    quadratic = 0.5 * x**2
    linear = delta * np.abs(x) - 0.5 * delta**2
    return np.where(np.abs(x) <= delta, quadratic, linear)


def _safe_norm(vectors: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(vectors**2, axis=-1, keepdims=True) + EPSILON)


def _tanh_loss(x: np.ndarray) -> np.ndarray:
    return x * np.tanh(x / 2.0)


def _offset_pairs(b: _Batch, kernel) -> tuple[float, float]:
    """Pairwise energy-offset loss over all ordered pairs of structures, the diagonal included."""
    e_true = (b.e_true / b.n_atoms).ravel()
    e_pred = (b.e_pred / b.n_atoms).ravel()
    weight = b.e_weight.ravel()
    num = den = 0.0
    for i in range(len(weight)):
        for j in range(len(weight)):
            w_ij = weight[i] * weight[j]
            w_ij = np.sqrt(w_ij + 1e-16) if w_ij != 0 else 0.0
            offset_error = abs(e_true[i] - e_true[j]) - abs(e_pred[i] - e_pred[j])
            num += w_ij * kernel(offset_error)
            den += w_ij
    return num, den


def _force_norm_term(b: _Batch, kernel) -> float:
    v_true, v_pred = _safe_norm(b.f_true), _safe_norm(b.f_pred)
    return float(np.mean(3.0 * b.f_weight * kernel(v_pred - v_true)))


def _stress_invariants(voigt: np.ndarray) -> np.ndarray:
    """First, second and third invariant of the symmetric tensors given as xx, yy, zz, xy, xz, yz."""
    out = np.empty((len(voigt), 3))
    for k, (xx, yy, zz, xy, xz, yz) in enumerate(voigt):
        tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        out[k] = (
            np.trace(tensor),
            0.5 * (np.trace(tensor) ** 2 - np.trace(tensor @ tensor)),
            np.linalg.det(tensor),
        )
    return out


def _sse_energy(b):
    return np.sum(b.e_weight * (b.e_pred - b.e_true) ** 2), np.sum(b.e_weight)


def _sse_epa(b):
    return np.sum(b.e_weight * ((b.e_pred - b.e_true) / b.n_atoms) ** 2), np.sum(
        b.e_weight
    )


def _mae_epa(b):
    error = (b.e_pred - b.e_true) / b.n_atoms
    return np.sum(b.e_weight * np.abs(error + EPSILON)), np.sum(b.e_weight)


def _huber_epa(b):
    error = (b.e_pred - b.e_true) / b.n_atoms
    return np.sum(b.e_weight * _huber(error, DELTA)), np.sum(b.e_weight)


def _offset(b):
    return _offset_pairs(b, lambda x: 0.5 * x**2)


def _offset_huber(b):
    return _offset_pairs(b, lambda x: 0.5 * _huber(np.array(x), DELTA))


def _sse_force(b, compute_norm=False):
    num = np.sum(b.f_weight * (b.f_pred - b.f_true) ** 2)
    if compute_norm:
        num += _force_norm_term(b, lambda x: x**2)
    return num, np.sum(b.f_weight) * 3


def _mae_force(b, compute_norm=False):
    num = np.sum(b.f_weight * _tanh_loss(b.f_pred - b.f_true))
    if compute_norm:
        num += _force_norm_term(b, _tanh_loss)
    return num, np.sum(b.f_weight) * 3


def _huber_force_comp_norm(b, compute_norm=False):
    num = np.sum(b.f_weight * _huber(b.f_pred - b.f_true, DELTA))
    if compute_norm:
        num += _force_norm_term(b, lambda x: _huber(x, DELTA))
    return num, np.sum(b.f_weight) * 3


def _huber_force(b, compute_norm=False, scale_w_by_norm=False, use_l2_norm=False):
    weight = b.f_weight
    if scale_w_by_norm:
        norm = _safe_norm(b.f_true)
        weight = np.where(norm < NORM_THRESHOLD, weight, weight / norm)
    error = b.f_pred - b.f_true
    if use_l2_norm:
        error = _safe_norm(error)
    num = np.sum(weight * _huber(error, DELTA))
    if compute_norm:
        v_err = _safe_norm(b.f_pred) - _safe_norm(b.f_true)
        num += float(np.mean(3.0 * weight * _huber(v_err, DELTA)))
    den = np.sum(weight) * (1 if use_l2_norm else 3)
    return num, den


def _sse_virial(b):
    return np.sum(b.v_weight * (b.v_pred - b.v_true) ** 2), np.sum(b.v_weight) * 6


def _sse_stress(b):
    delta = (b.v_pred - b.v_true) / b.volume
    return np.sum(b.v_weight * delta**2), np.sum(b.v_weight) * 6


def _mae_stress(b):
    delta = (b.v_pred - b.v_true) / b.volume
    return np.sum(b.v_weight * np.abs(delta + EPSILON)), np.sum(b.v_weight) * 6


def _huber_virial(b):
    return np.sum(b.v_weight * _huber(b.v_pred - b.v_true, DELTA)), np.sum(
        b.v_weight
    ) * 6


def _huber_stress(b, compute_invariants=False):
    s_true, s_pred = b.v_true / b.volume, b.v_pred / b.volume
    num = np.sum(b.v_weight * _huber(s_pred - s_true, DELTA))
    if compute_invariants:
        inv_true, inv_pred = _stress_invariants(s_true), _stress_invariants(s_pred)
        for column, divisor in enumerate((9.0, 18.0, 18.0)):
            error = (inv_true[:, column] - inv_pred[:, column]).reshape(-1, 1)
            num += np.sum(b.v_weight * _huber(error, DELTA)) / divisor
    return num, np.sum(b.v_weight) * 6


@dataclass(frozen=True)
class _Case:
    """A loss component, the keyword arguments that build it, and the oracle of its value."""

    cls: type[LossComponent]
    kwargs: dict
    oracle: Callable[[_Batch], tuple[float, float]]


def _case(cls, oracle, **kwargs) -> _Case:
    return _Case(cls, kwargs, oracle)


def _with(oracle, **options):
    return lambda b: oracle(b, **options)


CASES = {
    "sse_energy": _case(WeightedSSEEnergyLoss, _sse_energy),
    "sse_energy_per_atom": _case(WeightedSSEEnergyPerAtomLoss, _sse_epa),
    "mae_energy_per_atom": _case(WeightedMAEEPALoss, _mae_epa),
    "huber_energy_per_atom": _case(
        WeightedHuberEnergyPerAtomLoss, _huber_epa, delta=DELTA
    ),
    "offset_energy": _case(WeightedOffsetEnergyLoss, _offset),
    "offset_energy_huber": _case(
        WeightedOffsetEnergyHuberLoss, _offset_huber, delta=DELTA
    ),
    "sse_force": _case(WeightedSSEForceLoss, _sse_force),
    "sse_force_norm": _case(
        WeightedSSEForceLoss, _with(_sse_force, compute_norm=True), compute_norm=True
    ),
    "mae_force": _case(WSMAEForceLoss, _mae_force),
    "mae_force_norm": _case(
        WSMAEForceLoss, _with(_mae_force, compute_norm=True), compute_norm=True
    ),
    "huber_force_comp_norm": _case(
        WHuberForceCompNormLoss, _huber_force_comp_norm, delta=DELTA
    ),
    "huber_force_comp_norm_norm": _case(
        WHuberForceCompNormLoss,
        _with(_huber_force_comp_norm, compute_norm=True),
        delta=DELTA,
        compute_norm=True,
    ),
    "huber_force": _case(WeightedHuberForceLoss, _huber_force, delta=DELTA),
    "huber_force_norm": _case(
        WeightedHuberForceLoss,
        _with(_huber_force, compute_norm=True),
        delta=DELTA,
        compute_norm=True,
    ),
    "huber_force_scaled": _case(
        WeightedHuberForceLoss,
        _with(_huber_force, scale_w_by_norm=True),
        delta=DELTA,
        scale_w_by_norm=True,
        norm_threshold=NORM_THRESHOLD,
    ),
    "huber_force_scaled_norm": _case(
        WeightedHuberForceLoss,
        _with(_huber_force, scale_w_by_norm=True, compute_norm=True),
        delta=DELTA,
        scale_w_by_norm=True,
        compute_norm=True,
        norm_threshold=NORM_THRESHOLD,
    ),
    "huber_force_l2": _case(
        WeightedHuberForceLoss,
        _with(_huber_force, use_l2_norm=True),
        delta=DELTA,
        use_l2_norm=True,
    ),
    "huber_force_l2_scaled": _case(
        WeightedHuberForceLoss,
        _with(_huber_force, use_l2_norm=True, scale_w_by_norm=True),
        delta=DELTA,
        use_l2_norm=True,
        scale_w_by_norm=True,
        norm_threshold=NORM_THRESHOLD,
    ),
    "sse_virial": _case(WeightedSSEVirialLoss, _sse_virial),
    "sse_stress": _case(WeightedSSEStressLoss, _sse_stress),
    "mae_stress": _case(WeightedMAEStressLoss, _mae_stress),
    "huber_virial": _case(WeightedHuberVirialLoss, _huber_virial, delta=DELTA),
    "huber_stress": _case(WeightedHuberStressLoss, _huber_stress, delta=DELTA),
    "huber_stress_invariants": _case(
        WeightedHuberStressLoss,
        _with(_huber_stress, compute_invariants=True),
        delta=DELTA,
        compute_invariants=True,
    ),
}


def _build(case: _Case, weight: float = 1.0, **overrides) -> LossComponent:
    component = case.cls(loss_component_weight=weight, **{**case.kwargs, **overrides})
    component.build(tf.float64)
    return component


def _spec_only(component: LossComponent, input_data: dict) -> dict:
    return {key: input_data[key] for key in component.input_tensor_spec}


@pytest.fixture(scope="module")
def batch() -> _Batch:
    return _make_batch()


@pytest.fixture(scope="module")
def tensors(batch):
    return _tensors(batch)


# --- values against the oracles ---


def test_the_batch_has_residuals_on_both_sides_of_every_threshold(batch):
    """The oracles only discriminate the kernels when the batch reaches every branch of them."""
    for error in (
        (batch.e_pred - batch.e_true) / batch.n_atoms,
        batch.f_pred - batch.f_true,
        batch.v_pred - batch.v_true,
        (batch.v_pred - batch.v_true) / batch.volume,
    ):
        assert np.any(np.abs(error) <= DELTA)
    for error in (
        (batch.e_pred - batch.e_true) / batch.n_atoms,
        batch.f_pred - batch.f_true,
        batch.v_pred - batch.v_true,
    ):
        assert np.any(np.abs(error) > DELTA)
    norms = np.linalg.norm(batch.f_true, axis=1)
    assert np.any(norms < NORM_THRESHOLD)
    assert np.any(norms >= NORM_THRESHOLD)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_a_component_equals_its_oracle_from_the_tensors_of_its_spec_only(
    case, batch, tensors
):
    input_data, predictions = tensors
    component = _build(case)
    num, den = case.oracle(batch)
    value = component.compute_loss_component(
        input_data=_spec_only(component, input_data), predictions=predictions
    )
    _assert_close(value.numpy(), num / den)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_without_sample_normalisation_a_component_is_the_bare_sum(case, batch, tensors):
    input_data, predictions = tensors
    component = _build(case, normalize_by_samples=False)
    num, _ = case.oracle(batch)
    value = component.compute_loss_component(
        input_data=input_data, predictions=predictions
    )
    _assert_close(value.numpy(), num)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_calling_a_component_multiplies_its_value_by_the_component_weight(
    case, batch, tensors
):
    input_data, predictions = tensors
    component = _build(case, weight=2.5)
    num, den = case.oracle(batch)
    _assert_close(component(input_data, predictions).numpy(), 2.5 * num / den)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_a_component_leaves_its_input_dictionaries_alone(case, tensors):
    input_data, predictions = tensors
    component = _build(case)
    before = {k: v.numpy().copy() for k, v in {**input_data, **predictions}.items()}
    component.compute_loss_component(input_data=input_data, predictions=predictions)
    assert set(input_data) == {
        constants.DATA_REFERENCE_ENERGY,
        constants.DATA_ENERGY_WEIGHTS,
        constants.DATA_REFERENCE_FORCES,
        constants.DATA_FORCE_WEIGHTS,
        constants.DATA_REFERENCE_VIRIAL,
        constants.DATA_VIRIAL_WEIGHTS,
        constants.DATA_VOLUME,
        constants.ATOMS_TO_STRUCTURE_MAP,
        constants.N_STRUCTURES_BATCH_TOTAL,
    }
    for key, value in {**input_data, **predictions}.items():
        np.testing.assert_array_equal(value.numpy(), before[key])


def test_the_sse_energy_loss_of_two_structures_by_hand():
    # weights 1 and 3, residuals 1 and 2: (1 * 1 + 3 * 4) / (1 + 3) = 13 / 4
    input_data = {
        constants.DATA_REFERENCE_ENERGY: _const([[0.0], [0.0]], tf.float64),
        constants.DATA_ENERGY_WEIGHTS: _const([[1.0], [3.0]], tf.float64),
    }
    predictions = {constants.PREDICT_TOTAL_ENERGY: _const([[1.0], [-2.0]], tf.float64)}
    component = WeightedSSEEnergyLoss(loss_component_weight=1.0)
    component.build(tf.float64)
    assert (
        component.compute_loss_component(
            input_data=input_data, predictions=predictions
        ).numpy()
        == 3.25
    )


def test_the_energy_per_atom_losses_divide_by_the_atom_count_of_each_structure_by_hand():
    # structures of 2 and 4 atoms, residuals 1 and 2 eV: per-atom residuals 0.5 and 0.5, equal weights
    input_data = {
        constants.DATA_REFERENCE_ENERGY: _const([[0.0], [0.0]], tf.float64),
        constants.DATA_ENERGY_WEIGHTS: _const([[1.0], [1.0]], tf.float64),
        constants.ATOMS_TO_STRUCTURE_MAP: _const([0, 0, 1, 1, 1, 1], tf.int32),
        constants.N_STRUCTURES_BATCH_TOTAL: _const(2, tf.int32),
    }
    predictions = {constants.PREDICT_TOTAL_ENERGY: _const([[1.0], [2.0]], tf.float64)}
    for cls, expected in (
        (WeightedSSEEnergyPerAtomLoss, 0.25),
        (
            WeightedMAEEPALoss,
            0.5 + EPSILON,
        ),  # |residual + epsilon| keeps the loss finite at zero
        (
            WeightedHuberEnergyPerAtomLoss,
            0.125,
        ),  # delta 1: quadratic region, 0.5 * 0.5 ** 2
    ):
        component = cls(loss_component_weight=1.0)
        component.build(tf.float64)
        _assert_close(
            component.compute_loss_component(
                input_data=input_data, predictions=predictions
            ).numpy(),
            expected,
        )


def test_a_structure_without_atoms_counts_as_one_atom_instead_of_dividing_by_zero():
    input_data = {
        constants.DATA_REFERENCE_ENERGY: _const([[0.0], [0.0]], tf.float64),
        constants.DATA_ENERGY_WEIGHTS: _const([[1.0], [1.0]], tf.float64),
        constants.ATOMS_TO_STRUCTURE_MAP: _const([0, 0], tf.int32),
        constants.N_STRUCTURES_BATCH_TOTAL: _const(2, tf.int32),
    }
    predictions = {constants.PREDICT_TOTAL_ENERGY: _const([[2.0], [3.0]], tf.float64)}
    np.testing.assert_array_equal(
        compute_nat_per_structure(input_data).numpy(), [[2], [1]]
    )
    # structure 0: 2 / 2 = 1; structure 1: no atom, 3 / 1 = 3
    np.testing.assert_array_equal(
        compute_energy_per_atom_error(input_data, predictions).numpy(), [[1.0], [3.0]]
    )
    for cls in (
        WeightedSSEEnergyPerAtomLoss,
        WeightedHuberEnergyPerAtomLoss,
        WeightedOffsetEnergyLoss,
    ):
        component = cls(loss_component_weight=1.0)
        component.build(tf.float64)
        assert np.isfinite(
            component.compute_loss_component(
                input_data=input_data, predictions=predictions
            ).numpy()
        )


@pytest.mark.parametrize(
    "name",
    ["sse_energy", "sse_energy_per_atom", "mae_energy_per_atom", "offset_energy"],
)
def test_a_zero_weight_structure_does_not_contribute(name, batch, tensors):
    """A structure of weight zero gives the value of the batch without it, atom counts of the others kept."""
    input_data, predictions = tensors
    zero = batch.e_weight.copy()
    zero[1] = 0.0
    dropped = {**input_data, constants.DATA_ENERGY_WEIGHTS: _const(zero, tf.float64)}
    keep = [0, 2]
    without = replace(
        batch,
        e_true=batch.e_true[keep],
        e_pred=batch.e_pred[keep],
        e_weight=batch.e_weight[keep],
        n_atoms=batch.n_atoms[keep],
    )
    case = CASES[name]
    found = _build(case).compute_loss_component(
        input_data=dropped, predictions=predictions
    )
    num, den = case.oracle(without)
    _assert_close(found.numpy(), num / den)


def test_a_perfect_prediction_has_zero_loss_in_every_smooth_component():
    reference = _make_batch()
    perfect = replace(
        reference,
        e_pred=reference.e_true,
        f_pred=reference.f_true,
        v_pred=reference.v_true,
    )
    input_data, predictions = _tensors(perfect)
    for name, case in CASES.items():
        component = _build(case)
        value = component.compute_loss_component(
            input_data=input_data, predictions=predictions
        ).numpy()
        # a norm term is guarded by the epsilon, so it is the square root of 1e-10 apart at most
        assert abs(value) < PERFECT_PREDICTION_LOSS, name


# --- the energy offset helper and the Huber kernel ---


def test_energy_offset_error_builds_all_pairwise_absolute_differences_by_hand():
    e_true = _const([[1.0], [4.0], [9.0]], tf.float64)
    e_pred = _const([[0.0], [2.0], [3.0]], tf.float64)
    weight = _const([[1.0], [0.0], [4.0]], tf.float64)
    true_off, pred_off, w_off = energy_offset_error(e_true, e_pred, weight)
    # rows are ordered by the first structure of the pair, columns by the second
    np.testing.assert_array_equal(
        true_off.numpy().reshape(3, 3), [[0, 3, 8], [3, 0, 5], [8, 5, 0]]
    )
    np.testing.assert_array_equal(
        pred_off.numpy().reshape(3, 3), [[0, 2, 3], [2, 0, 1], [3, 1, 0]]
    )
    assert w_off is not None
    # sqrt(w_i w_j + 1e-16), and exactly zero when the product is zero
    expected = np.array([[1, 0, 2], [0, 0, 0], [2, 0, 4]], dtype=float)
    _assert_close(w_off.numpy().reshape(3, 3), expected, ARITH)
    assert np.all(w_off.numpy().reshape(3, 3)[expected == 0] == 0)


def test_energy_offset_error_without_weights_returns_no_weights():
    e = _const([[1.0], [2.0]], tf.float64)
    _, _, w_off = energy_offset_error(e, e)
    assert w_off is None


def test_huber_is_quadratic_inside_delta_and_linear_outside_by_hand():
    values = huber(_const([0.0, 0.5, -0.5, 1.0, 3.0, -3.0], tf.float64), delta=1.0)
    # 0.5 x^2 for |x| <= 1 (the boundary included), delta |x| - delta^2 / 2 beyond
    np.testing.assert_allclose(
        values.numpy(), [0.0, 0.125, 0.125, 0.5, 2.5, 2.5], rtol=ARITH.rtol
    )


def test_huber_follows_the_dtype_of_its_error():
    assert huber(_const([2.0], tf.float32)).dtype == tf.float32


# --- derivatives ---


def _shifted(tensors, directions, scale):
    """The tensors with ``scale`` times the direction added to those that have one."""
    return {
        k: v + scale * directions[k] if k in directions else v
        for k, v in tensors.items()
    }


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_the_gradient_of_a_component_matches_a_finite_difference_of_its_value(
    case, tensors
):
    input_data, predictions = tensors
    component = _build(case)
    rng = np.random.default_rng(1)
    floats = {
        key: value
        for key, value in {**input_data, **predictions}.items()
        if value.dtype == tf.float64
    }
    directions = {
        key: _const(rng.normal(size=value.shape), tf.float64)
        for key, value in floats.items()
    }

    # the tape watches every float tensor; one the loss does not read has a zero derivative, and the
    # central difference below would disagree with a loss that ignored an input it should read
    with tf.GradientTape() as tape:
        for tensor in floats.values():
            tape.watch(tensor)
        value = component.compute_loss_component(
            input_data=input_data, predictions=predictions
        )
    gradients = tape.gradient(
        value, list(floats.values()), unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    analytic = sum(
        float(tf.reduce_sum(g * directions[k]))
        for k, g in zip(floats, gradients, strict=True)
    )

    def loss_at(scale):
        return component.compute_loss_component(
            input_data=_shifted(input_data, directions, scale),
            predictions=_shifted(predictions, directions, scale),
        )

    numeric = float(loss_at(FD_STEP) - loss_at(-FD_STEP)) / (2 * FD_STEP)
    _assert_close(analytic, numeric, FD)


# --- constructors, options, errors ---


def test_the_huber_force_loss_refuses_the_vector_norm_together_with_the_norm_term():
    with pytest.raises(
        ValueError, match="use_l2_norm=True already couples force components"
    ):
        WeightedHuberForceLoss(
            loss_component_weight=1.0, use_l2_norm=True, compute_norm=True
        )


def test_the_norm_threshold_of_the_huber_force_loss_becomes_a_variable_only_when_it_is_used():
    plain = WeightedHuberForceLoss(loss_component_weight=1.0, norm_threshold=3.0)
    plain.build(tf.float64)
    assert plain.norm_threshold == 3.0
    scaled = WeightedHuberForceLoss(
        loss_component_weight=1.0, scale_w_by_norm=True, norm_threshold=3.0
    )
    scaled.build(tf.float64)
    assert isinstance(scaled.norm_threshold, tf.Variable)
    assert scaled.norm_threshold.dtype == tf.float64
    assert _value(scaled.norm_threshold) == 3.0
    assert not scaled.norm_threshold.trainable


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_building_a_component_makes_a_fixed_weight_variable_and_the_epsilon(case):
    component = case.cls(loss_component_weight=0.75, **case.kwargs)
    assert not hasattr(component, "loss_component_weight")
    component.build(tf.float32)
    assert isinstance(component.loss_component_weight, tf.Variable)
    assert component.loss_component_weight.dtype == tf.float32
    assert not component.loss_component_weight.trainable
    assert _value(component.loss_component_weight) == 0.75
    assert component.epsilon.dtype == tf.float32
    np.testing.assert_allclose(
        float(component.epsilon), EPSILON, rtol=FLOAT32_ELEMENTWISE.rtol
    )


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_the_constructor_keeps_the_normalisation_flag_and_the_name(case):
    assert (
        case.cls(loss_component_weight=1.0, **case.kwargs).normalize_by_samples is True
    )
    off = case.cls(loss_component_weight=1.0, normalize_by_samples=False, **case.kwargs)
    assert off.normalize_by_samples is False
    named = case.cls(loss_component_weight=1.0, name="mine", **case.kwargs)
    assert named.name == "mine"


def test_the_corresponding_metrics_of_a_component_are_none_unless_a_force_class_sets_them():
    from tensorpotential.metrics import ForceMetrics

    assert (
        WeightedOffsetEnergyLoss(loss_component_weight=1.0).corresponding_metrics
        is None
    )
    assert (
        WeightedSSEEnergyLoss(loss_component_weight=1.0).corresponding_metrics is None
    )
    assert (
        WSMAEForceLoss(loss_component_weight=1.0).corresponding_metrics is ForceMetrics
    )
    assert (
        WHuberForceCompNormLoss(loss_component_weight=1.0).corresponding_metrics
        is ForceMetrics
    )


# --- weight and parameter access, repr ---


def test_the_component_weight_is_set_and_read_back_through_the_variable():
    component = WeightedSSEEnergyLoss(loss_component_weight=1.0)
    component.build(tf.float64)
    component.set_loss_component_weight(4.0)
    assert component.get_loss_component_weight() == 4.0
    assert _value(component.loss_component_weight) == 4.0


def test_set_loss_component_params_reads_the_keys_it_knows_and_ignores_the_rest():
    component = WeightedSSEEnergyLoss(loss_component_weight=1.0)
    component.build(tf.float64)
    component.set_loss_component_params({"delta": 9.0, "other": 1})
    assert component.get_loss_component_weight() == 1.0
    component.set_loss_component_params({"weight": 7.0})
    assert component.get_loss_component_weight() == 7.0
    component.set_loss_component_params({})
    assert component.get_loss_component_weight() == 7.0


def test_a_huber_component_also_takes_the_delta_parameter():
    component = WeightedHuberVirialLoss(loss_component_weight=2.0, delta=0.5)
    component.build(tf.float64)
    component.set_loss_component_params({"delta": 3.0})
    assert _value(component.delta) == 3.0
    assert component.get_loss_component_weight() == 2.0
    component.set_loss_component_params({"weight": 5.0, "delta": 0.25})
    assert (component.get_loss_component_weight(), _value(component.delta)) == (
        5.0,
        0.25,
    )
    component.set_loss_component_params({"weight": 6.0})
    assert _value(component.delta) == 0.25


def test_a_new_delta_changes_the_value_of_a_built_huber_component(batch, tensors):
    input_data, predictions = tensors
    component = _build(CASES["huber_virial"])
    component.set_loss_component_params({
        "delta": 100.0
    })  # every residual is now quadratic
    expected = np.sum(batch.v_weight * 0.5 * (batch.v_pred - batch.v_true) ** 2) / (
        np.sum(batch.v_weight) * 6
    )
    _assert_close(
        component.compute_loss_component(
            input_data=input_data, predictions=predictions
        ).numpy(),
        expected,
    )


def test_repr_shows_the_weight_before_and_after_the_build():
    component = WeightedSSEForceLoss(loss_component_weight=2.0)
    assert repr(component) == "2.0*WeightedSSEForceLoss"
    component.build(tf.float64)
    component.set_loss_component_weight(3.5)
    assert repr(component) == "3.5*WeightedSSEForceLoss"


def test_huber_repr_adds_the_delta_as_a_float_before_the_build_and_as_a_value_after():
    component = WeightedHuberForceLoss(loss_component_weight=1.0, delta=0.25)
    assert repr(component) == "1.0*WeightedHuberForceLoss(delta=0.25)"
    component.build(tf.float64)
    assert repr(component) == "1.0*WeightedHuberForceLoss(delta=0.25)"


# --- the abstract bases ---


def test_the_abstract_bases_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        LossComponent(loss_component_weight=1.0, name="x")  # ty: ignore[call-non-callable]
    with pytest.raises(TypeError, match="abstract"):
        HuberLoss(loss_component_weight=1.0)  # ty: ignore[call-non-callable]


def test_the_abstract_compute_methods_raise_not_implemented_error():
    class _Plain(LossComponent):
        def compute_loss_component(self, input_data, predictions, **kwargs):
            return super().compute_loss_component(input_data, predictions, **kwargs)

    class _Huber(HuberLoss):
        def compute_loss_component(self, input_data, predictions, **kwargs):
            return super().compute_loss_component(input_data, predictions, **kwargs)

    for component in (_Plain(1.0, name="plain"), _Huber(1.0)):
        with pytest.raises(NotImplementedError):
            component.compute_loss_component(input_data={}, predictions={})
        with pytest.raises(NotImplementedError):
            component(input_data={}, predictions={})


def test_every_component_names_the_tensors_it_reads_in_its_input_spec():
    for case in CASES.values():
        spec = case.cls.input_tensor_spec
        assert spec, case.cls.__name__
        for entry in spec.values():
            assert entry["dtype"] in ("float", "int")
    assert LossComponent.input_tensor_spec == {}


# --- LossFunction ---


def _loss_function() -> LossFunction:
    loss = LossFunction(
        {
            "energy": WeightedSSEEnergyLoss(loss_component_weight=1.0),
            "forces": WeightedSSEForceLoss(loss_component_weight=10.0),
        },
        name="total",
    )
    loss.build(tf.float64)
    return loss


def test_the_loss_function_sums_its_components_and_reports_each_one(batch, tensors):
    input_data, predictions = tensors
    result = _loss_function()(input_data=input_data, predictions=predictions)
    num_e, den_e = _sse_energy(batch)
    num_f, den_f = _sse_force(batch)
    assert list(result) == ["total", "energy", "forces"]
    _assert_close(result["energy"].numpy(), num_e / den_e)
    _assert_close(result["forces"].numpy(), 10.0 * num_f / den_f)
    _assert_close(result["total"].numpy(), num_e / den_e + 10.0 * num_f / den_f)


def test_the_loss_function_of_no_component_is_the_float_zero():
    loss = LossFunction({})
    assert loss(input_data={}, predictions={}) == {"total_loss": 0.0}


def test_the_loss_function_collects_the_input_signatures_of_its_components():
    signatures = _loss_function().get_input_signatures()
    assert set(signatures) == {
        constants.DATA_REFERENCE_ENERGY,
        constants.DATA_ENERGY_WEIGHTS,
        constants.DATA_REFERENCE_FORCES,
        constants.DATA_FORCE_WEIGHTS,
    }
    assert signatures[constants.DATA_REFERENCE_FORCES] == {
        "shape": [None, 3],
        "dtype": "float",
    }


def test_the_loss_function_sets_and_reads_the_component_weights():
    loss = _loss_function()
    assert loss.get_loss_component_weights() == {"energy": 1.0, "forces": 10.0}
    loss.set_loss_component_weights({"forces": 0.5})
    assert loss.get_loss_component_weights() == {"energy": 1.0, "forces": 0.5}


def test_the_loss_function_sets_component_parameters_by_name():
    loss = LossFunction({
        "virial": WeightedHuberVirialLoss(loss_component_weight=1.0, delta=1.0)
    })
    loss.build(tf.float64)
    loss.set_loss_component_params({"virial": {"weight": 2.0, "delta": 0.1}})
    component = cast(HuberLoss, loss.loss_components["virial"])
    assert (component.get_loss_component_weight(), _value(component.delta)) == (
        2.0,
        0.1,
    )


def test_an_unknown_component_name_is_refused_with_the_known_names():
    loss = _loss_function()
    with pytest.raises(
        AssertionError,
        match=r"stress not in loss_components \(dict_keys\(\['energy', 'forces'\]\)\)",
    ):
        loss.set_loss_component_weights({"stress": 1.0})
    with pytest.raises(AssertionError, match="stress not in loss_components"):
        loss.set_loss_component_params({"stress": {"weight": 1.0}})
    assert loss.get_loss_component_weights() == {"energy": 1.0, "forces": 10.0}


def test_a_component_of_an_unknown_type_is_refused_when_the_loss_is_computed():
    loss = LossFunction({"odd": object()})  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Unknown loss type"):
        loss(input_data={}, predictions={})


def test_the_loss_function_repr_lists_every_component_with_its_name():
    text = repr(_loss_function())
    assert "1.0*WeightedSSEEnergyLoss(energy)" in text
    assert "10.0*WeightedSSEForceLoss(forces)" in text
    assert " + " in text


def test_the_loss_function_input_signatures_start_empty():
    assert LossFunction({}).input_signatures == {}


# --- regularisation ---


def _regularised_model():
    """A real, cheap model whose instructions report an L2 loss, all trainable weights set to 0.5."""
    from tensorpotential.potentials import get_preset
    from tensorpotential.tensorpot import TensorPotential

    preset = get_preset("GRACE_2LAYER_v1_24")
    instructions = preset(element_map={"Be": 0, "Li": 1}, lmax=0).get_instructions()
    model = TensorPotential(instructions, param_dtype=tf.float64).model
    for variable in model.variables_to_train:
        variable.assign(tf.fill(variable.shape, _const(0.5, variable.dtype)))
    # each instruction that reports an L2 loss counts every trainable variable it owns, the variables of
    # the instructions it refers to included: with every weight 0.5 the sum is 0.25 per element
    elements = sum(
        int(np.prod(v.shape)) for v in l2_regularised_variables(model.instructions)
    )
    return model, 0.25 * elements


def test_the_l2_loss_scales_the_model_l2_sum_by_its_weight():
    model, expected = _regularised_model()
    component = L2Loss(loss_component_weight=0.5)
    component.build(tf.float64)
    _assert_close(component.compute_loss_component(model).numpy(), expected)
    _assert_close(component(model).numpy(), 0.5 * expected)
    assert repr(component) == f"{component.loss_component_weight}*L2Loss"
    assert isinstance(component, RegularizationLossComponent)
    assert component.name == "L2Loss"
    assert L2Loss(loss_component_weight=1.0, name="mine").name == "mine"


def test_the_loss_function_adds_a_regularisation_component_computed_from_the_model(
    batch, tensors
):
    input_data, predictions = tensors
    model, l2_sum = _regularised_model()
    loss = LossFunction({
        "energy": WeightedSSEEnergyLoss(loss_component_weight=1.0),
        "l2": L2Loss(loss_component_weight=0.5),
    })
    loss.build(tf.float64)
    result = loss(input_data=input_data, predictions=predictions, model=model)
    num, den = _sse_energy(batch)
    _assert_close(result["l2"].numpy(), 0.5 * l2_sum)
    _assert_close(result["total_loss"].numpy(), num / den + 0.5 * l2_sum)
