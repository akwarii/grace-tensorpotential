"""Characterization tests for ``grace_2`` (preset ``TENSOR_2L``, ``extra/gen_tensor/model.py``).

Logic layer: the instruction list for every ``max_order`` and every set of tensor
components, the ``compute_energy`` branch, the sizes passed on to the instructions, and the
four early errors. Physics layer: the real model, built with random float64 weights, is
rotated, translated and permuted. The rank-2 output must transform as ``R T R^T`` whatever
the weights, the energy must be invariant, and the forces must equal minus the central
finite difference of the energy.
"""

from __future__ import annotations

import os
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf
from ase import Atoms
from scipy.spatial.transform import Rotation

from tensorpotential import TPModel, constants
from tensorpotential.data.databuilder import GeometricalDataBuilder
from tensorpotential.extra.gen_tensor import constants as tensor_constants
from tensorpotential.extra.gen_tensor.model import ComputeBatchEFTensor, grace_2
from tensorpotential.potentials import get_preset
from tests.tolerances import COVARIANCE_F64, FINITE_DIFFERENCE_F64

ELEMENTS = {"Si": 0, "O": 1}
RCUT = 3.5
SMALL = dict(
    element_map=ELEMENTS,
    rcut=RCUT,
    lmax=2,
    n_rad_base=4,
    n_rad_max=[6, 8],
    embedding_size=6,
    n_mlp_dens=4,
)

BASE = ["BondLength", "ScaledBondVector", "RadialBasis", "R", "Y", "Z", "A"]
# Instruction names a layer adds for each order (from reading grace_2 once, by hand).
LAYER_1 = {1: [], 2: ["A1", "AA"], 3: ["AA1", "AAA"], 4: ["AA2", "AAAA"]}
LAYER_2 = {1: [], 2: ["B1", "BB"], 3: ["BB1", "BBB"], 4: ["BB2", "BBBB"]}
REDUCE_1 = ["IM", "I", "R1", "B0", "YI", "B"]
OUTPUT = {
    0: ["predict_L0", "I0", "a0"],
    1: ["predict_L1", "I1", "a1"],
    2: ["predict_L2", "I2", "a2"],
}
ENERGY = [
    "rho_1",
    "I_e_LN",
    "rho_2",
    "I2_e_LN",
    "atomic_energy",
    "LinMLPOut2ScalarTarget",
    "ConstantScaleShiftTarget",
]


def expected_names(
    max_order: int, components: list[int], energy: bool = False
) -> list[str]:
    names = list(BASE)
    for order in range(2, max_order + 1):
        names += LAYER_1[order]
    names += REDUCE_1
    for order in range(2, max_order + 1):
        names += LAYER_2[order]
    if energy:
        names += ENERGY
    for component in components:
        names += OUTPUT[component]
    return names


# ----------------------------------------------------------------------------- logic


def test_preset_is_registered_under_tensor_2l():
    assert get_preset("TENSOR_2L") is grace_2


@pytest.mark.parametrize("max_order", [1, 2, 3, 4])
def test_instruction_list_for_each_order(max_order):
    ins = grace_2(**SMALL, max_order=max_order)
    assert list(ins) == expected_names(max_order, [0, 1, 2])


@pytest.mark.parametrize("components", [[0, 1, 2], [1, 2], [0, 2], [1], [2], [2, 1]])
def test_instruction_list_for_each_component_set(components):
    ins = grace_2(**SMALL, max_order=2, tensor_components=components)
    assert list(ins) == expected_names(2, components)


def test_default_components_are_all_three():
    assert list(grace_2(**SMALL, max_order=1)) == list(
        grace_2(**SMALL, max_order=1, tensor_components=[0, 1, 2])
    )


def test_compute_energy_adds_the_energy_head_before_the_outputs():
    ins = grace_2(**SMALL, max_order=2, tensor_components=[2], compute_energy=True)
    assert list(ins) == expected_names(2, [2], energy=True)
    assert ins["atomic_energy"].name == constants.PREDICT_ATOMIC_ENERGY


def test_energy_head_is_absent_by_default():
    assert "atomic_energy" not in grace_2(**SMALL, max_order=2)


def test_scale_shift_target_receives_the_given_constants():
    ins = grace_2(
        **SMALL,
        max_order=1,
        tensor_components=[2],
        compute_energy=True,
        constant_out_scale=2.5,
        atomic_shift_map={0: -1.0, 1: 3.0},
    )
    shift = ins["ConstantScaleShiftTarget"]
    assert shift.scale == 2.5
    np.testing.assert_array_equal(shift.atomic_shift_map, [-1.0, 3.0])


def test_output_targets_carry_their_tensor_rank():
    ins = grace_2(**SMALL, max_order=1)
    assert [ins[n].l for n in ("predict_L0", "predict_L1", "predict_L2")] == [0, 1, 2]
    assert [ins[n].name for n in ("predict_L0", "predict_L1", "predict_L2")] == [
        tensor_constants.PREDICT_L0_term,
        tensor_constants.PREDICT_L1_term,
        tensor_constants.PREDICT_L2_term,
    ]


@pytest.mark.parametrize("n_elements", [1, 2, 3])
def test_atom_type_dependent_reductions_follow_the_element_map(n_elements):
    element_map = {s: i for i, s in enumerate(["Si", "O", "H"][:n_elements])}
    ins = grace_2(**{**SMALL, "element_map": element_map}, max_order=2)
    for name in ("IM", "I0", "I1", "I2"):
        assert ins[name].number_of_atom_types == n_elements
        assert ins[name].is_central_atom_type_dependent
    assert not ins["I"].is_central_atom_type_dependent


def test_layer_widths_come_from_the_arguments():
    ins = grace_2(**{**SMALL, "n_mlp_dens": 5}, max_order=2)
    assert ins["R"].n_rad_max == 6
    assert ins["B"].n_out == 8  # n_rad_max[1]
    assert ins["IM"].n_out == 12
    assert ins["I"].n_out == 32
    assert ins["I0"].n_out == 6  # n_mlp_dens + 1
    assert ins["I1"].n_out == ins["I2"].n_out == 1
    assert ins["Z"].embedding_size == 6


@pytest.mark.parametrize(
    ("max_order", "expected"),
    [(1, [1]), (2, [1, 1]), (3, [1, 1, 1]), (4, [1, 1, 1, 1])],
)
def test_first_layer_indicator_ls_max_follows_lmax_indicator(max_order, expected):
    ins = grace_2(**SMALL, max_order=max_order, lmax_indicator=1)
    assert list(ins["IM"].ls_max) == expected
    assert ins["I"].ls_max == [1]


def test_highest_order_products_are_limited_by_the_tensor_rank():
    """``lmax`` of a product is the highest rank it keeps (its ``Lmax`` argument)."""
    ins = grace_2(**SMALL, max_order=4, tensor_components=[1])
    assert [ins[n].lmax for n in ("AA", "AAA", "AAAA")] == [
        2,
        1,
        1,
    ]  # AAA: max(1, lmax_indicator)
    assert [ins[n].lmax for n in ("BB", "BBB", "BBBB")] == [3, 1, 1]
    ins = grace_2(**SMALL, max_order=4, tensor_components=[1, 2])
    assert [ins[n].lmax for n in ("AAA", "AAAA", "BBB", "BBBB")] == [2, 2, 2, 2]


def test_extra_keyword_arguments_are_ignored():
    assert list(grace_2(**SMALL, max_order=1, not_an_option=1)) == list(
        grace_2(**SMALL, max_order=1)
    )


def test_a_single_scalar_component_is_refused():
    with pytest.raises(AssertionError, match="single component can not be a scalar"):
        grace_2(**SMALL, tensor_components=[0])


@pytest.mark.parametrize("components", [[3], [1, 3], [0, 1, 2, 3]])
def test_rank_above_two_is_refused(components):
    with pytest.raises(AssertionError, match="Maximum output tensor rank is 2"):
        grace_2(**SMALL, tensor_components=components)


def test_order_above_four_is_not_implemented():
    with pytest.raises(NotImplementedError, match="order > 4"):
        grace_2(**SMALL, max_order=5)


def test_indicator_lmax_is_limited_to_three():
    with pytest.raises(AssertionError, match="lmax_indicator is limited to 3"):
        grace_2(**SMALL, max_order=1, lmax_indicator=4)
    grace_2(**SMALL, max_order=1, lmax_indicator=3)  # the limit itself is accepted


# ---------------------------------------------------------------------------- physics


def structure_data(atoms: Atoms) -> dict:
    """Model input for one structure, as the calculator would feed it."""
    builder = GeometricalDataBuilder(
        elements_map=ELEMENTS, cutoff=RCUT, float_dtype="float64"
    )
    data = tf.data.Dataset.from_tensors(builder.extract_from_ase_atoms(atoms))
    data = data.get_single_element()
    n_bonds = data[constants.BOND_IND_I].shape[0]
    n_atoms = int(data[constants.N_ATOMS_BATCH_REAL])
    data[constants.ATOMIC_MU_I_LOCAL] = data[constants.ATOMIC_MU_I]
    data[constants.N_ATOMS_BATCH_TOTAL] = data[constants.N_ATOMS_BATCH_REAL]
    data[constants.BONDS_TO_STRUCTURE_MAP] = tf.zeros([n_bonds], tf.int32)
    data[constants.ATOMS_TO_STRUCTURE_MAP] = tf.zeros([n_atoms], tf.int32)
    data[constants.N_STRUCTURES_BATCH_TOTAL] = tf.constant(1)
    return data


def tensor_model(
    components: list[int], energy: bool = False, max_order: int = 2
) -> TPModel:
    tf.random.set_seed(7)
    np.random.seed(7)
    instructions = grace_2(
        **SMALL,
        max_order=max_order,
        tensor_components=components,
        compute_energy=energy,
    )
    compute = ComputeBatchEFTensor({
        "tensor_components": components,
        "compute_energy": energy,
        "compute_forces": energy,
    })
    model = TPModel(
        instructions=instructions, compute_function=compute, train_function=compute
    )
    model.build(param_dtype=tf.float64, jit_compile=False)
    return model


def cluster() -> Atoms:
    # three atoms inside the cutoff of each other; no symmetry that would hide a bug
    return Atoms("SiO2", positions=[[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [-0.4, 1.3, 0.7]])


def rotation() -> np.ndarray:
    axis = np.array([1.0, 2.0, 3.0])
    return Rotation.from_rotvec(0.7 * axis / np.linalg.norm(axis)).as_matrix()


def predict(model: TPModel, atoms: Atoms) -> dict:
    compute: Any = model.compute  # attached to the model by build()
    return {k: v.numpy() for k, v in compute(structure_data(atoms)).items()}


def assert_close(actual, expected, tol) -> None:
    np.testing.assert_allclose(actual, expected, rtol=tol.rtol, atol=tol.atol)


@pytest.mark.parametrize("components", [[0, 1, 2], [1, 2], [0, 2], [2]])
def test_rank_two_tensor_rotates_as_R_T_Rt(components):
    model = tensor_model(components)
    atoms = cluster()
    rot = rotation()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rot.T
    t0 = predict(model, atoms)[tensor_constants.PREDICT_TENSOR].reshape(-1, 3, 3)
    t1 = predict(model, rotated)[tensor_constants.PREDICT_TENSOR].reshape(-1, 3, 3)
    assert np.abs(t0).max() > 1e-3  # the check is not vacuous
    assert_close(t1, np.einsum("ij,ajk,lk->ail", rot, t0, rot), COVARIANCE_F64)


def test_vector_output_rotates_as_R_v():
    model = tensor_model([1])
    atoms = cluster()
    rot = rotation()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rot.T
    v0 = predict(model, atoms)[tensor_constants.PREDICT_TENSOR]
    v1 = predict(model, rotated)[tensor_constants.PREDICT_TENSOR]
    assert v0.shape == (3, 3)
    assert np.abs(v0).max() > 1e-3
    assert_close(v1, v0 @ rot.T, COVARIANCE_F64)


def test_tensor_is_invariant_to_translation():
    model = tensor_model([0, 1, 2])
    atoms = cluster()
    moved = atoms.copy()
    moved.positions = atoms.positions + np.array([0.3, -0.2, 5.0])
    key = tensor_constants.PREDICT_TENSOR
    assert_close(predict(model, moved)[key], predict(model, atoms)[key], COVARIANCE_F64)


def test_swapping_two_atoms_of_one_species_swaps_their_tensors():
    model = tensor_model([0, 1, 2])
    atoms = cluster()  # atoms 1 and 2 are both O
    swapped = atoms.copy()
    swapped.positions = atoms.positions[[0, 2, 1]]
    key = tensor_constants.PREDICT_TENSOR
    t0 = predict(model, atoms)[key]
    t1 = predict(model, swapped)[key]
    assert_close(t1, t0[[0, 2, 1]], COVARIANCE_F64)


def test_energy_is_invariant_and_forces_rotate_as_vectors():
    model = tensor_model([1, 2], energy=True)
    atoms = cluster()
    rot = rotation()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rot.T + np.array([0.3, -0.2, 5.0])
    out0, out1 = predict(model, atoms), predict(model, rotated)
    assert_close(
        out1[constants.PREDICT_TOTAL_ENERGY],
        out0[constants.PREDICT_TOTAL_ENERGY],
        COVARIANCE_F64,
    )
    assert_close(
        out1[constants.PREDICT_FORCES],
        out0[constants.PREDICT_FORCES] @ rot.T,
        COVARIANCE_F64,
    )
    assert_close(
        out0[constants.PREDICT_FORCES].sum(axis=0), np.zeros(3), COVARIANCE_F64
    )


def test_forces_are_minus_the_finite_difference_of_the_energy():
    model = tensor_model([1, 2], energy=True)
    atoms = cluster()
    step = 1e-4
    forces = predict(model, atoms)[constants.PREDICT_FORCES]
    numeric = np.zeros_like(forces)
    for i in range(len(atoms)):
        for k in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                shifted = atoms.copy()
                shifted.positions[i, k] += sign * step
                energies.append(
                    predict(model, shifted)[constants.PREDICT_TOTAL_ENERGY].item()
                )
            numeric[i, k] = -(energies[0] - energies[1]) / (2 * step)
    assert np.abs(forces).max() > 1e-3
    assert_close(forces, numeric, FINITE_DIFFERENCE_F64)
