"""``TPCalculator.get_data``: the padded model input it stores, as tensors, for one structure.

Logic layer: the keys, tensor types, dtypes and values it stores against the padded arrays of its own
padding manager, and its two errors. Physical layer: the real bonds against a brute-force periodic
neighbour list (``tests/neighbour_oracle.py``), which uses neither the calculator nor matscipy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensorpotential import constants
from tensorpotential.calculator import TPCalculator
from tests.neighbour_oracle import brute_force_pairs

HERE = Path(__file__).parent.resolve()
SAVED_MODEL = str(HERE / "test_calculator_model")
STRUCTURES = HERE / "data" / "MoNbTaW_test50.pkl.gz"
CUTOFF = 6.0  # metadata.yaml of the saved model


@pytest.fixture(scope="module")
def calculator() -> TPCalculator:
    return TPCalculator(model=SAVED_MODEL)


@pytest.fixture(scope="module")
def atoms():
    frame = pd.read_pickle(STRUCTURES)  # noqa: S301 - repo test data
    return next(a for a in frame["ase_atoms"] if len(a) == 16)


def test_stored_data_are_tensors_of_the_dtype_and_values_of_the_padded_arrays(
    calculator, atoms
):
    import tensorflow as tf  # after tensorpotential, which applies the TensorFlow options first

    calculator.get_data(atoms.copy())
    stored = calculator.data
    padded = calculator.padding_manager.get_data(atoms.copy())

    assert set(stored) == set(calculator.data_keys) & set(padded)
    assert constants.BOND_VECTOR in stored
    for key, tensor in stored.items():
        assert isinstance(tensor, tf.Tensor), key
        expected = np.asarray(padded[key])
        assert tensor.dtype.as_numpy_dtype == expected.dtype, key
        np.testing.assert_array_equal(tensor.numpy(), expected, err_msg=key)


def test_the_stored_bonds_are_the_bonds_of_a_brute_force_neighbour_list(
    calculator, atoms
):
    calculator.get_data(atoms.copy())
    # the real-bond count is not a model input: it is read from the padded arrays of the builder
    n_real = int(
        calculator.padding_manager.get_data(atoms.copy())[constants.N_NEIGHBORS_REAL]
    )
    vectors = calculator.data[constants.BOND_VECTOR].numpy()[:n_real]

    pairs = brute_force_pairs(
        atoms.get_positions(), atoms.get_cell()[:], atoms.get_chemical_symbols(), CUTOFF
    )

    assert n_real == len(pairs)
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(vectors, axis=1)),
        np.sort([np.linalg.norm(p.vector) for p in pairs]),
        rtol=1e-10,
    )
    assert calculator.current_min_dist == pytest.approx(
        min(np.linalg.norm(p.vector) for p in pairs), rel=1e-10
    )


def test_a_structure_with_a_bond_shorter_than_min_dist_is_refused(atoms):
    calc = TPCalculator(model=SAVED_MODEL, min_dist=100.0)

    with pytest.raises(RuntimeError, match=r"smaller than 100\.0"):
        calc.get_data(atoms.copy())


def test_an_element_the_model_does_not_know_is_refused(calculator, atoms):
    foreign = atoms.copy()
    foreign.symbols[0] = "Cu"

    with pytest.raises(AssertionError, match="following elements only"):
        calculator.get_data(foreign)
