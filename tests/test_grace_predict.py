"""``tensorpotential.scripts.grace_predict``: predict energies, forces and stresses of a dataframe (``main``).

The model is the committed random-weight SavedModel ``tests/test_calculator_model`` (Mo, Nb, Ta, W; cutoff 6 A). Logic layer:
default and explicit file names, the columns written, the order of the rows, a structure that cannot be evaluated (warn
or raise), the magnetic-moment helpers. Physical values layer: every row equals what ``ase`` gets from a fresh ``TPCalculator`` on
the same structure (an independent call path: no ``predict_structures``, no padding options), and the forces of each
structure sum to zero (translation invariance).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from tensorpotential.calculator import TPCalculator
from tensorpotential.scripts import grace_predict
from tests import cli_help
from tests.tolerances import COVARIANCE_F64 as TOL

HERE = Path(__file__).resolve().parent
SAVED_MODEL = HERE / "test_calculator_model"
N_STRUCTURES = 4


@pytest.fixture(scope="module")
def structures() -> pd.DataFrame:
    """Four structures of different size from the MoNbTaW test set, with a name column that must survive."""
    df = pd.read_pickle(HERE / "data" / "MoNbTaW_test50.pkl.gz")
    order = df["ase_atoms"].map(len).sort_values(kind="stable").index
    chosen = df.loc[order[[0, 10, 25, 40]]].reset_index(drop=True)
    assert chosen["ase_atoms"].map(len).nunique() > 1
    return chosen[["ase_atoms", "name"]]


@pytest.fixture(scope="module")
def direct() -> list[dict]:
    """Energy, forces and stress of the same structures through the plain ASE interface of a fresh calculator."""
    df = pd.read_pickle(HERE / "data" / "MoNbTaW_test50.pkl.gz")
    order = df["ase_atoms"].map(len).sort_values(kind="stable").index
    calc = TPCalculator(model=str(SAVED_MODEL))
    results = []
    for atoms in df.loc[order[[0, 10, 25, 40]], "ase_atoms"]:
        atoms = atoms.copy()
        atoms.calc = calc
        results.append({
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(),
        })
    return results


def run_predict(*argv: str) -> tuple[int, str, str]:
    return cli_help.run_main(grace_predict.main, "grace_predict", list(argv))


@pytest.fixture
def predicted(structures, tmp_path) -> pd.DataFrame:
    source = tmp_path / "in.pkl.gz"
    structures.to_pickle(source, compression="gzip")
    target = tmp_path / "out.pkl.gz"
    code, _, _ = run_predict(
        "-m", str(SAVED_MODEL), "-d", str(source), "-o", str(target)
    )
    assert code == 0
    return pd.read_pickle(target, compression="gzip")


def test_the_output_has_the_input_columns_without_the_structures_and_the_three_predictions(
    predicted,
):
    assert list(predicted.columns) == [
        "name",
        "energy_predicted",
        "forces_predicted",
        "stress_predicted",
    ]
    assert len(predicted) == N_STRUCTURES


def test_the_other_columns_of_the_input_are_kept(predicted, structures):
    assert list(predicted["name"]) == list(structures["name"])


def test_each_row_equals_the_plain_calculator_result_in_the_order_of_the_input(
    predicted, direct
):
    for (_, row), reference in zip(predicted.iterrows(), direct, strict=True):
        np.testing.assert_allclose(
            row["energy_predicted"], reference["energy"], rtol=TOL.rtol, atol=TOL.atol
        )
        np.testing.assert_allclose(
            row["forces_predicted"], reference["forces"], rtol=TOL.rtol, atol=TOL.atol
        )
        np.testing.assert_allclose(
            row["stress_predicted"], reference["stress"], rtol=TOL.rtol, atol=TOL.atol
        )


def test_the_forces_of_every_structure_sum_to_zero(predicted):
    for forces in predicted["forces_predicted"]:
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_the_force_arrays_have_one_row_per_atom(predicted, structures):
    for forces, atoms in zip(
        predicted["forces_predicted"], structures["ase_atoms"], strict=True
    ):
        assert forces.shape == (len(atoms), 3)


def test_default_file_names_and_model_directory(structures, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_model").symlink_to(SAVED_MODEL, target_is_directory=True)
    structures.head(2).to_pickle(tmp_path / "dataset.pkl.gz", compression="gzip")
    code, _, _ = run_predict()
    assert code == 0
    result = pd.read_pickle(tmp_path / "predicted_dataset.pkl.gz", compression="gzip")
    assert len(result) == 2


def test_a_structure_the_model_cannot_evaluate_is_a_warning_and_leaves_none(
    structures, tmp_path, caplog
):
    frame = pd.concat(
        [
            structures.head(2),
            pd.DataFrame({"ase_atoms": [bulk("Cu", cubic=True)], "name": ["copper"]}),
        ],
        ignore_index=True,
    )
    source = tmp_path / "in.pkl.gz"
    frame.to_pickle(source, compression="gzip")
    with caplog.at_level(logging.WARNING):
        run_predict(
            "-m",
            str(SAVED_MODEL),
            "-d",
            str(source),
            "-o",
            str(tmp_path / "out.pkl.gz"),
        )
    result = pd.read_pickle(tmp_path / "out.pkl.gz", compression="gzip")
    assert np.isnan(result["energy_predicted"].iloc[2])
    assert result["forces_predicted"].iloc[2] is None
    assert result["stress_predicted"].iloc[2] is None
    assert np.isfinite(result["energy_predicted"].iloc[:2]).all()
    assert "prediction failed for 1 of 3 structures" in caplog.text


@pytest.mark.parametrize("flag", ["-e", "--raise-errors"])
def test_raise_errors_stops_at_the_first_unsupported_structure(
    structures, tmp_path, flag
):
    frame = pd.concat(
        [
            structures.head(1),
            pd.DataFrame({"ase_atoms": [bulk("Cu", cubic=True)], "name": ["copper"]}),
        ],
        ignore_index=True,
    )
    source = tmp_path / "in.pkl.gz"
    frame.to_pickle(source, compression="gzip")
    with pytest.raises(
        AssertionError, match="configured to process the following elements only"
    ):
        run_predict(
            "-m",
            str(SAVED_MODEL),
            "-d",
            str(source),
            "-o",
            str(tmp_path / "out.pkl.gz"),
            flag,
        )
    assert not (tmp_path / "out.pkl.gz").exists()


def test_a_missing_dataset_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_predict(
            "-m",
            str(SAVED_MODEL),
            "-d",
            str(tmp_path / "absent.pkl.gz"),
            "-o",
            str(tmp_path / "out.pkl.gz"),
        )


def test_the_model_path_is_made_absolute_before_loading(
    structures, tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(HERE)
    structures.head(1).to_pickle(tmp_path / "in.pkl.gz", compression="gzip")
    with caplog.at_level(logging.INFO):
        run_predict(
            "-m",
            "test_calculator_model",
            "-d",
            str(tmp_path / "in.pkl.gz"),
            "-o",
            str(tmp_path / "out.pkl.gz"),
        )
    assert f"Loading model from: {SAVED_MODEL}" in caplog.text


# --- magnetic moments --------------------------------------------------------------------------------------------------


def test_set_magmom_with_three_components_per_atom_is_stored_as_given():
    moments = np.arange(12.0).reshape(4, 3)
    atoms = grace_predict.set_magmom(bulk("Cu", cubic=True), moments)
    np.testing.assert_array_equal(atoms.arrays["initial_magmoms"], moments)


def test_set_magmom_with_one_value_per_atom_puts_it_on_the_z_axis():
    atoms = grace_predict.set_magmom(
        bulk("Cu", cubic=True), np.array([1.0, 2.0, 3.0, 4.0])
    )
    expected = np.zeros((4, 3))
    expected[:, 2] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(atoms.arrays["initial_magmoms"], expected)


def test_set_magmom_with_a_column_of_values_fails_although_the_code_has_a_branch_for_it():
    # Current behaviour, reported on the CLI1 issue: the (n, 1) branch assigns a (n, 1) array to a (n,) slice.
    with pytest.raises(ValueError, match="could not broadcast"):
        grace_predict.set_magmom(
            bulk("Cu", cubic=True), np.array([[1.0], [2.0], [3.0], [4.0]])
        )


def test_set_magmom_rejects_an_unknown_shape():
    with pytest.raises(ValueError, match="mag_mom shape is not recognized"):
        grace_predict.set_magmom(bulk("Cu", cubic=True), np.zeros((4, 2)))


def test_set_magmom_rejects_a_wrong_number_of_atoms():
    with pytest.raises(AssertionError):
        grace_predict.set_magmom(bulk("Cu", cubic=True), np.zeros(3))


def test_atoms_without_a_mag_mom_column_are_handed_over_as_they_are(structures):
    handed = grace_predict._atoms_with_magmoms(structures)
    assert all(a is b for a, b in zip(handed, structures["ase_atoms"], strict=True))


def test_mag_mom_entries_are_applied_to_copies_and_none_leaves_the_structure_alone(
    structures,
):
    frame = structures.head(2).copy()
    n = len(frame["ase_atoms"].iloc[1])
    frame["mag_mom"] = [None, np.full(n, 2.0)]
    handed = grace_predict._atoms_with_magmoms(frame)
    assert handed[0] is frame["ase_atoms"].iloc[0]
    assert handed[1] is not frame["ase_atoms"].iloc[1]
    np.testing.assert_array_equal(
        handed[1].arrays["initial_magmoms"][:, 2], np.full(n, 2.0)
    )
    assert "initial_magmoms" not in frame["ase_atoms"].iloc[1].arrays
