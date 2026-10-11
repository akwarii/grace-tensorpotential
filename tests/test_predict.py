"""``tensorpotential.uq.cli.predict.predict_main``: the ``grace_uq predict`` command line, run in process.

The model is the random-weight GRACE-2L of the session fixture ``uq_setup`` (``model.yaml``, checkpoint and the UQ artifact
built from it) in the yaml mode of the command, on a few structures of ``tests/data``. One worker runs inline, so the
coverage run sees the worker code.

Logic layer: the columns written, the order of the rows, the options (``--save-features``, ``--no-ase-atoms``, ``--raise-errors``),
the exit code, the errors of the yaml mode. Physical values layer: per-atom sigma is non-negative, the forces of each
structure sum to zero, and a structure is predicted the same whatever its position in the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from tensorpotential.uq.cli.predict import predict_main

pytestmark = (
    pytest.mark.slow
)  # every run builds the calculator again: about two minutes in all

DATA = Path(__file__).resolve().parent / "data" / "MoNbTaW_test50.pkl.gz"
N_STRUCTURES = 4


@pytest.fixture(scope="module")
def candidates(tmp_path_factory) -> Path:
    frame = pd.read_pickle(DATA).head(N_STRUCTURES)[["ase_atoms", "name"]]
    path = tmp_path_factory.mktemp("uq_predict") / "candidates.pkl.gz"
    frame.to_pickle(path)
    return path


def yaml_args(uq_setup: dict, dataset: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--model", uq_setup["model_yaml"],
        "--checkpoint", uq_setup["checkpoint"],
        "--artifact", uq_setup["artifact_path"],
        "--dataset", str(dataset),
        "--output", str(output),
        "--n-workers", "1",
        *extra,
    ]  # fmt: skip


# Every run of the command builds a calculator and the process keeps about 350 MB of TensorFlow state for it (neither
# clear_session, a collection nor malloc_trim gives it back), and an xdist worker only grows. So a run is shared by the
# tests that read its output through a module fixture, and a test makes its own run only when it needs other options.
@pytest.fixture(scope="module")
def predicted_output(uq_setup, candidates, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("uq_predicted") / "predicted.pkl.gz"
    assert predict_main(yaml_args(uq_setup, candidates, output, "--save-features")) == 0
    return output


@pytest.fixture(scope="module")
def predicted(predicted_output) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_pickle(predicted_output))


@pytest.fixture(scope="module")
def predicted_without_atoms(uq_setup, candidates, tmp_path_factory) -> pd.DataFrame:
    output = tmp_path_factory.mktemp("uq_no_atoms") / "out.pkl.gz"
    assert predict_main(yaml_args(uq_setup, candidates, output, "--no-ase-atoms")) == 0
    return cast(pd.DataFrame, pd.read_pickle(output))


def test_the_output_has_the_input_columns_and_the_predictions(predicted):
    assert len(predicted) == N_STRUCTURES
    for column in (
        "ase_atoms",
        "name",
        "energy_predicted",
        "forces_predicted",
        "stress_predicted",
        "sigma",
        "features",
    ):
        assert column in predicted.columns, column


def test_the_rows_keep_the_order_of_the_dataset(predicted):
    expected = pd.read_pickle(DATA).head(N_STRUCTURES)["name"].tolist()
    assert predicted["name"].tolist() == expected


def test_sigma_has_one_non_negative_value_per_atom_and_gamma_needs_thresholds(
    predicted,
):
    assert (
        "gamma" not in predicted.columns
    )  # the artifact of the fixture has no calibrated thresholds
    for atoms, gamma in zip(predicted["ase_atoms"], predicted["sigma"], strict=True):
        gamma = np.asarray(gamma)
        assert gamma.shape == (len(atoms),)
        assert np.isfinite(gamma).all()
        assert (gamma >= 0).all()


def test_the_features_have_one_row_per_atom_and_the_dimension_of_the_artifact(
    predicted, uq_setup
):
    for atoms, features in zip(
        predicted["ase_atoms"], predicted["features"], strict=True
    ):
        assert np.asarray(features).shape == (len(atoms), uq_setup["feature_dim"])


def test_the_forces_of_every_structure_sum_to_zero(predicted):
    for atoms, forces in zip(
        predicted["ase_atoms"], predicted["forces_predicted"], strict=True
    ):
        assert forces.shape == (len(atoms), 3)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_a_structure_gets_the_same_prediction_whatever_its_place_in_the_dataset(
    uq_setup, candidates, predicted, tmp_path
):
    reversed_frame = pd.read_pickle(candidates).iloc[::-1]
    source = tmp_path / "reversed.pkl.gz"
    reversed_frame.to_pickle(source)
    output = tmp_path / "out.pkl.gz"
    assert predict_main(yaml_args(uq_setup, source, output)) == 0
    again = pd.read_pickle(output)
    np.testing.assert_allclose(
        again["energy_predicted"].to_numpy()[::-1],
        predicted["energy_predicted"].to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_without_save_features_no_features_column_is_written(predicted_without_atoms):
    assert "features" not in predicted_without_atoms.columns


def test_no_ase_atoms_drops_the_structures_from_the_output(predicted_without_atoms):
    assert "ase_atoms" not in predicted_without_atoms.columns
    assert len(predicted_without_atoms) == N_STRUCTURES


def test_an_extxyz_output_without_the_structures_fails_with_a_key_error(
    uq_setup, candidates, tmp_path
):
    # Current behaviour, reported on the CLI1 issue: --no-ase-atoms drops the column first, so the friendly
    # SystemExit of run_master ("extxyz output requires the ase_atoms column") is never reached.
    with pytest.raises(KeyError, match="ase_atoms"):
        predict_main(
            yaml_args(uq_setup, candidates, tmp_path / "out.xyz", "--no-ase-atoms")
        )


def test_an_extxyz_output_is_written_when_the_structures_are_kept(
    uq_setup, candidates, tmp_path
):
    from ase.io import read

    output = tmp_path / "out.xyz"
    assert predict_main(yaml_args(uq_setup, candidates, output)) == 0
    assert len(read(output, index=":")) == N_STRUCTURES


def test_the_shards_of_the_worker_are_removed_after_the_merge(predicted_output):
    assert sorted(p.name for p in predicted_output.parent.iterdir()) == [
        "predicted.pkl.gz"
    ]


def test_the_yaml_mode_needs_a_checkpoint_and_an_artifact(
    uq_setup, candidates, tmp_path
):
    argv = [
        "--model",
        uq_setup["model_yaml"],
        "--dataset",
        str(candidates),
        "--output",
        str(tmp_path / "o.pkl.gz"),
        "--n-workers",
        "1",
    ]
    with pytest.raises(SystemExit, match="--checkpoint and --artifact are required"):
        predict_main(argv)


def test_the_dataset_is_required(capsys):
    with pytest.raises(SystemExit) as stop:
        predict_main(["--model", "m"])
    assert stop.value.code == 2
    assert "the following arguments are required: --dataset" in capsys.readouterr().err


def test_a_worker_writes_its_shard_and_returns_its_exit_code(
    uq_setup, candidates, tmp_path
):
    output = tmp_path / "out.pkl.gz"
    code = predict_main(yaml_args(uq_setup, candidates, output, "--worker-id", "0"))
    assert code == 0
    assert not output.exists()  # a worker leaves a shard for the master to merge
    assert any(p.name != "out.pkl.gz" for p in tmp_path.iterdir())
