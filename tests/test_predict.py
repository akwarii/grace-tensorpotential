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
# clear_session, a collection nor malloc_trim gives it back), and an xdist worker only grows. Under --dist load the tests of a
# file go to different workers, so a module fixture would be built again on each of them: a run belongs to one test, which
# checks everything its output shows, and a test makes a second run only when it needs other options.
COLUMNS = ("ase_atoms", "name", "energy_predicted", "forces_predicted", "stress_predicted", "sigma")


def test_a_run_writes_the_predictions_of_every_structure_in_the_order_of_the_dataset(
    uq_setup, candidates, tmp_path
):
    output = tmp_path / "predicted.pkl.gz"
    assert predict_main(yaml_args(uq_setup, candidates, output, "--save-features")) == 0
    predicted = cast(pd.DataFrame, pd.read_pickle(output))

    # logic layer
    assert len(predicted) == N_STRUCTURES
    for column in (*COLUMNS, "features"):
        assert column in predicted.columns, column
    expected = pd.read_pickle(DATA).head(N_STRUCTURES)["name"].tolist()
    assert predicted["name"].tolist() == expected
    assert "gamma" not in predicted.columns  # the artifact of the fixture has no calibrated thresholds
    assert sorted(p.name for p in tmp_path.iterdir()) == ["predicted.pkl.gz"]  # the shards are merged and removed

    # physical values layer
    for atoms, sigma, features, forces in zip(
        predicted["ase_atoms"],
        predicted["sigma"],
        predicted["features"],
        predicted["forces_predicted"],
        strict=True,
    ):
        sigma = np.asarray(sigma)
        assert sigma.shape == (len(atoms),)  # one sigma per atom, finite and not negative
        assert np.isfinite(sigma).all()
        assert (sigma >= 0).all()
        assert np.asarray(features).shape == (len(atoms), uq_setup["feature_dim"])
        assert forces.shape == (len(atoms), 3)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_a_structure_gets_the_same_prediction_whatever_its_place_in_the_dataset(
    uq_setup, candidates, tmp_path
):
    # one run on the dataset followed by itself reversed: row k and row 2N-1-k are the same structure
    frame = pd.read_pickle(candidates)
    source = tmp_path / "doubled.pkl.gz"
    pd.concat([frame, frame.iloc[::-1]], ignore_index=True).to_pickle(source)
    output = tmp_path / "out.pkl.gz"
    assert predict_main(yaml_args(uq_setup, source, output, "--no-ase-atoms")) == 0
    predicted = cast(pd.DataFrame, pd.read_pickle(output))

    assert len(predicted) == 2 * N_STRUCTURES
    assert "ase_atoms" not in predicted.columns  # --no-ase-atoms drops the structures
    assert "features" not in predicted.columns  # and without --save-features there are no features
    energies = predicted["energy_predicted"].to_numpy()
    np.testing.assert_allclose(energies, energies[::-1], rtol=1e-9, atol=1e-9)


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
