"""``tensorpotential.uq.cli.build.master.build_main``: the ``grace_uq build`` command line, run in process.

``build_main`` validates the options, resolves the training data, and then either runs one worker step
(``--worker-id``), exports a SavedModel from an existing artifact, or hands over to ``run_master``, which starts one
subprocess per worker. The tests drive all three on a random-weight GRACE-2L model (the session fixture ``uq_setup``) and
twelve structures of ``tests/data``; every file is written to ``tmp_path``.

Logic layer: each validation error (message and exit code 2), the three modes, the resolution of the training data from
``../../input.yaml``, a sharded TF dataset, the filter function. Physical values layer (the artifact of the full run):
the per-element atom counts equal the hand count of the training set, the inverse covariances and scatter matrices are
symmetric and (semi-)definite, the thresholds are positive, and the stored random projection has the requested size.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensorpotential.instructions.base import load_instructions
from tensorpotential.scripts import grace_preprocess
from tensorpotential.tpmodel import extract_cutoff_and_elements
from tensorpotential.uq import constants as uqc
from tensorpotential.uq.cli.build import build_main
from tensorpotential.uq.cli.info import info_main

DATA = Path(__file__).resolve().parent / "data" / "MoNbTaW_test50.pkl.gz"
N_TRAIN = 12
RP_DIM = 16
ELEMENTS = ["Mo", "Nb", "Ta", "W"]


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    return pd.read_pickle(DATA).head(N_TRAIN)[["ase_atoms"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def model_dir(uq_setup, tmp_path_factory) -> Path:
    """Copies of the random-weight model files of ``uq_setup``: ``model.yaml`` and ``checkpoint.*``."""
    folder = tmp_path_factory.mktemp("uq_model")
    shutil.copy(uq_setup["model_yaml"], folder / "model.yaml")
    for source in Path(uq_setup["checkpoint"]).parent.glob(Path(uq_setup["checkpoint"]).name + "*"):
        shutil.copy(source, folder / source.name)
    return folder


def common_args(model_dir: Path, *extra: str) -> list[str]:
    return [
        "--n-workers",
        "1",
        "--rp-dim",
        str(RP_DIM),
        "--model-yaml",
        str(model_dir / "model.yaml"),
        "--checkpoint",
        str(model_dir / "checkpoint.index"),
        *extra,
    ]


@pytest.fixture(scope="module")
def master_run(model_dir, training_frame, tmp_path_factory):
    """One full ``grace_uq build`` (three worker steps in subprocesses) started from a fit-like folder.

    The folder is ``fit/seed/1`` with ``fit/input.yaml`` naming ``train.pkl.gz``, and the command is run in ``seed/1`` without
    ``--train-data``, so the data come from the ``../../input.yaml`` fallback.
    """
    fit = tmp_path_factory.mktemp("uq_fit")
    seed = fit / "seed" / "1"
    seed.mkdir(parents=True)
    (fit / "input.yaml").write_text("data:\n  filename: train.pkl.gz\n")
    training_frame.to_pickle(fit / "train.pkl.gz")
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.chdir(seed)
        build_main(common_args(model_dir, "--n-clusters", "2", "--no-export"))
    return {"seed": seed, "fit": fit, "artifact": seed / "gmm_artifacts.npz", "stdout": out.getvalue()}


def atoms_per_element(frame: pd.DataFrame) -> dict[str, int]:
    counts = dict.fromkeys(ELEMENTS, 0)
    for atoms in frame["ase_atoms"]:
        for symbol in atoms.get_chemical_symbols():
            counts[symbol] += 1
    return counts


# --- the full run ------------------------------------------------------------------------------------------------------


def test_the_training_data_come_from_the_input_yaml_two_levels_up(master_run):
    assert "training_set.pkl.gz not found; resolved --train-data=" in master_run["stdout"]
    assert str(master_run["fit"] / "train.pkl.gz") in master_run["stdout"]


def test_the_run_writes_the_artifact_next_to_the_working_directory_and_no_saved_model(master_run):
    assert master_run["artifact"].is_file()
    assert not (master_run["seed"] / "saved_model").exists()


def test_the_artifact_counts_every_training_atom_of_each_element_once(master_run, training_frame):
    expected = atoms_per_element(training_frame)
    with np.load(master_run["artifact"]) as artifact:
        assert list(artifact["element_map"]) == ELEMENTS
        for index, symbol in enumerate(ELEMENTS):
            assert artifact[f"counts_{index}"].sum() == expected[symbol]
            assert artifact[f"counts_{index}"].shape == (2,)


def test_the_artifact_stores_a_random_projection_of_the_requested_size(master_run):
    with np.load(master_run["artifact"]) as artifact:
        assert int(artifact["uq_rp_dim"]) == RP_DIM
        assert int(artifact["uq_rp_seed"]) == 42
        assert artifact["uq_rp_matrix"].shape[1] == RP_DIM
        assert int(artifact[uqc.SCHEMA_VERSION_KEY]) == uqc.SCHEMA_VERSION


def test_inverse_covariances_and_scatter_matrices_of_populated_clusters_are_symmetric_and_definite(master_run):
    checked = 0
    with np.load(master_run["artifact"]) as artifact:
        for index in range(len(ELEMENTS)):
            for cluster, count in enumerate(artifact[f"counts_{index}"]):
                inverse = artifact[f"inv_cov_{index}"][cluster].astype(np.float64)
                scatter = artifact[f"scatter_{index}"][cluster].astype(np.float64)
                if count == 0:  # an empty cluster keeps the sentinel values: no scatter, a very stiff metric
                    assert not scatter.any()
                    continue
                checked += 1
                np.testing.assert_allclose(inverse, inverse.T, rtol=1e-6, atol=0.0)
                np.testing.assert_allclose(scatter, scatter.T, rtol=1e-6, atol=1e-6 * np.abs(scatter).max())
                assert np.linalg.eigvalsh(inverse).min() > 0.0
                assert np.linalg.eigvalsh(scatter).min() > -1e-6 * np.abs(scatter).max()
    assert checked >= len(ELEMENTS)


def test_the_thresholds_are_finite_and_positive_for_every_element_and_cluster(master_run):
    with np.load(master_run["artifact"]) as artifact:
        thresholds = artifact["interp_thresholds"]
        assert thresholds.shape == (len(ELEMENTS), 2)
        assert np.isfinite(thresholds).all()
        assert (thresholds > 0).all()


def test_info_reads_the_artifact_of_the_run(master_run, capsys):
    assert info_main([str(master_run["artifact"])]) == 0
    out = capsys.readouterr().out
    assert "feature_dim D" in out
    assert all(symbol in out for symbol in ELEMENTS)


# --- one worker step ---------------------------------------------------------------------------------------------------


def worker_args(model_dir: Path, work: Path, data: Path, step: int, *extra: str) -> list[str]:
    return common_args(
        model_dir,
        "--worker-id",
        "0",
        "--step",
        str(step),
        "--train-data",
        str(data),
        "--artifact-path",
        str(work / "gmm_artifacts.npz"),
        *extra,
    )


@pytest.fixture
def data_file(training_frame, tmp_path) -> Path:
    path = tmp_path / "train.pkl.gz"
    training_frame.to_pickle(path)
    return path


def test_worker_step_1_writes_one_file_per_candidate_k_and_is_skipped_when_done(model_dir, data_file, tmp_path, capsys):
    argv = worker_args(model_dir, tmp_path, data_file, 1, "--n-clusters", "2", "3")
    build_main(argv)
    files = [tmp_path / ".step1_w0_k2.npz", tmp_path / ".step1_w0_k3.npz"]
    assert all(f.is_file() for f in files)
    assert "Worker 0 Step 1: 2 file(s) missing, running." in capsys.readouterr().out
    stamps = [f.stat().st_mtime_ns for f in files]
    build_main(argv)
    assert "Worker 0 Step 1 already completed, skipping." in capsys.readouterr().out
    assert [f.stat().st_mtime_ns for f in files] == stamps


def test_worker_steps_2_and_3_use_the_merged_artifacts_of_the_previous_step(model_dir, data_file, master_run, tmp_path):
    seed = master_run["seed"]
    build_main(worker_args(model_dir, tmp_path, data_file, 2, "--n-clusters", "2", "--step1-artifacts", str(seed / ".step1_final.npz")))
    assert (tmp_path / ".step2_w0.npz").is_file()
    build_main(worker_args(model_dir, tmp_path, data_file, 3, "--n-clusters", "2", "--step2-artifacts", str(seed / ".step2_final.npz")))
    assert (tmp_path / ".step3_w0.npz").is_file()


def test_a_worker_applies_the_filter_function_and_the_fraction(model_dir, data_file, tmp_path, monkeypatch):
    module = tmp_path / "keep_big_filter.py"
    module.write_text("def keep_big(atoms):\n    return len(atoms) >= 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    build_main(
        worker_args(
            model_dir, tmp_path, data_file, 1, "--n-clusters", "2", "--filter-fn", "keep_big_filter:keep_big", "--frac", "0.5"
        )
    )
    assert (tmp_path / ".step1_w0_k2.npz").is_file()


def test_a_worker_reads_a_sharded_tf_dataset_and_rejects_the_filter_function_for_it(model_dir, tmp_path, monkeypatch):
    source = tmp_path / "train.pkl.gz"
    pd.read_pickle(DATA).head(N_TRAIN).to_pickle(source)  # grace_preprocess needs the energy and force columns
    cutoff = extract_cutoff_and_elements(load_instructions(str(model_dir / "model.yaml")))[0]
    dataset = tmp_path / "tf_dataset"
    for stage in (1, 2, 3, 4):
        grace_preprocess.main([str(source), "-o", str(dataset), "-e", " ".join(ELEMENTS), "-c", str(cutoff), f"--stage-{stage}", "-b", "4"])
    assert (dataset / "stage3" / "stats.json").is_file()
    shards = dataset / "stage3"
    assert json.loads((shards / "stats.json").read_text())["total_num_structures"] == N_TRAIN

    build_main(worker_args(model_dir, tmp_path, shards, 1, "--n-clusters", "2"))
    assert (tmp_path / ".step1_w0_k2.npz").is_file()

    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "sharded_filter_for_tests.py").write_text("def keep(atoms):\n    return True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="require pickle/df shards"):
        build_main(worker_args(model_dir, other, shards, 1, "--n-clusters", "2", "--filter-fn", "sharded_filter_for_tests:keep"))


# --- standalone export ------------------------------------------------------------------------------------------------


def test_without_training_data_an_existing_artifact_is_exported(model_dir, master_run, tmp_path, capsys):
    artifact = tmp_path / "gmm_artifacts.npz"
    shutil.copy(master_run["artifact"], artifact)
    build_main(common_args(model_dir, "--artifact-path", str(artifact), "--train-data", str(tmp_path / "absent.pkl.gz")))
    out = capsys.readouterr().out
    assert f"Standalone export: {artifact} -> {tmp_path / 'saved_model'}" in out
    assert f"SUCCESS: SavedModel exported to {tmp_path / 'saved_model'}" in out
    assert (tmp_path / "saved_model" / "saved_model.pb").is_file()


# --- validation --------------------------------------------------------------------------------------------------------


def fails(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as stop:
        build_main(list(argv))
    assert stop.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "100.5", "nan"])
def test_threshold_percentile_outside_zero_to_hundred_is_an_error(tmp_path, capsys, value):
    err = fails(capsys, "--threshold-percentile", value, "--train-data", str(tmp_path / "x.pkl.gz"))
    assert f"--threshold-percentile must be in (0, 100], got {value}" in err


def test_threshold_percentile_hundred_is_accepted_up_to_the_data_check(tmp_path, capsys):
    err = fails(capsys, "--threshold-percentile", "100", "--train-data", str(tmp_path / "x.pkl.gz"), "--artifact-path", str(tmp_path / "a.npz"))
    assert "--threshold-percentile must be" not in err
    assert "--train-data files do not exist" in err


def test_weighted_and_plain_training_data_are_mutually_exclusive(tmp_path, capsys):
    data = tmp_path / "x.pkl.gz"
    data.write_bytes(b"")
    err = fails(capsys, "--train-data", str(data), "--train-data-weighted", "2.0", str(data))
    assert "--train-data and --train-data-weighted are mutually exclusive" in err


def test_a_worker_without_existing_training_data_is_an_error(tmp_path, capsys):
    err = fails(capsys, "--worker-id", "0", "--step", "1", "--train-data", str(tmp_path / "absent.pkl.gz"))
    assert f"--train-data files do not exist: ['{tmp_path / 'absent.pkl.gz'}']" in err


def test_the_master_without_data_and_without_artifact_is_an_error(tmp_path, capsys):
    err = fails(capsys, "--train-data", str(tmp_path / "absent.pkl.gz"), "--artifact-path", str(tmp_path / "a.npz"))
    assert "--train-data files do not exist" in err
    assert f"--artifact-path is not usable for export-only mode ({tmp_path / 'a.npz'})" in err
    assert "--restart was given" not in err


def test_restart_does_not_fall_through_to_the_export_of_an_existing_artifact(tmp_path, capsys):
    (tmp_path / "a.npz").write_bytes(b"")
    err = fails(capsys, "--train-data", str(tmp_path / "absent.pkl.gz"), "--artifact-path", str(tmp_path / "a.npz"), "--restart")
    assert "--restart was given but no training data could be located." in err


def test_exporting_nothing_is_an_error(tmp_path, capsys):
    (tmp_path / "a.npz").write_bytes(b"")
    err = fails(capsys, "--train-data", str(tmp_path / "absent.pkl.gz"), "--artifact-path", str(tmp_path / "a.npz"), "--no-export")
    assert "Nothing to do: --train-data not provided and --no-export set" in err


def test_a_filter_function_that_cannot_be_imported_fails_before_any_worker_starts(tmp_path):
    data = tmp_path / "x.pkl.gz"
    data.write_bytes(b"")
    with pytest.raises(ImportError, match="cannot import module"):
        build_main(["--train-data", str(data), "--filter-fn", "no_such_module_for_tests:fn", "--artifact-path", str(tmp_path / "a.npz")])


def test_paths_are_made_absolute_and_a_directory_artifact_gets_the_default_name(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    err = fails(capsys, "--train-data", "absent.pkl.gz", "--artifact-path", "out")
    assert f"['{tmp_path / 'absent.pkl.gz'}']" in err
    assert f"({tmp_path / 'out' / 'gmm_artifacts.npz'})" in err


def test_the_options_keep_their_documented_defaults():
    from tensorpotential.uq.cli.build.master import DEFAULT_TRAIN_DATA

    assert DEFAULT_TRAIN_DATA == ["training_set.pkl.gz"]
