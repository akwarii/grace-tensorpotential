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
import logging
import os
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
from tensorpotential.uq.cli.build import master
from tensorpotential.uq.cli.build.master import (
    _flatten_weighted_train_data,
    _worker_has_finished,
    build_parser,
)
from tensorpotential.uq.cli.info import info_main
from tests import cli_help

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
    for source in Path(uq_setup["checkpoint"]).parent.glob(
        Path(uq_setup["checkpoint"]).name + "*"
    ):
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
        code = build_main(common_args(model_dir, "--n-clusters", "2", "--no-export"))
    return {
        "code": code,
        "seed": seed,
        "fit": fit,
        "artifact": seed / "gmm_artifacts.npz",
        "stdout": out.getvalue(),
    }


def atoms_per_element(frame: pd.DataFrame) -> dict[str, int]:
    counts = dict.fromkeys(ELEMENTS, 0)
    for atoms in frame["ase_atoms"]:
        for symbol in atoms.get_chemical_symbols():
            counts[symbol] += 1
    return counts


# --- the full run ------------------------------------------------------------------------------------------------------


def test_the_training_data_come_from_the_input_yaml_two_levels_up(master_run):
    assert (
        "training_set.pkl.gz not found; resolved --train-data=" in master_run["stdout"]
    )
    assert str(master_run["fit"] / "train.pkl.gz") in master_run["stdout"]


def test_the_run_writes_the_artifact_next_to_the_working_directory_and_no_saved_model(
    master_run,
):
    assert master_run["code"] == 0
    assert master_run["artifact"].is_file()
    assert not (master_run["seed"] / "saved_model").exists()


def test_the_artifact_counts_every_training_atom_of_each_element_once(
    master_run, training_frame
):
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


def test_inverse_covariances_and_scatter_matrices_of_populated_clusters_are_symmetric_and_definite(
    master_run,
):
    checked = 0
    with np.load(master_run["artifact"]) as artifact:
        for index in range(len(ELEMENTS)):
            for cluster, count in enumerate(artifact[f"counts_{index}"]):
                inverse = artifact[f"inv_cov_{index}"][cluster].astype(np.float64)
                scatter = artifact[f"scatter_{index}"][cluster].astype(np.float64)
                if (
                    count == 0
                ):  # an empty cluster keeps the sentinel values: no scatter, a very stiff metric
                    assert not scatter.any()
                    continue
                checked += 1
                np.testing.assert_allclose(inverse, inverse.T, rtol=1e-6, atol=0.0)
                np.testing.assert_allclose(
                    scatter, scatter.T, rtol=1e-6, atol=1e-6 * np.abs(scatter).max()
                )
                assert np.linalg.eigvalsh(inverse).min() > 0.0
                assert np.linalg.eigvalsh(scatter).min() > -1e-6 * np.abs(scatter).max()
    assert checked >= len(ELEMENTS)


def test_the_thresholds_are_finite_and_positive_for_every_element_and_cluster(
    master_run,
):
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


def worker_args(
    model_dir: Path, work: Path, data: Path, step: int, *extra: str
) -> list[str]:
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


def test_worker_step_1_writes_one_file_per_candidate_k_and_is_skipped_when_done(
    model_dir, data_file, tmp_path, capsys
):
    argv = worker_args(model_dir, tmp_path, data_file, 1, "--n-clusters", "2", "3")
    build_main(argv)
    files = [tmp_path / ".step1_w0_k2.npz", tmp_path / ".step1_w0_k3.npz"]
    assert all(f.is_file() for f in files)
    assert "Worker 0 Step 1: 2 file(s) missing, running." in capsys.readouterr().out
    stamps = [f.stat().st_mtime_ns for f in files]
    build_main(argv)
    assert "Worker 0 Step 1 already completed, skipping." in capsys.readouterr().out
    assert [f.stat().st_mtime_ns for f in files] == stamps


def test_worker_steps_2_and_3_use_the_merged_artifacts_of_the_previous_step(
    model_dir, data_file, master_run, tmp_path
):
    seed = master_run["seed"]
    build_main(
        worker_args(
            model_dir,
            tmp_path,
            data_file,
            2,
            "--n-clusters",
            "2",
            "--step1-artifacts",
            str(seed / ".step1_final.npz"),
        )
    )
    assert (tmp_path / ".step2_w0.npz").is_file()
    build_main(
        worker_args(
            model_dir,
            tmp_path,
            data_file,
            3,
            "--n-clusters",
            "2",
            "--step2-artifacts",
            str(seed / ".step2_final.npz"),
        )
    )
    assert (tmp_path / ".step3_w0.npz").is_file()


def test_a_worker_applies_the_filter_function_and_the_fraction(
    model_dir, data_file, tmp_path, monkeypatch
):
    module = tmp_path / "keep_big_filter.py"
    module.write_text("def keep_big(atoms):\n    return len(atoms) >= 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    build_main(
        worker_args(
            model_dir,
            tmp_path,
            data_file,
            1,
            "--n-clusters",
            "2",
            "--filter-fn",
            "keep_big_filter:keep_big",
            "--frac",
            "0.5",
            "--checkpoint",
            str(model_dir / "checkpoint"),  # named without the .index suffix the other tests give it
        )
    )
    assert (tmp_path / ".step1_w0_k2.npz").is_file()


def test_a_worker_reads_a_sharded_tf_dataset_and_rejects_the_filter_function_for_it(
    model_dir, tmp_path, monkeypatch
):
    source = tmp_path / "train.pkl.gz"
    pd.read_pickle(DATA).head(N_TRAIN).to_pickle(
        source
    )  # grace_preprocess needs the energy and force columns
    cutoff = extract_cutoff_and_elements(
        load_instructions(str(model_dir / "model.yaml"))
    )[0]
    dataset = tmp_path / "tf_dataset"
    for stage in (1, 2, 3, 4):
        grace_preprocess.main([
            str(source),
            "-o",
            str(dataset),
            "-e",
            " ".join(ELEMENTS),
            "-c",
            str(cutoff),
            f"--stage-{stage}",
            "-b",
            "4",
        ])
    assert (dataset / "stage3" / "stats.json").is_file()
    shards = dataset / "stage3"
    assert (
        json.loads((shards / "stats.json").read_text())["total_num_structures"]
        == N_TRAIN
    )

    build_main(worker_args(model_dir, tmp_path, shards, 1, "--n-clusters", "2"))
    assert (tmp_path / ".step1_w0_k2.npz").is_file()

    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "sharded_filter_for_tests.py").write_text(
        "def keep(atoms):\n    return True\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="require pickle/df shards"):
        build_main(
            worker_args(
                model_dir,
                other,
                shards,
                1,
                "--n-clusters",
                "2",
                "--filter-fn",
                "sharded_filter_for_tests:keep",
            )
        )


# --- standalone export ------------------------------------------------------------------------------------------------


def test_without_training_data_an_existing_artifact_is_exported(
    model_dir, master_run, tmp_path, capsys
):
    artifact = tmp_path / "gmm_artifacts.npz"
    shutil.copy(master_run["artifact"], artifact)
    build_main(
        common_args(
            model_dir,
            "--artifact-path",
            str(artifact),
            "--train-data",
            str(tmp_path / "absent.pkl.gz"),
        )
    )
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
def test_threshold_percentile_outside_zero_to_hundred_is_an_error(
    tmp_path, capsys, value
):
    err = fails(
        capsys,
        "--threshold-percentile",
        value,
        "--train-data",
        str(tmp_path / "x.pkl.gz"),
    )
    assert f"--threshold-percentile must be in (0, 100], got {value}" in err


def test_threshold_percentile_hundred_is_accepted_up_to_the_data_check(
    tmp_path, capsys
):
    err = fails(
        capsys,
        "--threshold-percentile",
        "100",
        "--train-data",
        str(tmp_path / "x.pkl.gz"),
        "--artifact-path",
        str(tmp_path / "a.npz"),
    )
    assert "--threshold-percentile must be" not in err
    assert "--train-data files do not exist" in err


def test_weighted_and_plain_training_data_are_mutually_exclusive(tmp_path, capsys):
    data = tmp_path / "x.pkl.gz"
    data.write_bytes(b"")
    err = fails(
        capsys, "--train-data", str(data), "--train-data-weighted", "2.0", str(data)
    )
    assert "--train-data and --train-data-weighted are mutually exclusive" in err


def test_a_worker_without_existing_training_data_is_an_error(tmp_path, capsys):
    err = fails(
        capsys,
        "--worker-id",
        "0",
        "--step",
        "1",
        "--train-data",
        str(tmp_path / "absent.pkl.gz"),
    )
    assert f"--train-data files do not exist: ['{tmp_path / 'absent.pkl.gz'}']" in err


def test_the_master_without_data_and_without_artifact_is_an_error(tmp_path, capsys):
    err = fails(
        capsys,
        "--train-data",
        str(tmp_path / "absent.pkl.gz"),
        "--artifact-path",
        str(tmp_path / "a.npz"),
    )
    assert "--train-data files do not exist" in err
    assert (
        f"--artifact-path is not usable for export-only mode ({tmp_path / 'a.npz'})"
        in err
    )
    assert "--restart was given" not in err


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--restart", "--restart was given but no training data could be located."),
        ("--no-export", "Nothing to do: --train-data not provided and --no-export set"),
    ],
)
def test_an_existing_artifact_is_not_exported_when_the_options_forbid_it(
    tmp_path, capsys, flag, message
):
    # --restart asks for a full rebuild; --no-export leaves nothing to do without training data
    (tmp_path / "a.npz").write_bytes(b"")
    err = fails(
        capsys,
        "--train-data",
        str(tmp_path / "absent.pkl.gz"),
        "--artifact-path",
        str(tmp_path / "a.npz"),
        flag,
    )
    assert message in err


def test_a_filter_function_that_cannot_be_imported_fails_before_any_worker_starts(
    tmp_path,
):
    data = tmp_path / "x.pkl.gz"
    data.write_bytes(b"")
    with pytest.raises(ImportError, match="cannot import module"):
        build_main([
            "--train-data",
            str(data),
            "--filter-fn",
            "no_such_module_for_tests:fn",
            "--artifact-path",
            str(tmp_path / "a.npz"),
        ])


def test_paths_are_made_absolute_and_a_directory_artifact_gets_the_default_name(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    err = fails(capsys, "--train-data", "absent.pkl.gz", "--artifact-path", "out")
    assert f"['{tmp_path / 'absent.pkl.gz'}']" in err
    assert f"({tmp_path / 'out' / 'gmm_artifacts.npz'})" in err


def test_the_options_keep_their_documented_defaults():
    from tensorpotential.uq.cli.build.master import DEFAULT_TRAIN_DATA

    assert DEFAULT_TRAIN_DATA == ["training_set.pkl.gz"]


def test_a_sharded_dataset_without_statistics_is_read_all_the_same(model_dir, tmp_path):
    source = tmp_path / "train.pkl.gz"
    pd.read_pickle(DATA).head(N_TRAIN).to_pickle(source)
    cutoff = extract_cutoff_and_elements(
        load_instructions(str(model_dir / "model.yaml"))
    )[0]
    dataset = tmp_path / "tf_dataset"
    for stage in (1, 2, 3):  # no stage 4: stage3/stats.json is absent
        grace_preprocess.main([
            str(source),
            "-o",
            str(dataset),
            "-e",
            " ".join(ELEMENTS),
            "-c",
            str(cutoff),
            f"--stage-{stage}",
            "-b",
            "4",
        ])
    assert not (dataset / "stage3" / "stats.json").exists()
    build_main(
        worker_args(model_dir, tmp_path, dataset / "stage3", 1, "--n-clusters", "2")
    )
    assert (tmp_path / ".step1_w0_k2.npz").is_file()


def test_a_finished_worker_step_returns_zero_without_loading_the_model(
    tmp_path, data_file, monkeypatch, capsys
):
    # the expected output of step 1 exists, so no model is needed: nothing but the argument parsing runs
    for k in (2, 3):
        (tmp_path / f".step1_w0_k{k}.npz").write_bytes(b"")
    cli_help.poison_sys_argv(monkeypatch)
    argv = [
        "--worker-id",
        "0",
        "--step",
        "1",
        "--n-workers",
        "1",
        "--n-clusters",
        "2",
        "3",
        "--train-data",
        str(data_file),
        "--artifact-path",
        str(tmp_path / "gmm_artifacts.npz"),
        "--model-yaml",
        str(tmp_path / "absent.yaml"),
    ]
    assert build_main(argv) == 0
    assert "already completed, skipping" in capsys.readouterr().out


def test_the_parser_of_the_build_command_is_available_without_running_it():
    assert build_parser().prog == "grace_uq build"


def test_weighted_groups_become_the_flat_path_list_and_the_weight_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        ["--train-data-weighted", "2.0", "a.pkl.gz", "b.pkl.gz", "--train-data-weighted", "0.5", "c.pkl.gz"]
    )
    assert _flatten_weighted_train_data(parser, args) is True  # --train-data was left at its default
    paths = [str(tmp_path / name) for name in ("a.pkl.gz", "b.pkl.gz", "c.pkl.gz")]
    assert args.train_data == paths
    assert args.file_weight_map == dict(zip(paths, (2.0, 2.0, 0.5), strict=True))


def test_plain_train_data_gives_an_empty_weight_map_and_is_left_alone():
    parser = build_parser()
    args = parser.parse_args(["--train-data", "a.pkl.gz"])
    assert _flatten_weighted_train_data(parser, args) is False
    assert args.train_data == ["a.pkl.gz"]
    assert args.file_weight_map == {}


def test_a_worker_without_a_step_expects_no_file_and_is_never_finished(tmp_path, capsys):
    class Args:
        artifact_path = str(tmp_path / "gmm_artifacts.npz")
        step = None
        worker_id = 3
        n_clusters = [2]

    assert _worker_has_finished(Args) is False
    assert "Worker 3 Step None: 0 file(s) missing, running. (e.g. ?)" in capsys.readouterr().out


# --- the master driver: options, resuming, restarting, failures (characterization of run_master) ------------------------

WEIGHT = 2.0


def _run_build(folder: Path, model_dir: Path, *extra: str) -> str:
    """Run ``grace_uq build`` in ``folder`` and return what it printed.

    Logging is disabled meanwhile: the live-log handler of pytest swaps ``sys.stdout`` back to its own capture on the first
    record, which would discard the redirect.
    """
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.chdir(folder)
        logging.disable(logging.CRITICAL)
        try:
            assert build_main(common_args(model_dir, *extra)) == 0
        finally:
            logging.disable(logging.NOTSET)
    return out.getvalue()


def _make_importable(patch: pytest.MonkeyPatch, folder: Path) -> None:
    """Let the master (``sys.path``) and its worker subprocesses (``PYTHONPATH``) import the filter module of ``folder``."""
    patch.syspath_prepend(str(folder))
    patch.setenv("PYTHONPATH", str(folder) + os.pathsep + os.environ.get("PYTHONPATH", ""))


@pytest.fixture(scope="module")
def option_run(model_dir, training_frame, tmp_path_factory):
    """One build with every optional stage on: several k (elbow), a weighted group, a filter, a fraction, a percentile, the export."""
    folder = tmp_path_factory.mktemp("uq_options")
    data = folder / "train.pkl.gz"
    training_frame.to_pickle(data)
    (folder / "keep_all_filter.py").write_text("def keep(atoms):\n    return len(atoms) >= 1\n")
    args = (
        "--n-clusters", "2", "3",
        "--train-data-weighted", str(WEIGHT), str(data),
        "--filter-fn", "keep_all_filter:keep",
        "--frac", "1.0",
        "--threshold-percentile", "90",
        "--verbose",
    )  # fmt: skip
    with pytest.MonkeyPatch.context() as patch:
        _make_importable(patch, folder)
        stdout = _run_build(folder, model_dir, *args)
    return {"folder": folder, "args": args, "stdout": stdout, "artifact": folder / "gmm_artifacts.npz"}


def test_several_candidate_k_select_one_by_the_elbow_and_write_the_report(option_run):
    out = option_run["stdout"]
    assert "Per-element elbow results:" in out
    assert "Global optimal k = " in out
    assert (option_run["folder"] / "gmm_artifacts_elbow.txt").is_file()
    with np.load(option_run["artifact"]) as artifact:
        assert artifact["counts_0"].ndim == 1
        assert artifact["counts_0"].shape[0] in (2, 3)


def test_the_weight_of_a_group_scales_the_effective_counts_of_each_cluster(option_run, training_frame):
    """Every atom of the single weighted group counts WEIGHT times: effective count = WEIGHT x raw count, summing to WEIGHT x atoms."""
    expected = atoms_per_element(training_frame)
    with np.load(option_run["artifact"]) as artifact:
        for index, symbol in enumerate(ELEMENTS):
            counts = artifact[f"counts_{index}"]
            effective = artifact[f"{uqc.EFFECTIVE_COUNT}_{index}"]
            assert counts.sum() == expected[symbol]
            np.testing.assert_allclose(effective, WEIGHT * counts)


def test_the_unweighted_build_stores_effective_counts_equal_to_the_raw_counts(master_run):
    with np.load(master_run["artifact"]) as artifact:
        for index in range(len(ELEMENTS)):
            np.testing.assert_array_equal(
                artifact[f"{uqc.EFFECTIVE_COUNT}_{index}"], artifact[f"counts_{index}"].astype(np.float64)
            )


def test_the_percentile_estimator_is_announced_and_recorded_in_the_artifact(option_run):
    assert "thresholds: 90-th percentile of the training sigma distribution" in option_run["stdout"]
    with np.load(option_run["artifact"]) as artifact:
        assert str(artifact["threshold_mode"]) == "percentile_90"


def test_a_second_run_in_the_same_folder_skips_every_finished_stage(option_run, model_dir):
    before = {name: np.load(option_run["artifact"])[name] for name in ("counts_0", "interp_thresholds")}
    with pytest.MonkeyPatch.context() as patch:
        _make_importable(patch, option_run["folder"])
        out = _run_build(option_run["folder"], model_dir, *option_run["args"])
    for stage in ("Skipping Step 1.", "Skipping Step 2.", "Skipping Step 3.", "Skipping export."):
        assert stage in out
    assert "--threshold-percentile was given but step 3 was skipped" in out
    assert "Spawning" not in out
    with np.load(option_run["artifact"]) as artifact:
        for name, value in before.items():
            np.testing.assert_array_equal(artifact[name], value)


def test_restart_rebuilds_everything_with_the_same_result(option_run, model_dir):
    with np.load(option_run["artifact"]) as artifact:
        before = {name: artifact[name] for name in ("counts_0", "counts_3")}
    with pytest.MonkeyPatch.context() as patch:
        _make_importable(patch, option_run["folder"])
        out = _run_build(option_run["folder"], model_dir, *option_run["args"], "--restart")
    assert "Restarting: cleaning up old checkpoint files..." in out
    assert "Skipping" not in out
    assert out.count("Spawning 1 workers...") == 3
    with np.load(option_run["artifact"]) as artifact:
        for name, value in before.items():
            np.testing.assert_array_equal(artifact[name], value)


@pytest.fixture
def staged_build(option_run, model_dir, training_frame, tmp_path):
    """A folder with the data and the finished step files of ``option_run`` copied in, for the failure tests below."""

    def stage(*files: str) -> Path:
        training_frame.to_pickle(tmp_path / "train.pkl.gz")
        for name in files:
            shutil.copy(option_run["folder"] / name, tmp_path / name)
        return tmp_path

    return stage


@pytest.mark.parametrize(
    ("step", "present"),
    [(1, ()), (2, (".step1_final.npz",)), (3, (".step1_final.npz", ".step2_final.npz"))],
)
def test_workers_that_report_success_without_writing_their_file_are_an_error(
    staged_build, model_dir, monkeypatch, step, present
):
    """The master checks the checkpoint of every worker; a missing one names the step."""
    folder = staged_build(*present)
    monkeypatch.setattr(master, "spawn_and_monitor", lambda *a, **k: [])
    monkeypatch.chdir(folder)
    with pytest.raises(RuntimeError, match=rf"Worker file '.*' missing after Step {step} completed"):
        build_main(common_args(model_dir, "--n-clusters", "2", "--no-export", "--train-data", str(folder / "train.pkl.gz")))


def test_a_failed_worker_stops_the_pipeline_with_the_failed_ids(staged_build, model_dir, monkeypatch):
    folder = staged_build()
    monkeypatch.setattr(master, "spawn_and_monitor", lambda *a, **k: [0])
    monkeypatch.chdir(folder)
    with pytest.raises(RuntimeError, match=r"Step 1: worker\(s\) \[0\] failed"):
        build_main(common_args(model_dir, "--n-clusters", "2", "--no-export", "--train-data", str(folder / "train.pkl.gz")))


def test_workers_whose_files_are_all_present_are_not_spawned_again(staged_build, model_dir, option_run, monkeypatch, capsys):
    """Resuming after the master died between the workers and the merge: the worker files of the step are found on disk."""
    folder = staged_build()
    for k in (2, 3):
        (folder / f".step1_w0_k{k}.npz").write_bytes(b"")
    monkeypatch.setattr(
        master, "spawn_and_monitor", lambda *a, **k: pytest.fail("no worker may be spawned")
    )
    monkeypatch.chdir(folder)
    with pytest.raises(EOFError):  # the stub files are not npz: the merge that follows fails, after the check under test
        build_main(common_args(model_dir, "--n-clusters", "2", "3", "--no-export", "--train-data", str(folder / "train.pkl.gz")))
    assert "All workers for Step 1 already finished." in capsys.readouterr().out


def test_the_results_of_several_workers_are_merged_into_the_counts_of_the_whole_data(
    staged_build, model_dir, training_frame, monkeypatch
):
    """Two workers each see half of the atoms: the merged counts still add up to the atoms of each element."""
    folder = staged_build(".step1_final.npz")
    monkeypatch.chdir(folder)
    build_main(
        common_args(
            model_dir, "--n-clusters", "2", "3", "--n-workers", "2", "--no-export",
            "--train-data", str(folder / "train.pkl.gz"),
        )
    )  # fmt: skip
    expected = atoms_per_element(training_frame)
    with np.load(folder / "gmm_artifacts.npz") as artifact:
        for index, symbol in enumerate(ELEMENTS):
            assert artifact[f"counts_{index}"].sum() == expected[symbol]
            np.testing.assert_array_equal(
                artifact[f"{uqc.EFFECTIVE_COUNT}_{index}"], artifact[f"counts_{index}"].astype(np.float64)
            )


def test_restart_removes_the_old_step_files_and_survives_one_that_cannot_be_removed(
    staged_build, model_dir, monkeypatch, capsys
):
    """``--restart`` is best effort on the old checkpoints: a file that cannot be removed is left and the run goes on."""
    folder = staged_build(".step1_final.npz", ".step2_final.npz")
    (folder / ".step9_stuck.npz").write_bytes(b"")
    real_remove = os.remove

    def remove(path, *args, **kwargs):
        if str(path).endswith(".step9_stuck.npz"):
            raise PermissionError(path)
        real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", remove)
    monkeypatch.setattr(master, "spawn_and_monitor", lambda *a, **k: [0])
    monkeypatch.chdir(folder)
    with pytest.raises(RuntimeError, match="Step 1"):
        build_main(
            common_args(
                model_dir, "--n-clusters", "2", "--no-export", "--restart", "--threads-per-worker", "2",
                "--train-data", str(folder / "train.pkl.gz"),
            )
        )  # fmt: skip
    assert "Restarting: cleaning up old checkpoint files..." in capsys.readouterr().out
    assert not (folder / ".step1_final.npz").exists()
    assert not (folder / ".step2_final.npz").exists()
    assert (folder / ".step9_stuck.npz").exists()


def test_groups_of_different_weights_and_an_element_without_data(model_dir, training_frame, tmp_path, caplog):
    """Nine frames without tungsten in two weighted groups: W is absent from the artifact, and the effective counts of Mo, Nb and Ta are
    the weighted sums of the atoms of each group (computed here from the frames)."""
    light = [1, 2, 3, 6]
    heavy = [7, 8, 9, 10, 11]
    weights = {"light.pkl.gz": 1.0, "heavy.pkl.gz": 5.0}
    expected = dict.fromkeys(ELEMENTS, 0.0)
    for (name, rows), weight in zip((("light.pkl.gz", light), ("heavy.pkl.gz", heavy)), weights.values(), strict=True):
        training_frame.iloc[rows].reset_index(drop=True).to_pickle(tmp_path / name)
        for atoms in training_frame.iloc[rows]["ase_atoms"]:
            for symbol in atoms.get_chemical_symbols():
                expected[symbol] += weight
    assert expected["W"] == 0.0
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        build_main(
            common_args(
                model_dir, "--n-clusters", "2", "--no-export",
                "--train-data-weighted", "1.0", str(tmp_path / "light.pkl.gz"),
                "--train-data-weighted", "5.0", str(tmp_path / "heavy.pkl.gz"),
            )
        )  # fmt: skip
    assert "element W(3) has no training atoms" in caplog.text
    with np.load(tmp_path / "gmm_artifacts.npz") as artifact:
        for index, symbol in enumerate(ELEMENTS[:3]):
            effective = artifact[f"{uqc.EFFECTIVE_COUNT}_{index}"]
            assert effective.sum() == pytest.approx(expected[symbol])
        assert "counts_3" not in artifact.files
