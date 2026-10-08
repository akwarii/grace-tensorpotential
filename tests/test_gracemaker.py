"""Characterization tests of ``tensorpotential.cli.gracemaker.main``.

``main`` is driven the way the ``gracemaker`` script drives it (an ``argv`` list and an ``input.yaml``)
on random-weight LINEAR and FS models and a few dozen small structures of ``tests/data``; every run
writes into ``tmp_path`` only. The two boundaries that are replaced are named where they are used: the
download of a foundation checkpoint, and the end of a run by a keyboard interrupt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ase import Atoms  # noqa: E402

from tensorpotential.calculator.asecalculator import TPCalculator  # noqa: E402
from tensorpotential.cli import gracemaker  # noqa: E402
from tensorpotential.cli.gracemaker import main  # noqa: E402
from tensorpotential.instructions.base import load_instructions  # noqa: E402
from tensorpotential.metadata_utils import read_model_metadata  # noqa: E402
from tensorpotential.tensorpot import TensorPotential  # noqa: E402
from tensorpotential.tpmodel import TPModel  # noqa: E402
from tensorpotential.utils import get_dtype_by_name  # noqa: E402
from tests.tolerances import COVARIANCE_F64  # noqa: E402
from tensorpotential.utils import load_metrics  # noqa: E402

TESTS = Path(__file__).resolve().parent
SEED = 42
N_TRAIN = 12
N_TEST = 6


def _smallest(path: Path, n: int, symbols: set[str] | None = None) -> pd.DataFrame:
    """The ``n`` structures with fewest atoms of a dataset, optionally only those made of ``symbols``."""
    df = pd.read_pickle(path)
    if symbols is not None:
        keep = df["ase_atoms"].map(lambda a: set(a.get_chemical_symbols()) <= symbols)
        df = df[keep]
    order = df["ase_atoms"].map(len).sort_values(kind="stable").index[:n]
    return df.loc[order].reset_index(drop=True)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Small train and test sets: all four elements (``train``/``test``) and Mo and W only (``mow_*``)."""
    folder = tmp_path_factory.mktemp("gracemaker_data")
    src = TESTS / "data"
    _smallest(src / "MoNbTaW_train50.pkl.gz", N_TRAIN).to_pickle(
        folder / "train.pkl.gz"
    )
    _smallest(src / "MoNbTaW_test50.pkl.gz", N_TEST).to_pickle(folder / "test.pkl.gz")
    mow = {"Mo", "W"}
    _smallest(src / "MoNbTaW_train50.pkl.gz", N_TRAIN, mow).to_pickle(
        folder / "mow_train.pkl.gz"
    )
    _smallest(src / "MoNbTaW_test50.pkl.gz", N_TEST, mow).to_pickle(
        folder / "mow_test.pkl.gz"
    )
    return folder


def _base_config(train: str = "train", test: str = "test") -> dict:
    """The LINEAR input of ``tests/MoNbTaW-LINEAR`` on the small datasets, two epochs of Adam."""
    with (TESTS / "MoNbTaW-LINEAR" / "input.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg["data"] = {
        "filename": f"../data/{train}.pkl.gz",
        "test_filename": f"../data/{test}.pkl.gz",
    }
    cfg["fit"]["batch_size"] = 4
    cfg["fit"]["test_batch_size"] = N_TEST
    cfg["fit"]["maxiter"] = 2
    cfg["fit"]["checkpoint_freq"] = 1
    cfg["fit"]["learning_rate_reduction"] = {"patience": 5, "factor": 0.8, "min": 1e-3}
    # a ``learning_rate`` entry inside ``loss.switch`` is not a loss component: it breaks a restart after the switch
    cfg["fit"]["loss"]["switch"].pop("learning_rate")
    return cfg


def _merge(base: dict, overrides: dict) -> dict:
    """``base`` with ``overrides`` merged in recursively; a value of ``None`` deletes the key."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@contextmanager
def _own_log_handlers():
    """Close and remove the root-logger file handlers that ``main`` adds (it never removes them itself)."""
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


@pytest.fixture(autouse=True)
def _restore_log_handlers():
    with _own_log_handlers():
        yield


class Workspace:
    """A scratch run directory next to a ``data`` folder; inputs refer to ``../data/<file>``."""

    def __init__(self, root: Path, data: Path):
        (root / "data").symlink_to(data, target_is_directory=True)
        self.run = root / "run"
        self.run.mkdir()

    def write_input(
        self,
        name: str = "input.yaml",
        train: str = "train",
        test: str = "test",
        **overrides,
    ) -> str:
        cfg = _merge(_base_config(train, test), overrides)
        (self.run / name).write_text(yaml.safe_dump(cfg))
        return name

    @property
    def seed_dir(self) -> Path:
        return self.run / "seed" / str(SEED)

    def metrics(self, kind: str = "train") -> pd.DataFrame:
        return load_metrics(str(self.seed_dir / f"{kind}_metrics.yaml"))


@pytest.fixture
def ws(tmp_path, monkeypatch, data_dir) -> Workspace:
    workspace = Workspace(tmp_path, data_dir)
    monkeypatch.chdir(workspace.run)
    return workspace


def _fingerprint(folder: Path) -> str:
    """Hash of the names and contents of the files below ``folder``."""
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(folder)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def trained(tmp_path_factory, data_dir):
    """A finished two-epoch LINEAR run (float64, Adam); read-only: tests copy what they edit."""
    root = tmp_path_factory.mktemp("gracemaker_trained")
    workspace = Workspace(root, data_dir)
    name = workspace.write_input()
    with pytest.MonkeyPatch.context() as mp, _own_log_handlers():
        mp.chdir(workspace.run)
        main([name])
    before = _fingerprint(workspace.run)
    yield workspace
    assert _fingerprint(workspace.run) == before, (
        "a test modified the shared trained run"
    )


def _clone_seed(trained: Workspace, ws: Workspace) -> None:
    """Copy the checkpoints and model.yaml of the shared run into the scratch run."""
    shutil.copytree(trained.run / "seed", ws.run / "seed")


class TestCheckModel:
    def test_check_model_exits_before_any_training(self, ws, caplog):
        name = ws.write_input()
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
            main([name, "--check-model"])
        assert exc.value.code == 0
        assert "Model is constructed" in caplog.text
        assert not (ws.seed_dir / "train_metrics.yaml").exists()
        assert not (ws.seed_dir / "model.yaml").exists()


FROM_FILE = {"potential": {"preset": None, "param_dtype": None}}
"""Overrides for a run whose model comes from a ``model.yaml`` file: nothing left in ``potential`` to build one from."""


def _weights(
    seed_dir: Path, checkpoint: str = "checkpoints/checkpoint.best_test_loss"
) -> dict[str, np.ndarray]:
    """All variables of a run, read back through ``TensorPotential`` (not through ``main``)."""
    model_yaml = str(seed_dir / "model.yaml")
    dtype = get_dtype_by_name(read_model_metadata(model_yaml)["param_dtype"])
    tp = TensorPotential(potential=load_instructions(model_yaml), param_dtype=dtype)
    tp.load_checkpoint(checkpoint_name=str(seed_dir / checkpoint), model_only=True)
    return {v.name: v.numpy() for v in tp.model.variables if v.dtype.is_floating}


DIMER = Atoms("MoNb", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]], pbc=False)


def _checkpoint_energy(
    seed_dir: Path,
    checkpoint: str = "checkpoints/checkpoint.best_test_loss",
    atoms: Atoms = DIMER,
) -> float:
    """Energy of ``atoms`` from the model.yaml and checkpoint of a run, loaded without ``main``."""
    model_yaml = str(seed_dir / "model.yaml")
    dtype = get_dtype_by_name(read_model_metadata(model_yaml)["param_dtype"])
    tp = TensorPotential(potential=load_instructions(model_yaml), param_dtype=dtype)
    tp.load_checkpoint(checkpoint_name=str(seed_dir / checkpoint), model_only=True)
    return float(TPCalculator(tp.model).get_potential_energy(atoms.copy()))


def _assert_exported_energy_matches(
    exported_seed_dir: Path,
    source_seed_dir: Path,
    checkpoint: str = "checkpoints/checkpoint.best_test_loss",
) -> None:
    """The ``saved_model`` that ``--save-model`` wrote predicts what the source checkpoint predicts."""
    exported = TPCalculator(
        model=str(exported_seed_dir / "saved_model")
    ).get_potential_energy(DIMER.copy())
    expected = _checkpoint_energy(source_seed_dir, checkpoint)
    assert expected != 0.0
    assert exported == pytest.approx(
        expected, rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
    )


@pytest.fixture
def interrupted_training(monkeypatch):
    """Replace both optimisation loops by a recorder that ends the run as a keyboard interrupt does.

    The real loops are exercised by the runs of the other tests (and ``tests/test_integration_test.py``);
    here everything ``main`` does before the loop (dataset and ``maxiter`` resolution, model construction,
    checkpoint loading) is checked without paying for an optimisation. Returns the list of recorded calls.
    """
    calls: list[tuple[str, dict]] = []

    def _recorder(name):
        def _stop(tp, **kwargs):
            calls.append((name, kwargs))
            raise KeyboardInterrupt

        return _stop

    monkeypatch.setattr(gracemaker, "train_adam", _recorder("Adam"))
    monkeypatch.setattr(gracemaker, "train_bfgs", _recorder("BFGS"))
    return calls


def _run_to_training(argv: list[str], calls: list) -> dict:
    """Run ``main`` until the (replaced) optimisation loop; return the ``fit_config`` it was given."""
    with pytest.raises(SystemExit) as exc:
        gracemaker.main(argv)
    assert exc.value.code == 0
    assert len(calls) == 1, "the optimisation loop was not reached exactly once"
    return calls[0][1]["fit_config"]


class TestRestart:
    def test_restart_latest_continues_the_counters(self, ws, trained):
        _clone_seed(trained, ws)
        name = ws.write_input(fit={"maxiter": 3})
        main([name, "-rl"])
        train = ws.metrics("train")
        assert train["epoch"].tolist() == [1, 2, 3]
        assert train["step"].tolist() == [3, 6, 9]  # 12 structures in batches of 4

    def test_reset_epoch_and_step_restarts_the_counters(self, ws, trained, caplog):
        _clone_seed(trained, ws)
        name = ws.write_input()
        with caplog.at_level(logging.INFO):
            main([name, "-rl", "--no-jit", "--eager", "--reset-epoch-and-step"])
        # metrics of the first run, then of the restarted run, counted from one again
        assert ws.metrics("train")["epoch"].tolist() == [1, 2, 1, 2]
        assert ws.metrics("train")["step"].tolist() == [3, 6, 3, 6]
        assert "Reset epochs (2 -> 0) and steps (6 -> 0) counters" in caplog.text
        assert "--no-jit option is provided" in caplog.text
        assert "Eager execution" in caplog.text

    def test_explicit_model_file_and_checkpoint_name_are_exported(self, ws, trained):
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(**FROM_FILE)
        # the checkpoint is named with its ``.index`` suffix, which main strips
        ckpt = str(prev / "checkpoints" / "checkpoint.best_test_loss.index")
        with pytest.raises(SystemExit) as exc:
            main([name, "-p", str(prev / "model.yaml"), "-cn", ckpt, "--save-model"])
        assert exc.value.code == 0
        _assert_exported_energy_matches(ws.seed_dir, trained.seed_dir)

    def test_model_file_with_other_elements_is_rejected(self, ws, trained):
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(
            **_merge(FROM_FILE, {"potential": {"elements": ["Mo", "W"]}})
        )
        with pytest.raises(RuntimeError, match="differs from that read from model"):
            main([name, "-p", str(prev / "model.yaml")])


class TestParamDtype:
    def test_dtype_comes_from_the_model_file_metadata(
        self, ws, trained, interrupted_training
    ):
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(**FROM_FILE)
        _run_to_training([name, "-p", str(prev / "model.yaml")], interrupted_training)
        assert (
            read_model_metadata(str(ws.seed_dir / "model.yaml"))["param_dtype"]
            == "float64"
        )

    def test_model_file_without_metadata_is_float64(
        self, ws, trained, interrupted_training
    ):
        with (trained.seed_dir / "model.yaml").open() as f:
            old_format = list(
                yaml.safe_load(f)["instructions"].values()
            )  # no ``metadata`` section
        (ws.run / "old_model.yaml").write_text(yaml.safe_dump(old_format))
        name = ws.write_input(**FROM_FILE)
        _run_to_training([name, "-p", "old_model.yaml"], interrupted_training)
        assert (
            read_model_metadata(str(ws.seed_dir / "model.yaml")).get("param_dtype")
            == "float64"
        )


class TestSaveModel:
    def test_save_model_exports_the_saved_model_without_training(
        self, ws, trained, caplog
    ):
        _clone_seed(trained, ws)
        name = ws.write_input()
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
            main([name, "-r", "--save-model"])
        assert exc.value.code == 0
        assert (ws.seed_dir / "saved_model" / "saved_model.pb").exists()
        assert not (ws.seed_dir / "saved_model.yaml").exists()
        _assert_exported_energy_matches(ws.seed_dir, trained.seed_dir)
        assert ws.metrics("train")["epoch"].tolist() == [1, 2]  # nothing was trained
        assert (
            "Global TRAIN batch size" not in caplog.text
        )  # nor was a dataset prepared

    def test_save_model_without_a_previous_fit_fails(self, ws):
        # no dtype in the input, so main looks for the dtype in the (absent) model.yaml of the run first
        name = ws.write_input(potential={"param_dtype": None})
        with pytest.raises(FileNotFoundError):
            main([name, "-s"])


class TestMaxiterResolution:
    """``fit.maxiter`` is an epoch count; main turns ``target_total_updates`` or ``auto`` into one."""

    @pytest.mark.parametrize(
        ("fit", "expected"),
        [
            pytest.param(
                {"target_total_updates": 45}, 15, id="adam-target"
            ),  # ceil(45 updates / 3 batches)
            pytest.param(
                {
                    "optimizer": "L-BFGS-B",
                    "opt_params": None,
                    "target_total_updates": 7,
                },
                10,
                id="bfgs-target-floor",
            ),
            pytest.param(
                {"maxiter": None}, 5000, id="adam-auto"
            ),  # 50000 / 3 batches, capped at 5000
            pytest.param(
                {"maxiter": "auto", "optimizer": "L-BFGS-B", "opt_params": None},
                500,
                id="lbfgsb-auto",
            ),
        ],
    )
    def test_epochs_handed_to_the_optimiser(
        self, ws, interrupted_training, fit, expected
    ):
        name = ws.write_input(fit=fit)
        fit_config = _run_to_training([name], interrupted_training)
        assert fit_config["maxiter"] == expected

    def test_unknown_optimizer_falls_back_to_a_default_target_then_fails(
        self, ws, interrupted_training, caplog
    ):
        name = ws.write_input(fit={"optimizer": "SGD", "maxiter": None})
        with (
            caplog.at_level(logging.INFO),
            pytest.raises(ValueError, match="Unknown optimizer: SGD"),
        ):
            main([name])
        assert "maxiter auto (SGD, scratch): target=500" in caplog.text
        assert interrupted_training == []

    @pytest.mark.parametrize(
        ("after_iter", "expected"),
        [
            pytest.param(0.5, 2, id="fraction"),  # round(0.5 * 4 epochs)
            pytest.param("auto", 3, id="auto"),  # round(0.75 * 4)
        ],
    )
    def test_switch_after_iter_becomes_an_epoch(
        self, ws, interrupted_training, after_iter, expected
    ):
        name = ws.write_input(
            fit={"maxiter": 4, "loss": {"switch": {"after_iter": after_iter}}}
        )
        fit_config = _run_to_training([name], interrupted_training)
        assert fit_config["loss"]["switch"]["after_iter"] == expected


class TestEntryPoints:
    def test_no_argument_reads_input_yaml_of_the_working_directory(
        self, ws, interrupted_training
    ):
        # also the defaults of a fresh fit: float32 parameters, and no final model after an interrupt
        ws.write_input(
            "input.yaml", fit={"maxiter": 3}, potential={"param_dtype": None}
        )
        assert _run_to_training(None, interrupted_training)["maxiter"] == 3
        meta = read_model_metadata(str(ws.seed_dir / "model.yaml"))
        assert meta["param_dtype"] == "float32"
        assert not (ws.seed_dir / "final_model").exists()

    def test_strategy_given_by_the_caller_is_used(
        self, ws, interrupted_training, caplog
    ):
        name = ws.write_input()
        strategy = tf.distribute.get_strategy()
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit):
            main([name], strategy=strategy, strategy_desc="given by the test")
        assert "distributed strategy is already initialized" in caplog.text
        assert "Data distribution strategy: given by the test" in caplog.text
        assert interrupted_training[0][1]["strategy"] is strategy

    def test_mirrored_strategy_is_created_on_request(
        self, ws, interrupted_training, caplog
    ):
        name = ws.write_input()
        with caplog.at_level(logging.INFO):
            _run_to_training([name, "--multigpu"], interrupted_training)
        assert "Data distribution strategy: Single host/multi GPU" in caplog.text
        assert isinstance(
            interrupted_training[0][1]["strategy"], tf.distribute.MirroredStrategy
        )

    def test_template_option_starts_the_input_wizard(self, ws, monkeypatch):
        calls = []
        # replaces the interactive dialog on stdin that writes a template ``input.yaml``
        monkeypatch.setattr(
            gracemaker, "generate_template_input", lambda: calls.append(1)
        )
        name = ws.write_input()
        with pytest.raises(SystemExit):
            main([name, "--template", "--check-model"])
        assert calls == [1]


class TestTrainableVariables:
    def test_only_the_named_variables_are_trained(self, ws, trained):
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(
            **_merge(FROM_FILE, {"fit": {"trainable_variable_names": ["E/reducing_"]}})
        )
        ckpt = str(prev / "checkpoints" / "checkpoint.best_test_loss")
        main([
            name,
            "-p",
            str(prev / "model.yaml"),
            "-cn",
            ckpt,
            "--reset-epoch-and-step",
        ])
        before, after = _weights(prev), _weights(ws.seed_dir)
        assert before.keys() == after.keys()
        trained_names = {n for n in before if n.startswith("E/reducing_")}
        assert trained_names, "the pattern matched no variable"
        for n in before:
            changed = not np.array_equal(before[n], after[n])
            assert changed == (n in trained_names), n


@pytest.fixture(scope="module")
def foundation(trained, tmp_path_factory) -> Iterator[Path]:
    """Stand-in foundation checkpoint: the model.yaml and weights of the trained run as ``model.yaml`` and ``checkpoint``."""
    folder = tmp_path_factory.mktemp("gracemaker_foundation")
    shutil.copy(trained.seed_dir / "model.yaml", folder / "model.yaml")
    for src in (trained.seed_dir / "checkpoints").glob("checkpoint.best_test_loss.*"):
        shutil.copy(src, folder / src.name.replace(".best_test_loss", ""))
    before = _fingerprint(folder)
    yield folder
    assert _fingerprint(folder) == before, (
        "a test modified the shared foundation checkpoint"
    )


@pytest.fixture
def downloads(monkeypatch, foundation) -> list[str]:
    """Answer the download of a foundation checkpoint with the local stand-in; return the names asked for."""
    asked: list[str] = []

    def _get(model: str) -> str:
        asked.append(model)
        return str(foundation)

    # replaces ``get_or_download_checkpoint``, which would fetch a multi-hundred-megabyte archive from the network
    monkeypatch.setattr(gracemaker, "get_or_download_checkpoint", _get)
    return asked


FINETUNE = {
    "potential": {
        "preset": None,
        "param_dtype": None,
        "finetune_foundation_model": "stand-in-fm",
    }
}


class TestFoundationFinetune:
    def test_model_and_weights_come_from_the_foundation_checkpoint(
        self, ws, foundation, downloads
    ):
        name = ws.write_input(**FINETUNE)
        with pytest.raises(SystemExit):
            main([name, "--save-model"])
        assert downloads == ["stand-in-fm"]
        _assert_exported_energy_matches(ws.seed_dir, foundation, "checkpoint")

    def test_restart_does_not_download_again(
        self, ws, trained, downloads, interrupted_training
    ):
        _clone_seed(trained, ws)
        name = ws.write_input(potential={"finetune_foundation_model": "stand-in-fm"})
        _run_to_training([name, "-rl"], interrupted_training)
        assert downloads == []

    def test_finetune_target_and_automatic_shift(
        self, ws, downloads, foundation, interrupted_training
    ):
        # finetune target of L-BFGS-B is 100 iterations; ``shift: auto`` aligns the model with the data
        fit = {"maxiter": None, "optimizer": "L-BFGS-B", "opt_params": None}
        name = ws.write_input(
            **_merge(FINETUNE, {"fit": fit, "potential": {"shift": "auto"}})
        )
        assert _run_to_training([name], interrupted_training)["maxiter"] == 100

        def shifts(path: Path):
            with path.open() as f:
                return yaml.safe_load(f)["instructions"]["ConstantScaleShiftTarget"][
                    "atomic_shift_map"
                ]

        assert shifts(foundation / "model.yaml") is None
        assert shifts(ws.seed_dir / "model.yaml")  # a shift per element was injected


class TestReduceElements:
    def test_reduced_model_predicts_like_the_full_one_on_the_kept_elements(
        self, ws, foundation, interrupted_training
    ):
        potential = {
            "filename": str(foundation / "model.yaml"),
            "checkpoint_name": str(foundation / "checkpoint"),
            "reduce_elements": True,
            "elements": ["Mo", "W"],
        }
        name = ws.write_input(
            train="mow_train",
            test="mow_test",
            **_merge(FROM_FILE, {"potential": potential}),
        )
        _run_to_training([name], interrupted_training)
        with (ws.seed_dir / "model.yaml").open() as f:
            reduced = yaml.safe_load(f)["instructions"]
        assert reduced["Z"]["element_map"] == {"Mo": 0, "W": 1}
        mo_w = Atoms("MoW", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]], pbc=False)
        full_energy = _checkpoint_energy(foundation, "checkpoint", mo_w)
        reduced_energy = _checkpoint_energy(ws.seed_dir, "checkpoints/checkpoint", mo_w)
        assert full_energy != 0.0
        assert reduced_energy == pytest.approx(
            full_energy, rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
        )


@pytest.fixture(scope="module")
def trained_fs(tmp_path_factory, data_dir):
    """A finished two-epoch FS run (float64) that also exported ``FS_model.yaml``; read-only."""
    root = tmp_path_factory.mktemp("gracemaker_trained_fs")
    workspace = Workspace(root, data_dir)
    name = workspace.write_input(potential={"preset": "FS", "scale": True})
    with pytest.MonkeyPatch.context() as mp, _own_log_handlers():
        mp.chdir(workspace.run)
        main([name, "--save--fs"])
    before = _fingerprint(workspace.run)
    yield workspace
    assert _fingerprint(workspace.run) == before, "a test modified the shared FS run"


class TestFsExport:
    def test_fit_exports_the_fs_model_next_to_the_final_model(self, trained_fs):
        assert (trained_fs.seed_dir / "final_model" / "saved_model.pb").exists()
        with (trained_fs.seed_dir / "FS_model.yaml").open() as f:
            exported = yaml.safe_load(f)
        assert exported, "FS_model.yaml is empty"

    def test_save_model_with_save_fs_exports_both_formats_from_the_best_checkpoint(
        self, ws, trained_fs, caplog
    ):
        _clone_seed(trained_fs, ws)
        name = ws.write_input(potential={"preset": "FS", "scale": True})
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
            main([name, "-r", "-s", "--save--fs"])
        assert exc.value.code == 0
        assert "Exporting to `saved_model.yaml` done" in caplog.text
        assert (ws.seed_dir / "saved_model" / "saved_model.pb").exists()
        # both exports read the same best-test checkpoint, so the two FS files hold the same model
        with (ws.seed_dir / "saved_model.yaml").open() as f:
            from_save = yaml.safe_load(f)
        with (trained_fs.seed_dir / "FS_model.yaml").open() as f:
            from_fit = yaml.safe_load(f)
        assert from_save == from_fit


LORA = {"all": {"rank": 2, "alpha": 1}}
"""``potential::lora``: rank-2 updates on every instruction that supports LoRA."""


def _lora_input(
    ws: Workspace, from_file: bool = True, fit: dict | None = None, **potential
) -> str:
    """An input with ``potential::lora``; ``from_file`` leaves no preset (the model comes from ``-p``)."""
    overrides = {"potential": {"lora": LORA, **potential}, "fit": fit or {}}
    return ws.write_input(**_merge(FROM_FILE if from_file else {}, overrides))


def _run_lora(ws: Workspace, prev: Path, *extra: str) -> None:
    """A LoRA fit started from the model.yaml and best checkpoint of ``prev`` (counters reset)."""
    ckpt = str(prev / "checkpoints" / "checkpoint.best_test_loss")
    main([
        _lora_input(ws),
        "-p",
        str(prev / "model.yaml"),
        "-cn",
        ckpt,
        "--reset-epoch-and-step",
        *extra,
    ])


@pytest.fixture(scope="module")
def lora_run(tmp_path_factory, data_dir, trained):
    """A finished LoRA run on top of ``trained`` (``prev`` holds a copy of the base run); read-only."""
    root = tmp_path_factory.mktemp("gracemaker_lora")
    workspace = Workspace(root, data_dir)
    prev = workspace.run / "prev"
    shutil.copytree(trained.seed_dir, prev)
    with pytest.MonkeyPatch.context() as mp, _own_log_handlers():
        mp.chdir(workspace.run)
        _run_lora(workspace, prev)
    before = _fingerprint(workspace.run)
    yield workspace
    assert _fingerprint(workspace.run) == before, "a test modified the shared LoRA run"


@pytest.fixture
def training_probe(monkeypatch):
    """Replace the Adam loop by a recorder of the ``TensorPotential`` it is given, then end the run.

    Like ``interrupted_training`` but keeps ``tp``, so the model that ``main`` hands to the optimiser
    (restored, activated and reduced as the input asks) can be inspected. Returns the recorded list.
    """
    seen: list[TensorPotential] = []

    def _stop(tp, **kwargs):
        seen.append(tp)
        raise KeyboardInterrupt

    monkeypatch.setattr(gracemaker, "train_adam", _stop)
    return seen


def _run_to_probe(argv: list[str], seen: list) -> TensorPotential:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    assert len(seen) == 1, "the optimisation loop was not reached exactly once"
    return seen[0]


def _lora_names(weights: dict[str, np.ndarray]) -> list[str]:
    return sorted(n for n in weights if "/LORA/" in n)


def _model_is_lora(seed_dir: Path) -> bool:
    return TPModel(load_instructions(str(seed_dir / "model.yaml"))).is_lora_enabled()


class TestLoraFit:
    def test_only_the_update_tensors_are_trained(self, lora_run, trained):
        base, new = _weights(trained.seed_dir), _weights(lora_run.seed_dir)
        update = _lora_names(new)
        assert update, "no LoRA tensor was created"
        assert set(new) - set(base) == set(update)
        # weights of instructions that LoRA adapts are frozen; those of instructions without LoRA
        # support (here the reducing tensors) stay trainable and move
        frozen = {n for n in base if n.startswith(("Z/", "A_ChemIndTransf/"))}
        assert len(frozen) == 2
        for n in frozen:
            np.testing.assert_array_equal(new[n], base[n], err_msg=n)
        assert not np.array_equal(new["E/reducing_A:0"], base["E/reducing_A:0"])
        # the update starts at zero (B = 0), so a non-zero B means that it was trained
        assert any(np.abs(new[n]).max() > 0 for n in update if n.endswith("/B:0"))

    def test_the_final_model_equals_the_activated_model_of_the_seed_directory(
        self, lora_run, trained
    ):
        activated = _checkpoint_energy(lora_run.seed_dir)
        assert activated != _checkpoint_energy(trained.seed_dir)  # the update acts
        # the final model is saved reduced: it predicts what the model with the update tensors does
        _assert_exported_energy_matches_final(lora_run.seed_dir, activated)

    def test_model_yaml_and_checkpoints_keep_the_update_tensors(self, lora_run):
        assert _model_is_lora(lora_run.seed_dir)
        for ckpt in ("checkpoint.best_test_loss", "checkpoint"):
            assert _lora_names(_weights(lora_run.seed_dir, f"checkpoints/{ckpt}"))

    def test_the_final_model_holds_no_update_tensors(self, lora_run):
        ckpt = str(lora_run.seed_dir / "final_model" / "variables" / "variables")
        names = {name for name, _ in tf.train.list_variables(ckpt)}
        assert names
        assert not any("LORA" in n for n in names), names

    def test_model_yaml_keeps_the_parameter_dtype(self, lora_run):
        meta = read_model_metadata(str(lora_run.seed_dir / "model.yaml"))
        assert meta["param_dtype"] == "float64"


def _assert_exported_energy_matches_final(seed_dir: Path, expected: float) -> None:
    final = TPCalculator(model=str(seed_dir / "final_model")).get_potential_energy(
        DIMER.copy()
    )
    assert final == pytest.approx(
        expected, rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
    )


class TestLoraRestart:
    def test_a_restart_keeps_the_trained_update_tensors(
        self, ws, lora_run, training_probe, caplog
    ):
        _clone_seed(lora_run, ws)
        name = _lora_input(ws, from_file=False)  # the input still asks for ``lora``
        with caplog.at_level(logging.INFO):
            tp = _run_to_probe([name, "-rl"], training_probe)
        assert "LoRA is already active for Z, kept" in caplog.text
        before = _weights(lora_run.seed_dir, "checkpoints/checkpoint")
        after = {v.name: v.numpy() for v in tp.model.variables if v.dtype.is_floating}
        assert _lora_names(before) == _lora_names(after)
        for n in before:
            np.testing.assert_array_equal(after[n], before[n], err_msg=n)
        assert any(np.abs(after[n]).max() > 0 for n in _lora_names(after))
        assert _model_is_lora(
            ws.seed_dir
        )  # model.yaml was re-saved with the LoRA state

    def test_the_restarted_run_trains_on_and_exports_the_reduced_model(
        self, ws, lora_run
    ):
        _clone_seed(lora_run, ws)
        name = _lora_input(ws, from_file=False, fit={"maxiter": 3})
        main([name, "-rl"])
        assert ws.metrics("train")["epoch"].tolist() == [1, 2, 3]
        expected = _checkpoint_energy(ws.seed_dir)
        _assert_exported_energy_matches_final(ws.seed_dir, expected)
        assert _model_is_lora(ws.seed_dir)


class TestLoraReduction:
    def test_reduce_lora_merges_the_update_into_the_model_file_and_keeps_the_energy(
        self, ws, lora_run, training_probe
    ):
        _clone_seed(lora_run, ws)
        activated = _checkpoint_energy(ws.seed_dir, "checkpoints/checkpoint")
        name = _lora_input(ws, from_file=False, reduce_lora=True)
        tp = _run_to_probe([name, "-rl"], training_probe)
        assert not tp.is_lora_enabled()
        assert not _model_is_lora(ws.seed_dir)
        reduced = float(TPCalculator(tp.model).get_potential_energy(DIMER.copy()))
        assert reduced == pytest.approx(
            activated, rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
        )

    def test_reduce_lora_without_an_active_lora_warns_and_changes_nothing(
        self, ws, trained, training_probe, caplog
    ):
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(**_merge(FROM_FILE, {"potential": {"reduce_lora": True}}))
        with caplog.at_level(logging.WARNING):
            tp = _run_to_probe([name, "-p", str(prev / "model.yaml")], training_probe)
        assert "no LoRA is active" in caplog.text
        assert not tp.is_lora_enabled()
        assert not _model_is_lora(ws.seed_dir)


class TestLoraSaveModel:
    def test_save_model_exports_the_reduced_model_and_leaves_the_seed_directory(
        self, ws, lora_run
    ):
        _clone_seed(lora_run, ws)
        name = _lora_input(ws, from_file=False)
        with pytest.raises(SystemExit) as exc:
            main([name, "-r", "--save-model"])
        assert exc.value.code == 0
        _assert_exported_energy_matches(ws.seed_dir, lora_run.seed_dir)
        ckpt = str(ws.seed_dir / "saved_model" / "variables" / "variables")
        assert not any("LORA" in n for n, _ in tf.train.list_variables(ckpt))
        assert _model_is_lora(
            ws.seed_dir
        )  # the seed directory keeps the update tensors
        assert _lora_names(
            _weights(ws.seed_dir, "checkpoints/checkpoint.best_test_loss")
        )


class TestLoraParamDtype:
    def test_the_model_file_written_after_activation_keeps_float32(
        self, ws, data_dir, tmp_path_factory
    ):
        base = Workspace(tmp_path_factory.mktemp("gracemaker_f32"), data_dir)
        name = base.write_input(potential={"param_dtype": "float32"})
        with pytest.MonkeyPatch.context() as mp, _own_log_handlers():
            mp.chdir(base.run)
            main([name])
        assert (
            read_model_metadata(str(base.seed_dir / "model.yaml"))["param_dtype"]
            == "float32"
        )
        ckpt = str(base.seed_dir / "checkpoints" / "checkpoint.best_test_loss")
        name = _lora_input(ws, fit={"maxiter": 1})
        main([
            name,
            "-p",
            str(base.seed_dir / "model.yaml"),
            "-cn",
            ckpt,
            "--reset-epoch-and-step",
        ])
        meta = read_model_metadata(str(ws.seed_dir / "model.yaml"))
        assert meta["param_dtype"] == "float32"
        assert _model_is_lora(ws.seed_dir)


class TestLoraRestartModelFile:
    @staticmethod
    def _args(**kwargs):
        defaults = {
            "restart_best_test": False,
            "restart_latest": False,
            "restart_suffix": None,
        }
        return argparse.Namespace(**{**defaults, **kwargs})

    @pytest.mark.parametrize(
        "flags",
        [
            {"restart_best_test": True},
            {"restart_latest": True},
            {"restart_suffix": ".epoch_1"},
        ],
    )
    def test_every_restart_kind_with_lora_reads_the_saved_model(self, tmp_path, flags):
        (tmp_path / "model.yaml").write_text("x")
        found = gracemaker.lora_restart_model_file(
            self._args(**flags), {"lora": LORA}, str(tmp_path)
        )
        assert found == str(tmp_path / "model.yaml")

    @pytest.mark.parametrize(
        ("flags", "potential", "saved"),
        [
            ({}, {"lora": LORA}, True),  # not a restart
            ({"restart_latest": True}, {}, True),  # no LoRA asked for
            ({"restart_latest": True}, {"lora": {}}, True),  # an empty LoRA config
            ({"restart_latest": True}, {"lora": LORA}, False),  # nothing saved yet
        ],
    )
    def test_otherwise_the_model_is_built_as_before(
        self, tmp_path, flags, potential, saved
    ):
        if saved:
            (tmp_path / "model.yaml").write_text("x")
        assert (
            gracemaker.lora_restart_model_file(
                self._args(**flags), potential, str(tmp_path)
            )
            is None
        )


class TestLoraDocumentedExample:
    def test_the_example_of_the_input_file_documentation_activates(
        self, ws, trained, training_probe
    ):
        """The ``lora`` line of ``docs/gracemaker/inputfile.md`` (``I`` names no instruction of this model)."""
        doc = (
            Path(__file__).resolve().parents[1] / "docs/gracemaker/inputfile.md"
        ).read_text()
        line = next(ln for ln in doc.splitlines() if ln.strip().startswith("# lora:"))
        example = yaml.safe_load(line.split("# lora:", 1)[1])
        prev = ws.run / "prev"
        shutil.copytree(trained.seed_dir, prev)
        name = ws.write_input(**_merge(FROM_FILE, {"potential": {"lora": example}}))
        tp = _run_to_probe([name, "-p", str(prev / "model.yaml")], training_probe)
        assert tp.is_lora_enabled()
        ranks = {v.name: v.shape for v in tp.model.variables if "/LORA/" in v.name}
        assert 8 in ranks["Z/w/LORA/A:0"]  # Z overrides `all` (rank 16)
