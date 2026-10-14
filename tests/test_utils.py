"""Tests of ``utils.py``: ``get_param_dtype_from_config`` and ``enforce_pbc``.

``get_param_dtype_from_config`` is the ``param_dtype`` default of a training config. This is one of the code defaults of ``param_dtype`` that disagree (see ``test_metadata_utils.py``): a config
without ``param_dtype`` gives float32, a yaml without metadata gives float64. Pinned, not endorsed.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import logging
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
import pytest
import tensorflow as tf
from ase import Atoms
from ase.build import fcc111

from tensorpotential.calculator import TPCalculator
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.utils import (
    NumpyEncoder,
    ModelFormatError,
    convert_model_reduce_elements,
    discovery_fit_folders,
    enforce_pbc,
    get_common_prefix,
    get_param_dtype_from_config,
    is_chief,
    load_fit,
    load_metrics,
    plot_dashboard,
    plot_fit,
    plot_many_fits,
    process_fit_dict,
    select_elements_in_model,
    update_fit_metrics,
)
from tests.tolerances import FLOAT64_ARITHMETIC
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

pytestmark = pytest.mark.usefixtures("info_logs")


class RecordingLog:
    """The two methods of a logger that ``get_param_dtype_from_config`` calls, remembering the arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def info(self, *args, **kwargs):
        self.calls.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args, kwargs))


@pytest.mark.parametrize(
    ("name", "dtype"), [("float32", tf.float32), ("float64", tf.float64)]
)
def test_param_dtype_key_is_read(name, dtype):
    assert get_param_dtype_from_config({"param_dtype": name}) == (dtype, name)


def test_default_without_a_key_is_float32():
    # PINNED, not endorsed: float32 here; metadata_utils.resolve_param_dtype gives float64 for a yaml without the key
    assert get_param_dtype_from_config({}) == (tf.float32, "float32")


def test_unknown_name_is_a_key_error():
    with pytest.raises(KeyError, match="Unknown dtype name float16"):
        get_param_dtype_from_config({"param_dtype": "float16"})


def test_deprecated_float_dtype_is_used_when_param_dtype_is_absent():
    assert get_param_dtype_from_config({"float_dtype": "float64"}) == (
        tf.float64,
        "float64",
    )


def test_param_dtype_wins_over_the_deprecated_float_dtype():
    config = {"param_dtype": "float32", "float_dtype": "float64"}

    assert get_param_dtype_from_config(config) == (tf.float32, "float32")


def test_log_reports_the_chosen_dtype():
    log = RecordingLog()

    get_param_dtype_from_config({"param_dtype": "float64"}, log=log)

    assert log.calls == [("info", ("Model parameter dtype: float64",), {})]


def test_log_reports_the_default():
    log = RecordingLog()

    get_param_dtype_from_config({}, log=log)

    assert log.calls == [
        ("info", ("Using default model parameter dtype: float32",), {})
    ]


def test_deprecated_key_warns_with_the_stacklevel_and_class_as_extra_arguments():
    log = RecordingLog()

    get_param_dtype_from_config({"float_dtype": "float64"}, log=log)

    kind, args, kwargs = log.calls[0]
    assert kind == "warning"
    assert "'float_dtype' is deprecated" in args[0]
    assert args[1] is DeprecationWarning
    assert kwargs == {"stacklevel": 2}
    assert log.calls[1] == ("info", ("Model parameter dtype: float64",), {})


def test_deprecated_key_warning_cannot_be_formatted_by_a_real_logger():
    # PINNED, not endorsed (TEST8 finding): logging.Logger.warning(msg, DeprecationWarning, stacklevel=2) passes the
    # class as a %-argument of a message without placeholders, so formatting the record raises TypeError (logging
    # prints a "Logging error" to stderr instead of failing) and the text of the warning is lost.
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("test_utils.param_dtype")
    logger.setLevel(logging.DEBUG)
    logger.propagate = (
        False  # pytest's own handler formats records and re-raises the error
    )
    logger.addHandler(Collect())
    try:
        result = get_param_dtype_from_config({"float_dtype": "float64"}, log=logger)
    finally:
        logger.handlers.clear()

    assert result == (tf.float64, "float64")
    warning = next(r for r in records if r.levelno == logging.WARNING)
    with pytest.raises(TypeError, match="not all arguments converted"):
        warning.getMessage()


# ---------------------------------------------------------------------------------------------
# enforce_pbc. ``GeometricalDataBuilder.extract_from_ase_atoms`` calls it with the largest cutoff
# before it builds pairs. Every test below **pins current behaviour** and is not an endorsement:
# where a behaviour looks like a hazard (the caller's ``Atoms`` is edited in place, partial periodic
# boundaries are overridden, a zero cell vector is made periodic) the docstring says so, and a later
# fix is meant to change the test on purpose.
# ---------------------------------------------------------------------------------------------

CUTOFF = 4.0


def dimer(**kwargs) -> Atoms:
    return Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]], **kwargs)


def test_aperiodic_input_gets_a_cube_of_twice_the_extent_plus_cutoff() -> None:
    """Pins current behaviour: the edge is ``2 * (max |pos - pos[0]| + cutoff)``, here 2 * (2.5 + 4)."""
    atoms = enforce_pbc(dimer(), CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 13.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.get_volume() == pytest.approx(13.0**3, rel=ARITH.rtol)


def test_aperiodic_input_is_centred_in_the_new_cell() -> None:
    atoms = enforce_pbc(dimer(), CUTOFF)
    np.testing.assert_allclose(
        atoms.positions,
        [[5.25, 6.5, 6.5], [7.75, 6.5, 6.5]],
        rtol=ARITH.rtol,
        atol=ARITH.atol,
    )


def test_a_single_atom_gets_a_cell_of_twice_the_cutoff() -> None:
    atoms = enforce_pbc(Atoms("Cu", positions=[[1.0, 2.0, 3.0]]), CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 8.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )
    np.testing.assert_allclose(
        atoms.positions, [[4.0, 4.0, 4.0]], rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_the_extent_is_measured_from_the_first_atom_only() -> None:
    """Pins current behaviour: the same three collinear atoms give different cells depending on
    which one is listed first (extent 4 from an end atom, 2 from the middle atom)."""
    xs = [[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]]
    from_end = enforce_pbc(Atoms("Cu3", positions=xs), CUTOFF)
    from_middle = enforce_pbc(Atoms("Cu3", positions=[xs[1], xs[0], xs[2]]), CUTOFF)
    assert from_end.cell[0, 0] == pytest.approx(2 * (4.0 + CUTOFF), rel=ARITH.rtol)
    assert from_middle.cell[0, 0] == pytest.approx(2 * (2.0 + CUTOFF), rel=ARITH.rtol)


def test_new_cell_keeps_periodic_images_of_a_cluster_beyond_the_cutoff() -> None:
    """The cube is large enough that the closest image of any atom is farther than the cutoff:
    the box edge minus the cluster extent is at least ``2 * cutoff``."""
    rng = np.random.default_rng(4)
    positions = rng.normal(scale=2.0, size=(12, 3))
    atoms = enforce_pbc(Atoms("Cu12", positions=positions), CUTOFF)
    extent = np.ptp(positions, axis=0).max()
    assert atoms.cell[0, 0] - extent >= 2 * CUTOFF


def test_the_callers_atoms_is_edited_in_place_and_returned() -> None:
    """Pins current behaviour, a hazard for stress (the volume of an invented cell): the object
    passed in is changed (cell, pbc and positions), and the function returns that same object."""
    original = dimer()
    returned = enforce_pbc(original, CUTOFF)
    assert returned is original
    assert original.pbc.tolist() == [True, True, True]
    assert original.cell.array.any()
    assert original.positions[0, 1] == pytest.approx(6.5, rel=ARITH.rtol)


def test_a_copy_protects_the_caller() -> None:
    original = dimer()
    enforce_pbc(original.copy(), CUTOFF)
    assert original.pbc.tolist() == [False, False, False]
    assert not original.cell.array.any()
    assert original.positions[1, 0] == 2.5


def test_an_existing_cell_of_an_aperiodic_input_is_overwritten() -> None:
    """Pins current behaviour: a cell the caller set is replaced when ``pbc`` is all False."""
    atoms = dimer(cell=50.0 * np.eye(3), pbc=False)
    enforce_pbc(atoms, CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 13.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_a_fully_periodic_input_is_left_unchanged() -> None:
    cell = np.array([[5.0, 0, 0], [1.0, 6.0, 0], [0.0, 0.5, 7.0]])
    atoms = dimer(cell=cell, pbc=True)
    before = atoms.positions.copy()
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)
    np.testing.assert_array_equal(atoms.positions, before)


def test_partial_periodic_boundaries_are_overridden_without_touching_cell_or_positions() -> (
    None
):
    """Pins current behaviour, not an endorsement: a slab flagged ``pbc = (T, T, F)`` becomes
    periodic along z as well, with the cell and positions as they were."""
    slab = fcc111("Cu", size=(2, 2, 3), a=3.61, vacuum=1.0)
    assert slab.pbc.tolist() == [True, True, False]
    cell, positions = slab.cell.array.copy(), slab.positions.copy()
    enforce_pbc(slab, CUTOFF)
    assert slab.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(slab.cell.array, cell)
    np.testing.assert_array_equal(slab.positions, positions)


@pytest.mark.parametrize(
    "pbc", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_any_single_periodic_direction_prevents_a_new_cell(pbc) -> None:
    """Pins current behaviour: only an all-False ``pbc`` triggers the new cell."""
    cell = 20.0 * np.eye(3)
    atoms = dimer(cell=cell, pbc=pbc)
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)


def test_a_zero_third_cell_vector_is_made_periodic_and_stays_zero() -> None:
    """Pins current behaviour, not an endorsement: the ASE convention for a 2D system (a zero
    third vector with ``pbc = (T, T, F)``) ends with ``pbc`` True along that zero vector.
    The pair search then fails (see ``test_databuilder.py``)."""
    cell = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 0.0]])
    atoms = dimer(cell=cell, pbc=[True, True, False])
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)


def test_an_aperiodic_input_with_a_zero_cell_gets_a_valid_cell() -> None:
    """Pins current behaviour: with ``pbc`` all False a zero cell is not a problem, it is replaced."""
    atoms = dimer(cell=np.zeros((3, 3)), pbc=False)
    enforce_pbc(atoms, CUTOFF)
    assert np.linalg.det(atoms.cell.array) > 0


def test_the_cutoff_enters_the_cell_linearly() -> None:
    small = enforce_pbc(dimer(), 3.0).cell[0, 0]
    large = enforce_pbc(dimer(), 5.0).cell[0, 0]
    assert large - small == pytest.approx(2 * (5.0 - 3.0), rel=ARITH.rtol)


def test_reduced_element_model_reproduces_energy_and_forces_of_the_full_model(tmp_path):
    """``LINEAR`` ends in a ``FunctionReduce`` with one weight set per central atom type, which
    ``convert_model_reduce_elements`` reduces through ``prepare_variables_for_selected_elements``. The full
    model is the oracle on a cluster that holds only the kept elements (Mo and Ta of Mo, Nb, Ta, W)."""
    instructions = get_preset("LINEAR")(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3},
        lmax=1,
        n_rad_max=6,
        n_rad_base=4,
        embedding_size=4,
        rcut=5.0,
    ).get_instructions()
    full_yaml, full_ckpt = str(tmp_path / "full.yaml"), str(tmp_path / "full")
    reduced_yaml, reduced_ckpt = str(tmp_path / "reduced.yaml"), str(tmp_path / "reduced")
    full = TensorPotential(instructions, param_dtype=tf.float64)
    save_instructions_dict(full_yaml, instructions, param_dtype=tf.float64)
    full.save_checkpoint(checkpoint_name=full_ckpt)
    convert_model_reduce_elements(
        element_map={"Mo": 0, "Ta": 1},
        potential_file_name=full_yaml,
        checkpoint_name=full_ckpt,
        new_potential_file_name=reduced_yaml,
        new_checkpoint_name=reduced_ckpt,
        param_dtype=tf.float64,
    )
    reduced = TensorPotential(
        load_instructions(reduced_yaml), param_dtype=tf.float64
    )
    reduced.load_checkpoint(checkpoint_name=reduced_ckpt)

    def energy_and_forces(model):
        atoms = Atoms(
            "MoTaMo",
            positions=[[0.0, 0.0, 0.0], [0.3, 0.1, 2.2], [2.1, -0.4, 0.9]],
            pbc=False,
        )
        atoms.calc = TPCalculator(model.model)
        return atoms.get_potential_energy(), atoms.get_forces()

    e_full, f_full = energy_and_forces(full)
    e_reduced, f_reduced = energy_and_forces(reduced)
    assert np.abs(f_full).max() > 1e-6
    np.testing.assert_allclose(e_reduced, e_full, **FLOAT64_ARITHMETIC._asdict())
    np.testing.assert_allclose(f_reduced, f_full, **FLOAT64_ARITHMETIC._asdict())


def test_cutoff_helpers_are_reexported_from_utils_unchanged() -> None:
    """The code moved to ``tensorpotential.core.cutoffs``; the old import path gives the same objects."""
    import tensorpotential.utils as legacy
    from tensorpotential.core import cutoffs

    assert legacy.process_cutoff_dict is cutoffs.process_cutoff_dict
    assert legacy.CUTOFF_PRESETS is cutoffs.CUTOFF_PRESETS


# ------------------------------------------------------------------ the JSON encoder and the chief test


def test_the_encoder_writes_numpy_scalars_and_arrays_as_plain_json():
    text = json.dumps(
        {"i": np.int64(3), "f": np.float32(0.5), "a": np.arange(4).reshape(2, 2), "b": np.float64(1.25)},
        cls=NumpyEncoder,
    )
    assert json.loads(text) == {"i": 3, "f": 0.5, "a": [[0, 1], [2, 3]], "b": 1.25}
    assert isinstance(json.loads(text)["i"], int)


def test_the_encoder_refuses_what_it_does_not_know():
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"s": {1, 2}}, cls=NumpyEncoder)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (None, True),
        (SimpleNamespace(), True),  # a single-process strategy has no cluster resolver
        (SimpleNamespace(cluster_resolver=SimpleNamespace()), True),  # a resolver without a task type
        (SimpleNamespace(cluster_resolver=SimpleNamespace(task_type="chief", task_id=5)), True),
        (SimpleNamespace(cluster_resolver=SimpleNamespace(task_type="worker", task_id=0)), True),
        (SimpleNamespace(cluster_resolver=SimpleNamespace(task_type="worker", task_id=1)), False),
        (SimpleNamespace(cluster_resolver=SimpleNamespace(task_type="ps", task_id=0)), False),
        (SimpleNamespace(cluster_resolver=SimpleNamespace(task_type="worker")), False),
    ],
)
def test_only_the_chief_or_the_first_worker_is_chief(strategy, expected):
    # a namespace stands in for a multi-worker strategy, which needs a running cluster
    assert is_chief(strategy) is expected


# ------------------------------------------------------------------ the fit-metrics helpers


def _write_fit(folder, test_f, finished=False):
    """A fit folder as gracemaker writes it: one entry per epoch, flattened metric names as keys."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, scale in (("test", 1.0), ("train", 2.0)):
        rows = [
            {"epoch": i, "rmse/f_comp": scale * v, "rmse/depa": 0.1 * scale * v}
            for i, v in enumerate(test_f)
        ]
        (folder / f"{name}_metrics.yaml").write_text(yaml.safe_dump(rows))
    if finished:
        (folder / "finished.txt").write_text("done")
    return str(folder)


@pytest.fixture
def matplotlib_axes():
    import matplotlib

    matplotlib.use("Agg")  # no display on a test machine: the backend is the hardware boundary replaced
    from matplotlib import pyplot as plt

    yield plt
    plt.close("all")


def test_load_metrics_reads_the_yaml_into_a_frame_with_one_row_per_epoch(tmp_path):
    folder = _write_fit(tmp_path / "fit", [0.5, 0.4, 0.3])
    frame = load_metrics(f"{folder}/test_metrics.yaml")
    assert sorted(frame.columns) == ["epoch", "rmse/depa", "rmse/f_comp"]  # the YAML writer sorts the keys
    np.testing.assert_array_equal(frame["rmse/f_comp"], [0.5, 0.4, 0.3])


def test_load_fit_collects_both_metric_files_and_the_finished_flag(tmp_path):
    unfinished = load_fit(_write_fit(tmp_path / "a", [0.5, 0.4]))
    finished = load_fit(_write_fit(tmp_path / "b", [0.5], finished=True))
    assert unfinished["final"] is False and finished["final"] is True
    assert unfinished["name"] == str(tmp_path / "a")
    assert len(unfinished["test"]) == 2 and len(unfinished["train"]) == 2
    np.testing.assert_array_equal(unfinished["train"]["rmse/f_comp"], [1.0, 0.8])


def test_a_folder_without_metrics_cannot_be_loaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fit(str(tmp_path))


def test_fit_folders_are_found_below_a_root_when_both_metric_files_are_there(tmp_path):
    a = _write_fit(tmp_path / "x" / "a", [0.1])
    b = _write_fit(tmp_path / "x" / "y" / "b", [0.1])
    only_test = tmp_path / "c"
    only_test.mkdir()
    (only_test / "test_metrics.yaml").write_text("[]")
    assert sorted(discovery_fit_folders(str(tmp_path))) == sorted([a, b])
    assert discovery_fit_folders(str(tmp_path / "nothing-here")) == []


@pytest.mark.parametrize(
    ("names", "align", "expected"),
    [
        (["run/a/seed/1", "run/a/seed/2"], True, "run/a/seed/"),
        (["run/aa", "run/ab"], True, "run"),
        (["run/aa", "run/ab"], False, "run/a"),
        (["run/a/", "run/a/b"], True, "run/a/"),
        (["solo"], True, ""),
        (["x/one", "y/two"], True, ""),
    ],
)
def test_the_common_prefix_can_be_cut_at_the_last_folder_separator(names, align, expected):
    assert get_common_prefix(names, align_folder=align) == expected
    assert get_common_prefix(tuple(names), align_folder=align) == expected


def test_update_fit_metrics_loads_new_folders_and_reports_the_epochs(tmp_path, caplog):
    a = _write_fit(tmp_path / "a", [0.5, 0.4, 0.3])
    fits: dict = {}
    update_fit_metrics(fits, [a])
    assert list(fits) == [a] and len(fits[a]["test"]) == 3
    assert f"{a}: 3 epochs" in caplog.text


def test_update_fit_metrics_reloads_a_known_running_fit(tmp_path):
    a = _write_fit(tmp_path / "a", [0.5])
    fits: dict = {}
    update_fit_metrics(fits, [a])
    _write_fit(tmp_path / "a", [0.5, 0.4])
    update_fit_metrics(fits, [a])
    assert len(fits[a]["test"]) == 2


def test_update_fit_metrics_skips_a_fit_marked_finished_unless_asked_to_rerun(tmp_path):
    a = _write_fit(tmp_path / "a", [0.5])
    fits = {a: {"finished": True, "marker": "kept"}}
    update_fit_metrics(fits, [a])
    assert fits[a] == {"finished": True, "marker": "kept"}
    update_fit_metrics(fits, [a], rerun=True)
    assert fits[a]["name"] == a  # reloaded


def test_a_fit_that_has_not_changed_for_the_wait_time_is_marked_finished(tmp_path, caplog):
    a = _write_fit(tmp_path / "a", [0.5])
    old = time.time() - (3 * 3600 + 25 * 60 + 7)
    os.utime(f"{a}/test_metrics.yaml", (old, old))
    fits = {a: {"marker": "kept"}}
    update_fit_metrics(fits, [a], wait_time_seconds=3600)
    assert fits[a] == {"marker": "kept", "finished": True}
    assert "no update for 3h:25m:7s, mark as finished" in caplog.text
    fresh = {a: {"marker": "kept"}}
    update_fit_metrics(fresh, [a], wait_time_seconds=10 * 3600)
    assert "finished" not in fresh[a] and fresh[a]["name"] == a


def test_a_folder_that_cannot_be_read_is_reported_and_leaves_the_dictionary_alone(tmp_path, caplog):
    fits: dict = {}
    update_fit_metrics(fits, [str(tmp_path / "missing")])
    assert fits == {}
    assert caplog.messages[0].startswith("Error:")


def _fit_dict(tmp_path):
    """Four runs: two settings ("m/a" and "m/b") with seeds 1 and 2, best epochs known by hand."""
    runs = {
        "m/a/seed/1": [0.9, 0.30, 0.50],
        "m/a/seed/2": [0.8, 0.40, 0.45],
        "m/b/seed/1": [0.7, 0.20, 0.60],
        "m/b/seed/2": [0.6, 0.10, 0.50],
    }
    return {name: load_fit(_write_fit(tmp_path / name, values)) for name, values in runs.items()}, runs


def test_process_fit_dict_names_the_runs_and_picks_the_best_epoch_of_each(tmp_path):
    fits, runs = _fit_dict(tmp_path)
    prefix_names = {str(tmp_path / n): fd for n, fd in zip(runs, fits.values(), strict=True)}
    prefix, best, grouped = process_fit_dict(prefix_names, fkey="rmse/f_comp", align_folder=True)

    assert prefix.endswith("/m/")
    assert sorted(fd["shortname"] for fd in prefix_names.values()) == ["a", "a", "b", "b"]
    assert sorted(fd["shortname_and_seed"] for fd in prefix_names.values()) == [
        "a/seed/1",
        "a/seed/2",
        "b/seed/1",
        "b/seed/2",
    ]
    # the best test value of each run, sorted ascending
    assert list(best["rmse/f_comp"]) == [0.1, 0.2, 0.3, 0.4]
    assert list(best["shortname"]) == ["b", "b", "a", "a"]
    # grouped by setting: mean, std (n - 1), min and the list
    column = grouped["rmse/f_comp"]
    np.testing.assert_allclose(column.loc["a", "mean"], 0.35, rtol=ARITH.rtol)
    np.testing.assert_allclose(column.loc["a", "std"], np.std([0.3, 0.4], ddof=1), rtol=ARITH.rtol)
    np.testing.assert_allclose(column.loc["b", "min"], 0.1, rtol=ARITH.rtol)
    assert sorted(column.loc["b", "list"]) == [0.1, 0.2]


def test_process_fit_dict_without_alignment_cuts_the_prefix_at_the_character(tmp_path):
    fits = {
        str(tmp_path / "run_a/seed/1"): load_fit(_write_fit(tmp_path / "run_a/seed/1", [0.3])),
        str(tmp_path / "run_b/seed/1"): load_fit(_write_fit(tmp_path / "run_b/seed/1", [0.2])),
    }
    prefix, _, _ = process_fit_dict(fits, align_folder=False)
    assert prefix.endswith("/run_")
    assert sorted(fd["shortname"] for fd in fits.values()) == ["a", "b"]


# ------------------------------------------------------------------ plotting (the Agg backend, no display)


def _fit(test, train, name="fit"):
    def frame(values):
        return pd.DataFrame({
            "rmse/f_comp": values,
            "rmse/depa": [x / 10 for x in values],
            "step": [10 * i for i in range(len(values))],
        })

    return {"test": frame(test), "train": frame(train), "name": name, "shortname": name}


def test_plot_fit_draws_the_test_curve_with_its_minimum_in_meV_and_the_train_curve_dashed(matplotlib_axes):
    fig, ax = matplotlib_axes.subplots()
    plot_fit(_fit([0.5, 0.25, 0.3], [0.9, 0.8, 0.7]), ax=ax, name="run")
    test_line, train_line = ax.lines
    np.testing.assert_array_equal(test_line.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(test_line.get_ydata(), [0.5, 0.25, 0.3])
    assert test_line.get_label() == "run(250.0)"  # the minimum 0.25 times 1e3
    np.testing.assert_array_equal(train_line.get_ydata(), [0.9, 0.8, 0.7])
    assert train_line.get_linestyle() == "--"
    assert train_line.get_color() == test_line.get_color()
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Epoch", "rmse/f_comp")


def test_plot_fit_can_leave_out_a_curve_use_another_metric_and_another_abscissa(matplotlib_axes):
    fit = _fit([0.5, 0.25], [0.9, 0.8])
    _, ax = matplotlib_axes.subplots()
    plot_fit(fit, k="rmse/depa", ax=ax, name="n", plot_train=False, x_name="step")
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 10])
    assert ax.get_ylabel() == "rmse/depa"
    _, ax2 = matplotlib_axes.subplots()
    plot_fit(fit, ax=ax2, name="n", plot_test=False, x_name="step")
    assert len(ax2.lines) == 1
    np.testing.assert_array_equal(ax2.lines[0].get_xdata(), [0, 10])
    assert ax2.lines[0].get_linestyle() == "--"


def test_plot_fit_takes_the_current_axes_and_the_short_name_by_default(matplotlib_axes):
    matplotlib_axes.figure()
    plot_fit(_fit([0.5], [0.9], name="short"))
    assert matplotlib_axes.gca().lines[0].get_label() == "short(500.0)"


def test_many_fits_are_plotted_by_decreasing_best_test_value_from_a_dict_or_a_list(matplotlib_axes, caplog):
    fits = [_fit([0.5], [1.0], "mid"), _fit([0.9], [1.0], "worst"), _fit([0.1], [1.0], "best")]
    matplotlib_axes.figure()
    plot_many_fits({i: f for i, f in enumerate(fits)})
    assert caplog.messages == ["worst", "mid", "best"]
    caplog.clear()
    original = list(fits)
    plot_many_fits(fits, plot_train=True)
    assert caplog.messages == ["worst", "mid", "best"]
    assert fits == original  # the caller's list keeps its order
    with pytest.raises(UnboundLocalError):  # pinned: only a dict or a list is accepted
        plot_many_fits(tuple(fits))


def test_the_dashboard_plots_the_energy_and_force_metrics_of_every_included_fit(matplotlib_axes):
    fits = {"a": _fit([0.5, 0.4], [0.9, 0.8], "A"), "b": _fit([0.3, 0.2], [0.6, 0.5], "B"), "c": _fit([0.1], [0.1], "C")}
    fig, ax_e, ax_f = plot_dashboard(fits, exclude_list=["c"])
    assert len(ax_e.lines) == 2 and len(ax_f.lines) == 2
    assert ax_e.get_ylabel() == "rmse/depa" and ax_f.get_ylabel() == "rmse/f_comp"
    assert [line.get_label() for line in ax_f.lines] == ["A(400.0)", "B(200.0)"]
    np.testing.assert_allclose(ax_e.lines[0].get_ydata(), [0.05, 0.04], rtol=ARITH.rtol)
    assert fig is ax_e.figure


def test_the_dashboard_can_select_fits_a_metric_a_label_and_the_train_curves(matplotlib_axes):
    fits = {"a": _fit([0.5, 0.4], [0.9, 0.8], "A"), "b": _fit([0.3, 0.2], [0.6, 0.5], "B")}
    fits["a"]["other"] = "label-a"
    fits["b"]["other"] = "label-b"
    for fd in fits.values():
        for part in ("test", "train"):
            fd[part]["mae/depa"] = fd[part]["rmse/depa"]
            fd[part]["mae/f_comp"] = fd[part]["rmse/f_comp"]
    _, ax_e, ax_f = plot_dashboard(fits, metric="mae", include_list=["b"], label="other", plot_train=True)
    assert len(ax_f.lines) == 2  # test and train of b
    assert ax_f.lines[0].get_label() == "label-b(200.0)"
    assert ax_f.get_ylabel() == "mae/f_comp"
    _, _, ax_f2 = plot_dashboard(fits, x_name="step")
    np.testing.assert_array_equal(ax_f2.lines[0].get_xdata(), [0, 10])


def test_the_dashboard_uses_the_current_figure_when_an_axis_is_given(matplotlib_axes):
    fig, (ax_f, ax_e) = matplotlib_axes.subplots(2, 1)
    got_fig, got_e, got_f = plot_dashboard({"a": _fit([0.5], [0.5], "A")}, ax_e=ax_e, ax_f=ax_f)
    assert got_fig is fig and got_e is ax_e and got_f is ax_f


# ------------------------------------------------------------------ element selection errors


def _linear_model(tmp_path):
    instructions = get_preset("LINEAR")(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2}, lmax=1, n_rad_max=4, n_rad_base=4, embedding_size=4, rcut=5.0
    ).get_instructions()
    checkpoint = str(tmp_path / "ckpt")
    TensorPotential(instructions, param_dtype=tf.float64).save_checkpoint(checkpoint_name=checkpoint)
    return instructions, checkpoint


def test_selecting_an_element_the_model_does_not_have_is_refused_naming_the_known_ones(tmp_path):
    # the embedding refuses the unknown symbol first: the length check after it cannot be reached
    instructions, checkpoint = _linear_model(tmp_path)
    with pytest.raises(ValueError, match=r"Element W not found in the map \(\{'Mo': .*'Nb': .*'Ta': .*\}\)"):
        select_elements_in_model(["Mo", "W"], instructions, checkpoint, tf.float64)


def test_selecting_from_a_model_without_a_chemical_embedding_is_refused(tmp_path):
    from tensorpotential.instructions.compute import BondLength

    checkpoint = str(tmp_path / "ckpt")
    TensorPotential({"r": BondLength(name="r")}, param_dtype=tf.float64).save_checkpoint(checkpoint_name=checkpoint)
    with pytest.raises(ValueError, match="No ScalarChemicalEmbedding found in instructions dict"):
        select_elements_in_model(["Mo"], {"r": BondLength(name="r")}, checkpoint, tf.float64)


def test_the_selection_keeps_the_embedding_rows_of_the_chosen_elements_in_the_requested_order(tmp_path):
    instructions, checkpoint = _linear_model(tmp_path)
    embedding = next(v for v in instructions.values() if type(v).__name__ == "ScalarChemicalEmbedding")
    full = TensorPotential(instructions, param_dtype=tf.float64)
    full.load_checkpoint(checkpoint_name=checkpoint)
    rows = embedding.w.numpy().copy()  # one row per element of Mo, Nb, Ta
    tp = select_elements_in_model(["Ta", "Mo"], instructions, checkpoint, tf.float64)

    assert isinstance(tp, TensorPotential)
    np.testing.assert_array_equal(embedding.w.numpy(), rows[[2, 0]])
    assert [s.decode() for s in embedding.element_map_symbols.numpy()] == ["Ta", "Mo"]
    # the constructor arguments that go into the saved yaml name the elements in the requested order (read from the
    # instruction: `convert_model_reduce_elements` writes the yaml in the test above, not this function)
    assert embedding._init_args["element_map"] == {"Ta": 0, "Mo": 1}


def test_converting_with_an_unknown_parameter_dtype_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="Unknown dtype <dtype: 'int32'>. Supported dtypes"):
        convert_model_reduce_elements(
            {"Mo": 0}, str(tmp_path / "a.yaml"), str(tmp_path / "ckpt"), str(tmp_path / "b.yaml"), str(tmp_path / "ckpt2"), param_dtype=tf.int32
        )


def test_converting_a_model_in_the_old_list_format_is_refused_with_a_hint(tmp_path):
    instructions = get_preset("LINEAR")(
        element_map={"Mo": 0}, lmax=1, n_rad_max=4, n_rad_base=4, embedding_size=4, rcut=5.0
    ).get_instructions()
    new_format = tmp_path / "new.yaml"
    save_instructions_dict(str(new_format), instructions, param_dtype=tf.float64)
    # the very old format is the bare list of instruction records
    records = list(yaml.safe_load(new_format.read_text())["instructions"].values())
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(records))
    with pytest.raises(ModelFormatError, match="is NOT in new format") as refused:
        convert_model_reduce_elements({"Mo": 0}, str(old), str(tmp_path / "ckpt"), str(tmp_path / "b.yaml"), str(tmp_path / "ckpt2"), param_dtype=tf.float64)
    assert "update_model" in str(refused.value)
    assert isinstance(refused.value, ValueError)


def test_an_unexpected_error_while_loading_a_fit_is_not_hidden(tmp_path, monkeypatch):
    a = _write_fit(tmp_path / "a", [0.5])

    def broken(folder):
        raise RuntimeError("not a read error")

    monkeypatch.setattr("tensorpotential.utils.load_fit", broken)
    with pytest.raises(RuntimeError, match="not a read error"):
        update_fit_metrics({}, [a])
