"""``tensorpotential.scripts.grace_preprocess``: argument parser, data builders and the stress column of stage 1.

The physical layer runs the real entry point (``main``, stage 1) on a small dataframe and compares the
stored virial with one computed by hand from the input stress: ``virial = -stress * volume``, reordered
from the ASE Voigt order (xx, yy, zz, yz, xz, xy) to the order of the data pipeline (xx, yy, zz, xy, xz, yz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from ase.build import bulk

from tensorpotential import constants
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
)
from tensorpotential.data.process_df import (
    ENERGY_CORRECTED_COL,
    FORCES_COL,
    STRESS_COL,
)
from tensorpotential.scripts import grace_preprocess

from tests.tolerances import FLOAT64_ARITHMETIC

N_STRUCTURES = 3
# Stress in the input units, ASE Voigt order (xx, yy, zz, yz, xz, xy); one distinct value per component.
VOIGT_STRESS = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
VOIGT_TO_PIPELINE_ORDER = [0, 1, 2, 5, 4, 3]


def _make_dataframe() -> pd.DataFrame:
    """Rattled four-atom Cu cells of different size, each with the same stress ``VOIGT_STRESS``."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_STRUCTURES):
        atoms = bulk("Cu", "fcc", a=3.5 + 0.1 * i, cubic=True)
        atoms.rattle(stdev=0.05, rng=rng)
        rows.append({
            "ase_atoms": atoms,
            ENERGY_CORRECTED_COL: -3.5 * len(atoms),
            FORCES_COL: rng.normal(size=(len(atoms), 3)),
            STRESS_COL: VOIGT_STRESS.copy(),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dataframe() -> pd.DataFrame:
    return _make_dataframe()


@pytest.fixture(scope="module")
def input_file(dataframe, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("preprocess_input") / "data.pkl.gz"
    dataframe.to_pickle(path)
    return path


def run_stage_1(input_file: Path, output: Path, *options: str) -> list[dict]:
    """Run ``grace_preprocess --stage-1`` through ``main`` and return the stored samples."""
    grace_preprocess.main([
        str(input_file),
        "-o",
        str(output),
        "-e",
        "Cu",
        "-c",
        "4.0",
        "--is-fit-stress",
        "--stage-1",
        *options,
    ])
    dataset = tf.data.Dataset.load(
        str(output / "stage1" / "shard_1-of-1"), compression="GZIP"
    )
    return [{k: v.numpy() for k, v in sample.items()} for sample in dataset]


def hand_virial(atoms, voigt_stress: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """``-stress * volume`` in the pipeline's component order, times a unit factor, shape (1, 6)."""
    virial = -(voigt_stress * atoms.get_volume())[VOIGT_TO_PIPELINE_ORDER]
    return (virial * factor).reshape(1, 6)


# ---------------------------------------------------------------------------------------------------
# Logic: the argument parser and the data builders
# ---------------------------------------------------------------------------------------------------


def test_parser_defaults():
    args = grace_preprocess.build_parser().parse_args(["data.pkl.gz"])
    assert args.input == ["data.pkl.gz"]
    assert args.output == "tf_dataset"
    assert args.batch_size == 8
    assert args.max_n_buckets == 5
    assert args.cutoff == 6
    assert args.cutoff_dict is None
    assert args.compression == "GZIP"
    assert args.energy_col == ENERGY_CORRECTED_COL
    assert args.forces_col == FORCES_COL
    assert args.stress_col == STRESS_COL
    assert args.is_fit_stress is False
    assert args.precision == "float64"
    assert args.task_id == 0
    assert args.total_task_num == 1
    assert not any([args.stage_1, args.stage_2, args.stage_3, args.stage_4])
    assert args.dense_nbr is False


def test_parser_reads_every_option():
    args = grace_preprocess.build_parser().parse_args([
        "a.pkl.gz",
        "b.pkl.gz",
        "-o",
        "out",
        "--sharded-input",
        "-e",
        "Cu Al",
        "-b",
        "4",
        "-bu",
        "2",
        "-c",
        "5.5",
        "--energy-col",
        "E",
        "--forces-col",
        "F",
        "--stress-col",
        "S",
        "--is-fit-stress",
        "--precision",
        "float32",
        "--task-id",
        "1",
        "--total-task-num",
        "3",
        "--rerun",
        "--stage-3",
        "--remove_stage1",
        "--dense-nbr",
    ])
    assert args.input == ["a.pkl.gz", "b.pkl.gz"]
    assert (args.output, args.elements, args.sharded_input) == ("out", "Cu Al", True)
    assert (args.batch_size, args.max_n_buckets, args.cutoff) == (4, 2, 5.5)
    assert (args.energy_col, args.forces_col, args.stress_col) == ("E", "F", "S")
    assert (args.is_fit_stress, args.precision) == (True, "float32")
    assert (args.task_id, args.total_task_num, args.rerun) == (1, 3, True)
    assert (args.stage_3, args.remove_stage1, args.dense_nbr) == (True, True, True)


def test_get_databuilders_builds_geometry_then_reference_builder():
    args = grace_preprocess.build_parser().parse_args([
        "data.pkl.gz",
        "--is-fit-stress",
        "--energy-col",
        "E",
        "--forces-col",
        "F",
        "--stress-col",
        "S",
    ])
    geom_db, ref_db = grace_preprocess.get_databuilders(
        {"Cu": 0}, 4.0, args, precision="float64", cutoff_dict=None
    )
    assert isinstance(geom_db, GeometricalDataBuilder)
    assert isinstance(ref_db, ReferenceEnergyForcesStressesDataBuilder)
    assert geom_db.is_fit_stress and ref_db.is_fit_stress
    assert (ref_db.energy_col, ref_db.forces_col, ref_db.stress_col) == ("E", "F", "S")
    assert ref_db.stress_conversion_factor == 1.0


# ---------------------------------------------------------------------------------------------------
# Physical values: the stress column that stage 1 stores, through the real entry point
# ---------------------------------------------------------------------------------------------------


def test_stage_1_default_stores_the_input_stress_as_virial(
    input_file, dataframe, tmp_path
):
    samples = run_stage_1(input_file, tmp_path / "out")
    assert len(samples) == N_STRUCTURES
    for sample, atoms in zip(samples, dataframe["ase_atoms"], strict=True):
        virial = sample[constants.DATA_REFERENCE_VIRIAL]
        np.testing.assert_allclose(
            virial,
            hand_virial(atoms, VOIGT_STRESS),
            rtol=FLOAT64_ARITHMETIC.rtol,
            atol=FLOAT64_ARITHMETIC.atol,
        )
        # The default path multiplies by exactly 1.0: the stored value is the unconverted product.
        assert np.array_equal(virial, hand_virial(atoms, VOIGT_STRESS))


# ---------------------------------------------------------------------------------------------------
# --stress-units
# ---------------------------------------------------------------------------------------------------

# Multiplier from the input units to eV/A3, computed by hand: 1 eV/A3 = 160.2176621 GPa = 1602.176621 kbar.
STRESS_UNIT_FACTORS = {
    "eV/A3": 1.0,
    "GPa": 1 / 160.2176621,
    "kbar": 1 / 1602.176621,
    "-kbar": -1 / 1602.176621,
}


def parse(*argv: str):
    return grace_preprocess.build_parser().parse_args(["data.pkl.gz", *argv])


def test_stress_units_default_is_ev_per_a3():
    assert parse().stress_units == "eV/A3"


@pytest.mark.parametrize("units", sorted(STRESS_UNIT_FACTORS))
def test_stress_units_accepts_every_documented_value(units):
    # The `=` form is the only one that works for -kbar (see the next test).
    assert parse(f"--stress-units={units}").stress_units == units


def test_stress_units_choices_are_the_ones_the_builder_converts():
    assert set(grace_preprocess.STRESS_UNITS_CHOICES) == set(STRESS_UNIT_FACTORS)
    for units in STRESS_UNIT_FACTORS:
        builder = ReferenceEnergyForcesStressesDataBuilder(stress_units=units)
        assert builder.stress_conversion_factor == pytest.approx(
            STRESS_UNIT_FACTORS[units],
            rel=FLOAT64_ARITHMETIC.rtol,
            abs=FLOAT64_ARITHMETIC.atol,
        )


def test_stress_units_rejects_other_values_and_lists_the_accepted_ones(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("--stress-units", "psi")
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "invalid choice: 'psi'" in message
    for units in STRESS_UNIT_FACTORS:
        assert f"'{units}'" in message


def test_stress_units_minus_kbar_needs_the_equals_form(capsys):
    # argparse reads a separate `-kbar` as an option; the help text says to use `=`.
    with pytest.raises(SystemExit) as exit_info:
        parse("--stress-units", "-kbar")
    assert exit_info.value.code == 2
    assert "expected one argument" in capsys.readouterr().err
    assert "--stress-units=-kbar" in grace_preprocess.build_parser().format_help()


@pytest.mark.parametrize("units", sorted(STRESS_UNIT_FACTORS))
def test_get_databuilders_passes_the_units_to_the_reference_builder(units):
    args = parse("--is-fit-stress", f"--stress-units={units}")
    _, ref_db = grace_preprocess.get_databuilders({"Cu": 0}, 4.0, args)
    assert ref_db.stress_conversion_factor == pytest.approx(
        STRESS_UNIT_FACTORS[units],
        rel=FLOAT64_ARITHMETIC.rtol,
        abs=FLOAT64_ARITHMETIC.atol,
    )


@pytest.mark.parametrize("units", ["GPa", "kbar", "-kbar"])
def test_stage_1_converts_the_stress_to_ev_per_a3(
    units, input_file, dataframe, tmp_path
):
    samples = run_stage_1(input_file, tmp_path / "out", f"--stress-units={units}")
    assert len(samples) == N_STRUCTURES
    for sample, atoms in zip(samples, dataframe["ase_atoms"], strict=True):
        np.testing.assert_allclose(
            sample[constants.DATA_REFERENCE_VIRIAL],
            hand_virial(atoms, VOIGT_STRESS, STRESS_UNIT_FACTORS[units]),
            rtol=FLOAT64_ARITHMETIC.rtol,
            atol=FLOAT64_ARITHMETIC.atol,
        )


def test_stage_1_with_ev_per_a3_equals_stage_1_without_the_option(input_file, tmp_path):
    omitted = run_stage_1(input_file, tmp_path / "omitted")
    explicit = run_stage_1(input_file, tmp_path / "explicit", "--stress-units=eV/A3")
    assert len(omitted) == len(explicit) == N_STRUCTURES
    for before, after in zip(omitted, explicit, strict=True):
        assert before.keys() == after.keys()
        for key in before:
            assert np.array_equal(before[key], after[key]), key


# ---------------------------------------------------------------------------------------------------
# main: the four stages, driven in process. Physical layer: the statistics of stages 2 and 4 against hand sums of the
# input (atom counts, squared forces) and the brute-force neighbour oracle of ``tests/neighbour_oracle.py``.
# ---------------------------------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402

from tests.neighbour_oracle import brute_force_pairs  # noqa: E402

CUTOFF = 4.0
BATCH_SIZE = 2


def run_main(input_files, output: Path, stage: int | None, *options: str) -> None:
    """``grace_preprocess`` with ``-e Cu -c 4.0 --is-fit-stress`` and the given stage flag (none for ``None``)."""
    files = input_files if isinstance(input_files, (list, tuple)) else [input_files]
    argv = [*map(str, files), "-o", str(output), "-e", "Cu", "-c", str(CUTOFF), "--is-fit-stress"]
    if stage is not None:
        argv.append(f"--stage-{stage}")
    grace_preprocess.main([*argv, *options])


def load_dataset(path: Path) -> list[dict]:
    dataset = tf.data.Dataset.load(str(path), compression="GZIP")
    return [{k: v.numpy() for k, v in sample.items()} for sample in dataset]


def oracle_neighbour_count(atoms) -> int:
    return len(brute_force_pairs(atoms.positions, atoms.cell[:], atoms.get_chemical_symbols(), CUTOFF))


@pytest.fixture(scope="module")
def pipeline(input_file, tmp_path_factory) -> Path:
    """All four stages on the three-structure frame: batches of two structures, two padding buckets."""
    out = tmp_path_factory.mktemp("pipeline") / "out"
    for stage in (1, 2, 3, 4):
        run_main(input_file, out, stage, "-b", str(BATCH_SIZE), "-bu", "2")
    return out


def test_stage_1_stores_params_and_one_sample_per_structure(pipeline, dataframe):
    params = json.loads((pipeline / "params.json").read_text())
    assert params["elements_map"] == {"Cu": 0}
    assert params["cutoff"] == CUTOFF
    assert params["precision"] == "float64"
    assert len(load_dataset(pipeline / "stage1" / "shard_1-of-1")) == N_STRUCTURES


def test_stage_2_counts_atoms_neighbours_and_structures_of_every_batch(pipeline, dataframe):
    batches = pd.read_pickle(pipeline / "stage2" / "batches_df-1-of-1.pkl.gz")
    assert list(batches.columns) == ["n_atoms", "n_neighbours", "n_structures"]
    assert sorted(batches["n_structures"]) == [1, 2]
    assert batches["n_structures"].sum() == N_STRUCTURES
    assert batches["n_atoms"].sum() == sum(len(a) for a in dataframe["ase_atoms"])
    assert batches["n_neighbours"].sum() == sum(oracle_neighbour_count(a) for a in dataframe["ase_atoms"])
    assert batches["n_neighbours"].is_monotonic_decreasing


def test_stage_3_padding_bounds_cover_every_batch(pipeline):
    bounds = json.loads((pipeline / "stage3" / "reduced_padding_bounds.json").read_text())
    batches = pd.read_pickle(pipeline / "stage2" / "batches_df-1-of-1.pkl.gz")
    assert len(bounds) == 2
    assert bounds == sorted(bounds)
    for n_neigh, n_atoms, n_struct in batches[["n_neighbours", "n_atoms", "n_structures"]].itertuples(index=False):
        assert any(b[0] >= n_neigh and b[1] >= n_atoms and b[2] >= n_struct for b in bounds)


def test_stage_3_batches_hold_the_real_atoms_structures_and_forces_of_the_input(pipeline, dataframe):
    batches = load_dataset(pipeline / "stage3" / "shard_1-of-1")
    assert len(batches) == 2
    assert sum(int(b[constants.N_STRUCTURES_BATCH_REAL]) for b in batches) == N_STRUCTURES
    assert sum(int(b[constants.N_ATOMS_BATCH_REAL]) for b in batches) == sum(len(a) for a in dataframe["ase_atoms"])
    stored = sum(float(np.sum(b[constants.DATA_REFERENCE_FORCES][: int(b[constants.N_ATOMS_BATCH_REAL])] ** 2)) for b in batches)
    given = float(sum(np.sum(f**2) for f in dataframe[FORCES_COL]))
    np.testing.assert_allclose(stored, given, rtol=FLOAT64_ARITHMETIC.rtol, atol=FLOAT64_ARITHMETIC.atol)


def test_stage_4_statistics_equal_the_hand_sums_of_the_input(pipeline, dataframe):
    stats = json.loads((pipeline / "stage3" / "stats.json").read_text())
    n_atoms = sum(len(a) for a in dataframe["ase_atoms"])
    n_neigh = sum(oracle_neighbour_count(a) for a in dataframe["ase_atoms"])
    sum_sqr = float(sum(np.sum(f**2) for f in dataframe[FORCES_COL]))
    assert stats["total_num_structures"] == N_STRUCTURES
    assert stats["total_num_of_atoms"] == n_atoms
    assert stats["total_num_of_neighs"] == n_neigh
    assert stats["total_num_of_batches"] == 2
    np.testing.assert_allclose(stats["sum_sqr_forces"], sum_sqr, rtol=FLOAT64_ARITHMETIC.rtol)
    np.testing.assert_allclose(stats["avg_n_neigh"], n_neigh / n_atoms, rtol=FLOAT64_ARITHMETIC.rtol)
    np.testing.assert_allclose(stats["scale"], np.sqrt(sum_sqr / n_atoms / 3), rtol=FLOAT64_ARITHMETIC.rtol)
    assert stats["element_map"] == {"Cu": 0}
    assert (stats["cutoff"], stats["batch_size"], stats["precision"]) == (CUTOFF, BATCH_SIZE, "float64")
    assert stats["cutoff_dict"] is None


@pytest.mark.parametrize("stages", [[], [1, 2], [2, 3, 4]])
def test_exactly_one_stage_flag_is_required(input_file, tmp_path, caplog, stages):
    argv = [str(input_file), "-o", str(tmp_path / "out")] + [f"--stage-{s}" for s in stages]
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as stop:
        grace_preprocess.main(argv)
    assert stop.value.code == 1
    assert "--stage-1 or --stage-2 or --stage-3 options must be provided" in caplog.text
    assert not (tmp_path / "out").exists()


def test_stage_1_is_skipped_when_its_folder_exists_and_redone_with_rerun(input_file, tmp_path, caplog):
    out = tmp_path / "out"
    run_main(input_file, out, 1)
    marker = out / "stage1" / "shard_1-of-1" / "marker.txt"
    marker.write_text("kept")
    with caplog.at_level(logging.INFO):
        run_main(input_file, out, 1)
    assert marker.exists()
    assert "Stage 1 is already finished" in caplog.text
    run_main(input_file, out, 1, "--rerun")
    assert not marker.exists()
    assert len(load_dataset(out / "stage1" / "shard_1-of-1")) == N_STRUCTURES


def test_stage_2_is_skipped_when_its_file_exists_and_redone_with_rerun(input_file, tmp_path, caplog):
    out = tmp_path / "out"
    run_main(input_file, out, 1)
    run_main(input_file, out, 2)
    target = out / "stage2" / "batches_df-1-of-1.pkl.gz"
    target.write_text("not a pickle")
    with caplog.at_level(logging.INFO):
        run_main(input_file, out, 2)
    assert "already exists, stopping" in caplog.text
    assert target.read_text() == "not a pickle"
    run_main(input_file, out, 2, "--rerun")
    assert len(pd.read_pickle(target)) == 1


def test_stage_3_is_skipped_when_its_folder_exists_and_removes_stage_1_on_request(input_file, tmp_path, caplog):
    out = tmp_path / "out"
    for stage in (1, 2):
        run_main(input_file, out, stage)
    run_main(input_file, out, 3, "--remove_stage1")
    assert not (out / "stage1" / "shard_1-of-1").exists()
    assert len(load_dataset(out / "stage3" / "shard_1-of-1")) == 1
    with caplog.at_level(logging.INFO):
        run_main(input_file, out, 3)
    assert "already exists, skipping" in caplog.text


def test_stage_3_keeps_stage_1_without_the_remove_option(pipeline):
    assert (pipeline / "stage1" / "shard_1-of-1").is_dir()


def test_stage_3_needs_the_batch_statistics_of_stage_2(input_file, tmp_path):
    out = tmp_path / "out"
    run_main(input_file, out, 1)
    with pytest.raises(RuntimeError, match="Stage 2 is not completed"):
        run_main(input_file, out, 3)


@pytest.mark.parametrize("strategy", [grace_preprocess.EQUI_ATOMS_STRATEGY, grace_preprocess.EQUI_NEIGH_STRATEGY])
def test_stage_2_strategies_split_by_atoms_or_neighbours(input_file, tmp_path, strategy):
    out = tmp_path / "out"
    run_main(input_file, out, 1)
    run_main(input_file, out, 2, "-b", "10", "-s", strategy)
    batches = pd.read_pickle(out / "stage2" / "batches_df-1-of-1.pkl.gz")
    assert batches["n_structures"].sum() == N_STRUCTURES
    assert batches["n_atoms"].sum() == 12


def test_stage_2_rejects_an_unknown_strategy(input_file, tmp_path):
    out = tmp_path / "out"
    run_main(input_file, out, 1)
    with pytest.raises(NotImplementedError, match="Strategy shuffled not implemented"):
        run_main(input_file, out, 2, "-s", "shuffled")


@pytest.mark.parametrize(
    ("elements", "expected"),
    [
        ("ALL", grace_preprocess.FULL_ELEMENTS),
        ("MP", grace_preprocess.ALEXANDRIA_ELEMENTS),
        ("Alexandria", grace_preprocess.ALEXANDRIA_ELEMENTS),
        ("OMOL_N", grace_preprocess.OMOL_N_ELEMENTS),
        ("Zn Cu Al", ["Al", "Cu", "Zn"]),
    ],
)
def test_stage_1_elements_presets_and_lists(input_file, tmp_path, elements, expected):
    out = tmp_path / "out"
    argv = [str(input_file), "-o", str(out), "-e", elements, "-c", str(CUTOFF), "--stage-1"]
    if "Cu" in expected:
        grace_preprocess.main(argv)
    else:
        # the parameters are stored first; the data has Cu, which the preset lacks, so the run then fails
        with pytest.raises(KeyError, match="Cu"):
            grace_preprocess.main(argv)
    params = json.loads((out / "params.json").read_text())
    assert list(params["elements_map"]) == list(expected)
    assert list(params["elements_map"].values()) == list(range(len(expected)))


def test_stage_1_without_elements_extracts_them_from_the_data_with_a_warning(input_file, tmp_path, caplog):
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO):
        grace_preprocess.main([str(input_file), "-o", str(out), "-c", str(CUTOFF), "--stage-1"])
    assert "Elements are not provided" in caplog.text
    assert json.loads((out / "params.json").read_text())["elements_map"] == {"Cu": 0}


def test_stage_1_reuses_the_elements_of_a_previous_run_when_none_are_given(input_file, tmp_path, caplog):
    out = tmp_path / "out"
    grace_preprocess.main([str(input_file), "-o", str(out), "-e", "Al Cu", "-c", str(CUTOFF), "--stage-1"])
    with caplog.at_level(logging.INFO):
        grace_preprocess.main([str(input_file), "-o", str(out), "--stage-1", "--rerun"])
    assert "Loaded elements: ['Al', 'Cu']" in caplog.text
    assert json.loads((out / "params.json").read_text())["elements_map"] == {"Al": 0, "Cu": 1}


def test_stage_2_and_3_use_the_parameters_stored_by_stage_1(input_file, tmp_path):
    out = tmp_path / "out"
    grace_preprocess.main([str(input_file), "-o", str(out), "-e", "Cu", "-c", "4.0", "--stage-1", "--precision", "float32"])
    params = json.loads((out / "params.json").read_text())
    assert params["precision"] == "float32"
    # stages 2 and 3 are started without -c or --precision: the stored cutoff and precision are used
    grace_preprocess.main([str(input_file), "-o", str(out), "--stage-2"])
    grace_preprocess.main([str(input_file), "-o", str(out), "--stage-3"])
    batch = load_dataset(out / "stage3" / "shard_1-of-1")[0]
    assert batch[constants.DATA_REFERENCE_FORCES].dtype == np.float32


def test_load_params_of_an_unparsable_file_logs_and_returns_nothing(tmp_path, caplog):
    (tmp_path / "params.json").write_text("{not json")
    with caplog.at_level(logging.ERROR):
        assert grace_preprocess.load_params(str(tmp_path)) is None
    assert "Can't parse" in caplog.text


def test_load_params_without_a_file_returns_nothing(tmp_path):
    assert grace_preprocess.load_params(str(tmp_path)) is None


def test_load_params_round_trips_save_params(tmp_path):
    grace_preprocess.save_params(str(tmp_path), {"Cu": 0, "Zn": 1}, 5.5, "float32", {"Cu": {"Zn": 4.0}})
    assert grace_preprocess.load_params(str(tmp_path)) == ({"Cu": 0, "Zn": 1}, 5.5, {"Cu": {"Zn": 4.0}}, "float32")


# --- several tasks -----------------------------------------------------------------------------------


def structure_ids(path: Path) -> list[int]:
    return sorted(int(s["structure_id"]) for s in load_dataset(path))


def test_rows_are_split_between_tasks_when_the_input_is_not_sharded(input_file, tmp_path):
    out = tmp_path / "out"
    for task in (0, 1):
        run_main(input_file, out, 1, "--task-id", str(task), "--total-task-num", "2")
    assert structure_ids(out / "stage1" / "shard_1-of-2") == [0, 2]
    assert structure_ids(out / "stage1" / "shard_2-of-2") == [1]


def test_files_are_split_between_tasks_when_the_input_is_sharded(dataframe, tmp_path):
    files = []
    for k in range(2):
        path = tmp_path / f"part{k}.pkl.gz"
        dataframe.iloc[[k]].to_pickle(path)
        files.append(path)
    out = tmp_path / "out"
    for task in (0, 1):
        run_main(files, out, 1, "--sharded-input", "--task-id", str(task), "--total-task-num", "2")
    assert structure_ids(out / "stage1" / "shard_1-of-2") == [0]
    assert structure_ids(out / "stage1" / "shard_2-of-2") == [1]


def test_stages_2_to_4_combine_the_shards_of_every_task(input_file, dataframe, tmp_path):
    out = tmp_path / "out"
    for stage in (1, 2):
        for task in (0, 1):
            run_main(input_file, out, stage, "--task-id", str(task), "--total-task-num", "2", "-b", "2", "-bu", "2")
    for task in (0, 1):
        run_main(input_file, out, 3, "--task-id", str(task), "--total-task-num", "2", "-b", "2", "-bu", "2")
    run_main(input_file, out, 4, "--task-id", "1", "--total-task-num", "2")
    assert not (out / "stage3" / "stats.json").exists()  # only task 0 aggregates
    run_main(input_file, out, 4, "--task-id", "0", "--total-task-num", "2")
    stats = json.loads((out / "stage3" / "stats.json").read_text())
    assert stats["total_num_structures"] == N_STRUCTURES
    assert stats["total_num_of_atoms"] == sum(len(a) for a in dataframe["ase_atoms"])


def test_an_unreadable_input_file_is_logged_and_skipped(input_file, dataframe, tmp_path, caplog):
    broken = tmp_path / "broken.pkl.gz"
    broken.write_text("not a pickle")
    out = tmp_path / "out"
    with caplog.at_level(logging.ERROR):
        run_main([broken, input_file], out, 1, "--sharded-input")
    assert "Error reading file" in caplog.text
    assert len(load_dataset(out / "stage1" / "shard_1-of-1")) == N_STRUCTURES
