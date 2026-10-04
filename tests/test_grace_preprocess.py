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
