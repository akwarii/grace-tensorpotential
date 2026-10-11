"""``tensorpotential.scripts.df2extxyz``: dataframe (pkl.gz) to extxyz conversion, driven through ``main``.

Logic layer: output naming, the column options, a missing stress column, a missing column. Physical values layer:
the energies, forces, stresses, cells and positions of the written file, read back with ``ase.io.read``, equal the
dataframe's (an oracle that does not call the converter); the stress is passed through without a unit conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.io import read

from tensorpotential.scripts import df2extxyz
from tests.tolerances import EXTXYZ_TEXT_ROUNDTRIP as TOL

DATA = Path(__file__).parent / "data" / "MoNbTaW_train50.pkl.gz"
N_ROWS = 3


@pytest.fixture
def frame() -> pd.DataFrame:
    """The first rows of the MoNbTaW training set: energy_corrected, forces and stress columns."""
    df = pd.read_pickle(DATA)
    return (
        df[["ase_atoms", "energy_corrected", "forces", "stress"]]
        .head(N_ROWS)
        .reset_index(drop=True)
    )


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_pickle(path, compression="gzip")
    return path


def _assert_matches(
    path: Path, frame: pd.DataFrame, energy: str, forces: str, stress: str | None
) -> None:
    written = read(path, format="extxyz", index=":")
    assert len(written) == len(frame)
    for atoms, (_, row) in zip(written, frame.iterrows(), strict=True):
        reference = row["ase_atoms"]
        assert atoms.get_chemical_symbols() == reference.get_chemical_symbols()
        assert tuple(atoms.pbc) == tuple(reference.pbc)
        np.testing.assert_allclose(
            atoms.cell[:], reference.cell[:], rtol=0.0, atol=TOL.atol
        )
        np.testing.assert_allclose(
            atoms.positions, reference.positions, rtol=0.0, atol=TOL.atol
        )
        np.testing.assert_allclose(
            atoms.get_potential_energy(), row[energy], rtol=TOL.rtol, atol=TOL.atol
        )
        np.testing.assert_allclose(
            atoms.get_forces(), row[forces], rtol=TOL.rtol, atol=TOL.atol
        )
        if stress is not None:
            np.testing.assert_allclose(
                atoms.get_stress(), row[stress], rtol=TOL.rtol, atol=TOL.atol
            )


def test_default_output_name_replaces_the_pickle_extension(frame, tmp_path):
    source = _write(frame, tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source)])
    assert (tmp_path / "train.xyz").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.pkl.gz", "train.xyz"]


@pytest.mark.parametrize("name", ["train.pkl.gzip", "train.pckl.gzip"])
def test_default_output_name_for_the_other_pickle_extensions(frame, tmp_path, name):
    source = _write(frame, tmp_path / name)
    df2extxyz.main([str(source)])
    assert (tmp_path / "train.xyz").is_file()


def test_explicit_output_name(frame, tmp_path):
    source = _write(frame, tmp_path / "train.pkl.gz")
    target = tmp_path / "elsewhere.extxyz"
    df2extxyz.main([str(source), "-o", str(target)])
    assert target.is_file()
    assert not (tmp_path / "train.xyz").exists()


def test_default_columns_are_energy_corrected_forces_and_stress(frame, tmp_path):
    source = _write(frame, tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source)])
    _assert_matches(
        tmp_path / "train.xyz", frame, "energy_corrected", "forces", "stress"
    )


def test_the_reference_values_are_also_stored_in_info_and_arrays(frame, tmp_path):
    source = _write(frame, tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source)])
    for atoms, (_, row) in zip(
        read(tmp_path / "train.xyz", index=":"), frame.iterrows(), strict=True
    ):
        np.testing.assert_allclose(
            atoms.info["REF_energy"], row["energy_corrected"], rtol=0.0, atol=TOL.atol
        )
        np.testing.assert_allclose(
            atoms.arrays["REF_forces"], row["forces"], rtol=0.0, atol=TOL.atol
        )
        np.testing.assert_allclose(
            atoms.info["REF_stress"], row["stress"], rtol=0.0, atol=TOL.atol
        )


def test_column_options_select_other_columns(frame, tmp_path):
    frame = frame.rename(
        columns={"energy_corrected": "e_ref", "forces": "f_ref", "stress": "s_ref"}
    )
    source = _write(frame, tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source), "-e", "e_ref", "-f", "f_ref", "-s", "s_ref"])
    _assert_matches(tmp_path / "train.xyz", frame, "e_ref", "f_ref", "s_ref")
    df2extxyz.main([
        str(source),
        "--energy-column",
        "e_ref",
        "--force-column",
        "f_ref",
        "--stress-column",
        "s_ref",
        "-o",
        str(tmp_path / "long.xyz"),
    ])
    _assert_matches(tmp_path / "long.xyz", frame, "e_ref", "f_ref", "s_ref")


def test_without_a_stress_column_no_stress_is_written(frame, tmp_path):
    source = _write(frame.drop(columns="stress"), tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source)])
    written = read(tmp_path / "train.xyz", index=":")
    assert all("REF_stress" not in atoms.info for atoms in written)
    _assert_matches(tmp_path / "train.xyz", frame, "energy_corrected", "forces", None)


def test_the_stress_is_passed_through_without_a_unit_conversion(frame, tmp_path):
    scaled = frame.copy()
    scaled["stress"] = [np.arange(6.0) * 0.5 + i for i in range(N_ROWS)]
    source = _write(scaled, tmp_path / "train.pkl.gz")
    df2extxyz.main([str(source)])
    for i, atoms in enumerate(read(tmp_path / "train.xyz", index=":")):
        np.testing.assert_allclose(
            atoms.get_stress(), np.arange(6.0) * 0.5 + i, rtol=0.0, atol=TOL.atol
        )


def test_a_missing_energy_column_is_an_error(frame, tmp_path):
    source = _write(frame.drop(columns="energy_corrected"), tmp_path / "train.pkl.gz")
    with pytest.raises(KeyError, match="energy_corrected"):
        df2extxyz.main([str(source)])


def test_a_missing_input_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        df2extxyz.main([str(tmp_path / "absent.pkl.gz")])


def test_the_input_is_required(capsys):
    with pytest.raises(SystemExit) as stop:
        df2extxyz.main([])
    assert stop.value.code == 2
    assert "the following arguments are required: input" in capsys.readouterr().err


def test_build_parser_defaults():
    args = df2extxyz.build_parser().parse_args(["in.pkl.gz"])
    assert (args.energy_column, args.force_column, args.stress_column, args.output) == (
        "energy_corrected",
        "forces",
        "stress",
        None,
    )
