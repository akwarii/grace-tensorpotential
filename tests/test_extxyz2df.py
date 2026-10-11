"""``tensorpotential.scripts.extxyz2df``: extxyz to dataframe (pkl.gz) conversion, driven through ``main``.

Logic layer: output naming, the optional output name, an extxyz without stress, a missing input. Physical values
layer: the input is written with ``ase.io.write`` from structures whose energy, forces and stress are known
numbers, and the dataframe read back with ``pandas`` holds exactly those values and the same geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from tensorpotential.scripts import extxyz2df
from tests.tolerances import EXTXYZ_TEXT_ROUNDTRIP as TOL

N_STRUCTURES = 3


def _structures(with_stress: bool = True):
    """Rattled Cu cells with a known energy ``-3.5 n``, forces ``0.1 (k + 1)`` and stress ``0.01 (k + 1)`` in Voigt order."""
    rng = np.random.default_rng(1)
    structures = []
    for k in range(N_STRUCTURES):
        atoms = bulk("Cu", "fcc", a=3.6 + 0.05 * k, cubic=True)
        atoms.rattle(stdev=0.03, rng=rng)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-3.5 * len(atoms) - k,
            forces=np.full((len(atoms), 3), 0.1 * (k + 1)),
            stress=np.arange(6.0) * 0.01 * (k + 1) if with_stress else None,
        )
        structures.append(atoms)
    return structures


@pytest.fixture
def extxyz(tmp_path) -> Path:
    path = tmp_path / "frames.xyz"
    write(path, _structures(), format="extxyz")
    return path


def test_default_output_name_replaces_the_xyz_extension(extxyz, tmp_path):
    extxyz2df.main([str(extxyz)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frames.pkl.gz", "frames.xyz"]


def test_explicit_output_name(extxyz, tmp_path):
    target = tmp_path / "chosen.pkl.gz"
    extxyz2df.main([str(extxyz), "--output-dataset-filename", str(target)])
    assert target.is_file()
    assert not (tmp_path / "frames.pkl.gz").exists()


def test_the_dataframe_holds_energy_forces_stress_and_geometry(extxyz, tmp_path):
    extxyz2df.main([str(extxyz)])
    df = pd.read_pickle(tmp_path / "frames.pkl.gz", compression="gzip")
    assert list(df.columns) == ["ase_atoms", "energy", "forces", "stress", "info"]
    assert len(df) == N_STRUCTURES
    for k, (_, row) in enumerate(df.iterrows()):
        reference = _structures()[k]
        assert row["ase_atoms"].get_chemical_symbols() == reference.get_chemical_symbols()
        np.testing.assert_allclose(row["ase_atoms"].positions, reference.positions, rtol=0.0, atol=TOL.atol)
        np.testing.assert_allclose(row["ase_atoms"].cell[:], reference.cell[:], rtol=0.0, atol=TOL.atol)
        np.testing.assert_allclose(row["energy"], reference.get_potential_energy(), rtol=0.0, atol=TOL.atol)
        np.testing.assert_allclose(row["forces"], np.full((len(reference), 3), 0.1 * (k + 1)), rtol=0.0, atol=TOL.atol)
        np.testing.assert_allclose(row["stress"], np.arange(6.0) * 0.01 * (k + 1), rtol=0.0, atol=TOL.atol)


def test_a_file_without_stress_gives_a_dataframe_without_the_stress_column(tmp_path):
    path = tmp_path / "nostress.xyz"
    write(path, _structures(with_stress=False), format="extxyz")
    extxyz2df.main([str(path)])
    df = pd.read_pickle(tmp_path / "nostress.pkl.gz", compression="gzip")
    assert "stress" not in df.columns
    assert {"ase_atoms", "energy", "forces"} <= set(df.columns)


def test_a_missing_input_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extxyz2df.main([str(tmp_path / "absent.xyz")])


def test_the_input_is_required(capsys):
    with pytest.raises(SystemExit) as stop:
        extxyz2df.main([])
    assert stop.value.code == 2
    assert "the following arguments are required: extxyz_filename" in capsys.readouterr().err
