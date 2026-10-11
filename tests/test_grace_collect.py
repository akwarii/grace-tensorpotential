"""``tensorpotential.scripts.grace_collect``: collect VASP calculations into a dataframe, driven through ``main``.

The inputs are OUTCAR files written by ``write_outcar`` below from chosen numbers (energies, forces, positions, stress
in kB), in the layout ASE's ``vasp-out`` reader needs. Logic layer: file discovery, the selection of steps, the
free-atom reference options, an unreadable calculation, the option parser. Physical values layer: the energy of the
dataframe is the TOTEN written in the file, the corrected energy is the energy minus the free-atom energies counted
by element (a hand computation), the stress is the VASP kB value converted to eV/A^3 with the VASP sign and
component order (xx yy zz xy yz zx to the Voigt order xx yy zz yz xz xy), and the forces are the file's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from tensorpotential.scripts import grace_collect
from tests.tolerances import FLOAT64_ARITHMETIC as EXACT
from tests.tolerances import HAND_UNIT_CONVERSION as UNITS

# 1 kB = 0.1 GPa; 1 GPa = 1 / 160.21766208 eV/A^3.
EV_PER_A3_PER_KB = 0.1 / 160.21766208


def write_outcar(
    path: Path,
    species: list[str],
    steps: list[dict],
    cell_length: float = 4.0,
) -> None:
    """Write a minimal OUTCAR (header, one block per step) that ``ase.io.read(..., format="vasp-out")`` reads.

    ``species`` lists one symbol per atom, grouped by element; a step is ``{"positions", "forces", "energy",
    "stress_kb"}`` with the free energy ``energy`` in eV and the six numbers of the ``in kB`` line.
    """
    elements = list(dict.fromkeys(species))
    counts = [species.count(e) for e in elements]
    lines = [" vasp.6.0.0 synthetic OUTCAR for tests", ""]
    # a real OUTCAR lists the whole set of potentials twice
    lines += [f" POTCAR:    PAW_PBE {e} 01Jan2000" for _ in range(2) for e in elements]
    lines += ["   ISPIN  =      1    spin polarized calculation?"]
    lines += ["   ions per type =              " + "  ".join(str(c) for c in counts)]
    lines += [
        " ----------------------------------------- Iteration    1(   1)  -----------------"
    ]
    for step in steps:
        lines += [
            "      direct lattice vectors                 reciprocal lattice vectors"
        ]
        for i in range(3):
            row = [cell_length if i == j else 0.0 for j in range(3)]
            lines += ["  " + " ".join(f"{x:14.9f}" for x in row) + "   0.0 0.0 0.0"]
        lines += ["", "  FORCE on cell =-STRESS in cart. coord.  units (eV):"]
        lines += ["  in kB   " + " ".join(f"{x:11.5f}" for x in step["stress_kb"])]
        lines += [
            " POSITION                                       TOTAL-FORCE (eV/Angst)",
            " " + "-" * 70,
        ]
        for pos, force in zip(step["positions"], step["forces"], strict=True):
            lines += [" ".join(f"{x:13.8f}" for x in (*pos, *force))]
        lines += [" " + "-" * 70, "    total drift:  0.0 0.0 0.0", ""]
        lines += ["  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)", "  " + "-" * 50]
        lines += [f"  free  energy   TOTEN  =  {step['energy']:16.8f} eV", ""]
        lines += [
            f"  energy  without entropy= {step['energy']:16.8f}  energy(sigma->0) = {step['energy']:16.8f}"
        ]
        lines += [
            " ----------------------------------------- Iteration    2(   1)  -----------------"
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def make_step(n_atoms: int, energy: float, k: int = 0) -> dict:
    """A step with distinct, known numbers: forces ``0.1 (k + 1) (i + 1)``, stress ``(k + 1) (1..6)`` kB."""
    return {
        "positions": [[0.5 * i, 0.25 * i + 0.1 * k, 0.1 * i] for i in range(n_atoms)],
        "forces": [
            [0.1 * (k + 1) * (i + 1), -0.05 * (i + 1), 0.01 * (k + i)]
            for i in range(n_atoms)
        ],
        "energy": energy,
        "stress_kb": [(k + 1) * v for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)],
    }


def run_collect(tmp_path: Path, working_dir: Path, *options: str) -> pd.DataFrame:
    """Run ``main`` on ``working_dir`` and return the stored dataframe."""
    output = tmp_path / "collected.pkl.gz"
    grace_collect.main([
        "-wd",
        str(working_dir),
        "--output-dataset-filename",
        str(output),
        *options,
    ])
    return pd.read_pickle(output, compression="gzip")


@pytest.fixture
def calc_dir(tmp_path) -> Path:
    """Two directories: ``a`` holds Cu2 (E=-6.0) and ``b/c`` holds Cu1Ni1 (E=-7.5)."""
    root = tmp_path / "calcs"
    write_outcar(root / "a" / "OUTCAR", ["Cu", "Cu"], [make_step(2, -6.0)])
    write_outcar(root / "b" / "c" / "OUTCAR", ["Cu", "Ni"], [make_step(2, -7.5, k=1)])
    return root


def test_the_energy_forces_and_geometry_of_each_calculation_are_collected(
    calc_dir, tmp_path
):
    df = run_collect(tmp_path, calc_dir).sort_values("name").reset_index(drop=True)
    assert len(df) == 2
    expected = [
        (["Cu", "Cu"], -6.0, make_step(2, -6.0)),
        (["Cu", "Ni"], -7.5, make_step(2, -7.5, k=1)),
    ]
    for (_, row), (symbols, energy, step) in zip(df.iterrows(), expected, strict=True):
        assert row["ase_atoms"].get_chemical_symbols() == symbols
        np.testing.assert_allclose(
            row["energy"], energy, rtol=EXACT.rtol, atol=EXACT.atol
        )
        np.testing.assert_allclose(
            row["forces"], step["forces"], rtol=EXACT.rtol, atol=EXACT.atol
        )
        np.testing.assert_allclose(
            row["ase_atoms"].positions,
            step["positions"],
            rtol=EXACT.rtol,
            atol=EXACT.atol,
        )
        np.testing.assert_allclose(
            row["ase_atoms"].cell.lengths(), [4.0] * 3, rtol=EXACT.rtol
        )
        assert all(row["ase_atoms"].pbc)


def test_the_stress_is_converted_from_kb_with_the_vasp_sign_and_component_order(
    calc_dir, tmp_path
):
    df = run_collect(tmp_path, calc_dir).sort_values("name").reset_index(drop=True)
    for k, stress in enumerate(df["stress"]):
        kb = (k + 1) * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = -kb[[0, 1, 2, 4, 5, 3]] * EV_PER_A3_PER_KB
        np.testing.assert_allclose(stress, expected, rtol=UNITS.rtol, atol=UNITS.atol)


def test_the_name_is_the_directory_the_file_and_the_step(calc_dir, tmp_path):
    df = run_collect(tmp_path, calc_dir).sort_values("name")
    assert list(df["name"]) == [f"{calc_dir}/a/OUTCAR##0", f"{calc_dir}/b/c/OUTCAR##0"]


def test_the_dataframe_columns(calc_dir, tmp_path):
    df = run_collect(tmp_path, calc_dir)
    assert list(df.columns) == [
        "name",
        "ase_atoms",
        "energy",
        "forces",
        "stress",
        "results",
        "energy_corrected",
        "energy_corrected_per_atom",
    ]


def test_without_free_atom_energies_the_corrected_energy_is_the_energy(
    calc_dir, tmp_path
):
    df = run_collect(tmp_path, calc_dir)
    np.testing.assert_allclose(
        df["energy_corrected"], df["energy"], rtol=EXACT.rtol, atol=EXACT.atol
    )
    np.testing.assert_allclose(
        df["energy_corrected_per_atom"],
        df["energy"] / 2,
        rtol=EXACT.rtol,
        atol=EXACT.atol,
    )


def test_free_atom_energies_are_subtracted_per_element(calc_dir, tmp_path):
    df = run_collect(
        tmp_path, calc_dir, "--free-atom-energy", "Cu:-0.5", "Ni:-1.25"
    ).sort_values("name")
    # by hand: Cu2 gives -6.0 + 2 * 0.5 = -5.0 and Cu1Ni1 gives -7.5 + 0.5 + 1.25 = -5.75
    np.testing.assert_allclose(
        df["energy_corrected"], [-5.0, -5.75], rtol=EXACT.rtol, atol=EXACT.atol
    )
    np.testing.assert_allclose(
        df["energy_corrected_per_atom"],
        [-2.5, -2.875],
        rtol=EXACT.rtol,
        atol=EXACT.atol,
    )


def test_without_the_option_every_element_counts_zero_but_a_partial_list_does_not(
    calc_dir, tmp_path
):
    # The help says "default is zero", which holds for the default (a defaultdict) only: once the option is given, the
    # parsed value is a plain dict and an element missing from it raises KeyError (current behaviour, see the CLI1 issue).
    with pytest.raises(KeyError, match="Ni"):
        run_collect(tmp_path, calc_dir, "--free-atom-energy", "Cu:-0.5")


@pytest.fixture
def calc_dir_with_free_atoms(calc_dir) -> Path:
    """``calc_dir`` plus a single Cu atom (E=-0.2) and a single Ni atom (E=-0.9) in a 10 A cell (V = 1000 A^3)."""
    write_outcar(
        calc_dir / "free_cu" / "OUTCAR", ["Cu"], [make_step(1, -0.2)], cell_length=10.0
    )
    write_outcar(
        calc_dir / "free_ni" / "OUTCAR", ["Ni"], [make_step(1, -0.9)], cell_length=10.0
    )
    return calc_dir


@pytest.mark.parametrize("options", [["auto"], ["Cu:auto", "Ni:auto"]])
def test_auto_takes_the_energy_of_the_isolated_atoms_of_the_dataset(
    calc_dir_with_free_atoms, tmp_path, options
):
    df = run_collect(tmp_path, calc_dir_with_free_atoms, "--free-atom-energy", *options)
    corrected = dict(
        zip(
            df["name"].map(lambda n: Path(n.split("##")[0]).parent.name),
            df["energy_corrected"],
            strict=True,
        )
    )
    np.testing.assert_allclose(
        corrected["a"], -6.0 + 2 * 0.2, rtol=EXACT.rtol, atol=EXACT.atol
    )
    np.testing.assert_allclose(
        corrected["c"], -7.5 + 0.2 + 0.9, rtol=EXACT.rtol, atol=EXACT.atol
    )
    np.testing.assert_allclose(corrected["free_cu"], 0.0, atol=EXACT.atol)
    np.testing.assert_allclose(corrected["free_ni"], 0.0, atol=EXACT.atol)


def test_auto_for_one_element_and_a_number_for_the_other(
    calc_dir_with_free_atoms, tmp_path
):
    df = run_collect(
        tmp_path, calc_dir_with_free_atoms, "--free-atom-energy", "Cu:auto", "Ni:-1.0"
    )
    corrected = dict(
        zip(
            df["name"].map(lambda n: Path(n.split("##")[0]).parent.name),
            df["energy_corrected"],
            strict=True,
        )
    )
    np.testing.assert_allclose(
        corrected["c"], -7.5 + 0.2 + 1.0, rtol=EXACT.rtol, atol=EXACT.atol
    )


def test_auto_without_an_isolated_atom_warns_and_uses_zero(calc_dir, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        df = run_collect(tmp_path, calc_dir, "--free-atom-energy", "auto")
    assert "No reference free atom energy found for Cu" in caplog.text
    np.testing.assert_allclose(
        df["energy_corrected"], df["energy"], rtol=EXACT.rtol, atol=EXACT.atol
    )


def test_the_free_atom_options_and_the_subtracted_energies_are_logged(
    calc_dir, tmp_path, caplog
):
    with caplog.at_level(logging.INFO):
        run_collect(tmp_path, calc_dir, "--free-atom-energy", "Cu:-0.5", "Ni:-1.0")
    assert "Free atomic energy options: Cu:-0.5, Ni:-1.0" in caplog.text
    assert (
        "Following atomic reference energies will be subtracted: Cu:-0.5, Ni:-1.0"
        in caplog.text
    )


@pytest.fixture
def relaxation(tmp_path) -> Path:
    """One OUTCAR with three ionic steps of a Cu2 cell, energies -1, -2, -3."""
    root = tmp_path / "relax"
    write_outcar(
        root / "OUTCAR",
        ["Cu", "Cu"],
        [make_step(2, -1.0 * (k + 1), k=k) for k in range(3)],
    )
    return root


@pytest.mark.parametrize(
    ("selection", "energies"),
    [
        (None, [-3.0]),
        ("last", [-3.0]),
        ("first", [-1.0]),
        ("all", [-1.0, -2.0, -3.0]),
        ("first_and_last", [-1.0, -3.0]),
        ("1:", [-2.0, -3.0]),
        (":2", [-1.0, -2.0]),
        ("::-1", [-3.0, -2.0, -1.0]),
    ],
)
def test_selection_of_the_steps_of_one_calculation(
    relaxation, tmp_path, selection, energies
):
    options = [] if selection is None else ["--selection", selection]
    df = run_collect(tmp_path, relaxation, *options)
    assert sorted(df["energy"]) == pytest.approx(
        sorted(energies), rel=EXACT.rtol, abs=EXACT.atol
    )
    assert sorted(n.split("##")[1] for n in df["name"]) == sorted(
        str(int(-e) - 1) for e in energies
    )


def test_an_invalid_selection_is_an_error(relaxation, tmp_path):
    with pytest.raises(
        grace_collect.InvalidArgumentError,
        match="select_steps option `middle` is not valid",
    ):
        run_collect(tmp_path, relaxation, "--selection", "middle")


def test_a_single_file_can_be_the_working_directory(calc_dir, tmp_path):
    df = run_collect(tmp_path, calc_dir / "a" / "OUTCAR")
    assert len(df) == 1
    assert df["name"].iloc[0] == f"{calc_dir / 'a'}/OUTCAR##0"


def test_a_calculation_that_cannot_be_read_is_logged_and_skipped(
    calc_dir, tmp_path, caplog
):
    (calc_dir / "broken").mkdir()
    (calc_dir / "broken" / "OUTCAR").write_text("this is not an OUTCAR\n")
    with caplog.at_level(logging.ERROR):
        df = run_collect(tmp_path, calc_dir)
    assert len(df) == 2
    assert "Filename could not be read" in caplog.text


def test_modified_file_names_are_collected_when_there_is_no_plain_outcar(tmp_path):
    root = tmp_path / "calcs" / "run"
    write_outcar(root / "OUTCAR_000", ["Cu", "Cu"], [make_step(2, -6.0)])
    write_outcar(root / "OUTCAR.relax", ["Cu", "Cu"], [make_step(2, -5.0)])
    df = run_collect(tmp_path, tmp_path / "calcs")
    assert sorted(df["energy"]) == pytest.approx([-6.0, -5.0])
    assert sorted(Path(n.split("##")[0]).name for n in df["name"]) == [
        "OUTCAR.relax",
        "OUTCAR_000",
    ]


def test_a_plain_outcar_takes_precedence_over_modified_names(tmp_path):
    root = tmp_path / "calcs" / "run"
    write_outcar(root / "OUTCAR", ["Cu", "Cu"], [make_step(2, -6.0)])
    write_outcar(root / "OUTCAR_000", ["Cu", "Cu"], [make_step(2, -5.0)])
    df = run_collect(tmp_path, tmp_path / "calcs")
    assert list(df["energy"]) == pytest.approx([-6.0])


def test_vasprun_xml_is_tried_when_there_is_no_outcar(tmp_path, caplog):
    root = tmp_path / "calcs"
    root.mkdir()
    (root / "vasprun.xml").write_text("not xml\n")
    write_outcar(
        tmp_path / "calcs" / "other" / "OUTCAR", ["Cu", "Cu"], [make_step(2, -6.0)]
    )
    with caplog.at_level(logging.ERROR):
        df = run_collect(tmp_path, root)
    assert len(df) == 1
    assert "Filename could not be read" in caplog.text


def test_modified_vasprun_names_are_tried(tmp_path, caplog):
    root = tmp_path / "calcs" / "run"
    root.mkdir(parents=True)
    (root / "vasprun.xml.relax").write_text("not xml\n")
    write_outcar(
        tmp_path / "calcs" / "other" / "OUTCAR", ["Cu", "Cu"], [make_step(2, -6.0)]
    )
    with caplog.at_level(logging.ERROR):
        df = run_collect(tmp_path, tmp_path / "calcs")
    assert len(df) == 1
    assert "Filename could not be read" in caplog.text


def test_default_output_filename_is_collected_pkl_gz_in_the_current_directory(
    calc_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    grace_collect.main(["-wd", str(calc_dir)])
    assert (tmp_path / "collected.pkl.gz").is_file()


def test_default_working_directory_is_the_current_directory(
    calc_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(calc_dir)
    output = tmp_path / "out.pkl.gz"
    grace_collect.main(["--output-dataset-filename", str(output)])
    assert len(pd.read_pickle(output, compression="gzip")) == 2


# --- helpers ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "n", "expected"),
    [
        ("last", 4, [3]),
        ("first", 4, [0]),
        ("all", 4, [0, 1, 2, 3]),
        ("first_and_last", 4, [0, 3]),
        ("first_and_last", 1, [0]),
        ("1:3", 5, [1, 2]),
        ("::2", 5, [0, 2, 4]),
        (":", 3, [0, 1, 2]),
        ("::-1", 3, [2, 1, 0]),
        ("-2:", 5, [3, 4]),
    ],
)
def test_generate_selected_indices(selection, n, expected):
    assert list(grace_collect.generate_selected_indices(selection, n)) == expected


def test_generate_selected_indices_rejects_an_unknown_word():
    with pytest.raises(grace_collect.InvalidArgumentError, match="`everything`"):
        grace_collect.generate_selected_indices("everything", 3)


def test_parse_kwargs_reads_numbers_words_and_auto():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--e", nargs="*", action=grace_collect.ParseKwargs, dest="e")
    parsed = parser.parse_args(["--e", "Al:-0.123", "Cu:auto", "Zn:abc", "auto"])
    assert parsed.e == {"Al": -0.123, "Cu": "auto", "Zn": "abc", "auto": True}


def test_get_safe_volume_is_zero_unless_periodic_in_all_directions():
    periodic = Atoms("Cu", cell=[2.0, 3.0, 4.0], pbc=True)
    assert grace_collect.get_safe_volume(periodic) == pytest.approx(24.0)
    assert (
        grace_collect.get_safe_volume(
            Atoms("Cu", cell=[2.0, 3.0, 4.0], pbc=[True, True, False])
        )
        == 0
    )


def test_walk_file_or_dir_without_a_path_is_an_error():
    with pytest.raises(ValueError, match="Root directory not provided"):
        next(grace_collect.walk_file_or_dir(None))


def test_walk_file_or_dir_on_a_file_yields_its_directory_and_name(tmp_path):
    target = tmp_path / "OUTCAR"
    target.write_text("x")
    assert list(grace_collect.walk_file_or_dir(str(target))) == [
        (str(tmp_path), [], ["OUTCAR"])
    ]


def test_get_free_atom_energy_picks_the_most_isolated_single_atom():
    df = pd.DataFrame({
        "n_Cu": [1, 1, 2],
        "NUMBER_OF_ATOMS": [1, 1, 2],
        "volume_per_atom": [600.0, 900.0, 1000.0],
        "energy": [-0.3, -0.2, -6.0],
        "name": ["small_box", "big_box", "bulk"],
    })
    assert grace_collect.get_free_atom_energy(df, "Cu") == pytest.approx(-0.2)
    assert grace_collect.get_free_atom_energy(df.iloc[[2]], "Cu") == 0
