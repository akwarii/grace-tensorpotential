"""Tests of the fixture structure set (``tests_torch/structures/build_structures.py``).

Logic: the two files are what the generator writes, name and purpose are present, the formats agree,
the module needs no TensorFlow. Physical values: every feature tag and every purpose sentence is
checked against the brute-force neighbour oracle (``tests/neighbour_oracle.py``, numpy only) and
against numbers counted by hand (shell counts of fcc, the cell of a lone atom, bond lengths and
charge balance of the mineral cell).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ase import Atoms

from tests.fresh_python import run_fresh_python
from tests.neighbour_oracle import Pair, brute_force_pairs
from tests.tolerances import STORED_COORDINATE_DISTANCE as STORED
from tests.tolerances import sample_std_rtol
from tests_torch.structures import build_structures as bs

EXPECTED_NAMES = (
    "dimer",
    "self_image_cell",
    "fcc4",
    "rattled_multi8",
    "slab",
    "isolated_atom",
    "icosahedron_satellite",
    "mineral",
)
EXPECTED_N_ATOMS = (2, 2, 4, 8, 12, 1, 14, 29)
KNOWN_FEATURES = frozenset({
    "aperiodic",
    "partial_pbc",
    "self_image_pairs",
    "isolated_atom",
    "unequal_neighbour_counts",
    "multi_element",
    "four_elements",
    "rattled",
    "vacuum_gap_below_cutoff",
})
A_CU = 3.61
"""Lattice constant of fcc Cu stated in the purpose text (a literal, not the generator's constant)."""
O_H_BOND = 0.97
"""Hydroxyl O-H distance stated in the purpose text."""
H_CLEARANCE = 2.0
"""No atom except its oxygen is closer than this to a hydroxyl H (a hydrogen pointing at a cation would be)."""
SATELLITE_GAP = 3.6
"""Stated in the purpose text: the extra atom sits 3.6 A beyond the outer radius."""
PAIR_DISTANCE_FLOOR = 0.9
HEAVY_PAIR_DISTANCE_FLOOR = 1.5
FCC_SHELL_COUNTS = {3.0: 12, 4.0: 18, 5.0: 42}
"""Neighbours of every atom of fcc within a cutoff (a = 3.61): 12 at a/sqrt(2) = 2.55 A, 6 more at a = 3.61 A, 24 more at a*sqrt(1.5) = 4.42 A."""
MINERAL_FORMAL_CHARGE = {"Mg": 2, "Si": 4, "O": -2, "H": 1}
SI_O_RANGE = (1.55, 1.75)
MG_O_RANGE = (1.95, 2.35)
O_H_PAIR_MAX = 1.2


@pytest.fixture(scope="module")
def structures() -> dict[str, Atoms]:
    return bs.load_structures()


def _cell_for_oracle(atoms: Atoms) -> np.ndarray:
    """The cell, or a unit cube for an aperiodic structure without one (the oracle needs a non-singular matrix)."""
    return np.array(atoms.cell) if atoms.cell.rank == 3 else np.eye(3)


def _pairs(atoms: Atoms, cutoff: float, *, honour_pbc: bool = True) -> list[Pair]:
    periodic = tuple(bool(f) for f in atoms.pbc) if honour_pbc else (True, True, True)
    return brute_force_pairs(
        atoms.positions,
        _cell_for_oracle(atoms),
        atoms.get_chemical_symbols(),
        cutoff,
        periodic,
    )


def _counts(atoms: Atoms, pairs: list[Pair]) -> np.ndarray:
    return np.bincount([p.i for p in pairs], minlength=len(atoms))


def _has_self_image(pairs: list[Pair]) -> bool:
    return any(p.i == p.j and p.shift != (0, 0, 0) for p in pairs)


def _computed_features(atoms: Atoms) -> set[str]:
    """The neighbour- and species-based tags that ``atoms`` has at ``PROBE_CUTOFF`` (all but ``rattled``)."""
    pairs = _pairs(atoms, bs.PROBE_CUTOFF)
    counts = _counts(atoms, pairs)
    found = set()
    npbc = int(np.sum(atoms.pbc))
    if npbc == 0:
        found.add("aperiodic")
    if 0 < npbc < 3:
        found.add("partial_pbc")
        if len(_pairs(atoms, bs.PROBE_CUTOFF, honour_pbc=False)) > len(pairs):
            found.add("vacuum_gap_below_cutoff")
    if _has_self_image(pairs):
        found.add("self_image_pairs")
    if counts.min() == 0:
        found.add("isolated_atom")
    if len(set(counts.tolist())) > 1:
        found.add("unequal_neighbour_counts")
    species = set(atoms.get_chemical_symbols())
    if len(species) >= 3:
        found.add("multi_element")
    if species == {"Mg", "Si", "O", "H"}:
        found.add("four_elements")
    return found


# ---------------------------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------------------------


def test_the_set_has_the_expected_structures_in_order(structures):
    assert tuple(structures) == EXPECTED_NAMES
    assert tuple(len(a) for a in structures.values()) == EXPECTED_N_ATOMS
    assert tuple(spec.name for spec in bs.SPECS) == EXPECTED_NAMES


def test_every_structure_states_a_purpose(structures):
    for name, atoms in structures.items():
        purpose = atoms.info["purpose"]
        assert len(purpose) > 80, name
        assert purpose.endswith("."), name
    assert len({a.info["purpose"] for a in structures.values()}) == len(structures)


def test_feature_tags_are_from_the_vocabulary():
    for spec in bs.SPECS:
        assert set(spec.features) <= KNOWN_FEATURES, spec.name
        assert len(set(spec.features)) == len(spec.features), spec.name


def test_committed_files_are_what_the_generator_writes(tmp_path):
    bs.write_files(tmp_path)
    for path in (bs.JSON_PATH, bs.XYZ_PATH):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes(), path.name


def test_build_is_deterministic():
    first, second = bs.build_all(), bs.build_all()
    for name in first:
        assert np.array_equal(first[name].positions, second[name].positions), name
        assert np.array_equal(first[name].cell.array, second[name].cell.array), name


def test_json_loads_back_the_built_structures(structures):
    built = bs.build_all()
    for name, atoms in structures.items():
        assert atoms.get_chemical_symbols() == built[name].get_chemical_symbols(), name
        assert np.array_equal(atoms.positions, built[name].positions), name
        assert np.array_equal(atoms.cell.array, built[name].cell.array), name
        assert atoms.pbc.tolist() == built[name].pbc.tolist(), name
        assert atoms.info["purpose"] == built[name].info["purpose"], name


def test_json_and_xyz_hold_the_same_numbers(structures):
    from_xyz = bs.load_xyz()
    assert list(from_xyz) == list(structures)
    for name, atoms in structures.items():
        other = from_xyz[name]
        assert other.get_chemical_symbols() == atoms.get_chemical_symbols(), name
        assert np.array_equal(other.positions, atoms.positions), name
        assert np.array_equal(other.cell.array, atoms.cell.array), name
        assert other.pbc.tolist() == atoms.pbc.tolist(), name
        assert other.info["purpose"] == atoms.info["purpose"], name


def test_stored_numbers_are_their_own_eight_decimal_text(structures):
    for name, atoms in structures.items():
        for value in itertools.chain(atoms.positions.ravel(), atoms.cell.array.ravel()):
            assert float(f"{value:.{bs.DECIMALS}f}") == value, (name, value)


def test_record_counts_and_formula_match_the_structure():
    import json

    document = json.loads(bs.JSON_PATH.read_text(encoding="utf-8"))
    assert document["probe_cutoff"] == bs.PROBE_CUTOFF
    loaded = bs.load_structures()
    for record in document["structures"]:
        atoms = loaded[record["name"]]
        assert record["n_atoms"] == len(atoms) == len(record["positions"])
        assert record["formula"] == atoms.get_chemical_formula()


def test_the_issue_roles_are_all_present():
    tagged = {f: [s.name for s in bs.SPECS if f in s.features] for f in KNOWN_FEATURES}
    assert tagged["aperiodic"] and tagged["partial_pbc"] and tagged["isolated_atom"]
    assert tagged["self_image_pairs"] and tagged["unequal_neighbour_counts"]
    assert len(tagged["multi_element"]) >= 2
    assert tagged["four_elements"] == ["mineral"]


def test_module_needs_neither_tensorflow_nor_tensorpotential(tmp_path):
    code = (
        "import sys\n"
        "import tests_torch.structures.build_structures\n"
        "bad = [m for m in ('tensorflow', 'torch', 'tensorpotential') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------------------
# physical values: oracle and hand counts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_feature_tags_are_exactly_the_oracle_findings(structures, name):
    tagged = set(next(s for s in bs.SPECS if s.name == name).features) - {"rattled"}
    assert tagged == _computed_features(structures[name])


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_no_two_atoms_overlap(structures, name):
    atoms = structures[name]
    symbols = atoms.get_chemical_symbols()
    for p in _pairs(atoms, 3.0):
        length = float(np.linalg.norm(p.vector))
        assert length >= PAIR_DISTANCE_FLOOR, (name, p.i, p.j, length)
        if "H" not in (symbols[p.i], symbols[p.j]):
            assert length >= HEAVY_PAIR_DISTANCE_FLOOR, (name, p.i, p.j, length)


def test_dimer_is_one_bond_in_both_directions(structures):
    atoms = structures["dimer"]
    pairs = _pairs(atoms, bs.PROBE_CUTOFF)
    assert [(p.i, p.j) for p in pairs] == [(0, 1), (1, 0)]
    assert not atoms.pbc.any()
    for p in pairs:
        assert float(np.linalg.norm(p.vector)) == pytest.approx(2.5, abs=STORED.atol)
    assert np.allclose(pairs[0].vector, -pairs[1].vector, rtol=0.0, atol=STORED.atol)


@pytest.mark.parametrize(("cutoff", "count"), sorted(FCC_SHELL_COUNTS.items()))
def test_fcc4_has_the_exact_coordination_shells(structures, cutoff, count):
    atoms = structures["fcc4"]
    assert _counts(atoms, _pairs(atoms, cutoff)).tolist() == [count] * 4


def test_fcc4_is_the_conventional_cell_of_lattice_constant_361(structures):
    atoms = structures["fcc4"]
    assert np.allclose(atoms.cell.array, A_CU * np.eye(3), rtol=0.0, atol=STORED.atol)
    expected = A_CU * np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ])
    ours = atoms.positions[np.lexsort(atoms.positions.T)]
    theirs = expected[np.lexsort(expected.T)]
    assert np.allclose(ours, theirs, rtol=0.0, atol=STORED.atol)


def test_self_image_cell_has_the_shortest_lattice_vector_as_its_only_self_image(
    structures,
):
    atoms = structures["self_image_cell"]
    a1 = atoms.cell.array[0]
    shortest = float(np.linalg.norm(a1))
    assert (
        shortest
        < float(np.linalg.norm(atoms.cell.array[1]))
        < float(np.linalg.norm(atoms.cell.array[2]))
    )
    pairs = [p for p in _pairs(atoms, shortest + 0.05) if p.i == p.j]
    assert sorted((p.i, p.shift) for p in pairs) == [
        (0, (-1, 0, 0)),
        (0, (1, 0, 0)),
        (1, (-1, 0, 0)),
        (1, (1, 0, 0)),
    ]
    for p in pairs:
        assert np.allclose(p.vector, p.shift[0] * a1, rtol=0.0, atol=STORED.atol)


def test_self_image_cell_has_every_vector_shorter_than_the_cutoff(structures):
    lengths = np.linalg.norm(structures["self_image_cell"].cell.array, axis=1)
    assert lengths.max() < bs.PROBE_CUTOFF
    assert lengths.min() >= 3.0 - STORED.atol


def test_rattled_multi8_is_a_displaced_three_species_rocksalt(structures):
    atoms = structures["rattled_multi8"]
    assert sorted(atoms.get_chemical_symbols()) == [
        "Ca",
        "Ca",
        "Mg",
        "Mg",
        "O",
        "O",
        "O",
        "O",
    ]
    half = 4.4 / 2.0
    offsets = atoms.positions - half * np.round(atoms.positions / half)
    assert np.abs(offsets).max() < 5 * 0.06
    assert np.abs(offsets).max() > 0.0
    assert np.std(offsets) == pytest.approx(
        0.06, rel=sample_std_rtol(offsets.size), abs=0.0
    )
    assert np.allclose(atoms.cell.array, 4.4 * np.eye(3), rtol=0.0, atol=STORED.atol)


def test_slab_is_three_layers_with_a_four_angstrom_gap(structures):
    atoms = structures["slab"]
    assert atoms.pbc.tolist() == [True, True, False]
    z = np.sort(atoms.positions[:, 2])
    layers = z[::4]
    spacing = A_CU / np.sqrt(3.0)
    assert np.allclose(np.diff(layers), spacing, rtol=0.0, atol=STORED.atol)
    assert np.allclose(z.reshape(3, 4), layers[:, None], rtol=0.0, atol=STORED.atol)
    gap = atoms.cell.array[2, 2] - (z[-1] - z[0])
    assert gap == pytest.approx(4.0, abs=STORED.atol)
    assert gap < bs.PROBE_CUTOFF
    in_plane = A_CU / np.sqrt(2.0) * 2
    assert np.linalg.norm(atoms.cell.array[0]) == pytest.approx(
        in_plane, abs=STORED.atol
    )


def test_slab_gains_edges_across_the_gap_when_the_third_axis_is_periodic(structures):
    atoms = structures["slab"]
    honoured = _pairs(atoms, bs.PROBE_CUTOFF)
    all_axes = _pairs(atoms, bs.PROBE_CUTOFF, honour_pbc=False)
    gained = {(p.i, p.j, p.shift) for p in all_axes} - {
        (p.i, p.j, p.shift) for p in honoured
    }
    assert gained
    assert all(shift[2] != 0 for _, _, shift in gained)


def test_isolated_atom_sees_only_its_own_images_at_the_cell_length(structures):
    atoms = structures["isolated_atom"]
    cell_length = 20.0
    assert np.allclose(
        atoms.cell.array, cell_length * np.eye(3), rtol=0.0, atol=STORED.atol
    )
    assert _pairs(atoms, cell_length - 0.1) == []
    pairs = _pairs(atoms, cell_length + 0.1)
    assert sorted(p.shift for p in pairs) == sorted([
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ])


def test_icosahedron_satellite_neighbour_counts_range_from_two_to_twelve(structures):
    atoms = structures["icosahedron_satellite"]
    counts = _counts(atoms, _pairs(atoms, bs.PROBE_CUTOFF))
    assert counts.min() == 2
    assert counts.max() == 12
    nearest_shell_cutoff = 3.0
    first_shell = _counts(atoms, _pairs(atoms, nearest_shell_cutoff))
    assert first_shell[:13].max() == 12
    assert first_shell[13] == 0
    centre_to_atoms = np.linalg.norm(
        atoms.positions[:13] - atoms.positions[:13].mean(axis=0), axis=1
    )
    assert centre_to_atoms.min() == pytest.approx(0.0, abs=STORED.atol)
    outer_radius = np.linalg.norm(atoms.positions[:13], axis=1).max()
    satellite_radius = np.linalg.norm(atoms.positions[13])
    assert satellite_radius - outer_radius == pytest.approx(
        SATELLITE_GAP, abs=STORED.atol
    )


def test_mineral_is_charge_neutral_with_the_expected_formula(structures):
    atoms = structures["mineral"]
    assert atoms.get_chemical_formula() == "H2Mg7O16Si4"
    assert sum(MINERAL_FORMAL_CHARGE[s] for s in atoms.get_chemical_symbols()) == 0


def test_mineral_cell_is_the_orthorhombic_forsterite_cell(structures):
    atoms = structures["mineral"]
    assert np.allclose(
        atoms.cell.array, np.diag([10.19, 5.98, 4.75]), rtol=0.0, atol=STORED.atol
    )
    assert atoms.pbc.all()


def _bonded(
    atoms: Atoms, centre: int, symbol: str, low: float, high: float
) -> list[int]:
    return [
        p.j
        for p in _pairs(atoms, high)
        if p.i == centre
        and atoms[p.j].symbol == symbol
        and low <= float(np.linalg.norm(p.vector)) <= high
    ]


def test_mineral_silicon_is_tetrahedral_and_magnesium_octahedral(structures):
    atoms = structures["mineral"]
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "Si":
            assert len(_bonded(atoms, i, "O", *SI_O_RANGE)) == 4, i
        if symbol == "Mg":
            assert len(_bonded(atoms, i, "O", *MG_O_RANGE)) == 6, i


def test_mineral_hydroxyl_hydrogens_each_bond_to_a_distinct_oxygen_only(structures):
    atoms = structures["mineral"]
    hydrogens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    assert len(hydrogens) == 2
    oxygens = []
    for h in hydrogens:
        near = [
            (p.j, float(np.linalg.norm(p.vector)))
            for p in _pairs(atoms, O_H_PAIR_MAX)
            if p.i == h
        ]
        assert len(near) == 1, h
        j, length = near[0]
        assert atoms[j].symbol == "O"
        assert length == pytest.approx(O_H_BOND, abs=STORED.atol)
        oxygens.append(j)
    assert len(set(oxygens)) == 2


def test_mineral_hydrogens_have_room_apart_from_their_oxygen(structures):
    atoms = structures["mineral"]
    hydrogens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    for h in hydrogens:
        others = sorted(
            float(np.linalg.norm(p.vector)) for p in _pairs(atoms, 3.0) if p.i == h
        )[1:]
        assert others
        assert others[0] >= H_CLEARANCE, (h, others[0])
