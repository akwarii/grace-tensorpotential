"""The fixture structure set of the torch backend: eight small ASE structures, each with a stated purpose.

The set feeds the golden-fixture generator (FIX2) and the neighbour-list twin tests. It is built by
code, not typed by hand, and written to two files that stay in sync with this module (a test
regenerates both and compares):

* ``structures.json``: the lossless record (symbols, cell, positions, pbc, purpose, feature tags);
* ``structures.xyz``: the same structures as extended XYZ, one frame each, for viewers and ASE.

Positions and cells are rounded to ``DECIMALS`` places before they are stored, so the two files hold
the same numbers. Units are Angstrom. No structure carries energies or forces: those come from the
model outputs of the golden fixtures, never from this file. The module needs ``ase`` and ``numpy``
only (no TensorFlow, no torch, no ``tensorpotential`` import).

Run ``python -m tests_torch.structures.build_structures`` from the repository root to rewrite the files.

Feature tags
------------
``aperiodic``
    ``pbc`` is all ``False`` (the TF builder gives such a structure an invented cube cell).
``partial_pbc``
    ``pbc`` is neither all ``True`` nor all ``False``.
``self_image_pairs``
    at ``PROBE_CUTOFF`` an atom has itself as a neighbour through a nonzero cell shift.
``isolated_atom``
    at ``PROBE_CUTOFF`` an atom has no neighbour at all (the TF builder adds a dummy bond).
``unequal_neighbour_counts``
    at ``PROBE_CUTOFF`` the atoms do not all have the same number of neighbours.
``multi_element``
    three or more chemical species.
``four_elements``
    exactly the species Mg, Si, O and H.
``rattled``
    positions displaced from the ideal lattice by seeded random noise.
``vacuum_gap_below_cutoff``
    a slab whose vacuum gap is shorter than ``PROBE_CUTOFF``: periodic images across the gap are
    neighbours when the third axis is made periodic, and are not when the per-axis ``pbc`` is honoured.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, fcc111
from ase.cluster import Icosahedron
from ase.geometry import find_mic
from ase.io import read, write
from ase.spacegroup import crystal

LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).parent
JSON_PATH = HERE / "structures.json"
XYZ_PATH = HERE / "structures.xyz"

FORMAT_VERSION = 1
DECIMALS = 8
PROBE_CUTOFF = 5.0
"""Cutoff (Angstrom) at which the neighbour-based feature tags are stated and tested."""

A_CU = 3.61
"""Lattice constant of fcc Cu (Angstrom)."""

O_H_BOND = 0.97
"""O-H distance of the hydroxyl groups of the mineral cell (Angstrom)."""

SATELLITE_GAP = 3.6
"""Distance (Angstrom) beyond the outer radius of the icosahedron at which the extra Cu atom sits."""

MAX_VACANCY_O_DISTANCE = 2.5
"""Oxygens closer than this to the Mg vacancy (Angstrom) are the six of its octahedron."""


@dataclass(frozen=True)
class StructureSpec:
    """One structure of the set: name, purpose, feature tags and the function that builds it."""

    name: str
    purpose: str
    features: tuple[str, ...]
    build: Callable[[], Atoms]


def _dimer() -> Atoms:
    return Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], pbc=False)


def _self_image_cell() -> Atoms:
    cell = [[3.0, 0.0, 0.0], [0.8, 3.1, 0.0], [0.5, 0.9, 3.3]]
    return Atoms(
        "CuAu",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=cell,
        pbc=True,
    )


def _fcc4() -> Atoms:
    return bulk("Cu", "fcc", a=A_CU, cubic=True)


def _rattled_multi8() -> Atoms:
    atoms = bulk("MgO", "rocksalt", a=4.4, cubic=True)
    mg = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Mg"]
    for i in mg[:2]:
        atoms[i].symbol = "Ca"
    atoms.rattle(stdev=0.06, rng=np.random.default_rng(20261004))
    return atoms


def _slab() -> Atoms:
    return fcc111("Cu", size=(2, 2, 3), a=A_CU, vacuum=2.0)


def _isolated_atom() -> Atoms:
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=20.0 * np.eye(3), pbc=True)


def _icosahedron_satellite() -> Atoms:
    atoms = Icosahedron("Cu", noshells=2, latticeconstant=A_CU)
    radius = float(np.linalg.norm(atoms.positions, axis=1).max())
    atoms.append(Atom("Cu", position=[0.0, 0.0, radius + SATELLITE_GAP]))
    atoms.pbc = False
    return atoms


def _hydroxyl_sites(atoms: Atoms, vacancy: np.ndarray, count: int) -> list[np.ndarray]:
    """Hydrogen positions on ``count`` oxygens around ``vacancy``, pointing into it, where the H has the most room."""
    scored = []
    for i in np.flatnonzero(atoms.numbers == 8):
        vectors, lengths = find_mic(
            atoms.positions[i][None] - vacancy, atoms.cell, atoms.pbc
        )
        direction, norm = vectors[0], float(lengths[0])
        if norm > MAX_VACANCY_O_DISTANCE:
            continue
        hydrogen = atoms.positions[i] - O_H_BOND * direction / norm
        trial = atoms.copy()
        trial.append("H")
        trial.positions[-1] = hydrogen
        others = [j for j in range(len(atoms)) if j != i]
        clearance = float(trial.get_distances(len(atoms), others, mic=True).min())
        scored.append((-clearance, int(i), hydrogen))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [hydrogen for _, _, hydrogen in scored[:count]]


def _mineral() -> Atoms:
    """Forsterite (Pnma, room-temperature coordinates) with one Mg(1) vacancy charge-balanced by two OH."""
    atoms = crystal(
        ["Mg", "Mg", "Si", "O", "O", "O"],
        basis=[
            (0.0, 0.0, 0.0),
            (0.2774, 0.25, 0.9915),
            (0.0940, 0.25, 0.4262),
            (0.0916, 0.25, 0.7657),
            (0.4474, 0.25, 0.2217),
            (0.1655, 0.0337, 0.2778),
        ],
        spacegroup=62,
        cellpar=[10.19, 5.98, 4.75, 90, 90, 90],
        primitive_cell=False,
    )
    vacancy_index = next(i for i in range(len(atoms)) if atoms[i].symbol == "Mg")
    vacancy = atoms.positions[vacancy_index].copy()
    del atoms[vacancy_index]
    sites = _hydroxyl_sites(atoms, vacancy, count=2)
    for hydrogen in sites:
        atoms.append("H")
        atoms.positions[-1] = hydrogen
    atoms.wrap()
    return atoms


SPECS: tuple[StructureSpec, ...] = (
    StructureSpec(
        "dimer",
        "Smallest non-trivial graph: two Cu atoms 2.5 A apart in vacuum, no cell, no pbc. "
        "Checks one bond in both directions, the force pair (equal and opposite) and the invented "
        "cube cell of an aperiodic input (zero stress).",
        ("aperiodic",),
        _dimer,
    ),
    StructureSpec(
        "self_image_cell",
        "Two atoms (Cu, Au) in a skewed cell with all vectors 3.0-3.3 A, shorter than the cutoff: "
        "every atom is its own neighbour through nonzero cell shifts (i == j), and shifts of both "
        "signs and several shells occur. Checks the shift convention and the virial of image bonds.",
        ("self_image_pairs",),
        _self_image_cell,
    ),
    StructureSpec(
        "fcc4",
        "Four-atom conventional fcc Cu cell (a = 3.61 A): perfect crystal with exact coordination "
        "shells (12, 6, 24, ...), equal neighbour counts, and a pair distance close to a shell edge. "
        "The reference for translation, rotation and permutation invariance.",
        ("self_image_pairs",),
        _fcc4,
    ),
    StructureSpec(
        "rattled_multi8",
        "Rocksalt cell of eight atoms with three species (Mg, Ca, O), two Mg sites replaced by Ca "
        "and all positions rattled (sigma 0.06 A, fixed seed): no symmetry, so forces are generic, "
        "and three element indices exercise the species embedding and the per-element tables.",
        ("self_image_pairs", "unequal_neighbour_counts", "multi_element", "rattled"),
        _rattled_multi8,
    ),
    StructureSpec(
        "slab",
        "Cu(111) slab, 2x2 surface cell and three layers, with 2.0 A of vacuum on each side "
        "(gap 4.0 A, below the cutoff) and pbc (True, True, False). A surface gives unequal neighbour "
        "counts, and the edge list differs between the TF all-axes-periodic default and honouring "
        "the per-axis pbc (decision D18).",
        ("partial_pbc", "unequal_neighbour_counts", "vacuum_gap_below_cutoff"),
        _slab,
    ),
    StructureSpec(
        "isolated_atom",
        "One Cu atom in a 20 A cubic cell: no neighbour within the cutoff. The TF builder adds a "
        "dummy bond (the atom never has zero bonds); the twin must reproduce or deliberately drop it. "
        "The energy is the free-atom term and the forces and stress are zero.",
        ("isolated_atom",),
        _isolated_atom,
    ),
    StructureSpec(
        "icosahedron_satellite",
        "Thirteen-atom Cu icosahedron plus one detached Cu atom 3.6 A beyond its outer radius, no "
        "pbc: neighbour counts run from 2 (the satellite) to 12 (the centre), so padding and the "
        "per-atom segment sums see very unequal atoms, in a non-periodic cluster with nontrivial "
        "point symmetry.",
        ("aperiodic", "unequal_neighbour_counts"),
        _icosahedron_satellite,
    ),
    StructureSpec(
        "mineral",
        "Forsterite (Mg2SiO4, Pnma, 28 atoms) with one Mg(1) vacancy and two hydroxyl H on "
        "neighbouring oxygens: 29 atoms of four species (Mg, Si, O, H) in an orthorhombic cell, "
        "the light element H with a short O-H bond (0.97 A), self-image pairs along the short axis, and unequal neighbour counts. Built "
        "from published room-temperature coordinates, not relaxed.",
        (
            "self_image_pairs",
            "multi_element",
            "four_elements",
            "unequal_neighbour_counts",
        ),
        _mineral,
    ),
)


def _round_decimal(values: np.ndarray) -> np.ndarray:
    """``values`` rounded to ``DECIMALS`` places as the decimal text would read back (``float('1.25000000')``)."""
    return np.vectorize(lambda x: float(f"{x:.{DECIMALS}f}"))(values)


def build_all() -> dict[str, Atoms]:
    """Build every structure of ``SPECS``, rounded as stored, keyed by name (in file order)."""
    built = {}
    for spec in SPECS:
        atoms = spec.build()
        atoms.set_cell(_round_decimal(atoms.cell.array))
        atoms.set_positions(_round_decimal(atoms.positions), apply_constraint=False)
        atoms.info = {}
        atoms.info["name"] = spec.name
        atoms.info["purpose"] = spec.purpose
        built[spec.name] = atoms
    return built


def _record(spec: StructureSpec, atoms: Atoms) -> dict[str, object]:
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "features": list(spec.features),
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "pbc": [bool(flag) for flag in atoms.pbc],
        "cell": atoms.cell.array.tolist(),
        "positions": atoms.positions.tolist(),
    }


def to_json_text(structures: dict[str, Atoms]) -> str:
    """The text of ``structures.json`` for the structures returned by ``build_all``."""
    document = {
        "format_version": FORMAT_VERSION,
        "units": {"length": "Angstrom"},
        "probe_cutoff": PROBE_CUTOFF,
        "structures": [_record(spec, structures[spec.name]) for spec in SPECS],
    }
    return json.dumps(document, indent=1) + "\n"


def write_files(directory: Path = HERE) -> None:
    """Write ``structures.json`` and ``structures.xyz`` into ``directory``."""
    structures = build_all()
    (directory / JSON_PATH.name).write_text(to_json_text(structures), encoding="utf-8")
    write(directory / XYZ_PATH.name, list(structures.values()), format="extxyz")
    LOGGER.info("wrote %d structures to %s", len(structures), directory)


def load_structures(path: Path = JSON_PATH) -> dict[str, Atoms]:
    """Read ``structures.json`` into ASE ``Atoms`` keyed by name; ``info`` holds ``name`` and ``purpose``."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    loaded = {}
    for record in document["structures"]:
        atoms = Atoms(
            symbols=record["symbols"],
            positions=record["positions"],
            cell=record["cell"],
            pbc=record["pbc"],
        )
        atoms.info["name"] = record["name"]
        atoms.info["purpose"] = record["purpose"]
        loaded[record["name"]] = atoms
    return loaded


def load_xyz(path: Path = XYZ_PATH) -> dict[str, Atoms]:
    """Read ``structures.xyz`` into ASE ``Atoms`` keyed by name."""
    frames = read(path, index=":", format="extxyz")
    return {atoms.info["name"]: atoms for atoms in frames}


def main() -> None:
    """Rewrite the two files next to this module."""
    logging.basicConfig(level=logging.INFO)
    write_files()


if __name__ == "__main__":
    main()
