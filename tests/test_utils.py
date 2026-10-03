"""Tests of ``enforce_pbc`` (``tensorpotential/utils.py``).

``GeometricalDataBuilder.extract_from_ase_atoms`` calls ``enforce_pbc(atoms, max_cutoff)`` before it
builds pairs. Every test here **pins current behaviour** and is not an endorsement: where a behaviour
looks like a hazard (the caller's ``Atoms`` is edited in place, partial periodic boundaries are
overridden, a zero cell vector is made periodic) the docstring says so, and a later fix is meant
to change the test on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from tensorpotential.utils import enforce_pbc
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

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
