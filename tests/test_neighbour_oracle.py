"""Tests of the brute-force neighbour oracle (``tests/neighbour_oracle.py``).

The oracle is the reference that ``tests/test_databuilder.py`` compares the TF builder with, so it
is checked here against numbers counted by hand (a dimer, simple cubic and fcc coordination
shells, the exact cutoff boundary) and against a second, naive enumeration over a fixed block of
images that does not use the exact image window.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tests.neighbour_oracle import (
    brute_force_pairs,
    pair_cutoff,
    recover_shifts,
    shift_window,
)
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

A_SC = 2.0
A_FCC = 4.0
FCC_CELL = A_FCC * np.eye(3)
FCC_POSITIONS = A_FCC * np.array(
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float
)

# Cumulative counts of neighbours of one atom in a perfect lattice, by shell radius in units of
# the cubic lattice constant a. Simple cubic: 6 at a, 12 at a sqrt(2), 8 at a sqrt(3).
# fcc: 12 at a/sqrt(2), 6 at a, 24 at a sqrt(3/2), 12 at a sqrt(2).
SC_SHELLS = [(1.0, 6), (math.sqrt(2), 12), (math.sqrt(3), 8)]
FCC_SHELLS = [
    (1 / math.sqrt(2), 12),
    (1.0, 6),
    (math.sqrt(1.5), 24),
    (math.sqrt(2), 12),
]


def cumulative(shells: list[tuple[float, int]], upto: int) -> int:
    return sum(count for _, count in shells[:upto])


def naive_pairs(positions, cell, symbols, cutoff, block: int = 6) -> set:
    """Edges by testing every shift in ``[-block, block]^3``: no reciprocal-vector window."""
    out = set()
    for i, j in itertools.product(range(len(positions)), repeat=2):
        rc = pair_cutoff(cutoff, symbols[i], symbols[j])
        for shift in itertools.product(range(-block, block + 1), repeat=3):
            if i == j and not any(shift):
                continue
            vector = positions[j] + np.asarray(shift) @ cell - positions[i]
            if np.linalg.norm(vector) < rc:
                out.add((i, j, shift))
    return out


def keys(pairs) -> list:
    return [(p.i, p.j, p.shift) for p in pairs]


# ---------------------------------------------------------------- hand-counted cases


def test_dimer_in_large_box_has_two_directed_edges() -> None:
    cell = 30.0 * np.eye(3)
    positions = np.array([[0.0, 0, 0], [3.0, 0, 0]])
    pairs = brute_force_pairs(positions, cell, ["Cu", "Cu"], 4.0)
    assert keys(pairs) == [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))]
    np.testing.assert_allclose(
        pairs[0].vector, [3.0, 0, 0], rtol=ARITH.rtol, atol=ARITH.atol
    )
    np.testing.assert_allclose(
        pairs[1].vector, [-3.0, 0, 0], rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_dimer_beyond_the_cutoff_has_no_edges() -> None:
    cell = 30.0 * np.eye(3)
    positions = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    assert brute_force_pairs(positions, cell, ["Cu", "Cu"], 4.0) == []


def test_cutoff_is_strict() -> None:
    """A pair at exactly the cutoff is not an edge; just inside it is."""
    cell = 30.0 * np.eye(3)
    assert (
        brute_force_pairs(np.array([[0.0, 0, 0], [4.0, 0, 0]]), cell, ["Cu"] * 2, 4.0)
        == []
    )
    inside = brute_force_pairs(
        np.array([[0.0, 0, 0], [3.9999, 0, 0]]), cell, ["Cu"] * 2, 4.0
    )
    assert len(inside) == 2


def test_dimer_across_the_boundary_uses_a_shift() -> None:
    """Atoms 1 A from opposite faces of a 10 A cell are 2 A apart through the boundary."""
    cell = 10.0 * np.eye(3)
    positions = np.array([[1.0, 5, 5], [9.0, 5, 5]])
    pairs = brute_force_pairs(positions, cell, ["Cu"] * 2, 3.0)
    assert keys(pairs) == [(0, 1, (-1, 0, 0)), (1, 0, (1, 0, 0))]
    np.testing.assert_allclose(
        pairs[0].vector, [-2.0, 0, 0], rtol=ARITH.rtol, atol=ARITH.atol
    )


@pytest.mark.parametrize("upto", [1, 2, 3])
def test_simple_cubic_coordination_numbers(upto: int) -> None:
    """One atom in a cell of 2 A: the neighbours are the lattice points of a simple cubic lattice."""
    radius = SC_SHELLS[upto - 1][0] * A_SC * 1.001
    pairs = brute_force_pairs(np.zeros((1, 3)), A_SC * np.eye(3), ["Cu"], radius)
    assert len(pairs) == cumulative(SC_SHELLS, upto)


@pytest.mark.parametrize("upto", [1, 2, 3, 4])
def test_fcc_coordination_numbers(upto: int) -> None:
    radius = FCC_SHELLS[upto - 1][0] * A_FCC * 1.001
    pairs = brute_force_pairs(FCC_POSITIONS, FCC_CELL, ["Cu"] * 4, radius)
    per_atom = np.bincount([p.i for p in pairs], minlength=4)
    assert per_atom.tolist() == [cumulative(FCC_SHELLS, upto)] * 4


def test_simple_cubic_shell_boundary_is_excluded() -> None:
    """With the cutoff equal to a, the six first neighbours at distance a are not edges."""
    assert brute_force_pairs(np.zeros((1, 3)), A_SC * np.eye(3), ["Cu"], A_SC) == []


def test_self_images_are_kept_and_the_zero_shift_is_not() -> None:
    pairs = brute_force_pairs(np.zeros((1, 3)), A_SC * np.eye(3), ["Cu"], 1.2 * A_SC)
    assert all(p.i == p.j == 0 and any(p.shift) for p in pairs)
    assert len(pairs) == 6


def test_full_list_is_symmetric() -> None:
    """(i, j, n) is an edge exactly when (j, i, -n) is, with the opposite vector."""
    rng = np.random.default_rng(7)
    cell = np.array([[5.0, 0, 0], [1.0, 4.5, 0], [0.5, 0.7, 6.0]])
    positions = rng.random((5, 3)) @ cell
    pairs = brute_force_pairs(positions, cell, ["Cu"] * 5, 5.5)
    by_key = {(p.i, p.j, p.shift): p.vector for p in pairs}
    assert len(by_key) == len(pairs)
    for (i, j, shift), vector in by_key.items():
        opposite = by_key[(j, i, tuple(-s for s in shift))]
        np.testing.assert_allclose(opposite, -vector, rtol=ARITH.rtol, atol=ARITH.atol)


def test_pairs_are_ordered_by_atom_and_shift() -> None:
    pairs = brute_force_pairs(FCC_POSITIONS, FCC_CELL, ["Cu"] * 4, 3.2)
    assert keys(pairs) == sorted(keys(pairs))


def test_per_pair_cutoff_function() -> None:
    """Cu-Cu 3.5, Cu-Al 2.0, Al-Al 1.0 on a dimer-plus-one chain 3 A apart."""
    table = {("Cu", "Cu"): 3.5, ("Cu", "Al"): 2.0, ("Al", "Cu"): 2.0, ("Al", "Al"): 1.0}
    cell = 40.0 * np.eye(3)
    positions = np.array([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])
    pairs = brute_force_pairs(
        positions, cell, ["Cu", "Cu", "Al"], lambda a, b: table[a, b]
    )
    assert keys(pairs) == [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))]


def test_pair_cutoff_accepts_number_and_function() -> None:
    assert pair_cutoff(4.0, "Cu", "Al") == 4.0
    assert pair_cutoff(lambda a, b: len(a + b), "Cu", "Al") == 4.0


# ---------------------------------------------------------------- image window


def test_shift_window_matches_the_reach() -> None:
    """Reach rc * |g| = 2.5 / 2 = 1.25 along a cell of 2 A; a pair displaced by 0.3 cells."""
    windows = shift_window(np.array([0.3, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]), 2.5)
    assert list(windows[0]) == [-1, 0]
    assert list(windows[1]) == [-1, 0, 1]
    narrow = shift_window(np.array([0.3, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]), 0.9)
    assert list(narrow[0]) == [0]


def test_shift_window_of_a_non_periodic_direction_is_zero_only() -> None:
    windows = shift_window(
        np.zeros(3), np.full(3, 0.5), 10.0, periodic=(True, False, True)
    )
    assert list(windows[1]) == [0]
    assert len(windows[0]) > 1


@pytest.mark.parametrize(
    "cell",
    [
        np.diag([3.0, 4.0, 5.0]),
        np.array([[3.0, 0, 0], [2.4, 1.8, 0], [0.4, -0.6, 3.2]]),
        np.array([[2.5, 0, 0], [-1.2, 2.2, 0], [1.1, 0.9, 6.0]]),
    ],
)
def test_exact_window_equals_the_naive_enumeration(cell: np.ndarray) -> None:
    rng = np.random.default_rng(11)
    positions = rng.random((3, 3)) @ cell
    symbols = ["Cu", "Al", "Cu"]
    rc = 4.5
    exact = brute_force_pairs(positions, cell, symbols, rc)
    assert set(keys(exact)) == naive_pairs(positions, cell, symbols, rc)
    assert len(exact) > 30


def test_partial_periodicity_drops_the_shifts_of_the_open_direction() -> None:
    cell = np.diag([3.0, 3.0, 3.0])
    full = brute_force_pairs(np.zeros((1, 3)), cell, ["Cu"], 3.2)
    slab = brute_force_pairs(
        np.zeros((1, 3)), cell, ["Cu"], 3.2, periodic=(True, True, False)
    )
    assert len(full) == 6
    assert len(slab) == 4
    assert all(p.shift[2] == 0 for p in slab)


# ---------------------------------------------------------------- recover_shifts


def test_recover_shifts_inverts_the_vector_definition() -> None:
    rng = np.random.default_rng(5)
    cell = np.array([[4.0, 0, 0], [1.0, 3.5, 0], [0.3, 0.2, 5.0]])
    positions = rng.random((4, 3)) @ cell
    pairs = brute_force_pairs(positions, cell, ["Cu"] * 4, 5.0)
    shifts = recover_shifts(
        positions,
        cell,
        np.array([p.i for p in pairs]),
        np.array([p.j for p in pairs]),
        np.array([p.vector for p in pairs]),
    )
    assert shifts == [p.shift for p in pairs]


def test_recover_shifts_rejects_a_vector_that_is_not_a_lattice_image() -> None:
    cell = 5.0 * np.eye(3)
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="lattice image"):
        recover_shifts(
            positions, cell, np.array([0]), np.array([1]), np.array([[1.4, 0, 0]])
        )
