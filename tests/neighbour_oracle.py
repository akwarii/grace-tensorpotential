"""A brute-force periodic neighbour list, the oracle for tests of ``GeometricalDataBuilder``.

It imports nothing from ``tensorpotential`` (nor matscipy): it only needs ``numpy``.

Rules (the contract the TF builder is compared with):

* the structure is periodic along the cell vectors flagged in ``periodic`` (all three by default,
  which is what the builder uses: it forces ``pbc=True`` before it builds pairs, see ``enforce_pbc``);
* every ordered pair ``(i, j, n)`` with integer cell shift ``n`` is a candidate, the vector is
  ``pos[j] + n @ cell - pos[i]``;
* a candidate is kept when its length is **strictly** below the pair cutoff;
* the list is full: ``(i, j, n)`` and ``(j, i, -n)`` are both present;
* the self-image ``(i, i, n)`` with ``n != 0`` is kept, the zero shift of ``i == i`` is not;
* the image range along each cell vector is exact: a vector shorter than ``rc`` has a fractional
  component along cell vector ``k`` of at most ``rc * |g_k|`` (``g_k`` the k-th reciprocal vector
  without the factor ``2 pi``), so no image outside that window can reach ``rc`` and the window
  is never wider than needed for a pair.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

Cutoff = float | Callable[[str, str], float]


class Pair(NamedTuple):
    """One directed edge: atom indices, integer cell shift and Cartesian vector."""

    i: int
    j: int
    shift: tuple[int, int, int]
    vector: np.ndarray


def pair_cutoff(cutoff: Cutoff, symbol_i: str, symbol_j: str) -> float:
    """Cutoff of one pair: a number, or a function of the two chemical symbols."""
    return float(cutoff(symbol_i, symbol_j)) if callable(cutoff) else float(cutoff)


def shift_window(
    delta_frac: np.ndarray,
    reciprocal_norm: np.ndarray,
    rc: float,
    periodic: Sequence[bool] = (True, True, True),
) -> list[range]:
    """Integer shifts along each cell vector that can give a vector shorter than ``rc``.

    ``delta_frac`` is the fractional displacement ``pos[j] - pos[i]`` in the cell basis and
    ``reciprocal_norm`` the lengths of the three reciprocal vectors (no ``2 pi``). A shift ``n``
    gives the fractional component ``delta + n``, and ``|delta + n| <= rc * |g|`` is necessary.
    A non-periodic direction only admits the shift 0.
    """
    windows = []
    for delta, g, wraps in zip(delta_frac, reciprocal_norm, periodic, strict=True):
        if not wraps:
            windows.append(range(1))
            continue
        reach = rc * g
        windows.append(range(math.ceil(-reach - delta), math.floor(reach - delta) + 1))
    return windows


def brute_force_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    symbols: Sequence[str],
    cutoff: Cutoff,
    periodic: Sequence[bool] = (True, True, True),
) -> list[Pair]:
    """All directed pairs closer than their cutoff in the infinite periodic crystal.

    Parameters
    ----------
    positions : (n, 3) Cartesian positions in Angstrom.
    cell : (3, 3) lattice vectors as rows, non-singular.
    symbols : chemical symbol of each atom.
    cutoff : a number, or a function ``(symbol_i, symbol_j) -> rc`` (must be symmetric).
    periodic : which cell vectors are periodic.

    Returns
    -------
    Pairs ordered by ``(i, j, shift)``.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    reciprocal = np.linalg.inv(cell).T
    reciprocal_norm = np.linalg.norm(reciprocal, axis=1)
    pairs = []
    for i, j in itertools.product(range(len(positions)), repeat=2):
        rc = pair_cutoff(cutoff, symbols[i], symbols[j])
        delta = positions[j] - positions[i]
        delta_frac = delta @ np.linalg.inv(cell)
        for shift in itertools.product(
            *shift_window(delta_frac, reciprocal_norm, rc, periodic)
        ):
            if i == j and not any(shift):
                continue
            vector = delta + np.asarray(shift) @ cell
            if np.linalg.norm(vector) < rc:
                pairs.append(Pair(i, j, shift, vector))
    return sorted(pairs, key=lambda p: (p.i, p.j, p.shift))


def recover_shifts(
    positions: np.ndarray,
    cell: np.ndarray,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    vectors: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Integer cell shift of each edge of a list given as ``(i, j, vector)``.

    Inverts ``vector = pos[j] + n @ cell - pos[i]``; the fractional result must be an integer
    to within a loose bound (an edge that is not a lattice image raises).
    """
    positions = np.asarray(positions, dtype=float)
    delta = positions[np.asarray(ind_j)] - positions[np.asarray(ind_i)]
    frac = (np.asarray(vectors, dtype=float) - delta) @ np.linalg.inv(cell)
    shifts = np.rint(frac)
    if not np.allclose(frac, shifts, atol=1e-3):
        raise ValueError("an edge vector is not a lattice image of its atom pair")
    return [(int(a), int(b), int(c)) for a, b, c in shifts]
