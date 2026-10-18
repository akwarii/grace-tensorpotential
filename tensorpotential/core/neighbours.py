"""The neighbour-list contract shared by the TensorFlow and the torch backends (NBR1, decision D18).

Numpy in, numpy out; the module imports neither TensorFlow nor torch, and the backend packages only when they are used.

The contract
------------
* The list is full: ``(i, j, S)`` and ``(j, i, -S)`` are both present. A self-image ``(i, i, S)`` with ``S != 0`` is
  kept; the zero shift of ``i == i`` is not. A pair is kept when its length is **strictly** below the cutoff.
* The vector of a pair is ``positions[j] - positions[i] + S @ cell`` (Angstrom), the ``bond_vector`` of the TF model.
* An atom without a neighbour has no pair: the fictitious bond beyond the cutoff that ``GeometricalDataBuilder`` adds
  to such an atom is not part of the contract.
* The functions never modify their inputs. The geometry the pairs were computed in (positions, cell, periodicity) is
  part of the result, because the TF policy may replace it:

  - ``enforce_pbc=True`` (the default, TF's ``enforce_pbc``): a structure with no periodic axis gets a cube cell of
    side ``2 (max_d + cutoff)`` (``max_d`` the largest distance from the first atom) and is centred in it the way
    ``ase.Atoms.center`` does; then every axis is periodic, also when the input was periodic along some axes only.
  - ``enforce_pbc=False`` honours ``pbc`` per axis. A cell that is singular because of the rows of its non-periodic
    axes (the ASE convention of a zero third row for a slab) gets those rows replaced by unit vectors orthogonal to
    the periodic rows; a non-periodic axis has the shift 0, so the replacement does not change any pair.

* The order of the pairs is the backend's own: matscipy's order is that of the TF path, which this module therefore
  reproduces bit for bit. :func:`canonical_order` sorts by ``(i, j, S)`` for comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np

from tensorpotential.core.backends import BackendNotInstalledError
from tensorpotential.errors import NeighbourListError

NeighbourBackend = Literal["matscipy", "vesin"]
NEIGHBOUR_BACKENDS: Final[tuple[NeighbourBackend, ...]] = get_args(NeighbourBackend)


@dataclass(frozen=True)
class Geometry:
    """Positions (Angstrom), cell (rows are the lattice vectors) and periodic axes a list was computed in."""

    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for array in (self.positions, self.cell, self.pbc):
            array.setflags(write=False)


@dataclass(frozen=True)
class NeighbourList:
    """Directed pairs: centre ``i``, neighbour ``j``, integer cell shift ``shifts`` and Cartesian ``vectors``.

    ``cutoff`` is the cutoff of the search; :func:`apply_pair_cutoffs` may have kept a subset of it since. All
    arrays are read-only: ``i`` and ``j`` (n_pairs,) int64, ``shifts`` (n_pairs, 3) int64, ``vectors``
    (n_pairs, 3) float64.
    """

    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray
    vectors: np.ndarray
    cutoff: float
    geometry: Geometry

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for array in (self.i, self.j, self.shifts, self.vectors):
            array.setflags(write=False)

    def subset(self, keep: np.ndarray) -> NeighbourList:
        """Return the pairs selected by ``keep`` (a boolean mask or an index array), in that order."""
        # indexing with a mask or an index array copies, so the result owns its arrays
        return NeighbourList(
            self.i[keep],
            self.j[keep],
            self.shifts[keep],
            self.vectors[keep],
            self.cutoff,
            self.geometry,
        )


def _as_geometry(
    positions: np.ndarray, cell: np.ndarray, pbc: bool | Sequence[bool]
) -> Geometry:
    positions = np.array(positions, dtype=np.float64)
    cell = np.array(cell, dtype=np.float64)
    periodic = np.asarray(pbc, dtype=bool)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        msg = f"positions must have the shape (n_atoms >= 1, 3), got {positions.shape}"
        raise NeighbourListError(msg)
    if periodic.shape not in {(), (3,)}:
        msg = f"pbc must be one flag or one flag per axis, got {pbc!r}"
        raise NeighbourListError(msg)
    if cell.shape != (3, 3):
        msg = f"cell must have the shape (3, 3) with the lattice vectors as rows, got {cell.shape}"
        raise NeighbourListError(msg)
    if not (np.isfinite(positions).all() and np.isfinite(cell).all()):
        msg = "positions and cell must be finite"
        raise NeighbourListError(msg)
    return Geometry(positions, cell, np.array(np.broadcast_to(periodic, (3,))))


def _invented_box(positions: np.ndarray, cutoff: float) -> Geometry:
    # The operations of TF's enforce_pbc and of ase.Atoms.center on a cube, in their order, so that the positions,
    # and hence the vectors, agree bit for bit.
    max_d = np.max(np.linalg.norm(positions - positions[0], axis=1))
    side = (max_d + cutoff) * 2
    shift = 0.5 * ((side - positions.max(axis=0)) - positions.min(axis=0))
    return Geometry(
        positions + shift * side / side, np.eye(3) * side, np.ones(3, dtype=bool)
    )


def _complete_cell(geometry: Geometry) -> Geometry:
    cell, periodic = geometry.cell, geometry.pbc
    if np.linalg.matrix_rank(cell) == 3:
        return geometry
    rows = cell[periodic]
    if np.linalg.matrix_rank(rows) < len(
        rows
    ):  # every singular cell with all axes periodic ends here
        hint = ", or pass enforce_pbc=False to keep the axes that are not periodic in the input open"
        msg = (
            f"the cell vectors of the periodic axes {np.flatnonzero(periodic).tolist()} are zero or linearly "
            f"dependent: {cell.tolist()}. Give independent lattice vectors for them{hint if periodic.all() else ''}"
        )
        raise NeighbourListError(msg)
    completed = cell.copy()
    # the right singular vectors beyond the rank of the periodic rows span their orthogonal complement
    # (all three of them, the identity, when no axis is periodic)
    completed[~periodic] = np.linalg.svd(rows, full_matrices=True)[2][len(rows) :]
    return Geometry(geometry.positions, completed, periodic)


def effective_geometry(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: bool | Sequence[bool],
    cutoff: float,
    *,
    enforce_pbc: bool = True,
) -> Geometry:
    """Return the geometry the pairs are computed in under the periodicity policy (module docstring, D18).

    Parameters
    ----------
    positions
        (n_atoms, 3) Cartesian positions in Angstrom; they may lie outside the cell.
    cell
        (3, 3) lattice vectors as rows.
    pbc
        Periodicity of each axis, or one flag for all three.
    cutoff
        Cutoff of the search in Angstrom (the largest pair cutoff); it sizes the invented box.
    enforce_pbc
        ``True`` reproduces TF's ``enforce_pbc``, ``False`` honours ``pbc`` per axis.

    Raises
    ------
    NeighbourListError
        Malformed input, or a periodic axis whose cell vector is zero or dependent on the others.
    """
    geometry = _as_geometry(positions, cell, pbc)
    if enforce_pbc:
        if not geometry.pbc.any():
            return _invented_box(geometry.positions, cutoff)
        geometry = Geometry(geometry.positions, geometry.cell, np.ones(3, dtype=bool))
    return _complete_cell(geometry)


def _matscipy_pairs(geometry: Geometry, cutoff: float) -> tuple[np.ndarray, ...]:
    from matscipy.neighbours import neighbour_list

    return neighbour_list(
        "ijSD",
        positions=geometry.positions,
        cell=geometry.cell,
        pbc=geometry.pbc,
        cutoff=cutoff,
    )


def _vesin_pairs(geometry: Geometry, cutoff: float) -> tuple[np.ndarray, ...]:
    try:
        import vesin
    except ModuleNotFoundError as error:
        msg = (
            "The vesin neighbour list needs the vesin package, which is not installed. Install it with: pip install "
            "vesin (it also comes with 'tensorpotential[torch-sim]'), or use backend='matscipy'."
        )
        raise BackendNotInstalledError("torch-sim", ("vesin",), msg) from error

    search = vesin.NeighborList(cutoff=cutoff, full_list=True)
    return tuple(
        search.compute(
            points=geometry.positions,
            box=geometry.cell,
            periodic=geometry.pbc,
            quantities="ijSD",
        )
    )


_SEARCHES: Final = {"matscipy": _matscipy_pairs, "vesin": _vesin_pairs}


def neighbour_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: bool | Sequence[bool],
    cutoff: float,
    *,
    backend: NeighbourBackend = "matscipy",
    enforce_pbc: bool = True,
) -> NeighbourList:
    """Every directed pair closer than ``cutoff``, under the contract of the module docstring.

    Parameters
    ----------
    positions, cell, pbc, enforce_pbc
        As for :func:`effective_geometry`.
    cutoff
        Cutoff in Angstrom, positive and finite; per-pair cutoffs below it are applied by :func:`apply_pair_cutoffs`.
    backend
        ``"matscipy"`` (the TF path, bit for bit) or ``"vesin"`` (needs the ``vesin`` package).

    Raises
    ------
    NeighbourListError
        An unknown backend, a cutoff that is not positive and finite, or a geometry :func:`effective_geometry`
        refuses.
    BackendNotInstalledError
        The backend package is not installed.
    """
    if backend not in _SEARCHES:
        msg = f"unknown neighbour-list backend {backend!r}; use one of {list(NEIGHBOUR_BACKENDS)}"
        raise NeighbourListError(msg)
    if not (np.isfinite(cutoff) and cutoff > 0):
        msg = f"the cutoff must be positive and finite, got {cutoff!r}"
        raise NeighbourListError(msg)
    geometry = effective_geometry(positions, cell, pbc, cutoff, enforce_pbc=enforce_pbc)
    i, j, shifts, vectors = _SEARCHES[backend](geometry, float(cutoff))
    return NeighbourList(
        np.asarray(i, dtype=np.int64),
        np.asarray(j, dtype=np.int64),
        np.asarray(shifts, dtype=np.int64).reshape(-1, 3),
        np.asarray(vectors, dtype=np.float64).reshape(-1, 3),
        float(cutoff),
        geometry,
    )


def pair_cutoff_matrix(
    elements: Sequence[str],
    cutoff: float,
    pair_cutoffs: Mapping[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Return the (n_elements, n_elements) cutoff of each pair of elements, ``cutoff`` where none is given.

    ``pair_cutoffs`` is the table of :func:`tensorpotential.core.cutoffs.process_cutoff_dict`; the default for a missing
    pair is the one of ``GeometricalDataBuilder``.

    Raises
    ------
    NeighbourListError
        The two orders of a pair have different cutoffs (the TF builder would then pick one by the iteration order
        of a set).
    """
    table = pair_cutoffs or {}
    matrix = np.array(
        [[table.get((a, b), cutoff) for b in elements] for a in elements],
        dtype=np.float64,
    )
    rows, cols = np.nonzero(matrix != matrix.T)
    if rows.size:
        pairs = sorted({
            tuple(sorted((elements[r], elements[c])))
            for r, c in zip(rows, cols, strict=True)
        })
        msg = f"the pair cutoffs are not symmetric for {pairs}; give one cutoff per pair of elements"
        raise NeighbourListError(msg)
    return matrix


def apply_pair_cutoffs(
    pairs: NeighbourList, species: np.ndarray, cutoff_matrix: np.ndarray
) -> NeighbourList:
    """Keep the pairs strictly shorter than the cutoff of their two species, in their order.

    Parameters
    ----------
    pairs
        A list searched with a cutoff at least as large as every entry of ``cutoff_matrix``.
    species
        (n_atoms,) index of each atom into the rows of ``cutoff_matrix``.
    cutoff_matrix
        Symmetric (n_species, n_species) cutoffs in Angstrom (:func:`pair_cutoff_matrix`).

    Raises
    ------
    NeighbourListError
        A pair cutoff above the cutoff of the search (pairs would be missing), or ``species`` that does not fit.
    """
    species = np.asarray(species)
    cutoff_matrix = np.asarray(cutoff_matrix, dtype=np.float64)
    if cutoff_matrix.max(initial=0.0) > pairs.cutoff:
        msg = f"a pair cutoff of {cutoff_matrix.max()} A exceeds the cutoff {pairs.cutoff} A the list was searched with"
        raise NeighbourListError(msg)
    if (
        species.shape != (len(pairs.geometry.positions),)
        or ((species < 0) | (species >= len(cutoff_matrix))).any()
    ):
        msg = f"species must give each of the {len(pairs.geometry.positions)} atoms a row of the cutoff matrix"
        raise NeighbourListError(msg)
    lengths = np.linalg.norm(pairs.vectors, axis=1)
    return pairs.subset(lengths < cutoff_matrix[species[pairs.i], species[pairs.j]])


def canonical_order(pairs: NeighbourList) -> NeighbourList:
    """Return the pairs sorted by centre, neighbour and shift, the order in which lists of two backends compare."""
    shifts = pairs.shifts
    return pairs.subset(
        np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], pairs.j, pairs.i))
    )
