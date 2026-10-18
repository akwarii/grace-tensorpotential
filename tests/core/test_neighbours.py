"""The neighbour-list contract of :mod:`tensorpotential.core.neighbours`: inputs, the periodicity policy, pair cutoffs.

Logic layer here; the physical values of every backend against the brute-force oracle are in
``tests_torch/test_neighbours_contract.py``, and the bit-for-bit equality with ``GeometricalDataBuilder`` is in
``tests/test_databuilder.py``. These tests need neither TensorFlow nor torch.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from tensorpotential.core import neighbours
from tensorpotential.core.backends import BackendNotInstalledError
from tensorpotential.errors import NeighbourListError
from tests.neighbour_oracle import brute_force_pairs

CLUSTER = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.4, 2.2, 1.1], [7.0, 0.0, 0.0]])
SLAB_CELL = np.diag([5.0, 5.0, 0.0])
CUBE = np.eye(3) * 6.0


def _pairs(
    positions: np.ndarray = CLUSTER,
    cell: np.ndarray = CUBE,
    pbc: bool | list[bool] = True,
    cutoff: float = 3.0,
    **options: Any,
) -> neighbours.NeighbourList:
    return neighbours.neighbour_pairs(positions, cell, pbc, cutoff, **options)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"positions": np.zeros((2, 2))}, "shape"),
        ({"positions": np.zeros((0, 3))}, "n_atoms >= 1"),
        ({"cell": np.eye(2)}, "lattice vectors as rows"),
        ({"pbc": [True, False]}, "one flag per axis"),
        ({"positions": CLUSTER * np.nan}, "finite"),
        ({"cell": np.diag([np.inf, 1.0, 1.0])}, "finite"),
        ({"cutoff": 0.0}, "positive and finite"),
        ({"cutoff": -1.0}, "positive and finite"),
        ({"cutoff": float("nan")}, "positive and finite"),
        ({"cutoff": float("inf")}, "positive and finite"),
        ({"backend": "ase"}, "unknown neighbour-list backend 'ase'"),
    ],
)
def test_malformed_input_raises_a_neighbour_list_error(
    overrides: dict[str, Any], message: str
):
    with pytest.raises(NeighbourListError, match=message):
        _pairs(**overrides)


def test_the_inputs_are_not_modified_and_the_result_is_read_only():
    positions, cell = CLUSTER.copy(), np.zeros((3, 3))
    found = _pairs(positions=positions, cell=cell, pbc=False)
    np.testing.assert_array_equal(positions, CLUSTER)
    np.testing.assert_array_equal(cell, np.zeros((3, 3)))
    assert positions.flags.writeable
    arrays = (
        found.i,
        found.j,
        found.shifts,
        found.vectors,
        *vars(found.geometry).values(),
    )
    assert not any(a.flags.writeable for a in arrays)


def test_the_result_has_the_documented_dtypes_and_shapes_also_when_empty():
    found = _pairs(positions=CLUSTER[:1], pbc=False)
    assert (found.i.dtype, found.j.dtype, found.shifts.dtype, found.vectors.dtype) == (
        np.int64,
        np.int64,
        np.int64,
        np.float64,
    )
    assert found.i.shape == (0,)
    assert found.shifts.shape == found.vectors.shape == (0, 3)
    assert found.cutoff == 3.0


def test_an_aperiodic_structure_gets_the_box_and_the_centring_of_the_tf_path():
    # TF's enforce_pbc and ase.Atoms.center, run on an Atoms: the contract must give the same numbers.
    atoms = Atoms("Cu4", positions=CLUSTER)
    max_d = np.max(np.linalg.norm(CLUSTER - CLUSTER[0], axis=1))
    atoms.set_cell(np.eye(3) * ((max_d + 3.0) * 2))
    atoms.center()
    geometry = neighbours.effective_geometry(CLUSTER, np.zeros((3, 3)), False, 3.0)
    np.testing.assert_array_equal(geometry.positions, atoms.positions)
    np.testing.assert_array_equal(geometry.cell, atoms.cell.array)
    assert geometry.pbc.tolist() == [True, True, True]


def test_partial_periodicity_is_overridden_by_default_and_honoured_on_request():
    cell = np.diag([5.0, 6.0, 7.0])
    enforced = neighbours.effective_geometry(CLUSTER, cell, [True, False, True], 3.0)
    honoured = neighbours.effective_geometry(
        CLUSTER, cell, [True, False, True], 3.0, enforce_pbc=False
    )
    assert enforced.pbc.tolist() == [True, True, True]
    assert honoured.pbc.tolist() == [True, False, True]
    for geometry in (enforced, honoured):
        np.testing.assert_array_equal(geometry.cell, cell)
        np.testing.assert_array_equal(geometry.positions, CLUSTER)


def test_a_zero_cell_row_made_periodic_by_default_raises_with_the_way_out():
    # TF raises numpy's LinAlgError (a ValueError) from matscipy here; the contract says what to do.
    with pytest.raises(
        NeighbourListError, match=r"axes \[0, 1, 2\].*enforce_pbc=False"
    ) as raised:
        neighbours.effective_geometry(CLUSTER, SLAB_CELL, [True, True, False], 3.0)
    assert isinstance(raised.value, ValueError)


def test_a_zero_cell_row_of_an_open_axis_is_completed_by_an_orthogonal_unit_vector():
    tilted = SLAB_CELL.copy()
    tilted[1, 0] = 1.5
    geometry = neighbours.effective_geometry(
        CLUSTER, tilted, [True, True, False], 3.0, enforce_pbc=False
    )
    np.testing.assert_array_equal(geometry.cell[:2], tilted[:2])
    np.testing.assert_allclose(abs(geometry.cell[2]), [0.0, 0.0, 1.0], atol=1e-15)
    assert geometry.pbc.tolist() == [True, True, False]


def test_an_open_structure_without_a_cell_is_searched_in_a_unit_cell_when_honoured():
    geometry = neighbours.effective_geometry(
        CLUSTER, np.zeros((3, 3)), False, 3.0, enforce_pbc=False
    )
    np.testing.assert_allclose(abs(geometry.cell), np.eye(3), atol=1e-15)
    np.testing.assert_array_equal(geometry.positions, CLUSTER)


def test_a_non_singular_cell_is_kept_as_it_is_when_honoured():
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 30.0]])
    geometry = neighbours.effective_geometry(
        CLUSTER, cell, [True, True, False], 3.0, enforce_pbc=False
    )
    np.testing.assert_array_equal(geometry.cell, cell)


def test_dependent_cell_vectors_of_periodic_axes_raise():
    cell = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(
        NeighbourListError, match=r"axes \[0, 1\] are zero or linearly dependent"
    ) as raised:
        neighbours.effective_geometry(
            CLUSTER, cell, [True, True, False], 3.0, enforce_pbc=False
        )
    assert "enforce_pbc" not in str(raised.value)


def test_the_honoured_open_axis_gives_the_open_space_pairs_of_the_oracle():
    found = neighbours.canonical_order(
        _pairs(cell=SLAB_CELL, pbc=[True, True, False], enforce_pbc=False)
    )
    expected = brute_force_pairs(
        CLUSTER,
        found.geometry.cell,
        ["Cu"] * len(CLUSTER),
        3.0,
        periodic=(True, True, False),
    )
    assert _rows(found) == [(p.i, p.j, *p.shift) for p in expected]
    assert (found.shifts[:, 2] == 0).all()


def test_vesin_that_is_not_installed_raises_the_backend_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        sys.modules, "vesin", None
    )  # stands in for an environment without the package
    with pytest.raises(
        BackendNotInstalledError,
        match=r"needs the vesin package.*pip install vesin.*backend='matscipy'",
    ):
        _pairs(backend="vesin")


def test_the_pair_cutoff_matrix_fills_missing_pairs_with_the_global_cutoff():
    matrix = neighbours.pair_cutoff_matrix(["Cu", "Al"], 4.0, {("Cu", "Cu"): 5.0})
    np.testing.assert_array_equal(matrix, [[5.0, 4.0], [4.0, 4.0]])
    np.testing.assert_array_equal(neighbours.pair_cutoff_matrix(["Cu"], 4.0), [[4.0]])


def test_an_asymmetric_pair_cutoff_table_raises_and_names_the_pair():
    table = {("Cu", "Al"): 3.0, ("Al", "Cu"): 3.5, ("Al", "Al"): 4.0}
    with pytest.raises(
        NeighbourListError, match=r"not symmetric for \[\('Al', 'Cu'\)\]"
    ):
        neighbours.pair_cutoff_matrix(["Cu", "Al"], 4.0, table)


def test_pair_cutoffs_keep_what_the_oracle_keeps_with_the_same_cutoffs():
    # Physical value: per-pair strict cutoffs against the oracle that applies them during its own search.
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.0, 6.0, (12, 3))
    symbols = ["Cu", "Al"] * 6
    table = {("Cu", "Cu"): 2.5, ("Cu", "Al"): 3.0, ("Al", "Cu"): 3.0, ("Al", "Al"): 3.5}
    matrix = neighbours.pair_cutoff_matrix(["Cu", "Al"], 3.5, table)
    species = np.array([0, 1] * 6)
    found = neighbours.apply_pair_cutoffs(
        _pairs(positions=positions, cutoff=3.5), species, matrix
    )
    expected = brute_force_pairs(
        positions, np.eye(3) * 6.0, symbols, lambda a, b: table[a, b]
    )
    assert _rows(neighbours.canonical_order(found)) == [
        (p.i, p.j, *p.shift) for p in expected
    ]
    assert len(found.i) < len(_pairs(positions=positions, cutoff=3.5).i)


def test_a_pair_exactly_at_its_pair_cutoff_is_dropped():
    dimer = np.array([[0.0, 0.0, 0.0], [3.25, 0.0, 0.0]])  # 3.25 is exact in binary
    wide = _pairs(positions=dimer, cell=np.zeros((3, 3)), pbc=False, cutoff=4.0)
    assert (
        len(neighbours.apply_pair_cutoffs(wide, np.array([0, 0]), np.array([[3.25]])).i)
        == 0
    )
    assert (
        len(neighbours.apply_pair_cutoffs(wide, np.array([0, 0]), np.array([[3.26]])).i)
        == 2
    )


@pytest.mark.parametrize(
    ("species", "matrix", "message"),
    [
        (np.zeros(4, dtype=int), np.array([[3.5]]), "exceeds the cutoff 3.0"),
        (np.zeros(3, dtype=int), np.array([[2.0]]), "each of the 4 atoms"),
        (np.array([0, 0, 1, 0]), np.array([[2.0]]), "each of the 4 atoms"),
        (np.array([0, -1, 0, 0]), np.array([[2.0]]), "each of the 4 atoms"),
    ],
)
def test_pair_cutoffs_that_do_not_fit_the_list_raise(
    species: np.ndarray, matrix: np.ndarray, message: str
):
    with pytest.raises(NeighbourListError, match=message):
        neighbours.apply_pair_cutoffs(_pairs(), species, matrix)


def test_canonical_sorts_by_centre_neighbour_and_shift_and_keeps_every_pair():
    found = _pairs(cell=np.eye(3) * 2.5)
    ordered = neighbours.canonical_order(found)
    rows = _rows(ordered)
    assert rows == sorted(rows)
    assert sorted(_rows(found)) == rows
    assert ordered.geometry is found.geometry
    lookup = dict(zip(_rows(found), found.vectors, strict=True))
    np.testing.assert_array_equal(ordered.vectors, [lookup[row] for row in rows])


def _rows(found: neighbours.NeighbourList) -> list[tuple[int, ...]]:
    return [
        (int(a), int(b), *map(int, s))
        for a, b, s in zip(found.i, found.j, found.shifts, strict=True)
    ]
