"""Conformance of every neighbour-list backend with the contract of :mod:`tensorpotential.core.neighbours` (NBR1).

Physical-value layer: each backend gives exactly the ``(i, j, S)`` set of the brute-force oracle
(``tests/neighbour_oracle.py``, numpy only, written apart from every backend) and its vectors to
``NEIGHBOUR_VECTOR_F64``, on the shared structures and on the cases the contract singles out. The torch-sim lists
get the geometry the contract computes in (they honour ``pbc`` per axis themselves); what they do with the raw cell
of a slab whose third row is zero is pinned apart, since that is how a torch-sim state may hold it. Pair order
differs between backends, so every comparison is in the canonical order.

Not covered here: ``vesin_nl_ts`` (it needs the ``vesin-torch`` package, which is not in the lock file) and any GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111

from tensorpotential.core import neighbours
from tests.neighbour_oracle import brute_force_pairs
from tests.tolerances import NEIGHBOUR_VECTOR_F64
from tests_torch.structures.build_structures import load_structures

Rows = list[tuple[int, ...]]


class Case(NamedTuple):
    """A structure and the policy it is searched under."""

    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, ...]
    cutoff: float
    enforce_pbc: bool = True


def _case(atoms: Atoms, cutoff: float, *, enforce_pbc: bool = True) -> Case:
    pbc = tuple(bool(p) for p in atoms.pbc)
    return Case(atoms.positions.copy(), np.array(atoms.cell), pbc, cutoff, enforce_pbc)


def _slab(*, zero_row: bool = False) -> Case:
    slab = fcc111("Cu", size=(2, 2, 3), a=3.61, vacuum=1.0)
    cell = np.array(slab.cell)
    if zero_row:
        cell[2] = 0.0
    return Case(
        slab.positions.copy(), cell, (True, True, False), 6.0, enforce_pbc=False
    )


_RNG = np.random.default_rng(20261018)
_TRICLINIC = np.array([[2.6, 0.0, 0.0], [1.1, 2.4, 0.0], [0.7, 0.9, 2.3]])
_PAIR = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_CLUSTER = _RNG.uniform(-1.5, 1.5, (3, 3)) + np.array([
    [0, 0, 0],
    [1.0, 1.0, 0],
    [0.5, 2.0, 1.0],
])
_OPEN = (False, False, False)

CASES: dict[str, Case] = {
    **{
        name: _case(atoms, 5.0)
        for name, atoms in load_structures().items()
        if name != "mineral"
    },
    "box_below_2rc": Case(
        _RNG.uniform(0, 3.0, (3, 3)), np.eye(3) * 3.0, (True,) * 3, 4.0
    ),
    "triclinic_two_atoms": Case(
        np.array([[0.1, 0.2, 0.3], [1.9, 1.4, 1.1]]), _TRICLINIC, (True,) * 3, 4.5
    ),
    "isolated_atom": Case(np.zeros((1, 3)), np.zeros((3, 3)), _OPEN, 5.0),
    "pair_below_rc": Case(_PAIR, np.zeros((3, 3)), _OPEN, 3.0 + 1e-9),
    "pair_above_rc": Case(_PAIR, np.zeros((3, 3)), _OPEN, 3.0 - 1e-9),
    "partial_pbc": Case(
        _CLUSTER, np.diag([3.5, 4.0, 9.0]), (True, False, True), 3.2, enforce_pbc=False
    ),
    "vacuum_slab_honoured": _slab(),
    "vacuum_slab_enforced": _slab()._replace(enforce_pbc=True),
    "zero_row_open_axis": _slab(zero_row=True),
    "left_handed_cell": Case(_CLUSTER, _TRICLINIC * [[1], [1], [-1]], (True,) * 3, 3.5),
    "positions_outside_cell": Case(
        _CLUSTER + np.array([7.3, -11.2, 4.1]), _TRICLINIC, (True,) * 3, 3.5
    ),
    "rhombohedral_bi": _case(bulk("Bi", "rhombohedral", a=4.75, alpha=57.23), 6.0),
}


def _rows(i: np.ndarray, j: np.ndarray, shifts: np.ndarray) -> Rows:
    return [
        (int(a), int(b), *(int(s) for s in shift))
        for a, b, shift in zip(i, j, shifts, strict=True)
    ]


def _geometry(case: Case) -> neighbours.Geometry:
    return neighbours.effective_geometry(*case[:4], enforce_pbc=case.enforce_pbc)


def _oracle(case: Case) -> tuple[Rows, np.ndarray]:
    geometry = _geometry(case)
    symbols = ["X"] * len(geometry.positions)
    pairs = brute_force_pairs(
        geometry.positions, geometry.cell, symbols, case.cutoff, tuple(geometry.pbc)
    )
    return [(p.i, p.j, *p.shift) for p in pairs], np.array([
        p.vector for p in pairs
    ]).reshape(-1, 3)


def _contract(backend: neighbours.Backend) -> Callable[[Case], tuple[Rows, np.ndarray]]:
    def run(case: Case) -> tuple[Rows, np.ndarray]:
        found = neighbours.neighbour_pairs(
            *case[:4], backend=backend, enforce_pbc=case.enforce_pbc
        )
        result = neighbours.canonical(found)
        return _rows(result.i, result.j, result.shifts), np.asarray(result.vectors)

    return run


def _torch_sim_batch(
    name: str, cases: list[Case], cells: list[np.ndarray]
) -> list[tuple[Rows, np.ndarray]]:
    import torch
    import torch_sim.neighbors

    geometries = [_geometry(case) for case in cases]
    sizes = [len(g.positions) for g in geometries]
    mapping, system, shifts = getattr(torch_sim.neighbors, name)(
        torch.tensor(np.concatenate([g.positions for g in geometries])),
        torch.tensor(np.stack(cells)),
        torch.tensor(np.stack([g.pbc for g in geometries])),
        cases[0].cutoff,
        torch.repeat_interleave(torch.arange(len(cases)), torch.tensor(sizes)),
    )
    offsets = np.cumsum([0, *sizes])
    results = []
    for k, geometry in enumerate(geometries):
        mine = system.numpy() == k
        i, j = mapping.numpy()[:, mine] - offsets[k]
        s = shifts.numpy()[mine].round().astype(np.int64)
        order = np.lexsort((s[:, 2], s[:, 1], s[:, 0], j, i))
        i, j, s = i[order], j[order], s[order]
        results.append((
            _rows(i, j, s),
            geometry.positions[j] - geometry.positions[i] + s @ geometry.cell,
        ))
    return results


def _torch_sim(name: str) -> Callable[[Case], tuple[Rows, np.ndarray]]:
    def run(case: Case) -> tuple[Rows, np.ndarray]:
        return _torch_sim_batch(name, [case], [_geometry(case).cell])[0]

    return run


TORCH_SIM = (
    "vesin_nl",
    "torch_nl_linked_cell",
    "torch_nl_n2",
    "alchemiops_nl_n2",
    "alchemiops_nl_cell_list",
)
CONTRACT_BACKENDS: dict[neighbours.Backend, tuple[pytest.MarkDecorator, ...]] = {
    "matscipy": (),
    "vesin": (pytest.mark.vesin,),
}


def _marks(name: str) -> tuple[pytest.MarkDecorator, ...]:
    return (
        (pytest.mark.torchsim, pytest.mark.vesin)
        if name == "vesin_nl"
        else (pytest.mark.torchsim,)
    )


BACKENDS: dict[
    str,
    tuple[Callable[[Case], tuple[Rows, np.ndarray]], tuple[pytest.MarkDecorator, ...]],
] = {
    **{name: (_contract(name), marks) for name, marks in CONTRACT_BACKENDS.items()},
    **{f"ts_{name}": (_torch_sim(name), _marks(name)) for name in TORCH_SIM},
}


def _every(names: dict[str, Case]) -> list[object]:
    return [
        pytest.param(backend, name, marks=marks, id=f"{backend}-{name}")
        for backend, (_, marks) in BACKENDS.items()
        for name in names
    ]


def _backends() -> list[object]:
    return [
        pytest.param(backend, marks=marks) for backend, (_, marks) in BACKENDS.items()
    ]


def _assert_equal(
    found: tuple[Rows, np.ndarray], expected: tuple[Rows, np.ndarray]
) -> None:
    assert found[0] == expected[0]
    np.testing.assert_allclose(found[1], expected[1], **NEIGHBOUR_VECTOR_F64._asdict())


@pytest.mark.parametrize(("backend", "name"), _every(CASES))
def test_every_backend_gives_the_oracle_pairs_and_vectors(backend: str, name: str):
    _assert_equal(BACKENDS[backend][0](CASES[name]), _oracle(CASES[name]))


@pytest.mark.parametrize("backend", _backends())
def test_the_slab_has_the_measured_edge_counts(backend: str):
    # TEST7 measured 600 edges with (T, T, F) honoured and 936 once every axis is periodic.
    run = BACKENDS[backend][0]
    assert len(run(CASES["vacuum_slab_honoured"])[0]) == 600
    assert len(run(CASES["vacuum_slab_enforced"])[0]) == 936


@pytest.mark.parametrize("backend", _backends())
def test_one_atom_in_a_small_cube_has_its_analytic_self_images(backend: str):
    # In a 2 A cube with a 3 A cutoff: 6 face images at 2 A and 12 edge images at 2.83 A; the corners are at 3.46 A.
    rows, vectors = BACKENDS[backend][0](
        Case(np.zeros((1, 3)), np.eye(3) * 2.0, (True,) * 3, 3.0)
    )
    assert len(rows) == 18
    assert all(i == j == 0 for i, j, *_ in rows)
    assert (
        sorted(np.round(np.linalg.norm(vectors, axis=1), 12))
        == [2.0] * 6 + [round(np.sqrt(8), 12)] * 12
    )


@pytest.mark.parametrize("backend", _backends())
def test_wrapped_and_unwrapped_positions_give_the_same_vectors(backend: str):
    case = CASES["positions_outside_cell"]
    fractional = case.positions @ np.linalg.inv(case.cell)
    wrapped = case._replace(positions=(fractional - np.floor(fractional)) @ case.cell)
    run = BACKENDS[backend][0]

    def by_vector(found: tuple[Rows, np.ndarray]) -> list[tuple[object, ...]]:
        return sorted(
            (row[0], row[1], *np.round(v, 9)) for row, v in zip(*found, strict=True)
        )

    assert by_vector(run(case)) == by_vector(run(wrapped))


@pytest.mark.torchsim
@pytest.mark.parametrize("name", TORCH_SIM)
def test_a_batch_of_mixed_cells_gives_each_structure_its_own_list(name: str):
    if name == "vesin_nl":
        pytest.importorskip("vesin")
    names = [
        "box_below_2rc",
        "triclinic_two_atoms",
        "vacuum_slab_enforced",
        "isolated_atom",
    ]
    cases = [CASES[n]._replace(cutoff=4.0) for n in names]
    found = _torch_sim_batch(name, cases, [_geometry(c).cell for c in cases])
    for case, result in zip(cases, found, strict=True):
        _assert_equal(result, _oracle(case))


# What the torch-sim lists do with a slab whose third cell row is zero and whose third axis is open, the ASE
# convention for a 2D structure (measured, NBR1): vesin is right, three lists refuse the singular cell, and the
# naive alchemiops list returns 130 of the 600 pairs without an error. A wrapper must hand them a completed cell
# (``effective_geometry``) instead of the raw one.
RAW_ZERO_ROW_REFUSALS = {
    "torch_nl_linked_cell": "singular",
    "torch_nl_n2": "singular",
    "alchemiops_nl_cell_list": "volume == 0.0",
}


def _raw_zero_row_params() -> list[object]:
    params = []
    for name in TORCH_SIM:
        marks = list(_marks(name))
        if name == "alchemiops_nl_n2":
            reason = "alchemiops_nl_n2 drops pairs silently on a zero cell row (130 of 600 measured)"
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
        params.append(pytest.param(name, marks=marks))
    return params


@pytest.mark.parametrize("name", _raw_zero_row_params())
def test_a_raw_zero_cell_row_is_refused_or_searched_right(name: str):
    case = CASES["zero_row_open_axis"]
    if name in RAW_ZERO_ROW_REFUSALS:
        with pytest.raises(RuntimeError, match=RAW_ZERO_ROW_REFUSALS[name]):
            _torch_sim_batch(name, [case], [case.cell])
        return
    _assert_equal(_torch_sim_batch(name, [case], [case.cell])[0], _oracle(case))


@pytest.mark.torchsim
@pytest.mark.parametrize("search", ["batch_naive_neighbor_list", "batch_cell_list"])
def test_an_alchemiops_buffer_that_overflows_raises(search: str):
    import torch
    from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
    from nvalchemiops.torch import neighbors as alchemiops

    case = CASES["fcc4"]
    with pytest.raises(NeighborOverflowError, match="larger than the maximum allowed"):
        getattr(alchemiops, search)(
            positions=torch.tensor(case.positions),
            cutoff=case.cutoff,
            batch_idx=torch.zeros(len(case.positions), dtype=torch.int32),
            cell=torch.tensor(case.cell).unsqueeze(0),
            pbc=torch.ones((1, 3), dtype=torch.bool),
            max_neighbors=4,
            return_neighbor_list=True,
        )
