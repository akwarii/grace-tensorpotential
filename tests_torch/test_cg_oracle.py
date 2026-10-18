"""GRACE's real Clebsch-Gordan couplings against two oracles that share no code with them (ORACLE1).

Oracle one, ``tests_torch/cg_oracle.py``: Racah's formula and the definition of the real harmonics. Oracle two:
cuEquivariance's ``clebsch_gordan`` (numpy-level, no kernels; tests marked ``cueq``). Both are compared, as whole
``[2L + 1, 2 l1 + 1, 2 l2 + 1]`` tensors in their index order, with ``gen_CG_matrix_REAL`` for every ``(l1, l2, L)``
the two shipped yamls couple (tiny and faithful tiers, collected from the coupling metadata the unmodified TF
``__init__`` builds under the stand-in). The physical values: the coupled product of two harmonics of one direction
is ``k`` times the harmonic of degree ``L`` with one ``k`` for every ``M`` (and ``CG_normalization_dict`` holds
``1 / |k|``), and a coupled feature of two directions rotates with the Wigner matrix of degree ``L``.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorpotential.functions.couplings import (
    CG_normalization_dict,
    gen_CG_matrix_REAL,
)
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)
from tests_torch import cg_oracle
from tests_torch.golden import GOLDEN
from tests_torch.golden.tolerances import FEATURE_F64, assert_close
from tests_torch.tools import tfstub

RNG_SEED = 20261018
YAMLS = (
    GOLDEN / "yamls" / "omat_tiny.yaml",
    GOLDEN / "yamls" / "large_tiny.yaml",
    GOLDEN.parent / "fixtures" / "yamls" / "omat_faithful.yaml",
    GOLDEN.parent / "fixtures" / "yamls" / "large_faithful.yaml",
)
COUPLING_CLASSES = ("ProductFunction", "SingleParticleBasisFunctionEquivariantInd")
UP_TO_4 = [
    (l1, l2, big_l)
    for l1 in range(5)
    for l2 in range(5)
    for big_l in range(abs(l1 - l2), min(l1 + l2, 4) + 1)
]
"""Every triple with all degrees at most 4 (65): the prototype set of the issue."""


@functools.cache
def _triples_of(yaml: Path) -> frozenset[tuple[int, int, int]]:
    found = set()
    for instruction in tfstub.load_model(yaml).values():
        if type(instruction).__name__ in COUPLING_CLASSES:
            meta = instruction.coupling_meta_data
            found |= set(
                zip(
                    meta.l1.astype(int),
                    meta.l2.astype(int),
                    meta.l.astype(int),
                    strict=True,
                )
            )
    return frozenset(found)


def _grace(l1: int, l2: int, big_l: int) -> np.ndarray:
    return np.asarray(gen_CG_matrix_REAL(l1, l2, big_l), dtype=np.float64)


def _directions(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------------ the oracle itself


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0, 0, 0, 0, 0, 0), 1.0),
        ((1, 0, 1, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((1, 1, 1, -1, 0, 0), 1.0 / math.sqrt(3.0)),
        ((1, 1, 1, 0, 2, 1), 1.0 / math.sqrt(2.0)),
        ((1, 0, 1, 0, 2, 0), math.sqrt(2.0 / 3.0)),
        ((2, 0, 2, 0, 0, 0), 1.0 / math.sqrt(5.0)),
        ((2, 1, 1, -1, 1, 0), math.sqrt(3.0 / 10.0)),
    ],
)
def test_hand_values_of_racahs_formula(args, expected):
    assert cg_oracle.clebsch_gordan(*args) == pytest.approx(expected, rel=1e-15, abs=0)


@pytest.mark.parametrize(
    "args",
    [(1, 1, 1, 1, 1, 1), (1, 0, 1, 0, 3, 0), (2, 2, 1, 0, 1, 2), (1, 2, 1, -1, 1, 1)],
    ids=["m not m1 + m2", "outside the triangle", "|m1| above j1", "|m| above j"],
)
def test_forbidden_components_are_zero(args):
    assert cg_oracle.clebsch_gordan(*args) == 0.0


@pytest.mark.parametrize(("l1", "l2"), [(1, 1), (2, 3), (4, 4)])
def test_the_complex_coefficients_are_an_orthogonal_change_of_basis(l1, l2):
    stacked = np.concatenate([
        cg_oracle.complex_cg(l1, l2, big_l).reshape(2 * big_l + 1, -1)
        for big_l in range(abs(l1 - l2), l1 + l2 + 1)
    ])
    assert_close(stacked @ stacked.T, np.eye(stacked.shape[0]), FEATURE_F64, "rows")
    assert_close(stacked.T @ stacked, np.eye(stacked.shape[1]), FEATURE_F64, "columns")


@pytest.mark.parametrize("degree", range(5))
def test_the_real_basis_is_unitary(degree):
    u = cg_oracle.real_basis(degree)
    assert_close(
        np.abs(u @ u.conj().T - np.eye(2 * degree + 1)),
        np.zeros((2 * degree + 1,) * 2),
        FEATURE_F64,
    )


@pytest.mark.parametrize("triple", UP_TO_4, ids=str)
def test_the_phase_makes_every_coupling_real(triple):
    imaginary = cg_oracle.coupled_real_basis(*triple).imag
    assert_close(imaginary, np.zeros_like(imaginary), FEATURE_F64, f"Im of {triple}")


def test_the_harmonics_of_degree_one_are_sqrt3_y_z_x():
    v = _directions(20, np.random.default_rng(RNG_SEED))
    assert_close(
        cg_oracle.real_harmonics(v, 1), math.sqrt(3.0) * v[:, [1, 2, 0]], FEATURE_F64
    )


def test_the_oracle_harmonics_are_the_twins_harmonics():
    v = _directions(200, np.random.default_rng(RNG_SEED))
    lmax = 6
    oracle = np.concatenate(
        [cg_oracle.real_harmonics(v, d) for d in range(lmax + 1)], axis=-1
    )
    twin = real_spherical_harmonics(
        torch.as_tensor(v, dtype=torch.float64), spherical_harmonics_plan(lmax)
    )
    assert_close(twin.numpy(), oracle, FEATURE_F64, "twin against oracle harmonics")


def test_wigner_matrices_are_orthogonal_and_compose():
    rng = np.random.default_rng(RNG_SEED)
    points = _directions(40, rng)
    q1, q2 = cg_oracle.random_rotation(rng), cg_oracle.random_rotation(rng)
    assert np.linalg.det(q1) == pytest.approx(1.0, rel=1e-14)
    for degree in range(5):
        d1, d2 = (cg_oracle.wigner_d(q, degree, points) for q in (q1, q2))
        assert_close(
            d1 @ d1.T, np.eye(2 * degree + 1), FEATURE_F64, f"orthogonal, l {degree}"
        )
        assert_close(
            cg_oracle.wigner_d(q1 @ q2, degree, points),
            d1 @ d2,
            FEATURE_F64,
            f"composition, l {degree}",
        )


# ------------------------------------------------------------------ GRACE against the first-principles oracle


def test_every_triple_the_yamls_couple_is_compared():
    # 31 triples in the omat yaml, 52 in the large one (a superset), the same in both tiers: all among the 65 that
    # every test of this file runs on
    omat_tiny, large_tiny, omat_faithful, large_faithful = (
        _triples_of(y) for y in YAMLS
    )
    assert (len(omat_tiny), len(large_tiny)) == (31, 52)
    assert omat_tiny == omat_faithful < large_tiny == large_faithful <= set(UP_TO_4)


@pytest.mark.parametrize("triple", UP_TO_4, ids=str)
def test_grace_couplings_equal_the_oracle(triple):
    assert_close(
        _grace(*triple),
        cg_oracle.real_cg(*triple),
        FEATURE_F64,
        f"gen_CG_matrix_REAL{triple}",
    )


@pytest.mark.parametrize("triple", UP_TO_4, ids=str)
def test_the_coupled_product_of_two_harmonics_is_k_times_the_harmonic_of_degree_l(
    triple,
):
    l1, l2, big_l = triple
    v = _directions(30, np.random.default_rng(RNG_SEED))
    product = np.einsum(
        "Mab,na,nb->nM",
        _grace(*triple),
        cg_oracle.real_harmonics(v, l1),
        cg_oracle.real_harmonics(v, l2),
    )
    k = cg_oracle.coupling_constant(*triple)
    assert_close(
        product,
        k * cg_oracle.real_harmonics(v, big_l),
        FEATURE_F64,
        f"product of {triple}",
    )
    if k:
        key = (l2, l1, big_l) if l1 < l2 else triple
        assert CG_normalization_dict[key] == pytest.approx(1.0 / abs(k), rel=1e-14)


def test_the_normalisation_table_holds_one_over_k_for_every_even_triple():
    for key, value in CG_normalization_dict.items():
        assert sum(key) % 2 == 0, key
        assert value == pytest.approx(
            1.0 / abs(cg_oracle.coupling_constant(*key)), rel=1e-14
        ), key


@pytest.mark.parametrize("triple", UP_TO_4, ids=str)
def test_a_coupled_feature_of_two_directions_rotates_with_the_wigner_matrix(triple):
    l1, l2, big_l = triple
    rng = np.random.default_rng(RNG_SEED)
    x, y, points = _directions(10, rng), _directions(10, rng), _directions(40, rng)
    q = cg_oracle.random_rotation(rng)
    c = _grace(*triple)

    def feature(a, b):
        return np.einsum(
            "Mab,na,nb->nM",
            c,
            cg_oracle.real_harmonics(a, l1),
            cg_oracle.real_harmonics(b, l2),
        )

    rotated = feature(x @ q.T, y @ q.T)
    assert_close(
        rotated,
        feature(x, y) @ cg_oracle.wigner_d(q, big_l, points).T,
        FEATURE_F64,
        f"{triple}",
    )


# ------------------------------------------------------------------ GRACE against cuEquivariance


def _cueq(l1: int, l2: int, big_l: int) -> np.ndarray:
    cue = pytest.importorskip("cuequivariance")
    table = np.asarray(cue.clebsch_gordan(cue.SO3(l1), cue.SO3(l2), cue.SO3(big_l)))
    assert table.shape == (1, 2 * l1 + 1, 2 * l2 + 1, 2 * big_l + 1)
    return table[0].transpose(2, 0, 1)


@pytest.mark.cueq
@pytest.mark.parametrize("triple", UP_TO_4, ids=str)
def test_cuequivariance_equals_grace_and_the_oracle(triple):
    assert_close(
        _cueq(*triple),
        _grace(*triple),
        FEATURE_F64,
        f"cuEquivariance against GRACE {triple}",
    )
    assert_close(
        _cueq(*triple),
        cg_oracle.real_cg(*triple),
        FEATURE_F64,
        f"cuEquivariance against the oracle {triple}",
    )
