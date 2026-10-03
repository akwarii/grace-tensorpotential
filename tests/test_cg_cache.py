"""Tests of the cache of ``gen_CG_matrix2`` (Clebsch-Gordan matrices in the complex basis).

Logic layer: repeated calls return the cached object, the result is read-only, the cached
value is bit-identical to a fresh evaluation, keyword and positional calls agree.
Value layer: closed-form coefficients (hand-computed) and the orthonormality of the
Clebsch-Gordan coefficients, ``sum_{m1,m2} C(L, M) C(L', M') = delta_{LL'} delta_{MM'}``
for every allowed ``L``.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest

from tensorpotential.functions.couplings import gen_CG_matrix2, gen_CG_matrix_REAL

ORTHONORMALITY_ATOL = 1e-14  # floats from sympy, sums of at most 25 terms
CLOSED_FORM_ATOL = 1e-15

TRIPLES = [(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 2, 3), (3, 2, 4)]


@pytest.fixture(autouse=True)
def _fresh_cache():
    gen_CG_matrix2.cache_clear()
    yield
    gen_CG_matrix2.cache_clear()


@pytest.mark.parametrize("l1,l2,L", TRIPLES)
def test_shape(l1, l2, L):
    assert gen_CG_matrix2(l1, l2, L).shape == (2 * L + 1, 2 * l1 + 1, 2 * l2 + 1)


def test_repeated_call_is_a_cache_hit_returning_the_same_object():
    first = gen_CG_matrix2(2, 1, 1)
    second = gen_CG_matrix2(2, 1, 1)
    assert first is second
    info = gen_CG_matrix2.cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_distinct_triples_are_distinct_entries():
    assert gen_CG_matrix2(1, 2, 1) is not gen_CG_matrix2(2, 1, 1)
    assert gen_CG_matrix2.cache_info().misses == 2


def test_keyword_and_positional_calls_agree():
    np.testing.assert_array_equal(
        gen_CG_matrix2(l1=2, l2=2, L=2), gen_CG_matrix2(2, 2, 2)
    )


def test_result_is_read_only():
    cg = gen_CG_matrix2(1, 1, 1)
    assert not cg.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cg[0, 0, 0] = 1.0


@pytest.mark.parametrize("l1,l2,L", TRIPLES)
def test_cached_value_equals_a_fresh_evaluation(l1, l2, L):
    fresh = gen_CG_matrix2.__wrapped__(l1, l2, L)
    np.testing.assert_array_equal(gen_CG_matrix2(l1, l2, L), fresh)


def test_real_matrix_does_not_modify_the_cached_one():
    before = gen_CG_matrix2(1, 1, 2).copy()
    gen_CG_matrix_REAL(1, 1, 2)
    gen_CG_matrix_REAL(1, 1, 2)
    np.testing.assert_array_equal(gen_CG_matrix2(1, 1, 2), before)


def test_closed_form_coefficients():
    # <1 m1; 1 m2 | 0 0> = (-1)^(1-m1) / sqrt(3) when m2 = -m1
    cg = gen_CG_matrix2(1, 1, 0)[0]
    expected = np.zeros((3, 3))
    for m1 in (-1, 0, 1):
        expected[m1 + 1, -m1 + 1] = (-1) ** (1 - m1) / np.sqrt(3)
    np.testing.assert_allclose(cg, expected, rtol=0, atol=CLOSED_FORM_ATOL)
    # <1 1; 1 1 | 2 2> = 1 and <1 1; 1 0 | 2 1> = 1/sqrt(2)
    cg2 = gen_CG_matrix2(1, 1, 2)
    assert cg2[2 + 2, 2, 2] == pytest.approx(1.0, rel=0, abs=CLOSED_FORM_ATOL)
    assert cg2[2 + 1, 2, 1] == pytest.approx(
        1 / np.sqrt(2), rel=0, abs=CLOSED_FORM_ATOL
    )


@pytest.mark.parametrize("l1,l2", [(1, 1), (2, 1), (2, 2)])
def test_coefficients_are_orthonormal(l1, l2):
    rows = []
    for L in range(abs(l1 - l2), l1 + l2 + 1):
        cg = gen_CG_matrix2(l1, l2, L)
        for M in range(2 * L + 1):
            rows.append(cg[M].ravel())
    gram = np.array(rows) @ np.array(rows).T
    np.testing.assert_allclose(
        gram, np.eye(len(rows)), rtol=0, atol=ORTHONORMALITY_ATOL
    )


def test_outside_the_triangle_rule_the_matrix_is_zero():
    assert not gen_CG_matrix2(1, 1, 3).any()
