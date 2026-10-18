"""A first-principles oracle of the real Clebsch-Gordan couplings and real harmonics of GRACE (ORACLE1).

Nothing here calls ``functions.couplings`` or ``sympy.physics``: the complex coefficients come from Racah's formula
in exact rational arithmetic, the complex harmonics from scipy (Condon-Shortley phase), and the change of basis from
the definition of the real harmonics: ``R[l, m] = sqrt(4 pi) sqrt(2) (-1)**m Re Y[l, m]`` for ``m > 0``,
``sqrt(4 pi) sqrt(2) (-1)**m Im Y[l, |m|]`` for ``m < 0`` and ``sqrt(4 pi) Y[l, 0]``, columns ``m = -l .. l``.

GRACE's real coupling of ``(l1, l2)`` to ``L`` is the complex one in that real basis times the phase
``(-i)**(l1 + l2 - L)``, which makes it real for either parity of ``l1 + l2 + L`` (the convention of
``c2r_harm_matrix``: each real basis carries ``(-i)**l``). Layout ``[2L + 1, 2 l1 + 1, 2 l2 + 1]``, as
``gen_CG_matrix_REAL``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.special import sph_harm_y


def _triangle(l1: int, l2: int, big_l: int) -> bool:
    return abs(l1 - l2) <= big_l <= l1 + l2


def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """``<j1 m1 j2 m2 | j m>`` for integer angular momenta, by Racah's formula, exact up to the final square root."""
    if (
        m != m1 + m2
        or not _triangle(j1, j2, j)
        or max(abs(m1) - j1, abs(m2) - j2, abs(m) - j) > 0
    ):
        return 0.0
    f = math.factorial
    prefactor = Fraction(
        (2 * j + 1) * f(j + j1 - j2) * f(j - j1 + j2) * f(j1 + j2 - j),
        f(j1 + j2 + j + 1),
    ) * (f(j + m) * f(j - m) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    total = Fraction(0)
    for k in range(j1 + j2 - j + 1):
        rest = (
            j1 + j2 - j - k,
            j1 - m1 - k,
            j2 + m2 - k,
            j - j2 + m1 + k,
            j - j1 - m2 + k,
        )
        if min(rest) >= 0:
            total += Fraction((-1) ** k, f(k) * math.prod(f(r) for r in rest))
    square = prefactor * total * total
    return math.copysign(math.sqrt(square.numerator / square.denominator), total)


def complex_cg(l1: int, l2: int, big_l: int) -> np.ndarray:
    """The complex-basis coefficients ``[2L + 1, 2 l1 + 1, 2 l2 + 1]``, indices ``m + l``."""
    table = np.zeros((2 * big_l + 1, 2 * l1 + 1, 2 * l2 + 1))
    for m in range(-big_l, big_l + 1):
        for m1 in range(max(-l1, m - l2), min(l1, m + l2) + 1):
            table[m + big_l, m1 + l1, m - m1 + l2] = clebsch_gordan(
                l1, m1, l2, m - m1, big_l, m
            )
    return table


def real_basis(degree: int) -> np.ndarray:
    """The unitary ``U`` with ``R[l] = U @ Y[l]`` (rows: real ``m``, columns: complex ``m``, both ``-l .. l``)."""
    u = np.zeros((2 * degree + 1, 2 * degree + 1), dtype=complex)
    u[degree, degree] = 1.0
    s = 1.0 / math.sqrt(2.0)
    for a in range(1, degree + 1):
        sign = (-1) ** a
        u[degree + a, degree + a], u[degree + a, degree - a] = sign * s, s
        u[degree - a, degree + a], u[degree - a, degree - a] = -1j * sign * s, 1j * s
    return u


def coupled_real_basis(l1: int, l2: int, big_l: int) -> np.ndarray:
    """The coupling in the real basis with GRACE's phase, as a complex array (its imaginary part vanishes)."""
    phase = (-1j) ** (l1 + l2 - big_l)
    return phase * np.einsum(
        "PM,Mij,ai,bj->Pab",
        real_basis(big_l),
        complex_cg(l1, l2, big_l),
        real_basis(l1).conj(),
        real_basis(l2).conj(),
    )


def real_cg(l1: int, l2: int, big_l: int) -> np.ndarray:
    """GRACE's real coupling ``[2L + 1, 2 l1 + 1, 2 l2 + 1]`` of ``(l1, l2)`` to ``L``."""
    return coupled_real_basis(l1, l2, big_l).real


def coupling_constant(l1: int, l2: int, big_l: int) -> float:
    """``k`` of ``sum C[M, m1, m2] R[l1, m1](x) R[l2, m2](x) = k R[L, M](x)`` (zero for odd ``l1 + l2 + L``).

    The product of two complex harmonics of one direction expands with ``sqrt((2 l1 + 1) (2 l2 + 1) / (4 pi (2 L +
    1))) <l1 0 l2 0 | L 0>``; the ``sqrt(4 pi)`` of the real harmonics and the phase of :func:`coupled_real_basis`
    (real for an even sum) complete it.
    """
    if (l1 + l2 + big_l) % 2:
        return 0.0
    scale = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * big_l + 1))
    return (
        (-1) ** ((l1 + l2 - big_l) // 2)
        * scale
        * clebsch_gordan(l1, 0, l2, 0, big_l, 0)
    )


def real_harmonics(directions: np.ndarray, degree: int) -> np.ndarray:
    """The real harmonics ``[n, 2l + 1]`` of degree ``l`` of unit vectors ``[n, 3]``, from scipy's complex ones."""
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    y = np.stack(
        [sph_harm_y(degree, m, theta, phi) for m in range(-degree, degree + 1)], axis=-1
    )
    return (math.sqrt(4.0 * math.pi) * y @ real_basis(degree).T).real


def wigner_d(rotation: np.ndarray, degree: int, directions: np.ndarray) -> np.ndarray:
    """``D`` with ``R[l](rotation @ x) = D @ R[l](x)``, fitted on ``directions`` (at least ``2l + 1`` of them)."""
    before = real_harmonics(directions, degree)
    after = real_harmonics(directions @ rotation.T, degree)
    solution, *_ = np.linalg.lstsq(before, after, rcond=None)
    return solution.T


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A proper rotation matrix from the QR decomposition of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q
