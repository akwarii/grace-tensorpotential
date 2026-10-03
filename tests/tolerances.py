"""The one table of tolerances for tests that compare numbers (``numpy.isclose`` semantics).

A comparison passes when ``|a - b| <= atol + rtol * |b|``. A test names a row here instead
of writing a number inline, and a tolerance is never widened to make a test pass: a row
changes only with a stated reason in the commit that changes it.
"""

from __future__ import annotations

from typing import NamedTuple


class Tolerance(NamedTuple):
    """A relative and an absolute tolerance."""

    rtol: float
    atol: float


# A handful of float64 operations (a ratio, a sum, a product): errors of a few ulp.
FLOAT64_ARITHMETIC = Tolerance(rtol=1e-12, atol=1e-15)

# Rotation, translation and permutation covariance of a float64 network with random weights:
# the observed error is about 1e-11 on outputs of order 1 (summation order differs between runs).
COVARIANCE_F64 = Tolerance(rtol=1e-9, atol=1e-9)

# Forces against a central finite difference of the energy in float64: step 1e-4 A gives a
# truncation error of order step**2 times the third derivative, about 1e-8 for the test cluster.
FINITE_DIFFERENCE_F64 = Tolerance(rtol=1e-6, atol=1e-7)


def sample_std_rtol(n_samples: int, n_sigma: float = 5.0) -> float:
    """Relative tolerance for the sample standard deviation of ``n_samples`` normal draws.

    The standard error of a sample standard deviation is ``1 / sqrt(2 n)`` of the true value,
    so a bound of ``n_sigma`` standard errors fails a correct initialiser only about once in
    ``1.7e6`` runs for 5 sigma; with a fixed seed the outcome is deterministic anyway.
    """
    return n_sigma / (2.0 * n_samples) ** 0.5

# A float64 least-squares solve (``numpy.linalg.lstsq``) on a small well-conditioned system whose
# exact solution is known: the observed error is about 1e-14 on values of order 1 to 10.
LEAST_SQUARES_F64 = Tolerance(rtol=1e-10, atol=1e-10)

# A float32 network with random weights (a few matrix products, an RMS normalisation, activations) against
# a float64 numpy evaluation of the same stored weights: float32 has a unit roundoff of 6e-8, and a handful of
# accumulations of order-one terms give errors of about 1e-6.
FLOAT32_NETWORK = Tolerance(rtol=1e-5, atol=1e-5)

# A few element-wise float32 operations (square, sum of three terms, square root, ratio) against the same
# formula in float64 on the same float32 inputs: unit roundoff 6e-8 per operation, errors of a few ulp.
FLOAT32_ELEMENTWISE = Tolerance(rtol=1e-6, atol=1e-7)
