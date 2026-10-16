"""The one tolerance table of the torch backend: every comparison of a torch output with an oracle or a golden.

A row is used exactly as ``numpy.isclose``: ``|actual - expected| <= atol + rtol * |expected|``, element by element,
``expected`` the reference. The absolute part is needed because a purely relative test fails on the force components
of a symmetric structure, which are a few 1e-15 eV/A where the exact value is 0. A test names a row (``FORCES_F64``,
or ``row_for("float64", "forces")`` in a parametrised test) and never writes a number of its own.

:func:`deviations` measures a comparison without asserting anything (the headroom a row leaves is read off it, not
found by loosening the row); :func:`assert_close` is the assertion. The table and the golden anchors are edit-locked
(``tests_torch/golden/LOCK.sha256``, checked by ``tests_torch/test_tolerances.py``): an existing row is never widened to
make a test pass, and a new or changed row comes in a commit of its own that states the measurement behind it.

How the rows are derived (measured on the committed tiny anchors; numbers in the comments on issue TORCH2):

* **float64**: the floor is the summation order. Forces rebuilt from the stored pair forces in another order, the
  energy and the virial summed again, differ from TF's by at most 2 float64 eps of the largest term. Each row is 100
  times that, ``rtol = 5e-14``, with ``atol = rtol * scale``, ``scale`` the size of the terms that are summed into the
  quantity (1 for energies, forces and stress, 10 for the virial and instruction outputs, whose terms are larger).
* **float32**: the floor is the float32 rounding of the chain. TF with float32 parameters differs from TF with float64
  parameters (same weights) by at most 28 float32 eps of the energy and 11 eps of the largest element of an
  instruction output; a twin and TF in float32 can each be that far from the float64 value, and the goldens were made
  on another CPU than the one a test runs on. Each row is ``rtol = 256 eps``, with the same scales.
* **across CPUs**: the same float32 torch computation on two CPUs; MACE measured 13 to 33 float32 eps, the row is 64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike

Dtype = Literal["float64", "float32"]
Quantity = Literal["energy", "forces", "virial", "stress", "feature", "any"]

EPS64: Final = float(np.finfo(np.float64).eps)
EPS32: Final = float(np.finfo(np.float32).eps)
F64_RTOL: Final = 5e-14
"""100 times the float64 summation-order floor (2 eps of the largest term)."""
F32_RTOL: Final = 256 * EPS32
"""About 10 times the float32 spread of TF (28 eps) doubled for two implementations; 3.05e-5."""
CROSS_CPU_RTOL: Final = 64 * EPS32
"""Twice the largest float32 spread MACE measured between CPUs (33 eps); 7.6e-6."""


@dataclass(frozen=True)
class Row:
    """A named tolerance: ``|actual - expected| <= atol + rtol * |expected|``."""

    name: str
    dtype: Dtype
    quantity: Quantity
    rtol: float
    atol: float
    unit: str
    rationale: str


@dataclass(frozen=True)
class Deviation:
    """How far ``actual`` is from ``expected`` under a row.

    ``ratio`` is the largest ``|actual - expected| / (atol + rtol * |expected|)``, at ``index``, where the two values are
    ``actual_at`` and ``expected_at``: the comparison passes when it is at most 1, and ``headroom`` (its inverse) says
    how many times the error could grow before it fails.
    """

    max_abs: float
    max_rel: float
    ratio: float
    index: tuple[int, ...]
    actual_at: float = 0.0
    expected_at: float = 0.0

    @property
    def headroom(self) -> float:
        """How many times the worst error fits in the bound (infinite when the arrays are equal)."""
        return float("inf") if self.ratio == 0.0 else 1.0 / self.ratio


def _row(
    dtype: Dtype, quantity: Quantity, rtol: float, scale: float, unit: str, why: str
) -> Row:
    suffix = "F64" if dtype == "float64" else "F32"
    return Row(
        f"{quantity.upper()}_{suffix}", dtype, quantity, rtol, rtol * scale, unit, why
    )


_F64_WHY = "float64 summation-order floor (2 eps of the largest term) times 100"
_F32_WHY = "float32 spread of TF (at most 28 eps) for two implementations on two CPUs, times about 5"

ENERGY_F64 = _row(
    "float64", "energy", F64_RTOL, 1.0, "eV", _F64_WHY + "; per structure or per atom"
)
FORCES_F64 = _row(
    "float64",
    "forces",
    F64_RTOL,
    1.0,
    "eV/A",
    _F64_WHY + "; symmetric forces are 7e-15",
)
VIRIAL_F64 = _row(
    "float64", "virial", F64_RTOL, 10.0, "eV", _F64_WHY + "; terms r (x) f up to 10 eV"
)
STRESS_F64 = _row(
    "float64",
    "stress",
    F64_RTOL,
    1.0,
    "eV/A^3",
    _F64_WHY + "; the virial over a volume",
)
FEATURE_F64 = _row(
    "float64",
    "feature",
    F64_RTOL,
    10.0,
    "1",
    _F64_WHY + "; instruction outputs up to 50",
)
ENERGY_F32 = _row(
    "float32", "energy", F32_RTOL, 1.0, "eV", _F32_WHY + "; measured 4.8e-7 eV"
)
FORCES_F32 = _row(
    "float32", "forces", F32_RTOL, 1.0, "eV/A", _F32_WHY + "; measured 8.0e-7 eV/A"
)
VIRIAL_F32 = _row(
    "float32", "virial", F32_RTOL, 10.0, "eV", _F32_WHY + "; measured 3.7e-6 eV"
)
STRESS_F32 = _row(
    "float32", "stress", F32_RTOL, 1.0, "eV/A^3", _F32_WHY + "; measured 1.2e-7 eV/A^3"
)
FEATURE_F32 = _row(
    "float32", "feature", F32_RTOL, 10.0, "1", _F32_WHY + "; measured 1.7e-6"
)
CROSS_CPU_F32 = Row(
    "CROSS_CPU_F32",
    "float32",
    "any",
    CROSS_CPU_RTOL,
    CROSS_CPU_RTOL,
    "1",
    "one float32 torch computation of order-one values on two CPUs: twice the 33 eps MACE measured",
)

ROWS: Final = (
    ENERGY_F64,
    FORCES_F64,
    VIRIAL_F64,
    STRESS_F64,
    FEATURE_F64,
    ENERGY_F32,
    FORCES_F32,
    VIRIAL_F32,
    STRESS_F32,
    FEATURE_F32,
    CROSS_CPU_F32,
)
"""Every row of the table."""


_BY_DTYPE_AND_QUANTITY: Final = {(r.dtype, r.quantity): r for r in ROWS}


def row_for(dtype: Dtype, quantity: Quantity) -> Row:
    """Return the row of ``quantity`` for a model whose parameter dtype is ``dtype``.

    Raises
    ------
    KeyError
        When the table has no such row.
    """
    return _BY_DTYPE_AND_QUANTITY[dtype, quantity]


def _as_float64(values: ArrayLike) -> np.ndarray:
    # a torch tensor: detached and on the CPU first
    as_numpy = getattr(values, "numpy", None)
    array = as_numpy(force=True) if callable(as_numpy) else values
    return np.asarray(array, dtype=np.float64)


def _ratio(diff: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """``diff / bound``, with 0 where ``diff`` is 0 and infinity where only ``bound`` is."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(diff == 0.0, 0.0, diff / bound)


def deviations(actual: ArrayLike, expected: ArrayLike, row: Row) -> Deviation:
    """Measure ``actual`` against ``expected`` under ``row`` without asserting.

    Parameters
    ----------
    actual
        The value under test: a numpy array or a torch tensor (any dtype, compared in float64).
    expected
        The reference, of the same shape.
    row
        The tolerance row.

    Raises
    ------
    ValueError
        When the shapes differ (nothing is broadcast).
    """
    a, b = _as_float64(actual), _as_float64(expected)
    if a.shape != b.shape:
        msg = f"{row.name}: shapes differ, {a.shape} against {b.shape}"
        raise ValueError(msg)
    if a.size == 0:
        return Deviation(0.0, 0.0, 0.0, ())
    diff = np.abs(a - b)
    ratio = _ratio(diff, row.atol + row.rtol * np.abs(b))
    worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return Deviation(
        float(diff.max()),
        float(_ratio(diff, np.abs(b)).max()),
        float(ratio[worst]),
        tuple(map(int, worst)),
        float(a[worst]),
        float(b[worst]),
    )


def assert_close(
    actual: ArrayLike, expected: ArrayLike, row: Row, what: str = "value"
) -> Deviation:
    """Assert that ``actual`` is within ``row`` of ``expected`` and return the deviation.

    The message names the row, the worst element and the headroom.
    """
    found = deviations(actual, expected, row)
    if found.ratio > 1.0:
        msg = (
            f"{what} not within {row.name} (rtol {row.rtol:.1e}, atol {row.atol:.1e} {row.unit}): at "
            f"{found.index} {found.actual_at:.17g} against {found.expected_at:.17g}, {found.ratio:.3g} times the bound"
        )
        raise AssertionError(msg)
    return found
