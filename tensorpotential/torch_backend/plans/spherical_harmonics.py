"""Columns and coefficients of the real spherical harmonics of the twins, in pure numpy.

The harmonics are those of TensorFlow's ``SphericalHarmonic`` (spec sheet ``SphericalHarmonic.md``): real, without the
Condon-Shortley phase, scaled by ``sqrt(4 pi)`` so that ``Y_00 = 1``, one column per ``(l, m)`` in the order
``l*l + l + m`` (G7: ``l`` ascending, ``m`` from ``-l`` to ``l``). Written as polynomials of the unit vector
``(x, y, z)``:

.. code-block:: text

    Y[l, 0]  = K[l, 0] Q[l, 0](z)
    Y[l, m]  = K[l, m] Q[l, m](z) Re (x + i y)**m      (m > 0)
    Y[l, -m] = K[l, m] Q[l, m](z) Im (x + i y)**m      (m > 0)

    K[l, m]  = sqrt((2 l + 1) (l - m)! / (l + m)!), times sqrt(2) when m > 0
    Q[m, m]  = (2 m - 1)!!                               (1 for m = 0)
    Q[m+1, m] = (2 m + 1) z Q[m, m]
    Q[l, m]  = ((2 l - 1) z Q[l-1, m] - (l + m - 1) Q[l-2, m]) / (l - m)      (l >= m + 2)

``Q[l, m] sin(theta)**m`` is the associated Legendre function without the Condon-Shortley phase, so the whole is a
polynomial: no square root of ``1 - z**2`` and no angle appear, and the zero vector (which ``ScaledBondVector`` gives for
a zero bond) yields the harmonics of the point ``z = 0`` with ``(x + i y)**m = 0``, as in TensorFlow. TensorFlow reaches the same
values through a complex recursion with other coefficients (``functions/spherical_harmonics.py``); the twin is checked
against its outputs, not against its tables, which no golden stores.

The table of the output columns that the downstream instructions read, ``coupling_meta_data``, is
:func:`uncoupled_meta_data`.

Layer: plans; imports numpy, pandas and the errors only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tensorpotential.torch_backend.errors import ShapeError


@dataclass(frozen=True)
class SphericalHarmonicsPlan:
    """The columns and coefficients of the real spherical harmonics up to degree ``lmax``.

    Attributes
    ----------
    lmax
        Highest degree.
    l, m
        ``[(lmax + 1)**2]`` int arrays: the degree and order of each output column.
    scale
        ``[(lmax + 1)**2]`` float64: ``K[l, |m|]`` of each column.
    diagonal
        ``[lmax + 1]`` float64: ``Q[m, m] = (2 m - 1)!!``.
    step_z, step_back
        ``[lmax + 1, lmax + 1]`` float64, indexed ``[l, m]``: ``Q[l, m] = step_z[l, m] z Q[l-1, m] - step_back[l, m]
        Q[l-2, m]``; ``step_z`` is set for ``l >= m + 1``, ``step_back`` for ``l >= m + 2`` (the term is absent for
        ``l = m + 1``), both 0 elsewhere.
    """

    lmax: int
    l: np.ndarray
    m: np.ndarray
    scale: np.ndarray
    diagonal: np.ndarray
    step_z: np.ndarray
    step_back: np.ndarray


def _scale(l: int, m: int) -> float:
    ratio = math.factorial(l - m) / math.factorial(l + m)
    return math.sqrt((2 * l + 1) * ratio * (2 if m else 1))


def spherical_harmonics_plan(lmax: int) -> SphericalHarmonicsPlan:
    """Return the plan of the real spherical harmonics up to degree ``lmax``.

    Parameters
    ----------
    lmax
        Highest degree, at least 0 (the shipped models use 4).

    Raises
    ------
    ShapeError
        When ``lmax`` is negative or not an integer.
    """
    if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 0:
        msg = f"lmax must be an integer of at least 0, got {lmax!r}"
        raise ShapeError(msg)
    l = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
    m = np.concatenate([np.arange(-d, d + 1) for d in range(lmax + 1)])
    scale = np.array([_scale(int(d), abs(int(o))) for d, o in zip(l, m, strict=True)])
    diagonal = np.array(
        [math.prod(range(2 * o - 1, 0, -2)) for o in range(lmax + 1)], dtype=np.float64
    )
    step_z = np.zeros((lmax + 1, lmax + 1))
    step_back = np.zeros((lmax + 1, lmax + 1))
    for o in range(lmax + 1):
        for d in range(o + 1, lmax + 1):
            step_z[d, o] = (2 * d - 1) / (d - o)
        for d in range(o + 2, lmax + 1):
            step_back[d, o] = (d + o - 1) / (d - o)
    return SphericalHarmonicsPlan(lmax, l, m, scale, diagonal, step_z, step_back)


def uncoupled_meta_data(plan: SphericalHarmonicsPlan) -> pd.DataFrame:
    """Return the ``coupling_meta_data`` of a ``SphericalHarmonic``: one row per output column, in output order.

    The columns are TF's ``init_uncoupled_meta_data`` (``instructions/base.py``): ``l``, ``m``, ``hist`` (empty),
    ``parity`` (``(-1)**l``) and ``sum_of_ls`` (``l``). The ``SingleParticleBasisFunction*`` instructions build their
    coupling tables from it.
    """
    l = plan.l.tolist()
    return pd.DataFrame(
        {
            "l": l,
            "m": plan.m.tolist(),
            "hist": [""] * len(l),
            "parity": [1 if d % 2 == 0 else -1 for d in l],
            "sum_of_ls": l,
        }
    )
