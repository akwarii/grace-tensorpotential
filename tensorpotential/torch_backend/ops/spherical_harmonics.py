"""Real spherical harmonics of unit vectors: the maths of TensorFlow's ``SphericalHarmonic``.

:func:`real_spherical_harmonics` evaluates the polynomials of
:mod:`tensorpotential.torch_backend.plans.spherical_harmonics` (the formulas and the conventions are there): ``Y_00 = 1``,
``l = 1`` is ``sqrt(3) (y, z, x)``, columns in the order ``l*l + l + m``. The loop over ``(l, m)`` is unrolled in
Python, as in TensorFlow; every operation is elementwise, so there are no data-dependent shapes and no host
synchronisation. The input is not normalised: the formulas assume a unit vector, as TensorFlow's do.

The result has the dtype of the input; the instruction twin casts its input to the geometry dtype first (float64, G1),
as TensorFlow computes the harmonics in float64 whatever the parameter dtype.

Layer: ops; imports torch, the errors, the dtype policy and the plan.
"""

from __future__ import annotations

import torch

from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.ops.precision import check_floating
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    SphericalHarmonicsPlan,
)


def _powers(
    x: torch.Tensor, y: torch.Tensor, lmax: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Real and imaginary parts of ``(x + i y)**m`` for ``m = 0 .. lmax``."""
    re, im = [torch.ones_like(x)], [torch.zeros_like(x)]
    for m in range(lmax):
        re.append(x * re[m] - y * im[m])
        im.append(x * im[m] + y * re[m])
    return re, im


def _legendre(
    z: torch.Tensor, plan: SphericalHarmonicsPlan
) -> dict[tuple[int, int], torch.Tensor]:
    """``Q[l, m](z)`` for ``0 <= m <= l <= lmax`` (the associated Legendre functions over ``sin(theta)**m``)."""
    q: dict[tuple[int, int], torch.Tensor] = {}
    for m in range(plan.lmax + 1):
        q[m, m] = torch.full_like(z, float(plan.diagonal[m]))
        for l in range(m + 1, plan.lmax + 1):
            q[l, m] = float(plan.step_z[l, m]) * z * q[l - 1, m]
            if l >= m + 2:
                q[l, m] = q[l, m] - float(plan.step_back[l, m]) * q[l - 2, m]
    return q


def real_spherical_harmonics(
    unit_vectors: torch.Tensor, plan: SphericalHarmonicsPlan
) -> torch.Tensor:
    """Return the real spherical harmonics of ``unit_vectors`` up to degree ``plan.lmax``, scaled so that ``Y_00 = 1``.

    Parameters
    ----------
    unit_vectors
        ``[..., 3]``: unit directions ``(x, y, z)`` (bond directions, ``ScaledBondVector``), any floating dtype.
    plan
        The columns and coefficients (``plans.spherical_harmonics.spherical_harmonics_plan``), built once by the
        caller rather than per call.

    Returns
    -------
    torch.Tensor
        ``[..., (plan.lmax + 1)**2]``, column ``l*l + l + m`` for ``m = -l .. l``, with the dtype and device of the input;
        differentiable with respect to it. ``mean over the sphere of Y[l, m] Y[l', m'] = delta``: orthonormal up to
        the factor ``4 pi`` of TensorFlow's default (``norm = False``).

    Raises
    ------
    ShapeError
        When the last axis is not of size 3.
    DtypeError
        When the input is not floating.
    """
    check_floating(unit_vectors)
    if unit_vectors.dim() == 0 or unit_vectors.shape[-1] != 3:
        msg = f"unit_vectors must have a last axis of size 3 (x, y, z), got shape {tuple(unit_vectors.shape)}"
        raise ShapeError(msg)
    x, y, z = unit_vectors.unbind(-1)
    re, im = _powers(x, y, plan.lmax)
    q = _legendre(z, plan)
    columns = []
    for l, m, scale in zip(
        plan.l.tolist(), plan.m.tolist(), plan.scale.tolist(), strict=True
    ):
        azimuthal = re[m] if m >= 0 else im[-m]
        columns.append(scale * q[l, abs(m)] * azimuthal)
    return torch.stack(columns, dim=-1)
