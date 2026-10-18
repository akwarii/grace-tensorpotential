"""Twins of TF's bond geometry: ``BondLength``, ``ScaledBondVector`` and ``SphericalHarmonic``.

Spec sheets of the three classes, section 5. With ``v`` the bond vectors ``[bonds, 3]``:

.. code-block:: text

    BondLength          d = sqrt(sum_k v[:, k]**2 + 1e-10)           ([bonds, 1]; the zero vector has length 1e-5)
    ScaledBondVector    vhat = v / d                                 ([bonds, 3]; the zero vector stays zero)
    SphericalHarmonic   Y = real harmonics of vhat up to lmax        ([bonds, (lmax + 1)**2], Y_00 = 1; / sqrt(4 pi)
                                                                       with norm)

The three have no weights. They keep the geometry dtype: the lengths and directions have the dtype of the bond
vectors (float64, G1), and the harmonics are computed in float64 whatever the parameter dtype, as TF converts its
input to float64.

Layer: instructions; imports torch, the spherical-harmonics op and plan, the dtype policy, the errors and the input keys
of ``constants``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch

from tensorpotential.constants import BOND_VECTOR
from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.ops.precision import check_floating
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
    uncoupled_meta_data,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec

BOND_LENGTH_EPSILON: Final = 1e-10
"""Added under the square root of ``BondLength`` (a Python float in TF, ``compute.py``)."""


def _vectors_key(ref: InstructionRef | None) -> str:
    """Return the entry of the bond vectors: the referenced instruction, or ``bond_vector`` when null."""
    return BOND_VECTOR if ref is None else ref.name


class BondLength(torch.nn.Module):
    """``BondLength``: ``sqrt(|v|**2 + 1e-10)`` of each bond vector, ``[bonds, 1]``.

    Parameters
    ----------
    spec
        The instruction (``instruction_with_bonds``: the entry of the vectors, ``bond_vector`` when null).
    """

    def __init__(self, spec: InstructionSpec) -> None:
        """Remember where the bond vectors are."""
        super().__init__()
        self.spec = spec
        self._vectors = _vectors_key(spec.options["instruction_with_bonds"])

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the softened lengths ``[bonds, 1]`` in the dtype of the vectors.

        Raises
        ------
        DtypeError
            When the vectors are not floating.
        """
        v = inputs[self._vectors]
        check_floating(v)
        return torch.sqrt(v.square().sum(dim=1, keepdim=True) + BOND_LENGTH_EPSILON)


class ScaledBondVector(torch.nn.Module):
    """``ScaledBondVector``: each bond vector divided by its softened length, ``[bonds, 3]``.

    Parameters
    ----------
    spec
        The instruction (``bond_length``: the entry of the lengths; ``bonds``: the entry of the vectors,
        ``bond_vector`` when null).

    Raises
    ------
    OptionValueError
        For a null ``bond_length`` (TF fails only at the first forward pass).
    """

    def __init__(self, spec: InstructionSpec) -> None:
        """Remember where the vectors and the lengths are."""
        super().__init__()
        self.spec = spec
        lengths = spec.options["bond_length"]
        # the option table refuses it too; this guards a hand-made spec
        if lengths is None:
            msg = f"instruction {spec.name!r} (ScaledBondVector): bond_length is null"
            raise OptionValueError(msg)
        self._lengths = lengths.name
        self._vectors = _vectors_key(spec.options["bonds"])

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the directions ``[bonds, 3]``: ``v / d``, no epsilon of its own."""
        return inputs[self._vectors] / inputs[self._lengths]


class SphericalHarmonic(torch.nn.Module):
    """``SphericalHarmonic``: the real spherical harmonics of the bond directions, ``[bonds, (lmax + 1)**2]``.

    Parameters
    ----------
    spec
        The instruction (``vhat``, ``lmax``; ``norm`` and ``type`` when the yaml gives them).
    policy
        Dtype policy of the model: the harmonics are in its geometry dtype (float64) whatever the parameter dtype.

    Attributes
    ----------
    coupling_meta_data
        One row per output column (``l``, ``m``, ``hist``, ``parity``, ``sum_of_ls``), as TF builds it; the
        ``SingleParticleBasisFunction*`` twins build their tables from it.

    Raises
    ------
    OptionValueError
        For ``type`` other than ``"real"`` (TF constructs it and fails at the first forward pass) or a null ``vhat``.
    ShapeError
        For a negative ``lmax``.
    """

    def __init__(self, spec: InstructionSpec, policy: DtypePolicy) -> None:
        """Build the plan of the harmonics and the table of the columns."""
        super().__init__()
        options = spec.options
        self.spec = spec
        self.policy = policy
        if options.get("type", "real") != "real":
            msg = f"instruction {spec.name!r} (SphericalHarmonic): type {options['type']!r} is not supported, TF cannot run it; use 'real'"
            raise OptionValueError(msg)
        if options["vhat"] is None:
            msg = f"instruction {spec.name!r} (SphericalHarmonic): vhat is null"
            raise OptionValueError(msg)
        self._directions = options["vhat"].name
        self.plan = spherical_harmonics_plan(options["lmax"])
        self.lmax = self.plan.lmax
        self.coupling_meta_data: pd.DataFrame = uncoupled_meta_data(self.plan)
        # norm = True leaves the orthonormal harmonics, Y_00 = 1 / sqrt(4 pi)
        self._factor = 1 / math.sqrt(4 * math.pi) if options.get("norm") else None

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the harmonics ``[bonds, (lmax + 1)**2]`` in float64, column ``l*l + l + m``."""
        directions = self.policy.to_geometry(inputs[self._directions])
        y = real_spherical_harmonics(directions, self.plan)
        return y if self._factor is None else y * self._factor
