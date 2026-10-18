"""Twin of TF's ``SingleParticleBasisFunctionScalarInd``: radial times angular times a species weight, summed per atom.

Spec sheet ``SingleParticleBasisFunctionScalarInd.md``, section 5, in the backend's layouts (``R [bonds, lm, n]``,
``Y [bonds, lm]``, the indicator table ``Z [n_elements, embedding_size]``):

.. code-block:: text

    a[j, lm, n] = R[j, lm, n] Y[j, lm] Zt[mu_j[j], n]          Zt = Z @ (w / sqrt(embedding_size))
                  (Zt[mu_j[j], n, l(lm)] with indicator_l_depend; no Zt without an indicator)
    out[i]      = inv_avg_n_neigh * sum_{bonds j with ind_i[j] = i} a[j]   (per central element with a dict avg_n_neigh)

The tables come from :func:`~tensorpotential.torch_backend.plans.spbf_scalar_ind.spbf_scalar_ind_plan`. The dtypes follow
TF: ``Y`` and ``Z`` are cast to the dtype of ``R``, the linear layer casts ``Z`` to its weight, and the bond product is
cast to the dtype of the transformed indicator, so a float32 model rounds ``Y`` to float32 before the product.
``sum_neighbors = False`` keeps one row per bond and does not average. The only weight is ``lin_transform.w``
(``[embedding_size, n_rad_max]``, or ``n_rad_max * (lmax + 1)`` columns with ``indicator_l_depend``), as in the
checkpoint. Per-atom indicators do not exist in the shipped models and are not ported.

Layer: instructions; imports torch, the plans, the ops, the errors and the input keys of ``constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import (
    ATOMIC_MU_I,
    BOND_IND_I,
    BOND_MU_J,
)
from tensorpotential.torch_backend.errors import ShapeError, SpecError
from tensorpotential.torch_backend.ops.dense import Linear
from tensorpotential.torch_backend.ops.precision import check_index, to_dtype_of
from tensorpotential.torch_backend.ops.scatter import segment_sum
from tensorpotential.torch_backend.plans.spbf_scalar_ind import spbf_scalar_ind_plan

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec


class SingleParticleBasisFunctionScalarInd(torch.nn.Module):
    """The first-layer basis ``A``: radial times angular times the neighbour's species weight, summed and averaged.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``radial``, ``angular``, ``indicator``, ``indicator_l_depend``,
        ``sum_neighbors``, ``avg_n_neigh``, ``lmax``; ``lm_first`` is a TF layout and changes nothing here).
    angular_meta
        The coupling table of the angular input.
    radial_lmax, n_rad_max
        ``lmax`` and the channel count of the radial input.
    embedding_size
        Width of the indicator table, ``None`` without an indicator.
    policy
        Dtype policy of the model: ``lin_transform.w`` has its parameter dtype, and ``inv_avg_n_neigh`` is made in it.

    Raises
    ------
    SpecError
        When ``embedding_size`` is given without an indicator, or missing with one.
    OptionValueError
        From the plan (degrees of the inputs, ``avg_n_neigh``).
    """

    def __init__(
        self,
        spec: InstructionSpec,
        angular_meta: pd.DataFrame,
        *,
        radial_lmax: int,
        n_rad_max: int,
        embedding_size: int | None,
        policy: DtypePolicy,
    ) -> None:
        """Build the tables and the linear map of the indicator."""
        super().__init__()
        options = spec.options
        indicator = options["indicator"]
        if (indicator is None) != (embedding_size is None):
            msg = (
                f"instruction {spec.name!r} (SingleParticleBasisFunctionScalarInd): its indicator is {indicator!r}, "
                f"but an embedding size of {embedding_size!r} was given"
            )
            raise SpecError(msg)
        self.spec = spec
        self.plan = spbf_scalar_ind_plan(
            angular_meta,
            lmax=options["lmax"],
            radial_lmax=radial_lmax,
            avg_n_neigh=options["avg_n_neigh"],
        )
        self.coupling_meta_data = self.plan.coupling_meta_data
        self.n_out = n_rad_max
        self.sum_neighbors = bool(options["sum_neighbors"])
        self.l_depend = bool(options["indicator_l_depend"])
        self._names = (options["radial"].name, options["angular"].name)
        self._indicator = "" if indicator is None else indicator.name
        self.lin_transform = None
        if embedding_size is not None:
            width = n_rad_max * (self.plan.lmax + 1) if self.l_depend else n_rad_max
            self.lin_transform = Linear(embedding_size, width, policy)
        cpu = torch.device("cpu")
        self.register_buffer(
            "l_tile", torch.as_tensor(self.plan.l_tile), persistent=False
        )
        inv_avg = policy.param_constant(self.plan.inv_avg_n_neigh, cpu)
        self.register_buffer("inv_avg_n_neigh", inv_avg, persistent=False)

    def _bond_product(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return ``R * Y`` ``[bonds, lm, n]``, with ``Y`` cut and cast to the dtype of ``R``."""
        r, y = (inputs[name] for name in self._names)
        if self.plan.slice_angular is not None:
            y = y[:, : self.plan.slice_angular]
        n_lm = len(self.plan.l_tile)
        if r.dim() != 3 or r.shape[1] != n_lm or y.dim() != 2 or y.shape[1] != n_lm:
            msg = (
                f"instruction {self.spec.name!r}: the radial input must be [bonds, {n_lm}, n] and the angular input "
                f"[bonds, {n_lm}] after the cut, got {tuple(r.shape)} and {tuple(y.shape)}"
            )
            raise ShapeError(msg)
        return r * to_dtype_of(y, r)[:, :, None]

    def _species_weight(
        self, inputs: Mapping[str, torch.Tensor], a: torch.Tensor, linear: Linear
    ) -> torch.Tensor:
        """Multiply each bond by the transformed indicator row of its neighbour's element."""
        z = linear(to_dtype_of(inputs[self._indicator], a))  # [n_elements, n_out]
        a = to_dtype_of(a, z)
        mu_j = inputs[BOND_MU_J]
        check_index(mu_j)
        if self.l_depend:
            z = z.reshape(z.shape[0], self.n_out, self.plan.lmax + 1)
            z = z[:, :, self.get_buffer("l_tile")].transpose(
                1, 2
            )  # [n_elements, lm, n]
            return a * z.index_select(0, mu_j)
        return a * z.index_select(0, mu_j)[:, None, :]

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Sum the weighted bond products over the neighbours of each atom.

        Parameters
        ----------
        inputs
            The radial ``[bonds, lm, n]`` and angular ``[bonds, lm]`` inputs and the indicator table by name,
            ``mu_j`` and ``ind_i`` ``[bonds]`` and ``atomic_mu_i`` ``[atoms]``.

        Returns
        -------
        torch.Tensor
            ``[atoms, lm, n]`` (``[bonds, lm, n]`` without ``sum_neighbors``), in the dtype of the transformed indicator
            (the parameter dtype), or of ``R`` without an indicator.
        """
        a = self._bond_product(inputs)
        if self.lin_transform is not None:
            a = self._species_weight(inputs, a, self.lin_transform)
        if not self.sum_neighbors:
            return a
        atomic_mu_i = inputs[ATOMIC_MU_I]
        a = segment_sum(a, inputs[BOND_IND_I], atomic_mu_i.shape[0])
        inv_avg = to_dtype_of(self.get_buffer("inv_avg_n_neigh"), a)
        if self.plan.per_element_average:
            check_index(atomic_mu_i)
            return a * inv_avg.index_select(0, atomic_mu_i)[:, :, None]
        return a * inv_avg
