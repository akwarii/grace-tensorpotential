"""Twin of TF's ``SingleParticleBasisFunctionEquivariantInd``: bonds times the neighbour's indicator, CG-coupled (naive port).

Spec sheet ``SingleParticleBasisFunctionEquivariantInd.md``, section 5, in the backend's layouts (``R`` and the indicator
``[.., lm, n]``, ``Y`` ``[bonds, lm]``):

.. code-block:: text

    prod[a, l, r, n] = inv_avg_n_neigh * sum_{bonds j with ind_i[j] = a} R[j, l, n] Y[j, l] I[ind_j[j], r, n]
    out[a, f, n]     = sum_{k with m_sum_ind[k] = f} cg[k] prod[a, lr_inds[k, 0], lr_inds[k, 1], n]

The tables come from :func:`~tensorpotential.torch_backend.plans.spbf_equivariant_ind.spbf_equivariant_ind_plan`.
``Y`` is cut to ``slice_angular`` columns and cast to the dtype of ``R``; the product follows the promoted dtype of its
inputs, as TF's does (sheet section 6), and ``cg`` and ``inv_avg_n_neigh`` are made in the parameter dtype and cast to
it. ``sum_neighbors = False`` keeps one row per bond and does not average. ``dense_nbr`` is accepted either way and
changes nothing: the sum over the bonds of an atom is the same whatever their layout (decided by the owner; the padding
bonds of TF's dense layout contribute zero). A chemical embedding needs the twin of ``ScalarChemicalEmbedding`` and is
refused. The twin has no weights. The coupling is a gather and a scatter sum where TF multiplies by a dense CG matrix:
the same terms, summed in another order.

Layer: instructions; imports torch, the plans, the ops, the errors and the input keys of ``constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import ATOMIC_MU_I, BOND_IND_I, BOND_IND_J
from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.precision import (
    check_index,
    promote_like_tf,
    to_dtype_of,
)
from tensorpotential.torch_backend.ops.scatter import segment_sum
from tensorpotential.torch_backend.plans.spbf_equivariant_ind import (
    EquivariantIndOptions,
    spbf_equivariant_ind_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec


class SingleParticleBasisFunctionEquivariantInd(torch.nn.Module):
    """The second-layer basis ``YI``: radial times angular times the neighbour's indicator, summed and coupled.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``angular``, ``indicator``, ``radial``, ``lmax``, ``Lmax``,
        ``keep_parity``, ``history_drop_list``, ``l_max_ind``, ``max_sum_l``, ``normalize``, ``avg_n_neigh``,
        ``sum_neighbors``; ``dense_nbr`` and ``lm_first`` change nothing here).
    angular_meta, indicator_meta
        The coupling tables of the angular input and of the indicator.
    radial_lmax
        ``lmax`` of the radial input, which has to fit ``lmax`` (TF asserts it).
    policy
        Dtype policy of the model: ``cg`` and ``inv_avg_n_neigh`` are made in its parameter dtype.

    Raises
    ------
    OptionValueError
        For a chemical embedding, and from the plan (degrees of the radial and angular inputs).
    ShapeError
        From the plan, when no coupling survives the limits.
    """

    def __init__(
        self,
        spec: InstructionSpec,
        angular_meta: pd.DataFrame,
        indicator_meta: pd.DataFrame,
        *,
        radial_lmax: int,
        policy: DtypePolicy,
    ) -> None:
        """Build the coupling plan and its constants."""
        super().__init__()
        if spec.options["chemical_embedding"] is not None:
            msg = (
                f"instruction {spec.name!r} (SingleParticleBasisFunctionEquivariantInd): a chemical_embedding needs "
                "the twin of ScalarChemicalEmbedding, which this instruction does not take yet; write null"
            )
            raise OptionValueError(msg)
        self.spec = spec
        self.plan = spbf_equivariant_ind_plan(
            angular_meta,
            indicator_meta,
            EquivariantIndOptions.from_options(spec.options),
            radial_lmax=radial_lmax,
        )
        self.coupling_meta_data = self.plan.coupling_meta_data
        self.sum_neighbors = bool(spec.options["sum_neighbors"])
        self._names = tuple(
            spec.options[k].name for k in ("radial", "angular", "indicator")
        )
        self._n_lm_indicator = len(indicator_meta)
        cpu = torch.device("cpu")
        lr = torch.as_tensor(self.plan.lr_inds, dtype=torch.int64)
        self.register_buffer("left", lr[:, 0].contiguous(), persistent=False)
        self.register_buffer("right", lr[:, 1].contiguous(), persistent=False)
        m_sum = torch.as_tensor(self.plan.m_sum_ind, dtype=torch.int64)
        self.register_buffer("m_sum_ind", m_sum, persistent=False)
        self.register_buffer(
            "cg", policy.param_constant(self.plan.cg, cpu), persistent=False
        )
        inv_avg = policy.param_constant(self.plan.inv_avg_n_neigh, cpu)
        self.register_buffer("inv_avg_n_neigh", inv_avg, persistent=False)

    def _bond_inputs(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``R [bonds, lm, n]``, ``Y [bonds, lm]`` cut and cast to ``R``, and ``I [atoms, lm_I, n]``."""
        r, y, indicator = (inputs[name] for name in self._names)
        if indicator.dim() != 3 or indicator.shape[1] != self._n_lm_indicator:
            msg = (
                f"instruction {self.spec.name!r}: the indicator {self._names[2]!r} must be "
                f"[atoms, {self._n_lm_indicator}, n], got {tuple(indicator.shape)}"
            )
            raise ShapeError(msg)
        if self.plan.slice_angular is not None:
            y = y[:, : self.plan.slice_angular]
        if r.dim() != 3 or r.shape[1] != y.shape[1]:
            msg = (
                f"instruction {self.spec.name!r}: the radial input {self._names[0]!r} must be "
                f"[bonds, {y.shape[1]}, n], got {tuple(r.shape)}"
            )
            raise ShapeError(msg)
        return r, to_dtype_of(y, r), indicator

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Couple the bonds with the indicators of their neighbours.

        Parameters
        ----------
        inputs
            The radial ``[bonds, lm, n]``, angular ``[bonds, lm]`` and indicator ``[atoms, lm_I, n]`` inputs by name,
            ``ind_i`` and ``ind_j`` ``[bonds]`` and ``atomic_mu_i`` ``[atoms]``.

        Returns
        -------
        torch.Tensor
            ``[atoms, n_func, n]`` (``[bonds, n_func, n]`` without ``sum_neighbors``), in the promoted dtype of the
            radial input and the indicator.
        """
        r, y, indicator = self._bond_inputs(inputs)
        ind_i, ind_j = inputs[BOND_IND_I], inputs[BOND_IND_J]
        check_index(ind_j)
        ry, bond_indicator = promote_like_tf(
            r * y[:, :, None], indicator.index_select(0, ind_j)
        )
        prod = ry[:, :, None, :] * bond_indicator[:, None, :, :]  # [bonds, lm, lm_I, n]
        if self.sum_neighbors:
            n_atoms = inputs[ATOMIC_MU_I].shape[0]
            prod = segment_sum(prod, ind_i, n_atoms)
            prod = prod * to_dtype_of(self.get_buffer("inv_avg_n_neigh"), prod)
        terms = prod[:, self.get_buffer("left"), self.get_buffer("right"), :]
        terms = terms * to_dtype_of(self.get_buffer("cg"), prod)[None, :, None]
        out = terms.new_zeros((terms.shape[0], self.plan.nfunc, terms.shape[2]))
        return out.index_add(1, self.get_buffer("m_sum_ind"), terms)
