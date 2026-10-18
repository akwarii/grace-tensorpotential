"""Twin of TF's ``ProductFunction``: the Clebsch-Gordan product of two atomic basis tensors, channel by channel.

Spec sheet ``ProductFunction.md``, section 5, in the backend's layout ``[atoms, lm, n]``:

.. code-block:: text

    out[a, f, n] = sum over pairs p with m_sum_ind[p] = f of cg[p] * left[a, left_ind[p], n] * right[a, right_ind[p], n]

The coupling table is :func:`~tensorpotential.torch_backend.plans.product_function.product_coupling_meta_data` of the
two operands' tables and the options; the index tables are
:func:`~tensorpotential.torch_backend.plans.product_function.product_function_plan` of it. As in TF the product
follows the promotion of its three factors (``cg`` is made in the parameter dtype), so a float64 operand promotes a
float32 model to float64. The channels never mix and the twin has no weights. TF scatters with
``unsorted_segment_sum``; the twin with ``index_add``: the same terms, summed in another order.

Layer: instructions; imports torch, the plans, the ops and the errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.ops.precision import promote_like_tf
from tensorpotential.torch_backend.plans.product_function import (
    ProductFunctionOptions,
    product_coupling_meta_data,
    product_function_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec


class ProductFunction(torch.nn.Module):
    """``ProductFunction``: couple the entries of ``left`` and ``right`` to functions of definite ``(L, M, parity)``.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``left``, ``right``, ``lmax``, ``Lmax``, ``is_left_right_equal``,
        ``keep_parity``, ``normalize`` and the extra limits of the coupling generator; ``lm_first`` is a TF layout and
        changes nothing here).
    left_meta, right_meta
        The coupling tables of the two operands.
    policy
        Dtype policy of the model: ``cg`` is made in its parameter dtype.

    Raises
    ------
    ShapeError
        From the plan, when no coupling survives the limits (TF: "No coupling channels found").
    """

    def __init__(
        self,
        spec: InstructionSpec,
        left_meta: pd.DataFrame,
        right_meta: pd.DataFrame,
        policy: DtypePolicy,
    ) -> None:
        """Build the coupling table and its index tables."""
        super().__init__()
        self.spec = spec
        options = ProductFunctionOptions.from_options(spec.options)
        self.coupling_meta_data = product_coupling_meta_data(
            left_meta, right_meta, options
        )
        plan = product_function_plan(self.coupling_meta_data)
        self.nfunc = plan.nfunc
        self._names = (spec.options["left"].name, spec.options["right"].name)
        self._n_lm = (len(left_meta), len(right_meta))
        for name, table in (
            ("left_ind", plan.left_ind),
            ("right_ind", plan.right_ind),
            ("m_sum_ind", plan.m_sum_ind),
        ):
            index = torch.as_tensor(table, dtype=torch.int64)
            self.register_buffer(name, index, persistent=False)
        cg = policy.param_constant(plan.cg, torch.device("cpu"))
        self.register_buffer("cg", cg, persistent=False)

    def _operands(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``left`` and ``right`` ``[atoms, lm, n]``, checked against the operands' tables."""
        left, right = (inputs[name] for name in self._names)
        for side, name, x, n_lm in zip(
            ("left", "right"), self._names, (left, right), self._n_lm, strict=True
        ):
            if x.dim() != 3 or x.shape[1] != n_lm:
                msg = (
                    f"instruction {self.spec.name!r}: the {side} operand {name!r} must be "
                    f"[atoms, {n_lm}, n], got {tuple(x.shape)}"
                )
                raise ShapeError(msg)
        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
            msg = (
                f"instruction {self.spec.name!r}: the operands must have the same atoms and channels, "
                f"got {tuple(left.shape)} and {tuple(right.shape)}"
            )
            raise ShapeError(msg)
        return left, right

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Multiply the paired entries of the two operands by ``cg`` and sum them into the output functions.

        Parameters
        ----------
        inputs
            The two operands ``[atoms, lm, n]`` by name.

        Returns
        -------
        torch.Tensor
            ``[atoms, n_func, n]``, in the promoted dtype of the operands and ``cg``.
        """
        left, right = self._operands(inputs)
        lft = left.index_select(1, self.get_buffer("left_ind"))
        rght = right.index_select(1, self.get_buffer("right_ind"))
        lft, rght, cg = promote_like_tf(lft, rght, self.get_buffer("cg"))
        prod = lft * rght * cg[None, :, None]
        out = prod.new_zeros((prod.shape[0], self.nfunc, prod.shape[2]))
        return out.index_add(1, self.get_buffer("m_sum_ind"), prod)
