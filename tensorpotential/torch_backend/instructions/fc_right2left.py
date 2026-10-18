"""Twin of TF's ``FCRight2Left``: a channel mixing of the left operand plus mixed right entries of equal ``(l, parity)``.

Spec sheet ``FCRight2Left.md``, section 5, in the backend's layout ``[atoms, lm, n]``:

.. code-block:: text

    out[a, row, k]  = norm_left * sum_n w_left[(mu_a,) k, n, w_tile_left[row]] * left[a, row, n]   (left itself
                      without left_coefs)
    out[a, to, k]  += norm_right * sum_n w_right[(mu_a,) k, n, w_tile_right[j]] * right[a, from, n]
                      for every pair j = (to, from) of collect_to and collect_from
    out            *= norm_map[row]                                                                  (norm_out only)

The tables come from :func:`~tensorpotential.torch_backend.plans.fc_right2left.fc_right2left_plan`; the output has the
coupling table of the left operand. As in TF there is **no cast**: an operand whose dtype differs from the weights raises
a ``DtypeError`` (TF fails there too), so the output has the parameter dtype. Per-element weights gather by
``atomic_mu_i``, and TF does not cut padded operands, so they must have one row per atom. The weights ``w_left`` (with
``left_coefs``) and ``w_right`` are the only state.

Layer: instructions; imports torch, the plans, the ops, the spec, the errors and the input keys of ``constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import ATOMIC_MU_I
from tensorpotential.torch_backend.errors import ShapeError, SpecError
from tensorpotential.torch_backend.ops.precision import check_index, same_dtype
from tensorpotential.torch_backend.plans.fc_right2left import (
    FCRight2LeftWidths,
    fc_right2left_plan,
)
from tensorpotential.torch_backend.spec.model import InstructionRef

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.plans.function_reduce_n import ReduceSource
    from tensorpotential.torch_backend.spec.model import InstructionSpec


def _check_operands(
    spec: InstructionSpec, left: ReduceSource, right: ReduceSource
) -> None:
    for key, given in (("left", left), ("right", right)):
        ref = spec.options[key]
        if not isinstance(ref, InstructionRef) or ref.name != given.name:
            msg = f"instruction {spec.name!r} (FCRight2Left): its {key} operand is {ref!r}, but {given.name!r} was given"
            raise SpecError(msg)


class FCRight2Left(torch.nn.Module):
    """``FCRight2Left``: mix the channels of ``left`` and add the mixed right entries of equal ``(l, parity)``.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``left``, ``right``, ``n_out``, ``left_coefs``,
        ``is_central_atom_type_dependent``, ``number_of_atom_types``, ``normalize``, ``norm_out``; ``lm_first`` is a
        TF layout and changes nothing here).
    left, right
        Name, coupling table and channel count of the two operands.
    policy
        Dtype policy of the model: the weights have its parameter dtype. They start at zero; the stored weights are
        loaded into ``w_left`` and ``w_right``.

    Raises
    ------
    SpecError
        When ``left`` or ``right`` is not the operand the spec names.
    OptionValueError
        For an ``is_central_atom_type_dependent`` that is not null, a bool or two bools, and from the plan
        (``number_of_atom_types`` with per-element weights, ``n_out`` without ``left_coefs``).
    ShapeError
        From the plan, when no ``(l, parity)`` is shared or matched groups differ in size.
    """

    def __init__(
        self,
        spec: InstructionSpec,
        left: ReduceSource,
        right: ReduceSource,
        policy: DtypePolicy,
    ) -> None:
        """Build the plan, the zero weights and the index tables of the instruction."""
        super().__init__()
        _check_operands(spec, left, right)
        widths = FCRight2LeftWidths.from_options(spec.options, left.n_out, right.n_out)
        self.spec = spec
        self.plan = fc_right2left_plan(
            left.coupling_meta_data, right.coupling_meta_data, widths
        )
        self.coupling_meta_data = left.coupling_meta_data.copy()
        self.n_out = widths.n_out
        self.per_element = widths.per_element
        self.norm_out = bool(spec.options["norm_out"])
        sides = (("left", self.plan.norm_left, 0), ("right", self.plan.norm_right, 1))
        # side -> (norm, per-element); a side without a weight (no left_coefs) has no entry
        self._sides = {
            side: (norm, widths.per_element[i])
            for side, norm, i in sides
            if norm is not None
        }
        self._operands = {
            "left": (left.name, len(left.coupling_meta_data)),
            "right": (right.name, len(right.coupling_meta_data)),
        }
        for name, shape in (
            ("w_left", self.plan.w_left_shape),
            ("w_right", self.plan.w_right_shape),
        ):
            if shape is not None:
                weight = torch.nn.Parameter(torch.zeros(shape, dtype=policy.param))
                self.register_parameter(name, weight)
        for name, table in (
            ("tile_left", self.plan.w_tile_left),
            ("tile_right", self.plan.w_tile_right),
            ("collect_to", self.plan.collect_to),
            ("collect_from", self.plan.collect_from),
        ):
            index = torch.as_tensor(table, dtype=torch.int64)
            self.register_buffer(name, index, persistent=False)
        self.register_buffer(
            "norm_map",
            policy.param_constant(
                self.plan.norm_map.reshape(1, -1, 1), torch.device("cpu")
            ),
            persistent=False,
        )

    def _operand(self, key: str, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        name, n_lm = self._operands[key]
        x = inputs[name]
        if x.dim() != 3 or x.shape[1] != n_lm:
            msg = (
                f"instruction {self.spec.name!r} (FCRight2Left): {key} operand {name!r} must be "
                f"[atoms, {n_lm}, n], got {tuple(x.shape)}"
            )
            raise ShapeError(msg)
        return x

    def _mix(
        self, side: str, x: torch.Tensor, inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """``norm * sum_n w[(mu_a,) k, n, j] x[a, j, n]`` of one side: ``[atoms, j, n_out]``; no cast, as in TF."""
        norm, per_element = self._sides[side]
        w = self.get_parameter(f"w_{side}").index_select(
            -1, self.get_buffer(f"tile_{side}")
        )
        same_dtype(w, x)
        if per_element:
            atomic_mu_i = inputs[ATOMIC_MU_I]
            check_index(atomic_mu_i)
            w = w.index_select(0, atomic_mu_i)
            return torch.einsum("aknw,awn->awk", w, x) * norm
        return torch.einsum("knw,awn->awk", w, x) * norm

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mix and add.

        Parameters
        ----------
        inputs
            The two operands ``[atoms, lm, n]`` by name, and ``atomic_mu_i`` ``[atoms]`` when a side has per-element
            weights.

        Returns
        -------
        torch.Tensor
            ``[atoms, lm_left, n_out]``, in the dtype of the weights.
        """
        left = self._operand("left", inputs)
        if "left" in self._sides:
            left = self._mix("left", left, inputs)
        right = self._operand("right", inputs).index_select(
            1, self.get_buffer("collect_from")
        )
        right = self._mix("right", right, inputs)
        same_dtype(left, right)
        out = left.index_add(1, self.get_buffer("collect_to"), right)
        return out * self.get_buffer("norm_map") if self.norm_out else out
