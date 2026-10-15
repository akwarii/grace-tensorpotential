"""The interface every instruction twin implements, and the only one the executor knows.

A twin is the torch counterpart of one TF instruction of a ``model.yaml``. The executor
(:mod:`~tensorpotential.torch_backend.model`) runs the twins in file order: it gives each one the values the
instructions before it produced, by name, and stores what :meth:`Twin.forward` returns under the twin's name, as
``TPModel`` does with ``frwrd``. It depends on this protocol and never on a concrete twin (import contract), so a
twin is added by registering it, not by changing the executor.

Layer: beside the ops, below the instructions and the model. torch is imported for type checking only, so the
package imports without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from tensorpotential.torch_backend.spec.model import InstructionSpec


class Twin(Protocol):
    """The torch twin of one TF instruction.

    A twin is usually a ``torch.nn.Module`` whose ``spec`` is set in its constructor. ``forward`` has no side
    effects: it changes neither the twin nor its inputs (rule R4).

    Examples
    --------
    >>> class Double(torch.nn.Module):
    ...     def __init__(self, spec: InstructionSpec) -> None:
    ...         super().__init__()
    ...         self.spec = spec
    ...
    ...     def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    ...         return 2.0 * inputs[self.spec.depends_on[0]]
    """

    @property
    def spec(self) -> InstructionSpec:
        """The instruction this twin computes, as the spec loader read it from the ``model.yaml``."""

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute the instruction's value.

        Parameters
        ----------
        inputs
            The values of the instructions run before this one (and the input tensors of the structure), by name.

        Returns
        -------
        torch.Tensor
            The value stored under ``spec.name``.
        """
