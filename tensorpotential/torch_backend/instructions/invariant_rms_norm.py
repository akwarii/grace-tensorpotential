"""Twin of TF's ``InvariantLayerRMSNorm``: an RMS normalisation of the invariant channels of each atom.

Spec sheet ``InvariantLayerRMSNorm.md``, section 5. With ``x`` the invariant feature ``[atoms, lm, n]`` (``lm`` is 1 in
the shipped models) and the mean over the channel axis ``n``:

.. code-block:: text

    full           out = x * rsqrt(mean(x**2) + 1e-10) * scale                            (scale over all n channels)
    only_nonlin    out = concat([x[..., :1], norm(x[..., 1:]) * scale])                    (channel 0 passes through)
    sep_lin_gate   out = concat([x[..., :1] * lin_scale, norm(x[..., 1:]) * scale])

The epsilon is made in the parameter dtype and, with the weights, cast to the dtype of ``x`` (:func:`channel_rms_norm`).
TF reads an ``lm_first`` input as ``[lm, atoms, n]``, transposes it, and always writes ``[atoms, n, lm]``; the twin
reads and writes the backend layout ``[atoms, lm, n]``. The weights keep their checkpoint shapes: ``scale``
``[1, n', 1]`` (``n' = n`` for ``full``, ``n - 1`` otherwise) and ``lin_scale`` ``[1, 1]``. The twins have no padding
atoms, so TF's masks of the padding rows are the identity here. ``init`` only sets the initial weights and is ignored.

Layer: instructions; imports torch, the dense ops and the errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.dense import channel_rms_norm
from tensorpotential.torch_backend.ops.precision import to_dtype_of

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec

TYPES: Final = ("full", "only_nonlin", "sep_lin_gate")
"""The ``type`` values of TF's constructor (an ``assert`` there)."""


class InvariantLayerRMSNorm(torch.nn.Module):
    """``InvariantLayerRMSNorm``: ``[atoms, lm, n]`` to ``[atoms, lm, n]``, the channels RMS-normalised per atom.

    Parameters
    ----------
    spec
        The instruction (``inpt``, ``type``; ``init`` is ignored).
    n_channels
        Channel count ``n`` of the input (TF's ``inpt.n_out``).
    policy
        Dtype policy of the model: ``scale`` and ``lin_scale`` have its parameter dtype.

    Raises
    ------
    OptionValueError
        For a ``type`` outside :data:`TYPES` or a null ``inpt``.
    ShapeError
        For fewer than one channel (``full``) or two (the other types, which keep channel 0 apart).
    """

    def __init__(
        self, spec: InstructionSpec, n_channels: int, policy: DtypePolicy
    ) -> None:
        """Make the weights of the type."""
        super().__init__()
        options = spec.options
        self.spec = spec
        self.type = options["type"]
        if self.type not in TYPES:
            msg = f"instruction {spec.name!r} (InvariantLayerRMSNorm): type must be one of {', '.join(TYPES)}, got {self.type!r}"
            raise OptionValueError(msg)
        # the option table refuses it too; this guards a hand-made spec
        if options["inpt"] is None:
            msg = f"instruction {spec.name!r} (InvariantLayerRMSNorm): inpt is null"
            raise OptionValueError(msg)
        self._inpt = options["inpt"].name
        normalised = n_channels if self.type == "full" else n_channels - 1
        if normalised < 1:
            msg = f"instruction {spec.name!r} (InvariantLayerRMSNorm): type {self.type!r} needs more than {n_channels} channel(s)"
            raise ShapeError(msg)
        self.scale = torch.nn.Parameter(
            torch.zeros((1, normalised, 1), dtype=policy.param)
        )
        self.lin_scale = None
        if self.type == "sep_lin_gate":
            self.lin_scale = torch.nn.Parameter(torch.zeros((1, 1), dtype=policy.param))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the normalised feature ``[atoms, lm, n]`` in the dtype of the input."""
        x = inputs[self._inpt]
        scale = self.scale.transpose(
            1, 2
        )  # the checkpoint's [1, n', 1] on the channel axis, last here
        if self.type == "full":
            return channel_rms_norm(x, scale)
        lin = x[..., :1]
        if self.lin_scale is not None:
            lin = lin * to_dtype_of(self.lin_scale, x)
        return torch.cat([lin, channel_rms_norm(x[..., 1:], scale)], dim=-1)
