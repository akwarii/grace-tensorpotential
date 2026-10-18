"""Twins of TF's energy chain: ``CreateOutputTarget``, ``LinMLPOut2ScalarTarget`` and ``ConstantScaleShiftTarget``.

Spec sheets of the three classes, section 5. The chain starts from the scalar ``initial_value`` of
``CreateOutputTarget``; every output instruction that follows reads the target, adds or applies its term and returns the
new value, which the model stores back **under the target's name** (``atomic_energy``):

.. code-block:: text

    CreateOutputTarget        value = initial_value                                     (a 0-d tensor)
    LinMLPOut2ScalarTarget    f_s = x_s[:, 0, :] for each origin s ([atoms, n], the l = 0 column)
                              lin = sum_s f_s[:, :1];  t = sum_s f_s[:, 1:];  t = scalar_rms_norm(t, scale) ("layer")
                              out = target + (mlp(t) + lin)                             ([atoms, 1])
    ConstantScaleShiftTarget  out = target * scale + shift                              (shift: a constant, a value per
                              element, or Z[mu_i] @ embedding_shift / sqrt(embedding_size))

Every constant is made in the parameter dtype and cast to the target, as TF builds them in ``float_dtype``: a float32
model rounds ``scale`` to float32 before the product. The twins have no padding atoms, so TF's masks of the padding
rows (``batch_tot_nat_real``) are the identity here. The weights are named as in the checkpoint (``mlp.layer<i>.w``,
``scale``, ``embedding_shift``).

Layer: instructions; imports torch, the dense ops, the dtype policy, the errors and the input keys of ``constants``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import ATOMIC_MU_I
from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.dense import MLP, scalar_rms_norm
from tensorpotential.torch_backend.ops.precision import check_index, to_dtype_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec

# hidden_layers of a LinMLPOut2ScalarTarget whose yaml gives null (output.py)
_DEFAULT_HIDDEN_LAYERS = (32,)


class CreateOutputTarget(torch.nn.Module):
    """``CreateOutputTarget``: the starting value of the energy, a 0-d tensor in the parameter dtype.

    Parameters
    ----------
    spec
        The instruction (``initial_value``, null meaning 0; ``l`` is 0, the only value the option table accepts).
    policy
        Dtype policy of the model.
    """

    def __init__(self, spec: InstructionSpec, policy: DtypePolicy) -> None:
        """Make the starting value."""
        super().__init__()
        self.spec = spec
        initial = spec.options["initial_value"]
        value = policy.param_constant(
            0.0 if initial is None else float(initial), torch.device("cpu")
        )
        self.register_buffer("value", value, persistent=False)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:  # noqa: ARG002 (reads no input)
        """Return the starting value ``[]``."""
        return self.get_buffer("value")


class LinMLPOut2ScalarTarget(torch.nn.Module):
    """``LinMLPOut2ScalarTarget``: channel 0 of the invariant features linearly, the others through an MLP.

    Parameters
    ----------
    spec
        The instruction (``origin``, ``target``, ``hidden_layers``, ``n_out``, ``normalize``, ``activation``,
        ``return_hidden_target``; ``full_origin_norm`` and ``init_norm`` are ignored, as in TF).
    origin_channels
        The channel count of each origin, in the order of ``spec.options["origin"]``; all must be equal (TF asserts it).
    policy
        Dtype policy of the model: the MLP weights and ``scale`` have its parameter dtype.

    Raises
    ------
    ShapeError
        When the origins have different channel counts, or fewer than two.
    OptionValueError
        For a ``normalize`` other than null and ``"layer"`` (TF asserts it), and from the MLP (activation).
    """

    def __init__(
        self,
        spec: InstructionSpec,
        origin_channels: Sequence[int],
        policy: DtypePolicy,
    ) -> None:
        """Build the MLP and the scale of the layer normalisation."""
        super().__init__()
        options = spec.options
        self.spec = spec
        self._origins = tuple(ref.name for ref in options["origin"])
        self._target = options["target"].name
        if (
            len(self._origins) != len(origin_channels)
            or len(set(origin_channels)) != 1
            or origin_channels[0] < 2
        ):
            msg = (
                f"instruction {spec.name!r} (LinMLPOut2ScalarTarget): the origins {list(self._origins)} must have one "
                f"channel count of at least 2, got {list(origin_channels)}"
            )
            raise ShapeError(msg)
        # the option table refuses it too; this guards a hand-made spec
        if options["normalize"] not in (None, "layer"):
            msg = f"instruction {spec.name!r} (LinMLPOut2ScalarTarget): normalize must be null or 'layer', got {options['normalize']!r}"
            raise OptionValueError(msg)
        n_in = origin_channels[0] - 1
        hidden = options["hidden_layers"]
        hidden = _DEFAULT_HIDDEN_LAYERS if hidden is None else tuple(hidden)
        sizes = [n_in, *hidden, int(options["n_out"])]
        self.mlp = MLP.fully_connected(sizes, options["activation"], policy)
        self.scale = None
        if options["normalize"] == "layer":
            self.scale = torch.nn.Parameter(torch.zeros((1, n_in), dtype=policy.param))
        self.hidden_key: str | None = options["return_hidden_target"] or None

    def _readout(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the new target, the summed linear channel and the input of the last MLP layer."""
        # [atoms, n], the l = 0 column of each origin [atoms, lm, n]; summed in TF's order
        first, *others = (inputs[name][:, 0, :] for name in self._origins)
        lin, transformed = first[:, :1], first[:, 1:]
        for scalars in others:
            lin = lin + scalars[:, :1]
            transformed = transformed + scalars[:, 1:]
        if self.scale is not None:
            transformed = scalar_rms_norm(transformed, self.scale)
        out, hidden = self.mlp.forward_with_hidden(transformed)
        return inputs[self._target] + (out + lin), lin, hidden

    def forward_with_hidden(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the new target ``[atoms, n_out]`` and the hidden features ``[atoms, 1 + width of the last layer]``.

        The hidden features are ``concat([lin, h])`` with ``h`` the input of the last MLP layer: what TF writes under
        ``return_hidden_target``.
        """
        out, lin, hidden = self._readout(inputs)
        return out, torch.cat([lin, hidden], dim=1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the new target ``[atoms, n_out]``: the old one plus the readout of the origins."""
        return self._readout(inputs)[0]


def _shift_map(spec: InstructionSpec, n_elements: int | None) -> list[float] | None:
    shift_map = spec.options["atomic_shift_map"]
    if shift_map is None:
        return None
    keys = sorted(int(k) for k in shift_map)
    if keys != list(range(len(keys))):
        msg = f"instruction {spec.name!r} (ConstantScaleShiftTarget): the atomic_shift_map keys must be 0 .. {len(keys) - 1}, got {keys}"
        raise OptionValueError(msg)
    if n_elements is not None and len(keys) < n_elements:
        msg = (
            f"instruction {spec.name!r} (ConstantScaleShiftTarget): the atomic_shift_map has {len(keys)} entries for "
            f"a model of {n_elements} elements"
        )
        raise OptionValueError(msg)
    return [float(shift_map[k]) for k in sorted(shift_map, key=int)]


class ConstantScaleShiftTarget(torch.nn.Module):
    """``ConstantScaleShiftTarget``: ``target * scale + shift``.

    Parameters
    ----------
    spec
        The instruction (``target``, ``scale``, ``shift``, ``atomic_shift_map``, ``chemical_embedding``; ``l`` is 0).
    policy
        Dtype policy of the model: the constants are made in its parameter dtype, ``embedding_shift`` too.
    n_elements
        Number of elements of the model; an ``atomic_shift_map`` with fewer entries is refused.
    embedding_size
        Width of the ``chemical_embedding`` table, ``None`` without one.

    Raises
    ------
    OptionValueError
        For ``atomic_shift_map`` keys other than ``0 .. n - 1`` or fewer entries than elements, a non-zero ``shift``
        together with a map or an embedding (TF asserts it), or an embedding without its size.
    """

    def __init__(
        self,
        spec: InstructionSpec,
        policy: DtypePolicy,
        *,
        n_elements: int | None = None,
        embedding_size: int | None = None,
    ) -> None:
        """Make the constants and the weight of the embedding shift."""
        super().__init__()
        options = spec.options
        self.spec = spec
        self._target = options["target"].name
        embedding = options["chemical_embedding"]
        self._embedding = "" if embedding is None else embedding.name
        shift_map = _shift_map(spec, n_elements)
        shift = float(options["shift"])
        if shift != 0 and (shift_map is not None or embedding is not None):
            msg = f"instruction {spec.name!r} (ConstantScaleShiftTarget): shift must be 0 with an atomic_shift_map or a chemical_embedding, got {shift}"
            raise OptionValueError(msg)
        if (embedding is None) != (embedding_size is None):
            msg = f"instruction {spec.name!r} (ConstantScaleShiftTarget): its chemical_embedding is {embedding!r}, but an embedding size of {embedding_size!r} was given"
            raise OptionValueError(msg)
        cpu = torch.device("cpu")
        self._constant = shift
        self.register_buffer(
            "scale",
            policy.param_constant(float(options["scale"]), cpu),
            persistent=False,
        )
        self.register_buffer(
            "constant_shift", policy.param_constant(shift, cpu), persistent=False
        )
        self._per_element = shift_map is not None
        values = policy.param_constant(shift_map or [], cpu).reshape(-1, 1)
        self.register_buffer("atomic_shift_map", values, persistent=False)
        self.embedding_shift = None
        if embedding_size is not None:
            self.embedding_shift = torch.nn.Parameter(
                torch.zeros((embedding_size, 1), dtype=policy.param)
            )
            norm = policy.param_constant(embedding_size, cpu).rsqrt()
            self.register_buffer("embedding_norm", norm, persistent=False)
        # TF's apply_shift: the shift sum is skipped when there is nothing to add
        self._apply_shift = bool(
            shift or self._per_element or self.embedding_shift is not None
        )

    def _shift(
        self, inputs: Mapping[str, torch.Tensor], total: torch.Tensor
    ) -> torch.Tensor:
        """Add the per-element, embedding and constant shifts to ``total`` (TF's ``total_shift``)."""
        if self._per_element or self.embedding_shift is not None:
            atomic_mu_i = inputs[ATOMIC_MU_I]
            check_index(atomic_mu_i)
            if self._per_element:
                shift_map = self.get_buffer("atomic_shift_map")
                per_element = shift_map.index_select(0, atomic_mu_i)
                total = total + to_dtype_of(per_element, total)
            if self.embedding_shift is not None:
                rows = inputs[self._embedding].index_select(0, atomic_mu_i)
                learned = (rows @ self.embedding_shift) * self.get_buffer(
                    "embedding_norm"
                )
                total = total + to_dtype_of(learned, total)
        if self._constant != 0:
            constant = self.get_buffer("constant_shift")
            total = total + to_dtype_of(constant, total)
        return total

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return ``target * scale`` plus the shifts, in the dtype of the target."""
        target = inputs[self._target]
        scaled = target * to_dtype_of(self.get_buffer("scale"), target)
        if not self._apply_shift:
            return scaled
        return scaled + self._shift(inputs, torch.zeros_like(target))
