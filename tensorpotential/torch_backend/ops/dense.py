"""Dense blocks of the twins: the linear layer, the MLP and the RMS norms of TF's ``functions/nn.py``.

* :class:`Linear` is TF's ``Linear`` and ``DenseLayer`` without activation: ``x @ (w * norm)``, the input cast to the
  dtype of the weight, ``norm = 1 / sqrt(n_in)`` (or 1 for a ``Linear`` with ``normalize = False``). ``norm`` is computed
  at run time from ``n_in`` and is not a weight: the checkpoint holds the unscaled ``w`` (spec sheets
  ``MLPRadialFunction`` and ``MLPRadialFunction_v2``, section 4).
* :class:`MLP` is ``FullyConnectedMLP``: an activation after every hidden layer, none after the last one, no bias.
  :meth:`MLP.fully_connected` resolves its ``activation`` argument as TF does (``None`` is ``silu(x) * 1.6759``); the
  per-layer list of ``MLPRadialFunction_v2`` goes to the constructor.
* The three RMS norms, each with the epsilon of its spec sheet and TF's order of operations:
  :func:`variance_norm` (``MLPRadialFunction`` with ``norm``), :func:`scalar_rms_norm` (``scalar_rms_ln`` of
  ``LinMLPOut2ScalarTarget``) and :func:`channel_rms_norm` (``InvariantLayerRMSNorm``).

No block has a bias: no instruction of the twin set passes ``use_bias`` (TF's ``DenseLayer`` cannot even run with it
when it has no activation, finding on issue DENSE1). Casts go through :mod:`.precision`.

Layer: ops; imports torch, the errors and the dtype policy (:mod:`.precision`).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

import torch

from tensorpotential.torch_backend.errors import ShapeError, UnsupportedModelError
from tensorpotential.torch_backend.ops.precision import same_dtype, to_dtype_of

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from tensorpotential.torch_backend.ops.precision import DtypePolicy

SILU_N2NORM_FACTOR: Final = 1.6759
"""Factor of TF's default activation ``silu_n2norm`` (``functions/nn.py:345``)."""
VARIANCE_NORM_EPSILON: Final = 1e-5
"""``MLPRadialFunction`` with ``norm``: made in the parameter dtype (sheet ``MLPRadialFunction``, section 4)."""
SCALAR_RMS_EPSILON: Final = 1e-16
"""``scalar_rms_ln``: a Python float (sheet ``LinMLPOut2ScalarTarget``, section 4)."""
CHANNEL_RMS_EPSILON: Final = 1e-10
"""``InvariantLayerRMSNorm``: made in the parameter dtype (sheet ``InvariantLayerRMSNorm``, section 4)."""


def silu_n2norm(x: torch.Tensor) -> torch.Tensor:
    """TF's default activation of ``FullyConnectedMLP``: ``silu(x) * 1.6759``."""
    return torch.nn.functional.silu(x) * SILU_N2NORM_FACTOR


ACTIVATIONS: Final[Mapping[str, Callable[[torch.Tensor], torch.Tensor]]] = {
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "silu_n2norm": silu_n2norm,
}
"""The activations by name: TF's ``ACTIVATION_DICT`` and the default of ``FullyConnectedMLP``."""


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in ACTIVATIONS:
        msg = f"unknown activation {name!r}: the twins have {', '.join(ACTIVATIONS)}"
        raise UnsupportedModelError(msg)
    return ACTIVATIONS[name]


class Linear(torch.nn.Module):
    """``x @ (w * norm)`` with ``norm = 1 / sqrt(n_in)``: TF's ``Linear`` and ``DenseLayer`` (no bias, no activation).

    Parameters
    ----------
    n_in, n_out
        Width of the input and of the output, at least 1.
    policy
        Dtype policy of the model: ``w`` has its parameter dtype.
    normalize
        ``False`` for a ``Linear`` built with ``normalize = False``: then ``norm`` is 1.
    device
        Device of ``w``.

    Raises
    ------
    ShapeError
        When a width is below 1.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        policy: DtypePolicy,
        *,
        normalize: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if n_in < 1 or n_out < 1:
            msg = f"a linear layer needs widths of at least 1, got {n_in} -> {n_out}"
            raise ShapeError(msg)
        self.n_in = n_in
        self.norm = 1.0 / n_in**0.5 if normalize else 1.0
        self.w = torch.nn.Parameter(
            torch.zeros((n_in, n_out), dtype=policy.param, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., n_in]`` to ``[..., n_out]`` in the dtype of ``w``.

        Raises
        ------
        ShapeError
            When the last axis of ``x`` is not ``n_in``.
        """
        if x.dim() == 0 or x.shape[-1] != self.n_in:
            msg = f"a linear layer of width {self.n_in} got an input of shape {tuple(x.shape)}"
            raise ShapeError(msg)
        w = self.w * self.norm
        return to_dtype_of(x, w) @ w


class MLP(torch.nn.Module):
    """A bias-free MLP of :class:`Linear` layers ``layer0``, ``layer1``, ...: TF's ``FullyConnectedMLP``.

    The state-dict key of a weight, ``layer<i>.w``, is the place of the weight under the MLP in the TF checkpoint
    (``<instruction>/mlp/layer<i>/w``).

    Parameters
    ----------
    sizes
        ``[n_in, *hidden_layers, n_out]``, at least two widths.
    activations
        One name of :data:`ACTIVATIONS` per hidden layer (``len(sizes) - 2``), applied after it; the last layer has none.
    policy
        Dtype policy of the model.
    normalize
        Passed to every layer.
    device
        Device of the weights.

    Raises
    ------
    ShapeError
        When ``sizes`` has fewer than two widths or ``activations`` does not have one name per hidden layer.
    UnsupportedModelError
        When an activation name is unknown.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        activations: Sequence[str],
        policy: DtypePolicy,
        *,
        normalize: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if len(sizes) < 2 or len(activations) != len(sizes) - 2:
            msg = (
                f"an MLP needs at least two widths and one activation per hidden layer, got widths {list(sizes)} "
                f"and activations {list(activations)}"
            )
            raise ShapeError(msg)
        self.activations = tuple(map(_activation, activations))
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            layer = Linear(n_in, n_out, policy, normalize=normalize, device=device)
            self.add_module(f"layer{i}", layer)

    @classmethod
    def fully_connected(
        cls,
        sizes: Sequence[int],
        activation: str | None,
        policy: DtypePolicy,
        *,
        device: torch.device | None = None,
    ) -> MLP:
        """Build the MLP of a ``FullyConnectedMLP(activation=...)``: one activation for every hidden layer.

        ``None`` is TF's default, ``silu(x) * 1.6759``; ``"silu"`` is the plain ``silu``.
        """
        name = "silu_n2norm" if activation is None else activation
        return cls(sizes, [name] * (len(sizes) - 2), policy, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., sizes[0]]`` to ``[..., sizes[-1]]``."""
        return self.forward_with_hidden(x)[0]

    def forward_with_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the output and the input of the last layer (``return_hidden=True`` in TF).

        With no hidden layer the second value is ``x`` itself.
        """
        *hidden, last = self.children()
        for layer, act in zip(hidden, self.activations, strict=True):
            x = act(layer(x))
        return last(x), x


def _inverse_rms(x: torch.Tensor, epsilon: torch.Tensor | float) -> torch.Tensor:
    return torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)


def variance_norm(y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``y * rsqrt(var(y) + 1e-5) * gamma`` over the last axis: the ``norm`` of ``MLPRadialFunction``.

    ``var`` is the population variance of each row around its mean, as ``tf.math.reduce_variance``; ``y`` itself is
    not centred. The epsilon is made in the dtype of ``gamma`` (the parameter dtype) and, with ``gamma``, cast to the
    dtype of ``y``.
    """
    epsilon = torch.as_tensor(
        VARIANCE_NORM_EPSILON, dtype=gamma.dtype, device=gamma.device
    )
    centred = y - y.mean(dim=-1, keepdim=True)
    inverse = torch.rsqrt(
        centred.square().mean(dim=-1, keepdim=True) + to_dtype_of(epsilon, y)
    )
    return y * inverse * to_dtype_of(gamma, y)


def scalar_rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * scale * rsqrt(mean(x**2) + 1e-16)`` over the last axis: TF's ``scalar_rms_ln`` without padding atoms.

    Raises
    ------
    DtypeError
        When ``x`` and ``scale`` have different dtypes (both are in the parameter dtype in a model).
    """
    same_dtype(x, scale)
    return x * scale * _inverse_rms(x, SCALAR_RMS_EPSILON)


def channel_rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * rsqrt(mean(x**2) + 1e-10) * scale`` over the channel axis, the last: ``InvariantLayerRMSNorm``.

    ``x`` is an invariant feature in the backend's layout ``[atoms, 1, n]`` (the ``full`` type), or its channels
    ``1 ..`` (``only_nonlin``, ``sep_lin_gate``); ``scale`` is ``[1, 1, n]``. The epsilon is made in the dtype of
    ``scale`` (the parameter dtype) and, with ``scale``, cast to the dtype of ``x``.
    """
    epsilon = torch.as_tensor(
        CHANNEL_RMS_EPSILON, dtype=scale.dtype, device=scale.device
    )
    return x * _inverse_rms(x, to_dtype_of(epsilon, x)) * to_dtype_of(scale, x)
