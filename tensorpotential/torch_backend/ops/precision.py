"""Dtype, device and precision policy of the torch backend: the one module that decides dtypes and casts (rule R5).

The policy every twin follows (section 6 of each spec sheet applies it to its class):

* **Parameters** have the parameter dtype, float32 or float64, read from the stored weights and never from the yaml or
  the torch default (spec sheets, G3): :class:`DtypePolicy`, :meth:`DtypePolicy.from_weights`.
* **Geometry** (bond vectors and what is computed from them alone: lengths, radial bases, spherical harmonics) is
  float64 whatever the parameter dtype (G1): :data:`GEOMETRY_DTYPE`.
* **Casts** happen here only: :meth:`DtypePolicy.to_param`, :meth:`DtypePolicy.to_geometry`, :func:`to_dtype_of` (TF's
  ``tf.cast(x, y.dtype)``), :func:`promote_like_tf` where TensorFlow promotes silently (below), and :func:`same_dtype`
  where TF refuses mixed operands. A twin never calls ``Tensor.to(dtype)`` itself.
* **Constants** are created with an explicit dtype and the device of the tensor they meet
  (:meth:`DtypePolicy.param_constant`, :meth:`DtypePolicy.geometry_constant`); the backend never reads
  ``torch.get_default_dtype()`` and never moves a tensor to another device (a test scans the package for both).
* **Energy**: atomic energies are summed per structure in float64 (``ops.scatter.structure_energy`` through :func:`to_energy`), as TF casts them to the
  dtype of the pair forces before its segment sum (``tpmodel.py:332-335``); a float32 sum would depend on the order of
  the terms, hence on the batch layout.
* **Layout**: every equivariant feature is ``[atoms, lm, n]`` (:data:`FEATURE_LAYOUT`), the flat ``ir_mul`` layout of
  cuEquivariance for operands of equal multiplicity, so no transpose appears in a forward pass. TF's default is
  ``[atoms, n, lm]`` and its ``lm_first`` option ``[lm, atoms, n]``; TF fixtures are permuted at the test boundary only.
* **TF32**: :func:`precision_scope` switches every reduced-precision float32 path off around a forward pass and
  restores the caller's settings; nothing in the backend sets them globally, and importing it changes no global state
  of torch (TF flips its own flags at import, which the twins must not copy: other models in the process would change).

Where TensorFlow promotes silently. With numpy behaviour on (``dtype_conversion_mode="all"``, G2), TF gives float64 for
a float32 tensor combined with any float64 operand, a 0-d tensor or a ``numpy.float64`` included; torch gives float32
when the float64 operand is 0-d or a numpy scalar, and float64 only for a float64 tensor of one or more dimensions
(measured on TF 2.20 and torch 2.14). A Python ``float`` takes the dtype of the tensor in both. A twin whose sheet says
TF promotes there calls :func:`promote_like_tf` on the operands first.

Layer: ops, the leaf of it: imports torch, numpy and the errors; the other ops import it.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from tensorpotential.torch_backend.errors import DtypeError, TorchBackendError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from numpy.typing import ArrayLike, DTypeLike

PARAM_DTYPES: Final = (torch.float32, torch.float64)
"""The parameter dtypes a model may have."""
GEOMETRY_DTYPE: Final = torch.float64
"""The dtype of bond vectors and of everything computed from them alone (G1)."""
ENERGY_DTYPE: Final = torch.float64
"""The dtype in which atomic energies are summed per structure."""
INDEX_DTYPES: Final = (torch.int32, torch.int64)
"""The dtypes of index tensors (TF's are int32, torch's own int64)."""
FEATURE_LAYOUT: Final = ("atoms", "lm", "n")
"""Axes of every equivariant feature inside the backend: atom, angular index ``l*l + l + m`` (G7), channel."""

PRECISION_KNOBS: Final = (
    "",
    "cuda.matmul",
    "cudnn.conv",
    "cudnn.rnn",
    "mkldnn.matmul",
    "mkldnn.conv",
    "mkldnn.rnn",
)
"""The ``fp32_precision`` attributes under ``torch.backends`` (``""``: the generic one) that :func:`precision_scope` sets.

Only this API of torch 2.9 is used: once a process mixes it with the legacy flags (``allow_tf32``,
``set_float32_matmul_precision``), reading a legacy flag raises, and saving the new attributes restores both exactly.
"""
FULL_PRECISION: Final = "ieee"
"""The ``fp32_precision`` value that computes float32 in float32 (no TF32, no bfloat16)."""

_TORCH_DTYPE: Final = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not tensor.is_floating_point():
        msg = f"only floating tensors are cast by the policy, got {tensor.dtype}"
        raise DtypeError(msg)
    return tensor.to(dtype)


@dataclass(frozen=True)
class DtypePolicy:
    """The dtypes of one model: its parameter dtype, and float64 for geometry.

    Parameters
    ----------
    param
        ``torch.float32`` or ``torch.float64``.

    Raises
    ------
    DtypeError
        For any other dtype.
    """

    param: torch.dtype

    def __post_init__(self) -> None:
        """Refuse a parameter dtype other than float32 and float64."""
        if self.param not in PARAM_DTYPES:
            msg = f"the parameter dtype must be float32 or float64, got {self.param}"
            raise DtypeError(msg)

    @classmethod
    def from_weights(cls, dtypes: Iterable[DTypeLike]) -> DtypePolicy:
        """Return the policy of a model from the dtypes of its stored weights.

        Integer and text arrays (element maps) are ignored; every floating weight must have the same dtype.

        Raises
        ------
        DtypeError
            When the floating weights have no dtype, several, or one other than float32 and float64.
        """
        floating = sorted(
            {np.dtype(d) for d in dtypes if np.dtype(d).kind == "f"}, key=str
        )
        if len(floating) != 1 or floating[0] not in _TORCH_DTYPE:
            found = ", ".join(map(str, floating)) or "none"
            msg = f"the floating weights must all be float32 or all float64, found {found}"
            raise DtypeError(msg)
        return cls(_TORCH_DTYPE[floating[0]])

    def to_param(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a floating tensor to the parameter dtype (no copy when it has it already)."""
        return _cast(tensor, self.param)

    def to_geometry(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a floating tensor to the geometry dtype (no copy when it has it already)."""
        return _cast(tensor, GEOMETRY_DTYPE)

    def param_constant(self, values: ArrayLike, device: torch.device) -> torch.Tensor:
        """Make a constant (a coupling table, an epsilon) in the parameter dtype on ``device``."""
        return torch.as_tensor(values, dtype=self.param, device=device)

    def geometry_constant(
        self, values: ArrayLike, device: torch.device
    ) -> torch.Tensor:
        """Make a constant in the geometry dtype on ``device``."""
        return torch.as_tensor(values, dtype=GEOMETRY_DTYPE, device=device)


def promote_like_tf(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cast the floating ``tensors`` to the dtype TF gives an elementwise op on them: the widest one, 0-d included.

    Non-floating tensors are returned unchanged.

    Raises
    ------
    DtypeError
        When none of the tensors is floating.
    """
    floating = [t.dtype for t in tensors if t.is_floating_point()]
    if not floating:
        msg = "promote_like_tf needs at least one floating tensor"
        raise DtypeError(msg)
    widest = max(floating, key=lambda d: torch.finfo(d).bits)
    return tuple(t.to(widest) if t.is_floating_point() else t for t in tensors)


def to_dtype_of(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cast a floating tensor to the dtype of ``reference``: TF's ``tf.cast(x, y.dtype)``, the commonest cast of the sheets.

    Raises
    ------
    DtypeError
        When ``tensor`` is not floating.
    """
    return _cast(tensor, reference.dtype)


def same_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Return the one dtype of ``tensors``, for an op that TF refuses to run on mixed dtypes (no cast, no promotion).

    Raises
    ------
    DtypeError
        When the tensors have more than one dtype, naming them; or when none is given.
    """
    dtypes = sorted({str(t.dtype) for t in tensors})
    if len(dtypes) != 1:
        msg = f"the operands must have one dtype, got {', '.join(dtypes) or 'none'}"
        raise DtypeError(msg)
    return tensors[0].dtype


def to_energy(atomic_energy: torch.Tensor) -> torch.Tensor:
    """Cast atomic energies to the dtype they are summed in, float64."""
    return _cast(atomic_energy, ENERGY_DTYPE)


def check_index(index: torch.Tensor) -> None:
    """Refuse an index tensor whose dtype is not ``int32`` or ``int64`` (reads no value).

    Raises
    ------
    DtypeError
        For any other dtype.
    """
    if index.dtype not in INDEX_DTYPES:
        msg = f"an index must be int32 or int64, got {index.dtype}"
        raise DtypeError(msg)


def check_floating(tensor: torch.Tensor) -> None:
    """Refuse a tensor that is not floating, for an op that keeps the dtype of its input (reads no value).

    Raises
    ------
    DtypeError
        For an integer, boolean or complex tensor.
    """
    if not tensor.is_floating_point():
        msg = f"a floating tensor is needed, got {tensor.dtype}"
        raise DtypeError(msg)


def _knob(path: str) -> object:
    return functools.reduce(getattr, filter(None, path.split(".")), torch.backends)


def fp32_precision_state() -> dict[str, str]:
    """Return the ``fp32_precision`` value of every attribute of :data:`PRECISION_KNOBS`."""
    return {path: _knob(path).fp32_precision for path in PRECISION_KNOBS}  # ty: ignore[unresolved-attribute]


def set_fp32_precision(state: Mapping[str, str]) -> None:
    """Set the ``fp32_precision`` attributes named in ``state`` (keys of :data:`PRECISION_KNOBS`), in its order.

    Raises
    ------
    TorchBackendError
        When a key is not in :data:`PRECISION_KNOBS` (nothing is set then).
    """
    unknown = sorted(state.keys() - set(PRECISION_KNOBS))
    if unknown:
        msg = f"not fp32_precision attributes of the policy: {', '.join(unknown)}"
        raise TorchBackendError(msg)
    for path, value in state.items():
        setattr(_knob(path), "fp32_precision", value)  # noqa: B010  (a property of torch, set by name)


@contextlib.contextmanager
def precision_scope() -> Iterator[None]:
    """Compute float32 in full precision inside the block, then restore the caller's settings.

    Sets every attribute of :data:`PRECISION_KNOBS` to ``"ieee"`` (no TF32 on CUDA and cuDNN, no reduced precision in
    oneDNN on the CPU) and writes the saved values back on exit, also when the block raises. The settings are
    process-wide while the block runs: a thread that runs another model at the same time sees them.

    Examples
    --------
    >>> with precision_scope():
    ...     energy = model(inputs)
    """
    saved = fp32_precision_state()
    try:
        set_fp32_precision(dict.fromkeys(PRECISION_KNOBS, FULL_PRECISION))
        yield
    finally:
        set_fp32_precision(saved)
