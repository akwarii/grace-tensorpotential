"""Dtype audit of torch code: a dispatch mode that records every op mixing float dtypes and every host read.

Inside ``with DtypeAudit() as audit:`` every aten op that torch dispatches (the forward pass and, when it runs inside
the block, the backward pass) is looked at before it runs:

* ``audit.mixed`` lists the ops whose tensor arguments have more than one floating dtype: an implicit promotion or
  demotion that the dtype policy (``torch_backend/ops/precision.py``) should have made explicit;
* ``audit.host_reads`` lists the ops that read a tensor value on the host (``.item()``, ``nonzero``, ``unique``):
  each one synchronises a GPU and breaks a compiled graph.

A test asserts both are empty with :meth:`DtypeAudit.assert_clean`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

HOST_READS: Final = frozenset({
    "aten::_local_scalar_dense",
    "aten::nonzero",
    "aten::unique",
    "aten::_unique",
    "aten::_unique2",
    "aten::unique_dim",
    "aten::unique_consecutive",
})
"""Schema names of the ops that move a value computed on the device to the host."""


class DtypeAudit(TorchDispatchMode):
    """Record the ops that mix float dtypes and the ops that read a value on the host (see the module)."""

    def __init__(self) -> None:
        super().__init__()
        self.mixed: list[tuple[str, tuple[str, ...]]] = []
        self.host_reads: list[str] = []

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> object:
        """Look at one aten op, then run it unchanged."""
        kwargs = kwargs or {}
        name = func._schema.name  # ty: ignore[unresolved-attribute]  (an OpOverload)
        floats = sorted({
            str(leaf.dtype)
            for leaf in tree_leaves((args, kwargs))
            if isinstance(leaf, torch.Tensor) and leaf.is_floating_point()
        })
        if len(floats) > 1:
            self.mixed.append((name, tuple(floats)))
        if name in HOST_READS:
            self.host_reads.append(name)
        return func(*args, **kwargs)

    def assert_clean(self) -> None:
        """Fail when an op mixed float dtypes or read a value on the host."""
        assert not self.mixed, f"ops mixing float dtypes: {self.mixed}"
        assert not self.host_reads, f"host reads: {self.host_reads}"
