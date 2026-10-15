"""The twin protocol: a ``torch.nn.Module`` with a ``spec`` and a ``forward`` over named inputs satisfies it.

``ty`` checks the assignments to :class:`Twin` below, so a protocol that an ``nn.Module`` twin cannot satisfy fails
the type check of this file; the test runs the twin through the protocol the way the executor will.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tensorpotential.torch_backend.protocol import Twin
from tensorpotential.torch_backend.spec.model import InstructionSpec


class Double(torch.nn.Module):
    """The twin of the protocol's docstring: twice the value of the instruction it depends on."""

    def __init__(self, spec: InstructionSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return 2.0 * inputs[self.spec.depends_on[0]]


def test_a_module_twin_runs_through_the_protocol():
    spec = InstructionSpec(
        name="doubled", cls="toy.Double", options={}, depends_on=("x",), index=1
    )
    twin: Twin = Double(spec)
    out = twin.forward({"x": torch.tensor([1.0, -2.0], dtype=torch.float64)})
    assert twin.spec is spec
    assert torch.equal(out, torch.tensor([2.0, -4.0], dtype=torch.float64))
