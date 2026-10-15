"""The twin protocol: a ``torch.nn.Module`` with a ``spec`` and a ``forward`` over named inputs satisfies it.

``ty`` checks the assignments to :class:`Twin` below, so a protocol that an ``nn.Module`` twin cannot satisfy fails
the type check of this file; the test runs the twin through the protocol the way the executor will.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tensorpotential.torch_backend.protocol import Twin
from tensorpotential.torch_backend.spec.model import InstructionSpec


class Scale(torch.nn.Module):
    """A twin that multiplies the value of the instruction it depends on by a stored weight."""

    def __init__(self, spec: InstructionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.weight = torch.nn.Parameter(
            torch.tensor(spec.options["factor"], dtype=torch.float64)
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.weight * inputs[self.spec.depends_on[0]]


def _spec() -> InstructionSpec:
    return InstructionSpec(
        name="scaled",
        cls="toy.Scale",
        options={"factor": 3.0},
        depends_on=("x",),
        index=1,
    )


def test_a_module_twin_runs_through_the_protocol():
    twin: Twin = Scale(_spec())
    inputs = {"x": torch.tensor([1.0, -2.0], dtype=torch.float64)}
    out = twin.forward(inputs)
    assert twin.spec.name == "scaled"
    assert torch.equal(out, torch.tensor([3.0, -6.0], dtype=torch.float64))
    assert torch.equal(inputs["x"], torch.tensor([1.0, -2.0], dtype=torch.float64))


def test_the_protocol_reads_the_spec_the_twin_was_built_from():
    spec = _spec()
    twin: Twin = Scale(spec)
    assert twin.spec is spec
