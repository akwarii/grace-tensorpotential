"""Twin of TF's ``ScalarChemicalEmbedding``: a learned vector per chemical element.

Spec sheet ``ScalarChemicalEmbedding.md``, section 5: the output is the weight table ``w [n_elements, embedding_size]``
itself, in the parameter dtype; the consumers gather its rows by element index and cast it. ``init`` and
``is_trainable`` only matter for training and are ignored; ``lora_config`` is refused by the option table.

The element map of the yaml (symbol -> row) is kept as ``element_map``. Its values must be the rows
``0 .. n_elements - 1``, since the element indices of the atoms address the rows. TF also stores the map as two
checkpoint variables (``element_map_symbols``, ``element_map_index``); they are not weights of the twin, and comparing
them with the yaml belongs to checkpoint loading.

Layer: instructions; imports torch and the errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.torch_backend.errors import OptionValueError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec


class ScalarChemicalEmbedding(torch.nn.Module):
    """``ScalarChemicalEmbedding``: the table ``w [n_elements, embedding_size]``.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``element_map``, ``embedding_size``).
    policy
        Dtype policy of the model: ``w`` has its parameter dtype. It starts at zero; the stored weight is loaded into it.

    Raises
    ------
    OptionValueError
        When the values of ``element_map`` are not the rows ``0 .. n_elements - 1``, or ``embedding_size`` is below 1.
    """

    def __init__(self, spec: InstructionSpec, policy: DtypePolicy) -> None:
        """Make the zero table of the element map's size."""
        super().__init__()
        self.spec = spec
        element_map = dict(spec.options["element_map"])
        if sorted(element_map.values()) != list(range(len(element_map))):
            msg = (
                f"instruction {spec.name!r} (ScalarChemicalEmbedding): the element_map values must be the rows "
                f"0 .. {len(element_map) - 1}, got {sorted(element_map.values())}"
            )
            raise OptionValueError(msg)
        self.embedding_size = int(spec.options["embedding_size"])
        if self.embedding_size < 1:
            msg = f"instruction {spec.name!r} (ScalarChemicalEmbedding): embedding_size must be at least 1, got {self.embedding_size}"
            raise OptionValueError(msg)
        self.element_map: Mapping[str, int] = element_map
        self.w = torch.nn.Parameter(
            torch.zeros((len(element_map), self.embedding_size), dtype=policy.param)
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:  # noqa: ARG002 (the table reads no input)
        """Return the table ``[n_elements, embedding_size]`` in the parameter dtype."""
        return self.w
