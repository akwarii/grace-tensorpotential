"""The values the loader returns: :class:`ModelSpec`, :class:`InstructionSpec` and :class:`InstructionRef`.

They live apart from :mod:`~tensorpotential.torch_backend.spec.loader` so that the registry, which checks them
against the option tables, and the loader, which builds them and calls the registry, do not import each other.
``loader`` re-exports the three names, which is where callers import them from.

The module imports neither TensorFlow nor torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

ModelFormat = Literal["wrapped", "flat", "list"]


@dataclass(frozen=True)
class InstructionRef:
    """An ``_instruction_`` reference of the yaml: the instruction that produced the value, by name."""

    name: str


@dataclass(frozen=True)
class InstructionSpec:
    """One instruction of the model.

    Parameters
    ----------
    name
        Instruction name, unique in the model (the key under which it stores its result).
    cls
        Dotted path of the TF class, as written in ``__cls__``.
    options
        Constructor arguments: the yaml's, then the pinned defaults of the keys the yaml omits. References are
        :class:`InstructionRef`; other nested values are as ``yaml.safe_load`` returns them.
    depends_on
        Names of the instructions referenced by ``options``, in order of first appearance, without repeats.
    index
        Position in the file (and in execution order).
    defaulted
        The keys of ``options`` that the yaml omits and the pinned defaults filled in; the registry words its error
        for such a key differently (the user did not write the value).
    """

    name: str
    cls: str
    options: Mapping[str, Any]
    depends_on: tuple[str, ...]
    index: int
    defaulted: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelSpec:
    """A whole model: its instructions in file order and the model-level options.

    Parameters
    ----------
    instructions
        The instructions in the order of the file, which is the order they run in; every reference points
        backwards.
    param_dtype
        ``"float32"`` or ``"float64"``: the ``metadata`` value, or the old-model default when there is none.
    format
        The yaml layout that was read.
    """

    instructions: tuple[InstructionSpec, ...]
    param_dtype: str
    format: ModelFormat

    def __getitem__(self, name: str) -> InstructionSpec:
        """Return the instruction called ``name``; ``KeyError`` when there is none."""
        for spec in self.instructions:
            if spec.name == name:
                return spec
        msg = name
        raise KeyError(msg)
