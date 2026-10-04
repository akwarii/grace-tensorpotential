"""Exception types of the spec loader.

Every error is a :class:`SpecError` (a ``ValueError``), so a caller that only wants to know that a ``model.yaml``
cannot be loaded catches one type, and a test can ask for the kind. The message names the instruction and, where
there is one, the class or the key involved: nothing is skipped or approximated silently (rule R3).
"""

from __future__ import annotations


class SpecError(ValueError):
    """A ``model.yaml`` cannot be turned into a :class:`~tensorpotential.torch_backend.spec.loader.ModelSpec`."""


class MalformedModelError(SpecError):
    """The yaml is not a model description: wrong container, missing ``__cls__`` or ``name``, duplicate names."""


class InstructionReferenceError(SpecError):
    """An ``_instruction_`` reference that cannot be resolved in the order of the file."""


class DanglingReferenceError(InstructionReferenceError):
    """A reference names an instruction that is not in the file."""


class CyclicReferenceError(InstructionReferenceError):
    """The references form a cycle (an instruction depends on itself, directly or not)."""


class ForwardReferenceError(InstructionReferenceError):
    """A reference names an instruction that comes later in the file.

    Instructions run in file order and an output instruction overwrites its target's entry in place, so the
    order is part of the model: TensorFlow's own loader cannot resolve such a reference either.
    """


class UnsupportedModelError(SpecError):
    """The model uses a class (or a model-level value) that the torch twins do not cover."""
