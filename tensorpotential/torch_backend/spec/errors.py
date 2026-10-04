"""Exception types of the spec loader.

Every error is a :class:`SpecError` (a ``ValueError``), so a caller that only wants to know that a ``model.yaml``
cannot be loaded catches one type, and a test can ask for the kind. The message names the instruction and, where
there is one, the class or the key involved: nothing is skipped or approximated silently (rule R3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProblemKind = Literal[
    "unknown_class",
    "rejected_class",
    "unknown_option",
    "missing_option",
    "unsupported_value",
    "rejected_value",
]


@dataclass(frozen=True)
class Problem:
    """One reason a model cannot be loaded: what kind, where, and the sentence that says it.

    Parameters
    ----------
    kind
        ``unknown_class`` (no entry in the registry), ``rejected_class`` (a registry entry with a reason),
        ``unknown_option`` (a key the class does not have), ``missing_option`` (a key the class needs and the
        yaml and the pinned defaults both lack), ``unsupported_value`` (a value outside what the twin accepts) or
        ``rejected_value`` (a value seen in a shipped yaml that has no twin, with the reason).
    instruction
        Name of the instruction.
    cls
        Dotted path of its class, as written in ``__cls__``.
    option
        The key, for the four kinds that concern one.
    message
        The sentence, which names the instruction, the class and the option.
    """

    kind: ProblemKind
    instruction: str
    cls: str
    option: str | None
    message: str


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
    """The model uses a class, an option or a model-level value that the torch twins do not cover.

    ``problems`` holds every finding of the check as :class:`Problem` records (empty for a model-level value):
    the message lists them all, so a user fixes a yaml in one pass, and a test asks for the kind.
    """

    def __init__(self, message: str, problems: tuple[Problem, ...] = ()) -> None:
        super().__init__(message)
        self.problems = problems
