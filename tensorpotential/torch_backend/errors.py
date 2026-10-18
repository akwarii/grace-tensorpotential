"""Exception types of the torch backend: the one module that defines them (rule R5).

Every error is a :class:`TorchBackendError`, which is a
:class:`~tensorpotential.errors.TensorPotentialError`, so one ``except`` catches every error the library raises on
purpose. Below it, each family also derives from the builtin type a caller expects: a ``model.yaml`` that cannot be
loaded is a :class:`SpecError` (a ``ValueError``), a dtype the dtype policy refuses is a :class:`DtypeError`
(a ``TypeError``), and a test can ask for the kind. The message names the object
involved (instruction, class, key, weight or atom) and what to do: nothing is skipped or approximated silently
(rule R3).

Layer: below every other module of the backend; it imports only :mod:`tensorpotential.errors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tensorpotential.errors import TensorPotentialError

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


class TorchBackendError(TensorPotentialError):
    """Base of the errors the torch backend raises on purpose."""


class SpecError(TorchBackendError, ValueError):
    """A ``model.yaml`` cannot be turned into a :class:`~tensorpotential.torch_backend.spec.loader.ModelSpec`."""


class WeightMismatchError(TorchBackendError, ValueError):
    """Stored weights do not fit the model: a key is missing or left over, or a shape or dtype differs."""


class DtypeError(TorchBackendError, TypeError):
    """A tensor or a weight has a dtype the dtype policy does not allow (``ops.precision``)."""


class ShapeError(TorchBackendError, ValueError):
    """Tensors given to an op have shapes that do not fit together (for example one segment index per row)."""


class OptionValueError(TorchBackendError, ValueError):
    """An op is given an option value outside its domain (a radial basis without functions, an unknown activation)."""


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
