"""Registry of the TF instruction classes, and the strict check of a loaded model against it.

:data:`REGISTRY` has one :class:`ClassEntry` for every ``TPInstruction`` subclass of the TF source: ``supported``
(a twin is planned for it, the option rules of
:data:`~tensorpotential.torch_backend.spec.options.SUPPORTED_OPTIONS` apply, and it has a spec sheet) or
``rejected`` with the reason. The completeness tests (``tests/test_registry.py``) and the scheduled job that runs
``tools/instruction_ast.py check`` compare this table with the classes the source defines, so a class added
upstream cannot slip in unnoticed (rule R3).

:func:`check_supported` is the strict half of the loader. It never stops at the first finding: every
unsupported class and option of the model is listed in one :class:`UnsupportedModelError`, after the MACE
loaders' lesson that a user should fix a yaml in one pass. Rules:

- a class matches by its **exact** dotted path; a subclass of a supported class is not supported, whatever it
  inherits (the twin cannot know what the subclass changed);
- an option the class does not have is an error that names the nearest valid key;
- an option the class needs, and that neither the yaml nor the pinned defaults give, is an error;
- an option value outside the rule is an error that says what is accepted (or, for a value seen in a shipped yaml
  and rejected on purpose, why).

The module imports neither TensorFlow nor torch.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from tensorpotential.torch_backend.spec.errors import Problem, UnsupportedModelError
from tensorpotential.torch_backend.spec.model import InstructionRef
from tensorpotential.torch_backend.spec.options import (
    DEFAULTS,
    REJECTED_CLASSES,
    REJECTED_OPTIONS,
    SUPPORTED_OPTIONS,
    OptionRule,
)

ClassStatus = Literal["supported", "rejected"]

SHEET_DIRECTORY: Final = "docs/torch_backend/spec"

# Class path -> the test file (relative to the repository root) that compares the twin with the golden fixtures
# of the TF class. Empty until Stage 3: each TWIN issue adds the classes it builds, and removes them from
# ``EXPECTED_WITHOUT_GOLDEN`` in ``tests/test_registry.py`` in the same pull request (which fails otherwise).
GOLDEN_TESTS: Final[Mapping[str, str]] = {}

_NEAREST_CLASS_CUTOFF: Final = 0.85
_VALUE_WIDTH: Final = 60


@dataclass(frozen=True)
class ClassEntry:
    """What the torch backend decided about one TF instruction class.

    Parameters
    ----------
    path
        Dotted path, as written in ``__cls__``.
    status
        ``supported`` or ``rejected``.
    reason
        Why a rejected class has no twin (empty for a supported one).
    sheet
        The spec sheet of a supported class, relative to the repository root (``None`` for a rejected class).
    golden_test
        The test file that compares the twin with the golden fixtures, ``None`` while there is none.
    """

    path: str
    status: ClassStatus
    reason: str
    sheet: str | None
    golden_test: str | None

    @property
    def name(self) -> str:
        """The class name without its module."""
        return self.path.rsplit(".", 1)[-1]


class InstructionLike(Protocol):
    """The part of :class:`~tensorpotential.torch_backend.spec.model.InstructionSpec` the check reads."""

    @property
    def name(self) -> str:
        """Instruction name."""
        ...

    @property
    def cls(self) -> str:
        """Dotted class path."""
        ...

    @property
    def options(self) -> Mapping[str, Any]:
        """Options after the pinned defaults were applied."""
        ...

    @property
    def defaulted(self) -> frozenset[str]:
        """The options that the pinned defaults filled in."""
        ...


def _build_registry() -> dict[str, ClassEntry]:
    entries = {
        path: ClassEntry(
            path,
            "supported",
            "",
            f"{SHEET_DIRECTORY}/{path.rsplit('.', 1)[-1]}.md",
            GOLDEN_TESTS.get(path),
        )
        for path in SUPPORTED_OPTIONS
    }
    entries |= {
        path: ClassEntry(path, "rejected", reason, None, None)
        for path, reason in REJECTED_CLASSES.items()
    }
    return dict(sorted(entries.items()))


REGISTRY: Final[Mapping[str, ClassEntry]] = _build_registry()


def supported_classes() -> tuple[str, ...]:
    """Dotted paths of the supported classes, sorted."""
    return tuple(p for p, e in REGISTRY.items() if e.status == "supported")


def value_kind(value: Any) -> str:
    """Type name of an option value, as the option rules spell it (``list[int]``, ``dict[str,int]``, ``ref``).

    An :class:`InstructionRef` is ``ref``. The names are those of ``tools/scan_options.value_kind``, which reads
    the raw yaml (where a reference is a mapping with an ``_instruction_`` key).
    """
    if value is None:
        return "null"
    if isinstance(value, InstructionRef):
        return "ref"
    if isinstance(value, dict):
        return f"dict[{_joined(value.keys())},{_joined(value.values())}]"
    if isinstance(value, list):
        return f"list[{_joined(value)}]"
    return type(value).__name__


def _joined(items: Iterable[Any]) -> str:
    kinds = sorted({value_kind(item) for item in items})
    return "|".join(kinds) if kinds else "empty"


def accepts(rule: OptionRule, value: Any) -> bool:
    """Whether ``rule`` allows ``value``: its kind is listed, and, when the rule lists values, so is the value.

    For a ``list[str]`` rule the values are those allowed as elements.
    """
    if value_kind(value) not in rule.kinds:
        return False
    if rule.values is None:
        return True
    items = value if isinstance(value, list) else [value]
    return all(item in rule.values for item in items)


def _shown(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _VALUE_WIDTH else text[: _VALUE_WIDTH - 3] + "..."


def _supported_text(rule: OptionRule) -> str:
    if rule.values is not None:
        return "supported values: " + ", ".join(repr(v) for v in rule.values)
    return "supported kinds: " + ", ".join(rule.kinds)


def class_problem(name: str, cls: str) -> Problem | None:
    """Return the problem with the class of an instruction, or ``None`` when the class is supported."""
    entry = REGISTRY.get(cls)
    if entry is None:
        near = difflib.get_close_matches(
            cls, REGISTRY, n=1, cutoff=_NEAREST_CLASS_CUTOFF
        )
        hint = f"; did you mean {near[0]}?" if near else ""
        message = (
            f"instruction {name!r}: class {cls} is unknown to the torch backend (no twin and no entry in the "
            "option table); classes are matched by exact name, so a subclass of a supported class is not "
            f"supported either{hint}"
        )
        return Problem("unknown_class", name, cls, None, message)
    if entry.status == "rejected":
        message = f"instruction {name!r}: class {cls} is not supported: {entry.reason}"
        return Problem("rejected_class", name, cls, None, message)
    return None


def option_problems(ins: InstructionLike) -> list[Problem]:
    """Return the unknown, missing and unsupported options of an instruction of a supported class."""
    rules = SUPPORTED_OPTIONS[ins.cls]
    where = f"instruction {ins.name!r} ({ins.cls.rsplit('.', 1)[-1]})"
    found: list[Problem] = []
    valid = [key for key, rule in rules.items() if not rule.ignored]
    for key in ins.options:
        if key not in rules:
            near = difflib.get_close_matches(key, valid, n=1)
            hint = f"; did you mean {near[0]!r}?" if near else ""
            message = f"{where}: unknown option {key!r}{hint}"
            found.append(Problem("unknown_option", ins.name, ins.cls, key, message))
    defaults = DEFAULTS.get(ins.cls, {})
    for key in valid:
        if key not in ins.options and key not in defaults:
            message = f"{where}: required option {key!r} is missing"
            found.append(Problem("missing_option", ins.name, ins.cls, key, message))
    for key, value in ins.options.items():
        if key in rules and not accepts(rules[key], value):
            found.append(_value_problem(ins, where, key, value))
    return found


def _value_problem(ins: InstructionLike, where: str, key: str, value: Any) -> Problem:
    kind = value_kind(value)
    for rejection in REJECTED_OPTIONS.get(ins.cls, {}).get(key, ()):
        if rejection.kind == kind:
            given = _shown(value) + (" (the default)" if key in ins.defaulted else "")
            message = f"{where}: option {key!r} = {given} is not supported: {rejection.reason}"
            return Problem("rejected_value", ins.name, ins.cls, key, message)
    rule = SUPPORTED_OPTIONS[ins.cls][key]
    if key in ins.defaulted:
        message = (
            f"{where}: option {key!r} is absent from the yaml, so the default {_shown(value)} applies, and "
            f"it is not supported ({_supported_text(rule)}); write the option explicitly"
        )
    else:
        message = f"{where}: option {key!r} = {_shown(value)} ({kind}) is not supported ({_supported_text(rule)})"
    return Problem("unsupported_value", ins.name, ins.cls, key, message)


def find_problems(instructions: Iterable[InstructionLike]) -> list[Problem]:
    """Every problem of the instructions, in file order (class problems first within an instruction)."""
    found: list[Problem] = []
    for ins in instructions:
        problem = class_problem(ins.name, ins.cls)
        if problem is not None:
            found.append(problem)
        else:
            found += option_problems(ins)
    return found


def check_supported(instructions: Iterable[InstructionLike]) -> None:
    """Raise one :class:`UnsupportedModelError` that lists every class and option the twins do not cover.

    Parameters
    ----------
    instructions
        The instructions of a loaded model, after the pinned defaults were applied.

    Raises
    ------
    UnsupportedModelError
        Its ``problems`` hold one :class:`Problem` per finding and its message one line each.
    """
    problems = find_problems(instructions)
    if problems:
        lines = "\n  ".join(p.message for p in problems)
        msg = f"the model cannot be loaded by the torch backend:\n  {lines}"
        raise UnsupportedModelError(msg, tuple(problems))
