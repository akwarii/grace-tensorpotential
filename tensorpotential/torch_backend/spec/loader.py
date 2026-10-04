"""TF-free loader of ``model.yaml``: parse, resolve references, apply the pinned defaults, order the instructions.

The result is a :class:`ModelSpec`: the instructions in the order of the file (the order they run in; see
:class:`~tensorpotential.torch_backend.spec.errors.ForwardReferenceError`), each as an :class:`InstructionSpec`
whose ``options`` hold every constructor argument that the yaml or
:data:`~tensorpotential.torch_backend.spec.options.DEFAULTS` gives. An ``_instruction_`` reference becomes an
:class:`InstructionRef`, wherever it sits in the options (a list of references too).

Three yaml layouts are read, as ``load_instructions`` reads them (``instructions/base.py``):

- wrapped: ``{metadata: {...}, instructions: {name: {...}, ...}}``;
- flat: ``{name: {...}, ...}`` (a ``__metadata__`` entry is dropped; the parameter dtype is then the old-model
  default, as ``metadata_utils.resolve_param_dtype`` does);
- list: ``[{...}, ...]`` (the oldest format).

The loader reads the file with ``yaml.safe_load`` and never executes a string of the file: ``__cls__`` is looked
up in tables, not imported (TF's ``str_to_class`` runs ``exec`` on it, a defect the twins must not copy).

Out of scope here (the registry and strict-loader checks of the issue that follows): the value of each option
against :data:`~tensorpotential.torch_backend.spec.options.SUPPORTED_OPTIONS`, unknown or missing keys. A class
that is not supported is rejected, and the model-level ``param_dtype`` is checked.

The module imports neither TensorFlow nor torch.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import yaml

from tensorpotential.torch_backend.spec.errors import (
    CyclicReferenceError,
    DanglingReferenceError,
    ForwardReferenceError,
    MalformedModelError,
    UnsupportedModelError,
)
from tensorpotential.torch_backend.spec.options import (
    DEFAULTS,
    MODEL_DEFAULTS,
    MODEL_OPTIONS,
    REJECTED_CLASSES,
    SUPPORTED_OPTIONS,
)

ModelFormat = Literal["wrapped", "flat", "list"]

_CLASS_KEY: Final = "__cls__"
_NAME_KEY: Final = "name"
_REF_KEY: Final = "_instruction_"
_LEGACY_METADATA_KEY: Final = "__metadata__"


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
    """

    name: str
    cls: str
    options: Mapping[str, Any]
    depends_on: tuple[str, ...]
    index: int


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


def load_model_spec(path: str | os.PathLike[str]) -> ModelSpec:
    """Read a ``model.yaml`` and return its :class:`ModelSpec`.

    Raises
    ------
    SpecError
        (a subclass) when the file is not a loadable model; see :func:`parse_model_spec`.
    """
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_model_spec(raw)


def parse_model_spec(raw: object) -> ModelSpec:
    """Turn the content of a ``model.yaml`` (as ``yaml.safe_load`` returns it) into a :class:`ModelSpec`.

    The checks run in this order, and the first failing one raises: layout and keys
    (:class:`MalformedModelError`), references (:class:`DanglingReferenceError`, :class:`CyclicReferenceError`,
    :class:`ForwardReferenceError`), then the model-level options and the classes
    (:class:`UnsupportedModelError`, which lists every offending class at once).
    """
    entries, metadata, layout = _split_layout(raw)
    specs = _build_specs(entries)
    _check_references(specs)
    _check_supported(specs)
    return ModelSpec(
        instructions=specs,
        param_dtype=_resolve_param_dtype(metadata),
        format=layout,
    )


def _split_layout(raw: object) -> tuple[list[Any], Mapping[str, Any], ModelFormat]:
    if isinstance(raw, list):
        return raw, {}, "list"
    if not isinstance(raw, Mapping):
        msg = f"a model.yaml holds a mapping or a list of instructions, not {type(raw).__name__}"
        raise MalformedModelError(msg)
    if "instructions" in raw:
        body, metadata = raw["instructions"], raw.get("metadata") or {}
        if not isinstance(body, Mapping):
            msg = f"'instructions' must map names to instructions, not {type(body).__name__}"
            raise MalformedModelError(msg)
        if not isinstance(metadata, Mapping):
            msg = f"'metadata' must be a mapping, not {type(metadata).__name__}"
            raise MalformedModelError(msg)
        return list(body.values()), metadata, "wrapped"
    flat = {k: v for k, v in raw.items() if k != _LEGACY_METADATA_KEY}
    return list(flat.values()), {}, "flat"


def _build_specs(entries: Iterable[Any]) -> tuple[InstructionSpec, ...]:
    specs: list[InstructionSpec] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        spec = _build_spec(entry, index)
        if spec.name in seen:
            msg = f"instruction name {spec.name!r} is used twice (entries {seen[spec.name]} and {index})"
            raise MalformedModelError(msg)
        seen[spec.name] = index
        specs.append(spec)
    return tuple(specs)


def _build_spec(entry: Any, index: int) -> InstructionSpec:
    if not isinstance(entry, Mapping):
        msg = f"entry {index} of the model is a {type(entry).__name__}, not a mapping"
        raise MalformedModelError(msg)
    name = entry.get(_NAME_KEY)
    if not isinstance(name, str) or not name:
        msg = f"entry {index} has no string 'name'"
        raise MalformedModelError(msg)
    cls = entry.get(_CLASS_KEY)
    if not isinstance(cls, str) or not cls:
        msg = f"instruction {name!r} has no string '__cls__'"
        raise MalformedModelError(msg)
    refs: list[str] = []
    given = {
        key: _resolve(value, refs, f"{name}.{key}")
        for key, value in entry.items()
        if key not in (_CLASS_KEY, _NAME_KEY)
    }
    defaults = DEFAULTS.get(cls, {})
    options = {**given, **{k: v for k, v in defaults.items() if k not in given}}
    return InstructionSpec(
        name=name,
        cls=cls,
        options=MappingProxyType(options),
        depends_on=tuple(dict.fromkeys(refs)),
        index=index,
    )


def _resolve(value: Any, refs: list[str], where: str) -> Any:
    """Replace every reference in ``value`` by an :class:`InstructionRef`, noting its target in ``refs``."""
    if isinstance(value, Mapping):
        if _REF_KEY in value:
            target = value.get(_NAME_KEY)
            if not isinstance(target, str) or not target:
                msg = f"{where}: a reference needs the string 'name' of its target, got {dict(value)!r}"
                raise MalformedModelError(msg)
            refs.append(target)
            return InstructionRef(target)
        return {k: _resolve(v, refs, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, refs, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def _check_references(specs: tuple[InstructionSpec, ...]) -> None:
    index = {spec.name: spec.index for spec in specs}
    for spec in specs:
        for target in spec.depends_on:
            if target not in index:
                msg = f"instruction {spec.name!r} refers to {target!r}, which is not in the model"
                raise DanglingReferenceError(msg)
    _check_acyclic(specs)
    for spec in specs:
        for target in spec.depends_on:
            if index[target] > spec.index:
                msg = (
                    f"instruction {spec.name!r} (entry {spec.index}) refers to {target!r}, which comes "
                    f"later (entry {index[target]}); instructions run in file order"
                )
                raise ForwardReferenceError(msg)


def _check_acyclic(specs: tuple[InstructionSpec, ...]) -> None:
    graph = {spec.name: spec.depends_on for spec in specs}
    done: set[str] = set()
    for root in graph:
        path: list[str] = []
        _visit(root, graph, path, done)


def _visit(
    node: str, graph: Mapping[str, tuple[str, ...]], path: list[str], done: set[str]
) -> None:
    if node in done:
        return
    if node in path:
        cycle = [*path[path.index(node) :], node]
        msg = "the references form a cycle: " + " -> ".join(repr(n) for n in cycle)
        raise CyclicReferenceError(msg)
    path.append(node)
    for target in graph[node]:
        _visit(target, graph, path, done)
    path.pop()
    done.add(node)


def _check_supported(specs: tuple[InstructionSpec, ...]) -> None:
    problems = [
        f"instruction {spec.name!r}: class {spec.cls} {_why_unsupported(spec.cls)}"
        for spec in specs
        if spec.cls not in SUPPORTED_OPTIONS
    ]
    if problems:
        msg = "the model cannot be loaded by the torch backend:\n  " + "\n  ".join(
            problems
        )
        raise UnsupportedModelError(msg)


def _why_unsupported(cls: str) -> str:
    if cls in REJECTED_CLASSES:
        return "is not supported: " + REJECTED_CLASSES[cls]
    return "is unknown to the torch backend (no twin and no entry in the option table)"


def _resolve_param_dtype(metadata: Mapping[str, Any]) -> str:
    rule = MODEL_OPTIONS["param_dtype"]
    value = metadata.get("param_dtype", MODEL_DEFAULTS["param_dtype"])
    if rule.values is not None and value not in rule.values:
        msg = f"metadata param_dtype {value!r} is not supported; supported: {', '.join(map(str, rule.values))}"
        raise UnsupportedModelError(msg)
    return str(value)
