"""A TensorFlow stand-in that constructs the TF instruction classes of this tree without TensorFlow (Appendix E).

The torch twins own a copy of the index-table ("plan") logic of the TF instructions (decision D7). Two oracles guard
that copy: the tables dumped by TensorFlow into the golden fixtures (FIX2), and the tables built here by the
**unmodified** ``__init__`` (and, on request, ``build``) of the TF classes, imported from this tree at each run, so the
second oracle follows any change of the TF source and needs no TensorFlow (CI's ``test-torch`` job has none).

How: inside :func:`stand_in` a meta-path finder answers every import of ``tensorflow``, ``keras`` and ``tf_keras`` with a
fake module. The few calls that make tables are real and numpy-backed (``constant``, ``convert_to_tensor``, ``cast``,
``reshape``, ``concat``, ``stack``, ``range``, ``zeros``, ``ones``, ``one_hot``, ``sqrt``, the dtypes, ``Module``, ``Variable``, ``random.normal`` and
``random.uniform``); every other name is a permissive :class:`Placeholder`, which ``frwrd`` would need and table code
never does. A tensor is a numpy array; a variable is a :class:`Variable` whose values are zeros (only its shape, dtype,
name and trainability are meaningful). On exit the fake modules and every ``tensorpotential`` module imported inside the
context are removed again, so the stand-in leaves nothing behind for code that later imports the real TensorFlow.

Typical use::

    model = load_model("tests_torch/golden/yamls/omat_tiny.yaml", float_dtype="float64")
    arrays(model["YI"])            # {"YI/lr_inds": int32 array, "YI/cg": ..., ...}, the keys of the fixtures' tables/
    variables(model["I0"])         # {"I0/reducing_A": VariableInfo(shape=(4, 4, 4, 2), ...), ...}
    with stand_in():               # one instruction from its constructor arguments
        y = construct("SphericalHarmonic", vhat="ScaledBondVector", name="Y", lmax=2)

Limits: ``frwrd`` cannot run (its operations are placeholders); :func:`placeholders` lists the attributes an
instruction got from a placeholder, so a test can assert that the tables it compares are real. A process that has
already imported the real TensorFlow cannot use the stand-in (:class:`StandInError`).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

ROOTS: Final = ("tensorflow", "keras", "tf_keras")
"""Top-level packages the stand-in answers."""

_SKIPPED: Final = ("_init_args", "_self_")
"""Attribute prefixes left out of a walk, as in the probe of the golden generator (``tools/grace_probe.py``)."""

_DTYPES: Final[Mapping[str, np.dtype]] = {
    name: np.dtype(name)
    for name in (
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "bool",
    )
}


class StandInError(RuntimeError):
    """The stand-in cannot run here (TensorFlow already imported) or was used outside :func:`stand_in`."""


class _PlaceholderMeta(type):
    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return cls

    def __getitem__(cls, item: object) -> Any:
        return cls


class Placeholder(metaclass=_PlaceholderMeta):
    """Any TensorFlow name the stand-in does not implement: callable, subscriptable, every attribute is another one."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return Placeholder()

    def __call__(self, *args: object, **kwargs: object) -> Any:
        del args, kwargs
        return Placeholder()

    def __getitem__(self, item: object) -> Any:
        return Placeholder()

    def __iter__(self) -> Iterator[Any]:
        return iter(())


def _dtype(dtype: object) -> np.dtype | None:
    return None if dtype is None else np.dtype(dtype)  # ty: ignore[no-matching-overload]


def constant(
    value: object, dtype: object = None, shape: object = None, name: object = None
) -> np.ndarray:
    """``tf.constant``: a numpy array; Python floats become float32 and Python ints int32, as in TensorFlow."""
    del name
    out = np.asarray(value)
    target = _dtype(dtype)
    if target is None and not isinstance(value, np.ndarray | np.generic):
        if out.dtype.kind == "f":
            target = np.dtype("float32")
        elif out.dtype.kind in "iu":
            target = np.dtype("int32")
    if target is not None:
        out = out.astype(target)
    if shape is not None:
        out = np.broadcast_to(out, shape).copy()  # ty: ignore[no-matching-overload]
    return out


def cast(value: object, dtype: object, name: object = None) -> np.ndarray:
    """``tf.cast``."""
    del name
    return np.asarray(value).astype(_dtype(dtype))


def reshape(value: object, shape: object, name: object = None) -> np.ndarray:
    """``tf.reshape``."""
    del name
    return np.reshape(np.asarray(value), shape)  # ty: ignore[no-matching-overload]


def zeros(shape: object, dtype: object = "float32", name: object = None) -> np.ndarray:
    """``tf.zeros``."""
    del name
    return np.zeros(shape, dtype=_dtype(dtype))  # ty: ignore[no-matching-overload]


def ones(shape: object, dtype: object = "float32", name: object = None) -> np.ndarray:
    """``tf.ones``."""
    del name
    return np.ones(shape, dtype=_dtype(dtype))  # ty: ignore[no-matching-overload]


def concat(values: Iterable[object], axis: int, name: object = None) -> np.ndarray:
    """``tf.concat``."""
    del name
    return np.concatenate([np.asarray(v) for v in values], axis=axis)


def stack(values: Iterable[object], axis: int = 0, name: object = None) -> np.ndarray:
    """``tf.stack``."""
    del name
    return np.stack([np.asarray(v) for v in values], axis=axis)


def range_(
    start: int,
    limit: int | None = None,
    delta: int = 1,
    dtype: object = None,
    name: object = None,
) -> np.ndarray:
    """``tf.range``; int32 unless ``dtype`` says otherwise, as in TensorFlow for integer bounds."""
    del name
    bounds = (0, start) if limit is None else (start, limit)
    return np.arange(*bounds, delta, dtype=_dtype(dtype) or np.dtype("int32"))


def sqrt(value: object, name: object = None) -> np.ndarray:
    """``tf.sqrt``."""
    del name
    return np.asarray(np.sqrt(np.asarray(value)))


def one_hot(
    indices: object, depth: int, dtype: object = "float32", **kwargs: object
) -> np.ndarray:
    """``tf.one_hot`` with the default on and off values."""
    del kwargs
    return np.eye(depth, dtype=_dtype(dtype))[np.asarray(indices)]


def _random(
    shape: object, *args: object, dtype: object = "float32", **kwargs: object
) -> np.ndarray:
    """``tf.random.normal`` and ``tf.random.uniform``: zeros of the shape and dtype (the values are never compared)."""
    del args, kwargs
    return zeros(shape, dtype)


class Variable:
    """``tf.Variable``: the value (as given, or zeros from :func:`_random`), its dtype, name and trainability."""

    def __init__(
        self,
        initial_value: object,
        trainable: bool = True,
        name: str | None = None,
        dtype: object = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        value = initial_value() if callable(initial_value) else initial_value
        self.value = constant(value, dtype)
        self.trainable = trainable
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the value."""
        return self.value.shape

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the value."""
        return self.value.dtype

    def numpy(self) -> np.ndarray:
        """The value."""
        return self.value


class Module:
    """``tf.Module``: a name, the name-scope decorator as the identity."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    @staticmethod
    def with_name_scope(method: Callable[..., Any]) -> Callable[..., Any]:
        """The identity (no name scopes without TensorFlow)."""
        return method

    @property
    def trainable_variables(self) -> tuple[Variable, ...]:
        """The trainable :class:`Variable` attributes of this module and the modules nested in it."""
        return tuple(v for v in self.variables if v.trainable)

    @property
    def submodules(self) -> tuple[Module, ...]:
        """Every module nested in this one, at any depth, each once (as ``tf.Module.submodules``)."""
        found: dict[int, Module] = {}
        pending: list[object] = [self]
        while pending:
            for _, value in _children(pending.pop(0)):
                if (
                    isinstance(value, Module)
                    and id(value) not in found
                    and value is not self
                ):
                    found[id(value)] = value
                    pending.append(value)
        return tuple(found.values())

    @property
    def variables(self) -> tuple[Variable, ...]:
        """The :class:`Variable` attributes of this module and the modules nested in it."""
        return tuple(v for _, v in _variable_paths(self, ""))


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return Placeholder


_TENSORFLOW: Final[Mapping[str, object]] = {
    **_DTYPES,
    "Module": Module,
    "Variable": Variable,
    "constant": constant,
    "convert_to_tensor": constant,
    "cast": cast,
    "reshape": reshape,
    "zeros": zeros,
    "ones": ones,
    "one_hot": one_hot,
    "concat": concat,
    "stack": stack,
    "range": range_,
    "sqrt": sqrt,
    "newaxis": None,
    "random": types.SimpleNamespace(normal=_random, uniform=_random),
}
"""The names of the fake ``tensorflow`` that are real (numpy-backed); every other name is a :class:`Placeholder`."""


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(
        self, fullname: str, path: object, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname.split(".")[0] in ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        module = _StubModule(spec.name)
        module.__path__ = []
        if spec.name == "tensorflow":
            for name, value in _TENSORFLOW.items():
                setattr(module, name, value)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        del module


def _real_tensorflow() -> list[str]:
    return sorted(n for n in sys.modules if n.split(".")[0] in ROOTS)


def active() -> bool:
    """Whether a :func:`stand_in` context is open."""
    return any(isinstance(f, _Finder) for f in sys.meta_path)


@contextlib.contextmanager
def stand_in() -> Iterator[types.ModuleType]:
    """Answer every import of TensorFlow with the stand-in while the context is open; yield the fake ``tensorflow``.

    Re-entrant: an inner context reuses the outer one. On exit of the outermost context the finder is removed and so is
    every ``tensorflow``, ``keras`` and ``tensorpotential.*`` module imported inside it (its binding in the parent package
    too); third-party modules imported meanwhile stay.

    Raises
    ------
    StandInError
        When the real TensorFlow (or Keras) is already imported in this process.
    """
    if active():
        yield importlib.import_module("tensorflow")
        return
    loaded = _real_tensorflow()
    if loaded:
        msg = (
            f"the TensorFlow stand-in needs a process without TensorFlow, but {loaded[:3]} "
            "are imported; run the stand-in tests in a process of their own (pytest tests_torch)"
        )
        raise StandInError(msg)
    before = set(sys.modules)
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module("tensorflow")
    finally:
        sys.meta_path.remove(finder)
        _forget(set(sys.modules) - before)


def _forget(names: Iterable[str]) -> None:
    # Only the stand-in and the library modules built on it: a third-party extension module (numpy.fft, scipy) cannot
    # be imported twice in one process, and it never holds a reference to the stand-in.
    ours = (
        n for n in names if n.split(".")[0] in ROOTS or n.startswith("tensorpotential.")
    )
    for name in sorted(ours, reverse=True):
        module = sys.modules.pop(name)
        parent, _, child = name.rpartition(".")
        if (
            parent in sys.modules
            and getattr(sys.modules[parent], child, None) is module
        ):
            delattr(sys.modules[parent], child)


def _require_active() -> None:
    if not active():
        msg = "open the stand-in first: with tfstub.stand_in(): ..."
        raise StandInError(msg)


def construct(class_name: str, /, **init_args: object) -> Any:
    """Construct the TF instruction ``class_name`` of ``tensorpotential.instructions`` from its constructor arguments.

    Arguments that are instructions are passed as the objects (from an earlier :func:`construct` or :func:`load_model`).
    Must be called inside :func:`stand_in`.
    """
    _require_active()
    instructions = importlib.import_module("tensorpotential.instructions")
    return getattr(instructions, class_name)(**init_args)


def load_model(path: str | Path, *, float_dtype: str | None = None) -> dict[str, Any]:
    """Construct every instruction of a ``model.yaml`` with the TF code, in file order, keyed by instruction name.

    With ``float_dtype`` (``"float64"``, ``"float32"``) every instruction is also built, as ``TPModel.build`` does,
    which creates its variables and the tables made in ``build``. Opens :func:`stand_in` when none is open.
    """
    with stand_in():
        base = importlib.import_module("tensorpotential.instructions.base")
        model = base.load_instructions(str(path))
        model = model if isinstance(model, dict) else {i.name: i for i in model}
        if float_dtype is not None:
            # numpy warns where TensorFlow silently makes inf or nan (SphericalHarmonics.pre_compute fills the
            # unused entries of its alm, blm tables that way), so the warnings of build are numpy's, not the library's
            with np.errstate(divide="ignore", invalid="ignore"):
                for instruction in model.values():
                    instruction.build(np.dtype(float_dtype))
        return model


def _is_instruction(value: object) -> bool:
    return any(c.__name__ == "TPInstruction" for c in type(value).__mro__)


def _children(obj: object) -> Iterator[tuple[str, Any]]:
    for name, value in vars(obj).items():
        if name.startswith(_SKIPPED):
            continue
        if isinstance(value, dict):
            yield from ((f"{name}[{k}]", v) for k, v in value.items())
        elif isinstance(value, list | tuple):
            yield from ((f"{name}[{i}]", v) for i, v in enumerate(value))
        else:
            yield name, value


def _walk(
    obj: object, path: str, kind: type | tuple[type, ...], seen: set[int]
) -> Iterator[tuple[str, Any]]:
    seen.add(id(obj))
    for name, value in _children(obj):
        key = f"{path}/{name}" if path else name
        if isinstance(value, kind):
            yield key, value
        elif (
            isinstance(value, Module)
            and not _is_instruction(value)
            and id(value) not in seen
        ):
            yield from _walk(value, key, kind, seen)


def arrays(instruction: Any) -> dict[str, np.ndarray]:
    """The arrays an instruction owns, keyed ``<instruction>/<attribute path>`` like the ``tables/`` of a fixture.

    The walk is that of the golden generator (``tools/make_golden.py``, ``_own_tables``): attributes and the items
    of dict and list attributes (``name[key]``), descending into nested modules but not into the other instructions
    an instruction refers to. Variables are left out (:func:`variables`). A numpy scalar is not a table (the TF code
    keeps some, ``lmax``, ``w_shape_right``, as numpy scalars), so an attribute that TensorFlow makes by arithmetic on
    a 0-d tensor and numpy turns into a scalar would be missed: none of the tables of the fixtures is such a case.
    """
    return dict(_walk(instruction, instruction.name, np.ndarray, set()))


@dataclass(frozen=True)
class VariableInfo:
    """Shape, dtype and trainability of a variable, the fields of the ``variables`` list of a fixture manifest."""

    shape: tuple[int, ...]
    dtype: str
    trainable: bool


def variables(instruction: Any) -> dict[str, VariableInfo]:
    """The variables of a built instruction, keyed ``<instruction>/<attribute path>`` like a fixture's weights.

    The key is the checkpoint path (``tools/grace_probe.py``, ``checkpoint_keys``): the attributes are walked breadth
    first, list and dict items as ``name/<index or key>``, so a variable reachable by several paths gets the shortest
    one (``RadialBasis/rc``, not ``RadialBasis/basis_function/rc``). Text variables have the dtype ``string``.
    """
    return {
        key: VariableInfo(tuple(v.shape), _dtype_name(v.dtype), bool(v.trainable))
        for key, v in _variable_paths(instruction, instruction.name)
    }


def _variable_paths(root: object, prefix: str) -> list[tuple[str, Variable]]:
    keys: dict[int, tuple[str, Variable]] = {}
    seen = {id(root)}
    queue: deque[tuple[object, str]] = deque([(root, prefix)])
    while queue:
        node, path = queue.popleft()
        for name, child in _tracked(node):
            key = f"{path}/{name}" if path else name
            if isinstance(child, Variable):
                keys.setdefault(id(child), (key, child))
            elif _is_container(child) and id(child) not in seen:
                seen.add(id(child))
                queue.append((child, key))
    return list(keys.values())


def _tracked(node: object) -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        yield from ((str(k), v) for k, v in node.items())
    elif isinstance(node, list | tuple):
        yield from ((str(i), v) for i, v in enumerate(node))
    else:
        yield from ((n, v) for n, v in vars(node).items() if not n.startswith(_SKIPPED))


def _is_container(value: object) -> bool:
    if isinstance(value, Module):
        return not _is_instruction(value)
    return isinstance(value, dict | list | tuple)


def _dtype_name(dtype: np.dtype) -> str:
    return "string" if dtype.kind in "USO" else dtype.name


def placeholders(instruction: Any) -> list[str]:
    """The attribute paths of an instruction whose value is a :class:`Placeholder` (not a real table)."""
    return sorted(
        key
        for key, _ in _walk(
            instruction, instruction.name, (Placeholder, _PlaceholderMeta), set()
        )
    )
