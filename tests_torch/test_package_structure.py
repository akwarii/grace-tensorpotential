"""Structure rules of ``tensorpotential.torch_backend`` that hold for every module, present and future.

Each module is found by walking the package, so a new module is checked without being listed. Imports: every module
imports in a fresh interpreter that refuses TensorFlow (R2), and the torch-free layers (the errors, the spec, the
plans, the weights, the protocol) also import when torch is refused; the import-linter contracts check the same
statically, this checks what really runs. Errors: every exception type of the backend is defined in its one
``errors`` module and derives from its base (R5). Immutability: every dataclass of the backend is frozen.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil
from types import ModuleType

import pytest

import tensorpotential.torch_backend as backend
from tensorpotential.torch_backend import errors
from tests.fresh_python import run_fresh_python

MODULES = tuple(
    sorted(
        [backend.__name__]
        + [
            m.name
            for m in pkgutil.walk_packages(backend.__path__, f"{backend.__name__}.")
        ]
    )
)
TORCH_FREE_PREFIXES = tuple(
    f"{backend.__name__}.{name}"
    for name in ("errors", "spec", "plans", "weights", "protocol")
)
TORCH_FREE = tuple(
    m for m in MODULES if m == backend.__name__ or m.startswith(TORCH_FREE_PREFIXES)
)

_IMPORT_REFUSING = """
import importlib, json, sys

refused = set(sys.argv[1].split(","))

class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in refused:
            raise ImportError(f"No module named {name!r} (refused by the test)", name=name)

sys.meta_path.insert(0, Refuse())
failed = {}
for module in json.loads(sys.stdin.read()):
    try:
        importlib.import_module(module)
    except ImportError as error:
        failed[module] = str(error)
print(json.dumps(failed))
"""


def _modules() -> list[ModuleType]:
    return [importlib.import_module(name) for name in MODULES]


def _defined_here(module: ModuleType) -> list[type]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    ]


def _import_failures(modules: tuple[str, ...], refused: str, cwd) -> dict[str, str]:
    done = run_fresh_python(
        _IMPORT_REFUSING, cwd, args=[refused], stdin=json.dumps(modules)
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_walk_finds_the_skeleton():
    layers = (
        "spec",
        "plans",
        "weights",
        "ops",
        "instructions",
        "model",
        "convert",
        "torchsim",
    )
    expected = {
        f"{backend.__name__}.{name}" for name in (*layers, "errors", "protocol")
    }
    assert expected <= set(MODULES)
    assert f"{backend.__name__}.spec.loader" in TORCH_FREE
    assert f"{backend.__name__}.ops" not in TORCH_FREE


def test_every_module_imports_without_tensorflow(tmp_path):
    assert _import_failures(MODULES, "tensorflow,keras", tmp_path) == {}


def test_the_torch_free_layers_import_without_torch(tmp_path):
    refused = "tensorflow,keras,torch,torch_sim"
    assert _import_failures(TORCH_FREE, refused, tmp_path) == {}


def test_the_refusing_interpreter_reports_a_refused_import(tmp_path):
    modules = ("tomllib", "tensorpotential.torch_backend.ops")
    failed = _import_failures(modules, "tomllib", tmp_path)
    assert list(failed) == ["tomllib"]
    assert "refused by the test" in failed["tomllib"]


def _violations(module: ModuleType) -> list[str]:
    found = []
    for cls in _defined_here(module):
        where = f"{module.__name__}.{cls.__name__}"
        if issubclass(cls, BaseException) and module is not errors:
            found.append(f"{where} is an exception outside {errors.__name__}")
        if issubclass(cls, BaseException) and not issubclass(
            cls, errors.TorchBackendError
        ):
            found.append(f"{where} does not derive from TorchBackendError")
        if dataclasses.is_dataclass(cls) and not cls.__dataclass_params__.frozen:  # ty: ignore[unresolved-attribute]
            found.append(f"{where} is a dataclass that is not frozen")
    return found


@pytest.mark.parametrize("module", _modules(), ids=lambda m: m.__name__)
def test_errors_live_in_one_module_and_dataclasses_are_frozen(module):
    assert _violations(module) == []


def test_the_structure_check_sees_a_violation():
    planted = ModuleType("planted")

    @dataclasses.dataclass
    class Loose:
        value: int

    class StrayError(ValueError):
        pass

    Loose.__module__ = StrayError.__module__ = planted.__name__
    planted.Loose, planted.StrayError = Loose, StrayError  # ty: ignore[unresolved-attribute]
    assert _violations(planted) == [
        "planted.Loose is a dataclass that is not frozen",
        "planted.StrayError is an exception outside tensorpotential.torch_backend.errors",
        "planted.StrayError does not derive from TorchBackendError",
    ]
