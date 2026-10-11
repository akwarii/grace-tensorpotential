"""Import gates: names that dead-code tools cannot see must keep resolving.

Dead-code removal (unused imports, unused names) is only safe when nothing resolves a name
dynamically.  Three things in this repository do:

* saved ``model.yaml`` files carry ``__cls__`` strings that ``str_to_class`` imports,
* user code, docs and notebooks do ``from tensorpotential.X import name``,
* two lazy ``__getattr__`` shims keep moved classes importable from their old paths.

Each gate below fails when one of them stops resolving.  Everything runs from any working
directory and writes nothing into the source tree.
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from pathlib import Path

import pytest

from tests.fresh_python import run_fresh_python

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "tensorpotential"

# Directories (below the repository root) scanned for imports and ``__cls__`` strings.
SCAN_DIRS = ("tensorpotential", "tests", "docs", "examples")
SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", "site"}

# Package that does not exist on this tree; its imports sit under ``try``/``except ImportError``
# or in a test that is skipped (``tests/test_structured_grid.py``).  Submodules count too.
KNOWN_ABSENT_PACKAGES = ("tensorpotential.experimental",)

# Files whose ``tensorpotential`` imports are stale on the untouched tree: the model files of
# the "skip custom model" integration test import ``tensorpotential.layers``,
# ``tensorpotential.tpinstruction`` and names that moved out of ``instructions.base``.
KNOWN_STALE_SOURCES = frozenset({
    "tests/MoNbTaW-CUSTOM/model.py",
    "tests/MoNbTaW-CUSTOM/model_mlp_emb.py",
})

# Modules that do not import on the untouched tree, mapped to the third-party module that is
# missing.  ``grace_dashboard`` needs ``flask``, which ``pyproject.toml`` does not declare.
# The gate accepts exactly this ``ModuleNotFoundError`` and nothing else for these modules.
BASELINED_IMPORT_FAILURES: dict[str, str] = {
    "tensorpotential.scripts.grace_dashboard": "flask",
}

IMPORT_TIMEOUT_S = 300

_CLS_RE = re.compile(r"""__cls__["']?\s*:\s*["']?([A-Za-z_][\w]*(?:\.[\w]+)+)""")
_MD_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_MD_IMPORT_RE = re.compile(
    r"^\s*from\s+(tensorpotential[\w.]*)\s+import\s+([\w, ]+?)\s*(?:#.*)?$"
)


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def _scan_files(*suffixes: str) -> list[Path]:
    """Files below ``SCAN_DIRS`` and the top-level files with one of ``suffixes``."""
    files = [p for p in REPO_ROOT.iterdir() if p.is_file() and p.suffix in suffixes]
    for name in SCAN_DIRS:
        root = REPO_ROOT / name
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
            files.extend(
                Path(dirpath) / f
                for f in sorted(filenames)
                if Path(f).suffix in suffixes
            )
    return sorted(files)


def package_modules() -> list[str]:
    """Dotted names of every module and package of ``tensorpotential``, in sorted order."""
    names = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if SKIP_DIR_NAMES & set(path.parts):
            continue
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return sorted(names)


def _imports_of_tree(tree: ast.AST) -> set[tuple[str, str]]:
    """``(module, name)`` of each absolute ``from tensorpotential... import name`` (no ``*``)."""
    pairs = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and (
                node.module == "tensorpotential"
                or node.module.startswith("tensorpotential.")
            )
        ):
            pairs.update((node.module, a.name) for a in node.names if a.name != "*")
    return pairs


def _imports_of_python(path: Path) -> set[tuple[str, str]]:
    return _imports_of_tree(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    )


def _imports_of_block(block: str) -> set[tuple[str, str]]:
    """Imports of a code block; a block that does not parse is read line by line."""
    try:
        return _imports_of_tree(ast.parse(block))
    except SyntaxError:
        matches = (_MD_IMPORT_RE.match(line) for line in block.splitlines())
        return {
            (m[1], n.strip())
            for m in matches
            if m
            for n in m[2].split(",")
            if n.strip()
        }


def _imports_of_markdown(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for block in _MD_BLOCK_RE.findall(path.read_text(encoding="utf-8")):
        pairs |= _imports_of_block(block)
    return pairs


def _imports_of_notebook(path: Path) -> set[tuple[str, str]]:
    pairs = set()
    for cell in json.loads(path.read_text(encoding="utf-8")).get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = (
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        # IPython magics (``%matplotlib``, ``!pip``) are not Python.
        source = "\n".join(
            ln if not ln.lstrip().startswith(("%", "!")) else ""
            for ln in source.splitlines()
        )
        try:
            pairs |= _imports_of_tree(ast.parse(source))
        except SyntaxError:
            continue
    return pairs


def collected_imports() -> dict[tuple[str, str], list[str]]:
    """Every ``(module, name)`` imported in code, tests, docs and notebooks, with its sources."""
    readers = {
        ".py": _imports_of_python,
        ".md": _imports_of_markdown,
        ".ipynb": _imports_of_notebook,
    }
    found: dict[tuple[str, str], list[str]] = {}
    for path in _scan_files(*readers):
        for pair in readers[path.suffix](path):
            found.setdefault(pair, []).append(path.relative_to(REPO_ROOT).as_posix())
    return found


def checked_imports() -> dict[tuple[str, str], list[str]]:
    """``collected_imports`` without the absent package, the baselined modules and stale files."""
    checked = {}
    for (module_name, name), sources in collected_imports().items():
        live = [s for s in sources if s not in KNOWN_STALE_SOURCES]
        if (
            live
            and not _is_absent(module_name)
            and module_name not in BASELINED_IMPORT_FAILURES
        ):
            checked[module_name, name] = live
    return checked


def _is_absent(module_name: str) -> bool:
    return any(
        module_name == p or module_name.startswith(p + ".")
        for p in KNOWN_ABSENT_PACKAGES
    )


def cls_strings() -> dict[str, list[str]]:
    """Every dotted ``__cls__`` value in yaml, markdown, python and notebook files."""
    found: dict[str, list[str]] = {}
    for path in _scan_files(".yaml", ".yml", ".md", ".py", ".ipynb"):
        if path == Path(__file__).resolve():  # its own test strings are not models
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in _CLS_RE.findall(text):
            found.setdefault(value, []).append(path.relative_to(REPO_ROOT).as_posix())
    return found


# --------------------------------------------------------------------------------------
# Subprocess helper
# --------------------------------------------------------------------------------------
# One interpreter imports ``tensorpotential`` and ``tensorpotential._tf_options`` (which loads and
# configures TensorFlow, costing seconds: the package ``__init__`` itself no longer does), then
# forks one child per module.  A child starts from the state a TensorFlow-side import finds and
# imports nothing else beforehand, so the modules are still checked independently of each other.
_FORK_IMPORT_CODE = """
import importlib, json, os, sys, traceback

root = sys.argv[1]
names = json.load(sys.stdin)
try:
    import tensorpotential
    import tensorpotential._tf_options
except BaseException:
    failure = {"returncode": 1, "stderr": traceback.format_exc()[-2000:]}
    print(json.dumps({name: failure for name in names}))
    sys.exit(0)
results = {}
for name in names:
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        code = 0
        try:
            module = importlib.import_module(name)
            assert module.__file__.startswith(root), module.__file__
        except BaseException:
            code = 1
            os.write(write_end, traceback.format_exc()[-2000:].encode())
        os._exit(code)
    os.close(write_end)
    chunks = []
    while True:
        chunk = os.read(read_end, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_end)
    _, status = os.waitpid(pid, 0)
    results[name] = {
        "returncode": os.waitstatus_to_exitcode(status),
        "stderr": b"".join(chunks).decode(errors="replace"),
    }
print(json.dumps(results))
"""

_RESOLVE_CODE = """
import importlib, json, sys
unresolved = []
for module_name, name in json.load(sys.stdin):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        unresolved.append([module_name, name, "module: " + type(exc).__name__ + ": " + str(exc)])
        continue
    if hasattr(module, name):
        continue
    try:
        importlib.import_module(module_name + "." + name)
    except Exception as exc:
        unresolved.append([module_name, name, "name: " + type(exc).__name__ + ": " + str(exc)])
print(json.dumps(unresolved))
"""


def _unresolved(pairs: list[tuple[str, str]], cwd: Path) -> list[list[str]]:
    """The ``[module, name, reason]`` of each pair that does not resolve, in one interpreter."""
    result = run_fresh_python(_RESOLVE_CODE, cwd, stdin=json.dumps(pairs))
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------------------
# Gate 1: every module imports in its own interpreter
# --------------------------------------------------------------------------------------
def _imports_in_forks(names: list[str], root: Path, cwd: Path) -> dict[str, dict]:
    """``{name: {"returncode", "stderr"}}`` of importing each of ``names`` from below ``root``."""
    result = run_fresh_python(
        _FORK_IMPORT_CODE, cwd, args=(str(root),), stdin=json.dumps(names)
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def import_results(tmp_path_factory):
    """The outcome of importing every module of the package, computed once per module."""
    return _imports_in_forks(
        package_modules(), REPO_ROOT, tmp_path_factory.mktemp("import_gate")
    )


@pytest.mark.parametrize("module_name", package_modules())
def test_module_imports_in_fresh_subprocess(module_name, import_results):
    returncode = import_results[module_name]["returncode"]
    stderr = import_results[module_name]["stderr"]
    missing = BASELINED_IMPORT_FAILURES.get(module_name)
    if returncode != 0 and missing is not None:
        assert f"No module named '{missing}'" in stderr, stderr[-2000:]
        return
    assert returncode == 0, f"import {module_name} failed:\n{stderr[-2000:]}"


def test_fork_import_reports_missing_modules_and_foreign_files(tmp_path):
    results = _imports_in_forks(
        ["tensorpotential.utils", "tensorpotential.no_such_module"], REPO_ROOT, tmp_path
    )
    assert results["tensorpotential.utils"]["returncode"] == 0
    assert results["tensorpotential.no_such_module"]["returncode"] != 0
    assert "ModuleNotFoundError" in results["tensorpotential.no_such_module"]["stderr"]
    # a module found outside the expected root is a failure (the assert in the child)
    elsewhere = _imports_in_forks(["tensorpotential.utils"], tmp_path, tmp_path)
    assert elsewhere["tensorpotential.utils"]["returncode"] != 0
    assert "AssertionError" in elsewhere["tensorpotential.utils"]["stderr"]


# --------------------------------------------------------------------------------------
# Gate 2: ``from tensorpotential.X import name`` in code, tests, docs and notebooks
# --------------------------------------------------------------------------------------
def test_imports_of_tensorpotential_names_resolve(tmp_path):
    found = checked_imports()
    pairs = sorted(found)
    assert len(pairs) > 100, (
        "the scan found suspiciously few imports"
    )  # guards a broken scan
    broken = [
        f"from {m} import {n}  ({r}; used in {', '.join(sorted(found[m, n])[:3])})"
        for m, n, r in _unresolved(pairs, tmp_path)
    ]
    assert not broken, "unresolved imports:\n" + "\n".join(broken)


# --------------------------------------------------------------------------------------
# Gate 3: ``__cls__`` strings
# --------------------------------------------------------------------------------------
def test_cls_strings_resolve(tmp_path):
    found = cls_strings()
    assert any(p.endswith(".yaml") for paths in found.values() for p in paths)
    pairs = sorted(tuple(value.rsplit(".", 1)) for value in found)
    pairs = [(m, n) for m, n in pairs if not _is_absent(m)]
    broken = [
        f"{m}.{n}  ({r}; in {', '.join(sorted(found[f'{m}.{n}'])[:3])})"
        for m, n, r in _unresolved(pairs, tmp_path)
    ]
    assert not broken, "unresolved __cls__ strings:\n" + "\n".join(broken)


# --------------------------------------------------------------------------------------
# Gate 4: the two lazy back-compat shims
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["RandomProjectedBasisFeatures", "generate_rp_matrix"])
def test_tpmodel_getattr_shim_resolves_moved_uq_symbols(name):
    from tensorpotential import tpmodel
    from tensorpotential.instructions.base import str_to_class
    from tensorpotential.uq import instructions as uq_instructions

    assert name not in vars(tpmodel), (
        "the symbol is defined in tpmodel again; the shim is dead"
    )
    assert getattr(tpmodel, name) is getattr(uq_instructions, name)
    # The route a saved ``model.yaml`` takes.
    assert str_to_class(f"tensorpotential.tpmodel.{name}") is getattr(
        uq_instructions, name
    )


def test_tpmodel_getattr_shim_rejects_unknown_names():
    from tensorpotential import tpmodel

    with pytest.raises(AttributeError, match="no attribute 'NoSuchSymbol'"):
        tpmodel.NoSuchSymbol  # noqa: B018 - attribute access is the behaviour under test


def test_instructions_getattr_shim_resolves_read_model_metadata_with_warning():
    import tensorpotential.instructions as instructions
    from tensorpotential import metadata_utils

    assert "read_model_metadata" not in vars(instructions)
    with pytest.warns(DeprecationWarning, match="metadata_utils.read_model_metadata"):
        func = instructions.read_model_metadata
    assert func is metadata_utils.read_model_metadata


def test_instructions_getattr_shim_rejects_unknown_names():
    import tensorpotential.instructions as instructions

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an unknown name must not even warn
        with pytest.raises(AttributeError, match="no attribute 'NoSuchSymbol'"):
            instructions.NoSuchSymbol  # noqa: B018 - attribute access is the behaviour under test


# --------------------------------------------------------------------------------------
# The scanners themselves
# --------------------------------------------------------------------------------------
def test_python_scanner_keeps_absolute_tensorpotential_imports_only():
    tree = ast.parse(
        "from tensorpotential.utils import Parity, capture_init_args\n"
        "from tensorpotential import constants\n"
        "from tensorpotential.utils import *\n"
        "from . import sibling\n"
        "from numpy import array\n"
        "from tensorpotential_other import x\n"
    )
    assert _imports_of_tree(tree) == {
        ("tensorpotential.utils", "Parity"),
        ("tensorpotential.utils", "capture_init_args"),
        ("tensorpotential", "constants"),
    }


def test_markdown_scanner_reads_unparseable_blocks_line_by_line():
    block = (
        "from tensorpotential.calculator import TPCalculator  # comment\n"
        "from tensorpotential.tpmodel import ExtractBasisFunctions, TPModel\n"
        "calc = TPCalculator(model, ...)  $ not python\n"
    )
    assert _imports_of_block(block) == {
        ("tensorpotential.calculator", "TPCalculator"),
        ("tensorpotential.tpmodel", "ExtractBasisFunctions"),
        ("tensorpotential.tpmodel", "TPModel"),
    }


def test_notebook_scanner_skips_magics_and_markdown_cells(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["from tensorpotential.x import y"]},
            {
                "cell_type": "code",
                "source": [
                    "%matplotlib inline\n",
                    "from tensorpotential.calculator import TPCalculator",
                ],
            },
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert _imports_of_notebook(path) == {
        ("tensorpotential.calculator", "TPCalculator")
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '"__cls__": "tensorpotential.instructions.compute.RadialBasis"',
            ["tensorpotential.instructions.compute.RadialBasis"],
        ),
        ("  - __cls__: tensorpotential.tpmodel.X\n", ["tensorpotential.tpmodel.X"]),
        ("{__cls__: a.b.C, name: d}", ["a.b.C"]),
        ("__cls__ is a key", []),
        ('"__cls__": "NoModule"', []),
    ],
)
def test_cls_regex(text, expected):
    assert _CLS_RE.findall(text) == expected


def test_is_absent_matches_package_and_submodules_only():
    assert _is_absent("tensorpotential.experimental")
    assert _is_absent("tensorpotential.experimental.mag.databuilder")
    assert not _is_absent("tensorpotential.experimentalish")
    assert not _is_absent("tensorpotential.utils")


# --------------------------------------------------------------------------------------
# Gate 4: the TensorFlow-free entry points (D1: lazy imports, TF-free shared code in core/)
# --------------------------------------------------------------------------------------
# Modules and names that must work in an interpreter where ``import tensorflow`` and
# ``import keras`` raise ImportError.  Add the entry points of the TF-free code here.
TF_FREE_MODULES = (
    "tensorpotential",
    "tensorpotential.core",
    "tensorpotential.core.cutoffs",
    "tensorpotential.core.lazy",
    "tensorpotential.torch_backend",
    "tensorpotential.torch_backend.spec",
    "tensorpotential.torch_backend.spec.errors",
    "tensorpotential.torch_backend.spec.loader",
    "tensorpotential.torch_backend.spec.options",
    "tensorpotential.constants",
    "tensorpotential.poly",
    "tensorpotential.functions.couplings",
    "tensorpotential.calculator",
    "tensorpotential.calculator.foundation_models",
)

_TF_BLOCKED_CODE = """
import importlib, json, sys, traceback

class RefuseTensorFlow:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("tensorflow", "keras"):
            raise ImportError(f"No module named {name!r} (refused by the test)", name=name)

sys.meta_path.insert(0, RefuseTensorFlow())
failures = {}
for name in json.load(sys.stdin):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith(sys.argv[1])]
        where = [f"{f.filename[len(sys.argv[1]):]}:{f.lineno}" for f in frames]
        failures[name] = {"error": str(exc), "where": where}
grace_fm = None
tf_error = ""
try:
    from tensorpotential.calculator import grace_fm
    import tensorpotential.calculator
    tensorpotential.calculator.TPCalculator
except ImportError as exc:
    tf_error = str(exc)
print(json.dumps({"failures": failures, "loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("tensorflow", "keras")), "tf_error": tf_error, "grace_fm": callable(grace_fm)}))
"""


@pytest.fixture(scope="module")
def tf_blocked_run(tmp_path_factory):
    result = run_fresh_python(
        _TF_BLOCKED_CODE,
        tmp_path_factory.mktemp("tf_blocked"),
        args=(str(REPO_ROOT) + os.sep,),
        stdin=json.dumps(list(TF_FREE_MODULES)),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_tf_free_entry_points_import_without_tensorflow(tf_blocked_run):
    """Each failure names the offending file and line, the first of the chain that reaches TensorFlow."""
    assert tf_blocked_run["failures"] == {}, "\n".join(
        f"{name}: {info['error']} via {' <- '.join(reversed(info['where']))}"
        for name, info in tf_blocked_run["failures"].items()
    )


def test_tf_free_entry_points_never_load_tensorflow(tf_blocked_run):
    assert tf_blocked_run["loaded"] == []


def test_tf_free_names_work_and_tf_names_explain_what_is_missing(tf_blocked_run):
    assert tf_blocked_run["grace_fm"] is True
    assert "'TPCalculator' needs TensorFlow" in tf_blocked_run["tf_error"]
    assert "tensorpotential[tf]" in tf_blocked_run["tf_error"]


def test_the_import_contract_is_kept():
    """``lint-imports`` reads ``[tool.importlinter]`` of pyproject.toml: no path from the TF-free modules to TensorFlow."""
    result = run_fresh_python(
        "from importlinter.cli import lint_imports_command as main; main()",
        REPO_ROOT,
        args=("--no-cache",),
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-1000:]
    assert "1 kept, 0 broken" in result.stdout


def test_the_contract_lists_torch_backend_once_it_exists():
    config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if (PACKAGE_DIR / "torch_backend").is_dir():
        assert '"tensorpotential.torch_backend"' in config
