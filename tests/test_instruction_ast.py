"""Tests of ``tools/instruction_ast.py``: the TF-free extractor of instruction classes and constructor defaults.

Two layers:

- logic: how classes are found (re-exported and aliased bases, relative imports, the class decorator), which
  constructor applies, the parameter kinds and the non-literal defaults, the packages that are skipped, and the
  two commands; each on a small source tree written to ``tmp_path``;
- physical values: the table read from the real source equals ``tests/data/instruction_constructor_defaults.json``,
  which ``tests/instruction_signatures.py`` writes from the classes *imported* with TensorFlow (``inspect``), so
  two independent readings of the same classes agree on all 39 of them, defaults included.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import instruction_ast as ia  # noqa: E402

BASE = (
    "from tensorpotential.instructions.base import TPInstruction, capture_init_args\n"
)


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``files`` (path relative to ``root``) and make every directory of ``tensorpotential`` a package."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def tree(tmp_path: Path, **modules: str) -> Path:
    """A source tree with the base module and ``modules`` (name -> source), the name being the module in ``instructions``."""
    files = {
        "tensorpotential/__init__.py": "",
        "tensorpotential/instructions/__init__.py": "",
        "tensorpotential/instructions/base.py": "class TPInstruction: ...\n"
        "def capture_init_args(cls): return cls\n",
    }
    files |= {
        f"tensorpotential/instructions/{name}.py": text
        for name, text in modules.items()
    }
    return write_tree(tmp_path, files)


def run(root: Path, path: str) -> ia.ClassInfo:
    return ia.extract(root)[f"tensorpotential.instructions.{path}"]


# ------------------------------------------------------------------------------------------------------ logic


def test_a_direct_subclass_of_the_root_is_found_with_its_location(tmp_path):
    root = tree(tmp_path, a=BASE + "\n\nclass A(TPInstruction):\n    pass\n")
    info = run(root, "a.A")
    assert (info.file, info.line) == ("tensorpotential/instructions/a.py", 4)
    assert list(ia.extract(root)) == [
        "tensorpotential.instructions.a.A",
        "tensorpotential.instructions.base.TPInstruction",
    ]


def test_classes_that_are_not_instructions_are_left_out(tmp_path):
    root = tree(tmp_path, a="class Plain:\n    pass\n\nclass Other(Plain):\n    pass\n")
    assert list(ia.extract(root)) == ["tensorpotential.instructions.base.TPInstruction"]


def test_a_base_in_the_same_file_is_followed(tmp_path):
    root = tree(
        tmp_path,
        a=BASE + "class Mid(TPInstruction):\n    pass\nclass Leaf(Mid):\n    pass\n",
    )
    assert {"a.Mid", "a.Leaf"} <= {
        p.removeprefix("tensorpotential.instructions.") for p in ia.extract(root)
    }


def test_a_base_imported_from_another_module_is_followed(tmp_path):
    root = tree(
        tmp_path,
        a=BASE + "class Mid(TPInstruction):\n    pass\n",
        b="from tensorpotential.instructions.a import Mid\nclass Leaf(Mid):\n    pass\n",
    )
    assert "tensorpotential.instructions.b.Leaf" in ia.extract(root)


def test_a_reexported_base_is_followed_to_its_definition(tmp_path):
    """``uq/instructions.py`` imports ``TPInstruction`` from ``compute``, which only imports it from ``base``."""
    root = tree(
        tmp_path,
        a="from tensorpotential.instructions.base import TPInstruction\n",
        b="from tensorpotential.instructions.a import TPInstruction\nclass Leaf(TPInstruction):\n    pass\n",
    )
    assert "tensorpotential.instructions.b.Leaf" in ia.extract(root)


def test_an_aliased_base_is_followed(tmp_path):
    root = tree(
        tmp_path,
        a="from tensorpotential.instructions.base import TPInstruction as Root\nclass Leaf(Root):\n    pass\n",
    )
    assert "tensorpotential.instructions.a.Leaf" in ia.extract(root)


def test_a_relative_import_is_resolved(tmp_path):
    root = tree(
        tmp_path,
        a="from .base import TPInstruction\nclass Leaf(TPInstruction):\n    pass\n",
    )
    assert "tensorpotential.instructions.a.Leaf" in ia.extract(root)


def test_a_relative_import_in_a_package_init_is_resolved_from_the_package(tmp_path):
    root = tree(tmp_path)
    write_tree(
        root,
        {
            "tensorpotential/instructions/__init__.py": "from .base import TPInstruction\n"
            "class Leaf(TPInstruction):\n    pass\n"
        },
    )
    assert "tensorpotential.instructions.Leaf" in ia.extract(root)


def test_a_base_written_as_an_attribute_of_an_imported_module_is_followed(tmp_path):
    root = tree(
        tmp_path,
        a="from tensorpotential.instructions import base\nclass Leaf(base.TPInstruction):\n    pass\n",
    )
    assert "tensorpotential.instructions.a.Leaf" in ia.extract(root)


def test_a_base_that_cannot_be_resolved_does_not_make_an_instruction(tmp_path):
    root = tree(
        tmp_path,
        a="import abc\nclass Leaf(abc.ABC, (object)):\n    pass\n",
    )
    assert "tensorpotential.instructions.a.Leaf" not in ia.extract(root)


def test_circular_bases_terminate(tmp_path):
    root = tree(tmp_path, a="class A(B):\n    pass\nclass B(A):\n    pass\n")
    assert "tensorpotential.instructions.a.A" not in ia.extract(root)


def test_the_compat_package_is_skipped(tmp_path):
    root = tree(tmp_path)
    write_tree(
        root,
        {
            "tensorpotential/compat/__init__.py": "",
            "tensorpotential/compat/old.py": BASE
            + "class Legacy(TPInstruction):\n    pass\n",
        },
    )
    assert not [p for p in ia.extract(root) if ".compat." in p]


def test_the_class_decorator_marks_a_class_as_captured(tmp_path):
    root = tree(
        tmp_path,
        a=BASE + "@capture_init_args\nclass Yes(TPInstruction):\n    pass\n"
        "class No(TPInstruction):\n    pass\n"
        "@capture_init_args()\nclass Called(TPInstruction):\n    pass\n"
        "@base.capture_init_args\nclass Dotted(TPInstruction):\n    pass\n"
        "@other\nclass Other(TPInstruction):\n    pass\n",
    )
    flags = {p.rsplit(".", 1)[-1]: i.captured for p, i in ia.extract(root).items()}
    assert (flags["Yes"], flags["Called"], flags["Dotted"]) == (True, True, True)
    assert (flags["No"], flags["Other"]) == (False, False)


def test_the_constructor_of_a_class_without_one_is_its_parents(tmp_path):
    root = tree(
        tmp_path,
        a=BASE
        + "class Parent(TPInstruction):\n    def __init__(self, x, y=2, name='P'):\n        pass\n"
        "class Child(Parent):\n    pass\n"
        "class Own(Parent):\n    def __init__(self, z=1):\n        pass\n",
    )
    assert [p.name for p in run(root, "a.Child").parameters] == ["x", "y", "name"]
    assert [p.name for p in run(root, "a.Own").parameters] == ["z"]


def test_a_class_with_no_constructor_anywhere_has_no_parameters(tmp_path):
    root = tree(tmp_path, a=BASE + "class Bare(TPInstruction):\n    pass\n")
    assert run(root, "a.Bare").parameters == ()


def test_parameter_kinds_defaults_and_non_literal_defaults(tmp_path):
    root = tree(
        tmp_path,
        a=BASE + "class K(TPInstruction):\n"
        "    def __init__(self, a, /, b, c=-1, *args, d, e=(1, 'x'), f=np.float64, **kw):\n"
        "        pass\n",
    )
    rows = {r["name"]: r for r in run(root, "a.K").as_row()["parameters"]}
    assert {n: r["kind"] for n, r in rows.items()} == {
        "a": "POSITIONAL_ONLY",
        "b": "POSITIONAL_OR_KEYWORD",
        "c": "POSITIONAL_OR_KEYWORD",
        "args": "VAR_POSITIONAL",
        "d": "KEYWORD_ONLY",
        "e": "KEYWORD_ONLY",
        "f": "KEYWORD_ONLY",
        "kw": "VAR_KEYWORD",
    }
    assert {
        n: r.get("default_repr") for n, r in rows.items() if "default_repr" in r
    } == {
        "c": "-1",
        "e": "(1, 'x')",
        "f": "np.float64",
    }
    info = run(root, "a.K")
    assert info.defaults() == {
        "c": -1,
        "e": (1, "x"),
    }  # a non-literal default is not a value


def test_name_is_not_a_default_of_the_class(tmp_path):
    root = tree(
        tmp_path,
        a=BASE
        + "class N(TPInstruction):\n    def __init__(self, name='N', k=1):\n        pass\n",
    )
    assert run(root, "a.N").defaults() == {"k": 1}


def test_constructor_info_reads_a_class_that_is_not_an_instruction(tmp_path):
    root = tree(
        tmp_path, a="class Plain:\n    def __init__(self, p=5):\n        pass\n"
    )
    info = ia.constructor_info(root, "tensorpotential.instructions.a.Plain")
    assert info is not None
    assert info.defaults() == {"p": 5}
    assert ia.constructor_info(root, "tensorpotential.instructions.a.Missing") is None


def test_module_names(tmp_path):
    root = tmp_path
    assert (
        ia.module_name(root, root / "tensorpotential/a/b.py") == "tensorpotential.a.b"
    )
    assert (
        ia.module_name(root, root / "tensorpotential/a/__init__.py")
        == "tensorpotential.a"
    )


def test_the_table_command_prints_the_rows_as_json(tmp_path, capsys):
    root = tree(
        tmp_path,
        a=BASE
        + "@capture_init_args\nclass A(TPInstruction):\n    def __init__(self, k=1):\n        pass\n",
    )
    assert ia.main(["table", "--root", str(root)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows["tensorpotential.instructions.a.A"] == {
        "captured": True,
        "parameters": [
            {"name": "k", "kind": "POSITIONAL_OR_KEYWORD", "default_repr": "1"}
        ],
    }


# ------------------------------------------------------------------------------ physical values (the oracle)


def test_the_table_of_the_real_source_equals_the_table_written_from_the_imported_classes():
    committed = json.loads(
        (ROOT / "tests/data/instruction_constructor_defaults.json").read_text()
    )
    assert ia.table_rows(ia.extract(ROOT)) == committed


def test_the_extractor_does_not_import_tensorflow():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'tools')\n"
        "import instruction_ast as ia\n"
        "ia.extract('.')\n"
        "assert 'tensorflow' not in sys.modules and 'tf_keras' not in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_the_check_command_runs_without_tensorflow_and_passes_on_this_tree():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('tensorflow', 'tf_keras', 'torch'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, 'tools')\n"
        "import instruction_ast as ia\n"
        "raise SystemExit(ia.main(['check']))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "39 instruction classes" in done.stdout


@pytest.mark.parametrize("name", ["table", "check"])
def test_the_script_runs_from_the_command_line(name):
    done = subprocess.run(
        [sys.executable, "tools/instruction_ast.py", name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
