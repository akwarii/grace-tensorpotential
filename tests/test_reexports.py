"""The old import paths of the modules CLI2 moved still work: every name they had is importable and is the moved object.

The names come from ``baselines/cli2_moves.json``, recorded on the tree before the move. ``cli/data.py`` and
``scripts/_kokkos_export.py`` moved whole and now only re-export; ``grace_preprocess`` and ``grace_utils`` keep their
parser, stages and handlers, and import the moved functions from the new modules.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "baselines" / "cli2_moves.json").read_text())
WHOLE_MODULES = ["tensorpotential/cli/data.py", "tensorpotential/scripts/_kokkos_export.py"]


def _module(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


@pytest.mark.parametrize("old", sorted(MANIFEST))
def test_every_name_of_the_old_module_is_importable_from_it(old):
    module = importlib.import_module(_module(old))
    missing = [name for name in MANIFEST[old]["names"] if not hasattr(module, name)]
    assert missing == []


@pytest.mark.parametrize("old", sorted(MANIFEST))
def test_a_moved_unit_is_the_object_of_the_new_module(old):
    entry = MANIFEST[old]
    old_module = importlib.import_module(_module(old))
    new_module = importlib.import_module(_module(entry["new"]))
    different = [n for n in entry["units"] if getattr(old_module, n) is not getattr(new_module, n)]
    assert different == []


@pytest.mark.parametrize("old", WHOLE_MODULES)
def test_a_module_that_moved_whole_holds_no_logic(old):
    tree = ast.parse((ROOT / old).read_text())
    kinds = {type(node).__name__ for node in tree.body}
    assert kinds <= {"Expr", "ImportFrom"}
    assert isinstance(tree.body[0], ast.Expr)  # the one-line docstring saying where the code lives
