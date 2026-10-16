"""The ``ProductFunction`` and ``FCRight2Left`` plans against TensorFlow's tables, for every instance of both models.

Two oracles (exit criterion of PLAN1): the ``tables/<instruction>/<attribute>`` that TensorFlow dumped into the golden
fixtures (FIX2; the eight tiny anchors, and the faithful fixtures where their git-ignored arrays exist), and the tables
that the unmodified TF ``__init__`` and ``build`` make under the TF stand-in (``tests_torch/tools/tfstub.py``) from
the yamls of both tiers. The plans are fed the coupling metadata that the stand-in's instances carry; instances are
selected by class, not by name (the output instruction of the omat yaml is named after another class). Every
comparison is exact: the tables are integers, and the floats are the same expressions in float64, cast to the
dtype of the fixture.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest

from tensorpotential.torch_backend.plans.fc_right2left import (
    FCRight2LeftWidths,
    fc_right2left_plan,
)
from tensorpotential.torch_backend.plans.product_function import (
    product_function_plan,
)
from tests_torch.golden import GOLDEN
from tests_torch.tools import tfstub

FIXTURES = GOLDEN.parent / "fixtures"
CLASSES = ("ProductFunction", "FCRight2Left")
YAMLS = {
    "omat_tiny": GOLDEN / "yamls" / "omat_tiny.yaml",
    "large_tiny": GOLDEN / "yamls" / "large_tiny.yaml",
    "omat_faithful": FIXTURES / "yamls" / "omat_faithful.yaml",
    "large_faithful": FIXTURES / "yamls" / "large_faithful.yaml",
}
TINY = [
    f"{m}_tiny_{v}"
    for m in ("omat", "large")
    for v in ("f64", "f32", "f64_dense", "f64_lm_first")
]
FAITHFUL = [t.replace("_tiny_", "_faithful_") for t in TINY]
COUNTS = {"ProductFunction": 6, "FCRight2Left": 6}
"""Instances of each class in every yaml of both models (A, AA, AAA, AAAA and B ... chains)."""


@functools.cache
def _model(yaml: str) -> dict[str, Any]:
    return tfstub.load_model(YAMLS[yaml], float_dtype="float64")


def _instances(yaml: str, cls: str) -> list[Any]:
    return [i for i in _model(yaml).values() if type(i).__name__ == cls]


def _widths(fc: Any) -> FCRight2LeftWidths:
    return FCRight2LeftWidths(
        n_out=fc.n_out,
        n_left=fc.left.n_out,
        n_right=fc.right.n_out,
        left_coefs=fc.left_coefs,
        per_element=tuple(fc.is_central_atom_type_dependent),
        number_of_atom_types=fc.number_of_atom_types,
        normalize=fc.normalize,
    )


def _tables(instruction: Any, lm_first: bool) -> dict[str, np.ndarray]:
    """The plan of ``instruction`` in TensorFlow's names and layout, keyed like the ``tables/`` of a fixture."""
    name = instruction.name
    if type(instruction).__name__ == "ProductFunction":
        p = product_function_plan(instruction.coupling_meta_data)
        cg = p.cg.reshape((-1, 1, 1) if lm_first else (1, 1, -1))
        found = {
            "left_ind": p.left_ind,
            "right_ind": p.right_ind,
            "m_sum_ind": p.m_sum_ind,
            "cg": cg,
        }
        found["nfunc"] = np.int32(p.nfunc)
    else:
        q = fc_right2left_plan(
            instruction.left.coupling_meta_data,
            instruction.right.coupling_meta_data,
            _widths(instruction),
        )
        found = {
            k: getattr(q, k)
            for k in (
                "w_tile_left",
                "collect_to",
                "collect_from",
                "w_tile_right",
                "norm_map",
            )
        }
        found["norm_right"] = np.float64(q.norm_right)
        if q.norm_left is not None:
            found["norm_left"] = np.float64(q.norm_left)
        if instruction.norm_out:
            found["norm_out_factor"] = q.norm_map.reshape(-1, 1, 1)
    return {f"{name}/{key}": np.asarray(value) for key, value in found.items()}


def _weight_shapes(instruction: Any) -> dict[str, tuple[int, ...]]:
    q = fc_right2left_plan(
        instruction.left.coupling_meta_data,
        instruction.right.coupling_meta_data,
        _widths(instruction),
    )
    shapes = {f"{instruction.name}/w_right": q.w_right_shape}
    if q.w_left_shape is not None:
        shapes[f"{instruction.name}/w_left"] = q.w_left_shape
    return shapes


def _assert_same(expected: np.ndarray, plan: np.ndarray, key: str) -> None:
    assert expected.shape == plan.shape, key
    assert expected.dtype.kind == plan.dtype.kind, key
    assert np.array_equal(expected, plan.astype(expected.dtype)), key


# ------------------------------------------------------------------ the stand-in oracle


@pytest.mark.parametrize("yaml", list(YAMLS))
@pytest.mark.parametrize("cls", CLASSES)
def test_the_plans_equal_the_tables_of_the_tf_init_under_the_stand_in(yaml, cls):
    instances = _instances(yaml, cls)
    assert len(instances) == COUNTS[cls]
    for instruction in instances:
        built = tfstub.arrays(instruction)
        tables = _tables(instruction, lm_first=instruction.lm_first)
        assert not set(tables) & set(tfstub.placeholders(instruction))
        for key, value in tables.items():
            _assert_same(built[key], value, key)


@pytest.mark.parametrize("yaml", list(YAMLS))
def test_the_weight_shapes_equal_the_variables_built_under_the_stand_in(yaml):
    for instruction in _instances(yaml, "FCRight2Left"):
        built = {k: v.shape for k, v in tfstub.variables(instruction).items()}
        assert built == _weight_shapes(instruction)


# ------------------------------------------------------------------ the golden fixtures


def _fixture(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folder = GOLDEN if "_tiny_" in name else FIXTURES
    if not (folder / f"{name}.npz").exists():
        pytest.skip(
            f"{name}.npz is git-ignored and not on this machine (make_golden.py write --tier faithful)"
        )
    with np.load(folder / f"{name}.npz", allow_pickle=False) as arrays:
        tables = {
            k.removeprefix("tables/"): arrays[k]
            for k in arrays.files
            if k.startswith("tables/")
        }
    return tables, json.loads((folder / f"{name}.json").read_text())


@pytest.mark.parametrize("name", TINY + FAITHFUL)
def test_the_plans_equal_the_tables_and_weight_shapes_of_the_fixtures(name):
    tables, manifest = _fixture(name)
    yaml = "_".join(name.split("_")[:2])
    lm_first = name.endswith("lm_first")
    shapes = {v["key"]: tuple(v["shape"]) for v in manifest["variables"]}
    seen = dict.fromkeys(CLASSES, 0)
    for instruction in _model(yaml).values():
        cls = type(instruction).__name__
        if cls not in CLASSES:
            continue
        assert manifest["instructions"][instruction.name]["class"] == cls
        seen[cls] += 1
        plan = _tables(instruction, lm_first=lm_first)
        own = {k for k in tables if k.startswith(f"{instruction.name}/")}
        assert own == set(plan), instruction.name
        for key, value in plan.items():
            _assert_same(tables[key], value, key)
        if cls == "FCRight2Left":
            for key, shape in _weight_shapes(instruction).items():
                assert shapes[key] == shape, key
    assert seen == COUNTS
