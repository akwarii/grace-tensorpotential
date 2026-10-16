"""The ``SingleParticleBasisFunctionEquivariantInd`` and ``FunctionReduceN`` plans against TensorFlow, every instance.

Two oracles (exit criterion of PLAN2): the ``tables/<instruction>/<attribute>`` and the variable list that TensorFlow
dumped into the golden fixtures (FIX2: the eight tiny anchors, the faithful manifests, and the faithful arrays where
the git-ignored files exist), and the tables that the unmodified TF ``__init__`` and ``build`` make under the TF
stand-in (``tests_torch/tools/tfstub.py``) from the yamls of both tiers. The plans get their options from the
constructor arguments the manifest recorded, the tables of their sources from the stand-in, and chain: the indicator
of ``YI`` is the output layout of the ``FunctionReduceN`` plan of ``I``. Instances are selected by class, never by
name. Every comparison is exact: integer tables, and float64 values cast to the dtype of the fixture as TF casts them.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tensorpotential.torch_backend.plans.function_reduce_n import (
    FunctionReduceNOptions,
    FunctionReduceNPlan,
    ReduceSource,
    function_reduce_n_plan,
)
from tensorpotential.torch_backend.plans.spbf_equivariant_ind import (
    EquivariantIndOptions,
    EquivariantIndPlan,
    spbf_equivariant_ind_plan,
)
from tests_torch.golden import GOLDEN
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

FIXTURES = GOLDEN.parent / "fixtures"
REDUCE = "FunctionReduceN"
EQUIV = "SingleParticleBasisFunctionEquivariantInd"
YAMLS = {
    "omat_tiny": GOLDEN / "yamls" / "omat_tiny.yaml",
    "large_tiny": GOLDEN / "yamls" / "large_tiny.yaml",
    "omat_faithful": FIXTURES / "yamls" / "omat_faithful.yaml",
    "large_faithful": FIXTURES / "yamls" / "large_faithful.yaml",
}
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in YAMLS for v in VARIANTS]
Plan = FunctionReduceNPlan | EquivariantIndPlan


def _directory(anchor: str) -> Path:
    return GOLDEN if "_tiny_" in anchor else FIXTURES


def _model_of(anchor: str) -> str:
    return "_".join(anchor.split("_")[:2])


def _manifest(anchor: str) -> dict[str, Any]:
    return json.loads((_directory(anchor) / f"{anchor}.json").read_text())


def _instances(model: str) -> list[tuple[str, str]]:
    entries = _manifest(f"{model}_f64")["instructions"]
    return [
        (n, e["class"]) for n, e in entries.items() if e["class"] in (REDUCE, EQUIV)
    ]


INSTANCES = [(model, name) for model in YAMLS for name, _ in _instances(model)]


def _ref(value: dict[str, Any]) -> str:
    return value["name"]


@functools.cache
def _stand_in(model: str) -> dict[str, Any]:
    return tfstub.load_model(YAMLS[model], float_dtype="float64")


@functools.cache
def _plans(model: str) -> dict[str, Plan]:
    """The plans of every instance of the two classes, in file order, from the manifest and the stand-in's sources."""
    tf_model = _stand_in(model)
    plans: dict[str, Plan] = {}
    for name, entry in _manifest(f"{model}_f64")["instructions"].items():
        args = entry["init_args"]
        if entry["class"] == REDUCE:
            sources = [tf_model[_ref(r)] for r in args["instructions"]]
            plans[name] = function_reduce_n_plan(
                [ReduceSource(s.name, s.coupling_meta_data, s.n_out) for s in sources],
                FunctionReduceNOptions.from_options(args),
            )
        elif entry["class"] == EQUIV:
            indicator = plans[_ref(args["indicator"])]
            assert isinstance(indicator, FunctionReduceNPlan)
            plans[name] = spbf_equivariant_ind_plan(
                tf_model[_ref(args["angular"])].coupling_meta_data,
                indicator.coupling_meta_data,
                EquivariantIndOptions.from_options(args),
                radial_lmax=tf_model[_ref(args["radial"])].lmax,
            )
    return plans


def _expected_tables(plan: Plan, name: str, dtype: np.dtype) -> dict[str, np.ndarray]:
    """The arrays TensorFlow keeps after ``build``, from the plan, keyed as in a fixture."""
    if isinstance(plan, EquivariantIndPlan):
        return {
            f"tables/{name}/cg": plan.cg.reshape(-1, 1, 1).astype(dtype),
            f"tables/{name}/inv_avg_n_neigh": np.asarray(plan.inv_avg_n_neigh, dtype),
            f"tables/{name}/lr_inds": plan.lr_inds,
            f"tables/{name}/m_sum_ind": plan.m_sum_ind,
            f"tables/{name}/nfunc": np.asarray(plan.nfunc, np.int32),
        }
    tables = {
        f"tables/{name}/norm_{c.name}": np.asarray(c.norm, dtype)
        for c in plan.collectors
    }
    tables[f"tables/{name}/norm_map"] = plan.runtime_norm_map.astype(dtype)
    return tables


def _assert_same(got: np.ndarray, expected: np.ndarray, key: str) -> None:
    assert (got.dtype, got.shape) == (expected.dtype, expected.shape), key
    np.testing.assert_array_equal(got, expected, err_msg=key)


@functools.cache
def _arrays(anchor: str) -> dict[str, np.ndarray]:
    path = _directory(anchor) / f"{anchor}.npz"
    if not path.exists():
        pytest.skip(
            f"{path.name} is not generated here (faithful arrays are git-ignored: make_golden.py write)"
        )
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files if k.startswith("tables/")}


@pytest.mark.parametrize("anchor", ANCHORS)
def test_tables_equal_the_fixture(anchor):
    arrays = _arrays(anchor)
    dtype = np.dtype(_manifest(anchor)["dtype"])
    for name, plan in _plans(_model_of(anchor)).items():
        expected = _expected_tables(plan, name, dtype)
        assert sorted(k for k in arrays if k.startswith(f"tables/{name}/")) == sorted(
            expected
        ), name
        for key, value in expected.items():
            _assert_same(value, arrays[key], key)


@pytest.mark.parametrize("anchor", ANCHORS)
def test_weight_shapes_equal_the_manifest(anchor):
    manifest = _manifest(anchor)
    rows = {r["key"]: r for r in manifest["variables"]}
    for name, plan in _plans(_model_of(anchor)).items():
        if isinstance(plan, FunctionReduceNPlan):
            expected = {
                f"{name}/reducing_{c.name}": c.weight_shape for c in plan.collectors
            }
            assert {k for k in rows if k.startswith(f"{name}/")} == set(expected), name
            for key, shape in expected.items():
                assert (
                    tuple(rows[key]["shape"]),
                    rows[key]["dtype"],
                    rows[key]["trainable"],
                ) == (
                    shape,
                    manifest["dtype"],
                    True,
                ), key
        else:
            assert not [k for k in rows if k.startswith(f"{name}/")], name


def _frame_without_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    # the "symbol" column is the symbolic polynomial of each row that poly.init_coupling_symbols adds for the export
    return frame.drop(columns="symbol", errors="ignore")


@pytest.mark.parametrize(("model", "name"), INSTANCES)
def test_plan_equals_the_stand_in(model, name):
    plan = _plans(model)[name]
    tf = _stand_in(model)[name]
    pd.testing.assert_frame_equal(
        plan.coupling_meta_data, _frame_without_symbols(tf.coupling_meta_data)
    )
    for key, value in _expected_tables(plan, name, np.dtype("float64")).items():
        _assert_same(value, tfstub.arrays(tf)[key.removeprefix("tables/")], key)
    if isinstance(plan, EquivariantIndPlan):
        assert (plan.slice_angular, plan.nfunc) == (tf.slice_angular, int(tf.nfunc))
        return
    assert (plan.lmax, plan.only_invar, plan.out_norm) == (
        tf.lmax,
        tf.only_invar,
        tf.out_norm,
    )
    assert [c.name for c in plan.collectors] == list(tf.collector)
    for c in plan.collectors:
        t = tf.collector[c.name]
        for key in ("func_collect_ind", "w_l_tile", "total_sum_ind"):
            _assert_same(getattr(c, key), np.asarray(t[key]), f"{name}/{c.name}/{key}")
        assert (c.w_shape, c.weight_shape) == (
            t["w_shape"],
            getattr(tf, f"reducing_{c.name}").shape,
        )
    variables = tfstub.variables(tf)
    assert {k: v.shape for k, v in variables.items()} == {
        f"{name}/reducing_{c.name}": c.weight_shape for c in plan.collectors
    }


def test_every_instance_of_both_models_is_covered():
    counts = {m: sorted(c for _, c in _instances(m)) for m in YAMLS}
    assert counts["omat_tiny"] == [REDUCE] * 4 + [EQUIV]
    assert counts["large_tiny"] == [REDUCE] * 5 + [EQUIV]
    assert counts["omat_faithful"] == counts["omat_tiny"]
    assert counts["large_faithful"] == counts["large_tiny"]
