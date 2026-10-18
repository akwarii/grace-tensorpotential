"""The ``ProductFunction`` twin against TensorFlow: every instance of the tiny anchors, teacher-forced.

For each anchor (both models; float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs) and each structure,
the twin of every ``ProductFunction`` instance gets the stored TF outputs of its two operands and the constructor
arguments the manifest recorded; its output must equal the stored TF output within the ``FEATURE`` row of the parameter
dtype. The twin builds its coupling table from the operands' tables and its options; that table must equal the one the
unmodified TF constructor builds under the stand-in (``tests_torch/tools/tfstub.py``), which also gives the operands'
tables, so no TensorFlow is needed. Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.product_function import (
    ProductFunction,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tests_torch.golden import GOLDEN, cases, load
from tests_torch.golden.tolerances import assert_close, row_for
from tests_torch.layout import to_internal
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

CLASS = "ProductFunction"
MODELS = ("omat_tiny", "large_tiny")
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in MODELS for v in VARIANTS]
PRODUCTS_PER_MODEL = (
    6  # AA, AAA, AAAA, BB, BBB, BBBB in both yamls (spec sheet, "Used in")
)
TABLE_COLUMNS = ("l", "m", "hist", "parity", "l1", "l2", "sum_of_ls")


@functools.cache
def _manifest(anchor: str) -> dict[str, Any]:
    return json.loads((GOLDEN / f"{anchor}.json").read_text())


@functools.cache
def _stand_in(model: str) -> dict[str, Any]:
    return tfstub.load_model(GOLDEN / "yamls" / f"{model}.yaml", float_dtype="float64")


def _tf_model(anchor: str) -> dict[str, Any]:
    manifest = _manifest(anchor)
    return _stand_in(f"{manifest['model']}_{manifest['tier']}")


@functools.cache
def _specs(anchor: str) -> dict[str, InstructionSpec]:
    """Return the instructions of the anchor from the constructor arguments its manifest recorded (options included)."""
    entries = [e["init_args"] for e in _manifest(anchor)["instructions"].values()]
    return {s.name: s for s in parse_model_spec(entries).instructions}


def _twin(anchor: str, name: str) -> ProductFunction:
    spec = _specs(anchor)[name]
    tf_model = _tf_model(anchor)
    weights = load(f"{_manifest(anchor)['weights_from'] or anchor}.weights")
    return ProductFunction(
        spec,
        tf_model[spec.options["left"].name].coupling_meta_data,
        tf_model[spec.options["right"].name].coupling_meta_data,
        policy=DtypePolicy.from_weights(w.dtype for w in weights.values()),
    )


INSTANCES = [
    (anchor, name)
    for anchor in ANCHORS
    for name, entry in _manifest(anchor)["instructions"].items()
    if entry["class"] == CLASS
]


def test_every_instance_of_both_models_is_compared():
    per_anchor = {a: sorted(n for b, n in INSTANCES if b == a) for a in ANCHORS}
    assert all(len(names) == PRODUCTS_PER_MODEL for names in per_anchor.values()), (
        per_anchor
    )
    lm_first = [_specs(a)["AA"].options["lm_first"] for a in ANCHORS]
    assert lm_first == [a.endswith("lm_first") for a in ANCHORS]


@pytest.mark.parametrize(
    ("anchor", "name"), [(a, n) for a, n in INSTANCES if a.endswith("_f64")]
)
def test_the_coupling_table_is_the_one_tf_builds(anchor, name):
    """The twin's table, built from its options, equals the one of the TF constructor (the layout does not change it)."""
    ours = _twin(anchor, name).coupling_meta_data
    theirs = _tf_model(anchor)[name].coupling_meta_data
    assert len(ours) == len(theirs)
    for column in TABLE_COLUMNS:
        assert list(ours[column]) == list(theirs[column]), column
    for column in ("left_inds", "right_inds", "cg_list"):
        for mine, tf in zip(ours[column], theirs[column], strict=True):
            np.testing.assert_array_equal(np.ravel(mine), np.ravel(tf), err_msg=column)


@pytest.mark.parametrize(("anchor", "name"), INSTANCES)
def test_the_twin_reproduces_tf_on_every_structure(anchor, name):
    twin = _twin(anchor, name)
    layout = "lm_first" if _manifest(anchor)["option"] == "lm_first" else "default"
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    assert twin.state_dict() == {}
    for case in cases(arrays):
        inputs = {
            twin.spec.options[k].name: torch.as_tensor(
                to_internal(arrays[f"{case}/ins/{twin.spec.options[k].name}"], layout)
            )
            for k in ("left", "right")
        }
        out = twin(inputs)
        expected = to_internal(arrays[f"{case}/ins/{name}"], layout)
        assert out.dtype == next(iter(inputs.values())).dtype
        assert_close(out, expected, row, f"{anchor} {case} {name}")
