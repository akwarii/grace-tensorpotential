"""The ``FCRight2Left`` twin against TensorFlow: every instance of the tiny anchors, teacher-forced.

For each anchor (both models; float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs) and each structure,
the twin of every ``FCRight2Left`` instance gets the stored TF outputs of its two operands, the stored weights, and the
constructor arguments the manifest recorded; its output must equal the stored TF output within the ``FEATURE`` row of
the parameter dtype. The coupling tables of the operands come from the TF stand-in (``tests_torch/tools/tfstub.py``),
so no TensorFlow is needed. Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.fc_right2left import FCRight2Left
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.plans.function_reduce_n import ReduceSource
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tests_torch.golden import GOLDEN, cases, load
from tests_torch.golden.tolerances import assert_close, row_for
from tests_torch.layout import to_internal
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

CLASS = "FCRight2Left"
MODELS = ("omat_tiny", "large_tiny")
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in MODELS for v in VARIANTS]


@functools.cache
def _manifest(anchor: str) -> dict[str, Any]:
    return json.loads((GOLDEN / f"{anchor}.json").read_text())


@functools.cache
def _stand_in(model: str) -> dict[str, Any]:
    return tfstub.load_model(GOLDEN / "yamls" / f"{model}.yaml", float_dtype="float64")


@functools.cache
def _specs(anchor: str) -> dict[str, InstructionSpec]:
    """Return the instructions of the anchor from the constructor arguments its manifest recorded (options included)."""
    entries = [e["init_args"] for e in _manifest(anchor)["instructions"].values()]
    return {s.name: s for s in parse_model_spec(entries).instructions}


def _operand(anchor: str, name: str) -> ReduceSource:
    manifest = _manifest(anchor)
    tf_model = _stand_in(f"{manifest['model']}_{manifest['tier']}")
    return ReduceSource(name, tf_model[name].coupling_meta_data, tf_model[name].n_out)


def _twin(anchor: str, name: str) -> FCRight2Left:
    spec = _specs(anchor)[name]
    weights = load(f"{_manifest(anchor)['weights_from'] or anchor}.weights")
    policy = DtypePolicy.from_weights(w.dtype for w in weights.values())
    left, right = (_operand(anchor, spec.options[k].name) for k in ("left", "right"))
    twin = FCRight2Left(spec, left, right, policy)
    state = {k: torch.as_tensor(weights[f"{name}/{k}"]) for k in twin.state_dict()}
    twin.load_state_dict(state, strict=True)
    return twin


INSTANCES = [
    (anchor, name)
    for anchor in ANCHORS
    for name, entry in _manifest(anchor)["instructions"].items()
    if entry["class"] == CLASS
]


def test_every_instance_of_both_models_is_compared():
    names = {m: sorted(n for a, n in INSTANCES if a == f"{m}_f64") for m in MODELS}
    assert names == {
        "omat_tiny": ["A1", "AA1", "AA2", "B1", "BB1", "BB2"],
        "large_tiny": ["A1", "AA1", "AA2", "B1", "BB1", "BB2"],
    }


@pytest.mark.parametrize(("anchor", "name"), INSTANCES)
def test_the_twin_reproduces_tf_on_every_structure(anchor, name):
    twin = _twin(anchor, name)
    layout = "lm_first" if _manifest(anchor)["option"] == "lm_first" else "default"
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    operands = [twin.spec.options[k].name for k in ("left", "right")]
    for case in cases(arrays):
        inputs = {
            op: torch.as_tensor(to_internal(arrays[f"{case}/ins/{op}"], layout))
            for op in operands
        }
        inputs["atomic_mu_i"] = torch.as_tensor(arrays[f"{case}/in/atomic_mu_i"])
        with torch.no_grad():
            out = twin(inputs)
        expected = to_internal(arrays[f"{case}/ins/{name}"], layout)
        assert out.dtype == twin.get_buffer("norm_map").dtype  # the parameter dtype
        assert_close(out, np.asarray(expected), row, f"{anchor} {case} {name}")
