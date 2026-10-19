"""The ``InvariantLayerRMSNorm`` twin against TensorFlow.

For each large tiny anchor (float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs; the omat model has no
such layer), every instance (``I_nl_LN`` of type ``only_nonlin``, ``I_0_LN`` of type ``full``), given the stored TF
output of its input and the stored ``scale``, reproduces the stored TF output on every structure within the
``FEATURE`` row of the parameter dtype. TF writes the input as ``[lm, atoms, n]`` when it has ``lm_first`` and the
output always as ``[atoms, n, lm]``; both are converted to the backend layout here.

The channel count of the input comes from the TF stand-in (``tests_torch/tools/tfstub.py``), so no TensorFlow is
needed. Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.invariant_rms_norm import (
    InvariantLayerRMSNorm,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tests_torch.golden import GOLDEN, cases, load
from tests_torch.golden.tolerances import assert_close, row_for
from tests_torch.layout import to_internal
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

MODELS = ("omat_tiny", "large_tiny")
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in MODELS for v in VARIANTS]


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


@functools.cache
def _weights(anchor: str) -> dict[str, np.ndarray]:
    return load(f"{_manifest(anchor)['weights_from'] or anchor}.weights")


def _policy(anchor: str) -> DtypePolicy:
    return DtypePolicy.from_weights(
        w.dtype for w in _weights(anchor).values() if w.dtype.kind == "f"
    )


NORMS = [
    (anchor, name)
    for anchor in ANCHORS
    for name, entry in _manifest(anchor)["instructions"].items()
    if entry["class"] == "InvariantLayerRMSNorm"
]


def test_every_instance_is_compared_with_both_types():
    expected = [
        (a, n) for a in ANCHORS if a.startswith("large") for n in ("I_nl_LN", "I_0_LN")
    ]
    assert sorted(NORMS) == sorted(expected)
    types = {_specs(a)[n].options["type"] for a, n in NORMS}
    assert types == {"full", "only_nonlin"}


def _norm(anchor: str, name: str) -> InvariantLayerRMSNorm:
    spec = _specs(anchor)[name]
    channels = _tf_model(anchor)[spec.options["inpt"].name].n_out
    twin = InvariantLayerRMSNorm(spec, channels, _policy(anchor))
    weights = _weights(anchor)
    state = {k: torch.as_tensor(weights[f"{name}/{k}"]) for k in twin.state_dict()}
    twin.load_state_dict(state, strict=True)
    return twin


@pytest.mark.parametrize(("anchor", "name"), NORMS)
def test_the_norm_reproduces_tf_on_every_structure(anchor, name):
    twin = _norm(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    inpt = twin.spec.options["inpt"].name
    layout = (
        "lm_first" if _specs(anchor)[inpt].options.get("lm_first", False) else "default"
    )
    for case in cases(arrays):
        x = torch.as_tensor(to_internal(arrays[f"{case}/ins/{inpt}"], layout))
        with torch.no_grad():
            out = twin({inpt: x})
        assert out.dtype == _policy(anchor).param
        expected = to_internal(arrays[f"{case}/ins/{name}"], "default")
        assert_close(out, expected, row, f"{anchor} {case} {name}")
