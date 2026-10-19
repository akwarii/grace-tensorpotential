"""The ``MLPRadialFunction`` twin against TensorFlow.

For each omat tiny anchor (float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs; the large model uses
``MLPRadialFunction_v2``), every ``MLPRadialFunction`` instance (``R``, ``R1``), given the stored TF radial basis and
the stored weights, reproduces the stored TF output on every structure within the ``FEATURE`` row of the parameter
dtype. TF writes ``R`` as ``[bonds, n_rad_max, lm]`` in both layouts; the twin returns ``[bonds, lm, n_rad_max]``.

The width of the basis comes from the TF stand-in (``tests_torch/tools/tfstub.py``), so no TensorFlow is needed.
Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.mlp_radial_function import (
    MLPRadialFunction,
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


def _instances(cls: str) -> list[tuple[str, str]]:
    return [
        (anchor, name)
        for anchor in ANCHORS
        for name, entry in _manifest(anchor)["instructions"].items()
        if entry["class"] == cls
    ]


RADIALS = _instances("MLPRadialFunction")


def test_every_instance_is_compared():
    expected = [(a, n) for a in ANCHORS if a.startswith("omat") for n in ("R", "R1")]
    assert sorted(RADIALS) == sorted(expected)


def _radial(anchor: str, name: str) -> MLPRadialFunction:
    spec = _specs(anchor)[name]
    basis = _tf_model(anchor)[spec.options["basis"].name]
    twin = MLPRadialFunction(spec, basis.basis_function.nfunc, _policy(anchor))
    weights = _weights(anchor)
    state = {
        k: torch.as_tensor(weights[f"{name}/{k.replace('.', '/')}"])
        for k in twin.state_dict()
    }
    twin.load_state_dict(state, strict=True)
    return twin


@pytest.mark.parametrize(("anchor", "name"), RADIALS)
def test_the_radial_function_reproduces_tf_on_every_structure(anchor, name):
    twin = _radial(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    basis = twin.spec.options["basis"].name
    for case in cases(arrays):
        inputs = {basis: torch.as_tensor(arrays[f"{case}/ins/{basis}"])}
        with torch.no_grad():
            out = twin(inputs)
        assert out.dtype == _policy(anchor).param
        # TF keeps the lm axis of R last in both layouts
        expected = to_internal(arrays[f"{case}/ins/{name}"], "default")
        assert_close(out, expected, row, f"{anchor} {case} {name}")
