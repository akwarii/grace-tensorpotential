"""The ``BondLength``, ``ScaledBondVector``, ``SphericalHarmonic`` and ``RadialBasis`` twins against TensorFlow.

For each tiny anchor (both models; float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs) and each
structure:

* every instance of the four classes, given the stored TF inputs (the bond vectors, or the stored output of the
  instruction it reads) and the stored ``rc``, reproduces the stored TF output within the ``FEATURE`` row of the
  parameter dtype; ``Y`` and the basis are float64 whatever the parameter dtype, as in TF;
* the chain run from the bond vectors alone (lengths, directions, harmonics, basis) gives the same outputs;
* the column table of ``SphericalHarmonic`` and the width and cutoff of ``RadialBasis`` equal those of the TF stand-in.

Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from tensorpotential.torch_backend.instructions.geometry import (
    BondLength,
    ScaledBondVector,
    SphericalHarmonic,
)
from tensorpotential.torch_backend.instructions.radial_basis import RadialBasis
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tests_torch.golden import GOLDEN, cases, load
from tests_torch.golden.tolerances import assert_close, row_for
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

MODELS = ("omat_tiny", "large_tiny")
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in MODELS for v in VARIANTS]
# in the order of their data dependencies: the chain test runs them in this order
CLASSES = ("BondLength", "ScaledBondVector", "SphericalHarmonic", "RadialBasis")


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


INSTANCES = {cls: _instances(cls) for cls in CLASSES}


def test_every_instance_of_both_models_is_compared():
    names = {"SphericalHarmonic": "Y"}
    for cls in CLASSES:
        expected = [(a, names.get(cls, cls)) for a in ANCHORS]
        assert sorted(INSTANCES[cls]) == sorted(expected), cls


def _twin(anchor: str, name: str) -> torch.nn.Module:
    spec = _specs(anchor)[name]
    cls = spec.cls.rsplit(".", 1)[-1]
    if cls == "BondLength":
        return BondLength(spec)
    if cls == "ScaledBondVector":
        return ScaledBondVector(spec)
    if cls == "SphericalHarmonic":
        return SphericalHarmonic(spec, _policy(anchor))
    twin = RadialBasis(spec, _policy(anchor))
    weights = _weights(anchor)
    state = {k: torch.as_tensor(weights[f"{name}/{k}"]) for k in twin.state_dict()}
    twin.load_state_dict(state, strict=True)
    return twin


def _stored_inputs(arrays, case: str) -> dict[str, torch.Tensor]:
    """The bond vectors and every stored instruction output of one structure."""
    prefix = f"{case}/"
    data = {}
    for key in arrays:
        if key.startswith((prefix + "in/", prefix + "ins/")):
            data[key.rsplit("/", 1)[-1]] = torch.as_tensor(arrays[key])
    return data


@pytest.mark.parametrize(
    ("anchor", "name"), [i for cls in CLASSES for i in INSTANCES[cls]]
)
def test_each_instance_reproduces_tf_on_every_structure(anchor, name):
    twin = _twin(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    for case in cases(arrays):
        with torch.no_grad():
            out = twin(_stored_inputs(arrays, case))
        assert out.dtype == torch.float64  # no parameter enters the four outputs
        assert_close(out, arrays[f"{case}/ins/{name}"], row, f"{anchor} {case} {name}")


@pytest.mark.parametrize("anchor", ANCHORS)
def test_the_chain_from_the_bond_vectors_gives_every_output(anchor):
    order = [n for cls in CLASSES for a, n in INSTANCES[cls] if a == anchor]
    twins = [_twin(anchor, n) for n in order]
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    for case in cases(arrays):
        data = {"bond_vector": torch.as_tensor(arrays[f"{case}/in/bond_vector"])}
        with torch.no_grad():
            for name, twin in zip(order, twins, strict=True):
                data[name] = twin(data)
        for name in order:
            assert_close(
                data[name], arrays[f"{case}/ins/{name}"], row, f"{anchor} {case} {name}"
            )


@pytest.mark.parametrize(("anchor", "name"), INSTANCES["SphericalHarmonic"])
def test_the_column_table_is_that_of_tf(anchor, name):
    tf_meta = _tf_model(anchor)[name].coupling_meta_data
    twin = _twin(anchor, name)
    assert isinstance(twin, SphericalHarmonic)
    # the model's simplify pass adds a `symbol` column to the shared frame later; the class builds these five
    columns = ["l", "m", "hist", "parity", "sum_of_ls"]
    pd.testing.assert_frame_equal(
        twin.coupling_meta_data.reset_index(drop=True),
        tf_meta[columns].reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(("anchor", "name"), INSTANCES["RadialBasis"])
def test_the_basis_width_and_cutoff_are_those_of_tf(anchor, name):
    twin = _twin(anchor, name)
    assert isinstance(twin, RadialBasis)
    tf_basis = _tf_model(anchor)[name]
    assert twin.nfunc == tf_basis.basis_function.nfunc
    assert twin.cutoff() == float(_weights(anchor)[f"{name}/rc"])
    assert twin.get_buffer("rc").dtype == _policy(anchor).param
