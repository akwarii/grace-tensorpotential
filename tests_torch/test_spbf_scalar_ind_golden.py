"""The ``ScalarChemicalEmbedding`` and ``SingleParticleBasisFunctionScalarInd`` twins against TensorFlow.

For each tiny anchor (both models; float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs):

* every ``ScalarChemicalEmbedding`` instance (``Z``), with the stored weights loaded, returns the stored TF output
  exactly, and the element map of the yaml equals the two element-map variables of the checkpoint;
* every ``SingleParticleBasisFunctionScalarInd`` instance (``A``, ``B0``), given the stored TF outputs of its radial,
  angular and indicator inputs, the stored bond and atom indices and weights, and the constructor arguments the
  manifest recorded, reproduces the stored TF output on every structure within the ``FEATURE`` row of the parameter
  dtype.

The coupling tables and widths of the inputs come from the TF stand-in (``tests_torch/tools/tfstub.py``), so no
TensorFlow is needed. Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.scalar_chemical_embedding import (
    ScalarChemicalEmbedding,
)
from tensorpotential.torch_backend.instructions.spbf_scalar_ind import (
    SingleParticleBasisFunctionScalarInd,
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


def _load_weights(twin: torch.nn.Module, anchor: str, name: str) -> None:
    weights = _weights(anchor)
    state = {
        k: torch.as_tensor(weights[f"{name}/{k.replace('.', '/')}"])
        for k in twin.state_dict()
    }
    twin.load_state_dict(state, strict=True)


def _instances(cls: str) -> list[tuple[str, str]]:
    return [
        (anchor, name)
        for anchor in ANCHORS
        for name, entry in _manifest(anchor)["instructions"].items()
        if entry["class"] == cls
    ]


EMBEDDINGS = _instances("ScalarChemicalEmbedding")
BASES = _instances("SingleParticleBasisFunctionScalarInd")


def test_every_instance_of_both_models_is_compared():
    assert sorted(EMBEDDINGS) == sorted((a, "Z") for a in ANCHORS)
    assert sorted(BASES) == sorted((a, n) for a in ANCHORS for n in ("A", "B0"))


def _embedding(anchor: str, name: str) -> ScalarChemicalEmbedding:
    twin = ScalarChemicalEmbedding(_specs(anchor)[name], _policy(anchor))
    _load_weights(twin, anchor, name)
    return twin


@pytest.mark.parametrize(("anchor", "name"), EMBEDDINGS)
def test_the_embedding_returns_the_stored_table_and_matches_the_checkpoint_map(
    anchor, name
):
    twin = _embedding(anchor, name)
    weights, arrays = _weights(anchor), load(anchor)
    assert list(twin.element_map) == weights[f"{name}/element_map_symbols"].tolist()
    assert (
        list(twin.element_map.values()) == weights[f"{name}/element_map_index"].tolist()
    )
    for case in cases(arrays):
        out = twin({})
        assert out.dtype == twin.w.dtype
        np.testing.assert_array_equal(
            out.detach().numpy(), arrays[f"{case}/ins/{name}"]
        )


def _basis(anchor: str, name: str) -> SingleParticleBasisFunctionScalarInd:
    spec, tf_model = _specs(anchor)[name], _tf_model(anchor)
    radial = tf_model[spec.options["radial"].name]
    indicator = spec.options["indicator"]
    twin = SingleParticleBasisFunctionScalarInd(
        spec,
        tf_model[spec.options["angular"].name].coupling_meta_data,
        radial_lmax=radial.lmax,
        n_rad_max=radial.n_rad_max,
        embedding_size=None
        if indicator is None
        else tf_model[indicator.name].embedding_size,
        policy=_policy(anchor),
    )
    _load_weights(twin, anchor, name)
    return twin


@pytest.mark.parametrize(("anchor", "name"), BASES)
def test_the_basis_reproduces_tf_on_every_structure(anchor, name):
    twin = _basis(anchor, name)
    layout = "lm_first" if _manifest(anchor)["option"] == "lm_first" else "default"
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    radial, angular, indicator = (
        twin.spec.options[k].name for k in ("radial", "angular", "indicator")
    )
    for case in cases(arrays):
        inputs = {
            # the radial input keeps its lm axis last in both TF layouts
            radial: torch.as_tensor(
                to_internal(arrays[f"{case}/ins/{radial}"], "default")
            ),
            angular: torch.as_tensor(arrays[f"{case}/ins/{angular}"]),
            indicator: torch.as_tensor(arrays[f"{case}/ins/{indicator}"]),
        }
        for key in ("ind_i", "mu_j", "atomic_mu_i"):
            inputs[key] = torch.as_tensor(arrays[f"{case}/in/{key}"])
        with torch.no_grad():
            out = twin(inputs)
        expected = to_internal(arrays[f"{case}/ins/{name}"], layout)
        assert out.dtype == _policy(anchor).param
        assert_close(out, expected, row, f"{anchor} {case} {name}")
