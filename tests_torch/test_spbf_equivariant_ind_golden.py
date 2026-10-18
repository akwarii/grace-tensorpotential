"""The ``SingleParticleBasisFunctionEquivariantInd`` twin against TensorFlow: every instance of the tiny anchors.

For each anchor (both models; float64, float32 and the ``lm_first`` and ``dense_nbr`` option pairs) and each structure,
the twin of every instance (``YI``) gets the stored TF outputs of its radial, angular and indicator inputs and the
stored bond indices, and the constructor arguments the manifest recorded; its output must equal the stored TF output
within the ``FEATURE`` row of the parameter dtype. The ``*_f64_dense`` anchors were made with ``dense_nbr`` on: the
twin sums over the bonds as it always does (decision of the owner). The tiny dense anchors have equal neighbour counts,
so only the faithful ``*_f64_dense`` fixtures hold TF's padding bonds (``icosahedron_satellite``); the faithful tier runs
where its git-ignored arrays were generated and skips otherwise. The coupling tables of the inputs come from the TF
stand-in (``tests_torch/tools/tfstub.py``). Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.spbf_equivariant_ind import (
    SingleParticleBasisFunctionEquivariantInd,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tests_torch.golden import GOLDEN, cases
from tests_torch.golden.tolerances import assert_close, row_for
from tests_torch.layout import to_internal
from tests_torch.tools import tfstub

pytestmark = pytest.mark.golden

CLASS = "SingleParticleBasisFunctionEquivariantInd"
FIXTURES = GOLDEN.parent / "fixtures"
MODELS = ("omat_tiny", "large_tiny", "omat_faithful", "large_faithful")
VARIANTS = ("f64", "f32", "f64_dense", "f64_lm_first")
ANCHORS = [f"{m}_{v}" for m in MODELS for v in VARIANTS]


def _directory(anchor: str) -> Path:
    return GOLDEN if "_tiny_" in anchor else FIXTURES


@functools.cache
def _manifest(anchor: str) -> dict[str, Any]:
    return json.loads((_directory(anchor) / f"{anchor}.json").read_text())


@functools.cache
def _stand_in(model: str) -> dict[str, Any]:
    yaml = _directory(f"{model}_") / "yamls" / f"{model}.yaml"
    return tfstub.load_model(yaml, float_dtype="float64")


def _load(anchor: str, suffix: str = "") -> dict[str, np.ndarray]:
    """Return the arrays of an anchor; skip a faithful one whose git-ignored file is not generated here."""
    path = _directory(anchor) / f"{anchor}{suffix}.npz"
    if not path.exists():
        pytest.skip(
            f"{path.name} is not generated here (faithful arrays are git-ignored: make_golden.py write)"
        )
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


@functools.cache
def _specs(anchor: str) -> dict[str, InstructionSpec]:
    """Return the instructions of the anchor from the constructor arguments its manifest recorded (options included)."""
    entries = [e["init_args"] for e in _manifest(anchor)["instructions"].values()]
    return {s.name: s for s in parse_model_spec(entries).instructions}


def _twin(anchor: str, name: str) -> SingleParticleBasisFunctionEquivariantInd:
    manifest = _manifest(anchor)
    spec = _specs(anchor)[name]
    tf_model = _stand_in(f"{manifest['model']}_{manifest['tier']}")
    weights = _load(manifest["weights_from"] or anchor, ".weights")
    return SingleParticleBasisFunctionEquivariantInd(
        spec,
        tf_model[spec.options["angular"].name].coupling_meta_data,
        tf_model[spec.options["indicator"].name].coupling_meta_data,
        radial_lmax=tf_model[spec.options["radial"].name].lmax,
        policy=DtypePolicy.from_weights(w.dtype for w in weights.values()),
    )


INSTANCES = [
    (anchor, name)
    for anchor in ANCHORS
    for name, entry in _manifest(anchor)["instructions"].items()
    if entry["class"] == CLASS
]


def test_every_instance_of_both_models_is_compared_with_both_neighbour_modes():
    assert sorted(INSTANCES) == sorted((a, "YI") for a in ANCHORS)
    dense = [_specs(a)["YI"].options["dense_nbr"] for a in ANCHORS if "dense" in a]
    assert dense == [True] * 4


@pytest.mark.parametrize(("anchor", "name"), INSTANCES)
def test_the_twin_reproduces_tf_on_every_structure(anchor, name):
    twin = _twin(anchor, name)
    layout = "lm_first" if _manifest(anchor)["option"] == "lm_first" else "default"
    arrays = _load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    radial, angular, indicator = (
        twin.spec.options[k].name for k in ("radial", "angular", "indicator")
    )
    assert twin.state_dict() == {}
    for case in cases(arrays):
        inputs = {
            # the radial input keeps its lm axis last in both TF layouts
            radial: torch.as_tensor(
                to_internal(arrays[f"{case}/ins/{radial}"], "default")
            ),
            angular: torch.as_tensor(arrays[f"{case}/ins/{angular}"]),
            indicator: torch.as_tensor(
                to_internal(arrays[f"{case}/ins/{indicator}"], layout)
            ),
        }
        for key in ("ind_i", "ind_j", "atomic_mu_i"):
            inputs[key] = torch.as_tensor(arrays[f"{case}/in/{key}"])
        out = twin(inputs)
        expected = to_internal(arrays[f"{case}/ins/{name}"], layout)
        assert out.dtype == inputs[indicator].dtype
        assert_close(out, expected, row, f"{anchor} {case} {name}")
