"""The ``CreateOutputTarget``, ``LinMLPOut2ScalarTarget``, ``ConstantScaleShiftTarget`` and ``TrainableShiftTarget`` twins
against TensorFlow.

The output instructions overwrite one key (``atomic_energy``), so the fixtures keep a snapshot of the target before and
after each of them (``out_before/<name>``, ``out_after/<name>``). For each tiny anchor (both models; float64, float32 and
the ``lm_first`` and ``dense_nbr`` option pairs) and each structure:

* ``CreateOutputTarget`` returns the stored starting value;
* every ``LinMLPOut2ScalarTarget``, ``ConstantScaleShiftTarget`` and ``TrainableShiftTarget`` instance, given the stored TF outputs of its origins
  and its stored ``out_before`` target, with the stored weights loaded, returns the stored ``out_after`` within the
  ``FEATURE`` row of the parameter dtype;
* the chain of each model run end to end from the stored origins (omat: create, readout, scale; large: create, readout,
  trainable shift) gives the model's ``atomic_energy``.

Instances are selected by class, never by name.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.output_targets import (
    ConstantScaleShiftTarget,
    CreateOutputTarget,
    LinMLPOut2ScalarTarget,
    TrainableShiftTarget,
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
def _stand_in_model(model: str) -> dict[str, Any]:
    return tfstub.load_model(GOLDEN / "yamls" / f"{model}.yaml", float_dtype="float64")


def _stand_in(anchor: str) -> dict[str, Any]:
    manifest = _manifest(anchor)
    return _stand_in_model(f"{manifest['model']}_{manifest['tier']}")


@functools.cache
def _specs(anchor: str) -> dict[str, InstructionSpec]:
    """Return the instructions of the anchor from the constructor arguments its manifest recorded (options included)."""
    entries = [e["init_args"] for e in _manifest(anchor)["instructions"].values()]
    return {s.name: s for s in parse_model_spec(entries).instructions}


@functools.cache
def _weights(anchor: str) -> dict[str, np.ndarray]:
    return load(f"{_manifest(anchor)['weights_from'] or anchor}.weights")


def _policy(anchor: str) -> DtypePolicy:
    floats = (w.dtype for w in _weights(anchor).values() if w.dtype.kind == "f")
    return DtypePolicy.from_weights(floats)


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


CREATES = _instances("CreateOutputTarget")
READOUTS = _instances("LinMLPOut2ScalarTarget")
SCALES = _instances("ConstantScaleShiftTarget")
SHIFTS = _instances("TrainableShiftTarget")


def test_every_instance_of_both_models_is_compared():
    assert sorted(CREATES) == sorted((a, "atomic_energy") for a in ANCHORS)
    # the omat yaml names its readout MLPOut2ScalarTarget
    readout = {"omat": "MLPOut2ScalarTarget", "large": "LinMLPOut2ScalarTarget"}
    assert sorted(READOUTS) == sorted((a, readout[a.split("_")[0]]) for a in ANCHORS)
    assert sorted(SCALES) == sorted(
        (a, "ConstantScaleShiftTarget") for a in ANCHORS if a.startswith("omat")
    )
    assert sorted(SHIFTS) == sorted(
        (a, "TrainableShiftTarget") for a in ANCHORS if a.startswith("large")
    )


def _create(anchor: str, name: str) -> CreateOutputTarget:
    return CreateOutputTarget(_specs(anchor)[name], _policy(anchor))


def _readout(anchor: str, name: str) -> LinMLPOut2ScalarTarget:
    spec = _specs(anchor)[name]
    tf_model = _stand_in(anchor)
    channels = [tf_model[ref.name].n_out for ref in spec.options["origin"]]
    twin = LinMLPOut2ScalarTarget(spec, channels, _policy(anchor))
    _load_weights(twin, anchor, name)
    return twin


def _scale(anchor: str, name: str) -> ConstantScaleShiftTarget:
    n_elements = len(_manifest(anchor)["elements"])
    return ConstantScaleShiftTarget(
        _specs(anchor)[name], _policy(anchor), n_elements=n_elements
    )


def _shift(anchor: str, name: str) -> TrainableShiftTarget:
    twin = TrainableShiftTarget(_specs(anchor)[name], _policy(anchor))
    _load_weights(twin, anchor, name)
    return twin


def _origins(
    anchor: str, twin: LinMLPOut2ScalarTarget, arrays, case: str
) -> dict[str, torch.Tensor]:
    """The stored origins in the backend layout: TF writes ``[lm, atoms, n]`` only for an origin with ``lm_first``."""
    inputs = {}
    for ref in twin.spec.options["origin"]:
        lm_first = _specs(anchor)[ref.name].options.get("lm_first", False)
        stored = arrays[f"{case}/ins/{ref.name}"]
        layout = "lm_first" if lm_first else "default"
        inputs[ref.name] = torch.as_tensor(to_internal(stored, layout))
    return inputs


@pytest.mark.parametrize(("anchor", "name"), CREATES)
def test_the_target_starts_from_the_stored_value(anchor, name):
    out = _create(anchor, name)({})
    assert out.shape == () and out.dtype == _policy(anchor).param
    for case in cases(arrays := load(anchor)):
        np.testing.assert_array_equal(out.numpy(), arrays[f"{case}/ins/{name}"])


@pytest.mark.parametrize(("anchor", "name"), READOUTS)
def test_the_readout_reproduces_tf_on_every_structure(anchor, name):
    twin = _readout(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    target = twin.spec.options["target"].name
    for case in cases(arrays):
        inputs = _origins(anchor, twin, arrays, case)
        inputs[target] = torch.as_tensor(arrays[f"{case}/out_before/{name}"])
        with torch.no_grad():
            out = twin(inputs)
        assert out.dtype == _policy(anchor).param
        assert_close(
            out, arrays[f"{case}/out_after/{name}"], row, f"{anchor} {case} {name}"
        )


@pytest.mark.parametrize(("anchor", "name"), SCALES)
def test_the_scale_reproduces_tf_on_every_structure(anchor, name):
    twin = _scale(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    target = twin.spec.options["target"].name
    for case in cases(arrays):
        inputs = {
            target: torch.as_tensor(arrays[f"{case}/out_before/{name}"]),
            "atomic_mu_i": torch.as_tensor(arrays[f"{case}/in/atomic_mu_i"]),
        }
        out = twin(inputs)
        assert out.dtype == _policy(anchor).param
        assert_close(
            out, arrays[f"{case}/out_after/{name}"], row, f"{anchor} {case} {name}"
        )


@pytest.mark.parametrize(("anchor", "name"), SHIFTS)
def test_the_trainable_shift_reproduces_tf_on_every_structure(anchor, name):
    twin = _shift(anchor, name)
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    target = twin.spec.options["target"].name
    for case in cases(arrays):
        inputs = {
            target: torch.as_tensor(arrays[f"{case}/out_before/{name}"]),
            "atomic_mu_i": torch.as_tensor(arrays[f"{case}/in/atomic_mu_i"]),
        }
        with torch.no_grad():
            out = twin(inputs)
        assert out.dtype == _policy(anchor).param
        assert_close(
            out, arrays[f"{case}/out_after/{name}"], row, f"{anchor} {case} {name}"
        )


@pytest.mark.parametrize("anchor", ANCHORS)
def test_the_chain_of_each_model_gives_its_atomic_energy(anchor):
    (create,) = (n for a, n in CREATES if a == anchor)
    (readout,) = (n for a, n in READOUTS if a == anchor)
    if anchor.startswith("omat"):
        (scale,) = (n for a, n in SCALES if a == anchor)
        last = _scale(anchor, scale)
    else:
        (shift,) = (n for a, n in SHIFTS if a == anchor)
        last = _shift(anchor, shift)
    twins = _create(anchor, create), _readout(anchor, readout), last
    arrays = load(anchor)
    row = row_for(_manifest(anchor)["dtype"], "feature")
    for case in cases(arrays):
        data = _origins(anchor, twins[1], arrays, case)
        data["atomic_mu_i"] = torch.as_tensor(arrays[f"{case}/in/atomic_mu_i"])
        with torch.no_grad():
            for twin in twins:  # each output instruction writes the target's name
                data["atomic_energy"] = twin(data)
        assert_close(
            data["atomic_energy"],
            arrays[f"{case}/res/atomic_energy"],
            row,
            f"{anchor} {case}",
        )
