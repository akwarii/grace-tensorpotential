"""The ``RadialBasis`` twin against TensorFlow for the four bases, on every golden fixture that holds one.

The architecture fixtures (FAM2) give each basis in both parameter dtypes: ``Cheb`` and ``SBessel`` from the preset
defaults and model files, ``Gaussian`` and ``RadSinBessel`` from the preset variants with that ``basis_type``. For each
fixture and structure:

* the instruction, read by the strict loader from the constructor arguments the manifest recorded (so a refused
  ``basis_type`` fails here), with the stored ``rc`` (and ``grid`` and ``scale`` for the Gaussian basis) loaded strictly,
  reproduces the stored TF output from the stored lengths within the ``FEATURE`` row of the parameter dtype;
* the width and the cutoff are those of the yaml and the checkpoint.

Physical values from a numpy oracle that does not call the twin: the stored Gaussian ``grid`` and ``scale`` are
``linspace(rmin, rcut, nfunc)`` and ``-0.5 / (init_gamma * spacing)**2`` of the pinned defaults (the twin reads them
from the checkpoint; this checks that the checkpoint holds what the yaml implies).
"""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.instructions.radial_basis import RadialBasis
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden import GOLDEN, cases, load
from tests_torch.golden.tolerances import assert_close, row_for

pytestmark = pytest.mark.golden

BASES = ("Cheb", "SBessel", "Gaussian", "RadSinBessel")
DTYPES = ("float32", "float64")
CHECKPOINT_KEYS = {
    "rc": "rc",
    "grid": "basis_function/grid",
    "scale": "basis_function/scale",
}
"""Twin buffer -> key of the TF checkpoint under the instruction name."""


@functools.cache
def _manifest(fixture: str) -> dict[str, Any]:
    return json.loads((GOLDEN / f"{fixture}.json").read_text())


def _radial_names(fixture: str) -> list[str]:
    entries = _manifest(fixture)["instructions"].items()
    return [name for name, entry in entries if entry["class"] == "RadialBasis"]


INSTANCES = [
    (path.stem, name)
    for path in sorted(GOLDEN.glob("*.json"))
    for name in _radial_names(path.stem)
]


def _init_args(fixture: str, name: str) -> dict[str, Any]:
    return _manifest(fixture)["instructions"][name]["init_args"]


def _id(instance: tuple[str, str]) -> str:
    return f"{instance[0]}-{instance[1]}"


@functools.cache
def _weights(fixture: str) -> dict[str, np.ndarray]:
    return load(f"{_manifest(fixture)['weights_from'] or fixture}.weights")


def _stored(fixture: str, name: str, key: str) -> np.ndarray:
    """Return the stored weight ``key`` of an instruction; the old list-format yaml keys it by position (``2/rc``)."""
    weights = _weights(fixture)
    if f"{name}/{key}" in weights:
        return weights[f"{name}/{key}"]
    position = list(_manifest(fixture)["instructions"]).index(name)
    return weights[f"{position}/{key}"]


def _twin(fixture: str, name: str) -> RadialBasis:
    """Build the twin from the instruction and the one it reads, through the loader, and load its stored weights."""
    args = _init_args(fixture, name)
    lengths = _manifest(fixture)["instructions"][args["bonds"]["name"]]["init_args"]
    spec = {s.name: s for s in parse_model_spec([lengths, args]).instructions}[name]
    policy = DtypePolicy.from_weights(
        w.dtype for w in _weights(fixture).values() if w.dtype.kind == "f"
    )
    twin = RadialBasis(spec, policy)
    state = {
        k: torch.as_tensor(_stored(fixture, name, CHECKPOINT_KEYS[k]))
        for k in twin.state_dict()
    }
    twin.load_state_dict(state, strict=True)
    return twin


def test_every_basis_is_compared_in_both_dtypes():
    seen = {
        (_init_args(f, n)["basis_type"], _manifest(f)["dtype"]) for f, n in INSTANCES
    }
    assert seen == {(b, d) for b in BASES for d in DTYPES}


@pytest.mark.parametrize(("fixture", "name"), INSTANCES, ids=map(_id, INSTANCES))
def test_the_twin_reproduces_tf_on_every_structure(fixture, name):
    twin = _twin(fixture, name)
    lengths = _init_args(fixture, name)["bonds"]["name"]
    arrays = load(fixture)
    row = row_for(_manifest(fixture)["dtype"], "feature")
    for case in cases(arrays):
        with torch.no_grad():
            out = twin({lengths: torch.as_tensor(arrays[f"{case}/ins/{lengths}"])})
        # float64 lengths give a float64 basis whatever the parameter dtype, as in TF
        assert out.dtype == torch.float64
        assert_close(out, arrays[f"{case}/ins/{name}"], row, f"{fixture} {case} {name}")


@pytest.mark.parametrize(("fixture", "name"), INSTANCES, ids=map(_id, INSTANCES))
def test_the_width_and_cutoff_are_those_of_the_yaml_and_the_checkpoint(fixture, name):
    twin = _twin(fixture, name)
    args = _init_args(fixture, name)
    assert twin.nfunc == args["nfunc"]
    rc = _stored(fixture, name, "rc")
    assert twin.cutoff() == float(rc)
    assert twin.get_buffer("rc").dtype == getattr(torch, str(rc.dtype))
    assert float(rc) == pytest.approx(args["rcut"], rel=1e-7)


GAUSSIAN = [i for i in INSTANCES if _init_args(*i)["basis_type"] == "Gaussian"]


@pytest.mark.parametrize(("fixture", "name"), GAUSSIAN, ids=map(_id, GAUSSIAN))
def test_the_stored_gaussian_grid_and_scale_are_those_the_yaml_implies(fixture, name):
    args = {**DEFAULTS[COMPUTE + "RadialBasis"], **_init_args(fixture, name)}
    grid = np.linspace(args["rmin"], args["rcut"], args["nfunc"])
    scale = -0.5 / (args["init_gamma"] * (grid[1] - grid[0])) ** 2
    dtype = _stored(fixture, name, "rc").dtype
    np.testing.assert_array_equal(
        _stored(fixture, name, "basis_function/grid"), grid.reshape(1, -1).astype(dtype)
    )
    np.testing.assert_array_equal(
        _stored(fixture, name, "basis_function/scale"), np.asarray(scale, dtype)
    )
