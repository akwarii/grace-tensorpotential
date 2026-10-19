"""The ``InvariantLayerRMSNorm`` twin: logic against a numpy re-computation of the sheet, and the physics of a norm.

Logic: the three types (``full``, ``only_nonlin``, ``sep_lin_gate``) against numpy (``x * rsqrt(mean over channels of
x**2 + 1e-10) * scale``, channel 0 apart and gated by ``lin_scale`` where the type says so), for one and two ``lm``
entries; the weights in their checkpoint shapes, zero at the start; a float32 model; ``init`` ignored; the typed errors.
Physical values: the normalised channels are invariant under a rescaling of the atom's features while the linear
channel follows it, each atom is normalised on its own features only, and the norm is odd in its input. The comparison
with TensorFlow is ``test_invariant_rms_norm_golden.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.instructions.invariant_rms_norm import (
    InvariantLayerRMSNorm,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_ATOMS, N_CH = 5, 6


def _spec(type_: str, **options: Any) -> InstructionSpec:
    cls = COMPUTE + "InvariantLayerRMSNorm"
    return InstructionSpec(
        name="I_LN",
        cls=cls,
        options={
            **DEFAULTS[cls],
            "inpt": InstructionRef("I"),
            "type": type_,
            **options,
        },
        depends_on=(),
        index=0,
    )


def _twin(
    type_="only_nonlin", policy=F64, n_channels=N_CH, **options: Any
) -> InvariantLayerRMSNorm:
    twin = InvariantLayerRMSNorm(_spec(type_, **options), n_channels, policy)
    rng = np.random.default_rng(RNG_SEED + 1)
    state = {
        k: torch.as_tensor(rng.normal(size=v.shape), dtype=policy.param)
        for k, v in twin.state_dict().items()
    }
    twin.load_state_dict(state, strict=True)
    return twin


def _features(rng, lm=1, dtype=np.float64) -> torch.Tensor:
    return torch.as_tensor(rng.normal(size=(N_ATOMS, lm, N_CH)).astype(dtype) * 3.0)


def _oracle(twin: InvariantLayerRMSNorm, x: np.ndarray) -> np.ndarray:
    """The sheet's forward pass in numpy, in the backend layout ``[atoms, lm, n]``."""
    scale = twin.scale.detach().double().numpy()[0, :, 0]

    def norm(y):
        return y / np.sqrt((y**2).mean(axis=-1, keepdims=True) + 1e-10) * scale

    if twin.spec.options["type"] == "full":
        return norm(x)
    lin = x[..., :1]
    if twin.lin_scale is not None:
        lin = lin * float(twin.lin_scale.detach())
    return np.concatenate([lin, norm(x[..., 1:])], axis=-1)


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize("type_", ["full", "only_nonlin", "sep_lin_gate"])
@pytest.mark.parametrize("lm", [1, 2])
def test_each_type_is_the_norm_of_the_sheet(type_, lm):
    twin = _twin(type_)
    x = _features(np.random.default_rng(RNG_SEED), lm)
    out = twin({"I": x})
    assert out.shape == x.shape and out.dtype == torch.float64
    assert_close(out, _oracle(twin, x.numpy()), FEATURE_F64, f"{type_} lm={lm}")


@pytest.mark.parametrize(
    ("type_", "shapes"),
    [
        ("full", {"scale": (1, N_CH, 1)}),
        ("only_nonlin", {"scale": (1, N_CH - 1, 1)}),
        ("sep_lin_gate", {"scale": (1, N_CH - 1, 1), "lin_scale": (1, 1)}),
    ],
)
def test_the_weights_have_their_checkpoint_shapes_and_start_at_zero(type_, shapes):
    state = InvariantLayerRMSNorm(_spec(type_), N_CH, F32).state_dict()
    assert {k: tuple(v.shape) for k, v in state.items()} == shapes
    assert all(v.dtype == torch.float32 and not v.any() for v in state.values())


def test_a_float32_model_normalises_float32_features():
    twin = _twin("full", F32)
    x = _features(np.random.default_rng(RNG_SEED), dtype=np.float32)
    out = twin({"I": x})
    assert out.dtype == torch.float32
    assert_close(out, _oracle(twin, x.double().numpy()), FEATURE_F32, "float32")


@pytest.mark.parametrize("init", ["zeros", "ones", "random", "near_zero"])
def test_init_is_ignored_the_weights_start_at_zero_until_loaded(init):
    state = InvariantLayerRMSNorm(_spec("full", init=init), N_CH, F64).state_dict()
    assert not state["scale"].any()


@pytest.mark.parametrize(
    ("type_", "options", "n_channels", "error", "match"),
    [
        (
            "layer",
            {},
            N_CH,
            OptionValueError,
            "type must be one of full, only_nonlin, sep_lin_gate",
        ),
        ("full", {"inpt": None}, N_CH, OptionValueError, "inpt is null"),
        ("only_nonlin", {}, 1, ShapeError, "needs more than 1 channel"),
        ("full", {}, 0, ShapeError, "needs more than 0 channel"),
    ],
)
def test_bad_options_and_too_few_channels_are_refused(
    type_, options, n_channels, error, match
):
    with pytest.raises(error, match=match):
        _twin(type_, n_channels=n_channels, **options)


# ------------------------------------------------------------------ physical values


@pytest.mark.parametrize("type_", ["full", "only_nonlin", "sep_lin_gate"])
def test_the_normalised_channels_ignore_the_scale_of_the_features(type_):
    twin = _twin(type_)
    x = _features(np.random.default_rng(RNG_SEED))
    factor = 7.5
    out = twin({"I": x}).detach().numpy()
    scaled = twin({"I": factor * x}).detach().numpy()
    first = 0 if type_ == "full" else 1
    # exact up to the epsilon: the ratio is sqrt((ms + e) / (ms + e / factor**2)) ~ 1 + e (1 - 1/factor**2) / (2 ms),
    # e = 1e-10, ms = mean(x**2); 1e-14 for the rounding
    ms = (x.numpy()[..., first:] ** 2).mean(axis=-1, keepdims=True)
    bound = 0.5e-10 * (1 - 1 / factor**2) / ms + 1e-14
    ratio = scaled[..., first:] / out[..., first:]
    assert np.all(np.abs(ratio - 1) <= bound)
    if first:
        assert_close(scaled[..., :1], factor * out[..., :1], FEATURE_F64, "linear")


def test_each_atom_is_normalised_on_its_own_features_and_the_norm_is_odd():
    twin = _twin("full")
    x = _features(np.random.default_rng(RNG_SEED))
    out = twin({"I": x})
    changed = x.clone()
    changed[2] = -changed[2]
    moved = twin({"I": changed})
    keep = [a for a in range(N_ATOMS) if a != 2]
    assert torch.equal(moved[keep], out[keep])
    assert torch.equal(moved[2], -out[2])
