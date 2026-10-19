"""The ``MLPRadialFunction_v2`` twin: logic against a numpy re-computation of the sheet, and the physics of a radial function.

Logic: the output against a numpy MLP written from the sheet (``x @ (w * norm)`` with ``norm = 1/sqrt(n_in)``, or 1
without ``normalize``; one activation per hidden layer, plain ``silu`` by default, a string repeated), the default
``[64, 64]``, the gate of the chemical embedding with each flag (``z[mu_j]`` for ``embed_j``, ``tanh`` of the product
with ``embed_i``, no gate without either), the embedding transform always normalised, the float32 casts, the weight
names and the typed errors. Physical values: a bond's radial functions depend on that bond only (and, with the gate, on
its two elements), relabelling the bonds permutes them, the ``2 l + 1`` orders of a degree share one value, and a basis
that vanishes (beyond the cutoff) gives zero radial functions. The comparison with TensorFlow is
``test_mlp_radial_function_golden.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
)
from tensorpotential.torch_backend.instructions.mlp_radial_function import (
    MLPRadialFunction_v2,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close
from tests_torch.init_only import assert_option_is_ignored

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_BONDS, NFUNC, N_ELEMENTS, EMB = 7, 5, 3, 4

ACTIVATIONS = {
    "silu": lambda x: x / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


def _spec(**options: Any) -> InstructionSpec:
    cls = COMPUTE + "MLPRadialFunction_v2"
    base = {
        "basis": InstructionRef("B"),
        "n_rad_max": 3,
        "lmax": 2,
        "hidden_layers": [4, 5],
    }
    return InstructionSpec(
        name="R",
        cls=cls,
        options={**DEFAULTS[cls], **base, **options},
        depends_on=(),
        index=0,
    )


def _twin(policy=F64, embedding_size=None, **options: Any) -> MLPRadialFunction_v2:
    twin = MLPRadialFunction_v2(
        _spec(**options), NFUNC, policy, embedding_size=embedding_size
    )
    rng = np.random.default_rng(RNG_SEED + 1)
    state = {
        k: torch.as_tensor(rng.normal(size=v.shape), dtype=policy.param)
        for k, v in twin.state_dict().items()
    }
    twin.load_state_dict(state, strict=True)
    return twin


def _gated(policy=F64, **options: Any) -> MLPRadialFunction_v2:
    return _twin(policy, EMB, chem_embedding=InstructionRef("Z"), **options)


def _inputs(rng, dtype=np.float64) -> dict[str, torch.Tensor]:
    return {
        "B": torch.as_tensor(rng.normal(size=(N_BONDS, NFUNC)).astype(dtype)),
        "Z": torch.as_tensor(rng.normal(size=(N_ELEMENTS, EMB)).astype(dtype)),
        "mu_i": torch.as_tensor(rng.integers(0, N_ELEMENTS, N_BONDS)),
        "mu_j": torch.as_tensor(rng.integers(0, N_ELEMENTS, N_BONDS)),
    }


def _layout(y: np.ndarray, n_rad_max: int, lmax: int) -> np.ndarray:
    """``R[b, l*l + l + m, n] = y[b, n * (lmax + 1) + l]``."""
    out = np.empty((y.shape[0], (lmax + 1) ** 2, n_rad_max))
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            for n in range(n_rad_max):
                out[:, l * l + l + m, n] = y[:, n * (lmax + 1) + l]
    return out


def _oracle(twin, data, activations, normalize=True, gate=None) -> np.ndarray:
    """The sheet's forward pass in numpy, ``[bonds, lm, n]``."""
    state = {k: v.double().numpy() for k, v in twin.state_dict().items()}
    weights = [state[f"layers.{i}.w"] for i in range(len(twin.layers))]
    x = data["B"].double().numpy()
    for w, act in zip(weights[:-1], activations, strict=True):
        x = ACTIVATIONS[act](x @ (w / np.sqrt(w.shape[0]) if normalize else w))
    last = weights[-1]
    y = x @ (last / np.sqrt(last.shape[0]) if normalize else last)
    if gate is not None:
        w = state["embed_transform.w"]
        z = data["Z"].double().numpy() @ (w / np.sqrt(w.shape[0]))
        embed_i, embed_j = gate
        g: Any = 1.0
        if embed_j:
            g = z[data["mu_j"].numpy()]
        if embed_i:
            g = np.tanh(g * z[data["mu_i"].numpy()])
        y = y * g
    return _layout(y, twin.n_rad_max, twin.lmax)


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(
    ("activation", "names"),
    [
        (None, ["silu", "silu"]),
        ("tanh", ["tanh", "tanh"]),
        (["sigmoid", "silu"], ["sigmoid", "silu"]),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_the_output_is_the_mlp_of_the_sheet(activation, names, normalize):
    twin = _twin(activation=activation, normalize=normalize)
    data = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(data)
    assert out.shape == (N_BONDS, 9, 3) and out.dtype == torch.float64
    expected = _oracle(twin, data, names, normalize)
    assert_close(out, expected, FEATURE_F64, f"{activation} {normalize}")


def test_no_hidden_layer_is_one_linear_map():
    twin = _twin(hidden_layers=[])
    data = _inputs(np.random.default_rng(RNG_SEED))
    assert_close(twin(data), _oracle(twin, data, []), FEATURE_F64, "no hidden")


def test_null_hidden_layers_mean_two_layers_of_64_named_as_in_the_checkpoint():
    shapes = {
        k: tuple(v.shape) for k, v in _twin(hidden_layers=None).state_dict().items()
    }
    assert shapes == {
        "layers.0.w": (NFUNC, 64),
        "layers.1.w": (64, 64),
        "layers.2.w": (64, 9),
    }


@pytest.mark.parametrize(
    "gate", [(False, True), (True, True), (True, False), (False, False)]
)
def test_the_gate_follows_each_flag_with_the_tanh_only_under_embed_i(gate):
    embed_i, embed_j = gate
    twin = _gated(embed_i=embed_i, embed_j=embed_j)
    data = _inputs(np.random.default_rng(RNG_SEED))
    expected = _oracle(twin, data, ["silu", "silu"], gate=gate)
    assert_close(twin(data), expected, FEATURE_F64, f"gate {gate}")


def test_the_embedding_transform_is_normalised_whatever_normalize_says():
    twin = _gated(normalize=False)
    assert twin.embed_transform is not None
    assert twin.embed_transform.norm == pytest.approx(1 / np.sqrt(EMB))
    assert twin.layers[0].norm == 1.0


def test_a_float32_model_casts_the_basis_down_and_gates_in_float32():
    twin = _gated(F32, embed_i=True)
    data = _inputs(np.random.default_rng(RNG_SEED))
    data["Z"] = data["Z"].float()
    out = twin(data)
    assert out.dtype == torch.float32
    expected = _oracle(twin, data, ["silu", "silu"], gate=(True, True))
    assert_close(out, expected, FEATURE_F32, "float32")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"basis": None}, "basis is null"),
        ({"n_rad_max": 0}, "n_rad_max must be at least 1"),
        ({"lmax": -1}, "lmax at least 0"),
        ({"activation": "relu"}, "one of tanh, silu, sigmoid per hidden layer"),
        ({"activation": ["silu"]}, r"per hidden layer \(2\)"),
        ({"activation": "silu_n2norm"}, "one of tanh, silu, sigmoid"),
    ],
)
def test_a_bad_option_is_refused(options, message):
    with pytest.raises(OptionValueError, match=message):
        _twin(**options)


def test_an_embedding_and_its_size_go_together():
    with pytest.raises(OptionValueError, match="chem_embedding is None"):
        _twin(embedding_size=EMB)
    with pytest.raises(
        OptionValueError, match=r"chem_embedding is .*embedding size of None"
    ):
        _twin(chem_embedding=InstructionRef("Z"))


def test_a_basis_of_another_width_and_a_floating_element_index_are_refused():
    data = _inputs(np.random.default_rng(RNG_SEED))
    with pytest.raises(ShapeError, match="width 5"):
        _twin()(dict(data, B=data["B"][:, :-1]))
    for key, flags in (("mu_j", {}), ("mu_i", {"embed_i": True})):
        with pytest.raises(DtypeError, match="int32 or int64"):
            _gated(**flags)(dict(data, **{key: data[key].double()}))


# ------------------------------------------------------------------ physical values


def test_each_bond_depends_on_its_own_basis_and_elements_and_relabelling_permutes():
    twin = _gated(embed_i=True)
    data = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(data)
    for key, change in (
        ("B", lambda t: t + 1.0),
        ("mu_j", lambda t: (t + 1) % N_ELEMENTS),
        ("mu_i", lambda t: (t + 1) % N_ELEMENTS),
    ):
        changed = dict(data, **{key: data[key].clone()})
        changed[key][3] = change(changed[key][3])
        moved = twin(changed)
        keep = [b for b in range(N_BONDS) if b != 3]
        assert torch.equal(moved[keep], out[keep]) and not torch.equal(moved[3], out[3])
    order = torch.as_tensor(np.random.default_rng(RNG_SEED + 2).permutation(N_BONDS))
    relabelled = {k: (v if k == "Z" else v[order]) for k, v in data.items()}
    assert_close(
        twin(relabelled), out[order].detach().numpy(), FEATURE_F64, "relabelled"
    )


def test_the_orders_of_a_degree_share_one_radial_value():
    out = _gated(lmax=3)(_inputs(np.random.default_rng(RNG_SEED)))
    for l in range(4):
        block = out[:, l * l : (l + 1) ** 2, :]
        assert torch.equal(block, block[:, :1, :].expand_as(block))


@pytest.mark.parametrize("activation", ["silu", "tanh"])
def test_a_vanishing_basis_gives_vanishing_radial_functions(activation):
    data = _inputs(np.random.default_rng(RNG_SEED))
    data["B"] = torch.zeros_like(data["B"])
    assert torch.count_nonzero(_gated(activation=activation)(data)) == 0


@pytest.mark.parametrize("normalize", [True, False])
def test_init_type_changes_neither_the_state_nor_the_output(normalize):
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    assert_option_is_ignored(
        lambda **o: _gated(normalize=normalize, **o),
        COMPUTE + "MLPRadialFunction_v2",
        "init_type",
        inputs,
    )
