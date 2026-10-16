"""The dense blocks against numpy references written from the spec sheets, and against the TF goldens.

Logic: the linear layer, the MLP and the three RMS norms against numpy re-computations of the formulas of the sheets
(``MLPRadialFunction``, ``MLPRadialFunction_v2``, ``LinMLPOut2ScalarTarget``, ``InvariantLayerRMSNorm``), with the
epsilons written here from the sheets, not imported; dtypes, shapes, errors, gradients, the dtype audit.
Physical values: the radial MLPs, the invariant RMS norms and the energy readouts of every tiny anchor, rebuilt from
the stored weights and TF's stored inputs, against TF's stored outputs.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OpArgumentError,
    ShapeError,
)
from tensorpotential.torch_backend.ops import dense
from tensorpotential.torch_backend.ops.precision import DtypePolicy, precision_scope
from tests_torch.dtype_audit import DtypeAudit
from tests_torch.golden import GOLDEN, TINY_F32, TINY_F64, cases, load
from tests_torch.golden.tolerances import (
    FEATURE_F32,
    FEATURE_F64,
    assert_close,
    row_for,
)
from tests_torch.layout import to_internal

RNG_SEED = 20261016
F64 = DtypePolicy(torch.float64)
F32 = DtypePolicy(torch.float32)
ROWS = [(F64, FEATURE_F64), (F32, FEATURE_F32)]

# the epsilons and the factor, as the spec sheets give them (section 4 of each)
SHEET_VARIANCE_EPS = 1e-5  # MLPRadialFunction
SHEET_SCALAR_RMS_EPS = 1e-16  # LinMLPOut2ScalarTarget (scalar_rms_ln)
SHEET_CHANNEL_RMS_EPS = 1e-10  # InvariantLayerRMSNorm
SHEET_SILU_FACTOR = 1.6759


def _silu(x):
    return x / (1.0 + np.exp(-x))


NP_ACTIVATIONS = {
    "tanh": np.tanh,
    "silu": _silu,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu_n2norm": lambda x: _silu(x) * SHEET_SILU_FACTOR,
}


def _dtype_name(policy):
    """``"float32"`` or ``"float64"``: the numpy dtype and the dtype of the tolerance rows."""
    return str(policy.param).removeprefix("torch.")


def _np_dtype(policy):
    return np.dtype(_dtype_name(policy)).type


def _np_mlp(x, weights, activations, *, normalize=True):
    """Compute the forward pass of the sheets: ``act(h @ (w / sqrt(n_in)))`` on hidden layers, none on the last."""
    h = x.astype(weights[0].dtype)
    for i, w in enumerate(weights):
        h = h @ (w / np.sqrt(w.shape[0]) if normalize else w)
        if i < len(activations):
            h = NP_ACTIVATIONS[activations[i]](h)
    return h


def _random_weights(rng, sizes, policy):
    return [
        rng.normal(size=(a, b)).astype(_np_dtype(policy))
        for a, b in itertools.pairwise(sizes)
    ]


def _load(mlp, weights):
    mlp.load_state_dict({
        f"layer{i}.w": torch.as_tensor(w) for i, w in enumerate(weights)
    })
    return mlp


def _rows_with_small_values(rng, shape, tiny=1e-9):
    """Draw normal rows, then set one row to ``tiny`` and one to zeros, where the epsilon decides the result."""
    x = rng.normal(size=shape)
    x[0] = tiny
    x[1] = 0.0
    return x


# ------------------------------------------------------------------ linear layer


@pytest.mark.parametrize(("policy", "row"), ROWS)
@pytest.mark.parametrize("shape", [(5,), (11, 5), (3, 4, 5)])
def test_linear_is_the_input_times_the_weight_over_sqrt_n_in(policy, row, shape):
    rng = np.random.default_rng(RNG_SEED)
    x, (w,) = rng.normal(size=shape), _random_weights(rng, [5, 3], policy)
    layer = dense.Linear(5, 3, policy)
    layer.load_state_dict({"w": torch.as_tensor(w)})
    out = layer(torch.as_tensor(x))
    assert out.dtype == policy.param
    assert out.shape == (*shape[:-1], 3)
    assert_close(out, x.astype(w.dtype) @ (w / np.sqrt(5)), row)


def test_linear_without_normalize_uses_the_weight_as_it_is():
    layer = dense.Linear(2, 1, F64, normalize=False)
    layer.load_state_dict({"w": torch.tensor([[3.0], [-1.0]], dtype=torch.float64)})
    out = layer(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    assert out.tolist() == [[2.0]]


def test_the_norm_is_not_a_weight_and_the_weight_is_not_changed():
    layer = dense.Linear(4, 2, F32)
    assert list(layer.state_dict()) == ["w"]
    assert layer.w.dtype == torch.float32 and torch.all(layer.w == 0)
    stored = torch.ones((4, 2), dtype=torch.float32)
    layer.load_state_dict({"w": stored})
    layer(torch.ones((1, 4), dtype=torch.float64))
    assert torch.equal(layer.w, stored)


@pytest.mark.parametrize(("n_in", "n_out"), [(0, 3), (3, 0)])
def test_a_linear_layer_without_width_is_refused(n_in, n_out):
    with pytest.raises(ShapeError, match="at least 1"):
        dense.Linear(n_in, n_out, F64)


@pytest.mark.parametrize("shape", [(), (2, 4), (4, 3, 2)])
def test_an_input_of_another_width_is_refused(shape):
    with pytest.raises(ShapeError, match="width 3"):
        dense.Linear(3, 1, F64)(torch.zeros(shape, dtype=torch.float64))


# ------------------------------------------------------------------ MLP


@pytest.mark.parametrize(("policy", "row"), ROWS)
@pytest.mark.parametrize("activation", ["tanh", "silu", "sigmoid", None])
def test_the_fully_connected_mlp_is_the_forward_pass_of_the_sheets(
    policy, row, activation
):
    rng = np.random.default_rng(RNG_SEED)
    sizes = [8, 6, 7, 20]
    weights = _random_weights(rng, sizes, policy)
    mlp = _load(dense.MLP.fully_connected(sizes, activation, policy), weights)
    x = rng.normal(size=(13, 8))
    out = mlp(torch.as_tensor(x))
    name = activation or "silu_n2norm"
    assert out.dtype == policy.param and out.shape == (13, 20)
    assert_close(out, _np_mlp(x, weights, [name, name]), row)


@pytest.mark.parametrize("normalize", [True, False])
def test_an_mlp_with_one_activation_per_layer_and_its_normalize(normalize):
    rng = np.random.default_rng(RNG_SEED)
    sizes = [10, 6, 7, 25]
    weights = _random_weights(rng, sizes, F64)
    mlp = dense.MLP(sizes, ["silu", "tanh"], F64, normalize=normalize)
    x = rng.normal(size=(9, 10))
    out = _load(mlp, weights)(torch.as_tensor(x))
    assert_close(
        out, _np_mlp(x, weights, ["silu", "tanh"], normalize=normalize), FEATURE_F64
    )


def test_the_hidden_output_is_the_input_of_the_last_layer():
    rng = np.random.default_rng(RNG_SEED)
    weights = _random_weights(rng, [4, 9, 1], F64)
    mlp = _load(dense.MLP.fully_connected([4, 9, 1], "tanh", F64), weights)
    x = rng.normal(size=(6, 4))
    out, hidden = mlp.forward_with_hidden(torch.as_tensor(x))
    assert_close(hidden, _np_mlp(x, weights[:1], ["tanh"]), FEATURE_F64)
    assert_close(out, _np_mlp(x, weights, ["tanh"]), FEATURE_F64)


def test_an_mlp_without_hidden_layer_is_one_linear_layer_and_hides_its_input():
    mlp = dense.MLP.fully_connected([3, 2], None, F64)
    assert list(mlp.state_dict()) == ["layer0.w"]
    x = torch.ones((2, 3), dtype=torch.float64)
    out, hidden = mlp.forward_with_hidden(x)
    assert hidden is x and out.shape == (2, 2)


def test_the_weights_of_the_mlp_have_the_keys_of_the_checkpoint():
    mlp = dense.MLP.fully_connected([8, 6, 7, 20], "tanh", F32)
    assert {k: tuple(v.shape) for k, v in mlp.state_dict().items()} == {
        "layer0.w": (8, 6),
        "layer1.w": (6, 7),
        "layer2.w": (7, 20),
    }


def test_an_unknown_activation_is_refused_with_its_name():
    with pytest.raises(OpArgumentError, match=r"'relu'.*tanh, silu, sigmoid"):
        dense.MLP.fully_connected([2, 3, 1], "relu", F64)


@pytest.mark.parametrize(
    ("sizes", "activations"), [([4], []), ([4, 1], ["tanh"]), ([4, 3, 1], [])]
)
def test_widths_and_activations_that_do_not_fit_are_refused(sizes, activations):
    with pytest.raises(ShapeError, match="one activation per hidden layer"):
        dense.MLP(sizes, activations, F64)


# ------------------------------------------------------------------ RMS norms


@pytest.mark.parametrize(("policy", "row"), ROWS)
def test_the_variance_norm_rescales_by_the_row_variance_without_centring(policy, row):
    rng = np.random.default_rng(RNG_SEED)
    dt = _np_dtype(policy)
    y = _rows_with_small_values(rng, (7, 12)).astype(dt)
    gamma = rng.normal(size=(1, 12)).astype(dt)
    out = dense.variance_norm(torch.as_tensor(y), torch.as_tensor(gamma))
    expected = (
        y / np.sqrt(y.var(axis=-1, keepdims=True) + dt(SHEET_VARIANCE_EPS)) * gamma
    )
    assert out.dtype == policy.param
    assert_close(out, expected, row)


@pytest.mark.parametrize(("policy", "row"), ROWS)
def test_the_scalar_rms_norm_is_scalar_rms_ln(policy, row):
    rng = np.random.default_rng(RNG_SEED)
    dt = _np_dtype(policy)
    # with the epsilon 1e-16, a row of 1e-12 is rescaled to 1e-12 / sqrt(1e-24 + 1e-16) = 1e-4, not 1
    x = _rows_with_small_values(rng, (7, 12), tiny=1e-12).astype(dt)
    scale = rng.normal(size=(1, 12)).astype(dt)
    out = dense.scalar_rms_norm(torch.as_tensor(x), torch.as_tensor(scale))
    mean_square = (x**2).mean(axis=-1, keepdims=True)
    expected = x * scale / np.sqrt(mean_square + dt(SHEET_SCALAR_RMS_EPS))
    assert out.dtype == policy.param
    assert_close(out, expected, row)


@pytest.mark.parametrize(("policy", "row"), ROWS)
def test_the_channel_rms_norm_is_that_of_invariant_layer_rms_norm(policy, row):
    rng = np.random.default_rng(RNG_SEED)
    dt = _np_dtype(policy)
    x = _rows_with_small_values(rng, (7, 1, 17)).astype(dt)
    scale = rng.normal(size=(1, 1, 17)).astype(dt)
    out = dense.channel_rms_norm(torch.as_tensor(x), torch.as_tensor(scale))
    mean_square = (x**2).mean(axis=-1, keepdims=True)
    expected = x / np.sqrt(mean_square + dt(SHEET_CHANNEL_RMS_EPS)) * scale
    assert out.dtype == policy.param
    assert_close(out, expected, row)


def test_the_epsilons_are_those_of_the_sheets():
    assert dense.VARIANCE_NORM_EPSILON == SHEET_VARIANCE_EPS
    assert dense.SCALAR_RMS_EPSILON == SHEET_SCALAR_RMS_EPS
    assert dense.CHANNEL_RMS_EPSILON == SHEET_CHANNEL_RMS_EPS
    assert dense.SILU_N2NORM_FACTOR == SHEET_SILU_FACTOR


def test_a_float32_scale_meets_float64_data_as_tf_casts_it():
    """The epsilon is made in the parameter dtype and cast to the data dtype: float32(1e-10), not 1e-10."""
    x = torch.full((1, 1, 1), 1e-6, dtype=torch.float64)
    scale = torch.ones((1, 1, 1), dtype=torch.float32)
    out = dense.channel_rms_norm(x, scale)
    eps = float(np.float32(SHEET_CHANNEL_RMS_EPS))
    assert out.dtype == torch.float64
    assert out.item() == pytest.approx(1e-6 / np.sqrt(1e-12 + eps), rel=1e-15, abs=0)
    gamma = torch.ones((1, 1), dtype=torch.float32)
    assert dense.variance_norm(x[0], gamma).dtype == torch.float64


def test_the_scalar_rms_norm_refuses_mixed_dtypes():
    with pytest.raises(DtypeError, match="one dtype"):
        dense.scalar_rms_norm(
            torch.ones((2, 3), dtype=torch.float64),
            torch.ones((1, 3), dtype=torch.float32),
        )


# ------------------------------------------------------------------ gradients and the dtype audit


def test_the_blocks_pass_gradcheck():
    rng = np.random.default_rng(RNG_SEED)
    mlp = _load(
        dense.MLP.fully_connected([3, 4, 2], None, F64),
        _random_weights(rng, [3, 4, 2], F64),
    )
    x = torch.as_tensor(rng.normal(size=(5, 1, 3))).requires_grad_()
    scale = torch.as_tensor(rng.normal(size=(1, 1, 3))).requires_grad_()
    assert torch.autograd.gradcheck(mlp, (x,))
    assert torch.autograd.gradcheck(dense.variance_norm, (x, scale[0]))
    assert torch.autograd.gradcheck(dense.scalar_rms_norm, (x, scale))
    assert torch.autograd.gradcheck(dense.channel_rms_norm, (x, scale))


def test_the_blocks_mix_no_dtypes_and_read_nothing_on_the_host():
    mlp = dense.MLP.fully_connected([3, 4, 2], None, F32)
    basis = torch.linspace(0.1, 1.0, 15, dtype=torch.float64).reshape(5, 3)
    bonds = basis.requires_grad_()
    scale = torch.ones((1, 1, 2), dtype=torch.float32)
    with precision_scope(), DtypeAudit() as audit:
        y = mlp(bonds)
        y = dense.variance_norm(y, scale[0])
        y = dense.scalar_rms_norm(y, scale[0])
        y = dense.channel_rms_norm(y[:, None, :].to(torch.float64), scale)
        (grad,) = torch.autograd.grad(y.sum(), bonds)
    audit.assert_clean()
    assert grad.dtype == torch.float64


# ------------------------------------------------------------------ physical values: TF goldens

ANCHORS = TINY_F64 + TINY_F32
MLP_PREFIXES = ("mlp/layer", "layers/")
"""Where the checkpoint keeps the weights of a ``FullyConnectedMLP`` and of the ``Linear`` list of the v2 radial MLP."""


def _setting(name):
    manifest = json.loads((GOLDEN / f"{name}.json").read_text())
    weights = load(f"{name}.weights")
    policy = DtypePolicy.from_weights(w.dtype for w in weights.values())
    return manifest["instructions"], weights, policy


def _mlp_weights(weights, instruction):
    """Return the layer weights of an MLP, in order, and its widths ``[n_in, ..., n_out]``."""
    (prefix,) = [p for p in MLP_PREFIXES if f"{instruction}/{p}0/w" in weights]
    ws = []
    while f"{instruction}/{prefix}{len(ws)}/w" in weights:
        ws.append(weights[f"{instruction}/{prefix}{len(ws)}/w"])
    return ws, [ws[0].shape[0]] + [w.shape[1] for w in ws]


def _instances(instructions, cls):
    return [n for n, s in instructions.items() if s["class"] == cls]


@pytest.mark.golden
@pytest.mark.parametrize("name", ANCHORS)
def test_the_radial_mlps_of_tf_are_rebuilt(name):
    instructions, weights, policy = _setting(name)
    dtype = _dtype_name(policy)
    arrays = load(name)
    radial = _instances(instructions, "MLPRadialFunction") + _instances(
        instructions, "MLPRadialFunction_v2"
    )
    assert radial
    for instruction in radial:
        args = instructions[instruction]["init_args"]
        ws, sizes = _mlp_weights(weights, instruction)
        if instructions[instruction]["class"] == "MLPRadialFunction":
            assert not args["norm"]
            mlp = dense.MLP.fully_connected(sizes, args["activation"], policy)
        else:
            assert args["chem_embedding"] is None
            mlp = dense.MLP(
                sizes, args["activation"], policy, normalize=args["normalize"]
            )
        _load(mlp, ws)
        n_rad, lmax = args["n_rad_max"], args["lmax"]
        l_tile = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
        for case in cases(arrays):
            basis = torch.as_tensor(arrays[f"{case}/ins/{args['basis']['name']}"])
            out = mlp(basis).reshape(-1, n_rad, lmax + 1)[..., l_tile]
            expected = arrays[f"{case}/ins/{instruction}"]
            assert out.dtype == policy.param
            assert_close(
                out, expected, row_for(dtype, "feature"), f"{name} {case} {instruction}"
            )


@pytest.mark.golden
@pytest.mark.parametrize("name", ["large_tiny_f64", "large_tiny_f32"])
def test_the_invariant_rms_norms_of_tf_are_rebuilt(name):
    instructions, weights, policy = _setting(name)
    dtype = _dtype_name(policy)
    arrays = load(name)
    norms = _instances(instructions, "InvariantLayerRMSNorm")
    assert {instructions[n]["init_args"]["type"] for n in norms} == {
        "full",
        "only_nonlin",
    }
    for instruction in norms:
        args = instructions[instruction]["init_args"]
        scale = torch.as_tensor(to_internal(weights[f"{instruction}/scale"], "default"))
        for case in cases(arrays):
            x = torch.as_tensor(
                to_internal(arrays[f"{case}/ins/{args['inpt']['name']}"], "default")
            )
            if args["type"] == "full":
                out = dense.channel_rms_norm(x, scale)
            else:
                out = torch.cat(
                    [x[..., :1], dense.channel_rms_norm(x[..., 1:], scale)], dim=-1
                )
            expected = to_internal(arrays[f"{case}/ins/{instruction}"], "default")
            assert_close(
                out, expected, row_for(dtype, "feature"), f"{name} {case} {instruction}"
            )


@pytest.mark.golden
@pytest.mark.parametrize("name", ANCHORS)
def test_the_energy_readouts_of_tf_are_rebuilt(name):
    instructions, weights, policy = _setting(name)
    dtype = _dtype_name(policy)
    arrays = load(name)
    (instruction,) = _instances(instructions, "LinMLPOut2ScalarTarget")
    args = instructions[instruction]["init_args"]
    ws, sizes = _mlp_weights(weights, instruction)
    mlp = _load(dense.MLP.fully_connected(sizes, args["activation"], policy), ws)
    for case in cases(arrays):
        scalars = [arrays[f"{case}/ins/{o['name']}"][:, :, 0] for o in args["origin"]]
        transformed = torch.as_tensor(sum(f[:, 1:] for f in scalars))
        lin = torch.as_tensor(sum(f[:, :1] for f in scalars))
        if args["normalize"] == "layer":
            scale = torch.as_tensor(weights[f"{instruction}/scale"])
            transformed = dense.scalar_rms_norm(transformed, scale)
        before = torch.as_tensor(arrays[f"{case}/out_before/{instruction}"])
        out = before + (mlp(transformed) + lin)
        assert_close(
            out,
            arrays[f"{case}/out_after/{instruction}"],
            row_for(dtype, "energy"),
            f"{name} {case}",
        )
