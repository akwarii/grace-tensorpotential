"""The ``MLPRadialFunction`` twin: logic against a numpy re-computation of the sheet, and the physics of a radial function.

Logic: the output against a numpy MLP written from the sheet (``x @ (w / sqrt(n_in))``, an activation after each hidden
layer, ``silu * 1.6759`` by default, the default ``[64, 64, 64]``), for every activation, no and several hidden layers,
``norm`` (population variance of the row, not centred, ``1e-5``), the chemical embeddings of the central atom and of the
neighbour (in that column order), the column layout ``R[b, l*l + l + m, n] = y[b, n * (lmax + 1) + l]``, the float32
casts and the typed errors. Physical values: a bond's radial functions depend on that bond only, relabelling the bonds
permutes them, the ``2 l + 1`` orders of a degree share one value, and a basis that vanishes (beyond the cutoff) gives
zero radial functions. The comparison with TensorFlow is ``test_mlp_radial_function_golden.py``.
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
    MLPRadialFunction,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_BONDS, NFUNC, N_ELEMENTS = 7, 5, 3

ACTIVATIONS = {
    None: lambda x: x / (1 + np.exp(-x)) * 1.6759,
    "tanh": np.tanh,
    "silu": lambda x: x / (1 + np.exp(-x)),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


def _spec(**options: Any) -> InstructionSpec:
    cls = COMPUTE + "MLPRadialFunction"
    base = {"basis": InstructionRef("B"), "n_rad_max": 3, "lmax": 2}
    return InstructionSpec(
        name="R",
        cls=cls,
        options={**DEFAULTS[cls], **base, "hidden_layers": [4, 5], **options},
        depends_on=(),
        index=0,
    )


def _twin(policy=F64, sizes=(None, None), **options: Any) -> MLPRadialFunction:
    twin = MLPRadialFunction(
        _spec(**options),
        NFUNC,
        policy,
        embedding_size_i=sizes[0],
        embedding_size_j=sizes[1],
    )
    rng = np.random.default_rng(RNG_SEED + 1)
    state = {
        k: torch.as_tensor(rng.normal(size=v.shape), dtype=policy.param)
        for k, v in twin.state_dict().items()
    }
    twin.load_state_dict(state, strict=True)
    return twin


def _inputs(rng, dtype=np.float64) -> dict[str, torch.Tensor]:
    return {
        "B": torch.as_tensor(rng.normal(size=(N_BONDS, NFUNC)).astype(dtype)),
        "mu_i": torch.as_tensor(rng.integers(0, N_ELEMENTS, N_BONDS)),
        "mu_j": torch.as_tensor(rng.integers(0, N_ELEMENTS, N_BONDS)),
    }


def _weights(twin: MLPRadialFunction) -> list[np.ndarray]:
    layers = sorted(k for k in twin.state_dict() if k.startswith("mlp."))
    return [twin.state_dict()[k].double().numpy() for k in layers]


def _oracle(twin, x, activation=None, gamma=None) -> np.ndarray:
    """The sheet's forward pass in numpy, ``[bonds, lm, n]``."""
    *hidden, last = _weights(twin)
    for w in hidden:
        x = ACTIVATIONS[activation](x @ (w / np.sqrt(w.shape[0])))
    y = x @ (last / np.sqrt(last.shape[0]))
    if gamma is not None:
        y = y / np.sqrt(y.var(axis=-1, keepdims=True) + 1e-5) * gamma
    n, lmax = twin.n_rad_max, twin.lmax
    out = np.empty((x.shape[0], (lmax + 1) ** 2, n))
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            for k in range(n):
                out[:, l * l + l + m, k] = y[:, k * (lmax + 1) + l]
    return out


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
@pytest.mark.parametrize("hidden", [[], [4], [4, 5]])
def test_the_output_is_the_mlp_of_the_sheet(activation, hidden):
    twin = _twin(activation=activation, hidden_layers=hidden)
    data = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(data)
    assert out.shape == (N_BONDS, 9, 3) and out.dtype == torch.float64
    expected = _oracle(twin, data["B"].numpy(), activation)
    assert_close(out, expected, FEATURE_F64, f"{activation} {hidden}")


def test_null_hidden_layers_mean_three_layers_of_64():
    twin = _twin(hidden_layers=None)
    shapes = [tuple(v.shape) for v in twin.state_dict().values()]
    assert shapes == [(NFUNC, 64), (64, 64), (64, 64), (64, 9)]


def test_the_weights_are_named_as_in_the_checkpoint():
    assert list(_twin().state_dict()) == [
        "mlp.layer0.w",
        "mlp.layer1.w",
        "mlp.layer2.w",
    ]
    assert set(_twin(norm=True).state_dict()) == {
        "gamma",
        "mlp.layer0.w",
        "mlp.layer1.w",
        "mlp.layer2.w",
    }


def test_norm_rescales_each_row_by_its_population_variance():
    twin = _twin(norm=True, activation="tanh")
    assert twin.gamma is not None and twin.gamma.shape == (1, 9)
    data = _inputs(np.random.default_rng(RNG_SEED))
    gamma = twin.gamma.detach().numpy()
    expected = _oracle(twin, data["B"].numpy(), "tanh", gamma)
    assert_close(twin(data), expected, FEATURE_F64, "norm")


@pytest.mark.parametrize("sides", [(2, None), (None, 3), (2, 3)])
def test_the_embeddings_of_the_central_atom_then_the_neighbour_are_appended(sides):
    refs = {
        f"chemical_embedding_{s}": InstructionRef(f"Z{s}") if n else None
        for s, n in zip("ij", sides, strict=True)
    }
    twin = _twin(sizes=sides, activation="tanh", **refs)
    rng = np.random.default_rng(RNG_SEED)
    data = _inputs(rng)
    columns = [data["B"].numpy()]
    for side, size in zip("ij", sides, strict=True):
        if size:
            table = rng.normal(size=(N_ELEMENTS, size))
            data[f"Z{side}"] = torch.as_tensor(table)
            columns.append(table[data[f"mu_{side}"].numpy()])
    expected = _oracle(twin, np.concatenate(columns, axis=1), "tanh")
    assert_close(twin(data), expected, FEATURE_F64, f"{sides}")


def test_a_float32_model_casts_the_float64_basis_and_embedding_down():
    twin = _twin(F32, sizes=(2, None), chemical_embedding_i=InstructionRef("Zi"))
    rng = np.random.default_rng(RNG_SEED)
    data = _inputs(rng)
    table = rng.normal(size=(N_ELEMENTS, 2)).astype(np.float32)
    data["Zi"] = torch.as_tensor(table)
    out = twin(data)
    assert out.dtype == torch.float32
    x = np.concatenate([data["B"].numpy(), table[data["mu_i"].numpy()]], axis=1)
    assert_close(out, _oracle(twin, x), FEATURE_F32, "float32")


def test_the_column_table_is_built_not_stored():
    twin = _twin(lmax=3)
    assert twin.get_buffer("l_tile").tolist() == [0] + [1] * 3 + [2] * 5 + [3] * 7
    assert "l_tile" not in twin.state_dict()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"basis": None}, "basis is null"),
        ({"n_rad_max": 0}, "n_rad_max must be at least 1"),
        ({"lmax": -1}, "lmax at least 0"),
        ({"activation": "relu"}, "unknown activation"),
    ],
)
def test_a_bad_option_is_refused(options, message):
    with pytest.raises(OptionValueError, match=message):
        _twin(**options)


def test_an_embedding_and_its_size_go_together():
    with pytest.raises(OptionValueError, match="chemical_embedding_i is None"):
        _twin(sizes=(2, None))
    with pytest.raises(OptionValueError, match="chemical_embedding_j is"):
        _twin(chemical_embedding_j=InstructionRef("Zj"))


def test_a_basis_of_another_width_is_refused():
    data = _inputs(np.random.default_rng(RNG_SEED))
    data["B"] = data["B"][:, :-1]
    with pytest.raises(ShapeError, match="width 5"):
        _twin()(data)


def test_an_element_index_must_be_an_integer():
    twin = _twin(sizes=(2, None), chemical_embedding_i=InstructionRef("Zi"))
    data = _inputs(np.random.default_rng(RNG_SEED))
    data["Zi"] = torch.zeros((N_ELEMENTS, 2), dtype=torch.float64)
    data["mu_i"] = data["mu_i"].double()
    with pytest.raises(DtypeError, match="int32 or int64"):
        twin(data)


# ------------------------------------------------------------------ physical values


def test_each_bond_depends_on_its_own_basis_only_and_relabelling_permutes():
    twin = _twin(activation="tanh")
    data = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(data)
    changed = dict(data, B=data["B"].clone())
    changed["B"][3] += 1.0
    moved = twin(changed)
    assert not torch.equal(moved[3], out[3])
    keep = [b for b in range(N_BONDS) if b != 3]
    assert torch.equal(moved[keep], out[keep])
    order = torch.as_tensor(np.random.default_rng(RNG_SEED + 2).permutation(N_BONDS))
    # the matmul may round a row differently at another position
    relabelled = twin(dict(data, B=data["B"][order]))
    assert_close(relabelled, out[order].detach().numpy(), FEATURE_F64, "relabelled")


def test_the_orders_of_a_degree_share_one_radial_value():
    out = _twin(lmax=3)(_inputs(np.random.default_rng(RNG_SEED)))
    for l in range(4):
        block = out[:, l * l : (l + 1) ** 2, :]
        assert torch.equal(block, block[:, :1, :].expand_as(block))


@pytest.mark.parametrize("activation", [None, "tanh", "silu"])
def test_a_vanishing_basis_gives_vanishing_radial_functions(activation):
    data = _inputs(np.random.default_rng(RNG_SEED))
    data["B"] = torch.zeros_like(data["B"])
    assert torch.count_nonzero(_twin(activation=activation)(data)) == 0
