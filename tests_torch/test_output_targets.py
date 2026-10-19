"""The output twins: logic against numpy re-computations of the sheets, and the physical properties of an energy readout.

Logic: ``CreateOutputTarget`` (the value, null as 0, the dtype); ``LinMLPOut2ScalarTarget`` against a numpy MLP written
from the sheet (``x @ (w / sqrt(n_in))``, ``silu * 1.6759`` by default, the layer normalisation with ``1e-16``) for one
and two origins, every activation, no and several hidden layers, ``n_out > 1`` and the hidden features of
``return_hidden_target``; ``ConstantScaleShiftTarget`` with the scale alone, a constant shift, a per-element map and an
embedding shift; ``TrainableShiftTarget`` against a numpy gather of its table, in the dtype of the target; the typed
errors. Physical values: each atom's energy depends on that atom's features only
(locality), relabelling the atoms permutes the energies, the learned shifts add one reference energy per atom of
each element to the total, and a float32 model rounds the omat scale to the float32 value
TF measured (spec sheet, section 4). The comparison with TensorFlow is ``test_output_targets_golden.py``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
)
from tensorpotential.torch_backend.instructions.output_targets import (
    ConstantScaleShiftTarget,
    CreateOutputTarget,
    LinMLPOut2ScalarTarget,
    TrainableShiftTarget,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import DEFAULTS, OUTPUT
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_ATOMS, N_CH = 5, 6
OMAT_SCALE = 1.892985414710868
OMAT_SCALE_F32 = (
    1.892985463142395  # measured in TF with float32 parameters (sheet, section 4)
)


def _spec(cls: str, name: str, **options: Any) -> InstructionSpec:
    return InstructionSpec(
        name=name,
        cls=OUTPUT + cls,
        options={**DEFAULTS[OUTPUT + cls], **options},
        depends_on=(),
        index=0,
    )


# ------------------------------------------------------------------ CreateOutputTarget


@pytest.mark.parametrize(("given", "value"), [(0.0, 0.0), (-1.25, -1.25), (None, 0.0)])
@pytest.mark.parametrize("policy", [F64, F32])
def test_the_target_starts_from_its_initial_value_in_the_parameter_dtype(
    given, value, policy
):
    twin = CreateOutputTarget(
        _spec("CreateOutputTarget", "atomic_energy", initial_value=given), policy
    )
    out = twin({})
    assert out.shape == () and out.dtype == policy.param and float(out) == value
    assert twin.state_dict() == {}


# ------------------------------------------------------------------ LinMLPOut2ScalarTarget


def _readout(
    policy=F64, origins=("I",), channels=None, **options: Any
) -> LinMLPOut2ScalarTarget:
    spec = _spec(
        "LinMLPOut2ScalarTarget",
        "readout",
        origin=[InstructionRef(o) for o in origins],
        target=InstructionRef("atomic_energy"),
        **{"hidden_layers": [4], **options},
    )
    twin = LinMLPOut2ScalarTarget(spec, channels or [N_CH] * len(origins), policy)
    rng = np.random.default_rng(RNG_SEED + 1)
    state = {
        k: torch.as_tensor(rng.normal(size=v.shape), dtype=policy.param)
        for k, v in twin.state_dict().items()
    }
    twin.load_state_dict(state, strict=True)
    return twin


def _features(rng, origins=("I",), dtype=np.float64, lm=1) -> dict[str, torch.Tensor]:
    inputs = {
        o: torch.as_tensor(rng.normal(size=(N_ATOMS, lm, N_CH)).astype(dtype))
        for o in origins
    }
    inputs["atomic_energy"] = torch.as_tensor(
        rng.normal(size=(N_ATOMS, 1)).astype(dtype)
    )
    return inputs


ACTIVATIONS = {
    None: lambda x: x / (1 + np.exp(-x)) * 1.6759,
    "silu": lambda x: x / (1 + np.exp(-x)),
    "tanh": np.tanh,
}


def _readout_reference(twin, inputs, *, activation=None, normalize=None):
    """Sheet section 5 in numpy, from the twin's weights (the weights are data; the formula is the sheet's)."""
    state = {k: v.double().numpy() for k, v in twin.state_dict().items()}
    origins = [ref.name for ref in twin.spec.options["origin"]]
    scalars = [inputs[o].double().numpy()[:, 0, :] for o in origins]
    lin = sum(s[:, :1] for s in scalars)
    x = sum(s[:, 1:] for s in scalars)
    if normalize == "layer":
        x = x * state["scale"] / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-16)
    layers = sorted(k for k in state if k.startswith("mlp."))
    for i, key in enumerate(layers):
        w = state[key]
        h = x
        x = x @ (w / np.sqrt(w.shape[0]))
        if i < len(layers) - 1:
            x = ACTIVATIONS[activation](x)
    target = inputs["atomic_energy"].double().numpy()
    return target + (x + lin), np.concatenate([lin, h], axis=1)


@pytest.mark.parametrize("activation", [None, "silu", "tanh"])
@pytest.mark.parametrize("normalize", [None, "layer"])
def test_the_readout_is_the_formula_of_the_sheet(activation, normalize):
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(activation=activation, normalize=normalize)
    inputs = _features(rng)
    expected, _ = _readout_reference(
        twin, inputs, activation=activation, normalize=normalize
    )
    out = twin(inputs)
    assert out.shape == (N_ATOMS, 1) and out.dtype == torch.float64
    assert_close(out, expected, FEATURE_F64)


def test_two_origins_are_added_channel_by_channel_before_the_mlp():
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(origins=("I0", "I1"), normalize="layer", activation="tanh")
    inputs = _features(rng, origins=("I0", "I1"))
    expected, _ = _readout_reference(twin, inputs, activation="tanh", normalize="layer")
    assert_close(twin(inputs), expected, FEATURE_F64)


@pytest.mark.parametrize(("hidden_layers", "n_out"), [([], 1), ([5, 3], 1), ([4], 2)])
def test_any_depth_and_output_width(hidden_layers, n_out):
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(hidden_layers=hidden_layers, n_out=n_out)
    # the architecture from the options, not from the twin
    widths = [N_CH - 1, *hidden_layers, n_out]
    shapes = [tuple(v.shape) for v in twin.state_dict().values()]
    assert shapes == list(itertools.pairwise(widths))
    inputs = _features(rng)
    expected, hidden = _readout_reference(twin, inputs)
    out, out_hidden = twin.forward_with_hidden(inputs)
    assert out.shape == (N_ATOMS, n_out)
    assert_close(out, expected, FEATURE_F64)
    assert_close(out_hidden, hidden, FEATURE_F64)


def test_null_hidden_layers_mean_one_layer_of_32():
    twin = _readout(hidden_layers=None)
    assert {k: tuple(v.shape) for k, v in twin.state_dict().items()} == {
        "mlp.layer0.w": (N_CH - 1, 32),
        "mlp.layer1.w": (32, 1),
    }


def test_the_hidden_features_are_the_linear_channel_and_the_input_of_the_last_layer():
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(return_hidden_target="gmm_features", normalize="layer")
    assert twin.hidden_key == "gmm_features"
    inputs = _features(rng)
    out, hidden = twin.forward_with_hidden(inputs)
    expected, expected_hidden = _readout_reference(twin, inputs, normalize="layer")
    assert hidden.shape == (N_ATOMS, 1 + 4)
    assert_close(hidden, expected_hidden, FEATURE_F64)
    assert_close(out, expected, FEATURE_F64)
    torch.testing.assert_close(
        out, twin(inputs), rtol=0, atol=0
    )  # the main output is unchanged
    assert (
        _readout(return_hidden_target="").hidden_key is None
    )  # an empty name is off, as in TF


def test_only_the_l0_column_of_an_origin_is_read():
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout()
    inputs = _features(rng, lm=4)
    scalar_only = {**inputs, "I": inputs["I"][:, :1, :]}
    torch.testing.assert_close(twin(inputs), twin(scalar_only), rtol=0, atol=0)


def test_the_state_holds_the_mlp_weights_and_the_scale_of_the_layer_normalisation():
    assert set(_readout(normalize="layer").state_dict()) == {
        "mlp.layer0.w",
        "mlp.layer1.w",
        "scale",
    }
    assert set(_readout().state_dict()) == {"mlp.layer0.w", "mlp.layer1.w"}
    assert _readout(normalize="layer").state_dict()["scale"].shape == (1, N_CH - 1)


def test_a_float32_model_reads_float32_features():
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(F32, normalize="layer")
    inputs = _features(rng, dtype=np.float32)
    out = twin(inputs)
    assert out.dtype == torch.float32
    assert_close(
        out, _readout_reference(twin, inputs, normalize="layer")[0], FEATURE_F32
    )
    with pytest.raises(
        DtypeError
    ):  # the layer norm needs features in the parameter dtype, as TF's does
        twin(_features(rng))


def test_gradcheck():
    rng = np.random.default_rng(RNG_SEED)
    twin = _readout(normalize="layer", activation="tanh")
    inputs = _features(rng)
    x = inputs["I"].requires_grad_()
    assert torch.autograd.gradcheck(lambda x: twin({**inputs, "I": x}), (x,))


@pytest.mark.parametrize(
    ("origins", "channels", "match"),
    [
        (("I0", "I1"), [N_CH, N_CH + 1], "one channel count"),
        (("I",), [1], "at least 2"),
        (("I0", "I1"), [N_CH], "one channel count"),
    ],
)
def test_origins_of_unequal_or_too_few_channels_are_refused(origins, channels, match):
    with pytest.raises(ShapeError, match=match):
        _readout(origins=origins, channels=channels)


def test_an_unknown_normalisation_is_refused():
    with pytest.raises(OptionValueError, match="normalize must be null or 'layer'"):
        _readout(normalize="batch")


# ------------------------------------------------------------------ ConstantScaleShiftTarget


def _scale(
    policy=F64, n_elements=None, embedding_size=None, **options: Any
) -> ConstantScaleShiftTarget:
    spec = _spec(
        "ConstantScaleShiftTarget",
        "scale",
        target=InstructionRef("atomic_energy"),
        **options,
    )
    return ConstantScaleShiftTarget(
        spec, policy, n_elements=n_elements, embedding_size=embedding_size
    )


def _energy(rng, dtype=np.float64) -> dict[str, torch.Tensor]:
    return {
        "atomic_energy": torch.as_tensor(rng.normal(size=(N_ATOMS, 1)).astype(dtype)),
        "atomic_mu_i": torch.as_tensor(np.array([0, 2, 1, 2, 0])),
    }


def test_the_scale_alone_multiplies_the_target():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng)
    twin = _scale(scale=OMAT_SCALE, shift=0)
    assert twin.state_dict() == {}
    torch.testing.assert_close(
        twin(inputs), inputs["atomic_energy"] * OMAT_SCALE, rtol=0, atol=0
    )


def test_a_constant_shift_is_added_after_the_scale():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng)
    out = _scale(scale=2.0, shift=-0.5)(inputs)
    torch.testing.assert_close(out, inputs["atomic_energy"] * 2.0 - 0.5, rtol=0, atol=0)


def test_a_per_element_map_shifts_each_atom_by_its_element():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng)
    shift_map = {
        2: 30.0,
        0: 10.0,
        1: 20.0,
    }  # given unsorted: the key is the element index
    out = _scale(scale=2.0, atomic_shift_map=shift_map, n_elements=3)(inputs)
    expected = inputs["atomic_energy"] * 2.0 + torch.tensor(
        [[10.0], [30.0], [20.0], [30.0], [10.0]], dtype=torch.float64
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_an_embedding_shift_projects_the_element_vector():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng)
    z = rng.normal(size=(3, 4))
    w = rng.normal(size=(4, 1))
    inputs["Z"] = torch.as_tensor(z)
    twin = _scale(scale=2.0, chemical_embedding=InstructionRef("Z"), embedding_size=4)
    assert set(twin.state_dict()) == {"embedding_shift"}
    twin.load_state_dict({"embedding_shift": torch.as_tensor(w)}, strict=True)
    mu = inputs["atomic_mu_i"].numpy()
    expected = inputs["atomic_energy"].numpy() * 2.0 + (z[mu] @ w) / 2.0
    assert_close(twin(inputs), expected, FEATURE_F64)


def test_a_float32_model_rounds_the_scale_as_tf_does():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng, dtype=np.float32)
    twin = _scale(F32, scale=OMAT_SCALE)
    out = twin(inputs)
    assert out.dtype == torch.float32
    assert float(twin.get_buffer("scale")) == OMAT_SCALE_F32
    torch.testing.assert_close(
        out, inputs["atomic_energy"] * torch.tensor(OMAT_SCALE_F32), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (
            {"atomic_shift_map": {0: 1.0, 2: 1.0}},
            r"keys must be 0 \.\. 1, got \[0, 2\]",
        ),
        ({"atomic_shift_map": {0: 1.0}}, "1 entries for a model of 3 elements"),
        (
            {"atomic_shift_map": {0: 1.0, 1: 1.0, 2: 1.0}, "shift": 1.0},
            "shift must be 0",
        ),
        ({"chemical_embedding": InstructionRef("Z")}, "an embedding size of None"),
    ],
)
def test_inconsistent_shifts_are_refused(options, match):
    with pytest.raises(OptionValueError, match=match):
        _scale(n_elements=3, **options)


def test_a_floating_element_index_is_refused():
    inputs = _energy(np.random.default_rng(RNG_SEED))
    inputs["atomic_mu_i"] = inputs["atomic_mu_i"].double()
    with pytest.raises(DtypeError, match="int32 or int64"):
        _scale(atomic_shift_map={0: 1.0, 1: 2.0, 2: 3.0})(inputs)


# ------------------------------------------------------------------ physical values


def test_each_atomic_energy_depends_on_its_own_atom_only_and_relabelling_permutes_them():
    rng = np.random.default_rng(RNG_SEED)
    chain = (
        _readout(origins=("I0", "I1"), normalize="layer"),
        _scale(scale=OMAT_SCALE, shift=0.1),
    )
    inputs = _features(rng, origins=("I0", "I1"))
    # the chain starts from CreateOutputTarget's scalar
    inputs["atomic_energy"] = torch.zeros((), dtype=torch.float64)

    def energies(data: dict[str, torch.Tensor]) -> torch.Tensor:
        data = dict(data)
        for twin in chain:
            data["atomic_energy"] = twin(data)
        return data["atomic_energy"]

    before = energies(inputs)
    changed = {**inputs, "I1": inputs["I1"].clone()}
    changed["I1"][2] += 1.0
    after = energies(changed)
    others = [a for a in range(N_ATOMS) if a != 2]
    torch.testing.assert_close(after[others], before[others], rtol=0, atol=0)
    assert not torch.equal(after[2], before[2])
    order = torch.as_tensor(rng.permutation(N_ATOMS))
    relabelled = {k: (v[order] if v.dim() else v) for k, v in inputs.items()}
    torch.testing.assert_close(energies(relabelled), before[order], rtol=0, atol=1e-15)


# ------------------------------------------------------------------ TrainableShiftTarget


def _trainable(policy=F64, n_types=3) -> TrainableShiftTarget:
    spec = _spec(
        "TrainableShiftTarget",
        "TrainableShiftTarget",
        target=InstructionRef("atomic_energy"),
        number_of_atom_types=n_types,
    )
    twin = TrainableShiftTarget(spec, policy)
    shifts = np.array([[-3.5], [0.25], [-11.0]][:n_types])
    twin.load_state_dict({"at_shifts": torch.as_tensor(shifts, dtype=policy.param)})
    return twin


def test_the_trainable_shift_adds_the_row_of_each_atoms_element():
    inputs = _energy(np.random.default_rng(RNG_SEED))
    out = _trainable()(inputs)
    table = np.array([-3.5, 0.25, -11.0])
    expected = inputs["atomic_energy"].numpy() + table[[0, 2, 1, 2, 0], None]
    assert out.dtype == torch.float64
    np.testing.assert_array_equal(out.detach().numpy(), expected)


def test_the_shift_table_starts_at_zero_and_is_the_only_weight():
    spec = _spec(
        "TrainableShiftTarget",
        "TrainableShiftTarget",
        target=InstructionRef("atomic_energy"),
        number_of_atom_types=4,
    )
    twin = TrainableShiftTarget(spec, F32)
    (key,) = twin.state_dict()
    assert key == "at_shifts" and twin.at_shifts.shape == (4, 1)
    assert twin.at_shifts.dtype == torch.float32
    assert torch.count_nonzero(twin.at_shifts) == 0


def test_a_float32_model_shifts_a_float32_target():
    inputs = _energy(np.random.default_rng(RNG_SEED), np.float32)
    out = _trainable(F32)(inputs)
    assert out.dtype == torch.float32
    table = np.array([-3.5, 0.25, -11.0], dtype=np.float32)
    expected = inputs["atomic_energy"].numpy() + table[[0, 2, 1, 2, 0], None]
    np.testing.assert_array_equal(out.detach().numpy(), expected)


def test_no_atom_type_and_a_floating_element_index_are_refused():
    spec = _spec(
        "TrainableShiftTarget",
        "t",
        target=InstructionRef("atomic_energy"),
        number_of_atom_types=0,
    )
    with pytest.raises(
        OptionValueError, match="number_of_atom_types must be at least 1"
    ):
        TrainableShiftTarget(spec, F64)
    inputs = _energy(np.random.default_rng(RNG_SEED))
    inputs["atomic_mu_i"] = inputs["atomic_mu_i"].double()
    with pytest.raises(DtypeError, match="int32 or int64"):
        _trainable()(inputs)


def test_the_learned_shifts_add_one_reference_energy_per_atom_to_the_total():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _energy(rng)
    twin = _trainable()
    total = float(twin(inputs).detach().sum() - inputs["atomic_energy"].sum())
    counts = np.bincount(inputs["atomic_mu_i"].numpy(), minlength=3)
    assert total == pytest.approx(
        float(counts @ np.array([-3.5, 0.25, -11.0])), abs=1e-12
    )
    order = torch.as_tensor(rng.permutation(N_ATOMS))
    relabelled = {k: v[order] for k, v in inputs.items()}
    assert torch.equal(twin(relabelled), twin(inputs)[order])
