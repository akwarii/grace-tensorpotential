"""Characterization tests for four output instructions of ``tensorpotential/instructions/output.py``.

Classes covered: ``CreateOutputTarget``, ``LinMLPOut2ScalarTarget``, ``ConstantScaleShiftTarget`` and
``TrainableShiftTarget`` (the other classes of the module have their own tests).

Logic layer: constructor assertions, defaults captured into ``_init_args``, ``build`` (variables, dtypes,
building twice), the branches of ``frwrd`` (``normalize``, ``return_hidden_target``, ``lm_first`` origins,
``local``, padding atoms, every kind of shift), the in-place overwrite of ``input_data[target.name]`` by
``__call__``, LoRA activation and merge, element selection (``prepare_variables_for_selected_elements`` and
``upd_init_args_new_elements``) and the yaml round trip of the constructor arguments.

Physical layer: every value is compared with a numpy re-computation that shares no code with the unit
(the MLP of ``LinMLPOut2ScalarTarget`` is written out layer by layer from the seeded weights), with hand
computed numbers (shifts per atom type, sorted ``atomic_shift_map``), with invariances (the first scalar
feature passes through linearly, the normalised part does not depend on the scale of its input, atoms can
be permuted), with extensivity (the shift of a union of structures is the sum of the shifts) and with
gradients whose value is a count of atoms.

A mock is used for one thing only: ``_Origin`` stands for the upstream ``FunctionReduce`` and exposes the
four attributes the targets read (``name``, ``n_out``, ``lmax`` and optionally ``lm_first``); it replaces
an instruction that would need a whole neighbour-list graph and is never a numeric path of the unit.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.instructions.compute import ScalarChemicalEmbedding
from tensorpotential.instructions.output import (
    ConstantScaleShiftTarget,
    CreateOutputTarget,
    LinMLPOut2ScalarTarget,
    TrainableShiftTarget,
)
from tests.seeded_weights import seed_trainable_variables, seeded_values
from tests.tolerances import FLOAT64_ARITHMETIC, FLOAT32_NETWORK

# Tolerance of every float64 comparison below (a handful of arithmetic operations per value).
TOL = FLOAT64_ARITHMETIC

N_TOTAL = 7  # atoms of the padded batch used by the readout tests
N_REAL = 5  # real atoms among them: the last two are padding
N_FEAT = 6  # features of an origin: one linear passthrough and five for the MLP
HIDDEN = [8, 6]
SCALE_INIT_STD = 1e-16  # stddev of the initial ``scale`` of the normalised readout
SILU_N2NORM = 1.6759  # factor of the default activation ``silu_n2norm``


class _Origin:
    """Stand-in for ``FunctionReduce``: the name of its output, its width and its ``lmax``.

    Parameters
    ----------
    name : str
        Key of the origin in the input dictionary.
    n_out : int
        Number of features per atom.
    lm_first : bool or None
        When not None the attribute ``lm_first`` is set; when None the attribute does not exist.
    """

    def __init__(self, name: str, n_out: int, lm_first: bool | None = None) -> None:
        self.name = name
        self.n_out = n_out
        self.lmax = 0
        if lm_first is not None:
            self.lm_first = lm_first


class _OriginWithoutLmax:
    """Origin without the attribute ``lmax`` (the constructors then assume ``lmax = 0``)."""

    def __init__(self, name: str, n_out: int) -> None:
        self.name = name
        self.n_out = n_out


# ------------------------------------------------------------------------------ numpy oracles


def _np_activation(name: str | None) -> Callable[[np.ndarray], np.ndarray]:
    """Activation function written in numpy (``None`` is the default ``silu_n2norm``)."""
    if name == "tanh":
        return np.tanh
    if name == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if name == "silu":
        return lambda x: x / (1.0 + np.exp(-x))
    return lambda x: SILU_N2NORM * x / (1.0 + np.exp(-x))


def _np_mlp(
    x: np.ndarray,
    weights: Sequence[np.ndarray],
    activation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bias-free MLP with weights divided by ``sqrt(n_in)``; returns the output and the last hidden layer."""
    act = _np_activation(activation)
    hidden = x
    for i, w in enumerate(weights):
        hidden = x  # the input of the last layer once the loop ends
        x = x @ (w / np.sqrt(w.shape[0]))
        if i < len(weights) - 1:
            x = act(x)
    return x, hidden


def _np_rms_ln(x: np.ndarray, scale: np.ndarray, n_real: int) -> np.ndarray:
    """RMS layer normalisation of every atom with the rows of padding atoms set to zero."""
    rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-16)
    rms[n_real:] = 0.0
    return x * scale * rms


def _np_linmlp(
    features: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    target: float,
    scale: np.ndarray | None,
    n_real: int,
    activation: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass of ``LinMLPOut2ScalarTarget`` from ``[atoms, n_out]`` scalar features.

    Returns the output, the linear part ``[atoms, 1]`` and the last hidden layer.
    """
    lin = sum(f[:, :1] for f in features)
    nonlin = sum(f[:, 1:] for f in features)
    if scale is not None:
        nonlin = _np_rms_ln(nonlin, scale, n_real)
    out, hidden = _np_mlp(nonlin, weights, activation)
    return target + out + lin, lin, hidden


def _mlp_weights(ins: LinMLPOut2ScalarTarget) -> list[np.ndarray]:
    """Weight matrices of the MLP of a built instruction, as numpy arrays."""
    return [getattr(ins.mlp, f"layer{i}").w.numpy() for i in range(ins.mlp.nlayers)]


def _mlp_weights_with_lora(ins: LinMLPOut2ScalarTarget) -> list[np.ndarray]:
    """Weights plus the low-rank (or additive) update held in the LoRA tensors."""
    out = []
    for i in range(ins.mlp.nlayers):
        layer = getattr(ins.mlp, f"layer{i}")
        tensors = [t.numpy() for t in layer.lora_tensors]
        if len(tensors) == 1:  # additive mode: one full matrix
            out.append(layer.w.numpy() + tensors[0])
        else:  # low rank: A [n_in, r] and B [n_out, r]
            out.append(layer.w.numpy() + tensors[0] @ tensors[1].T)
    return out


# ------------------------------------------------------------------------------ builders


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _built_target(
    dtype: tf.DType = tf.float64, initial_value: float = 0.0, name: str = "energy"
) -> CreateOutputTarget:
    """A built ``CreateOutputTarget``."""
    target = CreateOutputTarget(name=name, initial_value=initial_value)
    target.build(dtype)
    return target


def _linmlp(
    dtype: tf.DType = tf.float64,
    n_origin: int = 1,
    seed: bool = True,
    initial_value: float = 0.0,
    lm_first: bool | None = None,
    **kwargs: Any,
) -> LinMLPOut2ScalarTarget:
    """A built readout over ``n_origin`` origins of ``N_FEAT`` features, with seeded weights."""
    origins = [_Origin(f"origin{i}", N_FEAT, lm_first) for i in range(n_origin)]
    target = _built_target(dtype, initial_value)
    kwargs.setdefault("hidden_layers", list(HIDDEN))
    ins = LinMLPOut2ScalarTarget(origin=origins, target=target, **kwargs)
    ins.build(dtype)
    if seed:
        seed_trainable_variables(ins)
    return ins


def _features(
    n_origin: int = 1, n_lm: int = 1, n_atoms: int = N_TOTAL, seed: int = 7
) -> list[np.ndarray]:
    """Random ``[atoms, n_out, lm]`` features of ``n_origin`` origins."""
    rng = _rng(seed)
    return [rng.normal(size=(n_atoms, N_FEAT, n_lm)) for _ in range(n_origin)]


def _readout_data(
    ins: LinMLPOut2ScalarTarget,
    features: Sequence[np.ndarray],
    n_real: int = N_REAL,
    dtype: tf.DType = tf.float64,
) -> dict[str, tf.Tensor]:
    """Input dictionary of a readout: the target, the origins and the two atom counts."""
    n_total = features[0].shape[0]
    data = {
        ins.target.name: ins.target.frwrd({}),
        constants.N_ATOMS_BATCH_REAL: tf.constant(n_real, tf.int32),
        constants.N_ATOMS_BATCH_TOTAL: tf.constant(n_total, tf.int32),
    }
    for origin, f in zip(ins.origin, features):
        data[origin.name] = tf.constant(f, dtype)
    return data


def _scalar_features(features: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The ``lm = 0`` slice of ``[atoms, n_out, lm]`` features."""
    return [f[:, :, 0] for f in features]


def _scale_value(ins: LinMLPOut2ScalarTarget) -> np.ndarray | None:
    return ins.scale.numpy() if ins.normalize is not None else None


# ------------------------------------------------------------------------------ shift builders

ELEMENT_MAP = {"H": 0, "C": 1, "O": 2}


def _shift_batch(
    target: np.ndarray,
    mu: Sequence[int],
    n_real: int,
    dtype: tf.DType = tf.float64,
    local_mu: Sequence[int] | None = None,
) -> dict[str, tf.Tensor]:
    """Input dictionary of a shift instruction; ``target`` is ``[atoms, 1]`` (or a scalar)."""
    data = {
        "energy": tf.constant(target, dtype),
        constants.ATOMIC_MU_I: tf.constant(mu, tf.int32),
        constants.N_ATOMS_BATCH_REAL: tf.constant(n_real, tf.int32),
    }
    if local_mu is not None:
        data[constants.ATOMIC_MU_I_LOCAL] = tf.constant(local_mu, tf.int32)
    return data


def _constant_shift(
    dtype: tf.DType = tf.float64, **kwargs: Any
) -> ConstantScaleShiftTarget:
    """A built ``ConstantScaleShiftTarget`` on a built target named ``energy``."""
    ins = ConstantScaleShiftTarget(target=_built_target(dtype), **kwargs)
    ins.build(dtype)
    return ins


def _embedding(dtype: tf.DType = tf.float64, size: int = 5) -> ScalarChemicalEmbedding:
    """A built chemical embedding of the three elements of ``ELEMENT_MAP``."""
    emb = ScalarChemicalEmbedding(
        name="Z", element_map=ELEMENT_MAP, embedding_size=size
    )
    emb.build(dtype)
    return emb


def _trainable_shift(
    n_types: int = 3, dtype: tf.DType = tf.float64, seed: bool = True
) -> TrainableShiftTarget:
    """A built ``TrainableShiftTarget``; ``seed`` replaces its zero shifts by seeded values."""
    ins = TrainableShiftTarget(
        target=_built_target(dtype), number_of_atom_types=n_types
    )
    ins.build(dtype)
    if seed:
        seed_trainable_variables(ins)
    return ins


def _patch_selected_elements(ins: Any, index_to_select: Sequence[int]) -> None:
    """Replay what ``select_elements_in_model`` does to one instruction.

    It gathers the new tensors, sets them as attributes and then updates the saved constructor arguments
    (``tensorpotential/utils.py``, which cannot be called here without a checkpoint).
    """
    idx = tf.constant(index_to_select, tf.int32)
    for var_name, var in ins.prepare_variables_for_selected_elements(idx).items():
        setattr(ins, var_name, var)
    ins.upd_init_args_new_elements({i: k for k, i in enumerate(index_to_select)})


def _close(actual: Any, expected: Any, tol: Any = TOL) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=tol.rtol, atol=tol.atol
    )


# =========================================================================== CreateOutputTarget


def test_create_output_target_defaults_are_captured_in_init_args():
    target = CreateOutputTarget(name="energy")
    assert target.l == 0
    assert target.value == 0.0
    assert target._init_args == {"name": "energy", "initial_value": 0.0, "l": 0}
    assert not target.is_built


def test_create_output_target_none_initial_value_is_zero():
    target = CreateOutputTarget(name="energy", initial_value=None)
    assert target.value == 0.0
    # the saved yaml keeps what the caller passed
    assert target._init_args["initial_value"] is None


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_create_output_target_build_makes_a_scalar_constant(dtype):
    target = _built_target(dtype, initial_value=2.5)
    assert target.is_built
    assert target.value.dtype == dtype
    assert target.value.shape == ()
    assert float(target.value) == 2.5
    assert not target.trainable_variables


def test_create_output_target_build_twice_keeps_the_first_dtype():
    target = _built_target(tf.float32, initial_value=1.5)
    first = target.value
    target.build(tf.float64)  # no-op: ``is_built`` guards the conversion
    assert target.value is first
    assert target.value.dtype == tf.float32


def test_create_output_target_frwrd_ignores_its_arguments():
    target = _built_target(tf.float64, initial_value=-0.75)
    plain = target.frwrd({})
    assert float(plain) == -0.75
    other = target.frwrd({"anything": 1}, training=True, local=True)
    assert float(other) == -0.75


def test_create_output_target_frwrd_before_build_returns_the_python_number():
    # Pins current behaviour (reported as a finding): no build, no tensor; the value is a Python float
    target = CreateOutputTarget(name="energy", initial_value=0.25)
    value = target.frwrd({})
    assert isinstance(value, float)
    assert value == 0.25


def test_create_output_target_l_is_kept_and_checked_by_the_readout():
    vector = CreateOutputTarget(name="vec", l=1)
    assert vector.l == 1
    with pytest.raises(AssertionError, match="Target l=1 does not match origin l=0"):
        LinMLPOut2ScalarTarget(origin=[_Origin("o", N_FEAT)], target=vector)


def test_create_output_target_yaml_round_trip(tmp_path):
    target = CreateOutputTarget(name="atomic_energy", initial_value=3.5, l=0)
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), {"atomic_energy": target})
    loaded = load_instructions(str(path))["atomic_energy"]
    assert isinstance(loaded, CreateOutputTarget)
    assert loaded._init_args == target._init_args
    loaded.build(tf.float64)
    assert float(loaded.value) == 3.5


# =========================================================================== LinMLPOut2ScalarTarget: logic


def test_lin_mlp_out2_scalar_target_defaults_are_captured():
    ins = _linmlp(seed=False, hidden_layers=None)
    assert ins.hidden_layers == [32]
    assert ins.n_out == 1
    assert ins.normalize is None
    assert ins.return_hidden_target is None
    assert not ins.lora
    assert ins.lora_config is None
    assert ins.mlp.layers_config == [N_FEAT - 1, 32, 1]
    assert ins.mlp.name == "LinMLPOut2ScalarTarget_MLP"
    args = ins._init_args
    assert args["hidden_layers"] is None  # the default, not the resolved [32]
    assert args["normalize"] is None
    assert args["activation"] is None
    assert args["n_out"] == 1
    assert args["lora_config"] is None
    assert args["return_hidden_target"] is None


def test_lin_mlp_out2_scalar_target_input_tensor_spec():
    spec = LinMLPOut2ScalarTarget.input_tensor_spec
    assert set(spec) == {constants.N_ATOMS_BATCH_REAL, constants.N_ATOMS_BATCH_TOTAL}


def test_lin_mlp_out2_scalar_target_accepts_unknown_keyword_arguments():
    # the presets pass ``full_origin_norm``; it is swallowed and saved with the other arguments
    ins = _linmlp(seed=False, full_origin_norm=True)
    assert ins._init_args["full_origin_norm"] is True


def test_lin_mlp_out2_scalar_target_rejects_unknown_normalisation():
    with pytest.raises(AssertionError):
        _linmlp(seed=False, normalize="batch")


def test_lin_mlp_out2_scalar_target_rejects_non_scalar_origin():
    origin = _Origin("o", N_FEAT)
    origin.lmax = 1
    with pytest.raises(AssertionError, match="non-scalar origin"):
        LinMLPOut2ScalarTarget(origin=[origin], target=_built_target())


def test_lin_mlp_out2_scalar_target_rejects_origins_of_different_widths():
    origins = [_Origin("a", N_FEAT), _Origin("b", N_FEAT + 2)]
    with pytest.raises(AssertionError, match="Not all shapes are the same"):
        LinMLPOut2ScalarTarget(origin=origins, target=_built_target())


def test_lin_mlp_out2_scalar_target_origin_without_lmax_counts_as_scalar():
    target = _built_target()
    ins = LinMLPOut2ScalarTarget(
        origin=[_OriginWithoutLmax("o", N_FEAT)], target=target, hidden_layers=[4]
    )
    assert ins.mlp.layers_config == [N_FEAT - 1, 4, 1]


def test_lin_mlp_out2_scalar_target_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation must be a string"):
        _linmlp(seed=False, activation="relu")


def test_lin_mlp_out2_scalar_target_build_without_normalisation_has_no_scale():
    ins = _linmlp(seed=False)
    assert not hasattr(ins, "scale")
    assert ins.is_built
    assert ins.mlp.is_built
    assert len(ins.trainable_variables) == len(HIDDEN) + 1  # one weight per layer


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_lin_mlp_out2_scalar_target_build_with_normalisation_makes_a_tiny_scale(dtype):
    ins = _linmlp(dtype, seed=False, normalize="layer")
    assert ins.scale.shape == [1, N_FEAT - 1]
    assert ins.scale.dtype == dtype
    assert ins.scale.trainable
    assert np.max(np.abs(ins.scale.numpy())) < 10 * SCALE_INIT_STD
    assert ins.scale in ins.trainable_variables
    assert all(getattr(ins.mlp, f"layer{i}").w.dtype == dtype for i in range(3))


def test_lin_mlp_out2_scalar_target_build_twice_pins_scale_replacement():
    # Pins current behaviour (reported as a finding): build() is not guarded by is_built for the scale,
    # so a second call draws a new scale variable while the (guarded) MLP keeps its weights
    ins = _linmlp(seed=False, normalize="layer")
    first_scale = ins.scale
    first_weight = ins.mlp.layer0.w
    ins.build(tf.float64)
    assert ins.scale is not first_scale
    assert ins.mlp.layer0.w is first_weight
    assert ins.is_built


@pytest.mark.parametrize("n_out", [1, 3])
def test_lin_mlp_out2_scalar_target_frwrd_shape_and_target_offset(n_out):
    ins = _linmlp(n_out=n_out, initial_value=0.75, normalize="layer")
    features = _features()
    out = ins.frwrd(_readout_data(ins, features))
    assert out.shape == (N_TOTAL, n_out)
    expected, _, _ = _np_linmlp(
        _scalar_features(features), _mlp_weights(ins), 0.75, _scale_value(ins), N_REAL
    )
    _close(out, expected)


def test_lin_mlp_out2_scalar_target_call_overwrites_only_the_target_key():
    ins = _linmlp(normalize="layer")
    data = _readout_data(ins, _features())
    expected = ins.frwrd(data).numpy()
    others = {k: v for k, v in data.items() if k != "energy"}
    returned = ins(data)
    assert returned is data
    assert data["energy"].shape == (N_TOTAL, 1)
    _close(data["energy"], expected)
    assert set(data) == set(others) | {"energy"}
    assert all(data[k] is v for k, v in others.items())


def test_lin_mlp_out2_scalar_target_frwrd_does_not_modify_input_without_hidden():
    ins = _linmlp()
    data = _readout_data(ins, _features())
    before = dict(data)
    ins.frwrd(data)
    assert data == before


def test_lin_mlp_out2_scalar_target_ignores_training_and_local_flags():
    ins = _linmlp(normalize="layer")
    data = _readout_data(ins, _features())
    plain = ins.frwrd(data).numpy()
    flagged = ins.frwrd(data, training=True, local=True).numpy()
    np.testing.assert_array_equal(plain, flagged)


def test_lin_mlp_out2_scalar_target_return_hidden_target_writes_into_input_data():
    # the hidden features are an extra output written under another key (``weak contract`` in the source)
    ins = _linmlp(normalize="layer", return_hidden_target="hidden_feats")
    features = _features()
    data = _readout_data(ins, features)
    out = ins.frwrd(data)
    hidden = data["hidden_feats"].numpy()
    assert hidden.shape == (N_TOTAL, 1 + HIDDEN[-1])
    ins.return_hidden_target = None
    plain_data = _readout_data(ins, features)
    _close(out, ins.frwrd(plain_data))
    assert "hidden_feats" not in plain_data
    _, lin, last_hidden = _np_linmlp(
        _scalar_features(features), _mlp_weights(ins), 0.0, _scale_value(ins), N_REAL
    )
    _close(hidden[:, :1], lin)
    _close(hidden[:, 1:], last_hidden)


def test_lin_mlp_out2_scalar_target_call_with_hidden_target_adds_both_keys():
    ins = _linmlp(return_hidden_target="hidden_feats")
    data = _readout_data(ins, _features())
    result = ins(data)
    assert set(result) >= {"energy", "hidden_feats"}
    assert result["energy"].shape == (N_TOTAL, 1)
    assert result["hidden_feats"].shape == (N_TOTAL, 1 + HIDDEN[-1])


def test_lin_mlp_out2_scalar_target_empty_return_hidden_target_is_off():
    ins = _linmlp(return_hidden_target="")
    data = _readout_data(ins, _features())
    ins.frwrd(data)
    assert "" not in data


@pytest.mark.parametrize("lm_first", [False, True])
def test_lin_mlp_out2_scalar_target_origin_layout(lm_first):
    # [atoms, n_out, lm] is the default layout; with ``lm_first`` the origin is [lm, atoms, n_out]
    ins = _linmlp(normalize="layer", lm_first=lm_first)
    features = _features(n_lm=4)
    data = _readout_data(ins, features)
    if lm_first:
        data["origin0"] = tf.constant(np.transpose(features[0], (2, 0, 1)))
    out = ins.frwrd(data)
    expected, _, _ = _np_linmlp(
        _scalar_features(features),
        _mlp_weights(ins),
        0.0,
        _scale_value(ins),
        N_REAL,
    )
    _close(out, expected)


def test_lin_mlp_out2_scalar_target_mixed_layouts_of_two_origins():
    ins = _linmlp(n_origin=2, normalize="layer")
    ins.origin[1].lm_first = True
    features = _features(n_origin=2, n_lm=3)
    data = _readout_data(ins, features)
    data["origin1"] = tf.constant(np.transpose(features[1], (2, 0, 1)))
    out = ins.frwrd(data)
    expected, _, _ = _np_linmlp(
        _scalar_features(features), _mlp_weights(ins), 0.0, _scale_value(ins), N_REAL
    )
    _close(out, expected)


def test_lin_mlp_out2_scalar_target_float32_build_gives_float32_output():
    ins = _linmlp(tf.float32, normalize="layer")
    features = _features()
    data = _readout_data(ins, features, dtype=tf.float32)
    out = ins.frwrd(data)
    assert out.dtype == tf.float32
    expected, _, _ = _np_linmlp(
        _scalar_features(features),
        _mlp_weights(ins),
        0.0,
        _scale_value(ins),
        N_REAL,
    )
    # the oracle uses the float32 weights as stored but the unrounded float64 features
    _close(out, expected, FLOAT32_NETWORK)


# =========================================================================== LinMLPOut2ScalarTarget: physics


@pytest.mark.parametrize(
    ("normalize", "activation", "n_out"),
    [
        (None, None, 1),
        ("layer", None, 1),
        ("layer", "silu", 1),
        ("layer", "tanh", 1),
        ("layer", "sigmoid", 2),
    ],
)
def test_lin_mlp_out2_scalar_target_matches_numpy_forward_pass(
    normalize, activation, n_out
):
    ins = _linmlp(
        normalize=normalize, activation=activation, n_out=n_out, initial_value=0.5
    )
    features = _features(n_lm=3)
    out = ins.frwrd(_readout_data(ins, features))
    expected, _, _ = _np_linmlp(
        _scalar_features(features),
        _mlp_weights(ins),
        0.5,
        _scale_value(ins),
        N_REAL,
        activation,
    )
    assert out.shape == expected.shape
    _close(out, expected)


def test_lin_mlp_out2_scalar_target_two_origins_add_their_features():
    ins = _linmlp(n_origin=2, normalize="layer", activation="silu")
    features = _features(n_origin=2)
    out = ins.frwrd(_readout_data(ins, features))
    expected, _, _ = _np_linmlp(
        _scalar_features(features),
        _mlp_weights(ins),
        0.0,
        _scale_value(ins),
        N_REAL,
        "silu",
    )
    _close(out, expected)


def test_lin_mlp_out2_scalar_target_padding_atoms_with_normalisation():
    # the normalised part of a padding atom is zero, so the MLP adds nothing (no bias) and its output is
    # the target plus the linear feature; there is no mask on the linear part
    ins = _linmlp(normalize="layer", initial_value=0.5)
    features = _features()
    out = ins.frwrd(_readout_data(ins, features)).numpy()
    lin = features[0][:, :1, 0]
    _close(out[N_REAL:], 0.5 + lin[N_REAL:])
    assert np.max(np.abs(out[:N_REAL] - 0.5 - lin[:N_REAL])) > 1e-3


def test_lin_mlp_out2_scalar_target_padding_atoms_without_normalisation():
    # Pins current behaviour (reported as a finding): without ``normalize`` nothing masks padding atoms
    # (real_count is not read), they go through the MLP like real atoms
    ins = _linmlp()
    features = _features()
    data = _readout_data(ins, features, n_real=3)
    out = ins.frwrd(data)
    expected, _, _ = _np_linmlp(
        _scalar_features(features), _mlp_weights(ins), 0.0, None, 3
    )
    _close(out, expected)
    assert np.max(np.abs(out.numpy()[3:] - features[0][3:, :1, 0])) > 1e-3


def test_lin_mlp_out2_scalar_target_first_feature_passes_through_linearly():
    ins = _linmlp(normalize="layer", n_out=2)
    features = _features()
    base = ins.frwrd(_readout_data(ins, features)).numpy()
    shifted = [f.copy() for f in features]
    delta = np.linspace(-1.0, 2.0, N_TOTAL)
    shifted[0][:, 0, 0] += delta
    out = ins.frwrd(_readout_data(ins, shifted)).numpy()
    _close(out - base, np.repeat(delta[:, None], 2, axis=1))


def test_lin_mlp_out2_scalar_target_gradient_wrt_first_feature_is_one_per_output():
    ins = _linmlp(normalize="layer", n_out=2)
    data = _readout_data(ins, _features())
    x = tf.constant(_features()[0])
    data["origin0"] = x
    with tf.GradientTape() as tape:
        tape.watch(x)
        total = tf.reduce_sum(ins.frwrd(data))
    grad = tape.gradient(total, x).numpy()
    _close(
        grad[:, 0, 0], np.full(N_TOTAL, 2.0)
    )  # n_out = 2 outputs each see the passthrough
    assert np.max(np.abs(grad[:N_REAL, 1:, 0])) > 1e-3
    np.testing.assert_array_equal(
        grad[N_REAL:, 1:, 0], 0.0
    )  # padding atoms: normaliser is off


def test_lin_mlp_out2_scalar_target_normalised_part_ignores_the_scale_of_its_input():
    ins = _linmlp(normalize="layer")
    features = _features()
    base = ins.frwrd(_readout_data(ins, features)).numpy()
    rescaled = [f.copy() for f in features]
    rescaled[0][:, 1:, :] *= 37.0
    out = ins.frwrd(_readout_data(ins, rescaled)).numpy()
    _close(out, base)


def test_lin_mlp_out2_scalar_target_without_normalisation_the_scale_matters():
    ins = _linmlp()
    features = _features()
    base = ins.frwrd(_readout_data(ins, features)).numpy()
    rescaled = [f.copy() for f in features]
    rescaled[0][:, 1:, :] *= 37.0
    out = ins.frwrd(_readout_data(ins, rescaled)).numpy()
    assert np.max(np.abs(out - base)) > 1e-3


def test_lin_mlp_out2_scalar_target_is_additive_in_the_target():
    features = _features()
    ins = _linmlp(normalize="layer", initial_value=0.0)
    base = ins.frwrd(_readout_data(ins, features)).numpy()
    data = _readout_data(ins, features)
    data["energy"] = tf.constant(np.linspace(-2.0, 2.0, N_TOTAL)[:, None])
    out = ins.frwrd(data).numpy()
    _close(out - base, np.linspace(-2.0, 2.0, N_TOTAL)[:, None])


def test_lin_mlp_out2_scalar_target_is_a_per_atom_function():
    ins = _linmlp(normalize="layer")
    features = _features()
    out = ins.frwrd(_readout_data(ins, features)).numpy()
    order = np.array([3, 0, 4, 1, 2, 5, 6])  # permutes the real atoms only
    permuted = [f[order] for f in features]
    out_perm = ins.frwrd(_readout_data(ins, permuted)).numpy()
    _close(out_perm, out[order])


# =========================================================================== LinMLPOut2ScalarTarget: LoRA

LORA_CONFIGS = [
    {"rank": 2, "alpha": 1.0},
    {"mode": "additive"},
]


def _assign_seeded_lora_tensors(ins: LinMLPOut2ScalarTarget) -> None:
    """Replace the zero ``B`` factors (or the zero additive deltas) by seeded values."""
    for i in range(ins.mlp.nlayers):
        for j, tensor in enumerate(getattr(ins.mlp, f"layer{i}").lora_tensors):
            tensor.assign(seeded_values(f"lora{i}_{j}", tuple(tensor.shape)) * 0.5)


@pytest.mark.parametrize("config", LORA_CONFIGS)
@pytest.mark.parametrize("normalize", [None, "layer"])
def test_lin_mlp_out2_scalar_target_lora_enable_and_finalize(config, normalize):
    ins = _linmlp(normalize=normalize)
    features = _features()
    data = _readout_data(ins, features)
    plain = ins.frwrd(data).numpy()
    plain_weights = _mlp_weights(ins)

    ins.enable_lora_adaptation(config)
    assert ins.lora
    assert ins.lora_config == config
    assert ins._init_args["lora_config"] == config
    for i in range(ins.mlp.nlayers):
        layer = getattr(ins.mlp, f"layer{i}")
        assert layer.lora
        assert not layer.w.trainable
    if normalize:
        assert not ins.scale.trainable
    assert {v.name for v in ins.trainable_variables} == {
        t.name
        for i in range(ins.mlp.nlayers)
        for t in getattr(ins.mlp, f"layer{i}").lora_tensors
    }
    _close(ins.frwrd(data), plain)  # the update starts at zero

    _assign_seeded_lora_tensors(ins)
    adapted_weights = _mlp_weights_with_lora(ins)
    expected, _, _ = _np_linmlp(
        _scalar_features(features),
        adapted_weights,
        0.0,
        _scale_value(ins),
        N_REAL,
    )
    _close(ins.frwrd(data), expected)
    assert np.max(np.abs(expected - plain)) > 1e-3

    ins.finalize_lora_update()
    assert not ins.lora
    assert ins.lora_config is None
    assert "lora_config" not in ins._init_args
    for i in range(ins.mlp.nlayers):
        layer = getattr(ins.mlp, f"layer{i}")
        assert not layer.lora
        assert layer.w.trainable
        assert not hasattr(layer, "lora_tensors")
    if normalize:
        assert ins.scale.trainable
    _close(ins.frwrd(data), expected)  # merged weights give the adapted output
    for merged, original, adapted in zip(
        _mlp_weights(ins), plain_weights, adapted_weights
    ):
        _close(merged, adapted)
        assert np.max(np.abs(merged - original)) > 1e-3


@pytest.mark.parametrize("normalize", [None, "layer"])
def test_lin_mlp_out2_scalar_target_lora_config_at_construction(normalize):
    config = {"rank": 2, "alpha": 1.0}
    origins = [_Origin("origin0", N_FEAT)]
    ins = LinMLPOut2ScalarTarget(
        origin=origins,
        target=_built_target(),
        hidden_layers=list(HIDDEN),
        normalize=normalize,
        lora_config=config,
    )
    assert ins.lora
    assert ins.lora_config == config
    assert ins._init_args["lora_config"] == config
    ins.build(tf.float64)
    for i in range(ins.mlp.nlayers):
        layer = getattr(ins.mlp, f"layer{i}")
        assert layer.lora
        assert not layer.w.trainable
        assert len(layer.lora_tensors) == 2
    if normalize:
        assert not ins.scale.trainable
    assert all("LORA" in v.name for v in ins.trainable_variables)


# =========================================================================== ConstantScaleShiftTarget: logic


def test_constant_scale_shift_target_defaults_and_apply_shift():
    target = _built_target()
    plain = ConstantScaleShiftTarget(target=target)
    assert plain.scale == 1.0
    assert plain.constant_shift == 0.0
    assert plain.atomic_shift_map is None
    assert plain.chemical_embedding is None
    assert not plain.apply_shift
    assert plain._init_args["shift"] == 0.0
    assert plain._init_args["atomic_shift_map"] is None
    assert plain.l == 0
    scale_only = ConstantScaleShiftTarget(target=target, scale=3.0)
    assert not scale_only.apply_shift
    for kwargs in (
        {"shift": 0.5},
        {"atomic_shift_map": {0: 1.0}},
        {"chemical_embedding": _embedding()},
    ):
        assert ConstantScaleShiftTarget(target=target, **kwargs).apply_shift


def test_constant_scale_shift_target_input_tensor_spec():
    spec = ConstantScaleShiftTarget.input_tensor_spec
    assert set(spec) == {constants.ATOMIC_MU_I, constants.N_ATOMS_BATCH_REAL}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atomic_shift_map": {0: 1.0}},
        {"chemical_embedding": "embedding"},
        {"atomic_shift_map": {0: 1.0}, "chemical_embedding": "embedding"},
    ],
)
def test_constant_scale_shift_target_rejects_constant_shift_with_map_or_embedding(
    kwargs,
):
    if "chemical_embedding" in kwargs:
        kwargs = {**kwargs, "chemical_embedding": _embedding()}
    with pytest.raises(AssertionError, match="constant shift must be 0"):
        ConstantScaleShiftTarget(target=_built_target(), shift=0.5, **kwargs)


def test_constant_scale_shift_target_sorts_the_shift_map_by_key():
    ins = ConstantScaleShiftTarget(
        target=_built_target(), atomic_shift_map={2: 4.0, 0: -1.5, 1: 0.5}
    )
    np.testing.assert_array_equal(ins.atomic_shift_map, [-1.5, 0.5, 4.0])


def test_constant_scale_shift_target_shift_map_position_not_key_is_the_element_index():
    # Pins current behaviour (reported as a finding): the keys are only used for ordering, so a map with
    # a gap ({0: a, 2: b}) gives b to element 1 and the instruction cannot see a missing element
    ins = _constant_shift(atomic_shift_map={0: 1.0, 2: 5.0})
    assert ins.atomic_shift_map.shape == (2, 1)
    data = _shift_batch(np.zeros((3, 1)), [0, 1, 0], 3)
    _close(ins.frwrd(data)[:, 0], [1.0, 5.0, 1.0])
    with pytest.raises(tf.errors.InvalidArgumentError):
        # the model has an element 2, the map has two entries: the gather runs out of range
        # (on CPU the gather op reports it as an error)
        ins.frwrd(_shift_batch(np.zeros((3, 1)), [0, 2, 0], 3))


def test_constant_scale_shift_target_build_converts_to_tensors_of_the_dtype():
    ins = _constant_shift(tf.float32, scale=2.0, shift=0.5)
    assert ins.scale.dtype == tf.float32
    assert ins.constant_shift.dtype == tf.float32
    mapped = _constant_shift(tf.float32, atomic_shift_map={1: 2.0, 0: 1.0})
    assert mapped.atomic_shift_map.dtype == tf.float32
    assert mapped.atomic_shift_map.shape == (2, 1)
    assert not hasattr(mapped, "embedding_shift")
    assert isinstance(mapped.constant_shift, float)  # zero stays a Python number


def test_constant_scale_shift_target_build_with_embedding_adds_a_variable_and_norm():
    emb = _embedding(size=5)
    ins = _constant_shift(chemical_embedding=emb)
    assert ins.embedding_shift.shape == (5, 1)
    assert ins.embedding_shift.trainable
    assert ins.embedding_shift in ins.trainable_variables
    _close(ins.embedding_norm, 1.0 / np.sqrt(5.0))
    assert ins.embedding_norm.dtype == tf.float64


def test_constant_scale_shift_target_build_twice_pins_current_behaviour():
    # Pins current behaviour (reported as a finding): build() has no is_built guard and never sets
    # ``is_built``; a second call with the same dtype converts the tensors again and is harmless for the
    # numbers, but a second call with another dtype fails on the scale that was already a float64 tensor
    ins = _constant_shift(scale=2.0, atomic_shift_map={0: 1.0, 1: 2.0})
    assert not ins.is_built
    ins.build(tf.float64)
    assert ins.atomic_shift_map.shape == (2, 1)
    out = ins.frwrd(_shift_batch(np.zeros((3, 1)), [0, 1, 1], 3))
    _close(out[:, 0], [1.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        ins.build(tf.float32)


def test_constant_scale_shift_target_build_twice_redraws_the_embedding_shift():
    # Pins current behaviour (reported as a finding): a second build replaces the random embedding_shift
    ins = _constant_shift(chemical_embedding=_embedding())
    first = ins.embedding_shift
    ins.build(tf.float64)
    assert ins.embedding_shift is not first


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_constant_scale_shift_target_scale_keeps_the_dtype_of_the_target(dtype):
    ins = _constant_shift(tf.float64, scale=2.5)
    data = _shift_batch(np.full((4, 1), 0.5), [0, 1, 0, 1], 4, dtype)
    out = ins.frwrd(data)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out.numpy(), np.full((4, 1), 1.25))


def test_constant_scale_shift_target_scalar_target_without_shift_stays_scalar():
    # a CreateOutputTarget value is a scalar: without a shift the result is a scalar, with a shift an
    # [atoms, 1] column
    plain = _constant_shift(scale=3.0)
    data = _shift_batch(np.float64(0.5), [0, 1, 0], 3)
    out = plain.frwrd(data)
    assert out.shape == ()
    assert float(out) == 1.5
    shifted = _constant_shift(scale=3.0, shift=1.0)
    out = shifted.frwrd(data)
    assert out.shape == (3, 1)
    _close(out[:, 0], [2.5, 2.5, 2.5])


def test_constant_scale_shift_target_shift_with_float32_target_fails_when_built_in_float64():
    # Pins current behaviour (reported as a finding): the casts to the target dtype after ``tf.where``
    # are unreachable for another dtype, ``tf.where`` fails first
    ins = _constant_shift(tf.float64, atomic_shift_map={0: 1.0, 1: 2.0})
    data = _shift_batch(np.zeros((3, 1)), [0, 1, 0], 3, tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError):
        ins.frwrd(data)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_constant_scale_shift_target_shift_dtype_follows_the_build(dtype):
    # all values are exact binary fractions, float32 arithmetic is exact
    ins = _constant_shift(dtype, scale=2.5, atomic_shift_map={0: 0.25, 1: -1.5})
    target = np.array([[0.125], [0.5], [1.0], [0.25]])
    out = ins.frwrd(_shift_batch(target, [0, 1, 1, 0], 3, dtype))
    assert out.dtype == dtype
    np.testing.assert_array_equal(out.numpy(), [[0.5625], [-0.25], [1.0], [0.625]])


def test_constant_scale_shift_target_local_uses_the_local_atom_types():
    ins = _constant_shift(atomic_shift_map={0: 10.0, 1: 20.0})
    data = _shift_batch(np.zeros((3, 1)), [0, 0, 0], 3, local_mu=[1, 0, 1])
    _close(ins.frwrd(data, local=True)[:, 0], [20.0, 10.0, 20.0])
    _close(ins.frwrd(data, local=False)[:, 0], [10.0, 10.0, 10.0])
    plain = _shift_batch(
        np.zeros((3, 1)), [0, 1, 1], 3
    )  # no local key needed when local=False
    _close(ins.frwrd(plain)[:, 0], [10.0, 20.0, 20.0])


def test_constant_scale_shift_target_call_overwrites_only_the_target_key():
    ins = _constant_shift(scale=2.0, shift=1.0)
    data = _shift_batch(np.full((4, 1), 3.0), [0, 0, 0, 0], 3)
    others = {k: v for k, v in data.items() if k != "energy"}
    returned = ins(data)
    assert returned is data
    _close(data["energy"][:, 0], [7.0, 7.0, 7.0, 6.0])
    assert all(data[k] is v for k, v in others.items())
    assert set(data) == set(others) | {"energy"}


def test_constant_scale_shift_target_embedding_with_local_atom_types():
    emb = _embedding()
    ins = _constant_shift(chemical_embedding=emb)
    seed_trainable_variables(ins)
    z = emb.frwrd({}).numpy()
    data = _shift_batch(np.zeros((3, 1)), [0, 0, 0], 3, local_mu=[2, 1, 0])
    data["Z"] = emb.frwrd({})
    out = ins.frwrd(data, local=True).numpy()
    expected = z[[2, 1, 0]] @ ins.embedding_shift.numpy() / np.sqrt(5.0)
    _close(out, expected)


def test_constant_scale_shift_target_yaml_round_trip(tmp_path):
    target = CreateOutputTarget(name="energy")
    emb = ScalarChemicalEmbedding(name="Z", element_map=ELEMENT_MAP, embedding_size=4)
    plain = ConstantScaleShiftTarget(
        target=target, scale=2.0, atomic_shift_map={1: 0.5, 0: -1.0}, name="by_type"
    )
    uniform = ConstantScaleShiftTarget(target=target, shift=3.0, name="uniform")
    embedded = ConstantScaleShiftTarget(
        target=target, chemical_embedding=emb, name="embedded"
    )
    path = tmp_path / "model.yaml"
    save_instructions_dict(
        str(path),
        {
            "energy": target,
            "Z": emb,
            "by_type": plain,
            "uniform": uniform,
            "embedded": embedded,
        },
    )
    loaded = load_instructions(str(path))
    assert loaded["by_type"].scale == 2.0
    np.testing.assert_array_equal(loaded["by_type"].atomic_shift_map, [-1.0, 0.5])
    assert loaded["uniform"].constant_shift == 3.0
    assert loaded["uniform"].apply_shift
    assert loaded["embedded"].chemical_embedding is loaded["Z"]
    assert loaded["by_type"].target is loaded["energy"]


# =========================================================================== ConstantScaleShiftTarget: physics


def test_constant_scale_shift_target_scale_and_shift_per_atom_type_hand_computed():
    # scale 2, H: -1.5, C: 0.5, O: 4; five atoms (H C O C H) and two padding atoms
    ins = _constant_shift(scale=2.0, atomic_shift_map={2: 4.0, 0: -1.5, 1: 0.5})
    target = np.array([[1.0], [2.0], [3.0], [0.5], [-1.0], [10.0], [20.0]])
    mu = [0, 1, 2, 1, 0, 0, 2]
    out = ins.frwrd(_shift_batch(target, mu, 5)).numpy()
    expected = [[0.5], [4.5], [10.0], [1.5], [-3.5], [20.0], [40.0]]
    _close(out, expected)  # padding atoms (rows 5, 6) are scaled but not shifted


def test_constant_scale_shift_target_uniform_shift_only_real_atoms():
    ins = _constant_shift(scale=0.5, shift=1.25)
    target = np.arange(6.0).reshape(6, 1)
    out = ins.frwrd(_shift_batch(target, [0] * 6, 4)).numpy()
    _close(out[:, 0], [1.25, 1.75, 2.25, 2.75, 2.0, 2.5])


def test_constant_scale_shift_target_negative_uniform_shift():
    ins = _constant_shift(shift=-2.0)
    out = ins.frwrd(_shift_batch(np.ones((3, 1)), [0, 0, 0], 2)).numpy()
    _close(out[:, 0], [-1.0, -1.0, 1.0])


def test_constant_scale_shift_target_chemical_embedding_shift_matches_numpy_einsum():
    emb = _embedding(size=5)
    ins = _constant_shift(scale=1.5, chemical_embedding=emb)
    seed_trainable_variables(ins)
    z = emb.w.numpy()
    w = ins.embedding_shift.numpy()
    mu = [2, 0, 1, 1, 0, 2]
    target = _rng().normal(size=(6, 1))
    data = _shift_batch(target, mu, 4)
    data["Z"] = emb.frwrd({})
    out = ins.frwrd(data).numpy()
    shifts = np.einsum("ae,eo->ao", z[mu], w) / np.sqrt(5.0)
    shifts[4:] = 0.0
    _close(out, 1.5 * target + shifts)
    assert np.max(np.abs(shifts[:4])) > 1e-3


def test_constant_scale_shift_target_map_and_embedding_shifts_add():
    emb = _embedding(size=3)
    ins = _constant_shift(
        atomic_shift_map={0: 1.0, 1: 2.0, 2: 3.0}, chemical_embedding=emb
    )
    seed_trainable_variables(ins)
    mu = [0, 1, 2, 0, 1]
    data = _shift_batch(np.zeros((5, 1)), mu, 4)
    data["Z"] = emb.frwrd({})
    out = ins.frwrd(data).numpy()
    embedded = emb.w.numpy()[mu] @ ins.embedding_shift.numpy() / np.sqrt(3.0)
    expected = np.array([[1.0], [2.0], [3.0], [1.0], [0.0]]) + embedded
    expected[4] = 0.0
    _close(out, expected)


def test_constant_scale_shift_target_gradient_wrt_embedding_shift_is_a_sum_of_embeddings():
    emb = _embedding(size=4)
    ins = _constant_shift(chemical_embedding=emb)
    seed_trainable_variables(ins)
    mu = [0, 2, 2, 1, 0]
    data = _shift_batch(np.zeros((5, 1)), mu, 4)
    data["Z"] = emb.frwrd({})
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(ins.frwrd(data))
    grad = tape.gradient(total, ins.embedding_shift).numpy()
    z = emb.w.numpy()
    expected = z[mu[:4]].sum(axis=0)[:, None] / np.sqrt(
        4.0
    )  # padding atom (last) excluded
    _close(grad, expected)


def test_constant_scale_shift_target_gradient_wrt_target_is_the_scale():
    ins = _constant_shift(scale=2.5, atomic_shift_map={0: 1.0, 1: 2.0})
    x = tf.constant(np.ones((4, 1)))
    data = _shift_batch(np.ones((4, 1)), [0, 1, 0, 1], 3)
    data["energy"] = x
    with tf.GradientTape() as tape:
        tape.watch(x)
        total = tf.reduce_sum(ins.frwrd(data))
    _close(tape.gradient(total, x), np.full((4, 1), 2.5))


def test_constant_scale_shift_target_total_shift_is_extensive():
    # E_total = scale * sum(target) + sum over real atoms of the shift of their element: the shift of a
    # structure is n_H E_H + n_C E_C + n_O E_O, and of two structures together the sum of both
    e0 = {0: -1.5, 1: 0.5, 2: 4.0}
    ins = _constant_shift(scale=1.0, atomic_shift_map=e0)

    def energy(mu_real: list[int], n_pad: int) -> float:
        mu = mu_real + [0] * n_pad
        data = _shift_batch(np.zeros((len(mu), 1)), mu, len(mu_real))
        return float(tf.reduce_sum(ins.frwrd(data)[: len(mu_real)]))

    a = [0, 0, 1]  # H2 C
    b = [2, 1, 1, 0]  # O C2 H
    comp_a = np.bincount(a, minlength=3)
    comp_b = np.bincount(b, minlength=3)
    shifts = np.array([e0[0], e0[1], e0[2]])
    assert energy(a, 2) == pytest.approx(
        float(comp_a @ shifts), rel=TOL.rtol, abs=TOL.atol
    )
    assert energy(b, 0) == pytest.approx(
        float(comp_b @ shifts), rel=TOL.rtol, abs=TOL.atol
    )
    union = energy(a + b, 3)
    assert union == pytest.approx(
        energy(a, 0) + energy(b, 1), rel=TOL.rtol, abs=TOL.atol
    )
    assert union == pytest.approx(
        float((comp_a + comp_b) @ shifts), rel=TOL.rtol, abs=TOL.atol
    )


def test_constant_scale_shift_target_scaled_total_energy():
    ins = _constant_shift(scale=3.0, shift=-0.5)
    target = np.array([[1.0], [2.0], [4.0], [8.0]])
    out = ins.frwrd(_shift_batch(target, [0, 0, 0, 0], 3)).numpy()
    total = out[:3].sum()
    assert total == pytest.approx(3.0 * 7.0 - 0.5 * 3, rel=TOL.rtol, abs=TOL.atol)


# =========================================================================== ConstantScaleShiftTarget: elements


def test_constant_scale_shift_target_prepare_variables_selects_rows_like_numpy():
    shifts = {0: -1.5, 1: 0.5, 2: 4.0}
    ins = _constant_shift(atomic_shift_map=shifts)
    full = ins.atomic_shift_map.numpy()
    for selection in ([1], [2, 0], [0, 1, 2], [2, 2]):
        patch = ins.prepare_variables_for_selected_elements(tf.constant(selection))
        assert set(patch) == {"atomic_shift_map"}
        np.testing.assert_array_equal(
            patch["atomic_shift_map"].numpy(), full[selection]
        )
    assert ins.atomic_shift_map.shape == (3, 1)  # the instruction itself is not changed


@pytest.mark.parametrize(
    "kwargs", [{}, {"shift": 0.5}, {"chemical_embedding": "embedding"}]
)
def test_constant_scale_shift_target_prepare_variables_without_map_is_empty(kwargs):
    if "chemical_embedding" in kwargs:
        kwargs = {"chemical_embedding": _embedding()}
    ins = _constant_shift(**kwargs)
    assert ins.prepare_variables_for_selected_elements(tf.constant([0])) == {}


def test_constant_scale_shift_target_upd_init_args_reindexes_the_selected_map():
    ins = _constant_shift(atomic_shift_map={0: -1.5, 1: 0.5, 2: 4.0})
    _patch_selected_elements(ins, [2, 0])
    saved = dict(ins._init_args["atomic_shift_map"])
    assert saved == {0: 4.0, 1: -1.5}
    assert all(isinstance(v, float) for v in saved.values())


def test_constant_scale_shift_target_upd_init_args_without_map_changes_nothing():
    ins = _constant_shift(shift=0.5)
    before = dict(ins._init_args)
    ins.upd_init_args_new_elements({"H": 0})
    assert dict(ins._init_args) == before


def test_constant_scale_shift_target_upd_init_args_needs_a_built_instruction():
    # Pins current behaviour (reported as a finding): the map is a numpy array until build() and has no
    # ``.numpy()``, so the saved arguments of an unbuilt instruction cannot be updated
    ins = ConstantScaleShiftTarget(
        target=_built_target(), atomic_shift_map={0: 1.0, 1: 2.0}
    )
    with pytest.raises(AttributeError, match="numpy"):
        ins.upd_init_args_new_elements({"H": 0, "C": 1})


def test_constant_scale_shift_target_selected_elements_keep_their_shifts():
    # physics of the selection: an atom of a kept element has the same energy shift as before, under its
    # new index
    shifts = {0: -1.5, 1: 0.5, 2: 4.0}
    ins = _constant_shift(scale=2.0, atomic_shift_map=shifts)
    target = _rng().normal(size=(4, 1))
    before = ins.frwrd(_shift_batch(target, [2, 0, 2, 0], 4)).numpy()
    _patch_selected_elements(ins, [2, 0])
    after = ins.frwrd(_shift_batch(target, [0, 1, 0, 1], 4)).numpy()  # O -> 0, H -> 1
    _close(after, before)
    rebuilt = ConstantScaleShiftTarget(
        target=ins.target,
        **{k: v for k, v in ins._init_args.items() if k not in ("target", "name", "l")},
    )
    np.testing.assert_array_equal(rebuilt.atomic_shift_map, [4.0, -1.5])


# =========================================================================== TrainableShiftTarget: logic


def test_trainable_shift_target_defaults_and_init_args():
    ins = TrainableShiftTarget(target=_built_target(), number_of_atom_types=3)
    assert ins.number_of_atom_types == 3
    assert ins.l == 0
    assert ins.name == "TrainableShiftTarget"
    assert not ins.is_built
    assert ins._init_args["number_of_atom_types"] == 3
    spec = TrainableShiftTarget.input_tensor_spec
    assert set(spec) == {constants.ATOMIC_MU_I, constants.N_ATOMS_BATCH_REAL}


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_trainable_shift_target_build_starts_at_zero(dtype):
    ins = _trainable_shift(4, dtype, seed=False)
    assert ins.is_built
    assert ins.at_shifts.shape == (4, 1)
    assert ins.at_shifts.dtype == dtype
    assert ins.at_shifts.trainable
    assert ins.at_shifts.name == "tr_atomic_shift:0"
    assert ins.trainable_variables == (ins.at_shifts,)
    np.testing.assert_array_equal(ins.at_shifts.numpy(), 0.0)


def test_trainable_shift_target_build_twice_keeps_the_variable():
    ins = _trainable_shift(3)
    first = ins.at_shifts
    values = first.numpy().copy()
    ins.build(tf.float64)
    assert ins.at_shifts is first
    np.testing.assert_array_equal(ins.at_shifts.numpy(), values)


def test_trainable_shift_target_zero_initial_shifts_leave_the_target_unchanged():
    ins = _trainable_shift(3, seed=False)
    target = _rng().normal(size=(5, 1))
    out = ins.frwrd(_shift_batch(target, [0, 1, 2, 1, 0], 4))
    np.testing.assert_array_equal(out.numpy(), target)


def test_trainable_shift_target_scalar_target_broadcasts_to_a_column():
    ins = _trainable_shift(2)
    out = ins.frwrd(_shift_batch(np.float64(1.0), [0, 1, 1], 3))
    assert out.shape == (3, 1)
    shifts = ins.at_shifts.numpy()[:, 0]
    _close(out[:, 0], 1.0 + shifts[[0, 1, 1]])


def test_trainable_shift_target_local_uses_the_local_atom_types():
    ins = _trainable_shift(3)
    shifts = ins.at_shifts.numpy()[:, 0]
    data = _shift_batch(np.zeros((3, 1)), [0, 0, 0], 3, local_mu=[2, 1, 2])
    _close(ins.frwrd(data, local=True)[:, 0], shifts[[2, 1, 2]])
    _close(ins.frwrd(data, local=False)[:, 0], shifts[[0, 0, 0]])


def test_trainable_shift_target_call_overwrites_only_the_target_key():
    ins = _trainable_shift(3)
    data = _shift_batch(np.ones((4, 1)), [0, 1, 2, 0], 3)
    others = {k: v for k, v in data.items() if k != "energy"}
    shifts = ins.at_shifts.numpy()[:, 0]
    returned = ins(data)
    assert returned is data
    _close(data["energy"][:, 0], 1.0 + np.array([*shifts[[0, 1, 2]], 0.0]))
    assert all(data[k] is v for k, v in others.items())
    assert set(data) == set(others) | {"energy"}


def test_trainable_shift_target_float32_build_gives_float32_output():
    ins = _trainable_shift(2, tf.float32)
    out = ins.frwrd(_shift_batch(np.ones((3, 1)), [0, 1, 0], 3, tf.float32))
    assert out.dtype == tf.float32
    shifts = ins.at_shifts.numpy()[:, 0]
    _close(out[:, 0], 1.0 + shifts[[0, 1, 0]], FLOAT32_NETWORK)


def test_trainable_shift_target_float32_target_with_float64_shifts_fails():
    # Pins current behaviour (reported as a finding): the cast after ``tf.where`` is dead for another
    # dtype, ``tf.where`` fails first
    ins = _trainable_shift(2, tf.float64)
    with pytest.raises(tf.errors.InvalidArgumentError):
        ins.frwrd(_shift_batch(np.ones((3, 1)), [0, 1, 0], 3, tf.float32))


def test_trainable_shift_target_prepare_variables_selects_rows_like_numpy():
    ins = _trainable_shift(4)
    full = ins.at_shifts.numpy()
    for selection in ([3], [2, 0], [0, 1, 2, 3], [1, 1]):
        patch = ins.prepare_variables_for_selected_elements(tf.constant(selection))
        assert set(patch) == {"at_shifts"}
        new = patch["at_shifts"]
        assert isinstance(new, tf.Variable)
        assert new.trainable
        np.testing.assert_array_equal(new.numpy(), full[selection])
        assert new is not ins.at_shifts
    new.assign(tf.zeros_like(new))
    np.testing.assert_array_equal(ins.at_shifts.numpy(), full)  # a copy, not a view


def test_trainable_shift_target_upd_init_args_sets_the_new_number_of_types():
    ins = _trainable_shift(4)
    ins.upd_init_args_new_elements({"O": 0, "H": 1})
    assert ins._init_args["number_of_atom_types"] == 2
    assert ins._init_args["l"] == 0
    assert ins.number_of_atom_types == 4  # only the saved arguments change


def test_trainable_shift_target_selected_elements_keep_their_shifts():
    ins = _trainable_shift(4)
    target = _rng().normal(size=(5, 1))
    before = ins.frwrd(_shift_batch(target, [3, 0, 3, 0, 3], 4)).numpy()
    _patch_selected_elements(ins, [3, 0])
    after = ins.frwrd(_shift_batch(target, [0, 1, 0, 1, 0], 4)).numpy()
    _close(after, before)
    assert ins.at_shifts.shape == (2, 1)
    assert ins._init_args["number_of_atom_types"] == 2


def test_trainable_shift_target_yaml_round_trip(tmp_path):
    target = CreateOutputTarget(name="energy")
    ins = TrainableShiftTarget(target=target, number_of_atom_types=89, name="tr")
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), {"energy": target, "tr": ins})
    loaded = load_instructions(str(path))["tr"]
    assert isinstance(loaded, TrainableShiftTarget)
    assert loaded.number_of_atom_types == 89
    assert loaded.target is not None


# =========================================================================== TrainableShiftTarget: physics


def test_trainable_shift_target_shifts_only_real_atoms_by_their_type():
    ins = _trainable_shift(3)
    shifts = ins.at_shifts.numpy()[:, 0]
    assert np.all(np.abs(shifts) > 1e-3)
    target = _rng().normal(size=(7, 1))
    mu = [0, 1, 2, 2, 1, 0, 1]
    out = ins.frwrd(_shift_batch(target, mu, 5)).numpy()
    expected = target.copy()
    expected[:5, 0] += shifts[mu[:5]]
    _close(out, expected)
    np.testing.assert_array_equal(out[5:], target[5:])  # padding atoms untouched


def test_trainable_shift_target_hand_computed_shifts():
    ins = _trainable_shift(2, seed=False)
    ins.at_shifts.assign([[10.0], [-3.0]])
    out = ins.frwrd(
        _shift_batch(np.array([[1.0], [2.0], [3.0], [4.0]]), [1, 0, 1, 0], 3)
    )
    _close(out[:, 0], [-2.0, 12.0, 0.0, 4.0])


def test_trainable_shift_target_gradient_wrt_shifts_counts_real_atoms_per_type():
    ins = _trainable_shift(4)
    mu = np.array([0, 3, 3, 1, 0, 3, 2, 2])
    n_real = 6  # the last two atoms are padding
    data = _shift_batch(np.zeros((8, 1)), mu, n_real)
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(ins.frwrd(data))
    # the gradient of a gather is an IndexedSlices
    grad = tf.convert_to_tensor(tape.gradient(total, ins.at_shifts)).numpy()
    expected = np.bincount(mu[:n_real], minlength=4)[:, None].astype(float)
    np.testing.assert_array_equal(grad, expected)
    assert expected.tolist() == [[2.0], [1.0], [0.0], [3.0]]


def test_trainable_shift_target_total_shift_is_extensive():
    ins = _trainable_shift(3)
    shifts = ins.at_shifts.numpy()[:, 0]

    def energy(mu_real: list[int], n_pad: int) -> float:
        mu = mu_real + [0] * n_pad
        data = _shift_batch(np.zeros((len(mu), 1)), mu, len(mu_real))
        return float(tf.reduce_sum(ins.frwrd(data)[: len(mu_real)]))

    a, b = [0, 2, 2], [1, 0, 1, 1]
    assert energy(a, 1) == pytest.approx(
        float(np.bincount(a, minlength=3) @ shifts), rel=TOL.rtol, abs=TOL.atol
    )
    union = energy(a + b, 2)
    assert union == pytest.approx(
        energy(a, 0) + energy(b, 3), rel=TOL.rtol, abs=TOL.atol
    )


def test_trainable_shift_target_target_passes_through_with_unit_gradient():
    ins = _trainable_shift(2)
    x = tf.constant(np.ones((4, 1)))
    data = _shift_batch(np.ones((4, 1)), [0, 1, 0, 1], 3)
    data["energy"] = x
    with tf.GradientTape() as tape:
        tape.watch(x)
        total = tf.reduce_sum(ins.frwrd(data))
    np.testing.assert_array_equal(tape.gradient(total, x).numpy(), np.ones((4, 1)))
