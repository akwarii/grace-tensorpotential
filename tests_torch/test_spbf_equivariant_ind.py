"""The ``SingleParticleBasisFunctionEquivariantInd`` twin: logic against a loop over bonds, and rotation equivariance.

Logic: the forward pass of the spec sheet (section 5) written as a loop over bonds and Clebsch-Gordan terms in numpy,
from the plan's tables and ``1 / avg_n_neigh`` of the options, with and without ``sum_neighbors``, a cut angular input,
both parameter dtypes and TF's promotion; padding bonds of TF's dense layout (``ind_i = ind_j = 0``, zero radial
part) change nothing; typed errors. Physical values: rotating the bonds (harmonics recomputed from the rotated
directions by scipy, ``tests_torch/cg_oracle.py``) and every ``l`` block of the indicator by its Wigner matrix rotates
every ``L`` block of the output by the same matrix. The comparison with TensorFlow is
``test_spbf_equivariant_ind_golden.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
)
from tensorpotential.torch_backend.instructions.spbf_equivariant_ind import (
    SingleParticleBasisFunctionEquivariantInd,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.cg_oracle import real_harmonics, wigner_d
from tests_torch.coupling_frames import uncoupled
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261018
CLS = COMPUTE + "SingleParticleBasisFunctionEquivariantInd"
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_ATOMS, N_RAD = 4, 3
ANGULAR, INDICATOR = uncoupled(2), uncoupled(1)
AVG = 2.5


def _spec(**options: object) -> InstructionSpec:
    given = {
        "angular": InstructionRef("Y"),
        "indicator": InstructionRef("I"),
        "radial": InstructionRef("R"),
        "lmax": 2,
        "Lmax": 3,
        "avg_n_neigh": AVG,
        "normalize": True,
        **options,
    }
    return InstructionSpec(
        name="YI",
        cls=CLS,
        options={**DEFAULTS[CLS], **given},
        depends_on=("R", "Y", "I"),
        index=3,
    )


def _twin(policy=F64, angular=ANGULAR, radial_lmax=2, **options: object):
    return SingleParticleBasisFunctionEquivariantInd(
        _spec(**options), angular, INDICATOR, radial_lmax=radial_lmax, policy=policy
    )


def _bonds(rng, n_bonds=10):
    """Random bonds: every atom has at least one."""
    ind_i = np.concatenate([
        np.arange(N_ATOMS),
        rng.integers(0, N_ATOMS, n_bonds - N_ATOMS),
    ])
    return ind_i, rng.integers(0, N_ATOMS, n_bonds), rng.normal(size=(n_bonds, 3))


def _harmonics(vectors: np.ndarray) -> np.ndarray:
    """Return the real harmonics ``[bonds, 9]`` of the directions of ``vectors`` up to ``l = 2``, from scipy."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.concatenate([real_harmonics(unit, l) for l in range(3)], axis=1)


def _inputs(rng, lm_y=9, dtype=np.float64, radial_dtype=None):
    ind_i, ind_j, vectors = _bonds(rng)
    y = _harmonics(vectors)
    return {
        "R": torch.as_tensor(
            rng.normal(size=(len(ind_i), 9, N_RAD)).astype(radial_dtype or dtype)
        ),
        "Y": torch.as_tensor(np.pad(y, ((0, 0), (0, lm_y - 9))).astype(dtype)),
        "I": torch.as_tensor(rng.normal(size=(N_ATOMS, 4, N_RAD)).astype(dtype)),
        "ind_i": torch.as_tensor(ind_i),
        "ind_j": torch.as_tensor(ind_j),
        "atomic_mu_i": torch.zeros(N_ATOMS, dtype=torch.int64),
    }


def _reference(twin, inputs, *, sum_neighbors=True) -> np.ndarray:
    """Sheet section 5, bond by bond and term by term, in float64."""
    plan = twin.plan
    r, y, ind = (inputs[k].double().numpy() for k in ("R", "Y", "I"))
    ind_i, ind_j = inputs["ind_i"].numpy(), inputs["ind_j"].numpy()
    rows = N_ATOMS if sum_neighbors else len(ind_i)
    out = np.zeros((rows, plan.nfunc, N_RAD))
    for b in range(len(ind_i)):
        row = ind_i[b] if sum_neighbors else b
        for (left, right), f, cg in zip(
            plan.lr_inds, plan.m_sum_ind, plan.cg, strict=True
        ):
            out[row, f] += cg * r[b, left] * y[b, left] * ind[ind_j[b], right]
    return out / AVG if sum_neighbors else out


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(("policy", "row"), [(F64, FEATURE_F64), (F32, FEATURE_F32)])
@pytest.mark.parametrize("sum_neighbors", [True, False])
def test_the_forward_pass_is_the_formula_of_the_sheet(policy, row, sum_neighbors):
    twin = _twin(policy, sum_neighbors=sum_neighbors)
    dtype = np.float64 if policy is F64 else np.float32
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=dtype)
    out = twin(inputs)
    assert out.dtype == policy.param
    assert_close(out, _reference(twin, inputs, sum_neighbors=sum_neighbors), row)


def test_an_angular_input_of_higher_degree_is_cut_to_lmax():
    twin = _twin(angular=uncoupled(3), radial_lmax=2)
    assert twin.plan.slice_angular == 9
    inputs = _inputs(np.random.default_rng(RNG_SEED), lm_y=16)
    assert_close(twin(inputs), _reference(twin, inputs), FEATURE_F64)


def test_dense_nbr_and_lm_first_change_nothing():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    plain = _twin()(inputs)
    for options in ({"dense_nbr": True}, {"dense_nbr": False}, {"lm_first": True}):
        torch.testing.assert_close(_twin(**options)(inputs), plain, rtol=0, atol=0)


def test_padding_bonds_of_the_dense_layout_add_nothing():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    padded = dict(inputs)
    n_pad = 3  # ind_i = ind_j = 0, a radial part that the cutoff envelope made zero
    padded["R"] = torch.cat([
        inputs["R"],
        torch.zeros(n_pad, 9, N_RAD, dtype=torch.float64),
    ])
    padded["Y"] = torch.cat([inputs["Y"], torch.ones(n_pad, 9, dtype=torch.float64)])
    for key in ("ind_i", "ind_j"):
        padded[key] = torch.cat([inputs[key], torch.zeros(n_pad, dtype=torch.int64)])
    twin = _twin(dense_nbr=True)
    torch.testing.assert_close(twin(padded), twin(inputs), rtol=0, atol=0)


def test_a_float64_radial_input_promotes_a_float32_model_as_tf_does():
    inputs = _inputs(
        np.random.default_rng(RNG_SEED), dtype=np.float32, radial_dtype=np.float64
    )
    twin = _twin(F32)
    out = twin(inputs)
    assert out.dtype == torch.float64
    assert_close(out, _reference(twin, inputs), FEATURE_F32)


def test_a_float64_indicator_promotes_a_float32_radial_input_as_tf_does():
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=np.float32)
    inputs["I"] = inputs["I"].double()
    twin = _twin(F32)
    out = twin(inputs)
    assert out.dtype == torch.float64
    assert_close(out, _reference(twin, inputs), FEATURE_F32)


def test_an_atom_without_bonds_gets_a_row_of_zeros():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    lonely = N_ATOMS - 1
    keep = inputs["ind_i"] != lonely
    for key in ("R", "Y", "ind_i", "ind_j"):
        inputs[key] = inputs[key][keep]
    twin = _twin()
    out = twin(inputs)
    assert out.shape[0] == N_ATOMS
    assert torch.all(out[lonely] == 0.0)
    assert_close(out, _reference(twin, inputs), FEATURE_F64)


def test_the_angular_input_is_cast_to_the_radial_dtype():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs["Y"] = inputs["Y"].float()
    assert _twin()(inputs).dtype == torch.float64


def test_the_twin_has_no_state_and_passes_gradcheck():
    twin = _twin()
    assert twin.state_dict() == {}
    inputs = _inputs(np.random.default_rng(RNG_SEED))

    def run(r: torch.Tensor, y: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        return twin({**inputs, "R": r, "Y": y, "I": ind})

    args = tuple(inputs[k].clone().requires_grad_() for k in ("R", "Y", "I"))
    assert torch.autograd.gradcheck(run, args)


def test_a_chemical_embedding_is_refused():
    with pytest.raises(
        OptionValueError, match="needs the twin of ScalarChemicalEmbedding"
    ):
        _twin(chemical_embedding=InstructionRef("Z"))


def test_option_errors_of_the_plan_reach_the_caller():
    with pytest.raises(OptionValueError, match="do not match"):
        _twin(radial_lmax=1)


@pytest.mark.parametrize(
    ("key", "cut", "match"),
    [
        ("I", (slice(None), slice(0, 3)), r"indicator 'I' must be \[atoms, 4, n\]"),
        ("R", (slice(None), slice(0, 8)), r"radial input 'R' must be \[bonds, 9, n\]"),
    ],
)
def test_inputs_of_another_angular_size_are_refused(key, cut, match):
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs[key] = inputs[key][cut]
    with pytest.raises(ShapeError, match=match):
        _twin()(inputs)


def test_a_floating_neighbour_index_is_refused():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs["ind_j"] = inputs["ind_j"].double()
    with pytest.raises(DtypeError, match="int32 or int64"):
        _twin()(inputs)


# ------------------------------------------------------------------ physical values


def _rotate(
    features: np.ndarray, table: pd.DataFrame, rotation: np.ndarray
) -> np.ndarray:
    """Rotate every ``(l, hist, parity)`` block (``m = -l .. l``, contiguous) of ``[rows, lm, n]``."""
    directions = np.random.default_rng(RNG_SEED).normal(size=(60, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    out = features.copy()
    start = 0
    while start < len(table):
        l = int(table["l"].iloc[start])
        block = slice(start, start + 2 * l + 1)
        out[:, block] = np.einsum(
            "mk,akn->amn", wigner_d(rotation, l, directions), features[:, block]
        )
        start += 2 * l + 1
    return out


def test_rotating_the_bonds_and_the_indicators_rotates_the_output():
    rng = np.random.default_rng(RNG_SEED)
    ind_i, ind_j, vectors = _bonds(rng)
    lengths = np.linalg.norm(vectors, axis=1)
    # a radial part of the bond length and of l only (one value for the 2l + 1 columns of l), so it is invariant
    per_l = np.cos(np.outer(lengths, np.arange(1, 3 * N_RAD + 1))).reshape(-1, 3, N_RAD)
    radial = np.repeat(per_l, [1, 3, 5], axis=1)
    indicator = rng.normal(size=(N_ATOMS, 4, N_RAD))
    twin = _twin()
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()

    def output(vec: np.ndarray, ind: np.ndarray) -> np.ndarray:
        inputs = {
            "R": radial,
            "Y": _harmonics(vec),
            "I": ind,
            "ind_i": ind_i,
            "ind_j": ind_j,
        }
        tensors = {k: torch.as_tensor(v) for k, v in inputs.items()}
        tensors["atomic_mu_i"] = torch.zeros(N_ATOMS, dtype=torch.int64)
        return twin(tensors).numpy()

    before = output(vectors, indicator)
    after = output(vectors @ rotation.T, _rotate(indicator, INDICATOR, rotation))
    assert_close(after, _rotate(before, twin.coupling_meta_data, rotation), FEATURE_F64)
    assert np.abs(before).max() > 1e-2
