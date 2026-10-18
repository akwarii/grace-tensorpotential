"""The ``SingleParticleBasisFunctionScalarInd`` twin: logic against a loop over bonds, rotation and permutation symmetry.

Logic: the forward pass of the spec sheet (section 5) written as a loop over bonds in numpy, with the weight of the
indicator, ``1 / avg_n_neigh`` and the degrees taken from the options: both parameter dtypes, an angular input of
higher degree, ``indicator_l_depend``, a per-element ``avg_n_neigh``, no indicator, ``sum_neighbors`` off; TF's casts
(``Y`` rounded to the dtype of ``R``, the product cast to the weight dtype); ``lm_first`` changes nothing; the state
and the typed errors. Physical values: rotating the bonds (harmonics recomputed by scipy, ``tests_torch/cg_oracle.py``)
rotates every ``l`` block of the output by its Wigner matrix, and reordering the bonds changes nothing. The comparison
with TensorFlow is ``test_spbf_scalar_ind_golden.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
    SpecError,
)
from tensorpotential.torch_backend.instructions.spbf_scalar_ind import (
    SingleParticleBasisFunctionScalarInd,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.cg_oracle import real_harmonics, wigner_d
from tests_torch.coupling_frames import uncoupled
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261018
CLS = COMPUTE + "SingleParticleBasisFunctionScalarInd"
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_ATOMS, N_ELEMENTS, N_RAD, EMB, N_BONDS = 4, 3, 3, 5, 12
LMAX = 2
AVG = 2.5


def _spec(**options: Any) -> InstructionSpec:
    given = {
        "radial": InstructionRef("R"),
        "angular": InstructionRef("Y"),
        "indicator": InstructionRef("Z"),
        "avg_n_neigh": AVG,
        **options,
    }
    return InstructionSpec(
        name="A",
        cls=CLS,
        options={**DEFAULTS[CLS], **given},
        depends_on=("R", "Y", "Z"),
        index=3,
    )


def _weight(l_depend: bool, lmax: int = LMAX) -> np.ndarray:
    width = N_RAD * (lmax + 1) if l_depend else N_RAD
    return np.random.default_rng(RNG_SEED + 7).normal(size=(EMB, width))


def _twin(
    policy=F64,
    angular_lmax=LMAX,
    radial_lmax=LMAX,
    embedding_size: int | None = EMB,
    **options: Any,
):
    twin = SingleParticleBasisFunctionScalarInd(
        _spec(**options),
        uncoupled(angular_lmax),
        radial_lmax=radial_lmax,
        n_rad_max=N_RAD,
        embedding_size=embedding_size,
        policy=policy,
    )
    if twin.lin_transform is not None:
        lmax = options.get("lmax")
        lmax = angular_lmax if lmax is None else lmax
        w = _weight(bool(options.get("indicator_l_depend")), lmax)
        twin.load_state_dict(
            {"lin_transform.w": torch.as_tensor(w, dtype=policy.param)}, strict=True
        )
    return twin


def _harmonics(vectors: np.ndarray, lmax: int = LMAX) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.concatenate([real_harmonics(unit, l) for l in range(lmax + 1)], axis=1)


def _bonds(rng):
    """Random bonds; every atom has at least one."""
    ind_i = np.concatenate([
        np.arange(N_ATOMS),
        rng.integers(0, N_ATOMS, N_BONDS - N_ATOMS),
    ])
    return ind_i, rng.integers(0, N_ELEMENTS, N_BONDS), rng.normal(size=(N_BONDS, 3))


def _inputs(rng, angular_lmax=LMAX, dtype=np.float64, angular_dtype=None):
    ind_i, mu_j, vectors = _bonds(rng)
    n_lm = (LMAX + 1) ** 2
    return {
        "R": torch.as_tensor(rng.normal(size=(N_BONDS, n_lm, N_RAD)).astype(dtype)),
        "Y": torch.as_tensor(
            _harmonics(vectors, angular_lmax).astype(angular_dtype or dtype)
        ),
        "Z": torch.as_tensor(rng.normal(size=(N_ELEMENTS, EMB)).astype(dtype)),
        "ind_i": torch.as_tensor(ind_i),
        "mu_j": torch.as_tensor(mu_j),
        "atomic_mu_i": torch.as_tensor(np.arange(N_ATOMS) % N_ELEMENTS),
    }


def _reference(
    inputs,
    *,
    indicator=True,
    l_depend=False,
    sum_neighbors=True,
    inv_avg=1 / AVG,
    y_dtype=np.float64,
):
    """Sheet section 5, bond by bond, in float64 (``Y`` first rounded to ``y_dtype``)."""
    r = inputs["R"].double().numpy()
    n_lm = r.shape[1]
    y = inputs["Y"].numpy()[:, :n_lm].astype(y_dtype).astype(np.float64)
    l_of_column = np.concatenate([[l] * (2 * l + 1) for l in range(LMAX + 1)])
    ind_i, mu_j = inputs["ind_i"].numpy(), inputs["mu_j"].numpy()
    if indicator:
        zt = inputs["Z"].double().numpy() @ (_weight(l_depend) / np.sqrt(EMB))
        if l_depend:
            zt = zt.reshape(N_ELEMENTS, N_RAD, LMAX + 1)[:, :, l_of_column].transpose(
                0, 2, 1
            )
        else:
            zt = np.repeat(zt[:, None, :], n_lm, axis=1)
    bonds = np.zeros_like(r)
    for b in range(len(ind_i)):
        bonds[b] = r[b] * y[b][:, None] * (zt[mu_j[b]] if indicator else 1.0)
    if not sum_neighbors:
        return bonds
    out = np.zeros((N_ATOMS, n_lm, N_RAD))
    for b, i in enumerate(ind_i):
        out[i] += bonds[b]
    scale = np.asarray(inv_avg)
    return out * (
        scale[inputs["atomic_mu_i"].numpy()][:, :, None] if scale.ndim else scale
    )


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(("policy", "row"), [(F64, FEATURE_F64), (F32, FEATURE_F32)])
def test_the_forward_pass_is_the_formula_of_the_sheet(policy, row):
    rng = np.random.default_rng(RNG_SEED)
    dtype = np.float64 if policy is F64 else np.float32
    inputs = _inputs(rng, dtype=dtype)
    out = _twin(policy)(inputs)
    assert out.dtype == policy.param and out.shape == (N_ATOMS, 9, N_RAD)
    assert_close(out, _reference(inputs), row)


def test_an_angular_input_of_higher_degree_is_cut():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng, angular_lmax=3)
    twin = _twin(angular_lmax=3, lmax=LMAX)
    assert len(twin.coupling_meta_data) == 9
    assert_close(twin(inputs), _reference(inputs), FEATURE_F64)


def test_an_l_dependent_indicator_weights_each_degree():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    twin = _twin(indicator_l_depend=True)
    assert twin.state_dict()["lin_transform.w"].shape == (EMB, N_RAD * (LMAX + 1))
    assert_close(twin(inputs), _reference(inputs, l_depend=True), FEATURE_F64)


def test_a_per_element_average_scales_by_the_central_element():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    avg = {0: 2.0, 1: 4.0, 2: -1.0}
    out = _twin(avg_n_neigh=avg)(inputs)
    assert_close(out, _reference(inputs, inv_avg=[[0.5], [0.25], [1.0]]), FEATURE_F64)


def test_without_an_indicator_the_bonds_are_not_weighted_and_there_is_no_state():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    twin = _twin(indicator=None, embedding_size=None)
    assert twin.state_dict() == {}
    assert_close(twin(inputs), _reference(inputs, indicator=False), FEATURE_F64)


def test_without_sum_neighbors_one_row_per_bond_is_kept_unaveraged():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    out = _twin(sum_neighbors=False)(inputs)
    assert out.shape == (N_BONDS, 9, N_RAD)
    assert_close(out, _reference(inputs, sum_neighbors=False), FEATURE_F64)


def test_lm_first_changes_nothing():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    torch.testing.assert_close(
        _twin(lm_first=True)(inputs), _twin()(inputs), rtol=0, atol=0
    )


def test_a_float32_model_rounds_the_angular_input_before_the_product():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng, dtype=np.float32, angular_dtype=np.float64)
    twin = _twin(F32)
    # the bond product is R * float32(Y) exactly, not a float64 product rounded afterwards
    exact = inputs["R"] * inputs["Y"].float()[:, :, None]
    torch.testing.assert_close(twin._bond_product(inputs), exact, rtol=0, atol=0)
    out = twin(inputs)
    assert out.dtype == torch.float32
    assert_close(out, _reference(inputs, y_dtype=np.float32), FEATURE_F32)


def test_float64_inputs_of_a_float32_model_give_its_parameter_dtype():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    out = _twin(F32)(inputs)
    assert out.dtype == torch.float32
    assert_close(out, _reference(inputs), FEATURE_F32)


def test_an_atom_without_bonds_gets_a_row_of_zeros():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    inputs["atomic_mu_i"] = torch.cat([
        inputs["atomic_mu_i"],
        torch.zeros(1, dtype=torch.int64),
    ])
    out = _twin()(inputs)
    assert out.shape[0] == N_ATOMS + 1 and bool((out[N_ATOMS] == 0).all())


def test_gradcheck():
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    twin = _twin(indicator_l_depend=True)
    r, z = inputs["R"].requires_grad_(), inputs["Z"].requires_grad_()
    assert torch.autograd.gradcheck(
        lambda r, z: twin({**inputs, "R": r, "Z": z}), (r, z)
    )


def test_an_embedding_size_must_come_with_an_indicator_and_only_then():
    with pytest.raises(
        SpecError, match="indicator is None, but an embedding size of 5"
    ):
        _twin(indicator=None)
    with pytest.raises(SpecError, match="an embedding size of None"):
        _twin(embedding_size=None)


def test_option_errors_of_the_plan_reach_the_caller():
    with pytest.raises(OptionValueError, match="radial part"):
        _twin(radial_lmax=1)
    with pytest.raises(OptionValueError, match="must be a float or a dict"):
        _twin(avg_n_neigh=40)


@pytest.mark.parametrize(
    ("key", "shape"),
    [
        ("R", (N_BONDS, 4, N_RAD)),
        ("R", (N_BONDS, 9)),
        ("Y", (N_BONDS, 4)),
        ("Y", (N_BONDS, 9, 1)),
    ],
)
def test_inputs_of_another_angular_size_are_refused(key, shape):
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs[key] = torch.zeros(shape, dtype=torch.float64)
    with pytest.raises(ShapeError, match=r"must be \[bonds, 9, n\]"):
        _twin()(inputs)


@pytest.mark.parametrize("key", ["mu_j", "atomic_mu_i"])
def test_a_floating_element_index_is_refused(key):
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs[key] = inputs[key].double()
    with pytest.raises(DtypeError, match="int32 or int64"):
        _twin(avg_n_neigh={0: 1.0, 1: 2.0, 2: 3.0})(inputs)


# ------------------------------------------------------------------ physical values


def _rotate_blocks(features: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate every ``l`` block of ``[atoms, lm, n]`` by its Wigner matrix."""
    fit = np.random.default_rng(RNG_SEED).normal(size=(60, 3))
    fit /= np.linalg.norm(fit, axis=1, keepdims=True)
    out = features.copy()
    for l in range(LMAX + 1):
        block = slice(l * l, (l + 1) ** 2)
        out[:, block] = np.einsum(
            "mk,akn->amn", wigner_d(rotation, l, fit), features[:, block]
        )
    return out


def _physical_inputs(
    vectors: np.ndarray, ind_i: np.ndarray, mu_j: np.ndarray, z: np.ndarray
):
    lengths = np.linalg.norm(vectors, axis=1)
    # a radial part of the bond length and of l only (one value for the 2l + 1 columns of l), so it is invariant
    per_l = np.cos(np.outer(lengths, np.arange(1, 3 * N_RAD + 1))).reshape(-1, 3, N_RAD)
    return {
        "R": torch.as_tensor(np.repeat(per_l, [1, 3, 5], axis=1)),
        "Y": torch.as_tensor(_harmonics(vectors)),
        "Z": torch.as_tensor(z),
        "ind_i": torch.as_tensor(ind_i),
        "mu_j": torch.as_tensor(mu_j),
        "atomic_mu_i": torch.zeros(N_ATOMS, dtype=torch.int64),
    }


@pytest.mark.parametrize("l_depend", [False, True])
def test_rotating_the_bonds_rotates_every_block_and_their_order_does_not_matter(
    l_depend,
):
    rng = np.random.default_rng(RNG_SEED)
    ind_i, mu_j, vectors = _bonds(rng)
    z = rng.normal(size=(N_ELEMENTS, EMB))
    twin = _twin(indicator_l_depend=l_depend)
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    with torch.no_grad():
        before = twin(_physical_inputs(vectors, ind_i, mu_j, z)).numpy()
        after = twin(_physical_inputs(vectors @ rotation.T, ind_i, mu_j, z)).numpy()
        order = rng.permutation(N_BONDS)
        shuffled = twin(
            _physical_inputs(vectors[order], ind_i[order], mu_j[order], z)
        ).numpy()
    assert_close(after, _rotate_blocks(before, rotation), FEATURE_F64)
    assert_close(shuffled, before, FEATURE_F64)
    assert np.abs(before).max() > 1e-2
