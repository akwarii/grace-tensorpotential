"""The ``ProductFunction`` twin: logic against a loop over the coupling rows, and rotation equivariance.

Logic: the forward pass of the spec sheet (section 5) written in numpy as a loop over the rows of the twin's coupling
table and the pairs of each row (not over the flattened plan), in both parameter dtypes, for distinct operands and for
a square (``is_left_right_equal``); TF's promotion; ``lm_first`` changes nothing; no state; typed errors. Physical
values: real harmonics of a few directions per atom (scipy, ``tests_torch/cg_oracle.py``) give the basis ``A``; rotating
the directions rotates every ``L`` block of ``AA`` and of ``AAA = AA x A`` by its Wigner matrix, and their ``L = 0``
functions are invariant. The comparison with TensorFlow is ``test_product_function_golden.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.instructions.product_function import (
    ProductFunction,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.cg_oracle import real_harmonics, wigner_d
from tests_torch.coupling_frames import coupled, uncoupled
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261018
CLS = COMPUTE + "ProductFunction"
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_ATOMS, N_CH, LMAX = 5, 3, 2
A = uncoupled(LMAX)
AA = coupled(LMAX, LMAX)


def _spec(name: str = "AB", left: str = "A", right: str = "B", **options: object):
    given = {
        "left": InstructionRef(left),
        "right": InstructionRef(right),
        "lmax": LMAX,
        "Lmax": LMAX,
        "normalize": True,
        **options,
    }
    return InstructionSpec(
        name=name,
        cls=CLS,
        options={**DEFAULTS[CLS], **given},
        depends_on=tuple(dict.fromkeys((left, right))),
        index=2,
    )


def _twin(policy=F64, left_meta=A, right_meta=A, **spec_args: Any) -> ProductFunction:
    return ProductFunction(_spec(**spec_args), left_meta, right_meta, policy)


def _features(rng, n_lm: int, dtype=np.float64) -> torch.Tensor:
    return torch.as_tensor(rng.normal(size=(N_ATOMS, n_lm, N_CH)).astype(dtype))


def _reference(twin: ProductFunction, left: torch.Tensor, right: torch.Tensor):
    """Sheet section 5, row by row and pair by pair, in float64."""
    meta = twin.coupling_meta_data
    lft, rght = left.double().numpy(), right.double().numpy()
    out = np.zeros((N_ATOMS, len(meta), N_CH))
    for f, (lefts, rights, cgs) in enumerate(
        zip(meta["left_inds"], meta["right_inds"], meta["cg_list"], strict=True)
    ):
        for i, j, cg in zip(
            np.ravel(lefts), np.ravel(rights), np.ravel(cgs), strict=True
        ):
            out[:, f] += cg * lft[:, i] * rght[:, j]
    return out


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(("policy", "row"), [(F64, FEATURE_F64), (F32, FEATURE_F32)])
def test_the_forward_pass_is_the_formula_of_the_sheet(policy, row):
    rng = np.random.default_rng(RNG_SEED)
    dtype = np.float64 if policy is F64 else np.float32
    twin = _twin(policy, left_meta=AA, right_meta=A, Lmax=1)
    left, right = _features(rng, len(AA), dtype), _features(rng, len(A), dtype)
    out = twin({"A": left, "B": right})
    assert out.dtype == policy.param
    assert out.shape == (N_ATOMS, len(twin.coupling_meta_data), N_CH)
    assert_close(out, _reference(twin, left, right), row)


def test_a_square_uses_the_symmetric_table():
    rng = np.random.default_rng(RNG_SEED)
    square = _twin(right="A")
    distinct = _twin()
    assert len(square.coupling_meta_data) < len(distinct.coupling_meta_data)
    a = _features(rng, len(A))
    out = square({"A": a})
    assert_close(out, _reference(square, a, a), FEATURE_F64)
    # the symmetric table spans the same functions: every distinct-operand row of l1 >= l2 appears once
    assert set(square.coupling_meta_data["hist"]) <= set(
        distinct.coupling_meta_data["hist"]
    )


def test_lm_first_changes_nothing():
    rng = np.random.default_rng(RNG_SEED)
    inputs = {"A": _features(rng, len(A)), "B": _features(rng, len(A))}
    expected = _twin()(inputs)
    torch.testing.assert_close(_twin(lm_first=True)(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("left", "right"), [(np.float64, np.float32), (np.float32, np.float64)]
)
def test_a_float64_operand_promotes_a_float32_model_as_tf_does(left, right):
    rng = np.random.default_rng(RNG_SEED)
    twin = _twin(F32)
    a, b = _features(rng, len(A), left), _features(rng, len(A), right)
    out = twin({"A": a, "B": b})
    assert out.dtype == torch.float64
    assert_close(out, _reference(twin, a, b), FEATURE_F32)


def test_the_twin_has_no_state_and_passes_gradcheck():
    twin = _twin(Lmax=1)
    assert twin.state_dict() == {}
    rng = np.random.default_rng(RNG_SEED)
    args = (
        _features(rng, len(A)).requires_grad_(),
        _features(rng, len(A)).requires_grad_(),
    )
    assert torch.autograd.gradcheck(lambda a, b: twin({"A": a, "B": b}), args)


def test_no_surviving_coupling_is_refused():
    with pytest.raises(ShapeError, match="no row"):
        _twin(Lmax=0, keep_parity=[[0, -1]])


@pytest.mark.parametrize(
    ("shapes", "match"),
    [
        (
            ((N_ATOMS, 4, N_CH), (N_ATOMS, 9, N_CH)),
            r"left operand 'A' must be \[atoms, 9, n\]",
        ),
        (
            ((N_ATOMS, 9, N_CH), (N_ATOMS, 9)),
            r"right operand 'B' must be \[atoms, 9, n\]",
        ),
        (((N_ATOMS, 9, N_CH), (N_ATOMS, 9, 1)), "same atoms and channels"),
        (((N_ATOMS, 9, N_CH), (N_ATOMS + 1, 9, N_CH)), "same atoms and channels"),
    ],
)
def test_operands_of_another_shape_are_refused(shapes, match):
    inputs = {
        k: torch.zeros(s, dtype=torch.float64)
        for k, s in zip("AB", shapes, strict=True)
    }
    with pytest.raises(ShapeError, match=match):
        _twin()(inputs)


# ------------------------------------------------------------------ physical values


def _basis(directions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``A[a, lm, n] = sum_k weights[a, k, l, n] Y_lm(directions[a, k])``: an atomic basis of real harmonics."""
    per_l = [
        np.einsum(
            "akm,akn->amn",
            real_harmonics(directions.reshape(-1, 3), l).reshape(
                *directions.shape[:2], 2 * l + 1
            ),
            weights[:, :, l],
        )
        for l in range(LMAX + 1)
    ]
    return np.concatenate(per_l, axis=1)


def _rotate(
    features: np.ndarray, table: pd.DataFrame, rotation: np.ndarray
) -> np.ndarray:
    """Rotate every ``(l, parity, hist)`` block (``m = -l .. l``, contiguous) of ``[atoms, lm, n]``."""
    fit = np.random.default_rng(RNG_SEED).normal(size=(60, 3))
    fit /= np.linalg.norm(fit, axis=1, keepdims=True)
    out = features.copy()
    start = 0
    while start < len(table):
        l = int(table["l"].iloc[start])
        block = slice(start, start + 2 * l + 1)
        out[:, block] = np.einsum(
            "mk,akn->amn", wigner_d(rotation, l, fit), features[:, block]
        )
        start += 2 * l + 1
    return out


def test_rotating_the_directions_rotates_every_block_of_the_products():
    rng = np.random.default_rng(RNG_SEED)
    directions = rng.normal(size=(N_ATOMS, 4, 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    weights = rng.normal(size=(N_ATOMS, 4, LMAX + 1, N_CH))
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    aa = _twin(name="AA", right="A")
    aaa = _twin(
        name="AAA", left="AA", right="A", left_meta=aa.coupling_meta_data, Lmax=1
    )

    def products(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = torch.as_tensor(_basis(dirs, weights))
        out_aa = aa({"A": a})
        return out_aa.numpy(), aaa({"AA": out_aa, "A": a}).numpy()

    before = products(directions)
    after = products(directions @ rotation.T)
    for twin, b, r in zip((aa, aaa), before, after, strict=True):
        table = twin.coupling_meta_data
        assert_close(r, _rotate(b, table, rotation), FEATURE_F64)
        scalars = np.flatnonzero(table["l"].to_numpy() == 0)
        assert_close(r[:, scalars], b[:, scalars], FEATURE_F64)
        assert np.abs(b).max() > 1e-2
