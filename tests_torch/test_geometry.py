"""The geometry twins: logic against the formulas of the sheets, and the physics of lengths, directions and harmonics.

Logic: ``BondLength`` is ``sqrt(|v|**2 + 1e-10)`` (the zero vector has length ``1e-5`` and a zero gradient), read from
``bond_vector`` or a named entry; ``ScaledBondVector`` is ``v / d`` (the zero vector stays zero); ``SphericalHarmonic``
gives the hand values of the sheet for the axes and the zero vector, ``Y_00 = 1`` (``1 / sqrt(4 pi)`` with ``norm``),
float64 for a float32 model, TF's column table, and refuses ``type: complex``. ``RadialBasis`` is tested in
``test_radial_basis.py``. Physical values: lengths are invariant and directions covariant under a rotation, directions
are unit vectors, and the harmonics obey the addition theorem ``sum_m Y_lm(a) Y_lm(b) = (2 l + 1) P_l(a . b)``
(``scipy.special.eval_legendre``, independent of the twin's recursion), hence are rotation invariant in that sum.
The comparison with TensorFlow is ``test_geometry_golden.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
)
from tensorpotential.torch_backend.instructions.geometry import (
    BondLength,
    ScaledBondVector,
    SphericalHarmonic,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden.tolerances import FEATURE_F64, assert_close

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_BONDS = 9


def _spec(cls: str, name: str, **options: Any) -> InstructionSpec:
    return InstructionSpec(
        name=name,
        cls=COMPUTE + cls,
        options={**DEFAULTS.get(COMPUTE + cls, {}), **options},
        depends_on=(),
        index=0,
    )


def _length(**options: Any) -> BondLength:
    return BondLength(_spec("BondLength", "d", **options))


def _direction(**options: Any) -> ScaledBondVector:
    return ScaledBondVector(
        _spec(
            "ScaledBondVector",
            "vhat",
            **{"bond_length": InstructionRef("d"), **options},
        )
    )


def _harmonics(lmax=4, policy=F64, **options: Any) -> SphericalHarmonic:
    spec = _spec(
        "SphericalHarmonic",
        "Y",
        **{"vhat": InstructionRef("vhat"), "lmax": lmax, **options},
    )
    return SphericalHarmonic(spec, policy)


def _vectors(rng, n=N_BONDS) -> torch.Tensor:
    return torch.as_tensor(rng.normal(size=(n, 3)) * 2.0)


def _chain(v: torch.Tensor, lmax=4) -> dict[str, torch.Tensor]:
    data = {"bond_vector": v}
    data["d"] = _length()(data)
    data["vhat"] = _direction()(data)
    data["Y"] = _harmonics(lmax)(data)
    return data


# ------------------------------------------------------------------ logic


def test_the_length_is_softened_by_1e_10_under_the_root():
    v = _vectors(np.random.default_rng(RNG_SEED))
    expected = np.sqrt((v.numpy() ** 2).sum(axis=1, keepdims=True) + 1e-10)
    out = _length()({"bond_vector": v})
    assert out.shape == (N_BONDS, 1) and out.dtype == torch.float64
    np.testing.assert_array_equal(out.numpy(), expected)


def test_the_zero_vector_has_length_1e_5_and_a_zero_gradient():
    v = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    out = _length()({"bond_vector": v})
    assert float(out.detach()) == pytest.approx(1e-5, rel=1e-12)
    (grad,) = torch.autograd.grad(out.sum(), v)
    assert torch.count_nonzero(grad) == 0


def test_lengths_and_directions_can_read_the_vectors_from_a_named_entry():
    v = _vectors(np.random.default_rng(RNG_SEED))
    d = _length(instruction_with_bonds=InstructionRef("other"))({"other": v})
    assert torch.equal(d, _length()({"bond_vector": v}))
    direction = _direction(bonds=InstructionRef("other"))
    assert torch.equal(direction({"other": v, "d": d}), v / d)


def test_integer_vectors_are_refused():
    with pytest.raises(DtypeError, match="floating"):
        _length()({"bond_vector": torch.ones((2, 3), dtype=torch.int64)})


def test_the_direction_is_the_vector_over_its_softened_length():
    v = _vectors(np.random.default_rng(RNG_SEED))
    data = _chain(v)
    np.testing.assert_array_equal(data["vhat"].numpy(), v.numpy() / data["d"].numpy())
    zero = {"bond_vector": torch.zeros((1, 3), dtype=torch.float64)}
    zero["d"] = _length()(zero)
    assert torch.count_nonzero(_direction()(zero)) == 0


def test_a_direction_without_lengths_is_refused():
    with pytest.raises(OptionValueError, match="bond_length is null"):
        _direction(bond_length=None)


SQRT3, SQRT5, SQRT15 = math.sqrt(3), math.sqrt(5), math.sqrt(15)
HAND_VALUES = {  # sheet, section 5, lmax = 2
    (0.0, 0.0, 1.0): [1, 0, SQRT3, 0, 0, 0, SQRT5, 0, 0],
    (1.0, 0.0, 0.0): [1, 0, 0, SQRT3, 0, 0, -SQRT5 / 2, 0, SQRT15 / 2],
    (0.0, 1.0, 0.0): [1, SQRT3, 0, 0, 0, 0, -SQRT5 / 2, 0, -SQRT15 / 2],
    (0.0, 0.0, 0.0): [1, 0, 0, 0, 0, 0, -SQRT5 / 2, 0, 0],
}


@pytest.mark.parametrize("direction", list(HAND_VALUES))
def test_the_harmonics_have_the_hand_values_of_the_sheet(direction):
    vhat = torch.tensor([direction], dtype=torch.float64)
    out = _harmonics(lmax=2)({"vhat": vhat})
    np.testing.assert_allclose(out.numpy()[0], HAND_VALUES[direction], atol=1e-14)


def test_norm_leaves_the_orthonormal_harmonics():
    vhat = _chain(_vectors(np.random.default_rng(RNG_SEED)))["vhat"]
    plain = _harmonics()({"vhat": vhat})
    normed = _harmonics(norm=True)({"vhat": vhat})
    assert_close(normed, plain.numpy() / math.sqrt(4 * math.pi), FEATURE_F64, "norm")
    assert _harmonics(norm=False)({"vhat": vhat}).equal(plain)


def test_the_harmonics_are_float64_for_a_float32_model():
    vhat = _chain(_vectors(np.random.default_rng(RNG_SEED)))["vhat"]
    out = _harmonics(policy=F32)({"vhat": vhat.float()})
    assert out.dtype == torch.float64
    assert_close(
        out, _harmonics()({"vhat": vhat.float().double()}).numpy(), FEATURE_F64, "f32"
    )


def test_the_column_table_lists_l_m_and_the_parity():
    meta = _harmonics(lmax=2).coupling_meta_data
    assert list(meta.columns) == ["l", "m", "hist", "parity", "sum_of_ls"]
    assert meta["l"].tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert meta["m"].tolist() == [0, -1, 0, 1, -2, -1, 0, 1, 2]
    assert meta["parity"].tolist() == [1, -1, -1, -1, 1, 1, 1, 1, 1]
    assert meta["sum_of_ls"].tolist() == meta["l"].tolist()
    assert set(meta["hist"]) == {""}


def test_complex_harmonics_a_null_direction_and_a_negative_lmax_are_refused():
    with pytest.raises(OptionValueError, match="type 'complex' is not supported"):
        _harmonics(type="complex")
    with pytest.raises(OptionValueError, match="vhat is null"):
        _harmonics(vhat=None)
    with pytest.raises(ShapeError, match="lmax"):
        _harmonics(lmax=-1)
    assert _harmonics(type="real").plan.lmax == 4


# ------------------------------------------------------------------ physical values


def test_lengths_are_invariant_and_directions_covariant_under_a_rotation():
    rng = np.random.default_rng(RNG_SEED)
    v = _vectors(rng)
    rotation = torch.as_tensor(Rotation.random(random_state=RNG_SEED).as_matrix())
    data, turned = _chain(v), _chain(v @ rotation.T)
    assert_close(turned["d"], data["d"].numpy(), FEATURE_F64, "length")
    assert_close(
        turned["vhat"], (data["vhat"] @ rotation.T).numpy(), FEATURE_F64, "direction"
    )


def test_directions_are_unit_vectors_to_the_softening():
    data = _chain(_vectors(np.random.default_rng(RNG_SEED)))
    norms = data["vhat"].norm(dim=1).numpy()
    r = np.linalg.norm(data["bond_vector"].numpy(), axis=1)
    np.testing.assert_allclose(norms, r / np.sqrt(r**2 + 1e-10), rtol=1e-15)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_the_harmonics_obey_the_addition_theorem():
    rng = np.random.default_rng(RNG_SEED)
    lmax = 4
    a = _chain(_vectors(rng), lmax)
    b = _chain(_vectors(rng), lmax)
    cosines = (a["vhat"] * b["vhat"]).sum(dim=1).numpy()
    ya, yb = a["Y"].numpy(), b["Y"].numpy()
    for l in range(lmax + 1):
        cols = slice(l * l, (l + 1) ** 2)
        summed = (ya[:, cols] * yb[:, cols]).sum(axis=1)
        np.testing.assert_allclose(
            summed,
            (2 * l + 1) * eval_legendre(l, cosines),
            atol=1e-12,
            err_msg=f"l={l}",
        )
