"""The ``RadialBasis`` twin: the instruction around the bases of ``ops/radial.py`` (tested in ``test_radial.py``).

Logic: each ``basis_type`` evaluates its basis with the keywords the yaml gives, on the stored ``rc`` (rule R6: a loaded
``rc`` that differs from ``rcut`` is the one used, and it keeps the parameter dtype), the Gaussian basis on its stored
``grid`` and ``scale``; the weights are named as in the checkpoint; a keyword the basis does not read is refused unless
it is at its pinned default (owner decision, sheet section 7), with the Gaussian allow-list; an unknown ``basis_type``
is refused. Physical values: the basis is a function of the bond length only (invariant under a rotation of the bonds)
and vanishes beyond the cutoff. The comparison with TensorFlow is ``test_geometry_golden.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.instructions.radial_basis import RadialBasis
from tensorpotential.torch_backend.ops import radial
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.golden.tolerances import FEATURE_F64, assert_close

RNG_SEED = 20261019
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
RCUT = 6.0


def _twin(policy=F64, **options: Any) -> RadialBasis:
    cls = COMPUTE + "RadialBasis"
    given = {
        "bonds": InstructionRef("d"),
        "basis_type": "Cheb",
        "nfunc": 6,
        "rcut": RCUT,
    }
    spec = InstructionSpec(
        name="RadialBasis",
        cls=cls,
        options={**DEFAULTS[cls], **given, **options},
        depends_on=(),
        index=0,
    )
    return RadialBasis(spec, policy)


def _lengths(rng, n=11) -> torch.Tensor:
    return torch.as_tensor(rng.uniform(0.5, RCUT + 1.0, size=(n, 1)))


RC = torch.tensor(RCUT, dtype=torch.float64)
GRID = torch.linspace(0.0, RCUT, 6, dtype=torch.float64).reshape(1, 6)
SCALE = torch.tensor(-0.5, dtype=torch.float64)
CASES = {
    "Cheb": (
        {"p": 7, "kind": 2, "reversed": True, "normalized": True},
        lambda r: radial.Chebyshev(
            6, RCUT, p=7, kind=2, reversed=True, normalized=True
        )(r, RC),
    ),
    "RadSinBessel": (
        {"p": 7, "normalized": True},
        lambda r: radial.SineBessel(6, RCUT, p=7, normalized=True)(r, RC),
    ),
    "SBessel": ({}, lambda r: radial.SimplifiedBessel(6, RCUT)(r, RC)),
    "Gaussian": ({"p": 7}, lambda r: radial.Gaussian(6, RCUT, p=7)(r, RC, GRID, SCALE)),
}


@pytest.mark.parametrize("basis_type", list(CASES))
def test_each_basis_type_evaluates_its_basis_with_the_keywords_of_the_yaml(basis_type):
    keywords, expected = CASES[basis_type]
    twin = _twin(basis_type=basis_type, **keywords)
    if basis_type == "Gaussian":
        twin.load_state_dict({"rc": RC, "grid": GRID, "scale": SCALE}, strict=True)
    r = _lengths(np.random.default_rng(RNG_SEED))
    out = twin({"d": r})
    assert out.shape == (11, 6) and out.dtype == torch.float64
    assert torch.equal(out, expected(r))
    assert twin.nfunc == 6


def test_the_weights_are_named_as_in_the_checkpoint_and_keep_the_parameter_dtype():
    assert list(_twin().state_dict()) == ["rc"]
    gaussian = _twin(F32, basis_type="Gaussian")
    assert list(gaussian.state_dict()) == ["rc", "grid", "scale"]
    assert {v.dtype for v in gaussian.state_dict().values()} == {torch.float32}
    assert gaussian.get_buffer("grid").shape == (1, 6)


def test_the_stored_cutoff_is_used_not_the_yaml_one():
    twin = _twin(F32, rcut=5.6)
    assert twin.cutoff() == pytest.approx(5.6, abs=1e-6)
    assert twin.cutoff() == float(np.float32(5.6))
    twin.load_state_dict({"rc": torch.tensor(5.0, dtype=torch.float32)})
    assert twin.cutoff() == 5.0
    r = torch.tensor([[2.0], [5.5]], dtype=torch.float64)
    expected = radial.Chebyshev(6, 5.6)(r, torch.tensor(5.0, dtype=torch.float32))
    assert torch.equal(twin({"d": r}), expected)


@pytest.mark.parametrize(
    ("basis_type", "options"),
    [
        ("SBessel", {"p": 7}),
        ("SBessel", {"normalized": True}),
        ("RadSinBessel", {"kind": 2}),
        ("Gaussian", {"normalized": True}),
        ("Cheb", {"rmin": 1.0}),
        ("Cheb", {"normalised": True}),
    ],
)
def test_a_keyword_the_basis_does_not_read_is_refused_away_from_its_default(
    basis_type, options
):
    (key,) = options
    with pytest.raises(OptionValueError, match=f"does not read {key}="):
        _twin(basis_type=basis_type, **options)


def test_pinned_defaults_and_the_gaussian_allow_list_are_accepted():
    assert isinstance(_twin(basis_type="SBessel", p=5).basis, radial.SimplifiedBessel)
    gaussian = _twin(basis_type="Gaussian", rmin=0.5, init_gamma=2.0, trainable=True)
    assert isinstance(gaussian.basis, radial.Gaussian)


def test_an_unknown_basis_type_and_values_the_basis_refuses_are_refused():
    with pytest.raises(OptionValueError, match="unknown radial basis_type 'Bessel'"):
        _twin(basis_type="Bessel")
    with pytest.raises(OptionValueError, match="kind must be 1 or 2"):
        _twin(kind=3)
    with pytest.raises(OptionValueError, match="nfunc=0"):
        _twin(nfunc=0)


# ------------------------------------------------------------------ physical values


def test_the_basis_depends_on_the_bond_length_only():
    rng = np.random.default_rng(RNG_SEED)
    v = torch.as_tensor(rng.normal(size=(11, 3)) * 2.5)
    turned = v @ torch.as_tensor(Rotation.random(random_state=RNG_SEED).as_matrix()).T
    twin = _twin()
    out = twin({"d": v.norm(dim=1, keepdim=True)})
    assert_close(
        twin({"d": turned.norm(dim=1, keepdim=True)}),
        out.numpy(),
        FEATURE_F64,
        "rotated",
    )


@pytest.mark.parametrize("basis_type", list(CASES))
def test_the_basis_vanishes_beyond_the_cutoff(basis_type):
    twin = _twin(basis_type=basis_type, **CASES[basis_type][0])
    r = torch.tensor([[RCUT + 1e-9], [RCUT + 0.5], [20.0]], dtype=torch.float64)
    assert torch.count_nonzero(twin({"d": r})) == 0
