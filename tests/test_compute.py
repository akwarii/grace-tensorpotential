"""Characterization tests for thirteen instructions of ``tensorpotential/instructions/compute.py``.

Units: ``BondLength``, ``ScaledBondVector``, ``SphericalHarmonic``, ``RadialBasis``,
``MLPRadialFunction``, ``MLPRadialFunction_v2``, ``ScalarChemicalEmbedding`` (first part, helpers
``_geo_*``), then ``SingleParticleBasisFunctionScalarInd``, ``SingleParticleBasisFunctionEquivariantInd``,
``ProductFunction``, ``FunctionReduceN``, ``FCRight2Left`` and ``InvariantLayerRMSNorm`` (second part,
helpers ``_eqv_*``). The prefixes keep the helpers of the two parts apart.

Logic layer: every branch of ``__init__``, ``build``, ``frwrd`` and of the LoRA, element-selection and
regularisation methods, every assertion and error with its message, shapes and dtypes (float32 and
float64 parameters), the option values used by the presets and the shipped ``model.yaml`` files,
building twice, no mutation of the inputs. A test whose comment starts "Pins current behaviour (reported
as a finding)" fixes a behaviour that looks like a defect; the findings are on the issue TEST6, and no
library code is changed by these tests.

Physics layer: every expected value comes from an oracle that neither calls nor shares code with the
unit:

* ``BondLength`` and ``ScaledBondVector``: ``numpy.linalg.norm`` and hand vectors (the unit adds ``1e-10``
  under the square root, so the oracle is ``sqrt(r.r + 1e-10)``), unit norm, rotation invariance of the
  length and covariance of the direction, inversion.
* ``SphericalHarmonic``: ``scipy.special.sph_harm_y`` with the real-harmonic convention worked out from
  the code (columns ``l, m = -l..l``, Condon-Shortley phase removed, scaled by ``sqrt(4 pi)`` unless
  ``norm=True``), hand formulas for ``l <= 2``, orthonormality by Gauss-Legendre quadrature, parity
  ``(-1)^l``, the addition theorem and rotation covariance with Wigner matrices built from the scipy
  harmonics.
* ``RadialBasis``: closed forms from ``numpy`` / ``scipy.special`` (sine Bessel and its normalisation by
  ``scipy.integrate.quad``, Chebyshev polynomials of both kinds, Gaussians), orthonormality of the
  simplified spherical Bessel functions against ``r^2 dr``, envelope equal to 1 at 0 and vanishing with
  zero slope and curvature at the cutoff.
* ``MLPRadialFunction`` and ``MLPRadialFunction_v2``: a numpy re-computation of the forward pass from the
  seeded weights and the LoRA update ``W + A B^T``.
* ``ScalarChemicalEmbedding``: exact values of the ``zeros`` / ``delta`` initialisers, the sample standard
  deviation of ``random``, row selection against numpy fancy indexing.
* The neighbour sums, products and contractions: an explicit loop over bonds with real harmonics from
  ``scipy.special``, real Clebsch-Gordan coefficients from ``sympy`` rotated to the real basis by an
  analytic matrix, real Wigner matrices fitted on those harmonics, and explicit numpy re-computations
  from the stored weights; rotation (``out(R x) = D(R) out(x)``), translation (bond vectors are
  differences of positions), permutation of the neighbour order and of the atoms, the exchange symmetry
  of a product of a function with itself; ``InvariantLayerRMSNorm``: numpy re-computation of its three
  types, padding atoms, invariance of the normalised channels under a rescaling of the input.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.special import eval_chebyt, eval_chebyu, eval_legendre, sici, sph_harm_y
from sympy import S
from sympy.physics.quantum.cg import CG

from tensorpotential import constants
from tensorpotential.instructions import (
    FCRight2Left,
    FunctionReduceN,
    InstructionManager,
    InvariantLayerRMSNorm,
    LinearRadialFunction,
    ProductFunction,
    SingleParticleBasisFunctionEquivariantInd,
    SingleParticleBasisFunctionScalarInd,
)
from tensorpotential.instructions.base import TPInstruction
from tensorpotential.instructions.compute import (
    BondLength,
    MLPRadialFunction,
    MLPRadialFunction_v2,
    RadialBasis,
    ScalarChemicalEmbedding,
    ScaledBondVector,
    SphericalHarmonic,
)
from tests.seeded_weights import seed_trainable_variables
from tests.tolerances import (
    COVARIANCE_F64,
    FLOAT32_ELEMENTWISE,
    FLOAT32_NETWORK,
    FLOAT64_ARITHMETIC,
    sample_std_rtol,
)

_GEO_DTYPES = [
    pytest.param(tf.float32, np.float32, id="float32"),
    pytest.param(tf.float64, np.float64, id="float64"),
]
_GEO_LORA = {"rank": 2, "alpha": 1.0}
_GEO_N_SIGMA = (
    5.0  # statistical bounds below are five standard errors, as in sample_std_rtol
)


# --------------------------------------------------------------------------------------
# Helpers for the oracles and the fixtures
# --------------------------------------------------------------------------------------
class _GeoStub(TPInstruction):
    """A bare neighbouring instruction: only a name and an optional ``nfunc`` are read."""

    def __init__(self, name: str, nfunc: int | None = None) -> None:
        super().__init__(name=name)
        if nfunc is not None:
            self.nfunc = nfunc

    def build(self, float_dtype) -> None:
        self.is_built = True

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        return input_data[self.name]


class _GeoPerAtomEmbedding:
    """Stands in for a chemical embedding that is indexed per atom.

    No embedding class of the repository sets ``is_per_atom``; ``MLPRadialFunction`` reads it with
    ``getattr(..., False)``, so only this stub reaches the per-atom gather branches. It supplies
    the three attributes the radial function reads: ``name``, ``embedding_size``, ``is_per_atom``.
    """

    is_per_atom = True

    def __init__(self, name: str, embedding_size: int) -> None:
        self.name = name
        self.embedding_size = embedding_size


def _geo_bond_vectors(
    n: int = 12, seed: int = 0, dtype=np.float64, rmin: float = 0.8, rmax: float = 4.5
) -> np.ndarray:
    """Random bond vectors with lengths in ``[rmin, rmax]``, shape ``[n, 3]``."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return (direction * rng.uniform(rmin, rmax, size=(n, 1))).astype(dtype)


def _geo_unit_vectors(n: int, seed: int = 1) -> np.ndarray:
    """Random unit vectors, shape ``[n, 3]``."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _geo_rotation_matrix(seed: int = 3) -> np.ndarray:
    """A proper rotation (3 x 3) from a fixed seed."""
    return Rotation.random(random_state=seed).as_matrix()


def _geo_real_sh(v: np.ndarray, lmax: int, norm: bool = False) -> np.ndarray:
    """Real spherical harmonics from ``scipy.special.sph_harm_y``, shape ``[n, (lmax+1)^2]``.

    Convention: column ``l^2 + l + m``; ``m > 0`` is ``sqrt(2) (-1)^m Re Y_l^m``, ``m < 0`` is
    ``sqrt(2) (-1)^m Im Y_l^|m|`` (scipy includes the Condon-Shortley phase, the real harmonics do
    not), ``m = 0`` is ``Y_l^0``. Multiplied by ``sqrt(4 pi)`` unless ``norm`` is true.
    """
    theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    out = np.zeros((len(v), (lmax + 1) ** 2))
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            if m < 0:
                val = np.sqrt(2.0) * (-1.0) ** m * y.imag
            elif m == 0:
                val = y.real
            else:
                val = np.sqrt(2.0) * (-1.0) ** m * y.real
            out[:, ell * ell + ell + m] = val
    return out if norm else out * np.sqrt(4.0 * np.pi)


def _geo_envelope(x: np.ndarray, p: int) -> np.ndarray:
    """Polynomial cutoff envelope: 1 at 0, zero value, slope and curvature at 1."""
    return (
        1.0
        - (p + 1) * (p + 2) / 2.0 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2.0 * x ** (p + 2)
    )


def _geo_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# Activations written out in numpy, keyed like ``tensorpotential.functions.nn.ACTIVATION_DICT``;
# ``None`` is the ``silu_n2norm`` default of ``MLPRadialFunction`` (silu times 1.6759).
_GEO_ACTIVATIONS = {
    "tanh": np.tanh,
    "silu": _geo_silu,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    None: lambda x: _geo_silu(x) * 1.6759,
}


def _geo_l_gather(y: np.ndarray, n_rad_max: int, lmax: int) -> np.ndarray:
    """``[n, n_rad_max*(lmax+1)]`` to ``[n, n_rad_max, (lmax+1)^2]``: column ``l`` repeated ``2l+1`` times."""
    y = y.reshape(-1, n_rad_max, lmax + 1)
    cols = np.concatenate([[ell] * (2 * ell + 1) for ell in range(lmax + 1)])
    return y[:, :, cols]


def _geo_effective_weight(layer, lora: bool, mode: str = "lora") -> np.ndarray:
    """``w`` (plus ``A B^T`` or the additive delta when ``lora``) of a layer as numpy."""
    w = layer.w.numpy()
    if not lora:
        return w
    if mode == "lora":
        a, b = (t.numpy() for t in layer.lora_tensors)
        return w + a @ b.T
    return w + layer.lora_tensors[0].numpy()


def _geo_chem(
    n_elem: int = 4, emb: int = 3, name: str = "Z"
) -> ScalarChemicalEmbedding:
    """A built float64 chemical embedding with seeded non-trivial weights."""
    z = ScalarChemicalEmbedding(
        name=name, element_map={f"E{i}": i for i in range(n_elem)}, embedding_size=emb
    )
    z.build(tf.float64)
    seed_trainable_variables(z)
    return z


def _geo_bond_data(n: int, nfunc: int, n_elem: int = 4, seed: int = 5) -> dict:
    """Radial basis values and element / atom indices of ``n`` fake bonds."""
    rng = np.random.default_rng(seed)
    return {
        "basis": rng.normal(size=(n, nfunc)),
        constants.BOND_MU_I: rng.integers(0, n_elem, size=n).astype(np.int32),
        constants.BOND_MU_J: rng.integers(0, n_elem, size=n).astype(np.int32),
        constants.BOND_IND_I: rng.integers(0, 6, size=n).astype(np.int32),
        constants.BOND_IND_J: rng.integers(0, 6, size=n).astype(np.int32),
    }


# --------------------------------------------------------------------------------------
# BondLength
# --------------------------------------------------------------------------------------
def test_bond_length_default_reads_the_bond_vector_entry():
    bl = BondLength()
    bl.build(tf.float64)
    out = bl.frwrd({constants.BOND_VECTOR: _geo_bond_vectors(5)})
    assert out.shape == (5, 1)
    assert out.dtype == tf.float64
    assert bl.name == "BondLength"
    assert bl.is_built


def test_bond_length_reads_the_named_entry_given_as_string_or_instruction():
    r = _geo_bond_vectors(4, seed=2)
    data = {constants.BOND_VECTOR: _geo_bond_vectors(4, seed=9), "my_vectors": r}
    for source in ("my_vectors", _GeoStub("my_vectors")):
        bl = BondLength(instruction_with_bonds=source, name="d")
        bl.build(tf.float64)
        np.testing.assert_allclose(
            bl.frwrd(data).numpy()[:, 0],
            np.sqrt((r**2).sum(1) + 1e-10),
            **FLOAT64_ARITHMETIC._asdict(),
        )


def test_bond_length_rejects_an_unknown_source_type():
    with pytest.raises(TypeError, match="Unexpected type"):
        BondLength(instruction_with_bonds=3)


def test_bond_length_call_stores_under_name_and_refuses_to_overwrite():
    bl = BondLength(name="d_ij")
    bl.build(tf.float64)
    data = {constants.BOND_VECTOR: _geo_bond_vectors(3)}
    assert bl(data) is data
    assert data["d_ij"].shape == (3, 1)
    with pytest.raises(AssertionError, match="already exists"):
        bl(data)


def test_bond_length_build_twice_and_flags_do_not_change_the_result():
    bl = BondLength()
    bl.build(tf.float64)
    bl.build(tf.float32)
    assert bl.is_built
    data = {constants.BOND_VECTOR: _geo_bond_vectors(6)}
    plain = bl.frwrd(data).numpy()
    flagged = bl.frwrd(data, training=True, local=True).numpy()
    np.testing.assert_array_equal(plain, flagged)


def test_bond_length_does_not_modify_its_input():
    bl = BondLength()
    bl.build(tf.float64)
    r = _geo_bond_vectors(6)
    before = r.copy()
    data = {constants.BOND_VECTOR: r}
    bl.frwrd(data)
    np.testing.assert_array_equal(r, before)
    assert set(data) == {constants.BOND_VECTOR}


@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_bond_length_keeps_the_dtype_and_matches_the_exact_formula(tf_dtype, np_dtype):
    # Oracle: the unit adds 1e-10 under the square root, so the exact formula is sqrt(r.r + 1e-10).
    bl = BondLength()
    bl.build(tf_dtype)
    r = _geo_bond_vectors(20, seed=4, dtype=np_dtype)
    out = bl.frwrd({constants.BOND_VECTOR: r})
    assert out.dtype == tf_dtype
    oracle = np.sqrt((r.astype(np.float64) ** 2).sum(axis=1, keepdims=True) + 1e-10)
    tol = FLOAT64_ARITHMETIC if np_dtype is np.float64 else FLOAT32_ELEMENTWISE
    np.testing.assert_allclose(out.numpy(), oracle, **tol._asdict())


def test_bond_length_hand_vectors_and_the_softening_term():
    bl = BondLength()
    bl.build(tf.float64)
    r = np.array([
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 2.0],
        [-1.0, -2.0, -2.0],
        [0.0, 0.0, 0.0],
    ])
    out = bl.frwrd({constants.BOND_VECTOR: r}).numpy()[:, 0]
    # |r| = 5, 2, 3 and 0 by hand; the unit returns sqrt(|r|^2 + 1e-10), so a zero vector gives 1e-5.
    exact = np.sqrt(np.array([25.0, 4.0, 9.0, 0.0]) + 1e-10)
    np.testing.assert_allclose(out, exact, **FLOAT64_ARITHMETIC._asdict())
    np.testing.assert_allclose(out[3], 1e-5, **FLOAT64_ARITHMETIC._asdict())


def test_bond_length_agrees_with_the_norm_up_to_the_softening_and_is_rotation_invariant():
    bl = BondLength()
    bl.build(tf.float64)
    r = _geo_bond_vectors(30, seed=8)
    rot = _geo_rotation_matrix()
    d = bl.frwrd({constants.BOND_VECTOR: r}).numpy()[:, 0]
    d_rot = bl.frwrd({constants.BOND_VECTOR: r @ rot.T}).numpy()[:, 0]
    # sqrt(r.r + 1e-10) - |r| is about 1e-10 / (2 |r|), below 1e-10 for |r| >= 0.8
    np.testing.assert_allclose(d, np.linalg.norm(r, axis=1), **COVARIANCE_F64._asdict())
    np.testing.assert_allclose(d_rot, d, **COVARIANCE_F64._asdict())


def test_bond_length_is_even_in_the_vector_and_has_a_finite_gradient_at_zero():
    bl = BondLength()
    bl.build(tf.float64)
    r = _geo_bond_vectors(7, seed=11)
    d = bl.frwrd({constants.BOND_VECTOR: r}).numpy()
    d_inv = bl.frwrd({constants.BOND_VECTOR: -r}).numpy()
    np.testing.assert_array_equal(d, d_inv)
    x = tf.constant([[0.0, 0.0, 0.0], [0.3, -0.4, 1.2]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = bl.frwrd({constants.BOND_VECTOR: x})
    grad = tape.gradient(y, x).numpy()
    assert np.all(np.isfinite(grad))
    np.testing.assert_array_equal(grad[0], 0.0)
    # d|r|/dr = r / |r| for the second vector (|r|^2 = 1.69), with the same softening
    np.testing.assert_allclose(
        grad[1],
        [0.3, -0.4, 1.2] / np.sqrt(1.69 + 1e-10),
        **FLOAT64_ARITHMETIC._asdict(),
    )


# --------------------------------------------------------------------------------------
# ScaledBondVector
# --------------------------------------------------------------------------------------
def test_scaled_bond_vector_reads_the_default_and_the_named_entries():
    r = _geo_bond_vectors(5)
    d = np.linalg.norm(r, axis=1, keepdims=True)
    sbv = ScaledBondVector(bond_length="d")
    sbv.build(tf.float64)
    assert sbv.bonds == constants.BOND_VECTOR
    assert sbv.bond_length == "d"
    out = np.asarray(sbv.frwrd({constants.BOND_VECTOR: r, "d": d}))
    np.testing.assert_allclose(out, r / d, **FLOAT64_ARITHMETIC._asdict())

    other = _geo_bond_vectors(5, seed=77)
    for bonds in ("v2", _GeoStub("v2")):
        sbv = ScaledBondVector(bond_length=_GeoStub("d"), bonds=bonds)
        assert sbv.bonds == "v2"
        assert sbv.bond_length == "d"
        out = np.asarray(sbv.frwrd({constants.BOND_VECTOR: r, "v2": other, "d": d}))
        np.testing.assert_allclose(out, other / d, **FLOAT64_ARITHMETIC._asdict())


def test_scaled_bond_vector_silently_ignores_arguments_of_an_unknown_type():
    # Pins current behaviour (reported as a finding): a bond_length or bonds of another type than
    # TPInstruction / str is accepted by __init__ and leaves the attribute undefined (bond_length)
    # or replaced by the default (bonds), so the failure comes later, in frwrd, as an
    # AttributeError instead of a TypeError at construction.
    broken_length = ScaledBondVector(bond_length=3.0)
    assert not hasattr(broken_length, "bond_length")
    with pytest.raises(AttributeError):
        broken_length.frwrd({constants.BOND_VECTOR: np.ones((2, 3))})
    broken_bonds = ScaledBondVector(bond_length="d", bonds=3.0)
    assert not hasattr(broken_bonds, "bonds")


def test_scaled_bond_vector_build_twice_and_call_protocol():
    sbv = ScaledBondVector(bond_length="d", name="rhat")
    sbv.build(tf.float64)
    sbv.build(tf.float64)
    assert sbv.is_built
    r = _geo_bond_vectors(4)
    d = np.linalg.norm(r, axis=1, keepdims=True)
    before = (r.copy(), d.copy())
    data = {constants.BOND_VECTOR: r, "d": d}
    assert sbv(data) is data
    assert data["rhat"].shape == (4, 3)
    np.testing.assert_array_equal(r, before[0])
    np.testing.assert_array_equal(d, before[1])
    with pytest.raises(AssertionError, match="already exists"):
        sbv(data)


@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_scaled_bond_vector_dtype_follows_the_inputs(tf_dtype, np_dtype):
    r = _geo_bond_vectors(6, dtype=np_dtype)
    bl = BondLength()
    bl.build(tf_dtype)
    sbv = ScaledBondVector(bond_length=bl)
    sbv.build(tf_dtype)
    out = sbv.frwrd(bl({constants.BOND_VECTOR: r}))
    assert out.dtype == tf_dtype
    assert out.shape == (6, 3)
    tol = FLOAT64_ARITHMETIC if np_dtype is np.float64 else FLOAT32_ELEMENTWISE
    r64 = r.astype(np.float64)
    np.testing.assert_allclose(
        out.numpy(),
        r64 / np.sqrt((r64**2).sum(axis=1, keepdims=True) + 1e-10),
        **tol._asdict(),
    )


def test_scaled_bond_vector_hand_values():
    sbv = ScaledBondVector(bond_length="d")
    sbv.build(tf.float64)
    r = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]])
    d = np.array([[5.0], [2.0]])
    out = np.asarray(sbv.frwrd({constants.BOND_VECTOR: r, "d": d}))
    np.testing.assert_allclose(
        out, [[0.6, 0.8, 0.0], [0.0, 0.0, -1.0]], **FLOAT64_ARITHMETIC._asdict()
    )


def test_scaled_bond_vector_after_bond_length_has_unit_norm_up_to_the_softening():
    # |r / d|^2 = r.r / (r.r + 1e-10) exactly, because BondLength adds 1e-10 under the root.
    bl = BondLength()
    sbv = ScaledBondVector(bond_length=bl)
    bl.build(tf.float64)
    sbv.build(tf.float64)
    r = _geo_bond_vectors(40, seed=21, rmin=0.05)
    rhat = sbv.frwrd(bl({constants.BOND_VECTOR: r})).numpy()
    rr = (r**2).sum(axis=1)
    np.testing.assert_allclose(
        np.linalg.norm(rhat, axis=1),
        np.sqrt(rr / (rr + 1e-10)),
        **FLOAT64_ARITHMETIC._asdict(),
    )
    # for bonds of ordinary length the vector is a unit vector to 1e-9
    r = _geo_bond_vectors(40, seed=22)
    rhat = sbv.frwrd(bl({constants.BOND_VECTOR: r})).numpy()
    np.testing.assert_allclose(
        np.linalg.norm(rhat, axis=1), 1.0, **COVARIANCE_F64._asdict()
    )


def test_scaled_bond_vector_is_covariant_under_rotation_and_odd_under_inversion():
    bl = BondLength()
    sbv = ScaledBondVector(bond_length=bl)
    bl.build(tf.float64)
    sbv.build(tf.float64)
    rot = _geo_rotation_matrix(seed=5)
    r = _geo_bond_vectors(15, seed=6)

    def rhat(vectors: np.ndarray) -> np.ndarray:
        return sbv.frwrd(bl({constants.BOND_VECTOR: vectors})).numpy()

    base = rhat(r)
    np.testing.assert_allclose(
        rhat(r @ rot.T), base @ rot.T, **COVARIANCE_F64._asdict()
    )
    np.testing.assert_allclose(rhat(-r), -base, **FLOAT64_ARITHMETIC._asdict())
    # only the direction is kept: scaling the bond by 3 leaves it unchanged up to the softening
    np.testing.assert_allclose(rhat(3.0 * r), base, **COVARIANCE_F64._asdict())


# --------------------------------------------------------------------------------------
# SphericalHarmonic
# --------------------------------------------------------------------------------------
def _geo_sh(lmax: int, **kwargs) -> SphericalHarmonic:
    y = SphericalHarmonic(vhat="v", lmax=lmax, name="Y", **kwargs)
    y.build(tf.float64)
    return y


def test_spherical_harmonic_arguments_and_meta_data():
    y = SphericalHarmonic(vhat="rhat", lmax=3, name="Y")
    assert y.vhat == "rhat"
    assert y.lmax == 3
    assert y.coupling_origin is None
    meta = y.coupling_meta_data
    assert len(meta) == 16
    assert meta["l"].tolist() == [ell for ell in range(4) for _ in range(2 * ell + 1)]
    assert meta["m"].tolist() == [m for ell in range(4) for m in range(-ell, ell + 1)]
    # The parity of Y_l is (-1)^l; the meta data must say so.
    assert meta["parity"].tolist() == [(-1) ** ell for ell in meta["l"]]
    assert SphericalHarmonic(vhat=_GeoStub("rhat"), lmax=3, name="Y2").vhat == "rhat"


def test_spherical_harmonic_rejects_an_unknown_vhat():
    with pytest.raises(ValueError, match="Unknown entry for vhat"):
        SphericalHarmonic(vhat=3, lmax=2, name="Y")


def test_spherical_harmonic_forwards_options_to_the_function_and_records_them():
    y = SphericalHarmonic(vhat="v", lmax=2, name="Y", norm=True, type="complex")
    assert y.sh.norm is True
    assert y.sh.type == "complex"
    assert y.sh.lmax == 2
    assert y.to_dict()["norm"] is True


def test_spherical_harmonic_build_twice_keeps_the_tables():
    y = SphericalHarmonic(vhat="v", lmax=2, name="Y")
    assert not y.sh.is_built
    y.build(tf.float64)
    alm = y.sh.alm
    y.build(tf.float64)
    assert y.sh.is_built
    assert y.sh.alm is alm


@pytest.mark.parametrize("lmax", [0, 1, 4])
def test_spherical_harmonic_output_shape_and_call_protocol(lmax):
    y = SphericalHarmonic(vhat="v", lmax=lmax, name="Y")
    y.build(tf.float64)
    data = {"v": _geo_unit_vectors(7)}
    assert y(data) is data
    assert data["Y"].shape == (7, (lmax + 1) ** 2)
    with pytest.raises(AssertionError, match="already exists"):
        y(data)


def test_spherical_harmonic_computes_in_float64_whatever_the_build_dtype():
    # Pins current behaviour (reported as a finding): build(float_dtype) ignores its argument,
    # SphericalHarmonics always sets float_dtype = tf.float64, so a float32 model gets float64 harmonics.
    y = SphericalHarmonic(vhat="v", lmax=2, name="Y")
    y.build(tf.float32)
    v32 = _geo_unit_vectors(5).astype(np.float32)
    out = y.frwrd({"v": v32})
    assert out.dtype == tf.float64
    # a float32 unit vector has norm 1 only to float32 rounding, which the polynomials inherit
    np.testing.assert_allclose(
        out.numpy(),
        _geo_real_sh(v32.astype(np.float64), 2),
        **FLOAT32_ELEMENTWISE._asdict(),
    )


def test_spherical_harmonic_does_not_modify_its_input():
    y = _geo_sh(3)
    v = _geo_unit_vectors(5)
    before = v.copy()
    y.frwrd({"v": v})
    np.testing.assert_array_equal(v, before)


def test_spherical_harmonic_unknown_type_fails_at_the_forward_pass():
    y = _geo_sh(1, type="quaternion")
    with pytest.raises(ValueError, match="Unknown type quaternion"):
        y.frwrd({"v": _geo_unit_vectors(2)})


def test_spherical_harmonic_matches_scipy_up_to_lmax_6():
    lmax = 6
    v = _geo_unit_vectors(25, seed=2)
    out = _geo_sh(lmax).frwrd({"v": v}).numpy()
    np.testing.assert_allclose(
        out, _geo_real_sh(v, lmax), **FLOAT64_ARITHMETIC._asdict()
    )


def test_spherical_harmonic_normalised_option_divides_by_sqrt_4pi():
    v = _geo_unit_vectors(9, seed=4)
    out = _geo_sh(3, norm=True).frwrd({"v": v}).numpy()
    np.testing.assert_allclose(
        out, _geo_real_sh(v, 3, norm=True), **FLOAT64_ARITHMETIC._asdict()
    )
    np.testing.assert_allclose(
        out[:, 0], 1.0 / np.sqrt(4.0 * np.pi), **FLOAT64_ARITHMETIC._asdict()
    )


def test_spherical_harmonic_complex_type_cannot_be_evaluated():
    # Pins current behaviour (reported as a finding): type="complex" is accepted by the
    # constructor and by build, but the forward pass fails inside SphericalHarmonics._compute_ylm
    # (a rank-3 transpose permutation applied to a rank-2 stack), so no complex harmonics can be
    # computed through the instruction.
    y = _geo_sh(2, type="complex")
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="transpose expects a vector of size 2"
    ):
        y.frwrd({"v": _geo_unit_vectors(4)})


def test_spherical_harmonic_hand_formulas_up_to_l_2():
    v = _geo_unit_vectors(10, seed=9)
    x, y, z = v.T
    out = _geo_sh(2).frwrd({"v": v}).numpy()
    expected = np.stack(
        [
            np.ones_like(x),
            np.sqrt(3.0) * y,
            np.sqrt(3.0) * z,
            np.sqrt(3.0) * x,
            np.sqrt(15.0) * x * y,
            np.sqrt(15.0) * y * z,
            np.sqrt(5.0) / 2.0 * (3.0 * z**2 - 1.0),
            np.sqrt(15.0) * x * z,
            np.sqrt(15.0) / 2.0 * (x**2 - y**2),
        ],
        axis=1,
    )
    np.testing.assert_allclose(out, expected, **FLOAT64_ARITHMETIC._asdict())


def test_spherical_harmonic_is_orthonormal_by_quadrature():
    # Gauss-Legendre in cos(theta) times a uniform grid in phi integrates exactly every product of two
    # harmonics up to lmax; the unnormalised harmonics have the sphere average delta_ij.
    lmax = 4
    z, wz = np.polynomial.legendre.leggauss(lmax + 2)
    n_phi = 2 * lmax + 2
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    s = np.sqrt(1.0 - zz**2)
    pts = np.stack([s * np.cos(pp), s * np.sin(pp), zz], axis=-1).reshape(-1, 3)
    weights = (wz[:, None] * np.full((1, n_phi), 1.0 / n_phi)).reshape(-1) / 2.0
    out = _geo_sh(lmax).frwrd({"v": pts}).numpy()
    gram = out.T @ (weights[:, None] * out)
    np.testing.assert_allclose(
        gram, np.eye((lmax + 1) ** 2), **FLOAT64_ARITHMETIC._asdict()
    )


def test_spherical_harmonic_parity_is_minus_one_to_the_l():
    lmax = 5
    v = _geo_unit_vectors(11, seed=12)
    y = _geo_sh(lmax)
    out = y.frwrd({"v": v}).numpy()
    inverted = y.frwrd({"v": -v}).numpy()
    sign = np.concatenate([[(-1.0) ** ell] * (2 * ell + 1) for ell in range(lmax + 1)])
    np.testing.assert_allclose(inverted, out * sign, **FLOAT64_ARITHMETIC._asdict())


def test_spherical_harmonic_addition_theorem():
    # sum_m Z_lm(a) Z_lm(b) = (2l + 1) P_l(a.b) for the unnormalised real harmonics: an invariant
    # under simultaneous rotation, written with the Legendre polynomials of scipy.
    lmax = 4
    a = _geo_unit_vectors(14, seed=13)
    b = _geo_unit_vectors(14, seed=14)
    y = _geo_sh(lmax)
    za = y.frwrd({"v": a}).numpy()
    zb = y.frwrd({"v": b}).numpy()
    cosine = (a * b).sum(axis=1)
    for ell in range(lmax + 1):
        block = slice(ell * ell, (ell + 1) ** 2)
        np.testing.assert_allclose(
            (za[:, block] * zb[:, block]).sum(axis=1),
            (2 * ell + 1) * eval_legendre(ell, cosine),
            **COVARIANCE_F64._asdict(),
        )


def test_spherical_harmonic_rotation_covariance_with_wigner_matrices():
    # out(R x) = D(R) out(x), block by block in l. D^l is solved from the scipy harmonics, not
    # from the unit, so the unit is only checked, never used to define the oracle.
    lmax = 3
    rot = _geo_rotation_matrix(seed=7)
    probe = _geo_unit_vectors(30, seed=15)
    y = _geo_sh(lmax)
    out = y.frwrd({"v": probe}).numpy()
    out_rot = y.frwrd({"v": probe @ rot.T}).numpy()
    ref = _geo_real_sh(probe, lmax)
    ref_rot = _geo_real_sh(probe @ rot.T, lmax)
    for ell in range(lmax + 1):
        block = slice(ell * ell, (ell + 1) ** 2)
        wigner = ref_rot[:, block].T @ np.linalg.pinv(ref[:, block].T)
        np.testing.assert_allclose(
            wigner @ wigner.T, np.eye(2 * ell + 1), **COVARIANCE_F64._asdict()
        )
        np.testing.assert_allclose(
            out_rot[:, block].T, wigner @ out[:, block].T, **COVARIANCE_F64._asdict()
        )


def test_spherical_harmonic_l_1_block_rotates_like_the_vector_in_the_order_y_z_x():
    # Hand case: Z_1 = sqrt(3) (y, z, x), so for l = 1 the Wigner matrix is R in the (y, z, x) basis.
    rot = _geo_rotation_matrix(seed=8)
    perm = [1, 2, 0]
    wigner_l1 = rot[np.ix_(perm, perm)]
    v = _geo_unit_vectors(6, seed=16)
    y = _geo_sh(1)
    out = y.frwrd({"v": v}).numpy()[:, 1:]
    out_rot = y.frwrd({"v": v @ rot.T}).numpy()[:, 1:]
    np.testing.assert_allclose(out_rot, out @ wigner_l1.T, **COVARIANCE_F64._asdict())


# --------------------------------------------------------------------------------------
# RadialBasis
# --------------------------------------------------------------------------------------
_GEO_RCUT = 5.5
_GEO_NFUNC = 6
_GEO_BASES = {
    "RadSinBessel": {"p": 5, "normalized": False},
    "RadSinBessel_norm": {"p": 5, "normalized": True},
    "SBessel": {"normalized": True, "p": 5},
    "Gaussian": {"p": 5},
    "Cheb": {"p": 16, "normalized": False},
}
_GEO_ENVELOPE_BASES = [key for key in _GEO_BASES if key != "SBessel"]


def _geo_basis(key: str, nfunc: int = _GEO_NFUNC, rcut: float = _GEO_RCUT, **extra):
    kwargs = dict(_GEO_BASES.get(key, {}))
    kwargs.update(extra)
    rb = RadialBasis(
        bonds="r",
        basis_type=key.removesuffix("_norm"),
        nfunc=nfunc,
        rcut=rcut,
        **kwargs,
    )
    rb.build(tf.float64)
    return rb


def _geo_r(n: int = 30, rcut: float = _GEO_RCUT) -> np.ndarray:
    return np.linspace(0.2, 0.97 * rcut, n).reshape(-1, 1)


@pytest.mark.parametrize("key", list(_GEO_BASES))
def test_radial_basis_every_type_builds_and_has_the_requested_shape(key):
    rb = _geo_basis(key, nfunc=7)
    assert rb.nfunc == 7
    r = _geo_r(11)
    data = {"r": r}
    before = r.copy()
    out = rb.frwrd(data)
    assert out.shape == (11, 7)
    assert out.dtype == tf.float64
    assert rb.get_cutoff() == _GEO_RCUT
    assert rb.rc is rb.basis_function.rc
    np.testing.assert_array_equal(r, before)
    assert set(data) == {"r"}


def test_radial_basis_arguments_and_errors():
    stub = RadialBasis(bonds=_GeoStub("d"), basis_type="SBessel", nfunc=2, rcut=4)
    assert stub.input_name == "d"
    assert (
        RadialBasis(bonds="d", basis_type="SBessel", nfunc=2, rcut=4).input_name == "d"
    )
    with pytest.raises(ValueError, match="Unknown entry for bonds"):
        RadialBasis(bonds=2, basis_type="SBessel", nfunc=2, rcut=4)
    with pytest.raises(ValueError, match="Unknown type of the radial basis, Fourier"):
        RadialBasis(bonds="d", basis_type="Fourier", nfunc=2, rcut=4)
    with pytest.raises(ValueError, match="Must be at least one basis function"):
        RadialBasis(bonds="d", basis_type="SBessel", nfunc=0, rcut=4)
    with pytest.raises(ValueError, match="Cutoff distance must be larger than 0"):
        RadialBasis(bonds="d", basis_type="SBessel", nfunc=2, rcut=0.0)
    rb = RadialBasis(bonds="d", basis_type="Cheb", nfunc=3, rcut=4, p=3, kind=2)
    assert rb.basis_type == "Cheb"
    assert rb.kwargs == {"nfunc": 3, "rcut": 4, "p": 3, "kind": 2}
    assert rb.name == "RadialBasis"


class _GeoPlainBasisFunction:
    """A radial function with no ``build`` method: a table of ``r`` powers and a fixed cutoff."""

    nfunc = 2
    rcut = 3.0

    def __init__(self) -> None:
        self.rc = tf.Variable(3.0, dtype=tf.float64, trainable=False)

    def __call__(self, r):
        return tf.concat([r, r**2], axis=1)


class _GeoRadialBasisWithPlainFunction(RadialBasis):
    """``RadialBasis`` whose function is swapped for one without a ``build`` method.

    Stands in for the collaborator, not for the unit: all four functions of the repository inherit a
    ``build`` from ``RadialBasisFunction``, so the defensive ``hasattr(..., "build")`` branch of
    ``RadialBasis.build`` can only be reached with a function that lacks it.
    """

    def __init__(self) -> None:
        super().__init__(bonds="d", basis_type="SBessel", nfunc=2, rcut=3.0)
        self.basis_function = _GeoPlainBasisFunction()


def test_radial_basis_build_accepts_a_function_without_a_build_method():
    rb = _GeoRadialBasisWithPlainFunction()
    rb.build(tf.float64)
    assert rb.rc is rb.basis_function.rc
    assert rb.get_cutoff() == 3.0
    r = np.array([[1.0], [2.0]])
    np.testing.assert_array_equal(rb.frwrd({"d": r}).numpy(), [[1.0, 1.0], [2.0, 4.0]])


def test_radial_basis_get_cutoff_before_and_after_build():
    # Before build the Variable `rc` does not exist and the plain rcut is returned.
    rb = RadialBasis(bonds="d", basis_type="Gaussian", nfunc=3, rcut=6, p=5)
    assert not hasattr(rb.basis_function, "rc")
    assert rb.get_cutoff() == 6.0
    rb.build(tf.float32)
    assert rb.get_cutoff() == 6.0
    rb32 = RadialBasis(bonds="d", basis_type="Cheb", nfunc=3, rcut=6.2, p=5)
    rb32.build(tf.float32)
    assert rb32.get_cutoff() == float(np.float32(6.2))
    assert rb32.get_cutoff() != 6.2


@pytest.mark.parametrize("key", list(_GEO_BASES))
def test_radial_basis_build_twice_keeps_the_cutoff_variable(key):
    rb = _geo_basis(key)
    rc = rb.rc
    rb.build(tf.float64)
    assert rb.rc is rc
    assert rb.basis_function.rc is rc


@pytest.mark.parametrize("key", list(_GEO_BASES))
def test_radial_basis_vanishes_beyond_the_cutoff_and_at_it(key):
    rb = _geo_basis(key)
    r = np.array([[_GEO_RCUT], [_GEO_RCUT + 1e-6], [_GEO_RCUT + 1.0], [40.0]])
    out = rb.frwrd({"r": r}).numpy()
    # strictly beyond the cutoff: exactly zero; at the cutoff: zero up to rounding of sin(n pi)
    np.testing.assert_array_equal(out[1:], 0.0)
    np.testing.assert_allclose(out[0], 0.0, **FLOAT64_ARITHMETIC._asdict())


@pytest.mark.parametrize("key", list(_GEO_BASES))
def test_radial_basis_zero_distance_is_finite(key):
    rb = _geo_basis(key)
    out = rb.frwrd({"r": np.zeros((3, 1))}).numpy()
    assert out.shape == (3, _GEO_NFUNC)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("key", _GEO_ENVELOPE_BASES)
def test_radial_basis_slope_and_curvature_vanish_at_the_cutoff(key):
    # The envelope has a zero of order 3 at rc (value, first and second derivative vanish), so
    # energies and forces are smooth where a neighbour leaves the cutoff sphere.
    rb = _geo_basis(key)
    r = tf.constant(np.full((1, 1), _GEO_RCUT))
    check_curvature = (
        key != "Cheb"
    )  # second derivative of the Chebyshev basis is NaN at rc, see below
    for k in range(_GEO_NFUNC):
        with tf.GradientTape() as outer:
            outer.watch(r)
            with tf.GradientTape() as inner:
                inner.watch(r)
                value = rb.frwrd({"r": r})[:, k]
            slope = inner.gradient(value, r)
        curvature = outer.gradient(slope, r)
        np.testing.assert_allclose(slope.numpy(), 0.0, **COVARIANCE_F64._asdict())
        if check_curvature:
            np.testing.assert_allclose(
                curvature.numpy(), 0.0, **COVARIANCE_F64._asdict()
            )


def test_radial_basis_chebyshev_second_derivative_is_nan_exactly_at_the_cutoff():
    # Pins current behaviour (reported as a finding): |1 - r / rc| ** 1 has an undefined second
    # derivative at r = rc (the power rule takes log(0)), so the Chebyshev basis returns NaN for the
    # curvature exactly at the cutoff; the first derivative is fine (see the test above) and just
    # inside the cutoff the curvature is finite.
    rb = _geo_basis("Cheb")
    for r_value, finite in ((_GEO_RCUT, False), (0.99 * _GEO_RCUT, True)):
        r = tf.constant(np.full((1, 1), r_value))
        with tf.GradientTape() as outer:
            outer.watch(r)
            with tf.GradientTape() as inner:
                inner.watch(r)
                value = rb.frwrd({"r": r})[:, 1]
            slope = inner.gradient(value, r)
        assert bool(np.all(np.isfinite(outer.gradient(slope, r).numpy()))) is finite


def test_radial_basis_sine_bessel_matches_the_closed_form():
    rb = _geo_basis("RadSinBessel")
    r = _geo_r(40)
    n = np.arange(1, _GEO_NFUNC + 1)
    expected = (
        np.sqrt(2.0 / _GEO_RCUT)
        * np.sin(n * np.pi * r / _GEO_RCUT)
        / r
        * _geo_envelope(r / _GEO_RCUT, 5)
    )
    np.testing.assert_allclose(
        rb.frwrd({"r": r}).numpy(), expected, **FLOAT64_ARITHMETIC._asdict()
    )


def test_radial_basis_sine_bessel_zero_distance_limit():
    # sin(n pi r / rc) / r tends to n pi / rc, and the envelope is 1 at r = 0.
    rb = _geo_basis("RadSinBessel", nfunc=4)
    out = rb.frwrd({"r": np.zeros((1, 1))}).numpy()[0]
    n = np.arange(1, 5)
    np.testing.assert_allclose(
        out,
        np.sqrt(2.0 / _GEO_RCUT) * n * np.pi / _GEO_RCUT,
        **FLOAT64_ARITHMETIC._asdict(),
    )


def test_radial_basis_normalised_sine_bessel_matches_closed_form_with_quadrature_constants():
    # Mean and standard deviation of sqrt(2) sin(n pi x) / x over [0, 1] by scipy.integrate.quad,
    # independent of the sici based constants of the unit; the unit works with x = r / rc.
    nfunc = 5
    rb = _geo_basis("RadSinBessel_norm", nfunc=nfunc)
    mu, sigma = [], []
    for n in range(1, nfunc + 1):
        mean = quad(lambda x, n=n: np.sqrt(2.0) * np.sin(n * np.pi * x) / x, 0, 1)[0]
        second = quad(
            lambda x, n=n: (np.sqrt(2.0) * np.sin(n * np.pi * x) / x) ** 2, 0, 1
        )[0]
        mu.append(mean)
        sigma.append(np.sqrt(second - mean**2))
    r = _geo_r(40)
    x = r / _GEO_RCUT
    n = np.arange(1, nfunc + 1)
    expected = (
        (np.sqrt(2.0) * np.sin(n * np.pi * x) / x - np.array(mu))
        / np.array(sigma)
        * _geo_envelope(x, 5)
    )
    np.testing.assert_allclose(
        rb.frwrd({"r": r}).numpy(), expected, **COVARIANCE_F64._asdict()
    )
    # the mean is sqrt(2) Si(n pi), the sine integral of scipy
    np.testing.assert_allclose(
        mu, np.sqrt(2.0) * sici(n * np.pi)[0], **COVARIANCE_F64._asdict()
    )


def _geo_sbessel_raw(r: np.ndarray, rc: float, nfunc: int) -> np.ndarray:
    """Raw functions ``(-1)^n sqrt(2) pi rc^-3/2 (n+1)(n+2)/sqrt((n+1)^2+(n+2)^2) (j0 + j0)``."""
    cols = []
    for n in range(nfunc):
        pref = (
            (-1.0) ** n
            * np.sqrt(2.0)
            * np.pi
            / rc**1.5
            * (n + 1)
            * (n + 2)
            / np.sqrt((n + 1) ** 2 + (n + 2) ** 2)
        )
        # numpy sinc(x) = sin(pi x) / (pi x)
        cols.append(pref * (np.sinc(r * (n + 1) / rc) + np.sinc(r * (n + 2) / rc)))
    return np.stack(cols, axis=1)


def _geo_r2_quadrature(rcut: float, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes on ``[0, rcut]`` and weights times ``r^2``."""
    x, w = np.polynomial.legendre.leggauss(n)
    r = (x + 1.0) / 2.0 * rcut
    return r, w * rcut / 2.0 * r**2


def test_radial_basis_simplified_bessel_is_orthonormal_with_weight_r_squared():
    rb = _geo_basis("SBessel")
    r, w = _geo_r2_quadrature(_GEO_RCUT)
    g = rb.frwrd({"r": r[:, None]}).numpy()
    np.testing.assert_allclose(
        g.T @ (w[:, None] * g), np.eye(_GEO_NFUNC), **COVARIANCE_F64._asdict()
    )


def test_radial_basis_simplified_bessel_is_the_gram_schmidt_of_the_raw_functions():
    # Orthonormalising the raw functions with the Cholesky factor of their Gram matrix (weight r^2)
    # gives the unique orthonormal set with positive leading coefficients, which is what the
    # recursion of the unit computes.
    nfunc = 5
    rb = _geo_basis("SBessel", nfunc=nfunc)
    r, w = _geo_r2_quadrature(_GEO_RCUT)
    raw = _geo_sbessel_raw(r, _GEO_RCUT, nfunc)
    chol = np.linalg.cholesky(raw.T @ (w[:, None] * raw))
    probe = _geo_r(25)[:, 0]
    expected = _geo_sbessel_raw(probe, _GEO_RCUT, nfunc) @ np.linalg.inv(chol).T
    out = rb.frwrd({"r": probe[:, None]}).numpy()
    np.testing.assert_allclose(out, expected, **COVARIANCE_F64._asdict())


def test_radial_basis_simplified_bessel_zero_distance_limit():
    # sinc(0) = 1, so the first function at r = 0 is sqrt(2) pi rc^-3/2 * 2 / sqrt(5) * (1 + 1).
    rb = _geo_basis("SBessel", nfunc=1)
    out = rb.frwrd({"r": np.zeros((1, 1))}).numpy()
    expected = np.sqrt(2.0) * np.pi / _GEO_RCUT**1.5 * 2.0 / np.sqrt(5.0) * 2.0
    np.testing.assert_allclose(out[0, 0], expected, **FLOAT64_ARITHMETIC._asdict())


@pytest.mark.parametrize(("init_gamma", "rmin"), [(1.0, 0.0), (0.5, 1.0)])
def test_radial_basis_gaussian_matches_the_closed_form(init_gamma, rmin):
    nfunc = 7
    rb = _geo_basis("Gaussian", nfunc=nfunc, init_gamma=init_gamma, rmin=rmin)
    r = _geo_r(30)
    centres = np.linspace(rmin, _GEO_RCUT, nfunc).reshape(1, -1)
    width = init_gamma * (centres[0, 1] - centres[0, 0])
    expected = np.exp(-0.5 * (r - centres) ** 2 / width**2) * _geo_envelope(
        r / _GEO_RCUT, 5
    )
    np.testing.assert_allclose(
        rb.frwrd({"r": r}).numpy(), expected, **FLOAT64_ARITHMETIC._asdict()
    )


def test_radial_basis_gaussian_trainable_flag_controls_the_variables():
    assert _geo_basis("Gaussian", nfunc=4).trainable_variables == ()
    trainable = _geo_basis("Gaussian", nfunc=4, trainable=True)
    assert sorted(v.shape.rank for v in trainable.trainable_variables) == [0, 2]


def test_radial_basis_gaussian_ground_state_is_the_envelope_at_zero():
    # The first Gaussian is centred at rmin = 0, so at r = 0 its value is the envelope, 1.
    rb = _geo_basis("Gaussian", nfunc=3)
    out = rb.frwrd({"r": np.zeros((1, 1))}).numpy()
    np.testing.assert_allclose(out[0, 0], 1.0, **FLOAT64_ARITHMETIC._asdict())


@pytest.mark.parametrize(
    ("kind", "reversed_", "normalized"),
    [(1, False, False), (1, True, False), (2, False, False), (1, False, True)],
)
def test_radial_basis_chebyshev_matches_numpy_and_scipy(kind, reversed_, normalized):
    nfunc = 6
    rb = _geo_basis(
        "Cheb", nfunc=nfunc, p=5, kind=kind, reversed=reversed_, normalized=normalized
    )
    r = _geo_r(35)
    x = 2.0 * (1.0 - np.abs(1.0 - r / _GEO_RCUT)) - 1.0
    if reversed_:
        x = -x
    degrees = np.arange(1, nfunc + 1)
    if kind == 1:
        poly = np.polynomial.chebyshev.chebvander(x[:, 0], nfunc)[:, 1:]
        np.testing.assert_allclose(
            poly, eval_chebyt(degrees, x), **FLOAT64_ARITHMETIC._asdict()
        )
    else:
        poly = eval_chebyu(degrees, x)
    expected = poly * _geo_envelope(r / _GEO_RCUT, 5)
    if normalized:
        expected = expected * np.sqrt(1.0 / np.pi)
    np.testing.assert_allclose(
        rb.frwrd({"r": r}).numpy(), expected, **FLOAT64_ARITHMETIC._asdict()
    )


def test_radial_basis_chebyshev_rejects_an_unknown_kind_at_the_forward_pass():
    rb = _geo_basis("Cheb", nfunc=3, kind=3)
    with pytest.raises(ValueError, match="kind must be 1 or 2"):
        rb.frwrd({"r": _geo_r(4)})


def test_radial_basis_chebyshev_at_zero_distance_is_alternating_sign_times_envelope_one():
    # At r -> 0 the Chebyshev argument is -1 where T_n(-1) = (-1)^n, and the envelope is 1.
    rb = _geo_basis("Cheb", nfunc=4, p=5)
    out = rb.frwrd({"r": np.zeros((1, 1))}).numpy()[0]
    np.testing.assert_allclose(
        out, [(-1.0) ** n for n in range(1, 5)], **COVARIANCE_F64._asdict()
    )


def test_radial_basis_float32_build_rounds_the_cutoff_and_returns_float64():
    # Pins current behaviour (reported as a finding): float64 constants inside the basis functions
    # promote a float32 distance, so the output is float64 for a float32 build; the cutoff Variable
    # is float32 and holds the rounded value of rcut.
    rb = RadialBasis(bonds="d", basis_type="Cheb", nfunc=4, rcut=6.2, p=5)
    rb.build(tf.float32)
    assert rb.rc.dtype == tf.float32
    r = _geo_r(9, 6.2).astype(np.float32)
    out = rb.frwrd({"d": r})
    assert out.dtype == tf.float64
    rc32 = float(np.float32(6.2))
    r64 = r.astype(np.float64)
    x = 2.0 * (1.0 - np.abs(1.0 - r64 / rc32)) - 1.0
    expected = np.polynomial.chebyshev.chebvander(x[:, 0], 4)[:, 1:] * _geo_envelope(
        r64 / rc32, 5
    )
    np.testing.assert_allclose(out.numpy(), expected, **FLOAT64_ARITHMETIC._asdict())


# --------------------------------------------------------------------------------------
# MLPRadialFunction
# --------------------------------------------------------------------------------------
def _geo_mlp(
    *,
    hidden=(16, 12),
    activation="tanh",
    norm=False,
    n_rad_max=3,
    lmax=2,
    nfunc=5,
    emb_i=None,
    emb_j=None,
    lora_config=None,
    seeded=True,
    dtype=tf.float64,
):
    r = MLPRadialFunction(
        n_rad_max=n_rad_max,
        lmax=lmax,
        basis="basis",
        input_shape=nfunc,
        hidden_layers=list(hidden),
        activation=activation,
        norm=norm,
        name="R",
        chemical_embedding_i=emb_i,
        chemical_embedding_j=emb_j,
        lora_config=lora_config,
    )
    r.build(dtype)
    if seeded:
        seed_trainable_variables(r)
    return r


def _geo_mlp_forward(
    r, data, act_name, *, emb=(), per_atom=(False, False), lora=False, mode="lora"
):
    """Numpy forward pass of ``MLPRadialFunction`` from the current variable values.

    ``emb`` lists the data keys of the embedding tables (i first, then j) that are concatenated.
    """
    x = data["basis"]
    index_keys = {
        0: (constants.BOND_MU_I, constants.BOND_IND_I),
        1: (constants.BOND_MU_J, constants.BOND_IND_J),
    }
    for slot, key in emb:
        index = data[index_keys[slot][int(per_atom[slot])]]
        x = np.concatenate([x, data[key][index]], axis=1)
    layers = [getattr(r.mlp, f"layer{i}") for i in range(r.mlp.nlayers)]
    for i, layer in enumerate(layers):
        w = _geo_effective_weight(layer, lora, mode)
        x = x @ (w / np.sqrt(w.shape[0]))
        if i < len(layers) - 1:
            x = _GEO_ACTIVATIONS[act_name](x)
    if r.norm:
        x = x / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * r.gamma.numpy()
    return _geo_l_gather(x, r.n_rad_max, r.lmax)


def test_mlp_radial_function_input_shape_from_the_basis_argument():
    basis = RadialBasis(bonds="d", basis_type="Cheb", nfunc=9, rcut=5.0, p=5)
    r = MLPRadialFunction(n_rad_max=2, lmax=1, basis=basis, input_shape=123)
    assert r.basis_name == "RadialBasis"
    assert r.input_shape == 9  # the basis wins over input_shape
    stub = MLPRadialFunction(n_rad_max=2, lmax=1, basis=_GeoStub("g", nfunc=4))
    assert (stub.basis_name, stub.input_shape) == ("g", 4)
    named = MLPRadialFunction(n_rad_max=2, lmax=1, basis="g", input_shape=6)
    assert (named.basis_name, named.input_shape) == ("g", 6)


def test_mlp_radial_function_without_basis_cannot_run():
    # Pins current behaviour (reported as a finding): basis=None is accepted with an input_shape,
    # but basis_name is never set, so the instruction can be built and only fails in frwrd.
    bare = MLPRadialFunction(n_rad_max=2, lmax=1, input_shape=3)
    assert bare.input_shape == 3
    assert not hasattr(bare, "basis_name")
    bare.build(tf.float64)
    with pytest.raises(AttributeError, match="basis_name"):
        bare.frwrd({"basis": np.ones((2, 3))})


def test_mlp_radial_function_argument_errors():
    with pytest.raises(AssertionError, match="Need to provide shape"):
        MLPRadialFunction(n_rad_max=2, lmax=1, basis="g")
    with pytest.raises(AssertionError, match="Need to provide shape"):
        MLPRadialFunction(n_rad_max=2, lmax=1)
    with pytest.raises(ValueError, match="Unknown basis"):
        MLPRadialFunction(n_rad_max=2, lmax=1, basis=4.5)
    with pytest.raises(
        ValueError, match="MLP activation must be predefined str or None"
    ):
        MLPRadialFunction(n_rad_max=2, lmax=1, input_shape=3, activation=tf.nn.tanh)
    with pytest.raises(ValueError, match="activation must be a string"):
        MLPRadialFunction(n_rad_max=2, lmax=1, input_shape=3, activation="relu")


def test_mlp_radial_function_defaults_and_derived_attributes():
    r = MLPRadialFunction(n_rad_max=4, lmax=2, input_shape=7)
    assert r.hidden_layers == [64, 64, 64]
    assert r.n_out == 4 * 3
    assert r.norm is False
    assert r.mlp.layers_config == [7, 64, 64, 64, 12]
    assert r.l_tile.numpy().tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert r.l_tile.dtype == tf.int32
    assert not r.chem_i_is_per_atom
    assert not r.chem_j_is_per_atom
    z = _geo_chem(n_elem=3, emb=2)
    r2 = MLPRadialFunction(
        n_rad_max=4,
        lmax=0,
        input_shape=7,
        chemical_embedding_i=z,
        chemical_embedding_j=z,
        hidden_layers=[8],
    )
    assert r2.mlp.layers_config == [7 + 2 + 2, 8, 4]


def test_mlp_radial_function_build_creates_the_weights_once():
    r = MLPRadialFunction(
        n_rad_max=3, lmax=1, input_shape=5, hidden_layers=[8], norm=True
    )
    r.build(tf.float64)
    assert sorted(tuple(v.shape) for v in r.trainable_variables) == [
        (1, 6),
        (5, 8),
        (8, 6),
    ]
    assert r.is_built
    np.testing.assert_allclose(r.epsilon.numpy(), 1e-5, **FLOAT64_ARITHMETIC._asdict())
    first = [v.ref() for v in r.trainable_variables]
    r.build(tf.float64)
    assert [v.ref() for v in r.trainable_variables] == first
    no_norm = MLPRadialFunction(n_rad_max=3, lmax=1, input_shape=5, hidden_layers=[8])
    no_norm.build(tf.float32)
    assert not hasattr(no_norm, "gamma")
    assert all(v.dtype == tf.float32 for v in no_norm.trainable_variables)


@pytest.mark.parametrize("activation", ["tanh", "silu", "sigmoid", None])
def test_mlp_radial_function_forward_matches_the_numpy_recomputation(activation):
    r = _geo_mlp(activation=activation)
    data = _geo_bond_data(9, 5)
    out = r.frwrd(data).numpy()
    assert out.shape == (9, 3, 9)
    np.testing.assert_allclose(
        out, _geo_mlp_forward(r, data, activation), **COVARIANCE_F64._asdict()
    )


def test_mlp_radial_function_output_is_constant_along_m_and_follows_the_l_blocks():
    lmax = 3
    r = _geo_mlp(lmax=lmax, n_rad_max=2)
    out = r.frwrd(_geo_bond_data(6, 5)).numpy()
    for ell in range(lmax + 1):
        block = out[:, :, ell * ell : (ell + 1) ** 2]
        np.testing.assert_array_equal(
            block, np.repeat(block[:, :, :1], 2 * ell + 1, axis=2)
        )
    # different l carry different radial functions (they are different columns of the network output)
    assert not np.allclose(out[:, :, 0], out[:, :, 1])


def test_mlp_radial_function_normalisation_rescales_each_bond():
    r = _geo_mlp(norm=True)
    data = _geo_bond_data(8, 5)
    np.testing.assert_allclose(
        r.frwrd(data).numpy(),
        _geo_mlp_forward(r, data, "tanh"),
        **COVARIANCE_F64._asdict(),
    )
    r.gamma.assign(tf.zeros_like(r.gamma))
    np.testing.assert_array_equal(r.frwrd(data).numpy(), 0.0)


def test_mlp_radial_function_chemical_embeddings_are_concatenated_in_order_i_then_j():
    z_i = _geo_chem(n_elem=4, emb=2, name="Zi")
    z_j = _geo_chem(n_elem=4, emb=3, name="Zj")
    r = _geo_mlp(nfunc=5, emb_i=z_i, emb_j=z_j)
    assert r.mlp.layers_config[0] == 5 + 2 + 3
    data = _geo_bond_data(10, 5)
    data["Zi"] = z_i.w.numpy()
    data["Zj"] = z_j.w.numpy()
    out = r.frwrd(data).numpy()
    np.testing.assert_allclose(
        out,
        _geo_mlp_forward(r, data, "tanh", emb=[(0, "Zi"), (1, "Zj")]),
        **COVARIANCE_F64._asdict(),
    )
    # the element indices matter: exchanging mu_i and mu_j changes the result
    swapped = dict(data)
    swapped[constants.BOND_MU_I] = data[constants.BOND_MU_J]
    swapped[constants.BOND_MU_J] = data[constants.BOND_MU_I]
    assert not np.allclose(r.frwrd(swapped).numpy(), out)


@pytest.mark.parametrize("which", ["i", "j"])
def test_mlp_radial_function_single_embedding(which):
    z = _geo_chem(n_elem=4, emb=2, name="Zx")
    r = _geo_mlp(nfunc=5, **{f"emb_{which}": z})
    data = _geo_bond_data(7, 5)
    data["Zx"] = z.w.numpy()
    slot = 0 if which == "i" else 1
    np.testing.assert_allclose(
        r.frwrd(data).numpy(),
        _geo_mlp_forward(r, data, "tanh", emb=[(slot, "Zx")]),
        **COVARIANCE_F64._asdict(),
    )


def test_mlp_radial_function_per_atom_embeddings_are_gathered_with_the_atom_index():
    # Branch reached with a stub: no embedding class of the repository sets is_per_atom.
    z_i = _GeoPerAtomEmbedding("Zi", 2)
    z_j = _GeoPerAtomEmbedding("Zj", 3)
    r = _geo_mlp(nfunc=5, emb_i=z_i, emb_j=z_j)
    assert r.chem_i_is_per_atom
    assert r.chem_j_is_per_atom
    rng = np.random.default_rng(3)
    data = _geo_bond_data(9, 5)
    # six atoms: the per-atom tables are indexed by ind_i / ind_j, not by the element indices
    data["Zi"] = rng.normal(size=(6, 2))
    data["Zj"] = rng.normal(size=(6, 3))
    out = r.frwrd(data).numpy()
    np.testing.assert_allclose(
        out,
        _geo_mlp_forward(
            r, data, "tanh", emb=[(0, "Zi"), (1, "Zj")], per_atom=(True, True)
        ),
        **COVARIANCE_F64._asdict(),
    )


@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_mlp_radial_function_dtype_follows_the_build_dtype(tf_dtype, np_dtype):
    r = _geo_mlp(dtype=tf_dtype)
    data = _geo_bond_data(6, 5)
    data["basis"] = data["basis"].astype(np_dtype)
    out = r.frwrd(data)
    assert out.dtype == tf_dtype
    tol = COVARIANCE_F64 if np_dtype is np.float64 else FLOAT32_NETWORK
    ref = {**data, "basis": data["basis"].astype(np.float64)}
    np.testing.assert_allclose(
        out.numpy(), _geo_mlp_forward(r, ref, "tanh"), **tol._asdict()
    )


def test_mlp_radial_function_does_not_modify_its_input():
    r = _geo_mlp()
    data = _geo_bond_data(5, 5)
    snapshot = {k: v.copy() for k, v in data.items()}
    r.frwrd(data)
    assert set(data) == set(snapshot)
    for key, value in snapshot.items():
        np.testing.assert_array_equal(data[key], value)


def test_mlp_radial_function_lora_built_from_the_configuration():
    r = _geo_mlp(lora_config=_GEO_LORA, seeded=False)
    assert r.lora
    assert r.lora_config == _GEO_LORA
    layers = [getattr(r.mlp, f"layer{i}") for i in range(r.mlp.nlayers)]
    assert all(layer.lora and len(layer.lora_tensors) == 2 for layer in layers)
    # base weights are frozen, only the LoRA tensors train
    assert len(r.trainable_variables) == 2 * len(layers)
    assert all("LORA" in v.name for v in r.trainable_variables)


@pytest.mark.parametrize("mode", ["lora", "additive"])
def test_mlp_radial_function_lora_life_cycle_matches_w_plus_delta(mode):
    config = dict(_GEO_LORA, mode=mode)
    r = _geo_mlp()
    data = _geo_bond_data(8, 5)
    plain = r.frwrd(data).numpy()
    assert r._init_args["lora_config"] is None
    r.enable_lora_adaptation(config)
    assert r.lora
    assert r._init_args["lora_config"] == config
    # B (or the additive delta) starts at zero: no change of the output
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), plain, **FLOAT64_ARITHMETIC._asdict()
    )
    seed_trainable_variables(r, seed=5)  # only the LoRA tensors are trainable now
    lora_out = r.frwrd(data).numpy()
    assert not np.allclose(lora_out, plain)
    np.testing.assert_allclose(
        lora_out,
        _geo_mlp_forward(r, data, "tanh", lora=True, mode=mode),
        **COVARIANCE_F64._asdict(),
    )
    r.finalize_lora_update()
    assert not r.lora
    assert r.lora_config is None
    assert "lora_config" not in r._init_args
    layers = [getattr(r.mlp, f"layer{i}") for i in range(r.mlp.nlayers)]
    assert all(
        not layer.lora and not hasattr(layer, "lora_tensors") for layer in layers
    )
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), lora_out, **COVARIANCE_F64._asdict()
    )
    assert len(r.trainable_variables) == len(layers)  # the base weights train again


# --------------------------------------------------------------------------------------
# MLPRadialFunction_v2
# --------------------------------------------------------------------------------------
def _geo_v2(
    *,
    hidden=(16, 12),
    activation=None,
    n_rad_max=3,
    lmax=2,
    nfunc=5,
    chem=None,
    embed_i=False,
    embed_j=True,
    lora_config=None,
    seeded=True,
    dtype=tf.float64,
    **kwargs,
):
    r = MLPRadialFunction_v2(
        n_rad_max=n_rad_max,
        lmax=lmax,
        name="R2",
        basis="basis",
        input_shape=nfunc,
        hidden_layers=list(hidden),
        activation=activation,
        chem_embedding=chem,
        embed_i=embed_i,
        embed_j=embed_j,
        lora_config=lora_config,
        **kwargs,
    )
    r.build(dtype)
    if seeded:
        seed_trainable_variables(r)
    return r


def _geo_v2_forward(r, data, *, chem_key=None, lora=False, mode="lora"):
    """Numpy forward pass of ``MLPRadialFunction_v2`` from the current variable values."""

    def layer_matmul(x, layer):
        w = _geo_effective_weight(layer, lora, mode)
        return x @ (w * (1.0 / np.sqrt(w.shape[0]) if layer.normalize else 1.0))

    x = data["basis"]
    for act_name, layer in zip(r.activation, r.layers[:-1]):
        x = _GEO_ACTIVATIONS[act_name](layer_matmul(x, layer))
    y = layer_matmul(x, r.layers[-1])
    if chem_key is not None:
        z = layer_matmul(data[chem_key], r.embed_transform)
        emb = np.ones_like(y)
        if r.embed_j:
            emb = emb * z[data[constants.BOND_MU_J]]
        if r.embed_i:
            emb = np.tanh(emb * z[data[constants.BOND_MU_I]])
        y = y * emb
    return _geo_l_gather(y, r.n_rad_max, r.lmax)


def test_mlp_radial_function_v2_basis_argument_and_errors():
    basis = RadialBasis(bonds="d", basis_type="SBessel", nfunc=9, rcut=5.0)
    r = MLPRadialFunction_v2(n_rad_max=2, lmax=1, basis=basis, input_shape=123)
    assert (r.basis_name, r.input_shape) == ("RadialBasis", 9)
    stub = MLPRadialFunction_v2(n_rad_max=2, lmax=1, basis=_GeoStub("g", nfunc=4))
    assert (stub.basis_name, stub.input_shape) == ("g", 4)
    named = MLPRadialFunction_v2(n_rad_max=2, lmax=1, basis="g", input_shape=6)
    assert (named.basis_name, named.input_shape) == ("g", 6)
    assert MLPRadialFunction_v2(n_rad_max=2, lmax=1, input_shape=3).input_shape == 3
    with pytest.raises(AssertionError, match="Need to provide shape"):
        MLPRadialFunction_v2(n_rad_max=2, lmax=1, basis="g")
    with pytest.raises(AssertionError, match="Need to provide shape"):
        MLPRadialFunction_v2(n_rad_max=2, lmax=1)
    with pytest.raises(ValueError, match="Unknown basis"):
        MLPRadialFunction_v2(n_rad_max=2, lmax=1, basis=1.5)


def test_mlp_radial_function_v2_hidden_layers_and_activation_normalisation():
    r = MLPRadialFunction_v2(n_rad_max=2, lmax=1, input_shape=3)
    assert r.hidden_layers == [64, 64]
    assert r.activation == ["silu", "silu"]
    assert [(layer.n_in, layer.n_out) for layer in r.layers] == [
        (3, 64),
        (64, 64),
        (64, 4),
    ]
    assert [layer.name for layer in r.layers] == [
        "MLPRadialFunction_Linear_0",
        "MLPRadialFunction_Linear_1",
        "MLPRadialFunction_Linear_2",
    ]
    one = MLPRadialFunction_v2(
        n_rad_max=2, lmax=1, input_shape=3, hidden_layers=[5, 6, 7], activation="tanh"
    )
    assert one.activation == ["tanh", "tanh", "tanh"]
    listed = MLPRadialFunction_v2(
        n_rad_max=2,
        lmax=1,
        input_shape=3,
        hidden_layers=[5, 6],
        activation=["tanh", "sigmoid"],
    )
    assert listed.activation == ["tanh", "sigmoid"]
    with pytest.raises(AssertionError):
        MLPRadialFunction_v2(
            n_rad_max=2,
            lmax=1,
            input_shape=3,
            hidden_layers=[5, 6],
            activation=["tanh"],
        )
    assert r.n_out == 4
    assert r.l_tile.numpy().tolist() == [0, 1, 1, 1]
    assert r.chem_embedding is None
    assert not hasattr(r, "embed_transform")


def test_mlp_radial_function_v2_build_creates_the_layers_once():
    z = _geo_chem(n_elem=4, emb=3)
    r = MLPRadialFunction_v2(
        n_rad_max=2, lmax=1, input_shape=5, hidden_layers=[8], chem_embedding=z
    )
    r.build(tf.float32)
    assert r.is_built
    assert [tuple(layer.w.shape) for layer in r.layers] == [(5, 8), (8, 4)]
    assert tuple(r.embed_transform.w.shape) == (3, 4)
    assert r.layers[0].w.dtype == tf.float32
    first = r.layers[0].w
    r.build(tf.float32)
    assert r.layers[0].w is first


@pytest.mark.parametrize("act", [None, "tanh", ["silu", "sigmoid"]])
def test_mlp_radial_function_v2_forward_matches_the_numpy_recomputation(act):
    r = _geo_v2(activation=act)
    data = _geo_bond_data(9, 5)
    out = r.frwrd(data).numpy()
    assert out.shape == (9, 3, 9)
    np.testing.assert_allclose(
        out, _geo_v2_forward(r, data), **COVARIANCE_F64._asdict()
    )


def test_mlp_radial_function_v2_without_hidden_layer_is_one_linear_map():
    r = _geo_v2(hidden=[], n_rad_max=2, lmax=1)
    assert len(r.layers) == 1
    data = _geo_bond_data(6, 5)
    expected = _geo_l_gather(data["basis"] @ (r.layers[0].w.numpy() / np.sqrt(5)), 2, 1)
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), expected, **COVARIANCE_F64._asdict()
    )


def test_mlp_radial_function_v2_unknown_activation_fails_only_in_the_forward_pass():
    # Pins current behaviour (reported as a finding): names are not validated at construction, the
    # lookup in ACTIVATION_DICT raises a bare KeyError at the first forward pass.
    r = _geo_v2(activation="relu")
    with pytest.raises(KeyError, match="relu"):
        r.frwrd(_geo_bond_data(3, 5))


@pytest.mark.parametrize("normalize", [True, False])
def test_mlp_radial_function_v2_normalize_option_moves_the_scale_from_the_forward_to_the_weights(
    normalize,
):
    r = _geo_v2(hidden=[200], normalize=normalize, seeded=False)
    first = r.layers[0]
    expected_std = 1.0 if normalize else 1.0 / np.sqrt(5)
    n_weights = first.w.shape[0] * first.w.shape[1]
    np.testing.assert_allclose(
        first.w.numpy().std(), expected_std, rtol=sample_std_rtol(n_weights), atol=0.0
    )
    seed_trainable_variables(r)
    data = _geo_bond_data(4, 5)
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), _geo_v2_forward(r, data), **COVARIANCE_F64._asdict()
    )


def test_mlp_radial_function_v2_init_type_options_reach_the_layers():
    zeros = _geo_v2(init_type="zeros", seeded=False)
    assert all(np.all(layer.w.numpy() == 0.0) for layer in zeros.layers)
    np.testing.assert_array_equal(zeros.frwrd(_geo_bond_data(3, 5)).numpy(), 0.0)
    uniform = _geo_v2(init_type="uniform", hidden=[40], seeded=False)
    w = uniform.layers[0].w.numpy()
    assert np.abs(w).max() <= 1.0  # normalize=True: U(-1, 1)
    assert np.abs(w).max() > 0.9
    with pytest.raises(AssertionError):
        MLPRadialFunction_v2(n_rad_max=2, lmax=1, input_shape=3, init_type="orthogonal")


@pytest.mark.parametrize(
    ("embed_i", "embed_j"), [(False, True), (True, False), (True, True), (False, False)]
)
def test_mlp_radial_function_v2_chemical_embedding_modulates_the_output(
    embed_i, embed_j
):
    z = _geo_chem(n_elem=4, emb=3)
    r = _geo_v2(chem=z, embed_i=embed_i, embed_j=embed_j)
    data = _geo_bond_data(11, 5)
    data[z.name] = z.w.numpy()
    out = r.frwrd(data).numpy()
    np.testing.assert_allclose(
        out, _geo_v2_forward(r, data, chem_key=z.name), **COVARIANCE_F64._asdict()
    )
    # the same layers without a chemical embedding give the unmodulated output
    plain = _geo_v2(chem=None)
    for src, dst in zip(r.layers, plain.layers):
        dst.w.assign(src.w)
    unmodulated = plain.frwrd(data).numpy()
    if embed_i or embed_j:
        assert not np.allclose(out, unmodulated)
    else:
        np.testing.assert_allclose(out, unmodulated, **FLOAT64_ARITHMETIC._asdict())


def test_mlp_radial_function_v2_embedding_i_is_squashed_by_tanh():
    # With embed_i the modulation is tanh(...), so |output| cannot exceed |unmodulated output|,
    # however large the embedding transform is.
    z = _geo_chem(n_elem=3, emb=3)
    r = _geo_v2(chem=z, embed_i=True, embed_j=False)
    r.embed_transform.w.assign(100.0 * r.embed_transform.w)
    data = _geo_bond_data(10, 5, n_elem=3)
    data[z.name] = z.w.numpy()
    out = r.frwrd(data).numpy()
    plain = _geo_v2(chem=None)
    for src, dst in zip(r.layers, plain.layers):
        dst.w.assign(src.w)
    bound = np.abs(plain.frwrd(data).numpy())
    assert np.all(np.abs(out) <= bound + 1e-12 * bound.max())
    assert np.any(np.abs(out) > 0.5 * bound)  # saturated, not vanishing


def test_mlp_radial_function_v2_embedding_is_not_available_in_local_mode():
    z = _geo_chem(n_elem=4, emb=3)
    r = _geo_v2(chem=z)
    data = _geo_bond_data(4, 5)
    data[z.name] = z.w.numpy()
    with pytest.raises(NotImplementedError):
        r.frwrd(data, local=True)
    assert _geo_v2().frwrd(data, local=True).shape == (4, 3, 9)


@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_mlp_radial_function_v2_dtype_follows_the_build_dtype(tf_dtype, np_dtype):
    z = _geo_chem(n_elem=4, emb=3)
    r = _geo_v2(dtype=tf_dtype, chem=z)
    data = _geo_bond_data(6, 5)
    data["basis"] = data["basis"].astype(np_dtype)
    data[z.name] = z.w.numpy().astype(np_dtype)
    out = r.frwrd(data)
    assert out.dtype == tf_dtype
    tol = COVARIANCE_F64 if np_dtype is np.float64 else FLOAT32_NETWORK
    ref = {
        k: v.astype(np.float64) if v.dtype == np_dtype else v for k, v in data.items()
    }
    np.testing.assert_allclose(
        out.numpy(), _geo_v2_forward(r, ref, chem_key=z.name), **tol._asdict()
    )


def test_mlp_radial_function_v2_does_not_modify_its_input():
    z = _geo_chem(n_elem=4, emb=3)
    r = _geo_v2(chem=z)
    data = _geo_bond_data(5, 5)
    data[z.name] = z.w.numpy()
    snapshot = {k: v.copy() for k, v in data.items()}
    r.frwrd(data)
    assert set(data) == set(snapshot)
    for key, value in snapshot.items():
        np.testing.assert_array_equal(data[key], value)


def test_mlp_radial_function_v2_lora_built_from_the_configuration():
    z = _geo_chem(n_elem=4, emb=3)
    r = _geo_v2(chem=z, lora_config=_GEO_LORA, seeded=False)
    assert r.lora
    assert all(layer.lora and len(layer.lora_tensors) == 2 for layer in r.layers)
    assert r.embed_transform.lora
    lora_tensors = [v for v in r.trainable_variables if "LORA" in v.name]
    # the chemical embedding itself stays trainable; every Linear carries two LoRA tensors
    assert len(lora_tensors) == 2 * (len(r.layers) + 1)


@pytest.mark.parametrize("mode", ["lora", "additive"])
@pytest.mark.parametrize("with_chem", [False, True])
def test_mlp_radial_function_v2_lora_life_cycle_matches_w_plus_delta(mode, with_chem):
    config = dict(_GEO_LORA, mode=mode)
    z = _geo_chem(n_elem=4, emb=3) if with_chem else None
    r = _geo_v2(chem=z)
    data = _geo_bond_data(8, 5)
    if z is not None:
        data[z.name] = z.w.numpy()
    plain = r.frwrd(data).numpy()
    r.enable_lora_adaptation(config)
    assert r.lora
    assert r._init_args["lora_config"] == config
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), plain, **FLOAT64_ARITHMETIC._asdict()
    )
    seed_trainable_variables(
        r, seed=5
    )  # the LoRA tensors (and the chemical embedding, if any)
    if z is not None:
        data[z.name] = z.w.numpy()
    lora_out = r.frwrd(data).numpy()
    assert not np.allclose(lora_out, plain)
    chem_key = z.name if z is not None else None
    np.testing.assert_allclose(
        lora_out,
        _geo_v2_forward(r, data, chem_key=chem_key, lora=True, mode=mode),
        **COVARIANCE_F64._asdict(),
    )
    r.finalize_lora_update()
    assert not r.lora
    assert "lora_config" not in r._init_args
    assert all(
        not layer.lora and not hasattr(layer, "lora_tensors") for layer in r.layers
    )
    if with_chem:
        assert not r.embed_transform.lora
    np.testing.assert_allclose(
        r.frwrd(data).numpy(), lora_out, **COVARIANCE_F64._asdict()
    )


# --------------------------------------------------------------------------------------
# ScalarChemicalEmbedding
# --------------------------------------------------------------------------------------
_GEO_ELEMENTS = {"H": 0, "C": 1, "N": 2, "O": 3}


def _geo_z(init: str = "random", size: int = 5, dtype=tf.float64, **kwargs):
    z = ScalarChemicalEmbedding(
        name="Z", element_map=_GEO_ELEMENTS, embedding_size=size, init=init, **kwargs
    )
    z.build(dtype)
    return z


def test_scalar_chemical_embedding_stores_the_element_map():
    z = ScalarChemicalEmbedding(name="Z", element_map=_GEO_ELEMENTS, embedding_size=5)
    assert z.number_of_elements == 4
    assert z.embedding_size == 5
    assert z.is_trainable
    assert z.init == "random"
    assert not z.lora
    assert z.element_map_symbols.numpy().tolist() == [b"H", b"C", b"N", b"O"]
    assert z.element_map_index.numpy().tolist() == [0, 1, 2, 3]
    element_map = z.get_element_map()
    assert element_map is not None
    symbols, index = element_map
    assert symbols.tolist() == ["H", "C", "N", "O"]
    assert symbols.dtype.kind == "U"
    assert index.tolist() == [0, 1, 2, 3]
    assert not z.is_built
    assert not hasattr(z, "w")


def test_scalar_chemical_embedding_element_map_keeps_the_index_values():
    z = ScalarChemicalEmbedding(
        name="Z", element_map={"Fe": 2, "Ni": 0, "Cr": 1}, embedding_size=2
    )
    element_map = z.get_element_map()
    assert element_map is not None
    symbols, index = element_map
    assert symbols.tolist() == ["Fe", "Ni", "Cr"]
    assert index.tolist() == [2, 0, 1]
    assert z.number_of_elements == 3


@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_scalar_chemical_embedding_random_init_is_standard_normal(tf_dtype, np_dtype):
    tf.random.set_seed(2024)
    n, size = 200, 50
    z = ScalarChemicalEmbedding(
        name="Z", element_map={f"E{i}": i for i in range(n)}, embedding_size=size
    )
    z.build(tf_dtype)
    w = z.w.numpy()
    assert z.w.dtype == tf_dtype
    assert w.shape == (n, size)
    assert w.dtype == np_dtype
    np.testing.assert_allclose(w.std(), 1.0, rtol=sample_std_rtol(n * size), atol=0.0)
    assert abs(w.mean()) < _GEO_N_SIGMA / np.sqrt(n * size)
    assert z.w.ref() in [v.ref() for v in z.trainable_variables]


@pytest.mark.parametrize("init", ["zero", "zeros"])
@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_scalar_chemical_embedding_zero_init(init, tf_dtype, np_dtype):
    z = _geo_z(init, size=3, dtype=tf_dtype)
    assert z.w.dtype == tf_dtype
    np.testing.assert_array_equal(z.w.numpy(), np.zeros((4, 3), dtype=np_dtype))


@pytest.mark.parametrize(("size", "diagonal"), [(3, 3), (4, 4), (6, 4)])
@pytest.mark.parametrize(("tf_dtype", "np_dtype"), _GEO_DTYPES)
def test_scalar_chemical_embedding_delta_init_is_a_truncated_identity(
    size, diagonal, tf_dtype, np_dtype
):
    z = _geo_z("delta", size=size, dtype=tf_dtype)
    expected = np.zeros((4, size), dtype=np_dtype)
    expected[np.arange(diagonal), np.arange(diagonal)] = 1.0
    assert z.w.dtype == tf_dtype
    np.testing.assert_array_equal(z.w.numpy(), expected)


def test_scalar_chemical_embedding_unknown_init_is_an_error_and_leaves_it_unbuilt():
    z = ScalarChemicalEmbedding(
        name="Z", element_map=_GEO_ELEMENTS, embedding_size=2, init="xavier"
    )
    with pytest.raises(
        NotImplementedError, match="Unknown initialization method.*xavier"
    ):
        z.build(tf.float64)
    assert not z.is_built


def test_scalar_chemical_embedding_is_trainable_flag_and_build_twice():
    z = _geo_z("random", is_trainable=False)
    assert z.w.trainable is False
    assert z.trainable_variables == ()
    first = z.w
    z.build(tf.float64)
    assert z.w is first
    trainable = _geo_z("zeros")
    assert [v.ref() for v in trainable.trainable_variables] == [trainable.w.ref()]


def test_scalar_chemical_embedding_forward_returns_the_table_and_call_stores_it():
    z = _geo_z("delta", size=4)
    seed_trainable_variables(z)
    np.testing.assert_array_equal(z.frwrd({}).numpy(), z.w.numpy())
    data = {}
    assert z(data) is data
    np.testing.assert_array_equal(data["Z"].numpy(), z.w.numpy())
    # the returned object is the variable itself, so a later update is visible without a new call
    z.w.assign(tf.zeros_like(z.w))
    np.testing.assert_array_equal(data["Z"].numpy(), 0.0)


@pytest.mark.parametrize("mode", ["lora", "additive"])
def test_scalar_chemical_embedding_lora_forward_adds_the_low_rank_update(mode):
    config = dict(_GEO_LORA, mode=mode)
    z = _geo_z("random", size=6)
    seed_trainable_variables(z)
    base = z.w.numpy().copy()
    z.enable_lora_adaptation(config)
    assert z.lora
    assert z._init_args["lora_config"] == config
    assert z.w.trainable is False
    np.testing.assert_allclose(
        z.frwrd({}).numpy(), base, **FLOAT64_ARITHMETIC._asdict()
    )
    seed_trainable_variables(z, seed=9)  # only the LoRA tensors are trainable now
    tensors = [t.numpy() for t in z.lora_tensors]
    if mode == "lora":
        assert [t.shape for t in tensors] == [(4, 2), (6, 2)]
        delta = tensors[0] @ tensors[1].T
    else:
        assert [t.shape for t in tensors] == [(4, 6)]
        delta = tensors[0]
    assert np.abs(delta).max() > 0.0
    np.testing.assert_allclose(
        z.frwrd({}).numpy(), base + delta, **FLOAT64_ARITHMETIC._asdict()
    )
    z.finalize_lora_update()
    assert not z.lora
    assert z.lora_config is None
    assert "lora_config" not in z._init_args
    assert not hasattr(z, "lora_tensors")
    assert z.w.trainable is True
    np.testing.assert_allclose(
        z.w.numpy(), base + delta, **FLOAT64_ARITHMETIC._asdict()
    )
    np.testing.assert_allclose(
        z.frwrd({}).numpy(), base + delta, **FLOAT64_ARITHMETIC._asdict()
    )


def test_scalar_chemical_embedding_lora_from_the_constructor_is_enabled_at_build():
    z = _geo_z("random", lora_config=_GEO_LORA)
    assert z.lora
    assert len(z.lora_tensors) == 2
    assert z.lora_config == _GEO_LORA
    assert z.w.trainable is False
    assert len(z.trainable_variables) == 2


def test_scalar_chemical_embedding_unknown_lora_mode_leaves_the_flag_set():
    # Pins current behaviour (reported as a finding): enable_lora_adaptation sets lora = True and
    # records lora_config before initialize_lora_tensors rejects the mode, so after the ValueError
    # the instruction claims LoRA is on with no tensors; frwrd then fails with an AttributeError.
    z = _geo_z("random")
    with pytest.raises(ValueError, match="Unknown mode: bogus"):
        z.enable_lora_adaptation({"mode": "bogus", "rank": 2, "alpha": 1.0})
    assert z.lora
    assert not hasattr(z, "lora_tensors")
    with pytest.raises(AttributeError):
        z.frwrd({})


def test_scalar_chemical_embedding_get_index_to_select():
    z = _geo_z()
    index = z.get_index_to_select(["O", "H", "N", "O"])
    assert index.dtype == tf.int32
    assert index.numpy().tolist() == [3, 0, 2, 3]  # order of the request, repeats kept
    assert z.get_index_to_select([]).numpy().tolist() == []
    shuffled = ScalarChemicalEmbedding(
        name="Z", element_map={"Fe": 2, "Ni": 0}, embedding_size=2
    )
    assert shuffled.get_index_to_select(["Ni", "Fe"]).numpy().tolist() == [0, 2]
    with pytest.raises(ValueError, match="Element Xe not found in the map"):
        z.get_index_to_select(["H", "Xe"])


def test_scalar_chemical_embedding_prepare_variables_selects_rows_like_numpy():
    z = _geo_z("random", size=5)
    seed_trainable_variables(z)
    index = z.get_index_to_select(["O", "H"])
    prepared = z.prepare_variables_for_selected_elements(index)
    assert set(prepared) == {"element_map_symbols", "element_map_index", "w"}
    np.testing.assert_array_equal(prepared["w"].numpy(), z.w.numpy()[np.array([3, 0])])
    assert prepared["element_map_symbols"].numpy().tolist() == [b"O", b"H"]
    assert prepared["element_map_index"].numpy().tolist() == [0, 1]
    assert prepared["element_map_index"].dtype == z.element_map_index.dtype
    assert prepared["w"].dtype == z.w.dtype
    assert prepared["w"].trainable
    assert not prepared["element_map_symbols"].trainable
    assert not prepared["element_map_index"].trainable
    # the original table is untouched and the new variables are independent copies
    assert z.w.shape == (4, 5)
    prepared["w"].assign(tf.zeros_like(prepared["w"]))
    assert np.abs(z.w.numpy()).max() > 0.0


@pytest.mark.parametrize("rows", [[0], [3, 2, 1, 0], [1, 1, 2]])
def test_scalar_chemical_embedding_prepare_variables_any_row_selection(rows):
    z = _geo_z("random", size=3)
    seed_trainable_variables(z)
    prepared = z.prepare_variables_for_selected_elements(
        tf.constant(rows, dtype=tf.int32)
    )
    np.testing.assert_array_equal(prepared["w"].numpy(), z.w.numpy()[rows])
    assert prepared["element_map_index"].numpy().tolist() == list(range(len(rows)))
    assert [s.decode() for s in prepared["element_map_symbols"].numpy()] == [
        list(_GEO_ELEMENTS)[i] for i in rows
    ]


def test_scalar_chemical_embedding_upd_init_args_new_elements_rewrites_the_saved_map():
    z = _geo_z()
    new_map = {"O": 0, "H": 1}
    assert z._init_args["element_map"] == _GEO_ELEMENTS
    z.upd_init_args_new_elements(new_map)
    assert z._init_args["element_map"] == new_map
    assert z.to_dict()["element_map"] == new_map
    assert ScalarChemicalEmbedding.from_dict(z.to_dict()).number_of_elements == 2


# --------------------------------------------------------------------------------------
# Saved yaml of every class of the group, written by to_dict and read by from_dict
# --------------------------------------------------------------------------------------
_GEO_ROUND_TRIP = {
    "BondLength": lambda: BondLength(name="d"),
    "ScaledBondVector": lambda: ScaledBondVector(bond_length="d", name="rhat"),
    "SphericalHarmonic": lambda: SphericalHarmonic(vhat="rhat", lmax=2, name="Y"),
    "RadialBasis": lambda: RadialBasis(
        bonds="d", basis_type="SBessel", nfunc=4, rcut=5.0, normalized=True, p=5
    ),
    "MLPRadialFunction": lambda: MLPRadialFunction(
        n_rad_max=2, lmax=1, basis="g", input_shape=4, activation="tanh"
    ),
    "MLPRadialFunction_v2": lambda: MLPRadialFunction_v2(
        n_rad_max=2, lmax=1, basis="g", input_shape=4, embed_i=True
    ),
    "ScalarChemicalEmbedding": lambda: ScalarChemicalEmbedding(
        name="Z", element_map=_GEO_ELEMENTS, embedding_size=3, init="delta"
    ),
}


@pytest.mark.parametrize("class_name", list(_GEO_ROUND_TRIP))
def test_group_a_to_dict_from_dict_round_trip(class_name):
    original = _GEO_ROUND_TRIP[class_name]()
    saved = original.to_dict()
    assert saved["__cls__"] == f"tensorpotential.instructions.compute.{class_name}"
    restored = type(original).from_dict(saved)
    assert type(restored) is type(original)
    assert restored.to_dict() == saved
    assert restored.name == original.name


# ------------------------------------------------------------------------------------------------
# Independent oracles: real harmonics, real Clebsch-Gordan coefficients, real Wigner matrices
# ------------------------------------------------------------------------------------------------


def _eqv_real_harmonics(vecs: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics ``sqrt(4 pi) Y_lm`` of unit vectors, columns ordered ``l, m=-l..l``.

    Built from ``scipy.special.sph_harm_y`` with the Condon-Shortley phase: ``m > 0`` is
    ``sqrt(2) (-1)^m Re Y_l^m``, ``m < 0`` is ``sqrt(2) (-1)^m Im Y_l^|m|``. The factor ``sqrt(4 pi)``
    is the normalisation of the library (the mean square over the sphere is 1).

    Parameters
    ----------
    vecs : numpy.ndarray
        Unit vectors, shape ``[n, 3]``.
    lmax : int
        Highest degree.

    Returns
    -------
    numpy.ndarray
        Shape ``[n, (lmax + 1) ** 2]``.
    """
    theta = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
    phi = np.arctan2(vecs[:, 1], vecs[:, 0])
    cols = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if m == 0:
                col = sph_harm_y(l, 0, theta, phi).real
            elif m > 0:
                col = np.sqrt(2) * (-1) ** m * sph_harm_y(l, m, theta, phi).real
            else:
                col = np.sqrt(2) * (-1) ** m * sph_harm_y(l, -m, theta, phi).imag
            cols.append(col)
    return np.sqrt(4 * np.pi) * np.stack(cols, axis=1)


def _eqv_c2r(l: int) -> np.ndarray:
    """Matrix ``U`` with ``Y_real = U @ Y_complex`` for the harmonics of ``_eqv_real_harmonics``.

    ``Y_{l,m>0} = (-1)^m / sqrt(2) Y^m + 1 / sqrt(2) Y^-m`` and
    ``Y_{l,m<0} = -i (-1)^a / sqrt(2) Y^a + i / sqrt(2) Y^-a`` with ``a = |m|``.
    """
    n = 2 * l + 1
    u = np.zeros((n, n), dtype=complex)
    u[l, l] = 1.0
    for a in range(1, l + 1):
        u[l + a, l + a] = (-1) ** a / np.sqrt(2)
        u[l + a, l - a] = 1 / np.sqrt(2)
        u[l - a, l + a] = -1j * (-1) ** a / np.sqrt(2)
        u[l - a, l - a] = 1j / np.sqrt(2)
    return u


@lru_cache(maxsize=None)
def _eqv_real_cg(l1: int, l2: int, big_l: int) -> np.ndarray:
    """Real coupling tensor ``C[M, m1, m2]`` of two real harmonic multiplets onto degree ``L``.

    The complex coefficients come from ``sympy.physics.quantum.cg.CG`` and are taken to the real basis
    with ``_eqv_c2r``; the factor ``(-i)^(l1 + l2 - L)`` makes them real (it is a sign for the natural
    parity ``l1 + l2 + L`` even and ``-/+ i`` otherwise, where the product is real after the
    conversion). This is the phase convention of the library: a function of one vector coupled with
    itself gives ``C^{L}_{M} Y_{l1} Y_{l2}`` proportional to ``Y_{LM}`` with the sign fixed by it.

    Returns
    -------
    numpy.ndarray
        Shape ``[2L + 1, 2 l1 + 1, 2 l2 + 1]``.
    """
    cg = np.zeros((2 * big_l + 1, 2 * l1 + 1, 2 * l2 + 1))
    for mm in range(-big_l, big_l + 1):
        for m1 in range(-l1, l1 + 1):
            m2 = mm - m1
            if abs(m2) <= l2:
                cg[mm + big_l, m1 + l1, m2 + l2] = float(
                    CG(S(l1), S(m1), S(l2), S(m2), S(big_l), S(mm)).doit()
                )
    real = np.einsum(
        "MN,Nab,ka,lb->Mkl",
        _eqv_c2r(big_l),
        cg,
        _eqv_c2r(l1).conj(),
        _eqv_c2r(l2).conj(),
    )
    real = real * (-1j) ** (l1 + l2 - big_l)
    assert np.abs(real.imag).max() < 1e-12, "the conversion must give real coefficients"
    return real.real


def _eqv_wigner(rot: Rotation, lmax: int) -> dict[int, np.ndarray]:
    """Real Wigner matrices ``D^l(R)`` with ``Y_l(R v) = D^l Y_l(v)``, fitted on random unit vectors."""
    rng = np.random.default_rng(99)
    vecs = rng.normal(size=(40, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    y0 = _eqv_real_harmonics(vecs, lmax)
    y1 = _eqv_real_harmonics(rot.apply(vecs), lmax)
    out = {}
    for l in range(lmax + 1):
        sl = slice(l * l, (l + 1) ** 2)
        out[l] = y1[:, sl].T @ np.linalg.pinv(y0[:, sl].T)
    return out


def _eqv_rotation() -> Rotation:
    return Rotation.from_rotvec(0.9 * np.array([2.0, -1.0, 1.0]) / 6**0.5)


# ------------------------------------------------------------------------------------------------
# A small cluster with a complete bond list
# ------------------------------------------------------------------------------------------------

_EQV_TYPES = np.array([0, 1, 0, 1, 1], dtype=np.int32)


def _eqv_positions(n_atoms: int = 5, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.5, 1.5, size=(n_atoms, 3))


def _eqv_bonds(
    pos: np.ndarray, types: np.ndarray = _EQV_TYPES, order: np.ndarray | None = None
) -> dict:
    """Bond keys of the complete directed graph of ``pos``, sorted by centre atom (``order`` permutes bonds)."""
    n = len(pos)
    ind_i = np.repeat(np.arange(n), n - 1)
    ind_j = np.array([j for i in range(n) for j in range(n) if j != i])
    if order is not None:
        ind_i, ind_j = ind_i[order], ind_j[order]
    return {
        constants.BOND_VECTOR: pos[ind_j] - pos[ind_i],
        constants.BOND_IND_I: ind_i.astype(np.int32),
        constants.BOND_IND_J: ind_j.astype(np.int32),
        constants.BOND_MU_I: types[ind_i],
        constants.BOND_MU_J: types[ind_j],
        constants.ATOMIC_MU_I: types.copy(),
        constants.N_ATOMS_BATCH_TOTAL: n,
        constants.N_ATOMS_BATCH_REAL: n,
    }


def _eqv_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------
# Objects: neighbouring instructions that only carry names, sizes and tables
# ------------------------------------------------------------------------------------------------


def _eqv_radial(n_rad: int = 3, lmax: int = 2, name: str = "R") -> LinearRadialFunction:
    """A real radial function; only ``name``, ``n_rad_max``, ``lmax`` and ``l_tile`` are read by the SPBFs."""
    return LinearRadialFunction(
        n_rad_max=n_rad, lmax=lmax, basis="B", input_shape=4, name=name
    )


def _eqv_harmonic(lmax: int = 2, name: str = "Y", n_out: int | None = None):
    y = SphericalHarmonic(vhat="vhat", lmax=lmax, name=name)
    y.n_out = n_out
    return y


def _eqv_embedding(emb: int = 4, name: str = "Z", seed: int = 3):
    z = ScalarChemicalEmbedding(
        name=name, element_map={"H": 0, "C": 1}, embedding_size=emb
    )
    z.build(tf.float64)
    z.w.assign(np.random.default_rng(seed).normal(size=z.w.shape))
    return z


def _eqv_ang_radial(
    pos: np.ndarray,
    lmax: int,
    n_rad: int,
    seed: int = 5,
    ang_lmax: int | None = None,
) -> dict:
    """Bond data plus a random radial tensor ``R[b, n, lm]`` and independent harmonics of the bond direction."""
    data = _eqv_bonds(pos)
    vec = data[constants.BOND_VECTOR]
    n_b = len(vec)
    rng = np.random.default_rng(seed)
    data["R"] = rng.normal(size=(n_b, n_rad, (lmax + 1) ** 2))
    data["Y"] = _eqv_real_harmonics(
        _eqv_unit(vec), lmax if ang_lmax is None else ang_lmax
    )
    return data


def _eqv_np(x) -> np.ndarray:
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)


def _eqv_assert_close(actual, expected, tol=FLOAT64_ARITHMETIC) -> None:
    np.testing.assert_allclose(_eqv_np(actual), expected, **tol._asdict())


# ------------------------------------------------------------------------------------------------
# The oracles of the oracles: the harmonics, the coupling and the Wigner matrices are themselves checked
# ------------------------------------------------------------------------------------------------


def test_eqv_oracle_harmonics_match_the_library_and_are_orthonormal():
    vecs = _eqv_unit(np.random.default_rng(1).normal(size=(4000, 3)))
    y = _eqv_real_harmonics(vecs, 3)
    gram = y.T @ y / len(vecs)
    # Monte-Carlo estimate of the orthonormality integral over the sphere: sampling error
    # 1/sqrt(n) of order 0.016 per entry; the bound is four standard errors of the worst entry.
    assert np.abs(gram - np.eye(16)).max() < 4 * np.sqrt(3.0 / len(vecs))
    lib = SphericalHarmonic(vhat="v", lmax=3, name="Yoracle")
    lib.build(tf.float64)
    _eqv_assert_close(lib.frwrd({"v": vecs[:30]}), y[:30])


def test_eqv_oracle_real_cg_couples_a_vector_with_itself_to_a_harmonic():
    """``Y_l1(r) Y_l2(r)`` projected on ``L`` is ``kappa Y_L(r)`` with the Gaunt factor ``kappa``."""
    vecs = _eqv_unit(np.random.default_rng(2).normal(size=(6, 3)))
    y = _eqv_real_harmonics(vecs, 3)
    for l1, l2, big_l in [(1, 1, 0), (2, 1, 1), (2, 2, 2), (3, 2, 1), (2, 1, 3)]:
        c = _eqv_real_cg(l1, l2, big_l)
        a = y[:, l1 * l1 : (l1 + 1) ** 2]
        b = y[:, l2 * l2 : (l2 + 1) ** 2]
        got = np.einsum("Mkl,ak,al->aM", c, a, b)
        yl = y[:, big_l * big_l : (big_l + 1) ** 2]
        kappa = (
            np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * big_l + 1))
            * float(CG(S(l1), S(0), S(l2), S(0), S(big_l), S(0)).doit())
            * (-1) ** ((l1 + l2 - big_l) // 2)
        )
        _eqv_assert_close(got, kappa * yl, COVARIANCE_F64)


def test_eqv_oracle_wigner_matrices_are_orthogonal_and_compose():
    d = _eqv_wigner(_eqv_rotation(), 3)
    for l, mat in d.items():
        _eqv_assert_close(mat @ mat.T, np.eye(2 * l + 1), COVARIANCE_F64)
    twice = _eqv_wigner(_eqv_rotation() * _eqv_rotation(), 3)
    for l in range(4):
        _eqv_assert_close(d[l] @ d[l], twice[l], COVARIANCE_F64)


# ================================================================================================
# SingleParticleBasisFunctionScalarInd
# ================================================================================================

_EQV_N_RAD = 3
_EQV_LMAX = 2
_EQV_EMB = 4


def _eqv_scalar_ind(
    indicator: bool = True, seed: int = 11, build_dtype=tf.float64, **kwargs
):
    """A built scalar-indicator SPBF with seeded weights (``lin_transform`` is the only variable)."""
    tf.random.set_seed(seed)
    radial = kwargs.pop("radial", _eqv_radial(_EQV_N_RAD, _EQV_LMAX))
    angular = kwargs.pop("angular", _eqv_harmonic(_EQV_LMAX))
    z = _eqv_embedding(_EQV_EMB) if indicator else None
    ins = SingleParticleBasisFunctionScalarInd(
        name="A", radial=radial, angular=angular, indicator=z, **kwargs
    )
    ins.build(build_dtype)
    if indicator:
        seed_trainable_variables(ins.lin_transform, seed=seed)
    return ins, z


def _eqv_scalar_data(pos=None, lmax=_EQV_LMAX, z=None, seed=5) -> dict:
    pos = _eqv_positions() if pos is None else pos
    data = _eqv_ang_radial(pos, lmax, _EQV_N_RAD, seed)
    if z is not None:
        data["Z"] = z.w.numpy()
    return data


def _eqv_scalar_oracle(
    data: dict,
    lin: np.ndarray | None,
    n_atoms: int,
    l_depend: bool = False,
    inv_avg: float | np.ndarray = 1.0,
    lmax: int = _EQV_LMAX,
    gather_key: str = constants.BOND_MU_J,
) -> np.ndarray:
    """``A[i, n, lm] = inv_avg * sum_{j in nbr(i)} R[ij, n, l] Y[ij, lm] w_n(z_j)``, an explicit bond loop.

    ``lin`` is the transform of the indicator, already scaled by ``1 / sqrt(n_in)`` as the layer does.
    """
    r, y = data["R"], data["Y"]
    ind_i, gather = data[constants.BOND_IND_I], data[gather_key]
    n_b, n_rad, n_lm = r.shape
    l_of_lm = np.concatenate([[l] * (2 * l + 1) for l in range(lmax + 1)])
    out = np.zeros((n_atoms, n_rad, n_lm))
    for b in range(n_b):
        term = r[b] * y[b][None, :]
        if lin is not None:
            zb = lin[gather[b]]
            if l_depend:
                zb = zb.reshape(n_rad, lmax + 1)[:, l_of_lm]
                term = term * zb
            else:
                term = term * zb[:, None]
        out[ind_i[b]] += term
    return out * inv_avg


def _eqv_lin_transform(ins, z) -> np.ndarray:
    """``z @ w / sqrt(n_in)``: what ``DenseLayer`` (no bias, no activation) does, from its stored weight."""
    return z.w.numpy() @ ins.lin_transform.w.numpy() / np.sqrt(z.embedding_size)


@pytest.mark.parametrize(
    ("indicator", "l_depend"),
    [(False, False), (True, False), (True, True)],
    ids=["no-indicator", "indicator", "indicator-l-dependent"],
)
def test_spbf_scalar_ind_equals_the_explicit_bond_loop(indicator, l_depend):
    ins, z = _eqv_scalar_ind(indicator, indicator_l_depend=l_depend)
    data = _eqv_scalar_data(z=z)
    out = ins.frwrd(data)
    lin = _eqv_lin_transform(ins, z) if indicator else None
    assert out.shape == (5, _EQV_N_RAD, 9)
    assert out.dtype == tf.float64
    _eqv_assert_close(out, _eqv_scalar_oracle(data, lin, 5, l_depend))


def test_spbf_scalar_ind_indicator_weights_have_the_documented_shapes():
    ins, _ = _eqv_scalar_ind(True)
    assert ins.lin_transform.w.shape.as_list() == [_EQV_EMB, _EQV_N_RAD]
    assert ins.n_out == _EQV_N_RAD
    ins_l, _ = _eqv_scalar_ind(True, indicator_l_depend=True)
    assert ins_l.lin_transform.w.shape.as_list() == [_EQV_EMB, _EQV_N_RAD * 3]
    ins_0, _ = _eqv_scalar_ind(False)
    assert ins_0.lin_transform is None
    assert not ins_0.trainable_variables


@pytest.mark.parametrize("avg", [4.0, 0.25, np.float64(2.5)])
def test_spbf_scalar_ind_scales_with_the_inverse_average_neighbour_count(avg):
    ins, _ = _eqv_scalar_ind(False, avg_n_neigh=avg)
    ref, _ = _eqv_scalar_ind(False, avg_n_neigh=1.0)
    data = _eqv_scalar_data()
    _eqv_assert_close(ins.frwrd(data), _eqv_np(ref.frwrd(data)) / avg)
    assert ins.per_specie_n_neigh is False
    assert float(ins.inv_avg_n_neigh) == pytest.approx(1.0 / avg)


def test_spbf_scalar_ind_per_species_average_scales_each_centre_by_its_own_type():
    avg = {0: 2.0, 1: 5.0}
    ins, _ = _eqv_scalar_ind(False, avg_n_neigh=avg)
    data = _eqv_scalar_data()
    inv = np.array([1 / 2.0, 1 / 5.0])[_EQV_TYPES]
    expected = _eqv_scalar_oracle(data, None, 5) * inv[:, None, None]
    assert ins.per_specie_n_neigh is True
    assert ins.inv_avg_n_neigh.shape.as_list() == [2, 1]
    _eqv_assert_close(ins.frwrd(data), expected)


def test_spbf_scalar_ind_per_species_average_with_no_neighbours_uses_one():
    """A species whose average count is not positive (never seen in the data) is scaled by 1."""
    ins, _ = _eqv_scalar_ind(False, avg_n_neigh={0: 0.0, 1: -3.0})
    np.testing.assert_array_equal(ins.inv_avg_n_neigh.numpy(), [[1.0], [1.0]])
    data = _eqv_scalar_data()
    _eqv_assert_close(ins.frwrd(data), _eqv_scalar_oracle(data, None, 5))


def test_spbf_scalar_ind_per_species_average_is_not_available_for_local_atoms():
    ins, _ = _eqv_scalar_ind(False, avg_n_neigh={0: 2.0, 1: 5.0})
    data = _eqv_scalar_data()
    data[constants.ATOMIC_MU_I_LOCAL] = data[constants.ATOMIC_MU_I]
    with pytest.raises(NotImplementedError):
        ins.frwrd(data, local=True)


@pytest.mark.parametrize("bad", [3, "4.0", None, [1.0]])
def test_spbf_scalar_ind_refuses_an_average_that_is_neither_float_nor_dict(bad):
    # Pins current behaviour (reported as a finding): an int such as ``avg_n_neigh=4`` is refused too.
    with pytest.raises(TypeError, match="avg_n_neigh must be float or dict"):
        _eqv_scalar_ind(False, avg_n_neigh=bad)


def test_spbf_scalar_ind_local_atoms_are_counted_from_the_local_index():
    """With ``local=True`` the output has one row per local atom (here the first three of five)."""
    ins, z = _eqv_scalar_ind(True)
    data = _eqv_scalar_data(z=z)
    keep = data[constants.BOND_IND_I] < 3
    for key in (
        "R",
        "Y",
        constants.BOND_IND_I,
        constants.BOND_IND_J,
        constants.BOND_MU_J,
    ):
        data[key] = data[key][keep]
    data[constants.ATOMIC_MU_I_LOCAL] = _EQV_TYPES[:3]
    out = ins.frwrd(data, local=True)
    assert out.shape == (3, _EQV_N_RAD, 9)
    _eqv_assert_close(out, _eqv_scalar_oracle(data, _eqv_lin_transform(ins, z), 3))


def test_spbf_scalar_ind_without_neighbour_sum_returns_one_row_per_bond():
    ins, z = _eqv_scalar_ind(True, sum_neighbors=False, avg_n_neigh=7.0)
    data = _eqv_scalar_data(z=z)
    lin = _eqv_lin_transform(ins, z)
    expected = (
        data["R"] * data["Y"][:, None, :] * lin[data[constants.BOND_MU_J]][:, :, None]
    )
    out = ins.frwrd(data)
    assert out.shape == (20, _EQV_N_RAD, 9)
    _eqv_assert_close(out, expected)  # the average neighbour count only enters the sum


@pytest.mark.parametrize("sum_neighbors", [True, False])
def test_spbf_scalar_ind_lm_first_is_the_transpose_of_the_standard_layout(
    sum_neighbors,
):
    kwargs = {"sum_neighbors": sum_neighbors}
    std, z = _eqv_scalar_ind(True, **kwargs)
    first, _ = _eqv_scalar_ind(True, lm_first=True, **kwargs)
    data = _eqv_scalar_data(z=z)
    a = _eqv_np(std.frwrd(data))
    b = _eqv_np(first.frwrd(data))
    assert b.shape == (a.shape[2], a.shape[0], a.shape[1])
    _eqv_assert_close(b, np.transpose(a, (2, 0, 1)))


def test_spbf_scalar_ind_angular_degrees_above_lmax_are_dropped():
    """``lmax=1`` of a degree-2 harmonic keeps the first four columns and the ``l <= 1`` table rows."""
    full, z = _eqv_scalar_ind(True)
    cut, _ = _eqv_scalar_ind(
        True, lmax=1, radial=_eqv_radial(_EQV_N_RAD, 1), indicator_l_depend=False
    )
    cut.lin_transform.w.assign(full.lin_transform.w)
    data = _eqv_scalar_data(z=z)
    cut_data = dict(data, R=data["R"][:, :, :4])
    out = cut.frwrd(cut_data)
    assert out.shape == (5, _EQV_N_RAD, 4)
    _eqv_assert_close(out, _eqv_np(full.frwrd(data))[:, :, :4])
    assert cut.slice_angular == 4
    assert cut.lmax == 1
    assert cut.coupling_meta_data["l"].max() == 1
    assert len(cut.coupling_meta_data) == 4
    assert full.slice_angular is None
    # the table of the harmonic is not edited by slicing
    assert cut.angular.coupling_meta_data["l"].max() == _EQV_LMAX


def test_spbf_scalar_ind_refuses_an_lmax_above_that_of_the_harmonics():
    with pytest.raises(AssertionError, match="is too small for specified"):
        _eqv_scalar_ind(False, lmax=3, radial=_eqv_radial(_EQV_N_RAD, 3))


def test_spbf_scalar_ind_refuses_a_radial_function_of_another_lmax():
    with pytest.raises(AssertionError, match=r"do not match \(1 vs 2\)"):
        _eqv_scalar_ind(False, radial=_eqv_radial(_EQV_N_RAD, 1))


def test_spbf_scalar_ind_float32_parameters_give_float32_output_from_float64_data():
    ins, z = _eqv_scalar_ind(True, build_dtype=tf.float32)
    data = _eqv_scalar_data(z=z)
    out = ins.frwrd(data)
    assert out.dtype == tf.float32
    assert ins.lin_transform.w.dtype == tf.float32
    assert ins.inv_avg_n_neigh.dtype == tf.float32
    ref = _eqv_scalar_oracle(data, _eqv_lin_transform(ins, z), 5)
    np.testing.assert_allclose(out.numpy(), ref, **FLOAT32_NETWORK._asdict())


def test_spbf_scalar_ind_casts_angular_to_the_radial_dtype():
    ins, _ = _eqv_scalar_ind(False)
    data = _eqv_scalar_data()
    data["Y"] = data["Y"].astype(np.float32)
    out = ins.frwrd(data)
    assert out.dtype == tf.float64
    ref = _eqv_scalar_oracle(dict(data, Y=data["Y"].astype(np.float64)), None, 5)
    _eqv_assert_close(out, ref)


def test_spbf_scalar_ind_per_atom_indicator_is_gathered_by_the_neighbour_atom():
    """An indicator with ``is_per_atom`` (one row per atom) is indexed by ``ind_j``, not by the type of ``j``.

    No shipped ``ScalarChemicalEmbedding`` sets ``is_per_atom``; a subclass that does stands in for the
    neighbouring instruction (the unit under test is the real SPBF).
    """

    class _PerAtomEmbedding(ScalarChemicalEmbedding):
        is_per_atom = True

    z = _PerAtomEmbedding(
        name="Z", element_map={"H": 0, "C": 1}, embedding_size=_EQV_EMB
    )
    ins = SingleParticleBasisFunctionScalarInd(
        name="A",
        radial=_eqv_radial(),
        angular=_eqv_harmonic(),
        indicator=z,
    )
    assert ins.indicator_is_per_atom
    ins.build(tf.float64)
    seed_trainable_variables(ins.lin_transform)
    data = _eqv_scalar_data()
    per_atom = np.random.default_rng(8).normal(size=(5, _EQV_EMB))
    data["Z"] = per_atom
    lin = per_atom @ ins.lin_transform.w.numpy() / np.sqrt(_EQV_EMB)
    out = ins.frwrd(data)
    _eqv_assert_close(
        out, _eqv_scalar_oracle(data, lin, 5, gather_key=constants.BOND_IND_J)
    )


def test_spbf_scalar_ind_building_twice_keeps_the_weights_and_constants():
    ins, _ = _eqv_scalar_ind(True, avg_n_neigh=2.0)
    w = ins.lin_transform.w
    before = w.numpy().copy()
    inv = ins.inv_avg_n_neigh
    ins.build(tf.float32)
    assert ins.lin_transform.w is w
    np.testing.assert_array_equal(w.numpy(), before)
    assert ins.inv_avg_n_neigh is inv
    assert inv.dtype == tf.float64


def test_spbf_scalar_ind_call_adds_the_output_and_never_edits_the_inputs():
    ins, z = _eqv_scalar_ind(True)
    data = _eqv_scalar_data(z=z)
    snapshot = {k: np.array(v, copy=True) for k, v in data.items() if k != "A"}
    out = ins(data)
    assert set(out) == set(snapshot) | {"A"}
    for key, value in snapshot.items():
        np.testing.assert_array_equal(np.asarray(data[key]), value)
    with pytest.raises(AssertionError, match="already exists"):
        ins(data)


def test_spbf_scalar_ind_symbols_and_origin_come_from_the_harmonic():
    ins, _ = _eqv_scalar_ind(False)
    meta = ins.coupling_meta_data
    assert meta["l"].tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert [str(s) for s in meta["symbol"]][:2] == ["A_(0,0)", "A_(1,-1)"]
    assert ins.coupling_origin is ins.angular.coupling_origin


def test_spbf_scalar_ind_constructor_arguments_are_captured_for_the_yaml():
    ins, _ = _eqv_scalar_ind(
        True, indicator_l_depend=True, avg_n_neigh=3.0, lm_first=True
    )
    args = ins._init_args
    assert args["indicator_l_depend"] is True
    assert args["lm_first"] is True
    assert args["avg_n_neigh"] == 3.0
    assert args["sum_neighbors"] is True
    assert args["lmax"] is None
    assert args["radial"].name == "R"
    assert ins.to_dict()["__cls__"].endswith("SingleParticleBasisFunctionScalarInd")


# ---- physical invariances of the full chain: radial basis, harmonics, indicator, neighbour sum -------


def _eqv_chain_scalar(lmax: int = 2, lm_first: bool = False):
    """The shipped chain ``BondLength -> ScaledBondVector -> Y, RadialBasis -> R, Z -> A`` with seeded weights."""
    from tensorpotential.instructions import (
        BondLength,
        RadialBasis,
        ScaledBondVector,
    )

    tf.random.set_seed(21)
    d_ij = BondLength()
    rhat = ScaledBondVector(bond_length=d_ij)
    g = RadialBasis(bonds=d_ij, basis_type="SBessel", nfunc=6, rcut=4.0)
    radial = LinearRadialFunction(n_rad_max=3, lmax=lmax, basis=g, name="R")
    y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    z = _eqv_embedding(4)
    a = SingleParticleBasisFunctionScalarInd(
        name="A",
        radial=radial,
        angular=y,
        indicator=z,
        avg_n_neigh=3.0,
        lm_first=lm_first,
    )
    steps = [d_ij, rhat, g, radial, y, a]
    for step in steps:
        step.build(tf.float64)
    seed_trainable_variables(a)
    return steps, z


def _eqv_run_chain(steps, z, data: dict) -> np.ndarray:
    data = dict(data)
    data["Z"] = z.w.numpy()
    for step in steps:
        data = step(data)
    return data["A"].numpy()


def test_spbf_scalar_ind_chain_is_equivariant_translation_and_permutation_invariant():
    steps, z = _eqv_chain_scalar()
    pos = _eqv_positions()
    base = _eqv_run_chain(steps, z, _eqv_bonds(pos))
    assert np.abs(base).max() > 1e-3

    # translation: the bond vectors are differences of positions, so a shift cancels exactly
    shifted = _eqv_run_chain(steps, z, _eqv_bonds(pos + np.array([3.0, -2.0, 7.5])))
    _eqv_assert_close(shifted, base, COVARIANCE_F64)

    # permutation of the neighbour order: the sum over bonds does not depend on it
    order = np.random.default_rng(3).permutation(20)
    permuted = _eqv_run_chain(steps, z, _eqv_bonds(pos, order=order))
    _eqv_assert_close(permuted, base, COVARIANCE_F64)

    # permutation of the atoms permutes the rows
    perm = np.array([3, 0, 4, 1, 2])
    relabelled = _eqv_run_chain(steps, z, _eqv_bonds(pos[perm], _EQV_TYPES[perm]))
    _eqv_assert_close(relabelled, base[perm], COVARIANCE_F64)

    # rotation: l = 0 is invariant, each degree-l block rotates with the real Wigner matrix
    rot = _eqv_rotation()
    rotated = _eqv_run_chain(steps, z, _eqv_bonds(rot.apply(pos)))
    d = _eqv_wigner(rot, _EQV_LMAX)
    _eqv_assert_close(rotated[:, :, :1], base[:, :, :1], COVARIANCE_F64)
    for l in range(_EQV_LMAX + 1):
        sl = slice(l * l, (l + 1) ** 2)
        expected = np.einsum("ab,ikb->ika", d[l], base[:, :, sl])
        _eqv_assert_close(rotated[:, :, sl], expected, COVARIANCE_F64)


def test_spbf_scalar_ind_chain_in_lm_first_layout_is_the_transpose():
    steps, z = _eqv_chain_scalar()
    steps_t, z_t = _eqv_chain_scalar(lm_first=True)
    data = _eqv_bonds(_eqv_positions())
    std = _eqv_run_chain(steps, z, data)
    first = _eqv_run_chain(steps_t, z_t, data)
    _eqv_assert_close(first, np.transpose(std, (2, 0, 1)), COVARIANCE_F64)


@pytest.mark.parametrize(
    "config",
    [{"rank": 2, "alpha": 1.0}, {"mode": "additive"}],
    ids=["lora", "additive"],
)
def test_spbf_scalar_ind_lora_adds_the_low_rank_update_to_the_indicator_weight(config):
    ins, z = _eqv_scalar_ind(True)
    data = _eqv_scalar_data(z=z)
    plain = _eqv_np(ins.frwrd(data))
    w = ins.lin_transform.w
    w_before = w.numpy().copy()

    ins.enable_lora_adaptation(config)
    assert ins.lora
    assert ins._init_args["lora_config"] == config
    assert not w.trainable
    tensors = ins.lin_transform.lora_tensors
    assert len(tensors) == (1 if "mode" in config else 2)
    _eqv_assert_close(ins.frwrd(data), plain)  # the update starts at zero

    rng = np.random.default_rng(4)
    for t in tensors:
        t.assign(rng.normal(size=t.shape))
    delta = (
        tensors[0].numpy()
        if "mode" in config
        else tensors[0].numpy() @ tensors[1].numpy().T
    )
    lin = z.w.numpy() @ (w_before + delta) / np.sqrt(_EQV_EMB)
    updated = _eqv_np(ins.frwrd(data))
    assert np.abs(updated - plain).max() > 1e-3
    _eqv_assert_close(updated, _eqv_scalar_oracle(data, lin, 5))

    ins.finalize_lora_update()
    assert not ins.lora
    assert "lora_config" not in ins._init_args
    assert w.trainable
    assert not hasattr(ins.lin_transform, "lora_tensors")
    np.testing.assert_allclose(
        w.numpy(), w_before + delta, **FLOAT64_ARITHMETIC._asdict()
    )
    _eqv_assert_close(ins.frwrd(data), updated)


def test_spbf_scalar_ind_built_with_a_lora_config_starts_with_a_frozen_indicator_weight():
    config = {"rank": 3, "alpha": 2.0}
    ins, z = _eqv_scalar_ind(True, lora_config=config)
    assert ins.lora
    assert ins.lin_transform.lora
    assert not ins.lin_transform.w.trainable
    names = {v.name for v in ins.trainable_variables}
    assert any("LORA" in n for n in names)
    assert not any(n.startswith("A_ChemIndTransf/DenseLayer") for n in names)
    ins.finalize_lora_update()
    assert ins.lin_transform.w.trainable


def test_spbf_scalar_ind_lora_without_an_indicator_only_toggles_the_flag():
    ins, _ = _eqv_scalar_ind(False)
    ins.enable_lora_adaptation({"rank": 2, "alpha": 1.0})
    assert ins.lora
    assert not ins.trainable_variables
    ins.finalize_lora_update()
    assert not ins.lora
    assert ins.lora_config is None


# ------------------------------------------------------------------------------------------------
# Coupling oracle shared by ProductFunction and the equivariant SPBF
# ------------------------------------------------------------------------------------------------


def _eqv_strip(hist: str) -> str:
    return hist.replace("+", "").replace("-", "")


def _eqv_split_hist(hist: str) -> tuple[tuple[str, int], tuple[str, int]]:
    """``"((1,1)0,2)"`` -> ``(("(1,1)", 0), ("", 2))``: the history and degree of the two coupled functions."""
    s = _eqv_strip(hist)
    assert s[0] == "(" and s[-1] == ")"
    body, depth = s[1:-1], 0
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts = body[:i], body[i + 1 :]
            break
    out = []
    for p in parts:
        k = len(p)
        while k > 0 and p[k - 1].isdigit():
            k -= 1
        out.append((p[:k], int(p[k:])))
    return out[0], out[1]


def _eqv_index(meta) -> dict[tuple[int, int, str], int]:
    """``(l, m, history) -> column`` of a coupling table; the key must be unique."""
    idx = {}
    for i, (l, m, h) in enumerate(zip(meta["l"], meta["m"], meta["hist"])):
        key = (int(l), int(m), _eqv_strip(h))
        assert key not in idx
        idx[key] = i
    return idx


def _eqv_couple(
    out_meta,
    left_meta,
    right_meta,
    left: np.ndarray,
    right: np.ndarray,
    norms: dict[tuple[int, int, int], float] | None = None,
) -> np.ndarray:
    """``out[..., r] = sum_{m1 m2} C[L M, m1 m2] left[..., (l1 m1 h1)] right[..., (l2 m2 h2)]`` for every row ``r``.

    The rows, their degrees and their histories are read from the coupling tables (labels only); the
    numbers come from ``_eqv_real_cg``, times ``norms[(l1, l2, L)]`` when given. ``left`` and ``right``
    have the columns of their tables on the last axis; broadcasting applies to the others.
    """
    li, ri = _eqv_index(left_meta), _eqv_index(right_meta)
    shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    out = np.zeros((*shape, len(out_meta)))
    for r, (big_l, mm, hist) in enumerate(
        zip(out_meta["l"], out_meta["m"], out_meta["hist"])
    ):
        (h1, l1), (h2, l2) = _eqv_split_hist(hist)
        c = _eqv_real_cg(l1, l2, int(big_l))[int(mm) + int(big_l)]
        if norms is not None:
            c = c * norms[(l1, l2, int(big_l))]
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                if c[m1 + l1, m2 + l2] != 0.0:
                    out[..., r] += (
                        c[m1 + l1, m2 + l2]
                        * left[..., li[(l1, m1, h1)]]
                        * right[..., ri[(l2, m2, h2)]]
                    )
    return out


def _eqv_triples(meta) -> set[tuple[int, int, int]]:
    """The set ``(l1, l2, L)`` of the rows of a table of products of plain harmonics."""
    return {
        (_eqv_split_hist(h)[0][1], _eqv_split_hist(h)[1][1], int(big_l))
        for h, big_l in zip(meta["hist"], meta["l"])
    }


def _eqv_natural_triples(
    lmax1: int, lmax2: int, big_lmax: int
) -> set[tuple[int, int, int]]:
    """Triangle rule ``|l1 - l2| <= L <= l1 + l2`` with the parity ``(-1)^L = (-1)^(l1 + l2)``."""
    return {
        (l1, l2, big_l)
        for l1 in range(lmax1 + 1)
        for l2 in range(lmax2 + 1)
        for big_l in range(abs(l1 - l2), min(l1 + l2, big_lmax) + 1)
        if (l1 + l2 - big_l) % 2 == 0
    }


def _eqv_blocks(meta) -> list[tuple[int, slice]]:
    """Consecutive rows of one ``(L, parity, history)`` group: ``(L, slice)`` pairs (``M`` ascending)."""
    blocks, start = [], 0
    keys = list(zip(meta["l"], meta["parity"], meta["hist"]))
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            blocks.append((int(keys[start][0]), slice(start, i)))
            start = i
    return blocks


def _eqv_assert_equivariant(meta, out_rot, out, rot, tol=COVARIANCE_F64) -> None:
    """``out_rot[..., block L] = D^L out[..., block L]`` for every ``(L, parity, history)`` block."""
    d = _eqv_wigner(rot, int(meta["l"].max()))
    for big_l, sl in _eqv_blocks(meta):
        assert sl.stop - sl.start == 2 * big_l + 1
        expected = np.einsum("ab,...b->...a", d[big_l], out[..., sl])
        _eqv_assert_close(out_rot[..., sl], expected, tol)


# ================================================================================================
# SingleParticleBasisFunctionEquivariantInd
# ================================================================================================


def _eqv_equiv_ind(
    lmax: int = 2,
    big_lmax: int = 2,
    ind_lmax: int = 2,
    n_rad: int = 3,
    chem: bool | str = False,
    ang_lmax: int | None = None,
    build_dtype=tf.float64,
    seed: int = 13,
    **kwargs,
):
    """A built equivariant-indicator SPBF whose indicator is a harmonic of degree ``ind_lmax``."""
    tf.random.set_seed(seed)
    radial = _eqv_radial(n_rad, lmax)
    angular = _eqv_harmonic(lmax if ang_lmax is None else ang_lmax, "Y")
    ind = _eqv_harmonic(ind_lmax, "I", n_out=n_rad)
    z = _eqv_embedding(_EQV_EMB) if chem else None
    ins = SingleParticleBasisFunctionEquivariantInd(
        name="YI",
        angular=angular,
        indicator=ind,
        radial=radial,
        lmax=lmax,
        Lmax=big_lmax,
        chemical_embedding=z,
        **kwargs,
    )
    ins.build(build_dtype)
    if chem:
        seed_trainable_variables(ins.chem_linear, seed=seed)
    return ins, z


def _eqv_equiv_data(
    ins, z=None, pos=None, seed: int = 6, ind_scale: float = 1.0
) -> dict:
    """Bond data, random radial tensor, harmonics of the bonds, and a random indicator per atom."""
    pos = _eqv_positions() if pos is None else pos
    n_rad = ins.radial.n_rad_max
    data = _eqv_ang_radial(pos, ins.radial.lmax, n_rad, seed, ins.angular.lmax)
    n_lm_ind = len(ins.indicator.coupling_meta_data)
    data["I"] = ind_scale * np.random.default_rng(seed + 1).normal(
        size=(len(pos), n_rad, n_lm_ind)
    )
    if z is not None:
        data["Z"] = z.w.numpy()
    return data


def _eqv_equiv_oracle(
    ins,
    data: dict,
    ind_t: np.ndarray,
    n_atoms: int,
    inv_avg: float = 1.0,
    sum_neighbors: bool = True,
    zproj: np.ndarray | None = None,
    z_key: str = constants.BOND_MU_J,
) -> np.ndarray:
    """``YI[i, n, r] = inv_avg sum_{j in nbr(i)} couple(R Y_ij, I_j)`` by an explicit bond loop and coupling table."""
    n_ang = (
        (ins.angular.lmax + 1) ** 2 if ins.slice_angular is None else ins.slice_angular
    )
    ry = data["R"] * data["Y"][:, None, :n_ang]
    ind_j = data[constants.BOND_IND_J]
    bond_i = ind_t[ind_j].copy()
    if zproj is not None:
        l0 = ins.chem_l0_idx
        bond_i[:, :, l0] += zproj[data[z_key]]
    per_bond = _eqv_couple(
        ins.coupling_meta_data,
        ins.angular.coupling_meta_data,
        ins.indicator.coupling_meta_data,
        ry,
        bond_i,
    )
    if not sum_neighbors:
        return per_bond
    out = np.zeros((n_atoms, *per_bond.shape[1:]))
    for b, i in enumerate(data[constants.BOND_IND_I]):
        out[i] += per_bond[b]
    return out * inv_avg


def test_spbf_equiv_ind_table_is_the_natural_parity_coupling_of_harmonics():
    ins, _ = _eqv_equiv_ind(lmax=2, big_lmax=3, ind_lmax=2)
    assert _eqv_triples(ins.coupling_meta_data) == _eqv_natural_triples(2, 2, 3)
    meta = ins.coupling_meta_data
    assert len(meta) == sum(
        2 * big_l + 1 for _, _, big_l in _eqv_natural_triples(2, 2, 3)
    )
    assert set(zip(meta["l"], meta["parity"])) == {(0, 1), (1, -1), (2, 1), (3, -1)}
    assert ins.coupling_origin == ["Y", "I"]
    assert ins.n_out == _EQV_N_RAD
    assert ins.lmax == 3  # the output degree, not the input degree


@pytest.mark.parametrize("dense", [False, True], ids=["segment-sum", "dense"])
@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
def test_spbf_equiv_ind_equals_the_bond_loop_over_the_coupling(dense, lm_first):
    ins, _ = _eqv_equiv_ind(avg_n_neigh=4.0, lm_first=lm_first, dense_nbr=dense)
    data = _eqv_equiv_data(ins)
    ind = data["I"]
    feed = dict(data, I=np.transpose(ind, (2, 0, 1)) if lm_first else ind)
    out = _eqv_np(ins.frwrd(feed))
    expected = _eqv_equiv_oracle(ins, data, ind, 5, inv_avg=0.25)
    if lm_first:
        expected = np.transpose(expected, (2, 0, 1))
    n_rows = len(ins.coupling_meta_data)
    assert out.shape == ((n_rows, 5, 3) if lm_first else (5, 3, n_rows))
    _eqv_assert_close(out, expected)


def test_spbf_equiv_ind_dense_and_segment_sum_paths_agree_on_a_uniform_bond_layout():
    seg, _ = _eqv_equiv_ind(dense_nbr=False)
    dense, _ = _eqv_equiv_ind(dense_nbr=True)
    data = _eqv_equiv_data(seg)
    _eqv_assert_close(dense.frwrd(data), _eqv_np(seg.frwrd(data)))


def test_spbf_equiv_ind_dense_nbr_default_comes_from_the_instruction_manager():
    plain, _ = _eqv_equiv_ind()
    assert plain.dense_nbr is False
    assert plain._init_args["dense_nbr"] is False
    assert plain.dense_capable
    with InstructionManager(dense_nbr=True):
        inherited, _ = _eqv_equiv_ind()
    assert inherited.dense_nbr is True
    assert inherited._init_args["dense_nbr"] is True


def test_spbf_equiv_ind_explicit_dense_nbr_beats_the_manager_default():
    with InstructionManager(dense_nbr=True):
        ins, _ = _eqv_equiv_ind(dense_nbr=False)
    assert ins.dense_nbr is False


def test_spbf_equiv_ind_without_neighbour_sum_returns_one_row_per_bond():
    ins, _ = _eqv_equiv_ind(sum_neighbors=False, avg_n_neigh=9.0)
    data = _eqv_equiv_data(ins)
    out = _eqv_np(ins.frwrd(data))
    n_rows = len(ins.coupling_meta_data)
    assert out.shape == (20, 3, n_rows)
    # the average neighbour count only enters the sum over neighbours
    _eqv_assert_close(
        out, _eqv_equiv_oracle(ins, data, data["I"], 5, sum_neighbors=False)
    )


def test_spbf_equiv_ind_segment_sum_accepts_a_ragged_neighbour_list():
    """Atoms with different numbers of neighbours (no uniform layout): only the segment sum can handle this."""
    ins, _ = _eqv_equiv_ind(dense_nbr=False)
    data = _eqv_equiv_data(ins)
    keep = np.ones(20, dtype=bool)
    keep[[1, 2, 9, 14]] = False  # atoms 0, 2 and 3 lose neighbours, the others do not
    for key in ("R", "Y", constants.BOND_IND_I, constants.BOND_IND_J):
        data[key] = data[key][keep]
    counts = np.bincount(data[constants.BOND_IND_I], minlength=5)
    assert len(set(counts)) > 1
    _eqv_assert_close(ins.frwrd(data), _eqv_equiv_oracle(ins, data, data["I"], 5))


def test_spbf_equiv_ind_local_atoms_set_the_number_of_output_rows():
    ins, _ = _eqv_equiv_ind()
    data = _eqv_equiv_data(ins)
    keep = data[constants.BOND_IND_I] < 3
    for key in ("R", "Y", constants.BOND_IND_I, constants.BOND_IND_J):
        data[key] = data[key][keep]
    data[constants.ATOMIC_MU_I_LOCAL] = _EQV_TYPES[:3]
    out = _eqv_np(ins.frwrd(data, local=True))
    assert out.shape[0] == 3
    _eqv_assert_close(out, _eqv_equiv_oracle(ins, data, data["I"], 3))


@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
@pytest.mark.parametrize("per_atom", [False, True], ids=["per-element", "per-atom"])
def test_spbf_equiv_ind_chemical_embedding_adds_to_the_scalar_channel_of_the_indicator(
    per_atom, lm_first
):
    """The projected embedding is added to the ``L = 0, m = 0, p = +`` column of the indicator, per bond."""
    ins, z = _eqv_equiv_ind(chem=True, lm_first=lm_first)
    if per_atom:
        # No shipped ScalarChemicalEmbedding sets ``is_per_atom``: the flag is set on the instance read by the
        # SPBF (the neighbouring instruction), the unit under test is untouched.
        ins.chem_emb_is_per_atom = True
    data = _eqv_equiv_data(ins, z)
    z_rows = z.w.numpy()
    if per_atom:
        z_rows = np.random.default_rng(9).normal(size=(5, _EQV_EMB))
        data["Z"] = z_rows
    proj = z_rows @ ins.chem_linear.w.numpy() / np.sqrt(_EQV_EMB)
    feed = dict(data)
    if lm_first:
        feed["I"] = np.transpose(data["I"], (2, 0, 1))
    out = _eqv_np(ins.frwrd(feed))
    key = constants.BOND_IND_J if per_atom else constants.BOND_MU_J
    expected = _eqv_equiv_oracle(ins, data, data["I"], 5, zproj=proj, z_key=key)
    if lm_first:
        expected = np.transpose(expected, (2, 0, 1))
    _eqv_assert_close(out, expected)
    assert ins.chem_l0_idx == 0
    assert ins.n_lm_indicator == 9
    assert ins.chem_linear.w.shape.as_list() == [_EQV_EMB, _EQV_N_RAD]
    assert ins.chem_l0_mask.numpy().tolist() == [1.0] + [0.0] * 8
    assert constants.BOND_MU_J in ins.input_tensor_spec


def test_spbf_equiv_ind_without_chemical_embedding_does_not_need_the_bond_type():
    ins, _ = _eqv_equiv_ind()
    assert constants.BOND_MU_J not in ins.input_tensor_spec
    assert not hasattr(ins, "chem_linear")
    assert not ins.trainable_variables


def test_spbf_equiv_ind_refuses_an_indicator_without_a_unique_scalar_channel():
    y1 = _eqv_harmonic(1, "Y1", n_out=3)
    no_scalar = FunctionReduceN(
        instructions=[y1], name="P", ls_max=1, n_out=3, allowed_l_p=[[1, -1]]
    )
    with pytest.raises(AssertionError, match="found 0"):
        SingleParticleBasisFunctionEquivariantInd(
            name="YI",
            angular=_eqv_harmonic(1),
            indicator=no_scalar,
            radial=_eqv_radial(3, 1),
            lmax=1,
            Lmax=1,
            chemical_embedding=_eqv_embedding(),
        )
    y2 = _eqv_harmonic(2, "Y2", n_out=3)
    squares = ProductFunction(left=y2, right=y2, name="S", lmax=2, Lmax=0)
    with pytest.raises(AssertionError, match="found 3"):
        SingleParticleBasisFunctionEquivariantInd(
            name="YI",
            angular=_eqv_harmonic(1),
            indicator=squares,
            radial=_eqv_radial(3, 1),
            lmax=1,
            Lmax=1,
            chemical_embedding=_eqv_embedding(),
        )


def test_spbf_equiv_ind_refuses_a_radial_function_of_another_lmax():
    # angular degree above lmax: the radial function must be prepared for lmax
    with pytest.raises(AssertionError, match=r"prepared for lmax=1, while .* lmax=2"):
        SingleParticleBasisFunctionEquivariantInd(
            name="YI",
            angular=_eqv_harmonic(3),
            indicator=_eqv_harmonic(2, "I", n_out=3),
            radial=_eqv_radial(3, 1),
            lmax=2,
            Lmax=2,
        )
    # angular degree equal to lmax: radial and angular must match
    with pytest.raises(
        AssertionError, match="Radial part and angular part lmax do not match"
    ):
        SingleParticleBasisFunctionEquivariantInd(
            name="YI",
            angular=_eqv_harmonic(2),
            indicator=_eqv_harmonic(2, "I", n_out=3),
            radial=_eqv_radial(3, 1),
            lmax=2,
            Lmax=2,
        )


def test_spbf_equiv_ind_angular_degrees_above_lmax_are_dropped():
    ins, _ = _eqv_equiv_ind(lmax=1, big_lmax=2, ang_lmax=3)
    assert ins.slice_angular == 4
    data = _eqv_equiv_data(ins)
    assert data["Y"].shape[1] == 16 and data["R"].shape[2] == 4
    out = _eqv_np(ins.frwrd(data))
    # ``lmax`` limits the degree of both coupled functions, the harmonic and the indicator
    assert _eqv_triples(ins.coupling_meta_data) == _eqv_natural_triples(1, 1, 2)
    _eqv_assert_close(out, _eqv_equiv_oracle(ins, data, data["I"], 5))
    # no slicing without a higher angular degree
    plain, _ = _eqv_equiv_ind(lmax=2)
    assert plain.slice_angular is None


def test_spbf_equiv_ind_restricts_the_table_with_its_selection_options():
    base = _eqv_natural_triples(2, 2, 2)
    limited, _ = _eqv_equiv_ind(l_max_ind=1)  # degree of the indicator
    assert _eqv_triples(limited.coupling_meta_data) == {t for t in base if t[1] <= 1}
    summed, _ = _eqv_equiv_ind(
        max_sum_l=3
    )  # sum of the degrees of the two coupled functions
    assert _eqv_triples(summed.coupling_meta_data) == {
        t for t in base if t[0] + t[1] <= 3
    }
    dropped, _ = _eqv_equiv_ind(history_drop_list=[["(1-,1-)", 0], ["(2+,1-)", 1]])
    assert _eqv_triples(dropped.coupling_meta_data) == base - {(1, 1, 0), (2, 1, 1)}
    kept, _ = _eqv_equiv_ind(keep_parity=[[0, 1], [2, 1]])
    assert set(
        zip(kept.coupling_meta_data["l"], kept.coupling_meta_data["parity"])
    ) == {(0, 1), (2, 1)}


def test_spbf_equiv_ind_unnatural_parity_channels_follow_the_same_coupling():
    """``keep_parity`` may ask for ``(L, p)`` with ``p != (-1)^L`` (pseudo-tensors): the oracle covers them."""
    ins, _ = _eqv_equiv_ind(keep_parity=[[1, 1], [2, -1], [0, 1], [1, -1]])
    meta = ins.coupling_meta_data
    assert ((meta["l"] == 1) & (meta["parity"] == 1)).any()
    data = _eqv_equiv_data(ins)
    _eqv_assert_close(ins.frwrd(data), _eqv_equiv_oracle(ins, data, data["I"], 5))


def _eqv_sphere_quadrature(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact quadrature on the unit sphere for polynomials of degree up to ``2 lmax + 2``: points, weights (sum 1)."""
    x, w = np.polynomial.legendre.leggauss(lmax + 8)
    phi = np.linspace(0, 2 * np.pi, 2 * lmax + 12, endpoint=False)
    ct, ph = np.meshgrid(x, phi, indexing="ij")
    st = np.sqrt(1 - ct**2)
    vecs = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=-1).reshape(-1, 3)
    return vecs, np.repeat(w, len(phi)) / (2.0 * len(phi))


def _eqv_unit_norms(
    meta, left_meta, right_meta, lmax: int
) -> dict[tuple[int, int, int], float]:
    """``1 / sqrt(mean over the sphere and over M of (C Y_l1 Y_l2)^2)`` for every ``(l1, l2, L)`` of ``meta``.

    The harmonics are those of one direction, so the coupled value is a harmonic of ``L`` times a constant;
    this is the factor that makes each block of ``normalize=True`` have unit mean square.
    """
    vecs, weights = _eqv_sphere_quadrature(2 * lmax)
    y = _eqv_real_harmonics(vecs, lmax)
    raw = _eqv_couple(meta, left_meta, right_meta, y, y)
    out = {}
    for l1, l2, big_l in _eqv_triples(meta):
        rows = [
            r
            for r, (h, bl) in enumerate(zip(meta["hist"], meta["l"]))
            if bl == big_l
            and _eqv_split_hist(h)[0][1] == l1
            and _eqv_split_hist(h)[1][1] == l2
        ]
        out[(l1, l2, big_l)] = 1.0 / np.sqrt(
            np.mean([weights @ raw[:, r] ** 2 for r in rows])
        )
    return out


def test_spbf_equiv_ind_normalize_rescales_each_block_to_unit_mean_square():
    """``normalize=True`` scales every ``(l1, l2, L)`` block by the inverse rms of the coupled harmonics."""
    ins, _ = _eqv_equiv_ind(lmax=3, big_lmax=3, ind_lmax=3, normalize=True)
    plain, _ = _eqv_equiv_ind(lmax=3, big_lmax=3, ind_lmax=3)
    meta = ins.coupling_meta_data
    norms = _eqv_unit_norms(
        meta, ins.angular.coupling_meta_data, ins.indicator.coupling_meta_data, 3
    )
    assert norms[(1, 1, 0)] == pytest.approx(1 / np.sqrt(3.0))
    data = _eqv_equiv_data(ins)
    ry = data["R"] * data["Y"][:, None, :]
    bond_i = data["I"][data[constants.BOND_IND_J]]
    expected = np.zeros((5, 3, len(meta)))
    per_bond = _eqv_couple(
        meta,
        ins.angular.coupling_meta_data,
        ins.indicator.coupling_meta_data,
        ry,
        bond_i,
        norms,
    )
    for b, i in enumerate(data[constants.BOND_IND_I]):
        expected[i] += per_bond[b]
    _eqv_assert_close(ins.frwrd(data), expected, COVARIANCE_F64)
    # the un-normalised instruction differs by exactly these factors, block by block
    ratio = _eqv_np(ins.frwrd(data)) / _eqv_np(plain.frwrd(data))
    for (l1, l2, big_l), f in norms.items():
        rows = [
            r
            for r, (h, bl) in enumerate(zip(meta["hist"], meta["l"]))
            if bl == big_l
            and _eqv_split_hist(h)[0][1] == l1
            and _eqv_split_hist(h)[1][1] == l2
        ]
        _eqv_assert_close(
            ratio[:, :, rows], np.full((5, 3, len(rows)), f), COVARIANCE_F64
        )


def _eqv_equivariant_features(pos, rot=None, n=_EQV_N_RAD, lmax=_EQV_LMAX, seed=17):
    """Per-atom features that rotate like harmonics: ``Y(u_{a,k})`` of fixed random unit vectors ``u``.

    With ``rot`` the vectors are rotated first, so the features are those of the rotated system.
    """
    u = _eqv_unit(np.random.default_rng(seed).normal(size=(len(pos) * n, 3)))
    if rot is not None:
        u = rot.apply(u)
    return _eqv_real_harmonics(u, lmax).reshape(len(pos), n, -1)


def _eqv_radial_of_length(vec: np.ndarray, n_rad: int, lmax: int) -> np.ndarray:
    """A rotation-invariant radial tensor ``R[b, n, lm] = cos((n + 1) |r_b|) / (1 + |r_b|)`` (same for every lm)."""
    d = np.linalg.norm(vec, axis=1)
    r = np.cos(np.outer(d, np.arange(1, n_rad + 1))) / (1.0 + d)[:, None]
    return np.repeat(r[:, :, None], (lmax + 1) ** 2, axis=2)


def _eqv_equiv_run(ins, pos, rot=None, order=None) -> np.ndarray:
    pos_r = pos if rot is None else rot.apply(pos)
    data = _eqv_bonds(pos_r, order=order)
    data["Y"] = _eqv_real_harmonics(
        _eqv_unit(data[constants.BOND_VECTOR]), ins.angular.lmax
    )
    data["R"] = _eqv_radial_of_length(
        data[constants.BOND_VECTOR], ins.radial.n_rad_max, ins.radial.lmax
    )
    data["I"] = _eqv_equivariant_features(pos, rot, ins.radial.n_rad_max, 2)
    return _eqv_np(ins.frwrd(data))


@pytest.mark.parametrize("dense", [False, True], ids=["segment-sum", "dense"])
def test_spbf_equiv_ind_is_equivariant_and_invariant_under_translation_and_permutations(
    dense,
):
    ins, _ = _eqv_equiv_ind(
        keep_parity=[[0, 1], [1, -1], [2, 1], [1, 1]], dense_nbr=dense
    )
    pos = _eqv_positions()
    base = _eqv_equiv_run(ins, pos)
    assert np.abs(base).max() > 1e-3

    # the bond vectors are differences of positions: a translation of every position cancels exactly
    _eqv_assert_close(
        _eqv_equiv_run(ins, pos + np.array([4.0, -3.0, 1.0])), base, COVARIANCE_F64
    )

    # neighbour order: the sum over the bonds of an atom does not depend on it (segment sum only: the
    # dense layout requires the bonds sorted by centre atom, a permutation within each atom is allowed)
    order = np.arange(20).reshape(5, 4)
    order = np.stack([
        np.random.default_rng(i).permutation(row) for i, row in enumerate(order)
    ]).ravel()
    _eqv_assert_close(_eqv_equiv_run(ins, pos, order=order), base, COVARIANCE_F64)

    # rotation: every (L, parity, history) block rotates with D^L
    rot = _eqv_rotation()
    _eqv_assert_equivariant(
        ins.coupling_meta_data, _eqv_equiv_run(ins, pos, rot), base, rot
    )


def test_spbf_equiv_ind_atom_relabelling_permutes_the_rows():
    ins, _ = _eqv_equiv_ind()
    pos = _eqv_positions()
    base = _eqv_equiv_run(ins, pos)
    perm = np.array([2, 4, 0, 3, 1])
    data = _eqv_bonds(pos[perm], _EQV_TYPES[perm])
    data["Y"] = _eqv_real_harmonics(_eqv_unit(data[constants.BOND_VECTOR]), 2)
    data["R"] = _eqv_radial_of_length(data[constants.BOND_VECTOR], 3, 2)
    data["I"] = _eqv_equivariant_features(pos, None)[perm]
    _eqv_assert_close(ins.frwrd(data), base[perm], COVARIANCE_F64)


@pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
def test_spbf_equiv_ind_output_dtype_follows_the_data_not_the_parameters(data_dtype):
    """Float32 parameters with float64 data give float64: the tensors of the bonds set the precision."""
    ins, z = _eqv_equiv_ind(chem=True, build_dtype=tf.float32, avg_n_neigh=2.0)
    data64 = _eqv_equiv_data(ins, z)
    data = {
        k: (v.astype(data_dtype) if np.asarray(v).dtype == np.float64 else v)
        for k, v in data64.items()
    }
    out = ins.frwrd(data)
    assert out.dtype == tf.as_dtype(data_dtype)
    assert ins.cg.dtype == tf.float32
    assert ins.inv_avg_n_neigh.dtype == tf.float32
    assert ins.chem_linear.w.dtype == tf.float32
    proj = z.w.numpy() @ ins.chem_linear.w.numpy() / np.sqrt(_EQV_EMB)
    ref = _eqv_equiv_oracle(ins, data64, data64["I"], 5, inv_avg=0.5, zproj=proj)
    np.testing.assert_allclose(out.numpy(), ref, **FLOAT32_NETWORK._asdict())


def test_spbf_equiv_ind_building_twice_keeps_the_constants():
    ins, _ = _eqv_equiv_ind(chem=True)
    cg, w = ins.cg, ins.chem_linear.w
    before = w.numpy().copy()
    ins.build(tf.float32)
    assert ins.cg is cg
    assert cg.dtype == tf.float64
    assert ins.chem_linear.w is w
    np.testing.assert_array_equal(w.numpy(), before)


def test_spbf_equiv_ind_call_adds_the_output_and_never_edits_the_inputs():
    ins, _ = _eqv_equiv_ind()
    data = _eqv_equiv_data(ins)
    snapshot = {k: np.array(v, copy=True) for k, v in data.items()}
    out = ins(data)
    assert set(out) == set(snapshot) | {"YI"}
    for key, value in snapshot.items():
        np.testing.assert_array_equal(np.asarray(data[key]), value)


def test_spbf_equiv_ind_coupling_tables_are_consistent_with_the_flat_indices():
    ins, _ = _eqv_equiv_ind()
    meta = ins.coupling_meta_data
    n_cg = sum(len(c) for c in meta["cg_list"])
    assert ins.lr_inds.shape.as_list() == [n_cg, 2]
    assert ins.m_sum_ind.shape.as_list() == [n_cg]
    assert int(ins.nfunc) == len(meta)
    assert ins.cg.shape == (n_cg, 1, 1)
    assert ins.m_sum_ind.numpy().max() == len(meta) - 1


# ================================================================================================
# ProductFunction
# ================================================================================================

_EQV_N_ATOMS = 5


def _eqv_features(
    ins, n_out: int | None = None, seed: int = 31, n_atoms: int = _EQV_N_ATOMS
):
    """Random features ``[atoms, n, columns of the table of ins]``."""
    n = ins.n_out if n_out is None else n_out
    return np.random.default_rng(seed).normal(
        size=(n_atoms, n, len(ins.coupling_meta_data))
    )


def _eqv_product(
    lmax: int = 3,
    big_lmax: int = 2,
    n_out: int = 3,
    equal: bool = True,
    build_dtype=tf.float64,
    **kwargs,
):
    """``Y x Y`` (or ``Ya x Yb`` when ``equal`` is False) as a built product."""
    left = _eqv_harmonic(lmax, "Ya", n_out=n_out)
    right = left if equal else _eqv_harmonic(lmax, "Yb", n_out=n_out)
    ins = ProductFunction(
        left=left, right=right, name="AA", lmax=lmax, Lmax=big_lmax, **kwargs
    )
    ins.build(build_dtype)
    return ins


def _eqv_product_oracle(ins, data: dict, lm_first: bool = False) -> np.ndarray:
    a, b = data[ins.left.name], data[ins.right.name]  # standard layout [atoms, n, lm]
    out = _eqv_couple(
        ins.coupling_meta_data,
        ins.left.coupling_meta_data,
        ins.right.coupling_meta_data,
        a,
        b,
    )
    return (
        np.transpose(out, (2, 0, 1)) if lm_first else out
    )  # lm_first output [rows, atoms, n]


def test_product_function_table_of_harmonics_is_the_hand_derived_triangle_set():
    ins = _eqv_product(lmax=3, big_lmax=3)
    expected = {t for t in _eqv_natural_triples(3, 3, 3) if t[0] >= t[1]}
    assert _eqv_triples(ins.coupling_meta_data) == expected
    meta = ins.coupling_meta_data
    assert len(meta) == sum(2 * big_l + 1 for _, _, big_l in expected)
    assert ins.is_left_right_equal is True
    assert ins.coupling_origin == ["Ya", "Ya"]
    assert ins.lmax == 3  # the output degree
    assert ins.n_out == 3


def test_product_function_of_distinct_functions_keeps_both_orders():
    ins = _eqv_product(lmax=2, big_lmax=2, equal=False)
    assert ins.is_left_right_equal is False
    assert _eqv_triples(ins.coupling_meta_data) == _eqv_natural_triples(2, 2, 2)
    same = _eqv_product(lmax=2, big_lmax=2, equal=True, is_left_right_equal=False)
    assert (
        same.is_left_right_equal is False
    )  # explicit flag beats the comparison of the names
    assert _eqv_triples(same.coupling_meta_data) == _eqv_natural_triples(2, 2, 2)


def test_product_function_of_a_function_with_itself_has_no_antisymmetric_channels():
    """``(l, l) -> L`` odd is antisymmetric under exchange and vanishes for equal factors: no such row exists."""
    ins = _eqv_product(
        lmax=2,
        big_lmax=3,
        keep_parity=[[0, 1], [1, 1], [1, -1], [2, 1], [3, -1], [3, 1]],
    )
    meta = ins.coupling_meta_data
    for l1, l2, big_l in _eqv_triples(meta):
        assert not (l1 == l2 and big_l % 2 == 1)
    unequal = _eqv_product(
        lmax=2,
        big_lmax=3,
        equal=False,
        keep_parity=[[0, 1], [1, 1], [1, -1], [2, 1], [3, -1], [3, 1]],
    )
    assert (1, 1, 1) in _eqv_triples(unequal.coupling_meta_data)


@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
@pytest.mark.parametrize("equal", [True, False], ids=["equal", "distinct"])
def test_product_function_equals_the_coupling_oracle(equal, lm_first):
    ins = _eqv_product(equal=equal, lm_first=lm_first)
    data = {"Ya": _eqv_features(ins.left, seed=1)}
    if not equal:
        data["Yb"] = _eqv_features(ins.right, seed=2)
    expected = _eqv_product_oracle(ins, data, lm_first)
    feed = dict(data)
    if lm_first:
        feed = {k: np.transpose(v, (2, 0, 1)) for k, v in data.items()}
    out = _eqv_np(ins.frwrd(feed))
    n_rows = len(ins.coupling_meta_data)
    assert out.shape == ((n_rows, 5, 3) if lm_first else (5, 3, n_rows))
    _eqv_assert_close(out, expected)


def test_product_function_of_a_product_follows_the_nested_histories():
    """``AAA = AA x Y``: the oracle follows the history strings of the first product down to the harmonics."""
    y = _eqv_harmonic(2, "Y", n_out=3)
    aa = ProductFunction(left=y, right=y, name="AA", lmax=2, Lmax=2)
    aaa = ProductFunction(left=aa, right=y, name="AAA", lmax=2, Lmax=2)
    for ins in (aa, aaa):
        ins.build(tf.float64)
    data = {"Y": _eqv_features(y, seed=3)}
    data["AA"] = _eqv_np(aa.frwrd(data))
    _eqv_assert_close(data["AA"], _eqv_product_oracle(aa, data))
    out = _eqv_np(aaa.frwrd(data))
    assert any("(" in h[1:] for h in aaa.coupling_meta_data["hist"])  # nested histories
    _eqv_assert_close(out, _eqv_product_oracle(aaa, data))


def test_product_function_is_symmetric_under_exchange_up_to_the_clebsch_gordan_sign():
    """``C^L_{l2 l1}[M, m2, m1] = (-1)^(l1 + l2 - L) C^L_{l1 l2}[M, m1, m2]``: swap the factors, flip the odd blocks."""
    ab = _eqv_product(
        lmax=2,
        big_lmax=3,
        equal=False,
        keep_parity=[[0, 1], [1, 1], [1, -1], [2, 1], [2, -1], [3, 1], [3, -1]],
    )
    left = ab.left
    right = ab.right
    ba = ProductFunction(
        left=right,
        right=left,
        name="BB",
        lmax=2,
        Lmax=3,
        keep_parity=[[0, 1], [1, 1], [1, -1], [2, 1], [2, -1], [3, 1], [3, -1]],
    )
    ba.build(tf.float64)
    a, b = _eqv_features(left, seed=4), _eqv_features(right, seed=5)
    out_ab = _eqv_np(ab.frwrd({"Ya": a, "Yb": b}))
    out_ba = _eqv_np(ba.frwrd({"Ya": a, "Yb": b}))
    meta_ab, meta_ba = ab.coupling_meta_data, ba.coupling_meta_data
    n_checked = 0
    for r, (h, big_l, mm) in enumerate(
        zip(meta_ab["hist"], meta_ab["l"], meta_ab["m"])
    ):
        (_, l1), (_, l2) = _eqv_split_hist(h)
        swapped = f"({l2},{l1})"
        rr = [
            i
            for i, (h2, l2_, m2_) in enumerate(
                zip(meta_ba["hist"], meta_ba["l"], meta_ba["m"])
            )
            if h2 == swapped and l2_ == big_l and m2_ == mm
        ]
        assert len(rr) == 1
        _eqv_assert_close(
            out_ba[..., rr[0]], (-1) ** (l1 + l2 - big_l) * out_ab[..., r]
        )
        n_checked += 1
    assert n_checked == len(meta_ab) == len(meta_ba)


def test_product_function_normalize_rescales_each_block_to_unit_mean_square():
    ins = _eqv_product(lmax=3, big_lmax=3, normalize=True)
    plain = _eqv_product(lmax=3, big_lmax=3)
    meta = ins.coupling_meta_data
    norms = _eqv_unit_norms(
        meta, ins.left.coupling_meta_data, ins.right.coupling_meta_data, 3
    )
    assert norms[(1, 1, 0)] == pytest.approx(1 / np.sqrt(3.0))
    data = {"Ya": _eqv_features(ins.left, seed=6)}
    expected = _eqv_couple(
        meta,
        ins.left.coupling_meta_data,
        ins.right.coupling_meta_data,
        data["Ya"],
        data["Ya"],
        norms,
    )
    _eqv_assert_close(ins.frwrd(data), expected, COVARIANCE_F64)
    assert ins.normalize and not plain.normalize


def test_product_function_selection_options_restrict_the_table_as_documented():
    base = {t for t in _eqv_natural_triples(3, 3, 3) if t[0] >= t[1]}
    cases = {
        "lmax": (dict(lmax=2), {t for t in base if max(t[:2]) <= 2}),
        "lmax_left": (dict(lmax_left=1), {t for t in base if t[0] <= 1}),
        "lmax_right": (dict(lmax_right=1), {t for t in base if t[1] <= 1}),
        "max_sum_l": (dict(max_sum_l=3), {t for t in base if t[0] + t[1] <= 3}),
        "Lmax": (dict(big_lmax=1), {t for t in base if t[2] <= 1}),
        "keep_parity": (
            dict(keep_parity=[[0, 1], [2, 1]]),
            {t for t in base if t[2] in (0, 2)},
        ),
    }
    for name, (kw, expected) in cases.items():
        kw = dict(kw)
        big_l = kw.pop("big_lmax", 3)
        lmax = kw.pop("lmax", 3)
        ins = _eqv_product(lmax=lmax, big_lmax=big_l, **kw)
        assert _eqv_triples(ins.coupling_meta_data) == expected, name


def _eqv_hist_degree(hist: str) -> int:
    """Largest degree written in a history string (0 for the empty history)."""
    digits = "".join(c if c.isdigit() else " " for c in hist).split()
    return max(map(int, digits), default=0)


def test_product_function_history_options_act_on_the_histories_of_the_factors():
    y = _eqv_harmonic(2, "Y", n_out=3)
    aa = ProductFunction(left=y, right=y, name="AA", lmax=2, Lmax=2)
    full = ProductFunction(left=aa, right=y, name="F", lmax=2, Lmax=2)
    assert full.lmax_hist is None
    assert (
        max(
            _eqv_hist_degree(_eqv_split_hist(h)[0][0])
            for h in full.coupling_meta_data["hist"]
        )
        == 2
    )

    def kept(**kw):
        return ProductFunction(
            left=aa, right=y, name="K", lmax=2, Lmax=2, **kw
        ).coupling_meta_data

    for kw in [dict(lmax_hist=1), dict(lmax_hist_left=1)]:
        meta = kept(**kw)
        degrees = [_eqv_hist_degree(_eqv_split_hist(h)[0][0]) for h in meta["hist"]]
        assert max(degrees) <= 1, kw
        assert len(meta) < len(full.coupling_meta_data)
        assert set(meta["hist"]) < set(full.coupling_meta_data["hist"])
    # the right factor is a harmonic: its history is empty, nothing is cut
    assert len(kept(lmax_hist_right=1)) == len(full.coupling_meta_data)
    assert len(kept(lmax_hist_right=0)) == len(full.coupling_meta_data)

    victim = full.coupling_meta_data.iloc[0]
    drop = [[victim["hist"], int(victim["l"])]]
    dropped = kept(history_drop_list=drop)
    n_victim = (
        (full.coupling_meta_data["hist"] == victim["hist"])
        & (full.coupling_meta_data["l"] == victim["l"])
    ).sum()
    assert n_victim >= 1
    assert len(dropped) == len(full.coupling_meta_data) - n_victim


def test_product_function_refuses_factors_of_different_width():
    with pytest.raises(AssertionError, match="shapes of left and right do not match"):
        ProductFunction(
            left=_eqv_harmonic(1, "Ya", n_out=3),
            right=_eqv_harmonic(1, "Yb", n_out=4),
            name="AA",
            lmax=1,
            Lmax=1,
        )


def test_product_function_refuses_a_selection_that_leaves_no_channel():
    y = _eqv_harmonic(1, "Y", n_out=3)
    with pytest.raises(
        AssertionError, match="No coupling channels found between Y and Y"
    ):
        ProductFunction(left=y, right=y, name="AA", lmax=1, Lmax=1, keep_parity=[])


def test_product_function_float32_constants_and_output_dtype():
    ins = _eqv_product(build_dtype=tf.float32)
    assert ins.cg.dtype == tf.float32
    data64 = {"Ya": _eqv_features(ins.left, seed=7)}
    out32 = ins.frwrd({"Ya": data64["Ya"].astype(np.float32)})
    assert out32.dtype == tf.float32
    np.testing.assert_allclose(
        out32.numpy(), _eqv_product_oracle(ins, data64), **FLOAT32_NETWORK._asdict()
    )


def test_product_function_cg_layout_depends_on_the_feature_layout():
    std = _eqv_product()
    first = _eqv_product(lm_first=True)
    n_cg = int(std.m_sum_ind.shape[0])
    assert std.cg.shape == (1, 1, n_cg)
    assert first.cg.shape == (n_cg, 1, 1)
    assert int(std.nfunc) == len(std.coupling_meta_data)
    assert std.left_ind.shape == std.right_ind.shape == std.m_sum_ind.shape


def test_product_function_building_twice_keeps_the_constants():
    ins = _eqv_product()
    cg = ins.cg
    ins.build(tf.float32)
    assert ins.cg is cg
    assert cg.dtype == tf.float64


def test_product_function_ignores_the_local_flag_and_never_edits_its_inputs():
    ins = _eqv_product()
    data = {"Ya": _eqv_features(ins.left, seed=8)}
    snapshot = data["Ya"].copy()
    plain = _eqv_np(ins.frwrd(data))
    _eqv_assert_close(
        ins.frwrd(data, local=True), plain
    )  # no atom axis bookkeeping in a product
    np.testing.assert_array_equal(data["Ya"], snapshot)
    out = ins(data)
    assert "AA" in out


def test_product_function_repr_names_the_origin_and_degree():
    ins = _eqv_product(equal=False)
    assert repr(ins) == "ComputeProductFunction(name=AA, origin=Ya x Yb, l_max=2)"


def test_product_function_is_equivariant_through_two_levels():
    """Rotate the generating vectors: every block of ``AA`` and ``AAA`` rotates with its Wigner matrix."""
    y = _eqv_harmonic(2, "Y", n_out=3)
    aa = ProductFunction(left=y, right=y, name="AA", lmax=2, Lmax=2)
    aaa = ProductFunction(left=aa, right=y, name="AAA", lmax=2, Lmax=2)
    for ins in (aa, aaa):
        ins.build(tf.float64)
    pos = _eqv_positions()
    rot = _eqv_rotation()

    def run(r):
        data = {"Y": _eqv_equivariant_features(pos, r)}
        data["AA"] = _eqv_np(aa.frwrd(data))
        return data["AA"], _eqv_np(aaa.frwrd(data))

    (a0, b0), (a1, b1) = run(None), run(rot)
    assert np.abs(b0).max() > 1e-3
    _eqv_assert_equivariant(aa.coupling_meta_data, a1, a0, rot)
    _eqv_assert_equivariant(aaa.coupling_meta_data, b1, b0, rot)


# ================================================================================================
# FunctionReduceN
# ================================================================================================

_EQV_ALLOWED = [[0, 1], [1, -1], [2, 1]]
_EQV_ALLOWED_BOTH = [[0, 1], [1, -1], [1, 1], [2, 1]]


def _eqv_reduce_inputs(n_in: int = 3, parities=None, n_aa: int = 5):
    """Harmonics ``Y`` (``n_in`` channels) and the product ``AA`` of other harmonics ``Yb`` (``n_aa`` channels).

    Both have degree 2; the widths differ so that a norm or a weight shape taken from the wrong input shows.
    """
    y = _eqv_harmonic(2, "Y", n_out=n_in)
    yb = _eqv_harmonic(2, "Yb", n_out=n_aa)
    aa = ProductFunction(
        left=yb,
        right=yb,
        name="AA",
        lmax=2,
        Lmax=2,
        keep_parity=parities or [[0, 1], [1, -1], [1, 1], [2, 1], [2, -1]],
    )
    return y, aa


def _eqv_reducer(
    allowed=None,
    ls_max=2,
    n_out: int = 4,
    n_types: int | None = None,
    build_dtype=tf.float64,
    seed: int | None = 41,
    **kwargs,
):
    """A built ``FunctionReduceN`` over ``(Y, AA)`` with seeded weights, plus its inputs."""
    y, aa = _eqv_reduce_inputs(kwargs.pop("n_in", 3))
    if n_types is not None:
        kwargs.update(is_central_atom_type_dependent=True, number_of_atom_types=n_types)
    red = FunctionReduceN(
        instructions=[y, aa],
        name="R",
        ls_max=ls_max,
        n_out=n_out,
        allowed_l_p=allowed or _EQV_ALLOWED,
        **kwargs,
    )
    red.build(build_dtype)
    if seed is not None:
        seed_trainable_variables(red, seed=seed)
    return red, y, aa


def _eqv_reduce_data(y, aa, n_atoms: int = _EQV_N_ATOMS, seed: int = 51) -> dict:
    return {
        "Y": _eqv_features(y, seed=seed, n_atoms=n_atoms),
        "AA": _eqv_features(aa, seed=seed + 1, n_atoms=n_atoms),
        constants.ATOMIC_MU_I: _EQV_TYPES[:n_atoms].copy(),
    }


def _eqv_reduce_oracle(
    red,
    data: dict,
    atom_types: np.ndarray,
    init: float = 0.0,
    delta: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """The reduced tensor ``[atoms, n_out, rows]`` by an explicit loop over the collected functions.

    Row ``(l, m, p)`` of the output collects, from every input, the functions ``(l, m, p)`` of the groups
    ``(parity, l, history)`` that pass ``l <= ls_max`` and ``[l, p]`` in ``allowed``; the weight of a group is
    the slice of the stored ``reducing_<name>`` indexed by the position of the group in the sorted list of
    groups of that input. ``delta`` adds a weight update (LoRA) per input.
    """
    meta = red.coupling_meta_data
    row_of = {
        (int(l), int(m), int(p)): r
        for r, (l, m, p) in enumerate(zip(meta["l"], meta["m"], meta["parity"]))
    }
    n_atoms = len(atom_types)
    out = np.full((n_atoms, red.n_out, 1 if red.only_invar else len(meta)), init)
    counts = np.zeros(len(meta))
    for instr, l_cut in zip(red.instructions, red.ls_max):
        imeta = instr.coupling_meta_data
        allowed = {tuple(lp) for lp in red.allowed_l_p}
        keys = sorted({
            (int(p), int(l), str(h))
            for l, p, h in zip(imeta["l"], imeta["parity"], imeta["hist"])
            if l <= l_cut and (int(l), int(p)) in allowed
        })
        w = getattr(red, f"reducing_{instr.name}").numpy()
        if delta is not None:
            w = w + delta[instr.name]
        n_in = instr.n_out
        norm = red.scale / n_in**0.5 if red.normalize else 1.0
        for f, (l, m, p, h) in enumerate(
            zip(imeta["l"], imeta["m"], imeta["parity"], imeta["hist"])
        ):
            key = (int(p), int(l), str(h))
            if key not in keys:
                continue
            col = 0 if red.only_invar else row_of[(int(l), int(m), int(p))]
            counts[row_of[(int(l), int(m), int(p))]] += 1
            g = keys.index(key)
            x = data[instr.name][:n_atoms, :, f]  # [atoms, n_in]
            if red.is_central_atom_type_dependent:
                out[:, :, col] += norm * np.einsum(
                    "akn,an->ak", w[atom_types][..., g], x
                )
            else:
                out[:, :, col] += norm * x @ w[..., g].T
    if red.out_norm:
        counts[counts == 0] = 1
        out = out * (1.0 / np.sqrt(counts))[None, None, :]
    return out


@pytest.mark.parametrize("target", ["zeros", "ones"])
@pytest.mark.parametrize("out_norm", [False, True], ids=["raw", "out-norm"])
@pytest.mark.parametrize("n_types", [None, 2], ids=["shared", "per-type"])
@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
def test_function_reduce_n_equals_the_explicit_sum_over_collected_functions(
    lm_first, n_types, out_norm, target
):
    """``out = (init + sum of weighted functions) * norm_map``, with ``init`` the ``init_target_value``."""
    red, y, aa = _eqv_reducer(
        n_types=n_types,
        out_norm=out_norm,
        lm_first=lm_first,
        init_target_value=target,
        allowed=_EQV_ALLOWED_BOTH,
    )
    data = _eqv_reduce_data(y, aa)
    feed = dict(data)
    if lm_first:
        feed["Y"], feed["AA"] = (np.transpose(data[k], (2, 0, 1)) for k in ("Y", "AA"))
    out = _eqv_np(red.frwrd(feed))
    expected = _eqv_reduce_oracle(
        red, data, _EQV_TYPES, init=1.0 if target == "ones" else 0.0
    )
    n_rows = len(red.coupling_meta_data)
    assert out.shape == ((n_rows, 5, 4) if lm_first else (5, 4, n_rows))
    assert n_rows == 1 + 3 + 3 + 5  # (0,+), (1,-), (1,+), (2,+)
    _eqv_assert_close(np.transpose(out, (1, 2, 0)) if lm_first else out, expected)


@pytest.mark.parametrize("n_types", [None, 2], ids=["shared", "per-type"])
@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
def test_function_reduce_n_of_invariants_sums_every_collected_scalar(lm_first, n_types):
    """With ``ls_max = 0`` the output has one column: the weighted sum of every collected ``l = 0`` function."""
    red, y, aa = _eqv_reducer(
        ls_max=0, allowed=[[0, 1]], lm_first=lm_first, n_types=n_types, out_norm=True
    )
    assert red.only_invar
    data = _eqv_reduce_data(y, aa)
    feed = dict(data)
    if lm_first:
        feed["Y"], feed["AA"] = (np.transpose(data[k], (2, 0, 1)) for k in ("Y", "AA"))
    out = _eqv_np(red.frwrd(feed))
    n_scalars = 1 + sum(
        1
        for l, p in zip(aa.coupling_meta_data["l"], aa.coupling_meta_data["parity"])
        if l == 0 and p == 1
    )
    assert n_scalars == 4  # Y: one; AA: (0,0), (1,1), (2,2)
    assert out.shape == ((1, 5, 4) if lm_first else (5, 4, 1))
    expected = _eqv_reduce_oracle(red, data, _EQV_TYPES)
    assert red.norm_map.shape.as_list() == [1, 1, 1]
    assert float(red.norm_map[0, 0, 0]) == pytest.approx(1 / np.sqrt(n_scalars))
    _eqv_assert_close(np.transpose(out, (1, 2, 0)) if lm_first else out, expected)


def test_function_reduce_n_ones_target_with_zero_weights_returns_the_unit_collection():
    red, y, aa = _eqv_reducer(
        init_target_value="ones", init_vars="zeros", out_norm=True, seed=None
    )
    out = _eqv_np(red.frwrd(_eqv_reduce_data(y, aa)))
    expected = np.broadcast_to(red.norm_map.numpy().reshape(1, 1, -1), out.shape)
    _eqv_assert_close(out, expected)


def test_function_reduce_n_is_linear_in_its_weights():
    red, y, aa = _eqv_reducer()
    data = _eqv_reduce_data(y, aa)
    base = _eqv_np(red.frwrd(data))
    for name in ("Y", "AA"):
        w = getattr(red, f"reducing_{name}")
        w.assign(2.5 * w)
    _eqv_assert_close(red.frwrd(data), 2.5 * base)
    # linear in the input features as well
    scaled = dict(data, Y=-3.0 * data["Y"], AA=-3.0 * data["AA"])
    _eqv_assert_close(red.frwrd(scaled), -7.5 * base)


def test_function_reduce_n_local_atoms_drop_the_ghost_atoms_of_the_features():
    """With ``local=True`` and per-type weights, the features of the atoms beyond the local ones are ignored."""
    red, y, aa = _eqv_reducer(n_types=2)
    data = _eqv_reduce_data(y, aa)
    local_types = _EQV_TYPES[:3]
    data[constants.ATOMIC_MU_I_LOCAL] = local_types
    out = _eqv_np(red.frwrd(data, local=True))
    assert out.shape[0] == 3
    _eqv_assert_close(out, _eqv_reduce_oracle(red, data, local_types))
    # lm_first layout slices the atom axis, which is axis 1 there
    first, y2, aa2 = _eqv_reducer(n_types=2, lm_first=True)
    feed = {k: np.transpose(data[k], (2, 0, 1)) for k in ("Y", "AA")}
    feed[constants.ATOMIC_MU_I_LOCAL] = local_types
    out2 = _eqv_np(first.frwrd(feed, local=True))
    assert out2.shape[1] == 3
    _eqv_assert_close(
        np.transpose(out2, (1, 2, 0)), _eqv_reduce_oracle(first, data, local_types)
    )


def test_function_reduce_n_float32_weights_give_float32_output():
    red, y, aa = _eqv_reducer(build_dtype=tf.float32, out_norm=True)
    data = _eqv_reduce_data(y, aa)
    out = red.frwrd(data)
    assert out.dtype == tf.float32
    assert getattr(red, "reducing_Y").dtype == tf.float32
    assert red.norm_map.dtype == tf.float32
    np.testing.assert_allclose(
        out.numpy(),
        _eqv_reduce_oracle(red, data, _EQV_TYPES),
        **FLOAT32_NETWORK._asdict(),
    )


def test_function_reduce_n_call_adds_the_output_and_never_edits_the_inputs():
    red, y, aa = _eqv_reducer()
    data = _eqv_reduce_data(y, aa)
    snapshot = {k: np.array(v, copy=True) for k, v in data.items()}
    out = red(data)
    assert set(out) == set(snapshot) | {"R"}
    for key, value in snapshot.items():
        np.testing.assert_array_equal(np.asarray(data[key]), value)


def test_function_reduce_n_table_lists_the_allowed_blocks_by_degree_parity_and_order():
    red, _, _ = _eqv_reducer(
        allowed=[[2, 1], [1, 1], [0, 1], [1, -1], [3, 1]], ls_max=[2, 2]
    )
    meta = red.coupling_meta_data
    # sorted by (l, parity, m); degree 3 is dropped because lmax = max(ls_max) = 2
    assert list(zip(meta["l"], meta["parity"]))[::1] == (
        [(0, 1)] + [(1, -1)] * 3 + [(1, 1)] * 3 + [(2, 1)] * 5
    )
    assert meta["m"].tolist() == [0, -1, 0, 1, -1, 0, 1, -2, -1, 0, 1, 2]
    assert red.lmax == 2
    assert red.ls_max == [2, 2]
    assert red.allowed_l_p == [[2, 1], [1, 1], [0, 1], [1, -1], [3, 1]]
    assert not red.only_invar
    assert red.n_instr == 2


def test_function_reduce_n_integer_ls_max_applies_to_every_input_and_lists_stay_per_input():
    y, aa = _eqv_reduce_inputs()
    scalar = FunctionReduceN(
        instructions=[y, aa], name="a", ls_max=[2, 0], n_out=2, allowed_l_p=_EQV_ALLOWED
    )
    assert scalar.ls_max == [2, 0]
    assert not scalar.only_invar
    assert (
        len(scalar.collector["AA"]["func_collect_ind"]) == 3
    )  # only the l = 0 histories (0,0), (1,1), (2,2)
    assert len(scalar.collector["Y"]["func_collect_ind"]) == 1 + 3 + 5
    as_int = FunctionReduceN(
        instructions=[y, aa], name="c", ls_max=2, n_out=2, allowed_l_p=_EQV_ALLOWED
    )
    assert as_int.ls_max == [2, 2]
    allowed_as_tuples = FunctionReduceN(
        instructions=[y, aa], name="b", ls_max=2, n_out=2, allowed_l_p=[(0, 1), (2, 1)]
    )
    assert allowed_as_tuples.allowed_l_p == [[0, 1], [2, 1]]
    assert allowed_as_tuples.plist is allowed_as_tuples.allowed_l_p


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(init_vars="orthogonal"), 'Unknown variable initialization "orthogonal"'),
        (dict(init_target_value="twos"), 'Unknown target initialization "twos"'),
        (dict(is_central_atom_type_dependent=True), None),
    ],
    ids=["init_vars", "init_target_value", "missing-number-of-types"],
)
def test_function_reduce_n_refuses_inconsistent_options(kwargs, message):
    y, aa = _eqv_reduce_inputs()
    with pytest.raises(AssertionError) as info:
        FunctionReduceN(
            instructions=[y, aa],
            name="R",
            ls_max=2,
            n_out=2,
            allowed_l_p=_EQV_ALLOWED,
            **kwargs,
        )
    if message:
        assert message in str(info.value)


def test_function_reduce_n_refuses_duplicated_inputs_and_missing_degrees():
    y, aa = _eqv_reduce_inputs()
    with pytest.raises(AssertionError, match="duplicate instruction names"):
        FunctionReduceN(
            instructions=[y, y], name="R", ls_max=2, n_out=2, allowed_l_p=_EQV_ALLOWED
        )
    with pytest.raises(
        AssertionError, match="provide lmax to collect for every instruction"
    ):
        FunctionReduceN(
            instructions=[y, aa],
            name="R",
            ls_max=[2],
            n_out=2,
            allowed_l_p=_EQV_ALLOWED,
        )


def test_function_reduce_n_second_build_keeps_the_weights_and_norm_map():
    red, _, _ = _eqv_reducer(out_norm=True)
    weights, norm_map = getattr(red, "reducing_AA"), red.norm_map
    before = weights.numpy().copy()
    red.build(tf.float32)
    assert getattr(red, "reducing_AA") is weights
    assert red.norm_map is norm_map
    np.testing.assert_array_equal(weights.numpy(), before)


def test_function_reduce_n_l2_loss_is_the_sum_of_squares_of_the_trainable_weights():
    red, _, _ = _eqv_reducer(n_types=2)
    expected = sum(
        float(np.sum(getattr(red, f"reducing_{n}").numpy() ** 2)) for n in ("Y", "AA")
    )
    assert float(red.compute_l2_regularization_loss()) == pytest.approx(
        expected, rel=FLOAT64_ARITHMETIC.rtol
    )
    assert expected > 1.0
    zero, _, _ = _eqv_reducer(init_vars="zeros", seed=None)
    assert float(zero.compute_l2_regularization_loss()) == 0.0


def test_function_reduce_n_selected_elements_keep_the_rows_of_those_types_in_that_order():
    red, y, aa = _eqv_reducer(n_types=3)
    new = red.prepare_variables_for_selected_elements([2, 0])
    assert set(new) == {"reducing_Y", "reducing_AA"}
    for name, var in new.items():
        old = getattr(red, name).numpy()
        assert var.shape.as_list() == [2, *old.shape[1:]]
        np.testing.assert_array_equal(var.numpy(), old[[2, 0]])
        assert var.trainable
        assert var is not getattr(red, name)
    shared, _, _ = _eqv_reducer()
    assert shared.prepare_variables_for_selected_elements([0]) is None


def test_function_reduce_n_new_element_map_updates_the_recorded_type_count():
    red, _, _ = _eqv_reducer(n_types=3)
    assert red._init_args["number_of_atom_types"] == 3
    red.upd_init_args_new_elements({"H": 0, "C": 1})
    assert red._init_args["number_of_atom_types"] == 2


@pytest.mark.parametrize(
    "config",
    [{"rank": 2, "alpha": 1.0}, {"mode": "additive"}],
    ids=["lora", "additive"],
)
@pytest.mark.parametrize("n_types", [None, 2], ids=["shared", "per-type"])
def test_function_reduce_n_lora_adds_the_update_to_each_weight(config, n_types):
    red, y, aa = _eqv_reducer(n_types=n_types)
    data = _eqv_reduce_data(y, aa)
    plain = _eqv_np(red.frwrd(data))
    base = {n: getattr(red, f"reducing_{n}").numpy().copy() for n in ("Y", "AA")}

    red.enable_lora_adaptation(config)
    assert red.lora and red._init_args["lora_config"] == config
    _eqv_assert_close(red.frwrd(data), plain)  # the update starts at zero
    rng = np.random.default_rng(5)
    delta = {}
    for n in ("Y", "AA"):
        tensors = getattr(red, f"reducing_{n}_lora_tensors")
        assert not getattr(red, f"reducing_{n}").trainable
        for t in tensors:
            t.assign(rng.normal(size=t.shape))
        if "mode" in config:
            delta[n] = tensors[0].numpy()
        else:
            letters = "abcd"[: len(tensors)]
            delta[n] = np.einsum(
                ",".join(f"{c}r" for c in letters) + "->" + letters,
                *[t.numpy() for t in tensors],
            )
    updated = _eqv_np(red.frwrd(data))
    assert np.abs(updated - plain).max() > 1e-3
    _eqv_assert_close(updated, _eqv_reduce_oracle(red, data, _EQV_TYPES, delta=delta))
    # the weight update enters linearly: the l2 loss only sees the LoRA tensors
    lora_l2 = sum(
        float(np.sum(t.numpy() ** 2))
        for n in ("Y", "AA")
        for t in getattr(red, f"reducing_{n}_lora_tensors")
    )
    assert float(red.compute_l2_regularization_loss()) == pytest.approx(
        lora_l2, rel=FLOAT64_ARITHMETIC.rtol
    )

    red.finalize_lora_update()
    assert not red.lora and "lora_config" not in red._init_args
    for n in ("Y", "AA"):
        w = getattr(red, f"reducing_{n}")
        assert w.trainable and not hasattr(red, f"reducing_{n}_lora_tensors")
        np.testing.assert_allclose(
            w.numpy(), base[n] + delta[n], **FLOAT64_ARITHMETIC._asdict()
        )
    _eqv_assert_close(red.frwrd(data), updated)


def test_function_reduce_n_built_with_a_lora_config_freezes_the_weights_at_once():
    red, y, aa = _eqv_reducer(lora_config={"rank": 2, "alpha": 1.0}, seed=None)
    assert red.lora
    assert not getattr(red, "reducing_Y").trainable
    assert len(getattr(red, "reducing_AA_lora_tensors")) == 3
    data = _eqv_reduce_data(y, aa)
    _eqv_assert_close(red.frwrd(data), _eqv_reduce_oracle(red, data, _EQV_TYPES))


def test_function_reduce_n_simplify_records_unused_histories_but_leaves_the_tables_as_they_are():
    """Pins current behaviour (reported as a finding): ``simplify=True`` stores the unused histories (plain
    strings) in ``history_drop_list`` of each product, but the coupling tables only drop ``(history, L)`` pairs,
    so no row disappears."""
    from tensorpotential.instructions.compute import CropProductFunction

    def chain():
        a = SingleParticleBasisFunctionScalarInd(
            name="A",
            radial=_eqv_radial(),
            angular=_eqv_harmonic(),
            indicator=_eqv_embedding(),
        )
        a.n_out = 3
        kw = dict(
            lmax=2, Lmax=2, n_crop=3, keep_parity=_EQV_ALLOWED, history_drop_list=[]
        )
        aa = CropProductFunction(left=a, right=a, name="AA", **kw)
        aaa = CropProductFunction(left=aa, right=a, name="AAA", **kw)
        return a, aa, aaa

    a, aa, aaa = chain()
    sizes = (len(aa.coupling_meta_data), len(aaa.coupling_meta_data))
    red = FunctionReduceN(
        instructions=[a, aa, aaa],
        name="E",
        ls_max=0,
        n_out=2,
        allowed_l_p=[[0, 1]],
        simplify=True,
    )
    assert red.simplify
    assert (len(aa.coupling_meta_data), len(aaa.coupling_meta_data)) == sizes
    unused = list(aaa.history_drop_list)
    assert unused and all(isinstance(h, str) for h in unused)
    assert "(1,0)" in unused  # a product with the scalar L=1: no invariant needs it
    assert list(aa.history_drop_list) == unused
    plain_a, plain_aa, _ = chain()
    plain = FunctionReduceN(
        instructions=[plain_a, plain_aa],
        name="E",
        ls_max=0,
        n_out=2,
        allowed_l_p=[[0, 1]],
    )
    assert not plain.simplify
    assert list(plain_aa.history_drop_list) == []


def test_function_reduce_n_simplify_cannot_rebuild_a_plain_product_function():
    """Pins current behaviour (reported as a finding): ``drop_unused`` calls ``init_coupling()`` without the
    arguments of ``ProductFunction.init_coupling`` (and ``list(None)`` for the default drop list), so
    ``simplify=True`` over a ``ProductFunction`` raises; the presets use ``CropProductFunction``."""
    a = SingleParticleBasisFunctionScalarInd(
        name="A",
        radial=_eqv_radial(),
        angular=_eqv_harmonic(),
        indicator=_eqv_embedding(),
    )
    a.n_out = 3
    for kwargs, error in [
        ({}, "NoneType"),
        ({"history_drop_list": []}, "missing 2 required"),
    ]:
        p = ProductFunction(left=a, right=a, name="P", lmax=2, Lmax=2, **kwargs)
        with pytest.raises(TypeError, match=error):
            FunctionReduceN(
                instructions=[a, p],
                name="E",
                ls_max=0,
                n_out=2,
                allowed_l_p=[[0, 1]],
                simplify=True,
            )


# ================================================================================================
# FCRight2Left
# ================================================================================================


def _eqv_fc(
    n_out: int | None = 4,
    n_types: int | None = 2,
    build_dtype=tf.float64,
    seed: int | None = 61,
    **kwargs,
):
    """``FCRight2Left(left=Y, right=AA)`` over the harmonics and their product, built, with seeded weights."""
    y, aa = _eqv_reduce_inputs(parities=_EQV_ALLOWED)
    fc = FCRight2Left(
        left=y, right=aa, name="FC", n_out=n_out, number_of_atom_types=n_types, **kwargs
    )
    fc.build(build_dtype)
    if seed is not None:
        seed_trainable_variables(fc, seed=seed)
    return fc, y, aa


def _eqv_groups(meta) -> dict[tuple[int, int, str], list[int]]:
    """``(l, parity, hist) -> rows`` in the sorted order of the keys."""
    groups: dict[tuple[int, int, str], list[int]] = {}
    for r, key in enumerate(zip(meta["l"], meta["parity"], meta["hist"])):
        groups.setdefault((int(key[0]), int(key[1]), str(key[2])), []).append(r)
    return dict(sorted(groups.items()))


def _eqv_fc_oracle(fc, data: dict, atom_types: np.ndarray) -> np.ndarray:
    """``FCRight2Left`` by explicit loops: every left group gets the weighted right groups of its ``(l, parity)``."""
    left_meta = fc.left.coupling_meta_data
    left_groups, right_groups = (
        _eqv_groups(left_meta),
        _eqv_groups(fc.right.coupling_meta_data),
    )
    left, right = data[fc.left.name], data[fc.right.name]  # [atoms, n, columns]
    n_atoms = left.shape[0]

    def weighted(w, x, depends: bool, norm: float) -> np.ndarray:
        if depends:
            return norm * np.einsum("akn,an->ak", w[atom_types], x)
        return norm * x @ w.T

    dep_left, dep_right = fc.is_central_atom_type_dependent
    out = np.zeros((n_atoms, fc.n_out, len(left_meta)))
    n_left = 1 / np.sqrt(fc.left.n_out) if fc.normalize else 1.0
    n_right = 1 / np.sqrt(fc.right.n_out) if fc.normalize else 1.0
    for g, rows in enumerate(left_groups.values()):
        for f in rows:
            if fc.left_coefs:
                w = fc.w_left.numpy()[..., g]
                out[:, :, f] = weighted(w, left[:, :, f], dep_left, n_left)
            else:
                out[:, :, f] = left[:, :, f]
    count = 0
    matches = np.ones(len(left_groups))
    for g, ((l, p, _), rows) in enumerate(left_groups.items()):
        for (lr, pr, _), rrows in right_groups.items():
            if (lr, pr) != (l, p):
                continue
            w = fc.w_right.numpy()[..., count]
            for f, fr in zip(rows, rrows):
                out[:, :, f] += weighted(w, right[:, :, fr], dep_right, n_right)
            count += 1
            matches[g] += 1
    if fc.norm_out:
        for g, rows in enumerate(left_groups.values()):
            out[:, :, rows] *= 1 / np.sqrt(matches[g])
    return out


def _eqv_fc_data(y, aa, n_atoms: int = _EQV_N_ATOMS, seed: int = 71) -> dict:
    return _eqv_reduce_data(y, aa, n_atoms, seed)


@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
@pytest.mark.parametrize("norm_out", [False, True], ids=["raw", "out-norm"])
@pytest.mark.parametrize(
    "dependent", [None, False, [True, False], [False, True], True], ids=str
)
@pytest.mark.parametrize("left_coefs", [True, False], ids=["left-coefs", "plain-left"])
def test_fc_right2left_equals_the_explicit_sum_of_matching_groups(
    left_coefs, dependent, norm_out, lm_first
):
    fc, y, aa = _eqv_fc(
        n_out=4 if left_coefs else 3,
        left_coefs=left_coefs,
        is_central_atom_type_dependent=dependent,
        norm_out=norm_out,
        lm_first=lm_first,
    )
    data = _eqv_fc_data(y, aa)
    feed = dict(data)
    if lm_first:
        feed["Y"], feed["AA"] = (np.transpose(data[k], (2, 0, 1)) for k in ("Y", "AA"))
    out = _eqv_np(fc.frwrd(feed))
    n_cols = len(y.coupling_meta_data)
    assert out.shape == ((n_cols, 5, fc.n_out) if lm_first else (5, fc.n_out, n_cols))
    expected = _eqv_fc_oracle(fc, data, _EQV_TYPES)
    _eqv_assert_close(np.transpose(out, (1, 2, 0)) if lm_first else out, expected)


def test_fc_right2left_tables_and_norm_follow_the_matching_of_degree_and_parity():
    """Left groups ``(0,+)``, ``(1,-)``, ``(2,+)`` meet 3, 2 and 3 histories of ``Y x Y``: ``1/sqrt(1 + n)``."""
    fc, y, aa = _eqv_fc(norm_out=True)
    assert fc.w_shape_left == 3
    assert fc.w_shape_right == 3 + 2 + 3
    assert fc.w_left.shape.as_list() == [4, 3, 3]  # [n_out, left.n_out, left groups]
    # [n_out, right.n_out, matched right groups]
    assert fc.w_right.shape.as_list() == [4, 5, 8]
    expected = np.array([0.5] + [1 / np.sqrt(3)] * 3 + [0.5] * 5)
    np.testing.assert_allclose(
        np.asarray(fc.norm_map), expected, rtol=FLOAT64_ARITHMETIC.rtol
    )
    assert fc.norm_out_factor.shape.as_list() == [9, 1, 1]
    np.testing.assert_allclose(
        fc.norm_out_factor.numpy().ravel(), expected, rtol=FLOAT64_ARITHMETIC.rtol
    )
    assert fc.w_tile_left.numpy().tolist() == [0] + [1] * 3 + [2] * 5
    assert fc.collect_to.shape == fc.collect_from.shape
    # one column per matched right group, repeated over its rows: 3 groups of 1 row, 2 of 3 rows, 3 of 5 rows
    assert (
        fc.w_tile_right.numpy().tolist()
        == [0, 1, 2] + [3] * 3 + [4] * 3 + [5] * 5 + [6] * 5 + [7] * 5
    )
    assert len(fc.collect_to) == 3 + 6 + 15
    assert (
        fc.coupling_meta_data["hist"].tolist() == y.coupling_meta_data["hist"].tolist()
    )
    assert fc.coupling_meta_data is not y.coupling_meta_data  # a copy


def test_fc_right2left_weights_take_the_scales_of_the_options():
    for normalize in (True, False):
        fc, y, aa = _eqv_fc(normalize=normalize, seed=None)
        left_norm = 1 / np.sqrt(y.n_out) if normalize else 1.0
        right_norm = 1 / np.sqrt(aa.n_out) if normalize else 1.0
        assert float(fc.norm_left) == pytest.approx(left_norm)
        assert float(fc.norm_right) == pytest.approx(right_norm)


def test_fc_right2left_zero_init_sets_only_the_left_weight_to_zero():
    # Pins current behaviour (reported as a finding): ``init_vars="zeros"`` zeroes ``w_left`` but draws
    # ``w_right`` from a normal distribution (the third branch of the right-hand initialiser is a copy of
    # the "random" one), so a freshly built layer with zero init is not the identity on the left input.
    tf.random.set_seed(5)
    fc, y, aa = _eqv_fc(n_out=60, init_vars="zeros", seed=None)
    assert not fc.w_left.numpy().any()
    w = fc.w_right.numpy()
    assert w.std() == pytest.approx(1.0, rel=sample_std_rtol(w.size))
    out = _eqv_np(fc.frwrd(_eqv_fc_data(y, aa)))
    assert np.abs(out).max() > 1e-3


def test_fc_right2left_random_and_uniform_initialisers_have_their_stated_distribution():
    tf.random.set_seed(6)
    rnd, y, aa = _eqv_fc(n_out=60, seed=None)
    for w in (rnd.w_left.numpy(), rnd.w_right.numpy()):
        assert w.std() == pytest.approx(1.0, rel=sample_std_rtol(w.size))
    unscaled, _, _ = _eqv_fc(n_out=60, normalize=False, seed=None)
    assert unscaled.w_left.numpy().std() == pytest.approx(
        1 / np.sqrt(y.n_out), rel=sample_std_rtol(unscaled.w_left.numpy().size)
    )
    assert unscaled.w_right.numpy().std() == pytest.approx(
        1 / np.sqrt(aa.n_out), rel=sample_std_rtol(unscaled.w_right.numpy().size)
    )
    uni, _, _ = _eqv_fc(n_out=60, init_vars="uniform", seed=None)
    for w in (uni.w_left.numpy(), uni.w_right.numpy()):
        assert np.abs(w).max() <= 1.0
        assert np.abs(w).max() > 0.95
        assert w.std() == pytest.approx(1 / np.sqrt(3.0), rel=sample_std_rtol(w.size))


def test_fc_right2left_without_left_coefficients_has_no_left_weight():
    fc, y, aa = _eqv_fc(n_out=3, left_coefs=False)
    assert not hasattr(fc, "w_left")
    assert [v.name for v in fc.trainable_variables] == ["FC/w_right_FC:0"]
    assert not hasattr(fc, "norm_left")


def test_fc_right2left_central_atom_flags_are_normalised_to_a_pair():
    default, _, _ = _eqv_fc()
    assert default.is_central_atom_type_dependent == [False, False]
    both, _, _ = _eqv_fc(is_central_atom_type_dependent=True)
    assert both.is_central_atom_type_dependent == [True, True]
    pair, _, _ = _eqv_fc(is_central_atom_type_dependent=[True, False])
    assert pair.is_central_atom_type_dependent == [True, False]
    assert pair.w_left.shape.as_list() == [2, 4, 3, 3]
    assert pair.w_right.shape.as_list() == [4, 5, 8]
    right_only, _, _ = _eqv_fc(is_central_atom_type_dependent=[False, True])
    assert right_only.w_left.shape.as_list() == [4, 3, 3]
    assert right_only.w_right.shape.as_list() == [2, 4, 5, 8]


def test_fc_right2left_refuses_inconsistent_options():
    y, aa = _eqv_reduce_inputs(parities=_EQV_ALLOWED)
    with pytest.raises(
        ValueError, match="Unexpected type for :is_central_atom_type_dependent:"
    ):
        FCRight2Left(left=y, right=aa, name="FC", is_central_atom_type_dependent="yes")
    with pytest.raises(AssertionError, match="number_of_atom_types cannot be None"):
        FCRight2Left(
            left=y, right=aa, name="FC", is_central_atom_type_dependent=[False, True]
        )
    with pytest.raises(AssertionError):
        FCRight2Left(left=y, right=aa, name="FC", left_coefs=False, n_out=7)
    with pytest.raises(AssertionError, match='Unknown variable initialization "ortho"'):
        FCRight2Left(left=y, right=aa, name="FC", init_vars="ortho")
    # the default output width is that of the left factor
    assert FCRight2Left(left=y, right=aa, name="FC").n_out == y.n_out


def test_fc_right2left_refuses_a_right_factor_with_no_common_block():
    # Pins current behaviour (reported as a finding): no common (l, parity) between the factors ends in a bare
    # numpy error ("need at least one array to concatenate") instead of a message naming the instructions.
    y, _ = _eqv_reduce_inputs(parities=_EQV_ALLOWED)
    other = _eqv_harmonic(2, "Yb", n_out=y.n_out)
    aa = ProductFunction(
        left=y, right=other, name="AA", lmax=2, Lmax=1, keep_parity=[[1, 1]]
    )
    with pytest.raises(ValueError, match="at least one array to concatenate"):
        FCRight2Left(left=y, right=aa, name="FC")


@pytest.mark.parametrize("left_coefs", [True, False], ids=["left-coefs", "plain-left"])
def test_fc_right2left_build_refuses_an_initialiser_set_after_construction(left_coefs):
    y, aa = _eqv_reduce_inputs(parities=_EQV_ALLOWED)
    fc = FCRight2Left(
        left=y,
        right=aa,
        name="FC",
        n_out=3 if not left_coefs else 4,
        left_coefs=left_coefs,
    )
    fc.init_vars = "ortho"  # the constructor asserts on this; build must still refuse
    with pytest.raises(NotImplementedError, match="ortho"):
        fc.build(tf.float64)


def test_fc_right2left_second_build_keeps_the_weights_and_norms():
    fc, _, _ = _eqv_fc(norm_out=True)
    wl, wr, nf = fc.w_left, fc.w_right, fc.norm_out_factor
    before = wr.numpy().copy()
    fc.build(tf.float32)
    assert fc.w_left is wl and fc.w_right is wr and fc.norm_out_factor is nf
    np.testing.assert_array_equal(wr.numpy(), before)
    assert wr.dtype == tf.float64


def test_fc_right2left_float32_weights_and_data_give_float32_output():
    fc, y, aa = _eqv_fc(
        build_dtype=tf.float32, norm_out=True, is_central_atom_type_dependent=True
    )
    data64 = _eqv_fc_data(y, aa)
    data = {
        k: (v.astype(np.float32) if v.dtype == np.float64 else v)
        for k, v in data64.items()
    }
    out = fc.frwrd(data)
    assert out.dtype == tf.float32
    assert fc.w_left.dtype == fc.w_right.dtype == fc.norm_left.dtype == tf.float32
    np.testing.assert_allclose(
        out.numpy(), _eqv_fc_oracle(fc, data64, _EQV_TYPES), **FLOAT32_NETWORK._asdict()
    )


def test_fc_right2left_does_not_cast_float64_data_to_float32_weights():
    # Pins current behaviour (reported as a finding): unlike FunctionReduceN, which casts its inputs to the
    # dtype of the weights, FCRight2Left hands mixed dtypes to ``tf.einsum`` and TensorFlow refuses them.
    fc, y, aa = _eqv_fc(build_dtype=tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError, match="float"):
        fc.frwrd(_eqv_fc_data(y, aa))


def test_fc_right2left_local_types_select_the_per_type_weights():
    fc, y, aa = _eqv_fc(is_central_atom_type_dependent=True)
    data = _eqv_fc_data(y, aa)
    local_types = np.array([1, 1, 0, 0, 1], dtype=np.int32)
    data[constants.ATOMIC_MU_I_LOCAL] = local_types
    out = _eqv_np(fc.frwrd(data, local=True))
    _eqv_assert_close(out, _eqv_fc_oracle(fc, data, local_types))
    assert (
        np.abs(out - _eqv_np(fc.frwrd(data))).max() > 1e-3
    )  # and not those of the global types


def test_fc_right2left_l2_loss_is_the_sum_of_squares_of_the_trainable_weights():
    fc, _, _ = _eqv_fc()
    expected = float(np.sum(fc.w_left.numpy() ** 2) + np.sum(fc.w_right.numpy() ** 2))
    assert float(fc.compute_l2_regularization_loss()) == pytest.approx(
        expected, rel=FLOAT64_ARITHMETIC.rtol
    )
    fc.enable_lora_adaptation({"mode": "additive"})
    assert (
        float(fc.compute_l2_regularization_loss()) == 0.0
    )  # frozen weights, zero update


@pytest.mark.parametrize(
    "dependent", [False, True, [True, False], [False, True]], ids=str
)
def test_fc_right2left_element_selection_is_not_implemented_for_per_type_weights(
    dependent,
):
    fc, _, _ = _eqv_fc(is_central_atom_type_dependent=dependent)
    if dependent:  # any per-type weight, on either side
        with pytest.raises(NotImplementedError):
            fc.prepare_variables_for_selected_elements([0])
        with pytest.raises(NotImplementedError):
            fc.upd_init_args_new_elements({"H": 0})
    else:
        assert fc.prepare_variables_for_selected_elements([0]) is None
        assert fc.upd_init_args_new_elements({"H": 0}) is None


@pytest.mark.parametrize(
    "config",
    [{"rank": 2, "alpha": 1.0}, {"mode": "additive"}],
    ids=["lora", "additive"],
)
@pytest.mark.parametrize("left_coefs", [True, False], ids=["left-coefs", "plain-left"])
def test_fc_right2left_lora_adds_the_update_to_each_weight(config, left_coefs):
    fc, y, aa = _eqv_fc(n_out=4 if left_coefs else 3, left_coefs=left_coefs)
    data = _eqv_fc_data(y, aa)
    plain = _eqv_np(fc.frwrd(data))
    names = ["w_left", "w_right"] if left_coefs else ["w_right"]
    base = {n: getattr(fc, n).numpy().copy() for n in names}

    fc.enable_lora_adaptation(config)
    assert fc.lora and fc._init_args["lora_config"] == config
    _eqv_assert_close(fc.frwrd(data), plain)  # the update starts at zero
    rng = np.random.default_rng(6)
    for n in names:
        assert not getattr(fc, n).trainable
        tensors = getattr(fc, f"{n}_lora_tensors")
        assert len(tensors) == (1 if "mode" in config else 3)
        for t in tensors:
            t.assign(rng.normal(size=t.shape))
    assert not hasattr(fc, "w_left_lora_tensors") or left_coefs
    updated = _eqv_np(fc.frwrd(data))
    assert np.abs(updated - plain).max() > 1e-3

    def delta(name):
        ts = [t.numpy() for t in getattr(fc, f"{name}_lora_tensors")]
        return ts[0] if "mode" in config else np.einsum("ar,br,cr->abc", *ts)

    # oracle with the updated weights: write them into a twin with the same tables
    twin, ty, taa = _eqv_fc(
        n_out=4 if left_coefs else 3, left_coefs=left_coefs, seed=None
    )
    for n in names:
        getattr(twin, n).assign(base[n] + delta(n))
    _eqv_assert_close(updated, _eqv_fc_oracle(twin, data, _EQV_TYPES))
    deltas = {n: delta(n) for n in names}

    fc.finalize_lora_update()
    assert not fc.lora and "lora_config" not in fc._init_args
    for n in names:
        w = getattr(fc, n)
        assert w.trainable and not hasattr(fc, f"{n}_lora_tensors")
        np.testing.assert_allclose(
            w.numpy(), base[n] + deltas[n], **FLOAT64_ARITHMETIC._asdict()
        )
    _eqv_assert_close(fc.frwrd(data), updated)


def test_fc_right2left_built_with_a_lora_config_freezes_the_weights_at_once():
    fc, y, aa = _eqv_fc(lora_config={"rank": 2, "alpha": 1.0}, seed=None)
    assert fc.lora
    assert not fc.w_left.trainable and not fc.w_right.trainable
    assert len(fc.w_left_lora_tensors) == len(fc.w_right_lora_tensors) == 3
    data = _eqv_fc_data(y, aa)
    _eqv_assert_close(fc.frwrd(data), _eqv_fc_oracle(fc, data, _EQV_TYPES))


def test_fc_right2left_is_equivariant_and_adds_nothing_to_a_vanishing_right_factor():
    """Rotate the generating vectors: each block of the output rotates with its Wigner matrix."""
    fc, y, aa = _eqv_fc(is_central_atom_type_dependent=True)
    pos = _eqv_positions()
    rot = _eqv_rotation()

    def run(r):
        data = {
            "Y": _eqv_equivariant_features(pos, r, n=y.n_out),
            "Yb": _eqv_equivariant_features(pos, r, n=aa.n_out),
            constants.ATOMIC_MU_I: _EQV_TYPES,
        }
        data["AA"] = _eqv_np(aa.frwrd(data))
        return _eqv_np(fc.frwrd(data))

    base, rotated = run(None), run(rot)
    assert np.abs(base).max() > 1e-3
    _eqv_assert_equivariant(fc.coupling_meta_data, rotated, base, rot)


def test_fc_right2left_call_adds_the_output_and_never_edits_the_inputs():
    fc, y, aa = _eqv_fc()
    data = _eqv_fc_data(y, aa)
    snapshot = {k: np.array(v, copy=True) for k, v in data.items()}
    out = fc(data)
    assert set(out) == set(snapshot) | {"FC"}
    for key, value in snapshot.items():
        np.testing.assert_array_equal(np.asarray(data[key]), value)


# ================================================================================================
# InvariantLayerRMSNorm
# ================================================================================================

_EQV_N_REAL = 4  # the fifth atom of the batch is padding
_EQV_RMS_EPS = 1e-10  # the constant of the layer, written out here as an oracle input


def _eqv_rms_layer(
    kind: str = "full",
    init: str = "zeros",
    n_out: int = 5,
    lm_first: bool = False,
    build_dtype=tf.float64,
    seed: int | None = 81,
):
    """A built ``InvariantLayerRMSNorm`` on a stand-in producer that only carries ``name``, ``n_out``, ``lm_first``."""
    producer = _eqv_harmonic(2, "X", n_out=n_out)
    producer.lm_first = lm_first
    layer = InvariantLayerRMSNorm(inpt=producer, name="LN", type=kind, init=init)
    layer.build(build_dtype)
    if seed is not None:
        seed_trainable_variables(layer, seed=seed)
    return layer


def _eqv_rms_data(
    n_out: int = 5, n_lm: int = 3, seed: int = 91, lm_first: bool = False
) -> dict:
    x = np.random.default_rng(seed).normal(size=(_EQV_N_ATOMS, n_out, n_lm))
    return {
        "X": np.transpose(x, (2, 0, 1)) if lm_first else x,
        constants.ATOMIC_MU_I: _EQV_TYPES.copy(),
        constants.N_ATOMS_BATCH_REAL: _EQV_N_REAL,
    }


def _eqv_rms_oracle(layer, x: np.ndarray, n_real: int = _EQV_N_REAL) -> np.ndarray:
    """The three normalisations of ``x[atoms, n, lm]`` written out from the stored scales."""
    scale = layer.scale.numpy()  # [1, n or n - 1, 1]
    real = (np.arange(len(x)) < n_real)[:, None, None]
    if layer.type == "full":
        rms = 1.0 / np.sqrt(np.mean(x**2, axis=1, keepdims=True) + _EQV_RMS_EPS)
        return np.where(real, x * rms * scale, 0.0)
    lin = x[:, 0, :]
    if layer.type == "sep_lin_gate":
        lin = lin * layer.lin_scale.numpy()
    nonlin = x[:, 1:, :]
    rms = 1.0 / np.sqrt(np.mean(nonlin**2, axis=1, keepdims=True) + _EQV_RMS_EPS)
    normalised = np.where(real, nonlin * rms * scale, 0.0)
    return np.concatenate([lin[:, None, :], normalised], axis=1)


@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
@pytest.mark.parametrize("kind", ["full", "only_nonlin", "sep_lin_gate"])
def test_invariant_layer_rms_norm_equals_the_numpy_recomputation(kind, lm_first):
    layer = _eqv_rms_layer(kind, lm_first=lm_first)
    data = _eqv_rms_data(lm_first=lm_first)
    x = np.transpose(data["X"], (1, 2, 0)) if lm_first else data["X"]
    out = _eqv_np(layer.frwrd(data))
    assert out.shape == x.shape
    _eqv_assert_close(out, _eqv_rms_oracle(layer, x))


@pytest.mark.parametrize(
    ("kind", "scale_shape", "has_gate"),
    [
        ("full", [1, 5, 1], False),
        ("only_nonlin", [1, 4, 1], False),
        ("sep_lin_gate", [1, 4, 1], True),
    ],
)
def test_invariant_layer_rms_norm_variables_follow_the_type(
    kind, scale_shape, has_gate
):
    layer = _eqv_rms_layer(kind, init="ones", seed=None)
    assert layer.scale.shape.as_list() == scale_shape
    assert hasattr(layer, "lin_scale") is has_gate
    if has_gate:
        assert layer.lin_scale.shape.as_list() == [1, 1]
    assert len(layer.trainable_variables) == 1 + has_gate
    assert float(layer.epsilon) == _EQV_RMS_EPS
    assert layer.n_out == 5


@pytest.mark.parametrize("kind", ["full", "sep_lin_gate"])
def test_invariant_layer_rms_norm_initialisers(kind):
    zeros = _eqv_rms_layer(kind, "zeros", n_out=50, seed=None)
    assert not zeros.scale.numpy().any()
    ones = _eqv_rms_layer(kind, "ones", n_out=50, seed=None)
    assert np.all(ones.scale.numpy() == 1.0)
    tf.random.set_seed(3)
    rnd = _eqv_rms_layer(kind, "random", n_out=400, seed=None)
    assert rnd.scale.numpy().std() == pytest.approx(
        1.0, rel=sample_std_rtol(rnd.scale.shape.num_elements())
    )
    near = _eqv_rms_layer(kind, "near_zero", n_out=400, seed=None)
    assert near.scale.numpy().std() == pytest.approx(
        1e-8, rel=sample_std_rtol(near.scale.shape.num_elements())
    )
    if kind == "sep_lin_gate":
        assert not zeros.lin_scale.numpy().any()
        assert float(ones.lin_scale[0, 0]) == 1.0
        assert float(rnd.lin_scale[0, 0]) != 0.0
        assert 0.0 < abs(float(near.lin_scale[0, 0])) < 1e-6


def test_invariant_layer_rms_norm_zero_init_keeps_only_the_linear_channel():
    """With the default zero scale the normalised channels vanish and channel 0 passes through unchanged."""
    layer = _eqv_rms_layer("only_nonlin", "zeros", seed=None)
    data = _eqv_rms_data()
    out = _eqv_np(layer.frwrd(data))
    np.testing.assert_array_equal(out[:, 0, :], data["X"][:, 0, :])
    assert not out[:, 1:, :].any()
    full = _eqv_rms_layer("full", "zeros", seed=None)
    assert not _eqv_np(full.frwrd(data)).any()


def test_invariant_layer_rms_norm_hand_computed_value():
    """``x = (3, 4)`` has mean square 12.5, so ``out = x / sqrt(12.5) = (0.8485, 1.1314)`` with unit scale."""
    layer = _eqv_rms_layer("full", "ones", n_out=2, seed=None)
    data = {
        "X": np.array([[[3.0], [4.0]]]),
        constants.ATOMIC_MU_I: np.array([0], dtype=np.int32),
        constants.N_ATOMS_BATCH_REAL: 1,
    }
    out = _eqv_np(layer.frwrd(data))
    expected = np.array([[[3.0 / np.sqrt(12.5)], [4.0 / np.sqrt(12.5)]]])
    np.testing.assert_allclose(
        out, expected, rtol=1e-9
    )  # the layer adds 1e-10 to the mean square
    assert np.sqrt(np.mean(out**2)) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("kind", ["full", "only_nonlin", "sep_lin_gate"])
def test_invariant_layer_rms_norm_padding_atoms_have_a_vanishing_normalised_part(kind):
    layer = _eqv_rms_layer(kind, "random", seed=None)
    data = _eqv_rms_data()
    out = _eqv_np(layer.frwrd(data))
    first = 0 if kind == "full" else 1
    assert np.all(out[_EQV_N_REAL:, first:, :] == 0.0)
    assert np.abs(out[:_EQV_N_REAL, first:, :]).min() > 0.0
    if kind != "full":
        # Pins current behaviour: the linear channel of a padding atom is not masked
        assert np.abs(out[_EQV_N_REAL:, 0, :]).min() > 0.0


def test_invariant_layer_rms_norm_normalised_channels_do_not_depend_on_the_input_scale():
    """``out(c x)`` equals ``out(x)`` for the normalised channels (up to the epsilon) and scales for the linear one."""
    layer = _eqv_rms_layer("sep_lin_gate", "random", seed=None)
    data = _eqv_rms_data()
    base = _eqv_np(layer.frwrd(data))
    scaled = _eqv_np(layer.frwrd(dict(data, X=3.0 * data["X"])))
    _eqv_assert_close(scaled[:, 1:, :], base[:, 1:, :], COVARIANCE_F64)
    _eqv_assert_close(scaled[:, 0, :], 3.0 * base[:, 0, :])
    full = _eqv_rms_layer("full", "random", seed=None)
    _eqv_assert_close(
        full.frwrd(dict(data, X=4.0 * data["X"])),
        _eqv_np(full.frwrd(data)),
        COVARIANCE_F64,
    )


def test_invariant_layer_rms_norm_is_invariant_under_a_permutation_of_the_normalised_channels():
    layer_a = _eqv_rms_layer("only_nonlin", "ones", seed=None)
    data = _eqv_rms_data()
    perm = np.array([0, 3, 1, 4, 2])  # the linear channel stays first
    permuted = dict(data, X=data["X"][:, perm, :])
    out, out_perm = _eqv_np(layer_a.frwrd(data)), _eqv_np(layer_a.frwrd(permuted))
    _eqv_assert_close(out_perm, out[:, perm, :])


def test_invariant_layer_rms_norm_local_atoms_set_the_masked_range():
    layer = _eqv_rms_layer("full", "random", seed=None)
    data = _eqv_rms_data()
    data[constants.ATOMIC_MU_I_LOCAL] = _EQV_TYPES[:3]
    data["X"] = data["X"][:3]
    data[constants.N_ATOMS_BATCH_REAL] = 2
    out = _eqv_np(layer.frwrd(data, local=True))
    assert out.shape == (3, 5, 3)
    assert np.all(out[2:] == 0.0)
    _eqv_assert_close(out, _eqv_rms_oracle(layer, data["X"], n_real=2))


def test_invariant_layer_rms_norm_float32_parameters_follow_the_data_dtype():
    layer = _eqv_rms_layer("sep_lin_gate", "random", build_dtype=tf.float32, seed=None)
    data = _eqv_rms_data()
    out64 = layer.frwrd(data)
    assert out64.dtype == tf.float64  # the scales are cast to the dtype of the data
    data32 = dict(data, X=data["X"].astype(np.float32))
    out32 = layer.frwrd(data32)
    assert out32.dtype == tf.float32
    assert layer.scale.dtype == layer.epsilon.dtype == tf.float32
    np.testing.assert_allclose(
        out32.numpy(), _eqv_rms_oracle(layer, data["X"]), **FLOAT32_NETWORK._asdict()
    )


def test_invariant_layer_rms_norm_refuses_unknown_options():
    producer = _eqv_harmonic(1, "X", n_out=4)
    with pytest.raises(AssertionError, match="Unknown type fancy"):
        InvariantLayerRMSNorm(inpt=producer, name="LN", type="fancy")
    with pytest.raises(AssertionError):
        InvariantLayerRMSNorm(inpt=producer, name="LN", init="orthogonal")
    layer = InvariantLayerRMSNorm(inpt=producer, name="LN")
    layer.type = "fancy"  # the constructor asserts on this; build must still refuse
    with pytest.raises(ValueError, match="Unknown type fancy"):
        layer.build(tf.float64)


def test_invariant_layer_rms_norm_second_build_keeps_the_scale_and_call_keeps_the_inputs():
    layer = _eqv_rms_layer("full", "random")
    scale = layer.scale
    before = scale.numpy().copy()
    layer.build(tf.float32)
    assert layer.scale is scale
    np.testing.assert_array_equal(scale.numpy(), before)
    assert scale.dtype == tf.float64
    data = _eqv_rms_data()
    snapshot = {k: np.array(v, copy=True) for k, v in data.items()}
    out = layer(data)
    assert set(out) == set(snapshot) | {"LN"}
    for key, value in snapshot.items():
        np.testing.assert_array_equal(np.asarray(data[key]), value)
    assert layer._init_args["type"] == "full"
    assert layer._init_args["init"] == "random"


@pytest.mark.parametrize("lm_first", [False, True], ids=["standard", "lm-first"])
@pytest.mark.parametrize("kind", ["full", "only_nonlin"])
def test_invariant_layer_rms_norm_after_a_reducer_of_invariants(kind, lm_first):
    """The shipped chain: ``FunctionReduceN`` of scalars (``[atoms, n, 1]`` or ``[1, atoms, n]``) then the norm."""
    red, y, aa = _eqv_reducer(ls_max=0, allowed=[[0, 1]], n_out=6, lm_first=lm_first)
    layer = InvariantLayerRMSNorm(inpt=red, name="LN", type=kind, init="ones")
    layer.build(tf.float64)
    data = _eqv_reduce_data(y, aa)
    data[constants.N_ATOMS_BATCH_REAL] = _EQV_N_REAL
    feed = dict(data)
    if lm_first:
        feed["Y"], feed["AA"] = (np.transpose(data[k], (2, 0, 1)) for k in ("Y", "AA"))
    reduced = _eqv_np(red.frwrd(feed))
    x = np.transpose(reduced, (1, 2, 0)) if lm_first else reduced
    assert x.shape == (5, 6, 1)
    out = _eqv_np(layer.frwrd(dict(feed, R=reduced)))
    _eqv_assert_close(out, _eqv_rms_oracle(layer, x))


def test_spbf_equiv_ind_casts_angular_to_the_radial_dtype():
    ins, _ = _eqv_equiv_ind()
    data = _eqv_equiv_data(ins)
    data["Y"] = data["Y"].astype(np.float32)
    out = ins.frwrd(data)
    assert out.dtype == tf.float64
    rounded = dict(data, Y=data["Y"].astype(np.float64))
    _eqv_assert_close(out, _eqv_equiv_oracle(ins, rounded, data["I"], 5))


@pytest.mark.parametrize("normalize", [True, False], ids=["normalize", "no-normalize"])
def test_function_reduce_n_initialisers_and_norms_of_a_single_build(normalize):
    """Random, uniform and zero initialisers of the weights, and the norm constant, for both ``normalize``.

    ``Y`` has 3 channels and ``AA`` has 5: each input uses its own width.
    """
    tf.random.set_seed(17)
    scale = 2.0
    kwargs = {"scale": scale, "normalize": normalize, "n_out": 60, "seed": None}
    built = {
        init: _eqv_reducer(init_vars=init, **kwargs)[0]
        for init in ("random", "uniform", "zeros")
    }
    for name, n_in in (("Y", 3), ("AA", 5)):
        # std of the normal, half-width of the uniform, and the constant that multiplies the product
        bound = scale if normalize else 1 / np.sqrt(n_in)
        norm = scale / np.sqrt(n_in) if normalize else 1.0
        w = getattr(built["random"], f"reducing_{name}").numpy()
        assert w.std() == pytest.approx(bound, rel=sample_std_rtol(w.size))
        w = getattr(built["uniform"], f"reducing_{name}").numpy()
        assert np.abs(w).max() <= bound
        assert np.abs(w).max() > 0.95 * bound
        assert w.std() == pytest.approx(
            bound / np.sqrt(3.0), rel=sample_std_rtol(w.size)
        )
        assert not getattr(built["zeros"], f"reducing_{name}").numpy().any()
        for red in built.values():
            assert float(getattr(red, f"norm_{name}")) == pytest.approx(norm)


def test_function_reduce_n_build_refuses_an_initialiser_set_after_construction():
    y, aa = _eqv_reduce_inputs()
    red = FunctionReduceN(
        instructions=[y, aa], name="R", ls_max=2, n_out=2, allowed_l_p=_EQV_ALLOWED
    )
    red.init_vars = (
        "orthogonal"  # the constructor asserts on this; build must still refuse
    )
    with pytest.raises(NotImplementedError, match="orthogonal"):
        red.build(tf.float64)


def test_invariant_layer_rms_norm_build_with_an_initialiser_set_after_construction_creates_nothing():
    # Pins current behaviour (reported as a finding): ``build`` has no ``else`` for ``init``, so an unknown
    # value assigned after the constructor's assertion leaves the layer built but without a ``scale``.
    layer = InvariantLayerRMSNorm(inpt=_eqv_harmonic(1, "X", n_out=4), name="LN")
    layer.init = "orthogonal"
    layer.build(tf.float64)
    assert layer.is_built
    assert not hasattr(layer, "scale")
    assert not layer.trainable_variables


def test_invariant_layer_rms_norm_frwrd_with_a_type_set_after_building_has_no_linear_channel():
    # Pins current behaviour (reported as a finding): the three types are validated only in the constructor,
    # a later change of ``type`` falls through the ``if`` chain of ``frwrd`` and fails on an unbound name.
    layer = _eqv_rms_layer("only_nonlin")
    layer.type = "fancy"
    with pytest.raises(UnboundLocalError):
        layer.frwrd(_eqv_rms_data())
