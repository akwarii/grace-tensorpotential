"""Characterization tests for the ``build`` of the four reducing instructions.

``FunctionReduce``, ``FunctionReduceN``, ``FunctionReduceParticular`` and
``CollectInvarBasis`` (``tensorpotential/instructions/compute.py``) turn the equivariant
tensors of earlier instructions into one tensor with the allowed ``(l, parity)`` blocks.
``build`` creates the weights ``reducing_<name>`` of shape ``[n_out, n_in, w_shape]``
(``[n_types, ...]`` when the central atom type matters) and the constants ``norm_<name>``.

Logic layer: shapes, initialisers, dtypes, the options that change them, building twice.
Value layer: every table is derived by hand from the selection rules of the coupling of two
spherical harmonics (triangle rule, parity ``(-1)^(l1+l2)``), initialisers are compared with
their stated standard deviation, and the built weights make the output rotate with the real
Wigner matrices: ``out(R x) = D(R) out(x)``, block by block in ``l``.
"""

from __future__ import annotations

import os
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf
from scipy.spatial.transform import Rotation

from tensorpotential import constants
from tensorpotential.instructions import (
    FunctionReduce,
    FunctionReduceN,
    ProductFunction,
    SphericalHarmonic,
)
from tensorpotential.instructions.compute import (
    CollectInvarBasis,
    FunctionReduceParticular,
)
from tests.test_instructions import _compute_wigner_d_real
from tests.tolerances import COVARIANCE_F64, sample_std_rtol

ALLOWED = [[0, 1], [1, -1], [2, 1]]
LMAX_IN = 3  # the harmonics, and the products of two of them, run up to l = 3
Y_NOUT = 3
AA_NOUT = 5


def product_groups(l_out: int, parity: int) -> list[tuple[int, int]]:
    """The ``(l1, l2)`` histories coupling to ``l_out`` with ``parity``, with ``l1 >= l2``.

    Triangle rule ``|l1 - l2| <= l_out <= l1 + l2`` and parity ``(-1)**(l1 + l2)`` of the
    product of two spherical harmonics; the symmetric partner ``l1 < l2`` is the same function.
    A product of a harmonic with itself (``l1 == l2``) is symmetric under exchange, which
    leaves only even ``l_out`` (the Clebsch-Gordan symmetry ``(-1)**(l1 + l2 - l_out)``).
    """
    return [
        (l1, l2)
        for l1 in range(LMAX_IN + 1)
        for l2 in range(l1 + 1)
        if abs(l1 - l2) <= l_out <= l1 + l2
        and (-1) ** (l1 + l2) == parity
        and (l1 != l2 or l_out % 2 == 0)
    ]


def n_groups(allowed: list[list[int]], max_l: int) -> int:
    """Weight columns for the product ``AA`` (one per history) restricted to ``allowed``."""
    return sum(len(product_groups(l, p)) for l, p in allowed if l <= max_l)


def make_inputs(
    y_nout: int = Y_NOUT, aa_nout: int = AA_NOUT
) -> tuple[SphericalHarmonic, ProductFunction]:
    """Harmonics up to l=3 and their product coupled to l<=2 (the real objects, built).

    ``n_out`` is the feature count the reducer sees; it is only bookkeeping for the tables
    and weight shapes, so the shape tests use distinct values and the rotation tests 1.
    """
    y = SphericalHarmonic(vhat="v", lmax=LMAX_IN, name="Y")
    y.build(tf.float64)
    y.n_out = y_nout
    aa = ProductFunction(
        left=y,
        right=y,
        name="AA",
        lmax=LMAX_IN,
        Lmax=2,
        keep_parity=[[0, 1], [1, -1], [1, 1], [2, 1], [2, -1]],
    )
    aa.n_out = aa_nout
    return y, aa


def reducing(ins, name: str) -> tf.Variable:
    return getattr(ins, f"reducing_{name}")


# --------------------------------------------------------------------- hand-derived tables


def test_hand_derived_product_groups_match_the_instruction():
    _, aa = make_inputs()
    meta = aa.coupling_meta_data
    for l_out, parity in [(0, 1), (1, -1), (2, 1), (1, 1), (2, -1)]:
        rows = meta[(meta["l"] == l_out) & (meta["parity"] == parity)]
        hist = {tuple(int(c) for c in h.strip("()").split(",")) for h in rows["hist"]}
        assert hist == set(product_groups(l_out, parity))


def test_weight_columns_equal_the_number_of_collected_groups():
    y, aa = make_inputs()
    red = FunctionReduceN(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=4, allowed_l_p=ALLOWED
    )
    assert red.collector["Y"]["w_shape"] == 3  # (l, p) = (0, +), (1, -), (2, +)
    assert red.collector["AA"]["w_shape"] == n_groups(ALLOWED, 2) == 12


# ----------------------------------------------------------------- FunctionReduce / N


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
@pytest.mark.parametrize("n_types", [None, 3])
def test_weights_have_shape_out_in_columns(cls, n_types):
    y, aa = make_inputs()
    red = cls(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=ALLOWED,
        is_central_atom_type_dependent=n_types is not None,
        number_of_atom_types=n_types,
    )
    red.build(tf.float64)
    lead = [] if n_types is None else [n_types]
    assert reducing(red, "Y").shape.as_list() == [*lead, 4, Y_NOUT, 3]
    assert reducing(red, "AA").shape.as_list() == [*lead, 4, AA_NOUT, 12]
    assert {v.name for v in red.trainable_variables} == {
        "R/reducing_Y:0",
        "R/reducing_AA:0",
    }
    assert red.is_built


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_weights_and_constants_follow_the_float_dtype(cls, dtype):
    y, aa = make_inputs()
    red = cls(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=4, allowed_l_p=ALLOWED
    )
    red.build(dtype)
    assert reducing(red, "Y").dtype == reducing(red, "AA").dtype == dtype
    assert red.norm_Y.dtype == red.norm_AA.dtype == dtype
    assert red.float_dtype == dtype


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
def test_second_build_keeps_the_weights(cls):
    y, aa = make_inputs()
    red = cls(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=4, allowed_l_p=ALLOWED
    )
    red.build(tf.float64)
    weights = reducing(red, "Y")
    before = weights.numpy().copy()
    red.build(tf.float32)  # ignored once built
    assert reducing(red, "Y") is weights
    np.testing.assert_array_equal(weights.numpy(), before)
    assert red.float_dtype == tf.float64


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
def test_zero_init_gives_zero_weights(cls):
    y, aa = make_inputs()
    red = cls(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=ALLOWED,
        init_vars="zeros",
    )
    red.build(tf.float64)
    assert not reducing(red, "Y").numpy().any()
    assert not reducing(red, "AA").numpy().any()


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
def test_unknown_initialiser_set_after_construction_is_refused(cls):
    y, aa = make_inputs()
    red = cls(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=4, allowed_l_p=ALLOWED
    )
    red.init_vars = (
        "orthogonal"  # the constructor asserts on this; build must still refuse
    )
    with pytest.raises(NotImplementedError, match="orthogonal"):
        red.build(tf.float64)


def test_function_reduce_random_init_has_the_he_standard_deviation():
    y, aa = make_inputs()
    tf.random.set_seed(11)
    red = FunctionReduce(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=40, allowed_l_p=ALLOWED
    )
    red.build(tf.float64)
    for name, n_in, columns in [("Y", Y_NOUT, 3), ("AA", AA_NOUT, 12)]:
        w = reducing(red, name).numpy()
        expected = (2.0 / (n_in * columns)) ** 0.5
        assert w.std() == pytest.approx(expected, rel=sample_std_rtol(w.size))
        assert abs(w.mean()) < 5 * expected / w.size**0.5
        assert float(getattr(red, f"norm_{name}")) == 1.0
    assert float(red.n_instr) == pytest.approx(0.5)  # 1 / number of instructions


@pytest.mark.parametrize(
    ("normalize", "scale", "weight_std", "norm_of"),
    [
        (True, 1.0, lambda s, n: s, lambda s, n: s / n**0.5),
        (True, 2.5, lambda s, n: s, lambda s, n: s / n**0.5),
        (False, 1.0, lambda s, n: 1 / n**0.5, lambda s, n: 1.0),
    ],
    ids=["normalize", "normalize-scaled", "no-normalize"],
)
def test_function_reduce_n_random_init_and_norm(normalize, scale, weight_std, norm_of):
    y, aa = make_inputs()
    tf.random.set_seed(12)
    red = FunctionReduceN(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=40,
        allowed_l_p=ALLOWED,
        normalize=normalize,
        scale=scale,
    )
    red.build(tf.float64)
    for name, n_in in [("Y", Y_NOUT), ("AA", AA_NOUT)]:
        w = reducing(red, name).numpy()
        assert w.std() == pytest.approx(
            weight_std(scale, n_in), rel=sample_std_rtol(w.size)
        )
        assert float(getattr(red, f"norm_{name}")) == pytest.approx(
            norm_of(scale, n_in)
        )


def test_function_reduce_n_uniform_init_stays_inside_its_bound():
    y, aa = make_inputs()
    tf.random.set_seed(13)
    red = FunctionReduceN(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=40,
        allowed_l_p=ALLOWED,
        init_vars="uniform",
        scale=2.0,
    )
    red.build(tf.float64)
    w = reducing(red, "AA").numpy()
    assert np.abs(w).max() <= 2.0
    assert np.abs(w).max() > 1.9  # fills the interval, so it is not a normal draw
    # a uniform on [-s, s] has standard deviation s / sqrt(3)
    assert w.std() == pytest.approx(2.0 / 3**0.5, rel=sample_std_rtol(w.size))


def test_function_reduce_n_output_norm_counts_the_contributing_instructions():
    """Channel (l, m) is fed by every group of every input that carries that l.

    l=0: Y gives 1 and AA gives 4 histories; l=1: 1 + 3; l=2: 1 + 5. The map is 1/sqrt(count).
    """
    y, aa = make_inputs()
    red = FunctionReduceN(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=ALLOWED,
        out_norm=True,
    )
    counts = [1 + len(product_groups(l, p)) for l, p in ALLOWED]
    expected = np.concatenate([
        [c**-0.5] * (2 * l + 1) for c, (l, _) in zip(counts, ALLOWED)
    ])
    np.testing.assert_allclose(np.asarray(red.norm_map), expected, rtol=1e-12)
    red.build(tf.float64)
    assert red.norm_map.shape.as_list() == [9, 1, 1]
    np.testing.assert_allclose(red.norm_map.numpy().ravel(), expected, rtol=1e-12)


def test_function_reduce_n_without_output_norm_builds_a_unit_constant():
    y, aa = make_inputs()
    red = FunctionReduceN(
        instructions=[y, aa], name="R", ls_max=[2, 2], n_out=4, allowed_l_p=ALLOWED
    )
    red.build(tf.float64)
    assert red.norm_map.shape.as_list() == []
    assert float(red.norm_map) == 1.0


@pytest.mark.parametrize("config", [{"rank": 2, "alpha": 1}, {"mode": "additive"}])
def test_function_reduce_n_with_lora_freezes_the_base_weights(config):
    y, aa = make_inputs()
    red = FunctionReduceN(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=ALLOWED,
        lora_config=config,
    )
    red.build(tf.float64)
    assert red.lora
    for name in ("Y", "AA"):
        assert not reducing(red, name).trainable
        assert len(getattr(red, f"reducing_{name}_lora_tensors")) == (
            1 if "mode" in config else 3
        )
    red.finalize_lora_update()
    assert not red.lora
    assert not hasattr(red, "reducing_Y_lora_tensors")


# ------------------------------------------------------------ FunctionReduceParticular


def particular(**kwargs) -> FunctionReduceParticular:
    y, aa = make_inputs()
    options = (
        dict(instructions=[y, aa], name="Q", selected_l=2, selected_p=1, n_out=3)
        | kwargs
    )
    return FunctionReduceParticular(**options)


@pytest.mark.parametrize("n_types", [None, 2])
def test_particular_weights_have_one_column_per_group_of_the_selected_block(n_types):
    q = particular(
        is_central_atom_type_dependent=n_types is not None, number_of_atom_types=n_types
    )
    q.build(tf.float64)
    lead = [] if n_types is None else [n_types]
    assert reducing(q, "Y").shape.as_list() == [*lead, 3, Y_NOUT, 1]
    assert reducing(q, "AA").shape.as_list() == [
        *lead,
        3,
        AA_NOUT,
        len(product_groups(2, 1)),
    ]
    assert q.is_built


def test_particular_random_init_is_unit_normal_and_scaled_by_inverse_sqrt_n_in():
    tf.random.set_seed(14)
    q = particular(n_out=80)
    q.build(tf.float64)
    for name, n_in in [("Y", Y_NOUT), ("AA", AA_NOUT)]:
        w = reducing(q, name).numpy()
        assert w.std() == pytest.approx(1.0, rel=sample_std_rtol(w.size))
        assert float(getattr(q, f"norm_{name}")) == pytest.approx(n_in**-0.5)
    assert float(q.norm) == 1.0


def test_particular_output_norm_counts_the_contributing_groups():
    # every one of the five m channels of (l=2, +) gets Y once and AA five times
    q = particular(out_norm=True)
    expected = np.full(5, (1 + len(product_groups(2, 1))) ** -0.5)
    np.testing.assert_allclose(q._out_norm_map, expected, rtol=1e-12)
    q.build(tf.float64)
    assert q.norm_map.shape.as_list() == [5, 1, 1]
    np.testing.assert_allclose(q.norm_map.numpy().ravel(), expected, rtol=1e-12)


def test_particular_without_output_norm_builds_a_unit_constant():
    q = particular()
    q.build(tf.float64)
    assert q.norm_map.shape.as_list() == []
    assert float(q.norm_map) == 1.0


def test_particular_second_build_keeps_the_weights():
    q = particular()
    q.build(tf.float64)
    weights = reducing(q, "AA")
    before = weights.numpy().copy()
    q.build(tf.float32)
    assert reducing(q, "AA") is weights
    np.testing.assert_array_equal(weights.numpy(), before)


def test_particular_refuses_a_selection_above_the_lmax_of_its_inputs():
    with pytest.raises(AssertionError, match="required lmax 3"):
        particular(selected_l=3)  # AA only reaches l = 2


# ------------------------------------------------------------------ CollectInvarBasis


class ConcreteCollectInvarBasis(CollectInvarBasis):
    """``CollectInvarBasis`` lacks ``upd_init_args_new_elements`` and cannot be instantiated.

    The existing layout test uses the same stub; it stands in for the missing method only.
    """

    def upd_init_args_new_elements(self, new_element_map):
        return None


def collector(**kwargs) -> ConcreteCollectInvarBasis:
    y, aa = make_inputs(1, 1)
    return ConcreteCollectInvarBasis(instructions=[y, aa], name="C", ls_max=0, **kwargs)


def test_collect_invar_basis_is_abstract_as_shipped():
    y, aa = make_inputs()
    shipped: Any = (
        CollectInvarBasis  # the abstract class itself; typed loosely on purpose
    )
    with pytest.raises(TypeError, match="upd_init_args_new_elements"):
        shipped(instructions=[y, aa], name="C", ls_max=0)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_collect_invar_basis_build_only_records_the_dtype(dtype):
    c = collector()
    assert not c.is_built
    c.build(dtype)
    assert c.is_built
    assert c.float_dtype == dtype
    assert not c.variables  # a plain collector owns no weights


def test_collect_invar_basis_second_build_keeps_the_first_dtype():
    c = collector()
    c.build(tf.float64)
    c.build(tf.float32)
    assert c.float_dtype == tf.float64


def test_collect_invar_basis_collects_the_invariants_of_each_input():
    c = collector()
    assert c.coupling_meta_data[["l", "m", "parity"]].values.tolist() == [[0, 0, 1]]
    # Y has a single l=0 function; AA has one per history that couples to l=0
    assert c.collector["Y"]["func_collect_ind"].tolist() == [0]
    assert len(c.collector["AA"]["func_collect_ind"]) == len(product_groups(0, 1)) == 4
    for name in ("Y", "AA"):
        assert not c.collector[name]["total_sum_ind"].numpy().any()  # all land in row 0


def test_collect_invar_basis_refuses_equivariant_inputs():
    y, aa = make_inputs()
    with pytest.raises(AssertionError):
        ConcreteCollectInvarBasis(instructions=[y, aa], name="C", ls_max=1)


# ------------------------------------------------------------------------- rotation


def unit_vectors(n: int) -> np.ndarray:
    v = np.random.default_rng(5).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def reduce_inputs(vectors: np.ndarray) -> dict:
    """Y and AA evaluated on ``vectors`` (one atom per vector), plus the bookkeeping keys."""
    y, aa = make_inputs(1, 1)
    data = {"v": vectors}
    data = y(data)
    n = len(vectors)
    data["Y"] = data["Y"].numpy().reshape(n, 1, (LMAX_IN + 1) ** 2)
    data = aa(data)
    data[constants.N_ATOMS_BATCH_TOTAL] = n
    data[constants.ATOMIC_MU_I] = np.zeros(n, dtype=np.int32)
    return data


def block_slices(allowed: list[list[int]]) -> list[tuple[int, slice]]:
    start, out = 0, []
    for l, _ in allowed:
        out.append((l, slice(start, start + 2 * l + 1)))
        start += 2 * l + 1
    return out


def wigner():
    rot = Rotation.from_rotvec(0.9 * np.array([2.0, -1.0, 1.0]) / 6**0.5)
    return rot, _compute_wigner_d_real(rot, 3)


def inputs_for_reduce(rot: Rotation):
    base = unit_vectors(7)
    return reduce_inputs(base), reduce_inputs(rot.apply(base))


def reducer_pair(cls, **kwargs):
    """Two reducers over (Y, AA) with identical weights, for the original and rotated data."""
    y, aa = make_inputs(1, 1)
    tf.random.set_seed(21)
    red = cls(
        instructions=[y, aa],
        name="R",
        ls_max=[2, 2],
        n_out=4,
        allowed_l_p=ALLOWED,
        **kwargs,
    )
    red.build(tf.float64)
    return red


@pytest.mark.parametrize("cls", [FunctionReduce, FunctionReduceN])
@pytest.mark.parametrize("n_types", [None, 1])
def test_reduced_blocks_rotate_with_the_wigner_matrices(cls, n_types):
    rot, d = wigner()
    original, rotated = inputs_for_reduce(rot)
    red = reducer_pair(
        cls,
        is_central_atom_type_dependent=n_types is not None,
        number_of_atom_types=n_types,
    )
    out0 = red.frwrd(original).numpy()  # [atoms, n_out, lm]
    out1 = red.frwrd(rotated).numpy()
    assert np.abs(out0).max() > 1e-3
    for l, rows in block_slices(ALLOWED):
        expected = np.einsum("ab,nkb->nka", d[l], out0[..., rows])
        np.testing.assert_allclose(
            out1[..., rows],
            expected,
            rtol=COVARIANCE_F64.rtol,
            atol=COVARIANCE_F64.atol,
        )


def test_particular_block_rotates_with_the_wigner_matrix():
    rot, d = wigner()
    original, rotated = inputs_for_reduce(rot)
    y, aa = make_inputs(1, 1)
    tf.random.set_seed(22)
    q = FunctionReduceParticular(
        instructions=[y, aa],
        name="Q",
        selected_l=2,
        selected_p=1,
        n_out=3,
        out_norm=True,
    )
    q.build(tf.float64)
    out0 = q.frwrd(original).numpy()  # [atoms, n_out, 5]
    out1 = q.frwrd(rotated).numpy()
    assert np.abs(out0).max() > 1e-3
    np.testing.assert_allclose(
        out1,
        np.einsum("ab,nkb->nka", d[2], out0),
        rtol=COVARIANCE_F64.rtol,
        atol=COVARIANCE_F64.atol,
    )


def test_collected_invariants_do_not_change_under_rotation():
    rot, _ = wigner()
    original, rotated = inputs_for_reduce(rot)
    c = collector()
    c.build(tf.float64)
    b0, b1 = c.frwrd(original).numpy(), c.frwrd(rotated).numpy()
    assert b0.shape == (7, 1 * 1 + 1 * 4)  # Y: 1 invariant x n_out 1; AA: 4 x n_out 1
    assert np.abs(b0).max() > 1e-3
    np.testing.assert_allclose(
        b1, b0, rtol=COVARIANCE_F64.rtol, atol=COVARIANCE_F64.atol
    )
