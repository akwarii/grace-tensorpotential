"""The ``SingleParticleBasisFunctionEquivariantInd`` plan (``plans.spbf_equivariant_ind``): logic and physics.

Logic: the flattening of the coupling table, the dtypes, ``nfunc``, natural parity as the default, the cut of the
angular columns and its checks, the options read from a yaml, the errors. Physical values: coupling the real harmonics
of one direction with an indicator made of the harmonics of another gives functions whose ``L = 0`` rows are rotation
invariant and whose ``L`` blocks keep their norm under a rotation of both directions, and the ``(1-, 1-) -> 0`` row is
proportional to the dot product of the two directions (oracles that do not use the plan). The equality with
TensorFlow's tables is in ``test_plans_spbf_reduce_against_tf.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.function_reduce_n import output_layout
from tensorpotential.torch_backend.plans.spbf_equivariant_ind import (
    EquivariantIndOptions,
    natural_parity,
    spbf_equivariant_ind_plan,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)

RNG_SEED = 20261016
INDICATOR = output_layout(1, [(0, 1), (1, -1)])
"""The table of a first-layer indicator ``I``: ``l = 0`` even, ``l = 1`` odd (the omat yaml)."""


def _angular(lmax: int) -> pd.DataFrame:
    return output_layout(lmax, natural_parity(lmax))


def _plan(lmax: int = 2, Lmax: int = 2, **options: object):
    return spbf_equivariant_ind_plan(
        _angular(lmax),
        INDICATOR,
        EquivariantIndOptions(lmax=lmax, Lmax=Lmax, normalize=True, **options),  # ty: ignore[invalid-argument-type]
        radial_lmax=lmax,
    )


# ------------------------------------------------------------------ logic


def test_natural_parity():
    assert natural_parity(3) == [[0, 1], [1, -1], [2, 1], [3, -1]]


def test_tables_flatten_the_coupling_rows():
    plan = _plan()
    meta = plan.coupling_meta_data
    sizes = [len(cg) for cg in meta["cg_list"]]
    assert plan.nfunc == len(meta)
    assert plan.m_sum_ind.tolist() == np.repeat(np.arange(len(meta)), sizes).tolist()
    assert (plan.m_sum_ind.dtype, plan.lr_inds.dtype, plan.cg.dtype) == (
        np.int32,
        np.int32,
        np.float64,
    )
    assert plan.lr_inds.shape == (sum(sizes), 2)
    assert plan.lr_inds[:, 0].tolist() == np.concatenate(meta["left_inds"]).tolist()
    assert plan.lr_inds[:, 1].tolist() == np.concatenate(meta["right_inds"]).tolist()
    np.testing.assert_array_equal(plan.cg, np.concatenate(meta["cg_list"]))
    assert plan.inv_avg_n_neigh == 1.0
    assert plan.slice_angular is None


def test_default_keeps_natural_parity_and_explicit_pairs_are_honoured():
    natural = _plan()
    meta = natural.coupling_meta_data
    assert set(zip(meta["l"], meta["parity"], strict=True)) == {(0, 1), (1, -1), (2, 1)}
    both = _plan(keep_parity=((0, 1), (1, 1), (1, -1)))
    pairs = set(
        zip(
            both.coupling_meta_data["l"], both.coupling_meta_data["parity"], strict=True
        )
    )
    assert pairs == {(0, 1), (1, 1), (1, -1)}


def test_avg_n_neigh_is_inverted():
    assert _plan(avg_n_neigh=4.0).inv_avg_n_neigh == 0.25


def test_angular_columns_are_cut_when_the_harmonics_go_higher():
    plan = spbf_equivariant_ind_plan(
        _angular(3), INDICATOR, EquivariantIndOptions(lmax=2, Lmax=1), radial_lmax=2
    )
    assert plan.slice_angular == 9
    assert plan.lr_inds[:, 0].max() < 9


@pytest.mark.parametrize(
    ("angular_lmax", "radial_lmax", "match"),
    [(3, 3, "prepared for lmax=3"), (2, 1, "do not match")],
)
def test_degree_mismatch_is_a_typed_error(angular_lmax, radial_lmax, match):
    with pytest.raises(OptionValueError, match=match):
        spbf_equivariant_ind_plan(
            _angular(angular_lmax),
            INDICATOR,
            EquivariantIndOptions(lmax=2, Lmax=1),
            radial_lmax=radial_lmax,
        )


def test_no_coupling_is_a_typed_error():
    with pytest.raises(ShapeError, match="no coupling"):
        _plan(keep_parity=((0, -1),), Lmax=0)


def test_options_from_a_yaml():
    yaml_args = {
        "lmax": 4,
        "Lmax": 3,
        "keep_parity": [[0, 1], [1, -1]],
        "avg_n_neigh": 39.5,
        "normalize": True,
        "radia_basis": None,
        "angular": {"name": "Y"},
    }
    assert EquivariantIndOptions.from_options(yaml_args) == EquivariantIndOptions(
        lmax=4, Lmax=3, keep_parity=((0, 1), (1, -1)), normalize=True, avg_n_neigh=39.5
    )
    defaults = EquivariantIndOptions.from_options({
        "lmax": 1,
        "Lmax": 1,
        "history_drop_list": [["(1-)", 0]],
    })
    assert defaults == EquivariantIndOptions(
        lmax=1, Lmax=1, history_drop_list=(("(1-)", 0),)
    )


def test_inputs_are_not_modified():
    angular, indicator = _angular(2), INDICATOR.copy()
    before = (angular.copy(), indicator.copy())
    spbf_equivariant_ind_plan(
        angular, indicator, EquivariantIndOptions(lmax=2, Lmax=2), radial_lmax=2
    )
    pd.testing.assert_frame_equal(angular, before[0])
    pd.testing.assert_frame_equal(indicator, before[1])


# ------------------------------------------------------------------ physical values


def _harmonics(direction: np.ndarray, lmax: int) -> np.ndarray:
    unit = direction / np.linalg.norm(direction)
    return real_spherical_harmonics(
        torch.as_tensor(unit), spherical_harmonics_plan(lmax)
    ).numpy()


def _couple(plan, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``out[f] = sum_k cg[k] Y(u)[lr_inds[k, 0]] I(v)[lr_inds[k, 1]]``, the indicator being the harmonics of ``v``."""
    terms = (
        plan.cg
        * _harmonics(u, 2)[plan.lr_inds[:, 0]]
        * _harmonics(v, 1)[plan.lr_inds[:, 1]]
    )
    return np.bincount(plan.m_sum_ind, weights=terms, minlength=plan.nfunc)


def test_coupled_functions_are_rotation_equivariant():
    rng = np.random.default_rng(RNG_SEED)
    plan = _plan(keep_parity=((0, 1), (1, 1), (1, -1), (2, 1), (2, -1)))
    u, v = rng.normal(size=3), rng.normal(size=3)
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    before, after = _couple(plan, u, v), _couple(plan, rotation @ u, rotation @ v)
    meta = plan.coupling_meta_data
    for _, block in meta.groupby(["l", "parity", "hist"]):
        rows = block.index.to_numpy()
        assert np.linalg.norm(before[rows]) > 1e-3
        np.testing.assert_allclose(
            np.linalg.norm(after[rows]), np.linalg.norm(before[rows]), rtol=1e-12
        )
    scalars = (meta["l"] == 0).to_numpy()
    np.testing.assert_allclose(after[scalars], before[scalars], rtol=1e-12)


def test_the_scalar_of_two_vectors_is_their_dot_product():
    rng = np.random.default_rng(RNG_SEED)
    plan = _plan()
    meta = plan.coupling_meta_data
    (row,) = meta.index[(meta["l"] == 0) & (meta["hist"] == "(1-,1-)")]
    ratios = []
    for _ in range(4):
        u, v = rng.normal(size=3), rng.normal(size=3)
        cosine = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
        ratios.append(_couple(plan, u, v)[row] / cosine)
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    assert abs(ratios[0]) > 1e-3
