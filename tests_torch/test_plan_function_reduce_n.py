"""The ``FunctionReduceN`` plan (``plans.function_reduce_n``): logic on hand-made tables, physics on real harmonics.

Logic: the output layout and its order, the grouping by ``(parity, l, hist)`` in ascending key order, the selection by
``ls_max`` and ``allowed_l_p``, the destination rows, the norms, the weight shapes with and without the element axis,
``only_invar``, the runtime ``norm_map``, the options read from a yaml, the errors. Physical values: a reduction of two
sources of real spherical harmonics with the plan's tables and random weights is rotation equivariant (its ``l = 0``
outputs are invariant, the norm of each ``l = 1`` block too), which fails as soon as a weight is shared across the wrong
entries or an entry lands on the wrong row. The equality with TensorFlow's tables is in
``test_plans_spbf_reduce_against_tf.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.spherical_harmonics import (
    real_spherical_harmonics,
)
from tensorpotential.torch_backend.plans.function_reduce_n import (
    FunctionReduceNOptions,
    ReduceSource,
    function_reduce_n_plan,
    output_layout,
    per_source,
)
from tensorpotential.torch_backend.plans.spherical_harmonics import (
    spherical_harmonics_plan,
)

RNG_SEED = 20261016
COLUMNS = ["l", "m", "hist", "parity", "sum_of_ls"]


def _table(blocks: list[tuple[int, str, int]]) -> pd.DataFrame:
    """A coupling table with one row per ``m`` of each ``(l, hist, parity)`` block, in the given order."""
    rows = [(l, m, h, p, l) for l, h, p in blocks for m in range(-l, l + 1)]
    return pd.DataFrame(rows, columns=pd.Index(COLUMNS))


def _options(**changes: object) -> FunctionReduceNOptions:
    base = FunctionReduceNOptions(ls_max=(1,), allowed_l_p=((0, 1), (1, -1)), n_out=3)
    return dataclasses.replace(base, **changes)


# ------------------------------------------------------------------ logic


def test_output_layout_is_sorted_by_l_then_parity_then_m():
    layout = output_layout(2, [(2, 1), (1, 1), (0, 1), (1, -1), (3, -1)])
    assert list(layout.columns) == COLUMNS
    assert list(zip(layout["l"], layout["parity"], layout["m"], strict=True)) == [
        (0, 1, 0),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
        *[(2, 1, m) for m in range(-2, 3)],
    ]
    assert set(layout["hist"]) == {""}
    assert (layout["sum_of_ls"] == layout["l"]).all()
    assert list(layout.index) == list(range(len(layout)))


def test_per_source_repeats_an_int_and_keeps_a_list():
    assert per_source(2, 3) == (2, 2, 2)
    assert per_source([1, 0], 2) == (1, 0)


def test_groups_are_kept_in_ascending_parity_l_hist_order():
    meta = _table([
        (1, "b", -1),
        (0, "a", 1),
        (0, "", 1),
        (1, "a", -1),
        (2, "", 1),
        (1, "", 1),
    ])
    plan = function_reduce_n_plan([ReduceSource("S", meta, 4)], _options())
    (c,) = plan.collectors
    # rows: (1,b,-) 0-2, (0,a,+) 3, (0,'',+) 4, (1,a,-) 5-7, (2,'',+) 8-12 (l > ls_max), (1,'',+) 13-15 (not allowed)
    assert c.func_collect_ind.tolist() == [5, 6, 7, 0, 1, 2, 4, 3]
    assert c.func_collect_ind.dtype == np.int64
    assert c.w_l_tile.tolist() == [0, 0, 0, 1, 1, 1, 2, 3]
    assert c.w_l_tile.dtype == np.int32
    assert c.w_shape == 4
    # output rows: (0,+,0) -> 0, (1,-,m) -> 1 + m + 1
    assert c.total_sum_ind.ravel().tolist() == [1, 2, 3, 1, 2, 3, 0, 0]
    assert (c.total_sum_ind.shape, c.total_sum_ind.dtype) == ((8, 1), np.int32)
    np.testing.assert_array_equal(plan.norm_map, np.full(4, 1 / 2**0.5))


def test_norm_map_counts_every_source_and_is_one_for_empty_rows():
    a = ReduceSource("A", _table([(0, "", 1)]), 2)
    b = ReduceSource("B", _table([(0, "x", 1), (0, "y", 1), (1, "", -1)]), 5)
    plan = function_reduce_n_plan(
        [a, b], _options(ls_max=(0, 2), allowed_l_p=((0, 1), (1, -1), (2, 1)))
    )
    np.testing.assert_array_equal(plan.norm_map, [1 / 3**0.5, 1, 1, 1, 1, 1, 1, 1, 1])
    assert [c.weight_shape for c in plan.collectors] == [(3, 2, 1), (3, 5, 3)]
    assert len(plan.coupling_meta_data) == 9


def test_weights_norms_and_element_axis():
    source = ReduceSource("S", _table([(0, "", 1), (1, "", -1)]), 16)
    plain = function_reduce_n_plan([source], _options())
    assert (plain.collectors[0].weight_shape, plain.collectors[0].norm) == (
        (3, 16, 2),
        0.25,
    )
    scaled = function_reduce_n_plan([source], _options(scale=2.0))
    assert scaled.collectors[0].norm == 0.5
    unnormalised = function_reduce_n_plan(
        [source], _options(normalize=False, scale=2.0)
    )
    assert unnormalised.collectors[0].norm == 1.0
    per_element = function_reduce_n_plan(
        [source], _options(is_central_atom_type_dependent=True, number_of_atom_types=89)
    )
    assert per_element.collectors[0].weight_shape == (89, 3, 16, 2)


def test_only_invar_and_runtime_norm_map():
    source = ReduceSource("S", _table([(0, "", 1), (0, "a", 1), (1, "", -1)]), 2)
    invariant = function_reduce_n_plan(
        [source], _options(ls_max=(0,), allowed_l_p=((0, 1),))
    )
    assert invariant.only_invar
    assert invariant.lmax == 0
    assert invariant.runtime_norm_map.shape == ()
    assert invariant.runtime_norm_map == 1.0
    equivariant = function_reduce_n_plan([source], _options(out_norm=True))
    assert not equivariant.only_invar
    np.testing.assert_array_equal(
        equivariant.runtime_norm_map, equivariant.norm_map.reshape(-1, 1, 1)
    )


def test_options_from_a_yaml():
    yaml_args = {
        "instructions": [{"name": "A"}, {"name": "B"}],
        "ls_max": 2,
        "allowed_l_p": [[0, 1], [1, -1]],
        "n_out": 7,
        "is_central_atom_type_dependent": True,
        "number_of_atom_types": 89,
        "downscale_embedding_size": 16,
    }
    options = FunctionReduceNOptions.from_options(yaml_args)
    assert options == FunctionReduceNOptions(
        ls_max=(2, 2),
        allowed_l_p=((0, 1), (1, -1)),
        n_out=7,
        is_central_atom_type_dependent=True,
        number_of_atom_types=89,
    )
    assert FunctionReduceNOptions.from_options({
        **yaml_args,
        "ls_max": [1, 0],
    }).ls_max == (1, 0)


def test_inputs_are_not_modified():
    meta = _table([(1, "b", -1), (0, "", 1)])
    before = meta.copy()
    function_reduce_n_plan([ReduceSource("S", meta, 2)], _options())
    pd.testing.assert_frame_equal(meta, before)


@pytest.mark.parametrize(
    ("sources", "options", "match"),
    [
        (["S", "S"], _options(ls_max=(1, 1)), "distinct names"),
        (["S", "T"], _options(), "one per source"),
        (
            ["S"],
            _options(is_central_atom_type_dependent=True),
            "needs number_of_atom_types",
        ),
    ],
)
def test_option_errors(sources, options, match):
    meta = _table([(0, "", 1)])
    with pytest.raises(OptionValueError, match=match):
        function_reduce_n_plan([ReduceSource(n, meta, 1) for n in sources], options)


def test_a_source_without_collected_entries_is_a_typed_error():
    with pytest.raises(ShapeError, match="source 'S' has no entry"):
        function_reduce_n_plan([ReduceSource("S", _table([(2, "", 1)]), 1)], _options())


# ------------------------------------------------------------------ physical values


def _harmonics(directions: np.ndarray, lmax: int) -> np.ndarray:
    plan = spherical_harmonics_plan(lmax)
    return real_spherical_harmonics(torch.as_tensor(directions), plan).numpy()


def _uncoupled(lmax: int, hist: str) -> pd.DataFrame:
    return _table([(l, hist, 1 if l % 2 == 0 else -1) for l in range(lmax + 1)])


def _reduce(plan, weights, features) -> np.ndarray:
    """The forward pass of TensorFlow (spec sheet section 5) on one atom: ``[n_out, n_rows]``."""
    out = np.zeros((plan.collectors[0].weight_shape[0], len(plan.coupling_meta_data)))
    for c, w, x in zip(plan.collectors, weights, features, strict=True):
        a = x[:, c.func_collect_ind]  # [n_in, n_collected]
        pr = np.einsum("knw,nw->wk", w[:, :, c.w_l_tile], a) * c.norm
        np.add.at(out.T, c.total_sum_ind.ravel(), pr)
    return out * plan.runtime_norm_map.reshape(-1)


def test_reduction_of_harmonics_is_rotation_equivariant():
    rng = np.random.default_rng(RNG_SEED)
    sources = [
        ReduceSource("A", _uncoupled(2, ""), 3),
        ReduceSource("B", _uncoupled(1, "(x)"), 2),
    ]
    plan = function_reduce_n_plan(
        sources,
        _options(ls_max=(2, 1), allowed_l_p=((0, 1), (1, -1), (2, 1)), out_norm=True),
    )
    weights = [rng.normal(size=c.weight_shape) for c in plan.collectors]
    directions = rng.normal(size=(5, 3))
    mixing = [rng.normal(size=(s.n_out, len(directions))) for s in sources]

    def output(dirs: np.ndarray) -> np.ndarray:
        unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        y = _harmonics(unit, 2)
        features = [
            m @ y[:, : len(s.coupling_meta_data)]
            for m, s in zip(mixing, sources, strict=True)
        ]
        return _reduce(plan, weights, features)

    before = output(directions)
    after = output(directions @ Rotation.random(random_state=RNG_SEED).as_matrix().T)
    layout = plan.coupling_meta_data
    for l in (0, 1, 2):
        rows = (layout["l"] == l).to_numpy()
        np.testing.assert_allclose(
            np.linalg.norm(after[:, rows], axis=1),
            np.linalg.norm(before[:, rows], axis=1),
            rtol=1e-12,
        )
    np.testing.assert_allclose(after[:, 0], before[:, 0], rtol=1e-12)
    assert np.abs(before[:, 1:4]).max() > 1e-3
