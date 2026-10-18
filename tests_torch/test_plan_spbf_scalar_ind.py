"""The ``SingleParticleBasisFunctionScalarInd`` plan (``plans.spbf_scalar_ind``): logic on the harmonics table.

``lmax`` from the option or the angular input, the angular cut and its coupling table, ``l_tile``, the float and
per-element forms of ``1 / avg_n_neigh`` (a non-positive count gives 1), and the errors TF raises as assertions or
``TypeError``. The comparison of the derived values with TensorFlow is the golden test of the twin.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.plans.spbf_scalar_ind import (
    inverse_avg_n_neigh,
    spbf_scalar_ind_plan,
)
from tests_torch.coupling_frames import uncoupled

HARMONICS = uncoupled(3)


def test_a_null_lmax_takes_the_degree_of_the_angular_input():
    plan = spbf_scalar_ind_plan(HARMONICS, lmax=None, radial_lmax=3, avg_n_neigh=4.0)
    assert plan.lmax == 3
    assert plan.slice_angular is None
    assert plan.coupling_meta_data is HARMONICS
    assert plan.l_tile.tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    assert plan.l_tile.dtype == np.int64


def test_a_smaller_lmax_cuts_the_angular_columns_and_the_table():
    plan = spbf_scalar_ind_plan(HARMONICS, lmax=1, radial_lmax=1, avg_n_neigh=4.0)
    assert plan.slice_angular == 4
    assert plan.coupling_meta_data["l"].tolist() == [0, 1, 1, 1]
    assert plan.coupling_meta_data.equals(HARMONICS.iloc[:4])
    assert plan.l_tile.tolist() == [0, 1, 1, 1]


def test_an_lmax_equal_to_the_angular_degree_cuts_nothing():
    plan = spbf_scalar_ind_plan(HARMONICS, lmax=3, radial_lmax=3, avg_n_neigh=4.0)
    assert plan.slice_angular is None


def test_the_plan_is_frozen():
    plan = spbf_scalar_ind_plan(HARMONICS, lmax=None, radial_lmax=3, avg_n_neigh=4.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.lmax = 2  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("lmax", "radial_lmax", "match"),
    [
        (4, 4, "larger than the degree of the angular input"),
        (None, 2, r"radial part \(lmax=2\) and the instruction \(lmax=3\)"),
        (2, 3, r"radial part \(lmax=3\) and the instruction \(lmax=2\)"),
    ],
)
def test_degrees_that_do_not_fit_are_refused(lmax, radial_lmax, match):
    with pytest.raises(OptionValueError, match=match):
        spbf_scalar_ind_plan(
            HARMONICS, lmax=lmax, radial_lmax=radial_lmax, avg_n_neigh=4.0
        )


def test_a_float_average_is_inverted_to_a_scalar():
    inv = inverse_avg_n_neigh(4.0)
    assert inv.shape == () and inv.dtype == np.float64 and float(inv) == 0.25
    # a numpy float is a float, as in TF
    assert float(inverse_avg_n_neigh(np.float64(8.0))) == 0.125
    plan = spbf_scalar_ind_plan(HARMONICS, lmax=None, radial_lmax=3, avg_n_neigh=4.0)
    assert not plan.per_element_average


def test_a_dict_average_is_one_factor_per_element_and_one_where_the_count_is_not_positive():
    inv = inverse_avg_n_neigh({1: 4.0, 0: 2.0, 2: 0.0, 3: -1.0})
    assert inv.shape == (4, 1)
    assert inv[:, 0].tolist() == [0.5, 0.25, 1.0, 1.0]
    plan = spbf_scalar_ind_plan(
        HARMONICS, lmax=None, radial_lmax=3, avg_n_neigh={0: 2.0}
    )
    assert plan.per_element_average


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (40, "must be a float or a dict"),
        (True, "must be a float or a dict"),
        ("40.0", "must be a float or a dict"),
        ({0: 1.0, 2: 1.0}, r"element indices 0 \.\. 1"),
    ],
)
def test_other_averages_are_refused(value, match):
    with pytest.raises(OptionValueError, match=match):
        inverse_avg_n_neigh(value)
