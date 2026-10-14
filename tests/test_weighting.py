"""Tests of ``tensorpotential/data/weighting.py``: ``EnergyBasedWeightingPolicy`` and ``check_df_non_empty``.

Logic layer: the option combinations of ``nfit``, ``n_lower`` and ``n_upper``, the reference type, the energy
reference frames, the two filters (``DEup``, ``DFup``), the errors and the messages, and the determinism of the
random selection. Physical-value layer: the weights against toy frames whose values are computed by hand: the
energy weight ``1 / (E + DE)^2`` normalised to one, the rescaling of the low tier to ``wlow``, the force weight
``1 / (|f|^2 + DF)`` per atom times the weight of its structure and normalised to one over all atoms, and the
virial weight equal to the energy weight.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from tensorpotential.constants import (
    DATA_ENERGY_WEIGHTS,
    DATA_FORCE_WEIGHTS,
    DATA_VIRIAL_WEIGHTS,
)
from tensorpotential.data.process_df import (
    E_CHULL_DIST_PER_ATOM,
    E_CORRECTED_PER_ATOM_COLUMN,
    ELEMENTAL_REF_ENERGIES_DICT,
    ENERGY_CORRECTED_COL,
    FORCES_COL,
    NUMBER_OF_ATOMS,
)
from tensorpotential.data.weighting import (
    EnergyBasedWeightingPolicy,
    check_df_non_empty,
)
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH


def _frame(
    hull: list[float],
    pbc: list[bool] | None = None,
    forces: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """A frame of Cu2 structures: ``hull`` is the distance to the convex hull per atom, ``forces`` one (2, 3) array each."""
    n = len(hull)
    rng = np.random.default_rng(0)
    pbc = pbc if pbc is not None else [True] * n
    forces = (
        forces if forces is not None else [rng.normal(size=(2, 3)) for _ in range(n)]
    )
    atoms = [
        Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[5] * 3, pbc=p)
        for p in pbc
    ]
    return pd.DataFrame({
        "ase_atoms": atoms,
        "pbc": pbc,
        E_CHULL_DIST_PER_ATOM: hull,
        E_CORRECTED_PER_ATOM_COLUMN: [-3.0 + h for h in hull],
        ENERGY_CORRECTED_COL: [2 * (-3.0 + h) for h in hull],
        FORCES_COL: pd.Series(forces, dtype=object),
        NUMBER_OF_ATOMS: [2] * n,
    })


def _weights(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return EnergyBasedWeightingPolicy(**kwargs).generate_weights(frame.copy())


# --- check_df_non_empty ---


def test_an_empty_frame_is_refused_with_a_hint_about_the_filters():
    with pytest.raises(
        RuntimeError,
        match=r"Couldn't operate with empty dataset\. Try to reduce filters",
    ):
        check_df_non_empty(pd.DataFrame())


def test_a_non_empty_frame_passes():
    assert check_df_non_empty(pd.DataFrame({"a": [1]})) is None


# --- the constructor and its text ---


def test_the_constructor_keeps_its_arguments_and_the_defaults():
    p = EnergyBasedWeightingPolicy()
    assert (p.nfit, p.cutoff, p.DElow, p.DEup, p.DE, p.DF) == (
        None,
        None,
        1.0,
        10.0,
        1.0,
        1.0,
    )
    assert (p.wlow, p.DFup, p.n_lower, p.n_upper) == (None, None, None, None)
    assert (p.reftype, p.seed, p.energy) == ("all", 42, "convex_hull")


@pytest.mark.parametrize("text", ["none", "None", "NONE"])
def test_the_string_none_for_wlow_means_no_adjustment(text):
    assert EnergyBasedWeightingPolicy(wlow=text).wlow is None


def test_a_numeric_wlow_is_kept_as_given():
    assert EnergyBasedWeightingPolicy(wlow=0.75).wlow == 0.75
    assert (
        EnergyBasedWeightingPolicy(wlow="0.75").wlow == "0.75"
    )  # converted to float only when the weights are made


def test_the_string_form_lists_every_selection_setting():
    text = str(
        EnergyBasedWeightingPolicy(
            nfit=5,
            DElow=0.5,
            DEup=4.0,
            DFup=9.0,
            DE=2.0,
            DF=3.0,
            wlow=0.5,
            reftype="bulk",
            seed=1,
        )
    )
    assert text == (
        "EnergyBasedWeightingPolicy(nfit=5, n_lower=None, n_upper=None, energy=convex_hull,"
        " DElow=0.5, DEup=4.0, DFup=9.0, DE=2.0, DF=3.0, wlow=0.5, reftype=bulk,seed=1)"
    )


# --- nfit, n_lower and n_upper ---


def test_without_any_count_the_whole_dataset_is_fitted():
    policy = EnergyBasedWeightingPolicy()
    result = policy.generate_weights(_frame([0.1, 0.2, 0.3]))
    assert policy.nfit == 3
    assert len(result) == 3


def test_both_tier_counts_set_nfit_to_their_sum():
    policy = EnergyBasedWeightingPolicy(n_lower=2, n_upper=1)
    policy.generate_weights(_frame([0.1, 0.2, 0.3, 2.0, 3.0]))
    assert policy.nfit == 3


@pytest.mark.parametrize("counts", [{"n_lower": 2}, {"n_upper": 2}])
def test_a_single_tier_count_without_nfit_is_refused(counts):
    with pytest.raises(
        ValueError, match=r"nfit is None\. Please provide both n_lower and n_upper"
    ):
        EnergyBasedWeightingPolicy(**counts).generate_weights(_frame([0.1, 0.2]))


@pytest.mark.parametrize(
    "counts", [{"n_lower": 1}, {"n_upper": 1}, {"n_lower": 1, "n_upper": 1}]
)
def test_nfit_together_with_a_tier_count_is_refused(counts):
    with pytest.raises(
        ValueError, match=r"nfit is not None\. No n_lower or n_upper is expected"
    ):
        EnergyBasedWeightingPolicy(nfit=3, **counts).generate_weights(
            _frame([0.1, 0.2])
        )


# --- reference type ---


def test_bulk_keeps_the_periodic_structures_and_cluster_the_others():
    frame = _frame([0.1, 0.2, 0.3, 0.4], pbc=[True, False, True, False])
    assert sorted(_weights(frame, reftype="bulk").index) == [0, 2]
    assert sorted(_weights(frame, reftype="cluster").index) == [1, 3]
    assert sorted(_weights(frame, reftype="all").index) == [0, 1, 2, 3]


def test_a_reference_type_that_leaves_nothing_is_refused():
    with pytest.raises(RuntimeError, match="empty dataset"):
        _weights(_frame([0.1, 0.2], pbc=[False, False]), reftype="bulk")


# --- the energy reference frame ---


def test_the_convex_hull_frame_uses_the_precomputed_distance():
    result = _weights(_frame([0.0, 0.5, 0.25]))
    assert sorted(result["effective_energy"]) == [0.0, 0.25, 0.5]


def test_the_cohesive_frame_counts_from_the_lowest_energy_per_atom():
    frame = _frame([0.4, 0.1, 0.7])
    result = _weights(frame, energy="cohesive")
    expected = (
        frame[E_CORRECTED_PER_ATOM_COLUMN] - frame[E_CORRECTED_PER_ATOM_COLUMN].min()
    )
    np.testing.assert_allclose(
        result["effective_energy"].sort_index(),
        expected,
        rtol=ARITH.rtol,
        atol=ARITH.atol,
    )


def test_the_zero_formation_energy_frame_clips_negative_formation_energies_and_shifts_to_zero():
    # Cu2 structures: formation energy per atom = E_per_atom - the elemental reference energy of Cu
    reference = ELEMENTAL_REF_ENERGIES_DICT["Cu"]
    per_atom = [reference - 0.5, reference + 0.2, reference + 0.9]
    frame = _frame([0.0, 0.0, 0.0])
    frame[ENERGY_CORRECTED_COL] = [2 * e for e in per_atom]
    result = _weights(frame, energy="zero_formation_energy", DEup=100.0, DElow=0.5)
    # the negative one becomes 0 before the shift, the minimum is then 0 already
    np.testing.assert_allclose(
        result["effective_energy"].sort_index(),
        [0.0, 0.2, 0.9],
        rtol=ARITH.rtol,
        atol=ARITH.atol,
    )


def test_an_unknown_energy_frame_is_refused_with_the_possible_values():
    with pytest.raises(
        ValueError,
        match=r"Unknown EnergyBasedWeightingPolicy.energy=bogus settings. Possible values",
    ):
        _weights(_frame([0.1]), energy="bogus")


def test_a_given_cutoff_is_reported_as_ignored(caplog):
    with caplog.at_level(logging.INFO):
        _weights(_frame([0.1, 0.2]), cutoff=5.0)
    assert "cutoff is provided but will be ignored" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _weights(_frame([0.1, 0.2]))
    assert "No cutoff for EnergyBasedWeightingPolicy is provided" in caplog.text


# --- the filters ---


def test_structures_at_or_above_the_upper_energy_are_removed():
    result = _weights(_frame([0.1, 2.0, 5.0, 9.99, 10.0, 12.0]), DEup=10.0)
    assert sorted(result.index) == [0, 1, 2, 3]


def test_a_frame_with_nothing_below_the_upper_energy_is_refused():
    with pytest.raises(RuntimeError, match="empty dataset"):
        _weights(_frame([20.0, 30.0]), DEup=10.0)


def test_structures_with_a_force_above_dfup_are_removed():
    frame = _frame(
        [0.1, 0.2, 0.3],
        forces=[
            np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]),  # longest force vector: 5
            np.array([[6.0, 8.0, 0.0], [0.0, 0.0, 0.0]]),  # 10
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),  # 0.5
        ],
    )
    assert sorted(_weights(frame, DFup=5.0).index) == [
        0,
        2,
    ]  # a force of exactly DFup stays
    assert sorted(_weights(frame, DFup=None).index) == [0, 1, 2]


def test_dfup_that_removes_everything_is_refused():
    with pytest.raises(RuntimeError, match="empty dataset"):
        _weights(_frame([0.1, 0.2], forces=[np.full((2, 3), 100.0)] * 2), DFup=1e-6)


# --- the selection of structures ---

LOW = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # below DElow = 1
UP = [2.0, 3.0, 4.0, 5.0]  # between DElow and DEup


def _energies(result: pd.DataFrame) -> list[float]:
    return list(result["effective_energy"])


def test_a_dataset_smaller_than_nfit_is_taken_whole():
    assert len(_weights(_frame(LOW + UP), nfit=100)) == 10


def test_nfit_below_the_low_tier_keeps_nfit_low_structures_only():
    result = _weights(_frame(LOW + UP), nfit=4)
    assert len(result) == 4
    assert all(e < 1.0 for e in _energies(result))


def test_nfit_fills_the_remainder_from_the_upper_tier():
    result = _weights(_frame(LOW[:2] + UP), nfit=4)
    energies = _energies(result)
    assert len(result) == 4
    assert sum(e < 1.0 for e in energies) == 2 and sum(e >= 1.0 for e in energies) == 2


def test_nfit_larger_than_the_candidates_takes_all_the_upper_structures():
    result = _weights(_frame(LOW[:2] + UP[:1]), nfit=50)
    assert len(result) == 3


def test_the_tier_counts_pick_that_many_structures_of_each_tier():
    result = _weights(_frame(LOW + UP), n_lower=3, n_upper=2)
    energies = _energies(result)
    assert (sum(e < 1.0 for e in energies), sum(e >= 1.0 for e in energies)) == (3, 2)


def test_tier_counts_larger_than_the_tiers_take_whole_tiers():
    result = _weights(_frame(LOW[:2] + UP[:1]), n_lower=5, n_upper=5)
    assert len(result) == 3


def test_the_same_seed_gives_the_same_selection_and_another_seed_another_one():
    frame = _frame(LOW + UP + [0.15, 0.25, 0.35, 0.45, 6.0, 7.0])
    first = _weights(frame, nfit=5, seed=1)
    again = _weights(frame, nfit=5, seed=1)
    other = _weights(frame, nfit=5, seed=2)
    assert list(first.index) == list(again.index)
    assert list(first.index) != list(other.index)


def test_the_order_of_the_selected_structures_is_shuffled_with_the_seed():
    frame = _frame(LOW)
    orders = {tuple(_weights(frame, seed=s).index) for s in range(6)}
    assert len(orders) > 1
    assert all(sorted(o) == list(range(6)) for o in orders)


# Selected rows in output order, taken from the code before it was split into steps (#177):
# numpy's legacy ``RandomState`` stream (``seed``, ``choice``, ``shuffle``) is stable across versions.
# Counts at the tier sizes (6 low, 4 upper) pin the ``<=`` boundaries: the whole tier is taken
# without drawing from the stream.
SELECTION_ORDERS = [
    ({"n_lower": 6, "n_upper": 4}, [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]),
    ({"nfit": 10}, [3, 8, 0, 6, 5, 2, 1, 7, 4, 9]),
    ({"nfit": 7, "seed": 3}, [9, 2, 1, 4, 5, 0, 3]),
    ({"n_lower": 3, "n_upper": 4, "seed": 5}, [8, 2, 7, 4, 5, 6, 9]),
    ({"n_lower": 6, "n_upper": 2, "seed": 9}, [9, 4, 1, 2, 3, 0, 7, 5]),
    ({"nfit": 8, "seed": 11}, [8, 0, 2, 3, 9, 4, 5, 1]),
]


@pytest.mark.parametrize(("kwargs", "order"), SELECTION_ORDERS)
def test_the_selection_and_its_order_follow_the_seeded_random_stream(kwargs, order):
    result = _weights(_frame(LOW + UP), **kwargs)
    assert list(result.index) == order


# --- the energy weights ---


def test_energy_weights_follow_one_over_the_squared_shifted_energy_and_sum_to_one():
    hull = [0.0, 1.0, 3.0]  # DE = 1: raw weights 1, 1/4, 1/16
    result = _weights(_frame(hull), DElow=0.5, DEup=10.0).sort_index()
    raw = np.array([1.0, 0.25, 1.0 / 16.0])
    np.testing.assert_allclose(
        result[DATA_ENERGY_WEIGHTS], raw / raw.sum(), rtol=ARITH.rtol
    )
    assert result[DATA_ENERGY_WEIGHTS].sum() == pytest.approx(1.0, rel=ARITH.rtol)


def test_the_energy_offset_de_enters_through_its_absolute_value():
    frame = _frame([0.0, 2.0])
    plus = _weights(frame, DE=2.0, DElow=0.5)[DATA_ENERGY_WEIGHTS].sort_index()
    minus = _weights(frame, DE=-2.0, DElow=0.5)[DATA_ENERGY_WEIGHTS].sort_index()
    np.testing.assert_allclose(plus, minus, rtol=ARITH.rtol)
    raw = np.array([1 / 4, 1 / 16])
    np.testing.assert_allclose(plus, raw / raw.sum(), rtol=ARITH.rtol)


def test_the_virial_weight_equals_the_energy_weight():
    result = _weights(_frame([0.0, 0.5, 2.0]))
    np.testing.assert_array_equal(
        result[DATA_VIRIAL_WEIGHTS], result[DATA_ENERGY_WEIGHTS]
    )


# --- the tier weights wlow ---


def test_wlow_rescales_the_low_tier_to_its_share_and_the_upper_tier_to_the_rest():
    result = _weights(_frame([0.0, 0.5, 2.0, 3.0]), DElow=1.0, wlow=0.75).sort_index()
    weights = result[DATA_ENERGY_WEIGHTS]
    assert weights[[0, 1]].sum() == pytest.approx(0.75, rel=ARITH.rtol)
    assert weights[[2, 3]].sum() == pytest.approx(0.25, rel=ARITH.rtol)
    # inside a tier the proportions of 1 / (E + DE)^2 stay
    assert weights[0] / weights[1] == pytest.approx(
        (0.5 + 1.0) ** 2 / (0.0 + 1.0) ** 2, rel=ARITH.rtol
    )
    assert weights[2] / weights[3] == pytest.approx(
        (3.0 + 1.0) ** 2 / (2.0 + 1.0) ** 2, rel=ARITH.rtol
    )


def test_wlow_given_as_a_string_is_converted():
    result = _weights(_frame([0.0, 2.0]), DElow=1.0, wlow="0.5")
    assert result[DATA_ENERGY_WEIGHTS].sort_index()[0] == pytest.approx(
        0.5, rel=ARITH.rtol
    )


def test_with_only_low_structures_wlow_is_raised_to_one_with_a_warning(caplog):
    policy = EnergyBasedWeightingPolicy(wlow=0.6)
    with caplog.at_level(logging.WARNING):
        result = policy.generate_weights(_frame([0.0, 0.5]))
    assert "All structures were taken from low-tier" in caplog.text
    assert policy.wlow == 1.0  # pinned: the policy object is changed by the call
    assert result[DATA_ENERGY_WEIGHTS].sum() == pytest.approx(1.0, rel=ARITH.rtol)
    assert "No weights adjustment possible" in caplog.text


def test_wlow_of_one_with_only_low_structures_does_not_warn_about_the_tier(caplog):
    with caplog.at_level(logging.WARNING):
        _weights(_frame([0.0, 0.5]), wlow=1.0)
    assert "All structures were taken from low-tier" not in caplog.text


def test_only_upper_structures_cannot_be_rescaled(caplog):
    with caplog.at_level(logging.WARNING):
        result = _weights(_frame([2.0, 3.0]), DElow=1.0, wlow=0.5)
    assert "No weights adjustment possible" in caplog.text
    raw = np.array([1 / 9.0, 1 / 16.0])
    np.testing.assert_allclose(
        result[DATA_ENERGY_WEIGHTS].sort_index(), raw / raw.sum(), rtol=ARITH.rtol
    )


def test_without_wlow_the_tiers_are_not_rescaled(caplog):
    with caplog.at_level(logging.WARNING):
        _weights(_frame([0.0, 2.0]))
    assert "wlow=None, no weights adjustment" in caplog.text


# --- the force weights ---


def test_force_weights_follow_one_over_force_squared_plus_df_times_the_energy_weight():
    frame = _frame(
        [0.0, 3.0],
        forces=[
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),  # |f|^2 = 1, 4
            np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]),  # |f|^2 = 9, 0
        ],
    )
    result = _weights(frame, DF=1.0, DElow=0.5).sort_index()
    e_raw = np.array([1.0, 1.0 / 16.0])
    e_w = e_raw / e_raw.sum()
    per_atom = [
        np.array([1 / 2.0, 1 / 5.0]) * e_w[0],
        np.array([1 / 10.0, 1 / 1.0]) * e_w[1],
    ]
    total = sum(p.sum() for p in per_atom)
    for found, expected in zip(result[DATA_FORCE_WEIGHTS], per_atom, strict=True):
        np.testing.assert_allclose(found, expected / total, rtol=ARITH.rtol)
    assert sum(w.sum() for w in result[DATA_FORCE_WEIGHTS]) == pytest.approx(
        1.0, rel=ARITH.rtol
    )


def test_the_force_offset_df_enters_through_its_absolute_value():
    frame = _frame([0.0, 1.0])
    plus = _weights(frame, DF=2.0)[DATA_FORCE_WEIGHTS].sort_index()
    minus = _weights(frame, DF=-2.0)[DATA_FORCE_WEIGHTS].sort_index()
    for a, b in zip(plus, minus, strict=True):
        np.testing.assert_allclose(a, b, rtol=ARITH.rtol)


def test_forces_are_converted_to_arrays_and_must_match_the_atom_count():
    frame = _frame([0.0, 1.0])
    frame[FORCES_COL] = frame[FORCES_COL].map(lambda f: f.tolist())
    result = _weights(frame)
    assert all(isinstance(f, np.ndarray) for f in result[FORCES_COL])

    # three force vectors for two atoms
    broken = _frame([0.0, 1.0], forces=[np.zeros((3, 3)), np.zeros((2, 3))])
    with pytest.raises(
        AssertionError, match="Number of atoms doesn't corresponds to shape of forces"
    ):
        _weights(broken)


# --- the arguments ---


def test_the_input_frame_gains_the_effective_energy_column_when_nothing_is_filtered():
    # pinned: the policy writes into the frame it is given (a copy is made only by a filter)
    frame = _frame([0.0, 0.5])
    EnergyBasedWeightingPolicy().generate_weights(frame)
    assert "effective_energy" in frame.columns
    assert DATA_ENERGY_WEIGHTS not in frame.columns


def test_the_weights_leave_the_input_values_unchanged():
    frame = _frame([0.0, 0.5, 2.0])
    before = frame[[E_CHULL_DIST_PER_ATOM, ENERGY_CORRECTED_COL]].copy()
    EnergyBasedWeightingPolicy().generate_weights(frame)
    pd.testing.assert_frame_equal(
        frame[[E_CHULL_DIST_PER_ATOM, ENERGY_CORRECTED_COL]], before
    )
