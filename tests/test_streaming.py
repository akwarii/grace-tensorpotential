"""``data/streaming.py``: the bucket choice, growth and outlier handling of ``ElasticBatchIterator._pad_batch``.

Logic layer: the empty batch, the first bucket that fits, the three outlier strategies, the
unknown strategy, the relative / absolute / per-axis growth, the cap of the metric axis, the log
lines and the padding statistics. Value layer: the bucket sizes after growth against the closed
form ``max(required, int(current * (1 + frac)))`` (``current + step`` for a step of one or more)
evaluated by hand from the real atom and neighbour counts of the structures.
"""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pytest

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
)
from tensorpotential.data.streaming import (
    ElasticBatchIterator,
    MultiBinMetricBatcher,
    StreamingConfig,
    StreamingDatasetWrapper,
)

DATA = os.path.join(os.path.dirname(__file__), "data", "MoNbTaW_train50.pkl.gz")
ATOMS, BONDS, STRUCTS = (
    tc.PAD_MAX_N_ATOMS,
    tc.PAD_MAX_N_NEIGHBORS,
    tc.PAD_MAX_N_STRUCTURES,
)


@pytest.fixture(scope="module")
def builders() -> list[Any]:
    geom = GeometricalDataBuilder(
        elements_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3},
        cutoff=5.0,
        is_fit_stress=False,
    )
    ref = ReferenceEnergyForcesStressesDataBuilder(
        normalize_weights=False, is_fit_stress=False
    )
    return [geom, ref]


@pytest.fixture(scope="module")
def structures(builders) -> list[dict[str, Any]]:
    """Per-structure dicts of the first rows of the MoNbTaW set, smallest first."""
    frame = pd.read_pickle(DATA).iloc[:12]
    out = []
    for i in range(len(frame)):
        data: dict[str, Any] = {}
        for builder in builders:
            data.update(
                builder.extract_from_row(
                    frame.iloc[i], **{tc.DATA_STRUCTURE_ID: frame.index[i]}
                )
            )
        out.append(data)
    return sorted(out, key=_real_atoms)


def _real_atoms(structure: dict[str, Any]) -> int:
    return int(structure[tc.N_ATOMS_BATCH_REAL])


def _real_bonds(structure: dict[str, Any]) -> int:
    return int(structure[tc.N_NEIGHBORS_REAL])


def _bucket(atoms: int, bonds: int, structs: int) -> dict[str, int]:
    return {ATOMS: atoms, BONDS: bonds, STRUCTS: structs}


def _iterator(builders, buckets, **kwargs) -> ElasticBatchIterator:
    """An iterator whose metric axis is the structure count with a target of one slot, so the
    cap on the metric axis never raises a bucket above the required size."""
    batcher = MultiBinMetricBatcher(num_bins=1, target_metric=10**9)
    options = {"metric_strategy": "structures", "target_metric": 1, **kwargs}
    return ElasticBatchIterator(
        structure_iter=iter(()),
        batcher=batcher,
        data_builders=builders,
        buckets=buckets,
        **options,
    )


def _pad(iterator: ElasticBatchIterator, group: list[dict[str, Any]]) -> dict[str, Any]:
    batch = iterator._pad_batch(group)
    assert batch is not None
    return batch


def test_an_empty_group_gives_no_batch(builders):
    assert _iterator(builders, [])._pad_batch([]) is None


def test_a_group_goes_to_the_first_bucket_of_the_list_that_fits(builders, structures):
    one = structures[:1]
    atoms, bonds = _real_atoms(one[0]), _real_bonds(one[0])
    buckets = [
        _bucket(atoms - 1, bonds, 2),  # too few atoms
        _bucket(atoms + 7, bonds + 3, 2),  # first that fits
        _bucket(atoms + 1, bonds, 2),  # also fits, listed later
    ]
    iterator = _iterator(builders, buckets)
    batch = _pad(iterator, one)
    assert int(batch[tc.N_ATOMS_BATCH_TOTAL]) == atoms + 7
    assert int(batch[tc.N_STRUCTURES_BATCH_TOTAL]) == 2
    assert iterator.buckets == buckets  # nothing grew


def test_a_group_with_more_structures_than_any_bucket_slot_grows_a_bucket(
    builders, structures
):
    iterator = _iterator(builders, [_bucket(10**6, 10**6, 2)])
    iterator._pad_batch(structures[:2])
    assert len(iterator.buckets) == 1  # two structures fit the two slots
    iterator._pad_batch(structures[:3])
    assert len(iterator.buckets) == 2


def test_the_first_bucket_is_the_real_size_plus_the_padding_slot(
    builders, structures, caplog
):
    group = structures[:2]
    atoms = sum(map(_real_atoms, group))
    bonds = sum(map(_real_bonds, group))
    iterator = _iterator(builders, [])
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        batch = _pad(iterator, group)
    # +1 atom and +1 structure for the padding slot; bonds exactly
    assert iterator.buckets == [_bucket(atoms + 1, bonds, 3)]
    assert int(batch[tc.N_ATOMS_BATCH_TOTAL]) == atoms + 1
    assert any("grew: atoms 0->" in r.message for r in caplog.records)


def _grown(current: int, required: int, frac: float | int) -> int:
    """Closed form of the growth along one axis."""
    if required <= current:
        return current
    if frac >= 1:
        return max(required, current + int(frac))
    return max(required, int(current * (1 + frac)))


GROWTH_FRACTIONS = [
    0.1,
    0.5,
    3,
    {"atoms": 4, "neighbours": 0.25, "structures": 2},
]


@pytest.mark.parametrize("frac", GROWTH_FRACTIONS)
def test_a_bucket_that_is_too_small_grows_by_the_growth_fraction(
    builders, structures, frac
):
    base, big = structures[0], structures[-1]
    a0, b0 = _real_atoms(base), _real_bonds(base)
    steps = (
        frac
        if isinstance(frac, dict)
        else {"atoms": frac, "neighbours": frac, "structures": frac}
    )
    iterator = _iterator(builders, [], growth_fraction=frac)
    iterator._pad_batch([base])
    # from an empty reference bucket (all zeros): required size, or the absolute step if larger
    first = _bucket(
        _grown(0, a0 + 1, steps["atoms"]),
        _grown(0, b0, steps["neighbours"]),
        2,  # the metric axis of the helper: its target of one slot never raises the bucket
    )
    assert iterator.buckets == [first]
    iterator._pad_batch([big, base])
    grown = _bucket(
        _grown(first[ATOMS], _real_atoms(big) + a0 + 1, steps["atoms"]),
        _grown(first[BONDS], _real_bonds(big) + b0, steps["neighbours"]),
        3,
    )
    assert grown in iterator.buckets
    assert len(iterator.buckets) == 2


def test_the_structure_axis_grows_by_its_own_step_when_it_is_not_the_metric(
    builders, structures
):
    iterator = _iterator(
        builders, [], growth_fraction={"structures": 5}, metric_strategy="atoms"
    )
    iterator._pad_batch(structures[:1])
    assert iterator.buckets[0][STRUCTS] == 5  # max(2, 0 + 5)


def test_the_metric_axis_is_capped_at_the_target_metric(builders, structures):
    group = structures[:1]
    iterator = _iterator(builders, [], metric_strategy="atoms", target_metric=500)
    iterator._pad_batch(group)
    assert iterator.buckets[0][ATOMS] == 500  # max(required, target)
    assert iterator.buckets[0][BONDS] == _real_bonds(group[0])


def test_the_us_spelling_of_the_metric_strategy_is_accepted(builders, structures):
    iterator = _iterator(builders, [], metric_strategy="neighbors", target_metric=10**6)
    iterator._pad_batch(structures[:1])
    assert iterator.buckets[0][BONDS] == 10**6


def test_buckets_are_kept_sorted_by_atoms_plus_neighbours(builders, structures):
    iterator = _iterator(builders, [])
    iterator._pad_batch(structures[-3:])  # the large bucket first
    iterator._pad_batch(structures[:1])  # fits it: nothing new
    assert len(iterator.buckets) == 1
    iterator._pad_batch(structures[-6:])  # larger: a new bucket, listed last
    iterator._buckets.reverse()  # the list order is the caller's; growth re-sorts it
    iterator._pad_batch(structures[-9:])
    totals = [b[ATOMS] + b[BONDS] for b in iterator.buckets]
    assert len(totals) == 3
    assert totals == sorted(totals)


def test_an_unknown_outlier_strategy_is_refused_when_no_bucket_fits(
    builders, structures
):
    iterator = _iterator(builders, [], outlier_strategy="bogus")
    with pytest.raises(ValueError, match="Unknown outlier strategy: bogus"):
        iterator._pad_batch(structures[:1])


def test_the_error_strategy_refuses_a_structure_larger_than_the_largest_bucket(
    builders, structures
):
    iterator = _iterator(builders, [_bucket(2, 2, 2)], outlier_strategy="error")
    big = structures[-1]
    with pytest.raises(ValueError, match="exceeds largest bucket"):
        iterator._pad_batch([big])


def test_the_error_strategy_without_any_bucket_refuses_every_structure(
    builders, structures
):
    iterator = _iterator(builders, [], outlier_strategy="error")
    with pytest.raises(ValueError, match="exceeds largest bucket"):
        iterator._pad_batch(structures[:1])


def test_the_warn_skip_strategy_drops_the_outliers_and_pads_the_rest(
    builders, structures, caplog
):
    small, big = structures[0], structures[-1]
    assert _real_atoms(big) > _real_atoms(small)
    bucket = _bucket(_real_atoms(small) + 2, _real_bonds(small) + 2, 3)
    iterator = _iterator(builders, [bucket], outlier_strategy="warn_skip")
    with caplog.at_level(logging.WARNING, logger="tensorpotential.data.streaming"):
        batch = _pad(iterator, [small, big])
    assert int(batch[tc.N_STRUCTURES_BATCH_REAL]) == 1
    assert int(batch[tc.N_ATOMS_BATCH_TOTAL]) == bucket[ATOMS]
    assert any("Skipping 1 structures" in r.message for r in caplog.records)
    assert iterator.buckets == [bucket]


def test_the_warn_skip_strategy_returns_nothing_when_every_structure_is_an_outlier(
    builders, structures
):
    iterator = _iterator(builders, [_bucket(1, 1, 1)], outlier_strategy="warn_skip")
    assert iterator._pad_batch(structures[-2:]) is None


def test_a_group_whose_members_fit_but_not_together_grows_even_under_the_error_strategy(
    builders, structures
):
    one = structures[-1]
    bucket = _bucket(_real_atoms(one) + 1, _real_bonds(one) + 1, 2)
    iterator = _iterator(builders, [bucket], outlier_strategy="error")
    batch = _pad(iterator, [one, one])
    assert len(iterator.buckets) == 2
    assert int(batch[tc.N_STRUCTURES_BATCH_REAL]) == 2


def test_the_padding_statistics_count_real_and_padded_entries(builders, structures):
    group = structures[:3]
    iterator = _iterator(builders, [_bucket(10**4, 10**5, 10)])
    iterator._pad_batch(group)
    stats = iterator.padding_stats
    atoms = sum(map(_real_atoms, group))
    bonds = sum(map(_real_bonds, group))
    assert stats == {
        "nreal_struc": 3,
        "nreal_atoms": atoms,
        "nreal_neigh": bonds,
        "pad_nstruct": 7,
        "pad_nat": 10**4 - atoms,
        "pad_nneigh": 10**5 - bonds,
    }
    iterator._pad_batch(group)
    assert iterator.padding_stats["nreal_struc"] == 6
    iterator.reset_padding_stats()
    assert set(iterator.padding_stats.values()) == {0}


def test_the_first_epoch_logs_intermediate_padding_statistics_with_each_new_bucket(
    builders, structures, caplog
):
    iterator = _iterator(builders, [], verbose=True, is_first_epoch=True)
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        iterator._pad_batch(structures[:1])  # no statistics yet
        iterator._pad_batch(structures[-3:])  # a new bucket, after one batch
    messages = [r.message for r in caplog.records]
    assert sum("New bucket" in m for m in messages) == 2
    assert sum("Intermediate padding stats" in m for m in messages) == 1


def test_later_epochs_log_the_new_bucket_but_no_intermediate_statistics(
    builders, structures, caplog
):
    iterator = _iterator(builders, [], verbose=True, is_first_epoch=False)
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        iterator._pad_batch(structures[:1])
        iterator._pad_batch(structures[-3:])
    messages = [r.message for r in caplog.records]
    assert sum("New bucket" in m for m in messages) == 2
    assert not any("Intermediate padding stats" in m for m in messages)


def _counts(group: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(map(_real_atoms, group)), sum(map(_real_bonds, group))


@pytest.mark.parametrize("axis", ["atoms", "bonds", "structs"])
def test_a_bucket_that_holds_the_group_exactly_is_used(builders, structures, axis):
    group = structures[-3:]
    atoms, bonds = _counts(group)
    exact = {"atoms": atoms, "bonds": bonds, "structs": 3}
    roomy = {"atoms": atoms + 5, "bonds": bonds + 5, "structs": 5}
    sizes = {k: (exact if axis == k else roomy)[k] for k in exact}
    iterator = _iterator(
        builders, [_bucket(sizes["atoms"], sizes["bonds"], sizes["structs"])]
    )
    iterator._pad_batch(group)
    assert len(iterator.buckets) == 1


@pytest.mark.parametrize("frac", [0.5, 1, 4])
def test_growth_from_a_nearly_sufficient_bucket_follows_the_closed_form(
    builders, structures, frac
):
    group = structures[-3:]
    atoms, bonds = _counts(group)
    assert atoms >= 20  # the fractional step must exceed the missing slot
    # atoms: one too few to hold the group (growth decides); bonds fit with room
    ref = _bucket(atoms - 1, bonds + 50, 10)
    iterator = _iterator(builders, [ref], growth_fraction=frac)
    iterator._pad_batch(group)
    (new,) = [b for b in iterator.buckets if b != ref]
    assert new[ATOMS] == _grown(atoms - 1, atoms + 1, frac)
    assert new[BONDS] == bonds + 50  # required <= current: unchanged
    assert new[STRUCTS] == 4  # the metric axis of the helper takes the required size


def test_an_axis_that_is_exactly_as_large_as_required_is_not_grown(
    builders, structures
):
    group = structures[-3:]
    atoms, bonds = _counts(group)
    ref = _bucket(
        atoms + 1, bonds - 1, 4
    )  # atoms: required == current; bonds: too small
    iterator = _iterator(builders, [ref], growth_fraction=0.5)
    iterator._pad_batch(group)
    (new,) = [b for b in iterator.buckets if b != ref]
    assert new[ATOMS] == atoms + 1


def test_the_log_names_the_axes_that_grew_including_the_padding_slot(
    builders, structures, caplog
):
    group = structures[-3:]
    atoms, bonds = _counts(group)
    ref = _bucket(atoms, bonds - 1, 4)  # atoms: one short because of the padding slot
    iterator = _iterator(builders, [ref])
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        iterator._pad_batch(group)
    (message,) = [r.message for r in caplog.records if "New bucket" in r.message]
    assert f"atoms {atoms}->" in message
    assert f"neighbours {bonds - 1}->" in message
    assert "structures" not in message.split("grew:")[1]


def test_a_quiet_first_epoch_logs_no_intermediate_statistics(
    builders, structures, caplog
):
    iterator = _iterator(builders, [], verbose=False, is_first_epoch=True)
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        iterator._pad_batch(structures[:1])
        iterator._pad_batch(structures[-3:])
    assert not any("Intermediate padding stats" in r.message for r in caplog.records)


# --- StreamingDatasetWrapper.__iter__ ---


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_pickle(DATA).iloc[:20]


def _structure_ids(batches: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for batch in batches:
        ids.extend(int(i) for i in batch[tc.DATA_STRUCTURE_ID].flatten() if int(i) >= 0)
    return ids


def _wrapper(source, builders, **config) -> StreamingDatasetWrapper:
    options = {"target_metric": 1500, "num_bins": 2, **config}
    return StreamingDatasetWrapper(
        source, builders, StreamingConfig(**options), shuffle=False
    )


def test_a_prefetching_wrapper_yields_the_batches_of_the_serial_one(builders, frame):
    serial = list(_wrapper(frame, builders))
    prefetched = list(_wrapper(frame, builders, prefetch_queue_size=3))
    assert len(prefetched) == len(serial)
    assert _structure_ids(prefetched) == _structure_ids(serial)
    for left, right in zip(serial, prefetched):
        assert int(left[tc.N_ATOMS_BATCH_TOTAL]) == int(right[tc.N_ATOMS_BATCH_TOTAL])


def test_a_dataframe_is_streamed_in_row_order_with_the_frame_index_as_structure_id(
    builders, frame
):
    batches = list(_wrapper(frame, builders))
    assert sorted(_structure_ids(batches)) == sorted(frame.index)


def test_a_generator_of_rows_is_streamed_with_its_running_number_as_structure_id(
    builders, frame
):
    rows = (frame.iloc[i] for i in range(8))
    wrapper = _wrapper(rows, builders)
    with pytest.raises(TypeError, match="no len"):
        len(wrapper)
    batches = list(wrapper)
    assert sorted(_structure_ids(batches)) == list(range(8))


def test_a_generator_of_bare_atoms_is_wrapped_into_rows(builders, frame):
    atoms = (a for a in list(frame["ase_atoms"])[:5])
    batches = list(_wrapper(atoms, builders[:1]))
    assert sum(int(b[tc.N_STRUCTURES_BATCH_REAL]) for b in batches) == 5


def test_the_epoch_updates_the_estimate_and_keeps_the_buckets_for_the_next_one(
    builders, frame
):
    wrapper = _wrapper(frame, builders)
    first = list(wrapper)
    assert len(wrapper) == len(first)
    assert wrapper._buckets
    kept = [dict(b) for b in wrapper._buckets]
    second = list(wrapper)
    assert len(second) == len(first)
    assert wrapper._buckets == kept  # the same shapes cover the second epoch


def test_the_first_epoch_logs_its_padding_statistics_and_later_epochs_do_not(
    builders, frame, caplog
):
    wrapper = _wrapper(frame, builders)
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        list(wrapper)
        first = [
            r.message for r in caplog.records if "Epoch padding stats" in r.message
        ]
        list(wrapper)
        total = [
            r.message for r in caplog.records if "Epoch padding stats" in r.message
        ]
    assert len(first) == 1
    assert len(total) == 1
    assert "num. real structures: 20" in first[0]


def test_the_length_during_the_first_epoch_follows_the_atoms_seen_so_far(
    builders, frame
):
    wrapper = _wrapper(frame, builders)
    total_atoms = sum(len(a) for a in frame["ase_atoms"])
    seen_atoms = 0
    for count, batch in enumerate(wrapper, start=1):
        seen_atoms += int(batch[tc.N_ATOMS_BATCH_REAL])
        expected = max(1, round(total_atoms / (seen_atoms / count)))
        assert len(wrapper) == expected


def test_the_shuffle_is_a_seeded_permutation_of_the_rows_independent_of_the_global_rng(
    builders, frame
):
    wrapper = StreamingDatasetWrapper(
        frame, builders, StreamingConfig(), shuffle=True, seed=5
    )
    np.random.seed(0)
    ids = [int(d[tc.DATA_STRUCTURE_ID]) for d in wrapper._structure_dicts()]
    expected = list(frame.index[np.random.default_rng(5).permutation(len(frame))])
    assert ids == expected
    assert ids != list(frame.index)  # the permutation really moves rows
    unshuffled = _wrapper(frame, builders)
    assert [
        int(d[tc.DATA_STRUCTURE_ID]) for d in unshuffled._structure_dicts()
    ] == list(frame.index)


def test_only_the_first_epoch_logs_intermediate_bucket_statistics(
    builders, frame, caplog
):
    wrapper = _wrapper(frame, builders, verbose=True, target_metric=600)
    with caplog.at_level(logging.INFO, logger="tensorpotential.data.streaming"):
        list(wrapper)
        first = sum("Intermediate padding stats" in r.message for r in caplog.records)
        list(wrapper)
        total = sum("Intermediate padding stats" in r.message for r in caplog.records)
    assert first > 0
    assert total == first
