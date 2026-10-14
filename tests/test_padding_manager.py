"""``PaddingManager``: the bounds it keeps, when it adapts its margins and what it reports at each verbosity.

Logic layer. The data builders are replaced by one stub that reports the real atoms and neighbours of the structure it is
given, so that every branch of ``get_data`` and ``_record_and_maybe_adapt`` is reached with numbers chosen by hand.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from tensorpotential import constants
from tensorpotential.calculator.asecalculator import (
    AdaptivePaddingConfig,
    PaddingManager,
)

pytestmark = pytest.mark.usefixtures("info_logs")


class _StubBuilder:
    """Reports ``(n_atoms, n_neighbors)`` as the real counts of the batch and records the padding bounds it is given."""

    def __init__(self) -> None:
        self.padded_with: list[dict] = []

    def extract_from_ase_atoms(self, atoms):
        return {"size": atoms}

    def join_to_batch(self, group):
        n_atoms, n_neighbors = group[0]["size"]
        return {
            constants.N_ATOMS_BATCH_REAL: n_atoms,
            constants.N_NEIGHBORS_REAL: n_neighbors,
        }

    def pad_batch(self, _batch, max_dict):
        self.padded_with.append(dict(max_dict))


def _manager(verbose=0, **kwargs):
    builder = _StubBuilder()
    kwargs.setdefault("adaptive_padding", False)
    return PaddingManager(
        cast(Any, [builder]), debug_padding_verbose=verbose, **kwargs
    ), builder


def _bound(builder):
    padded = builder.padded_with[-1]
    return padded[constants.PAD_MAX_N_ATOMS], padded[constants.PAD_MAX_N_NEIGHBORS]


def test_the_first_structure_sets_a_bound_with_the_fractions_added():
    manager, builder = _manager(pad_atoms_number=10, pad_neighbors_fraction=0.5)
    manager.get_data((4, 1000))
    assert _bound(builder) == (14, 1500)
    assert manager.padding_bounds == [(14, 1500)]
    assert builder.padded_with[-1][constants.PAD_MAX_N_STRUCTURES] == 1


def test_a_smaller_structure_reuses_the_existing_bound():
    manager, builder = _manager(
        pad_atoms_number=10,
        pad_neighbors_fraction=0.5,
        max_number_reduction_recompilation=None,
    )
    manager.get_data((4, 1000))
    manager.get_data((5, 1100))
    assert _bound(builder) == (14, 1500)
    assert manager.padding_bounds == [(14, 1500)]


def test_a_larger_structure_adds_a_new_bound():
    manager, builder = _manager(pad_atoms_number=10, pad_neighbors_fraction=0.5)
    manager.get_data((4, 1000))
    manager.get_data((20, 2000))
    assert _bound(builder) == (30, 3000)
    assert manager.padding_bounds == [(14, 1500), (30, 3000)]


def test_without_padding_options_the_batch_is_not_padded():
    manager, builder = _manager(pad_atoms_number=None, pad_neighbors_fraction=None)
    batch = manager.get_data((4, 1000))
    assert builder.padded_with == []
    assert batch[constants.N_ATOMS_BATCH_REAL] == 4


def test_an_excessive_bound_is_reduced_until_the_allowed_number_is_used_up():
    manager, _ = _manager(
        pad_atoms_number=10,
        pad_neighbors_fraction=0.5,
        max_number_reduction_recompilation=1,
    )
    manager.get_data((100, 10000))
    manager.get_data((4, 300))
    assert manager.number_reduction_recompilation == 1
    assert len(manager.padding_bounds) == 2
    manager.get_data((4, 280))
    assert manager.number_reduction_recompilation == 1
    assert len(manager.padding_bounds) == 2


def test_verbosity_two_reports_new_and_reduced_bounds(caplog):
    manager, _ = _manager(
        verbose=2,
        pad_atoms_number=10,
        pad_neighbors_fraction=0.5,
        max_number_reduction_recompilation=1,
    )
    manager.get_data((100, 10000))
    manager.get_data((200, 20000))
    manager.get_data((4, 300))
    out = caplog.messages
    assert out == [
        "Adding initial padding bound: (110, 15000)",
        "Adding new maximum padding bound: (210, 30000)",
        "Reducing padding and adding new bound: (14, 450)",
    ]


def test_verbosity_three_reports_every_decision(caplog):
    manager, _ = _manager(
        verbose=3,
        pad_atoms_number=10,
        pad_neighbors_fraction=0.5,
        max_number_reduction_recompilation=None,
    )
    manager.get_data((100, 10000))
    manager.get_data((100, 10000))
    manager.get_data((4, 300))
    out = caplog.messages
    assert "nreal_atoms=100, nreal_neigh=10000" in out
    assert "Adding initial padding bound: (110, 15000)" in out
    assert "extract padding bounds: (110, 15000)" in out
    assert "Using existing padding bound: (110, 15000)" in out
    assert (
        "Padding reduction skipped or max reductions reached. Keeping bound: (110, 15000)"
        in out
    )
    assert out.count("padding_fraction=0.50") == 2
    assert len(manager.padding_fraction_history) == 3


def test_verbosity_three_reports_the_miss_that_adds_a_bound(caplog):
    manager, _ = _manager(verbose=3, pad_atoms_number=10, pad_neighbors_fraction=0.5)
    manager.get_data((4, 1000))
    manager.get_data((20, 2000))
    assert "Adding new maximum padding bound: (30, 3000)" in caplog.text


def test_silent_by_default(caplog):
    manager, _ = _manager()
    manager.get_data((4, 1000))
    manager.get_data((20, 2000))
    assert caplog.text == ""


def _adaptive(verbose: int = 0, **config: Any) -> PaddingManager:
    config = {"miss_rate_window": 2, "miss_rate_threshold": 0.5, **config}
    return _manager(
        verbose=verbose,
        pad_atoms_number=10,
        pad_neighbors_fraction=0.2,
        adaptive_padding=True,
        adaptive_padding_config=AdaptivePaddingConfig(**config),
    )[0]


def test_adaptation_is_off_when_adaptive_padding_is_off():
    manager, _ = _manager(pad_atoms_number=10, pad_neighbors_fraction=0.2)
    manager._record_and_maybe_adapt(is_miss=True)
    assert manager._call_history == type(manager._call_history)(
        maxlen=manager._call_history.maxlen
    )
    assert manager._n_adaptations == 0


def test_nothing_is_adapted_before_the_window_is_full():
    manager = _adaptive()
    manager._record_and_maybe_adapt(is_miss=True)
    assert manager._n_adaptations == 0
    assert len(manager._call_history) == 1


def test_nothing_is_adapted_when_the_miss_rate_is_at_the_threshold():
    manager = _adaptive()
    manager._record_and_maybe_adapt(is_miss=True)
    manager._record_and_maybe_adapt(is_miss=False)
    assert manager._n_adaptations == 0
    assert manager._current_pad_atoms_number == 10


def test_a_miss_rate_over_the_threshold_grows_the_margins_and_clears_the_history():
    manager = _adaptive(atoms_growth_factor=2.0, neighbors_growth_factor=1.5)
    manager._record_and_maybe_adapt(is_miss=True)
    manager._record_and_maybe_adapt(is_miss=True)
    assert manager._n_adaptations == 1
    assert manager._current_pad_atoms_number == 20
    assert manager._current_pad_neighbors_fraction == pytest.approx(0.3)
    assert len(manager._call_history) == 0


def test_the_growth_stops_at_the_configured_maxima():
    manager = _adaptive(max_pad_atoms_number=15, max_pad_neighbors_fraction=0.25)
    for _ in range(2):
        manager._record_and_maybe_adapt(is_miss=True)
    assert manager._current_pad_atoms_number == 15
    assert manager._current_pad_neighbors_fraction == pytest.approx(0.25)


def test_the_first_growth_is_logged_once_at_info_and_later_ones_are_not(caplog):
    manager = _adaptive()
    with caplog.at_level(
        logging.INFO, logger="tensorpotential.calculator.asecalculator"
    ):
        for _ in range(4):
            manager._record_and_maybe_adapt(is_miss=True)
    first_growths = [
        r for r in caplog.records if "grew margins for the first time" in r.getMessage()
    ]
    assert len(first_growths) == 1
    assert manager._n_adaptations == 2


def test_verbosity_one_reports_each_growth(caplog):
    manager = _adaptive(verbose=1)
    for _ in range(2):
        manager._record_and_maybe_adapt(is_miss=True)
    out = caplog.text
    assert (
        "Adaptive padding: miss_rate=1.00 > 0.5, growing margins: pad_atoms 10 -> 20"
        in out
    )
    assert "pad_neighbors_frac 0.2000 -> 0.3000 (adaptation #1)" in out
