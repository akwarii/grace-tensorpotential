"""Characterization tests for ``process_cutoff_dict`` and ``CUTOFF_PRESETS``.

``process_cutoff_dict`` expands what the user writes (a preset name, or a dictionary keyed by
``"CuAl"``, ``"Cu"``, ``"Cu*"`` or ``("Cu", "Al")``) into a table ``(element, element) -> cutoff``
that holds both orders of every pair and only elements of the element map.

Logic layer: every key syntax, the order in which expansions win, the errors, no side effect on
the input. Physical-value layer: the table of a preset is the mean of the two single-element
radii for a mixed pair, and the table is symmetric. The expected numbers are written by hand
from the preset table, not produced by the function.
"""

from __future__ import annotations

import pytest

from tensorpotential.core.cutoffs import CUTOFF_PRESETS, process_cutoff_dict
from tests.tolerances import FLOAT64_ARITHMETIC as ARITHMETIC

CU_AL = {"Cu": 0, "Al": 1}
PRESET = "DEFAULT_CUTOFF_2L"


def test_pair_key_gives_both_orders() -> None:
    assert process_cutoff_dict({"CuAl": 3.5}, CU_AL) == {
        ("Cu", "Al"): 3.5,
        ("Al", "Cu"): 3.5,
    }


def test_single_element_key_is_the_homonuclear_pair() -> None:
    assert process_cutoff_dict({"Cu": 2.5}, CU_AL) == {("Cu", "Cu"): 2.5}


def test_two_letter_symbols_and_one_letter_symbols_split_on_capitals() -> None:
    table = process_cutoff_dict({"HeH": 1.0}, {"He": 0, "H": 1})
    assert table == {("He", "H"): 1.0, ("H", "He"): 1.0}


def test_tuple_key_is_taken_as_is() -> None:
    table = process_cutoff_dict({("Cu", "Al"): 3.0}, CU_AL)
    assert table == {("Cu", "Al"): 3.0, ("Al", "Cu"): 3.0}


def test_star_expands_over_the_element_map() -> None:
    table = process_cutoff_dict({"Cu*": 3.0}, CU_AL)
    assert table == {("Cu", "Cu"): 3.0, ("Cu", "Al"): 3.0, ("Al", "Cu"): 3.0}


def test_lone_star_gives_every_homonuclear_pair() -> None:
    assert process_cutoff_dict({"*": 2.0}, CU_AL) == {("Cu", "Cu"): 2.0, ("Al", "Al"): 2.0}


def test_star_wins_over_an_explicit_key_for_the_same_pair() -> None:
    """The expanded keys are applied after the explicit ones, whatever the order written."""
    table = process_cutoff_dict({"Cu*": 3.0, "CuAl": 1.0}, CU_AL)
    assert table[("Cu", "Al")] == 3.0
    assert table[("Al", "Cu")] == 3.0


def test_both_orders_written_with_different_values_end_up_swapped() -> None:
    """Quirk of the symmetrisation: each order receives the value written for the other."""
    table = process_cutoff_dict({"CuAl": 1.0, "AlCu": 2.0}, CU_AL)
    assert table == {("Cu", "Al"): 2.0, ("Al", "Cu"): 1.0}


def test_elements_outside_the_map_are_dropped() -> None:
    table = process_cutoff_dict({"CuAl": 3.0, "CuH": 2.0, "HH": 1.0}, CU_AL)
    assert set(table) == {("Cu", "Al"), ("Al", "Cu")}


def test_the_input_dictionary_is_not_modified() -> None:
    given = {"Cu*": 3.0, "AlAl": 4.0}
    process_cutoff_dict(given, CU_AL)
    assert given == {"Cu*": 3.0, "AlAl": 4.0}


def test_a_tuple_of_three_symbols_is_rejected() -> None:
    with pytest.raises(AssertionError):
        process_cutoff_dict({("Cu", "Al", "H"): 3.0}, CU_AL)


def test_a_tuple_with_a_non_string_is_rejected() -> None:
    with pytest.raises(AssertionError):
        process_cutoff_dict({("Cu", 1): 3.0}, CU_AL)


def test_a_key_that_is_neither_string_nor_tuple_is_rejected() -> None:
    with pytest.raises(ValueError, match="Can not parse"):
        process_cutoff_dict({frozenset({"Cu"}): 3.0}, CU_AL)


def test_an_unknown_preset_name_fails() -> None:
    with pytest.raises(AttributeError):
        process_cutoff_dict("NO_SUCH_PRESET", CU_AL)


def test_preset_table_holds_the_four_pairs_of_two_elements() -> None:
    table = process_cutoff_dict(PRESET, CU_AL)
    assert set(table) == {("Cu", "Cu"), ("Cu", "Al"), ("Al", "Cu"), ("Al", "Al")}


def test_preset_restricts_to_the_elements_of_the_map() -> None:
    table = process_cutoff_dict(PRESET, {"Cu": 0})
    assert set(table) == {("Cu", "Cu")}


@pytest.mark.parametrize("preset", sorted(CUTOFF_PRESETS))
def test_every_preset_is_expandable_and_symmetric(preset: str) -> None:
    elements = ["H", "O", "Cu", "Al"]
    table = process_cutoff_dict(preset, {e: i for i, e in enumerate(elements)})
    assert all(table[b, a] == r for (a, b), r in table.items())


def test_preset_mixed_pair_is_the_mean_of_the_two_radii() -> None:
    """Hand computation: the radius of a mixed pair is (r_a + r_b) / 2, a homonuclear pair r_a."""
    radii = CUTOFF_PRESETS[PRESET]
    r_cu, r_al = radii["Cu"], radii["Al"]
    assert r_cu != r_al
    table = process_cutoff_dict(PRESET, CU_AL)
    assert table[("Cu", "Cu")] == r_cu
    assert table[("Al", "Al")] == r_al
    assert table[("Cu", "Al")] == pytest.approx(
        (r_cu + r_al) / 2, rel=ARITHMETIC.rtol, abs=ARITHMETIC.atol
    )
