"""``tensorpotential/data/preprocess.py``: the batching, parameter and data-builder functions behind ``grace_preprocess``.

The command (parser, stages, ``main``) is tested in ``test_grace_preprocess.py``; the physical layer of the stages
through the real entry point is there too.
"""

from __future__ import annotations

import logging

import pytest

from tensorpotential.data import preprocess
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
)
from tensorpotential.scripts import grace_preprocess
from tests.tolerances import FLOAT64_ARITHMETIC

# Multiplier from the input units to eV/A3, computed by hand: 1 eV/A3 = 160.2176621 GPa = 1602.176621 kbar.
STRESS_UNIT_FACTORS = {
    "eV/A3": 1.0,
    "GPa": 1 / 160.2176621,
    "kbar": 1 / 1602.176621,
    "-kbar": -1 / 1602.176621,
}


def parse(*argv: str):
    return grace_preprocess.build_parser().parse_args(["data.pkl.gz", *argv])


def test_get_databuilders_builds_geometry_then_reference_builder():
    args = grace_preprocess.build_parser().parse_args([
        "data.pkl.gz",
        "--is-fit-stress",
        "--energy-col",
        "E",
        "--forces-col",
        "F",
        "--stress-col",
        "S",
    ])
    geom_db, ref_db = preprocess.get_databuilders(
        {"Cu": 0}, 4.0, args, precision="float64", cutoff_dict=None
    )
    assert isinstance(geom_db, GeometricalDataBuilder)
    assert isinstance(ref_db, ReferenceEnergyForcesStressesDataBuilder)
    assert geom_db.is_fit_stress and ref_db.is_fit_stress
    assert (ref_db.energy_col, ref_db.forces_col, ref_db.stress_col) == ("E", "F", "S")
    assert ref_db.stress_conversion_factor == 1.0


@pytest.mark.parametrize("units", sorted(STRESS_UNIT_FACTORS))
def test_get_databuilders_passes_the_units_to_the_reference_builder(units):
    args = parse("--is-fit-stress", f"--stress-units={units}")
    _, ref_db = preprocess.get_databuilders({"Cu": 0}, 4.0, args)
    assert ref_db.stress_conversion_factor == pytest.approx(
        STRESS_UNIT_FACTORS[units],
        rel=FLOAT64_ARITHMETIC.rtol,
        abs=FLOAT64_ARITHMETIC.atol,
    )


def test_load_params_of_an_unparsable_file_logs_and_returns_nothing(tmp_path, caplog):
    (tmp_path / "params.json").write_text("{not json")
    with caplog.at_level(logging.ERROR):
        assert preprocess.load_params(str(tmp_path)) is None
    assert "Can't parse" in caplog.text


def test_load_params_without_a_file_returns_nothing(tmp_path):
    assert preprocess.load_params(str(tmp_path)) is None


def test_load_params_round_trips_save_params(tmp_path):
    preprocess.save_params(str(tmp_path), {"Cu": 0, "Zn": 1}, 5.5, "float32", {"Cu": {"Zn": 4.0}})
    assert preprocess.load_params(str(tmp_path)) == ({"Cu": 0, "Zn": 1}, 5.5, {"Cu": {"Zn": 4.0}}, "float32")


@pytest.mark.parametrize(
    "content",
    ['{"cutoff": 4.0, "precision": "float64"}', "[1, 2, 3]", '{"elements_map": {}, "cutoff": 4.0}'],
    ids=["no-elements-map", "not-a-mapping", "no-precision"],
)
def test_load_params_of_a_file_that_is_not_a_complete_mapping_logs_and_returns_nothing(
    tmp_path, caplog, content
):
    (tmp_path / "params.json").write_text(content)
    with caplog.at_level(logging.ERROR):
        assert preprocess.load_params(str(tmp_path)) is None
    assert "Can't parse" in caplog.text
