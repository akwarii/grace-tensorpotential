"""Completeness test of the architecture matrix (``docs/torch_backend/architectures.json`` and ``.md``), without TensorFlow.

The matrix lists every architecture the TensorFlow side can build (``tools/architecture_matrix.py``). Here it is built
again with the TensorFlow stand-in (``tests_torch/tools/tfstub.py``), so a new preset, a new preset argument value in
``ARGUMENT_VALUES``, a class moved between ``SUPPORTED_OPTIONS`` and ``REJECTED_CLASSES``, or a changed instruction
constructor makes the committed JSON stale and this test fail until it is regenerated. Logic only: the matrix holds
classes, options and decisions, no physical values.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tensorpotential.calculator import foundation_models as fm
from tensorpotential.torch_backend.spec.options import (
    REJECTED_CLASSES,
    SUPPORTED_OPTIONS,
)
from tensorpotential.torch_backend.spec.registry import REGISTRY
from tests_torch.tools import tfstub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import architecture_matrix as am  # noqa: E402  # ty: ignore[unresolved-import]  (tools/ is on sys.path at run time)

ISSUE_ID = re.compile(r"^[A-Z]+[0-9]+$")
BUILT_IN_NAMES = list(fm.CORE_MODELS_NAME_LIST)
ALIASES = dict(fm.MODELS_ALIASES_DICT)
UNUSED_BY_DECISION = {
    "BondAvgSphericalHarmonic",
    "CollectInvarBasis",
    "EquivariantGate",
    "LinMLPScalarReadOut",
    "TrainableShiftTarget_v2",
}
"""Gap (g) of FAM1: classes of the TF source that no architecture uses (TWIN18 rejects each with a reason)."""


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    with tfstub.stand_in():
        return am.build_document(BUILT_IN_NAMES, ALIASES)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(am.JSON_PATH.read_text(encoding="utf-8"))


def _gaps(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(row["id"], gap) for row in document["rows"] for gap in row["gaps"]]


# ------------------------------------------------------------------ the committed files are current


def test_json_is_what_the_tool_writes(document):
    assert am.dumps(document) == am.JSON_PATH.read_text(encoding="utf-8"), (
        "docs/torch_backend/architectures.json is stale: run `python tools/architecture_matrix.py write`"
    )


def test_markdown_is_the_table_of_the_json(committed):
    assert am.to_markdown(committed) == am.MD_PATH.read_text(encoding="utf-8"), (
        "docs/torch_backend/architectures.md is stale: run `python tools/architecture_matrix.py table`"
    )


# ------------------------------------------------------------------ completeness


def test_every_gap_is_closed_by_an_issue_or_rejected_with_a_reason(committed):
    open_gaps = [
        (row, gap)
        for row, gap in _gaps(committed)
        if not ISSUE_ID.match(gap["issue"])
        and not (gap["issue"] == am.REJECTED and gap.get("reason"))
    ]
    assert not open_gaps, (
        "gaps without an issue: add them to GAP_ISSUES or GAP_REJECTIONS"
    )


def test_every_class_of_every_row_is_supported_or_a_gap_of_that_row(committed):
    for row in committed["rows"]:
        classes = set(row.get("classes", {})) | set(row.get("classes_added", []))
        gap_classes = {gap["class"] for gap in row["gaps"] if gap["option"] is None}
        for cls in classes - set(SUPPORTED_OPTIONS):
            assert cls in REJECTED_CLASSES, (
                f"{row['id']}: {cls} is neither supported nor rejected"
            )
            assert cls in gap_classes, (
                f"{row['id']}: rejected class {cls} is not listed as a gap"
            )


def test_every_supported_class_used_by_a_row_has_no_class_gap(committed):
    for row_id, gap in _gaps(committed):
        if gap["option"] is None:
            assert gap["class"] not in SUPPORTED_OPTIONS, (
                f"{row_id}: supported class {gap['class']} listed as a gap"
            )


def test_the_gaps_of_the_issue_are_found(committed):
    issues = {gap["issue"] for _, gap in _gaps(committed)}
    expected = {
        "TWIN11",
        "TWIN12",
        "TWIN13",
        "TWIN14",
        "TWIN15",
        "TWIN16",
        "TWIN17",
        "TWIN19",
    }
    assert "TWIN18" not in issues  # its classes are in no architecture
    assert expected <= issues


def test_every_radial_basis_type_is_accepted(committed):
    """RAD2 closed gap (h): no row has a ``RadialBasis`` gap, whichever ``basis_type`` its variant sets."""
    assert not [r for r, gap in _gaps(committed) if gap["class"] == "RadialBasis"]


def test_every_foundation_model_maps_to_a_row(committed):
    rows = {row["id"] for row in committed["rows"]}
    mapping = committed["foundation_models"]
    assert set(mapping) == set(BUILT_IN_NAMES) | set(ALIASES)
    assert all(entry["row"] in rows for entry in mapping.values()), (
        "a foundation model maps to no row"
    )


def test_classes_in_no_architecture_have_a_decision(committed):
    unused = committed["unused_classes"]
    assert all(
        REGISTRY[path].status == "rejected" and entry["decision"]
        for path, entry in unused.items()
    )
    by_issue = {
        path.rsplit(".", 1)[-1]
        for path, entry in unused.items()
        if entry["issue"] == "TWIN18"
    }
    assert by_issue == UNUSED_BY_DECISION


# ------------------------------------------------------------------ planted changes


def test_a_new_preset_argument_value_adds_a_variant_row(committed, monkeypatch):
    monkeypatch.setitem(am.ARGUMENT_VALUES, "n_rad_base", (4,))
    with tfstub.stand_in():
        rows = am.generated_rows("preset", "GRACE_1LAYER_v1_24")
    stored = [
        row
        for row in committed["rows"]
        if row["id"].startswith("preset:GRACE_1LAYER_v1_24")
    ]
    assert len(rows) == len(stored) + 1


def test_a_gap_without_issue_is_written_as_unassigned(monkeypatch):
    monkeypatch.delitem(am.GAP_ISSUES, ("ZBLPotential", None))
    raw = {
        "instructions": {
            "Z": {
                "__cls__": "tensorpotential.instructions.compute.ZBLPotential",
                "name": "Z",
            }
        }
    }
    assert {gap["issue"] for gap in am.gaps_of(raw)} == {am.UNASSIGNED}


# ------------------------------------------------------------------ helpers


def test_a_malformed_model_is_one_unassigned_gap():
    assert am.gaps_of({"instructions": {"A": {"name": "A"}}}) == [
        {
            "kind": "MalformedModelError",
            "class": None,
            "option": None,
            "issue": am.UNASSIGNED,
        }
    ]


def test_a_rejected_gap_carries_its_reason():
    raw = {
        "instructions": {
            "R": {
                "__cls__": "tensorpotential.instructions.compute.FunctionReduceN",
                "name": "R",
                "scale": [1.0, 2.0],
            }
        }
    }
    scale = [gap for gap in am.gaps_of(raw) if gap["option"] == "scale"]
    assert scale == [
        {
            "kind": "unsupported_value",
            "class": "tensorpotential.instructions.compute.FunctionReduceN",
            "option": "scale",
            "issue": am.REJECTED,
            "reason": am.GAP_REJECTIONS["FunctionReduceN", "scale"],
        }
    ]


def test_foundation_names_follow_files_fine_tunes_aliases_and_name_rules():
    rows = [
        "file:GRACE-2L-OMAT-large-base",
        "preset:GRACE_1LAYER_v1_24",
        "file:GRACE-FS-OMAT",
    ]
    aliases = {"GRACE-2L-OMAT-L": "GRACE-2L-OMAT-large-ft-E"}
    names = [
        "GRACE-2L-OMAT-large-base",
        "GRACE-2L-OMAT-large-ft-AM-fp64",
        "GRACE-1L-OAM",
        "GRACE-FS-OAM",
        "Mystery",
    ]
    table = am.foundation_rows(names, aliases, rows)
    assert {name: entry["row"] for name, entry in table.items()} == {
        "GRACE-2L-OMAT-large-base": "file:GRACE-2L-OMAT-large-base",
        "GRACE-2L-OMAT-large-ft-AM-fp64": "file:GRACE-2L-OMAT-large-base",
        "GRACE-2L-OMAT-L": "file:GRACE-2L-OMAT-large-base",
        "GRACE-1L-OAM": "preset:GRACE_1LAYER_v1_24",
        "GRACE-FS-OAM": "file:GRACE-FS-OMAT",
        "Mystery": am.UNASSIGNED,
    }
    assert table["GRACE-2L-OMAT-large-base"]["evidence"] == "model file"


def test_class_options_lists_distinct_values_and_shortens_long_ones():
    long_map = {f"E{i}": i for i in range(20)}
    instructions = {
        "A": {
            "__cls__": "X.C",
            "name": "A",
            "n": 2,
            "ref": {"_instruction_": True, "name": "B"},
            "m": long_map,
        },
        "B": {"__cls__": "X.C", "name": "B", "n": 1, "ref": None, "m": long_map},
    }
    assert am.class_options(instructions) == {
        "X.C": {"m": ["<dict[str,int]>"], "n": [1, 2], "ref": ["<ref>", None]}
    }


def test_variants_skip_defaults_and_arguments_a_preset_lacks():
    with tfstub.stand_in():
        labels = [label for label, _ in am.variants("preset", "GRACE_2LAYER_v2_25")]
    assert labels[0] == ""
    assert 'basis_type="Cheb"' not in labels  # the default
    assert not any(
        label.startswith(("mlp_radial", "parity", "compute_energy")) for label in labels
    )
    assert 'basis_type="Gaussian"' in labels


def test_fixture_dtypes_read_rows_dtypes_and_what_was_not_made(tmp_path, monkeypatch):
    def manifest(name: str, **fields: Any) -> None:
        (tmp_path / f"{name}.json").write_text(json.dumps(fields))

    manifest(
        "a_f64",
        tier="tiny",
        row="preset:A",
        dtype="float64",
        option=None,
        not_made={"f32": "why"},
    )
    manifest(
        "a_f64_dense",
        tier="tiny",
        row="preset:A",
        dtype="float64",
        option="dense_nbr",
        not_made={},
    )
    manifest(
        "b_f32", tier="tiny", row="preset:B", dtype="float32", option=None, not_made={}
    )
    manifest("c_f64", tier="faithful", row="preset:C", dtype="float64", option=None)
    manifest("d_f64", tier="tiny", dtype="float64", option=None)
    monkeypatch.setattr(am, "GOLDEN_DIR", tmp_path)
    am.fixture_dtypes.cache_clear()
    try:
        assert am.fixture_dtypes() == {
            "preset:A": frozenset({"float64", "float32"}),
            "preset:B": frozenset({"float32"}),
        }
        assert am.fixture_present({"id": "preset:A"})
        assert not am.fixture_present({"id": "preset:B"})
        assert am.fixture_cell({"id": "preset:B", "source": "preset"}) == "no"
        assert (
            am.fixture_cell({"id": "tensor:x", "source": "tensor"}) == "free (TWIN19)"
        )
    finally:
        am.fixture_dtypes.cache_clear()


def test_table_command_writes_the_markdown_of_the_json(
    tmp_path, monkeypatch, committed
):
    monkeypatch.setattr(am, "MD_PATH", tmp_path / "architectures.md")
    assert am.main(["table"]) == 0
    assert am.MD_PATH.read_text(encoding="utf-8") == am.to_markdown(committed)
