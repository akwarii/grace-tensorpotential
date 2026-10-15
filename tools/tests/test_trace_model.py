"""Tests for tools/trace_model.py.

Logic tests use hand-made HLO text, trace events and op names. The value test runs the tool in a
subprocess (the XLA flags must be set before XLA starts) on the 1-layer test model and checks that
every instruction with work in it is found, against the instruction list of the model's yaml.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import trace_model as tm  # noqa: E402

HLO = """\
HloModule m
%fused (p: f64[2]) -> f64[2] {
  %p = f64[2]{0} parameter(0)
  ROOT %neg = f64[2]{0} negate(%p), metadata={op_type="Neg" op_name="A/neg"}
}
ENTRY %main {
  %dot.3 = f64[54,64]{1,0} dot(%a, %b), metadata={op_type="Einsum" op_name="AA1/ein_left/Einsum"}
  %wrapped_scatter.2.clone = f64[54,8]{1,0} fusion(%x), kind=kLoop, metadata={op_type="UnsortedSegmentSum" op_name="gradient_tape/A/UnsortedSegmentSum"}
  ROOT %call.4 = f64[2]{0} call(%y), to_apply=%fused, metadata={op_name="A/call"}
  %copy.1 = f64[2]{0} copy(%z)
}
"""


# ------------------------------------------------------------------ logic


def test_hlo_metadata_maps_instruction_names_to_shape_and_tensorflow_op():
    meta = tm.hlo_metadata(HLO)

    assert meta["dot.3"] == ("f64[54,64]{1,0}", "AA1/ein_left/Einsum")
    assert meta["wrapped_scatter.2.clone"][1] == "gradient_tape/A/UnsortedSegmentSum"
    assert meta["neg"] == ("f64[2]{0}", "A/neg")
    assert "copy.1" not in meta  # no metadata, nothing to attribute it to


@pytest.mark.parametrize(
    ("op_name", "expected"),
    [
        ("AA1/ein_left/Einsum", ("AA1", "forward")),
        ("gradient_tape/AA1/ein_left/Einsum_1", ("AA1", "backward")),
        ("R_Linear_2/MatMul", ("R", "forward")),
        ("A1_2_red/GatherV2", ("A1_2_red", "forward")),
        ("A_ChemIndTransf/MatMul", ("A", "forward")),
        ("mul_7", (tm.MODEL, "forward")),
        ("gradient_tape/AddN_3", (tm.MODEL, "backward")),
    ],
)
def test_owner_is_the_longest_matching_instruction_name(op_name, expected):
    names = ["R", "A", "A1", "A1_2_red", "AA1"]

    assert tm.owner(op_name, names) == expected


def test_attribute_sums_per_instruction_and_phase_and_skips_unmapped_events():
    meta = tm.hlo_metadata(HLO)
    op_ms = {
        "dot.3": 4.0,
        "wrapped_scatter.2.clone": 3.0,
        "neg.clone": 1.0,  # the clone of a mapped name
        "call.4": 50.0,  # the thunk wrapper: its contents are timed on their own
        "ThunkExecutor::Execute": 99.0,
    }

    rows, ops = tm.attribute(op_ms, meta, ["A", "AA1"])

    assert rows == [
        {"instruction": "AA1", "forward_ms": 4.0, "backward_ms": 0.0},
        {"instruction": "A", "forward_ms": 1.0, "backward_ms": 3.0},
    ]
    assert [o["hlo"] for o in ops] == ["dot.3", "wrapped_scatter.2.clone", "neg"]


def test_render_prints_shares_totals_and_the_largest_ops():
    rows = [
        {"instruction": "AA1", "forward_ms": 6.0, "backward_ms": 0.0},
        {"instruction": "A", "forward_ms": 1.0, "backward_ms": 1.0},
    ]
    ops = [{"hlo": "dot.3", "shape": "f64[2]", "op_name": "AA1/x", "ms": 6.0}]

    text = tm.render(rows, ops, wall_ms=5.0, top_ops=1)

    lines = text.splitlines()
    assert lines[2] == "| AA1 | 6.0 | 0.0 | 75.0 |"
    assert lines[3] == "| A | 1.0 | 1.0 | 25.0 |"
    assert lines[4] == "| summed op time | | | 8.0 ms |"
    assert lines[5] == "| wall time of one call | | | 5.0 ms |"
    assert lines[-1] == "| dot.3 | f64[2] | AA1/x | 6.0 |"
    assert "HLO op" not in tm.render(rows, ops, 5.0, top_ops=0)


def test_render_of_no_rows_has_zero_totals():
    assert "| summed op time | | | 0.0 ms |" in tm.render([], [], 1.0, 0)


def test_trace_events_are_summed_per_name_and_divided_by_the_calls():
    xplane_pb2 = pytest.importorskip("tensorflow.tsl.profiler.protobuf.xplane_pb2")
    space = xplane_pb2.XSpace()
    plane = space.planes.add()
    plane.event_metadata[1].name = "dot.3"
    plane.event_metadata[2].name = "neg"
    line = plane.lines.add()
    for metadata_id, ps in [(1, 2_000_000_000), (1, 4_000_000_000), (2, 1_000_000_000)]:
        event = line.events.add()
        event.metadata_id = metadata_id
        event.duration_ps = ps

    assert tm.trace_events_ms(space, calls=2) == {"dot.3": 3.0, "neg": 0.5}


def test_the_newest_trace_and_the_largest_dump_are_read(tmp_path):
    xplane_pb2 = pytest.importorskip("tensorflow.tsl.profiler.protobuf.xplane_pb2")
    space = xplane_pb2.XSpace()
    space.planes.add().name = "/host:CPU"
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "a.xplane.pb").write_bytes(space.SerializeToString())
    (tmp_path / "x.after_optimizations.txt").write_text("small")
    (tmp_path / "y.after_optimizations.txt").write_text("the larger module")

    assert tm.read_xspace(tmp_path).planes[0].name == "/host:CPU"
    assert tm.largest_optimized_hlo(tmp_path) == "the larger module"


def test_instruction_names_of_a_list_or_a_dict_of_instructions():
    class Ins:
        def __init__(self, name):
            self.name = name

    class Model:
        instructions = [Ins("a"), Ins("b")]

    assert tm.instruction_names(Model()) == ["a", "b"]
    Model.instructions = {"a": Ins("a"), "b": Ins("b")}
    assert tm.instruction_names(Model()) == ["a", "b"]


# ------------------------------------------------------------------ values


def test_the_tool_attributes_the_whole_model_to_its_instructions(tmp_path):
    pytest.importorskip("tensorflow")
    out = tmp_path / "t.json"

    done = subprocess.run(  # noqa: S603 - the tool under test, with the interpreter running the tests
        [
            sys.executable,
            str(ROOT / "tools" / "trace_model.py"),
            "model_grace.yaml",
            "--repeats",
            "1",
            "--calls",
            "1",
            "--top-ops",
            "5",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(out.read_text())["trace"]
    found = {r["instruction"] for r in result["rows"]}
    spec = yaml.safe_load((ROOT / "tests" / "model_grace.yaml").read_text())
    names = {
        entry["name"] for entry in spec if isinstance(entry, dict) and "name" in entry
    }
    assert {"R", "A"} <= found  # the radial function and the basis do the heavy work
    assert found - {tm.MODEL} <= names
    assert result["wall_ms"] > 0
    assert len(result["ops"]) == 5
    assert sum(r["backward_ms"] for r in result["rows"]) > 0
    assert "| instruction | forward ms |" in done.stdout
