"""Tests for tools/grace_probe.py.

Logic tests use small hand-built TensorFlow modules whose creation and use dtypes are known by
construction. Physical-value tests probe the real 1-layer test model (seeded random weights) and
check it against oracles that do not use the probe: the values stored in a written checkpoint
(``tf.train.load_checkpoint``), a brute-force neighbour list (``tests/neighbour_oracle.py``) and
the dtype behaviour of the TensorFlow classes pinned by their own tests (a float32 model still
computes the radial basis and the spherical harmonics in float64).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import grace_probe as gp  # noqa: E402

pytest.importorskip("tensorpotential")
tf = pytest.importorskip("tensorflow")

YAML = "model_grace.yaml"

# One TensorFlow configuration for every test module of tools/ (one thread, deterministic ops):
# the thread pools cannot be changed once TensorFlow has run, and the oracle-snapshot tests
# set exactly this.
gp.osn.configure_tensorflow()


# ------------------------------------------------------------------ logic


def test_attribute_field_names_a_unique_match_and_truncates_ambiguous_ones():
    assert gp._attribute_field(["a/b"]) == {"attribute": "a/b"}  # noqa: SLF001
    many = [f"x/{'y' * n}" for n in (5, 1, 4, 2, 3)]
    field = gp._attribute_field(many)  # noqa: SLF001
    assert field["attribute"] is None
    assert field["n_candidates"] == 5
    assert field["candidates"] == [many[1], many[3], many[4]]


@pytest.mark.parametrize(
    ("cast", "expected"),
    [
        ("float32->float64", True),
        ("float64->float32", True),
        ("int32->float64", False),
        ("float64->int32", False),
    ],
)
def test_is_float_cast_needs_two_floating_dtypes(cast, expected):
    assert gp._is_float_cast(cast) is expected  # noqa: SLF001


def test_median_ms_calls_the_function_once_per_repeat():
    calls = []
    value = gp.median_ms(lambda: calls.append(time.perf_counter()), 7)
    assert len(calls) == 7
    assert value >= 0.0


def test_median_ms_is_the_median_not_the_mean():
    # one slow call among five fast ones must not move the median
    delays = iter([0.0, 0.0, 0.05, 0.0, 0.0])
    value = gp.median_ms(lambda: time.sleep(next(delays)), 5)
    assert value < 5.0


def test_resolve_yaml_takes_a_path_or_a_name_in_tests(tmp_path):
    own = tmp_path / "m.yaml"
    own.write_text("x: 1\n")
    assert gp.resolve_yaml(str(own)) == own
    assert gp.resolve_yaml(YAML) == gp.TESTS / YAML


def test_a_checkpoint_needs_exactly_one_yaml_and_one_dtype():
    ok = argparse.Namespace(checkpoint="c", yamls=["a.yaml"], dtypes=["float32"])
    gp._check_checkpoint_args(ok)  # noqa: SLF001
    for yamls, dtypes in ((["a", "b"], ["float32"]), (["a"], ["float32", "float64"])):
        bad = argparse.Namespace(checkpoint="c", yamls=yamls, dtypes=dtypes)
        with pytest.raises(SystemExit, match="exactly one yaml and one dtype"):
            gp._check_checkpoint_args(bad)  # noqa: SLF001
    gp._check_checkpoint_args(  # noqa: SLF001
        argparse.Namespace(checkpoint=None, yamls=["a", "b"], dtypes=["float32"] * 2)
    )


def test_foreign_cores_is_machine_cpu_minus_own_cpu_per_wall_second():
    assert gp.foreign_cores(100.0, 140.0, 30.0, 5.0) == pytest.approx(2.0)
    assert gp.foreign_cores(100.0, 130.0, 30.0, 5.0) == 0.0
    # accounting jitter must not give a negative core count
    assert gp.foreign_cores(100.0, 129.0, 30.0, 5.0) == 0.0


@pytest.mark.parametrize(
    ("before", "after", "wall"), [(None, 1.0, 1.0), (1.0, None, 1.0), (1.0, 2.0, 0.0)]
)
def test_foreign_cores_is_none_when_it_cannot_be_measured(before, after, wall):
    assert gp.foreign_cores(before, after, 0.1, wall) is None


def test_machine_busy_seconds_reads_the_first_line_of_proc_stat(tmp_path):
    stat = tmp_path / "stat"
    # user nice system idle iowait irq softirq steal: idle and iowait are left out
    stat.write_text("cpu  100 20 30 9999 8888 4 5 6 0 0\ncpu0 1 1 1 1 1 1 1 1\n")
    ticks = os.sysconf("SC_CLK_TCK")
    assert gp.machine_busy_seconds(stat) == pytest.approx(
        (100 + 20 + 30 + 4 + 5 + 6) / ticks
    )
    assert gp.machine_busy_seconds(tmp_path / "missing") is None
    stat.write_text("")
    assert gp.machine_busy_seconds(stat) is None


def test_busy_seconds_of_this_machine_only_grow():
    first = gp.machine_busy_seconds()
    sum(i * i for i in range(2_000_000))
    second = gp.machine_busy_seconds()
    assert first is not None and second is not None
    assert second >= first


class _Stub(tf.Module):
    """An instruction-shaped module: a float32 weight, a float64 constant, a float64 input."""

    def __init__(self, name="stub"):
        super().__init__(name=name)
        self.name = name
        self.w = tf.Variable(np.full((3,), 2.0, dtype=np.float32), name="w")
        self.scale = tf.constant(0.5, dtype=tf.float64)
        self.table = np.arange(3, dtype=np.float64)

    def __call__(self, input_data, training=False, local=False):
        x = tf.cast(input_data["x"], tf.float32) * self.w
        input_data[self.name] = tf.cast(x, tf.float64) * self.scale + self.table


@pytest.fixture(scope="module")
def stub_trace():
    stub = _Stub()
    snapshot = {"x": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)}
    keys = gp.checkpoint_keys(_Wrap(stub))
    return gp.trace_instruction(stub, snapshot, keys)


class _Wrap(tf.Module):
    def __init__(self, instruction):
        super().__init__()
        self.instructions = {instruction.name: instruction}


def test_trace_records_creation_dtype_and_the_operation_that_reads_each_value(
    stub_trace,
):
    captured = {c["attribute"]: c for c in stub_trace["captured"]}
    assert captured["stub/w"]["created"] == "float32"
    assert captured["stub/w"]["read_by"] == ["Mul:float32"]
    assert captured["stub/scale"]["created"] == "float64"
    assert captured["stub/scale"]["read_by"] == ["Mul:float64"]
    assert captured["stub/table"]["read_by"] == ["AddV2:float64"]


def test_trace_counts_the_casts_and_flags_the_mixed_precision(stub_trace):
    assert stub_trace["casts"] == {"float64->float32": 1, "float32->float64": 1}
    assert stub_trace["mixed_float_dtypes"] is True
    assert stub_trace["op_float_dtypes"] == {"float32": 2, "float64": 3}


def test_trace_names_the_input_by_its_dictionary_key(stub_trace):
    assert stub_trace["inputs"] == [
        {"key": "x", "dtype": "float64", "read_by": ["Cast:float64->float32"]}
    ]


def test_trace_gives_the_variable_a_checkpoint_key(stub_trace):
    rows = {a["path"]: a for a in stub_trace["attributes"]}
    assert rows["stub/w"]["checkpoint_key"] == (
        "model/instructions/stub/w/.ATTRIBUTES/VARIABLE_VALUE"
    )
    assert rows["stub/w"]["trainable"] is True
    assert rows["stub/scale"]["kind"] == "tensor"
    assert rows["stub/scale"]["checkpoint_key"] is None
    assert rows["stub/table"]["kind"] == "ndarray"


def test_a_pure_float64_instruction_is_not_flagged_as_mixed():
    class Pure(_Stub):
        def __call__(self, input_data, training=False, local=False):
            input_data[self.name] = input_data["x"] * 2.0

    snap = {"x": tf.constant([1.0], dtype=tf.float64)}
    out = gp.trace_instruction(Pure(), snap, {})
    assert out["mixed_float_dtypes"] is False
    assert out["casts"] == {}


def test_an_instruction_that_cannot_be_traced_is_reported_not_raised():
    class Broken(_Stub):
        def __call__(self, input_data, training=False, local=False):
            raise ValueError("no frwrd here")

    out = gp.trace_instruction(Broken(), {"x": tf.constant([1.0])}, {})
    assert out["trace_error"].startswith("ValueError: ")
    assert out["trace_error"].endswith("no frwrd here")
    assert out["class"] == "Broken"
    assert "captured" not in out


def test_variable_that_is_reachable_twice_gets_the_shortest_path():
    shared = _Stub("shared")
    holder = tf.Module()
    holder.a = tf.Module()
    holder.a.short = shared  # depth 2
    holder.z = tf.Module()
    holder.z.mid = tf.Module()
    holder.z.mid.inner = shared  # depth 3, visited first by a depth-first walk
    keys = gp.checkpoint_keys(holder)
    assert keys[id(shared.w)] == "model/a/short/w" + gp.SAVED_SUFFIX


def test_variable_records_report_variables_and_keys_that_do_not_match():
    class Untracked(tf.Module):
        """Holds a variable that the saver is not told about."""

        def __init__(self):
            super().__init__()
            self.v = tf.Variable([1.0], name="untracked")

        def _trackable_children(self, *args, **kwargs):
            return {}

    class Parent(tf.Module):
        def __init__(self):
            super().__init__()
            self.child = Untracked()
            self.saved_only = tf.Variable([2.0], name="saved_only")

    parent = Parent()
    # tracked by the saver but not an attribute, so absent from ``parent.variables``
    parent._track_trackable(tf.Variable([3.0]), name="hidden")  # noqa: SLF001
    report = gp.variable_records(parent)
    assert report["variables_without_key"] == ["untracked:0"]
    assert report["keys_without_variable"] == ["model/hidden" + gp.SAVED_SUFFIX]
    assert report["n_checkpoint_keys"] == 2


# ------------------------------------------------------------------ physics (real model)


@pytest.fixture(scope="module")
def probed():
    """Float32 and float64 probes of the 1-layer test model, with the models themselves."""
    out = {}
    atoms = gp.osn.load_structures(gp.TESTS / gp.osn.DEFAULT_STRUCTURES, 1)[0]
    for dtype in ("float64", "float32"):
        model = gp.build_probe_model(gp.resolve_yaml(YAML), dtype)
        out[dtype] = (model, gp.probe_model(model, atoms))
    return out


def test_every_variable_has_the_checkpoint_key_it_is_saved_under(probed, tmp_path):
    model, report = probed["float64"]
    prefix = tf.train.Checkpoint(model=model).write(str(tmp_path / "ckpt"))
    reader = tf.train.load_checkpoint(prefix)
    assert report["keys_without_variable"] == []
    assert report["variables_without_key"] == []
    by_name = {v.name: v for v in model.variables}
    for row in report["variables"]:
        saved = reader.get_tensor(row["checkpoint_key"])
        np.testing.assert_array_equal(saved, by_name[row["name"]].numpy())
        assert list(saved.shape) == row["shape"]


def test_probe_dtypes_are_those_of_the_built_variables(probed):
    for dtype, (model, report) in probed.items():
        assert {r["dtype"] for r in report["variables"] if r["trainable"]} == {dtype}
        assert len(report["variables"]) == len(model.variables)
        assert report["n_trainable"] == len(model.trainable_variables)


def test_float32_model_computes_radial_basis_and_harmonics_in_float64(probed):
    # pinned by tests/test_compute.py: SphericalHarmonics always float64, RadialBasis promoted
    _, report = probed["float32"]
    by_name = {i["name"]: i for i in report["instructions"]}
    assert set(by_name["Y"]["op_float_dtypes"]) == {"float64"}
    assert by_name["RadialBasis"]["op_float_dtypes"]["float64"] > 10
    rc = [
        c
        for c in by_name["RadialBasis"]["captured"]
        if c["attribute"] == "RadialBasis/rc"
    ]
    assert {c["created"] for c in rc} == {"float32"}
    assert "Cast:float32->float64" in {r for c in rc for r in c["read_by"]}
    assert "RadialBasis" in report["mixed_float_instructions"]


def test_float64_model_computes_and_stores_everything_in_float64(probed):
    _, report = probed["float64"]
    assert report["trace_errors"] == []
    for ins in report["instructions"]:
        assert set(ins["op_float_dtypes"]) <= {"float64"}, ins["name"]
        for cap in ins["captured"]:
            assert cap["created"] == "float64", (ins["name"], cap)


def test_float64_model_reads_the_swish_beta_as_a_float32_literal(probed):
    # Pins current behaviour (a finding of SPEC2): the activation constant ``beta`` is created
    # as a float32 literal and cast up, one per activated layer: R has three, the output MLP one.
    _, report = probed["float64"]
    literals = {
        i["name"]: i["literal_float_constants"]["float32"]
        for i in report["instructions"]
        if "float32" in i["literal_float_constants"]
    }
    assert literals == {"R": 3, "MLPOut2ScalarTarget": 1}
    assert set(report["mixed_float_instructions"]) == set(literals)


def test_float32_basis_function_casts_its_bond_input_down(probed):
    _, report = probed["float32"]
    a = next(i for i in report["instructions"] if i["name"] == "A")
    keys = {x["key"]: x["dtype"] for x in a["inputs"]}
    assert keys["Y"] == "float64"
    assert a["casts"].get("float64->float32") == 1


def test_timing_bond_count_equals_a_brute_force_neighbour_list(probed):
    from tensorpotential.tpmodel import extract_cutoff_and_elements
    from tests.neighbour_oracle import brute_force_pairs

    model, _ = probed["float64"]
    structures = gp.timing_structures(2, [])
    rows = gp.time_model(model, structures, repeats=2)
    cutoff, _, _ = extract_cutoff_and_elements(model.instructions)
    for (_, atoms), row in zip(structures, rows, strict=True):
        pairs = brute_force_pairs(
            atoms.positions,
            np.asarray(atoms.cell),
            atoms.get_chemical_symbols(),
            float(cutoff),
        )
        assert row["n_atoms"] == len(atoms)
        assert row["n_bonds"] == len(pairs)
        assert row["model_ms"] > 0
        assert row["neighbour_list_ms"] > 0
        assert row["neighbour_list_share"] == pytest.approx(
            row["neighbour_list_ms"] / (row["neighbour_list_ms"] + row["model_ms"])
        )


def test_supercells_multiply_the_atom_count_and_the_bond_count():
    structures = gp.timing_structures(1, [1, 2, 3])
    labels = [label for label, _ in structures]
    assert labels == ["s0", "s0x2", "s0x3"]
    n0 = len(structures[0][1])
    assert [len(a) for _, a in structures] == [n0, 8 * n0, 27 * n0]


def test_main_writes_one_json_file_per_yaml(tmp_path):
    code = gp.main([
        "write",
        str(tmp_path),
        "--yamls",
        YAML,
        "--dtypes",
        "float64",
        "--no-timing",
    ])
    assert code == 0
    data = json.loads((tmp_path / "probe_model_grace.json").read_text())
    assert data["label"] == "model_grace"
    assert data["random_weights"] is True
    assert "timings" not in data["dtypes"]["float64"]
    assert data["environment"]["tensorflow"] == tf.__version__


def test_reading_a_checkpoint_restores_its_weights(tmp_path):
    written = gp.build_probe_model(gp.resolve_yaml(YAML), "float64", seed=1)
    other = gp.build_probe_model(gp.resolve_yaml(YAML), "float64", seed=2)
    prefix = tf.train.Checkpoint(model=written).write(str(tmp_path / "ckpt"))
    restored = gp.build_probe_model(gp.resolve_yaml(YAML), "float64", checkpoint=prefix)
    differs = [
        not np.array_equal(a.numpy(), b.numpy())
        for a, b in zip(
            written.trainable_variables, other.trainable_variables, strict=True
        )
    ]
    assert any(differs)
    for a, b in zip(written.variables, restored.variables, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_a_checkpoint_of_another_model_is_refused(tmp_path):
    other = gp.build_probe_model(gp.resolve_yaml("model_grace_2L_omat.yaml"), "float64")
    prefix = tf.train.Checkpoint(model=other).write(str(tmp_path / "ckpt"))
    with pytest.raises(AssertionError):
        gp.build_probe_model(gp.resolve_yaml(YAML), "float64", checkpoint=prefix)


def test_main_with_timing_records_rows_and_the_foreign_cpu(tmp_path):
    code = gp.main([
        "write",
        str(tmp_path),
        "--yamls",
        YAML,
        "--dtypes",
        "float64",
        "--n-structures",
        "1",
        "--supercells",
        "--repeats",
        "1",
    ])
    assert code == 0
    section = json.loads((tmp_path / "probe_model_grace.json").read_text())["dtypes"][
        "float64"
    ]
    assert [row["structure"] for row in section["timings"]] == ["s0"]
    foreign = section["timings_foreign_cpu_cores"]
    assert foreign is None or foreign >= 0.0
