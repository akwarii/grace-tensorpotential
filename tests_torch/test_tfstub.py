"""Tests of the TensorFlow stand-in (``tests_torch/tools/tfstub.py``), without TensorFlow.

Physical values: the tables the unmodified TF code builds under the stand-in are the tables TensorFlow itself dumped
into the golden anchors (every key, dtype, shape and value), and its variables have the names, shapes, dtypes and
trainability of the anchors' manifests, for both models in both parameter dtypes and for the faithful widths. Logic:
the numpy-backed operations follow TensorFlow's dtype rules, everything else is a placeholder that the walk reports,
the context leaves no module behind, refuses a process that has TensorFlow, and is re-entrant.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tests_torch.golden import GOLDEN, TINY_F32, TINY_F64, load
from tests_torch.tools import tfstub

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests_torch" / "fixtures"
DTYPE = {"f64": "float64", "f32": "float32"}


def _model_of(anchor: str) -> tuple[Path, str]:
    model, tier, dtype = anchor.split("_")
    return GOLDEN / "yamls" / f"{model}_{tier}.yaml", DTYPE[dtype]


@pytest.fixture(scope="module", params=TINY_F64 + TINY_F32)
def built(request: pytest.FixtureRequest) -> tuple[str, dict]:
    path, dtype = _model_of(request.param)
    return request.param, tfstub.load_model(path, float_dtype=dtype)


@pytest.mark.golden
def test_tables_equal_the_tensorflow_dump(built):
    anchor, model = built
    expected = {k: v for k, v in load(anchor).items() if k.startswith("tables/")}
    got = {
        f"tables/{key}": value
        for instruction in model.values()
        for key, value in tfstub.arrays(instruction).items()
    }
    assert sorted(got) == sorted(expected)
    for key, value in expected.items():
        assert (got[key].dtype, got[key].shape) == (value.dtype, value.shape), key
        np.testing.assert_array_equal(got[key], value, err_msg=key)


def _manifest_variables(path: Path) -> dict[str, tfstub.VariableInfo]:
    rows = json.loads(path.read_text())["variables"]
    return {
        r["key"]: tfstub.VariableInfo(tuple(r["shape"]), r["dtype"], r["trainable"])
        for r in rows
    }


def _stand_in_variables(model: dict) -> dict[str, tfstub.VariableInfo]:
    return {k: v for ins in model.values() for k, v in tfstub.variables(ins).items()}


@pytest.mark.golden
def test_variables_equal_the_manifest(built):
    anchor, model = built
    assert _stand_in_variables(model) == _manifest_variables(GOLDEN / f"{anchor}.json")


@pytest.mark.golden
@pytest.mark.parametrize("anchor", ["omat_faithful_f64", "large_faithful_f32"])
def test_variables_equal_the_manifest_at_the_real_widths(anchor):
    path, dtype = _model_of(anchor)
    model = tfstub.load_model(FIXTURES / "yamls" / path.name, float_dtype=dtype)
    assert _stand_in_variables(model) == _manifest_variables(
        FIXTURES / f"{anchor}.json"
    )


def test_only_activations_are_placeholders(built):
    _, model = built
    found = [p for ins in model.values() for p in tfstub.placeholders(ins)]
    assert found
    assert all(p.endswith("/activation") for p in found)


def test_without_build_only_init_variables_and_init_tables_exist():
    path, _ = _model_of(TINY_F64[1])
    model = tfstub.load_model(path)
    assert sorted(_stand_in_variables(model)) == [
        "Z/element_map_index",
        "Z/element_map_symbols",
    ]
    assert "YI/cg" in tfstub.arrays(model["YI"])
    assert "Y/sh/alm" not in tfstub.arrays(model["Y"])


def _leftovers() -> list[str]:
    return [
        n
        for n in sys.modules
        if n.split(".")[0] in tfstub.ROOTS
        or n.startswith("tensorpotential.instructions")
    ]


def test_the_context_leaves_no_module_behind():
    assert not _leftovers()
    with tfstub.stand_in() as tf:
        assert tf.constant(1).dtype == np.int32
        with tfstub.stand_in() as inner:
            assert inner is tf
        y = tfstub.construct("SphericalHarmonic", vhat="v", name="Y", lmax=2)
    assert not tfstub.active()
    assert not _leftovers()
    assert not hasattr(sys.modules["tensorpotential"], "instructions")
    assert list(y.coupling_meta_data["l"]) == [0, 1, 1, 1, 2, 2, 2, 2, 2]


def test_refuses_a_process_with_tensorflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    with pytest.raises(tfstub.StandInError, match="without TensorFlow"):  # noqa: SIM117
        with tfstub.stand_in():
            pass
    assert not tfstub.active()


def test_construct_needs_the_context():
    with pytest.raises(tfstub.StandInError, match="open the stand-in"):
        tfstub.construct("SphericalHarmonic", vhat="v", name="Y", lmax=2)


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (1.5, None, "float32"),
        ([1, 2], None, "int32"),
        (np.float64(1.5), None, "float64"),
        (np.arange(3), None, "int64"),
        (1, "float64", "float64"),
        (["Cu", "H"], None, "<U2"),
    ],
)
def test_constant_follows_the_dtype_rules_of_tensorflow(value, dtype, expected):
    assert tfstub.constant(value, dtype=dtype).dtype == np.dtype(expected)


def test_numpy_backed_operations():
    assert tfstub.constant(2.0, shape=[2, 2]).tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert tfstub.range_(3).dtype == np.int32
    assert tfstub.range_(1, 3, dtype=np.float64).tolist() == [1.0, 2.0]
    assert tfstub.one_hot(1, 3, dtype=np.float64).tolist() == [0.0, 1.0, 0.0]
    assert tfstub.concat([[1], [2, 3]], axis=0).tolist() == [1, 2, 3]
    assert tfstub.stack([1, 2]).tolist() == [1, 2]
    assert tfstub.reshape(np.arange(4), [2, 2]).shape == (2, 2)
    assert tfstub.cast([1.7], np.int32).tolist() == [1]
    assert tfstub.zeros([2], np.int32).dtype == np.int32
    assert tfstub.ones([1]).dtype == np.float32
    assert isinstance(tfstub.sqrt(4.0), np.ndarray)
    with tfstub.stand_in() as tf:
        w = tf.Variable(
            tf.random.normal([2, 3], stddev=1.0, dtype=tf.float64), name="w"
        )
    assert (w.shape, w.dtype, w.trainable, w.name) == ((2, 3), np.float64, True, "w")
    assert not w.numpy().any()
    assert tfstub.Variable(lambda: [1, 2], trainable=False).dtype == np.int32


def test_placeholders_are_permissive():
    p = tfstub.Placeholder(1, key=2)
    assert isinstance(p.any.chain(3)[4], tfstub.Placeholder)
    assert list(p) == []
    assert tfstub.Placeholder.anything["x"] is tfstub.Placeholder
    with pytest.raises(AttributeError):
        _ = p.__wrapped__


def test_module_walks():
    with tfstub.stand_in() as tf:
        inner = tf.Module(name="inner")
        inner.w = tf.Variable(np.zeros(2), trainable=False)
        outer = tf.Module(name="outer")
        outer.layers = [inner, {"b": tf.Variable(np.ones(1))}]
        outer.again = inner
        assert outer.submodules == (inner,)
        assert len(outer.variables) == 2
        assert outer.trainable_variables == (outer.layers[1]["b"],)
        assert set(tfstub.variables(outer)) == {
            "outer/again/w",
            "outer/layers/1/b",
        }  # shortest path


def test_arrays_flatten_containers_one_level_and_skip_other_instructions():
    with tfstub.stand_in() as tf:
        y = tfstub.construct("SphericalHarmonic", vhat="v", name="Y", lmax=1)
        outer = tf.Module(name="outer")
        outer.maps = {"a": np.zeros(1), "deep": {"b": np.zeros(1)}}
        outer.seq = [np.ones(2)]
        outer.angular = y
        y.table = np.zeros(3)
        assert sorted(tfstub.arrays(outer)) == ["outer/maps[a]", "outer/seq[0]"]
