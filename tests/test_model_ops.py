"""``tensorpotential/model_ops.py``: the model operations behind ``grace_utils``, called with keyword arguments.

The commands (the ``Namespace`` handlers) are tested in ``test_grace_utils_ops.py``; here the operations are called as
library code, and ``resolve_param_dtype``, the default of the precision of a yaml without a ``param_dtype`` key, is
pinned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml

from tensorpotential import model_ops
from tensorpotential.scripts import grace_utils
from tests.saved_models import dimer_energy, save_seeded_model


def _yaml_with_metadata(tmp_path, metadata):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.dump({"metadata": metadata, "instructions": {}}))
    return str(path)


# ------------------------------------------------------------ resolve_param_dtype (a code default of param_dtype)
# The second of the code paths that decide param_dtype for a yaml without a param_dtype key; it agrees with
# metadata_utils.resolve_param_dtype (float64) and disagrees with TensorPotential and get_param_dtype_from_config
# (float32). PINNED, not endorsed (see test_metadata_utils.py).


@pytest.mark.parametrize(("name", "dtype"), [("float32", tf.float32), ("float64", tf.float64)])
def test_resolve_param_dtype_command_line_value_wins(tmp_path, name, dtype):
    other = "float64" if name == "float32" else "float32"
    path = _yaml_with_metadata(tmp_path, {"param_dtype": other})

    assert model_ops.resolve_param_dtype(name, path) is dtype


def test_resolve_param_dtype_command_line_value_does_not_need_the_file(tmp_path):
    assert model_ops.resolve_param_dtype("float32", str(tmp_path / "missing.yaml")) is tf.float32


@pytest.mark.parametrize(("name", "dtype"), [("float32", tf.float32), ("float64", tf.float64)])
def test_resolve_param_dtype_reads_the_metadata(tmp_path, name, dtype):
    path = _yaml_with_metadata(tmp_path, {"param_dtype": name})

    assert model_ops.resolve_param_dtype(None, path) is dtype


def test_resolve_param_dtype_without_the_key_gives_float64(tmp_path):
    path = _yaml_with_metadata(tmp_path, {"tensorpotential_version": "x"})

    assert model_ops.resolve_param_dtype(None, path) is tf.float64


def test_resolve_param_dtype_missing_file_gives_float64(tmp_path):
    # unlike metadata_utils.resolve_param_dtype it does not test for the file: read_model_metadata gives {}
    assert model_ops.resolve_param_dtype(None, str(tmp_path / "missing.yaml")) is tf.float64


def test_resolve_param_dtype_unknown_name_is_a_key_error(tmp_path):
    with pytest.raises(KeyError, match="Unknown dtype name float16"):
        model_ops.resolve_param_dtype("float16", str(tmp_path / "model.yaml"))


def test_the_command_handler_passes_the_attribute_of_the_namespace_on(tmp_path):
    path = _yaml_with_metadata(tmp_path, {"param_dtype": "float32"})

    assert grace_utils.resolve_param_dtype(argparse.Namespace(param_dtype="float64"), path) is tf.float64
    assert grace_utils.resolve_param_dtype(argparse.Namespace(param_dtype=None), path) is tf.float32
    # a Namespace without the attribute counts as unset
    assert grace_utils.resolve_param_dtype(argparse.Namespace(), path) is tf.float32


# ------------------------------------------------------------ the operations as library functions


def test_resave_checkpoint_is_callable_without_a_namespace_and_keeps_the_prediction(tmp_path):
    model, checkpoint = save_seeded_model(tmp_path, {"H": 0})
    before = dimer_energy(model, checkpoint, "H")

    model_ops.resave_checkpoint(
        potential=str(model),
        checkpoint_path=str(checkpoint),
        output_suffix="-copy",
        param_dtype_name="float32",
    )

    assert Path(f"{tmp_path / 'checkpoint-copy'}.index").exists()
    assert np.isclose(dimer_energy(model, tmp_path / "checkpoint-copy", "H"), before, rtol=1e-5, atol=1e-6)


def test_reduce_elements_keeps_only_the_requested_elements_when_called_with_keywords(tmp_path):
    model, checkpoint = save_seeded_model(tmp_path, {"H": 0, "He": 1})
    before = dimer_energy(model, checkpoint, "He")

    model_ops.reduce_elements(
        potential=str(model),
        checkpoint_path=str(checkpoint),
        output_suffix="-he",
        elements=["He"],
        param_dtype_name="float32",
    )

    reduced = yaml.safe_load((tmp_path / "model-he.yaml").read_text())["instructions"]
    assert all(list(ins["element_map"]) == ["He"] for ins in reduced.values() if "element_map" in ins)
    after = dimer_energy(tmp_path / "model-he.yaml", tmp_path / "checkpoint-he", "He")
    assert np.isclose(after, before, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name",
    [
        "update_model",
        "resave_checkpoint",
        "reduce_elements",
        "cast_model_param",
        "export_model",
        "export_kokkos",
        "model_summary",
        "aux_model",
    ],
)
def test_an_operation_takes_keyword_arguments_only(name):
    with pytest.raises(TypeError):
        getattr(model_ops, name)(argparse.Namespace())
