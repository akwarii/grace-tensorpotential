"""Tests of ``metadata_utils.py``: reading the metadata block of a ``model.yaml`` and resolving ``param_dtype``.

``param_dtype`` has several code defaults that disagree (SPEC2 records it): a yaml without a ``param_dtype``
key resolves to float64 here and in ``scripts/grace_utils.py``, while ``TensorPotential`` and
``utils.get_param_dtype_from_config`` default to float32. The disagreement is pinned as it is, not endorsed;
the tests of the other code paths are in ``test_grace_utils_export.py``, ``test_utils.py`` and
``test_tensorpot.py``.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
from pathlib import Path

import pytest
import tensorflow as tf
import yaml

from tensorpotential.metadata_utils import (
    get_dtype_by_name,
    read_model_metadata,
    resolve_param_dtype,
)

TESTS_DIR = Path(__file__).parent


def write_yaml(path: Path, data) -> str:
    path.write_text(yaml.dump(data, sort_keys=False))
    return str(path)


METADATA = {
    "float32": {"param_dtype": "float32", "tensorpotential_version": "x"},
    "float64": {"param_dtype": "float64"},
    "none": {"tensorpotential_version": "x", "saved_at": "2026-01-01T00:00:00Z"},
}


@pytest.fixture
def wrapped(tmp_path) -> dict[str, str]:
    """Wrapped-format yamls by the dtype their metadata names (``none``: no ``param_dtype`` key)."""
    return {
        kind: write_yaml(
            tmp_path / f"wrapped_{kind}.yaml", {"metadata": meta, "instructions": {}}
        )
        for kind, meta in METADATA.items()
    }


# ------------------------------------------------------------------------------ read_model_metadata


def test_read_metadata_of_the_wrapped_format(wrapped):
    assert read_model_metadata(wrapped["float32"]) == {
        "param_dtype": "float32",
        "tensorpotential_version": "x",
    }


def test_read_metadata_of_a_missing_file_is_empty(tmp_path):
    assert read_model_metadata(str(tmp_path / "missing.yaml")) == {}


def test_read_metadata_of_a_wrapped_file_without_a_metadata_key_is_empty(tmp_path):
    assert (
        read_model_metadata(write_yaml(tmp_path / "m.yaml", {"instructions": {}})) == {}
    )


def test_read_metadata_of_an_old_flat_dict_is_empty():
    assert read_model_metadata(str(TESTS_DIR / "model_grace_2L_omat.yaml")) == {}


def test_read_metadata_of_a_list_yaml_is_empty():
    assert read_model_metadata(str(TESTS_DIR / "model_grace.yaml")) == {}


def test_read_metadata_of_an_empty_file_is_empty(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    assert read_model_metadata(str(path)) == {}


def test_read_metadata_of_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("metadata: [unclosed")

    with pytest.raises(yaml.YAMLError):
        read_model_metadata(str(path))


# -------------------------------------------------------------------------------------------- dtype names


def test_dtype_names():
    assert get_dtype_by_name("float32") is tf.float32
    assert get_dtype_by_name("float64") is tf.float64


@pytest.mark.parametrize("name", ["float16", "double", "", "Float64"])
def test_unknown_dtype_name_is_a_key_error_naming_it(name):
    with pytest.raises(KeyError, match=f"Unknown dtype name {name}"):
        get_dtype_by_name(name)


# ------------------------------------------------------------------------------------ resolve_param_dtype


def test_resolve_reads_float32_from_the_metadata(wrapped):
    assert resolve_param_dtype(wrapped["float32"]) is tf.float32


def test_resolve_reads_float64_from_the_metadata(wrapped):
    assert resolve_param_dtype(wrapped["float64"]) is tf.float64


def test_resolve_explicit_override_wins_over_the_metadata(wrapped):
    assert resolve_param_dtype(wrapped["float32"], tf.float64) is tf.float64
    assert resolve_param_dtype(wrapped["float64"], tf.float32) is tf.float32


def test_resolve_explicit_override_does_not_need_the_file(tmp_path):
    assert resolve_param_dtype(str(tmp_path / "missing.yaml"), tf.float32) is tf.float32


def test_resolve_metadata_without_param_dtype_gives_float64_and_says_old_model(
    wrapped, caplog
):
    # PINNED, not endorsed: float64 here, while TensorPotential and get_param_dtype_from_config default to float32
    with caplog.at_level(logging.INFO):
        dtype = resolve_param_dtype(wrapped["none"])

    assert dtype is tf.float64
    assert "No param_dtype in model.yaml (old model)" in caplog.text


@pytest.mark.parametrize("filename", ["model_grace.yaml", "model_grace_2L_omat.yaml"])
def test_resolve_old_yamls_without_metadata_give_float64(filename):
    assert resolve_param_dtype(str(TESTS_DIR / filename)) is tf.float64


def test_resolve_missing_file_gives_float64_and_says_not_found(tmp_path, caplog):
    path = str(tmp_path / "missing.yaml")

    with caplog.at_level(logging.INFO):
        dtype = resolve_param_dtype(path)

    assert dtype is tf.float64
    assert f"model.yaml {path} not found" in caplog.text


def test_resolve_logs_where_the_dtype_was_inferred_from(wrapped, caplog):
    with caplog.at_level(logging.INFO):
        resolve_param_dtype(wrapped["float32"])

    assert (
        f"Inferred param_dtype: float32 ({tf.float32}) from {wrapped['float32']}"
        in caplog.text
    )


def test_resolve_unknown_dtype_in_the_metadata_is_a_key_error(tmp_path):
    path = write_yaml(
        tmp_path / "m.yaml",
        {"metadata": {"param_dtype": "float16"}, "instructions": {}},
    )

    with pytest.raises(KeyError, match="Unknown dtype name float16"):
        resolve_param_dtype(path)


def test_resolve_empty_param_dtype_counts_as_missing(tmp_path):
    path = write_yaml(
        tmp_path / "m.yaml", {"metadata": {"param_dtype": ""}, "instructions": {}}
    )

    assert resolve_param_dtype(path) is tf.float64


def test_a_model_saved_without_param_dtype_resolves_to_float64_whatever_it_was_built_with(
    tmp_path,
):
    # PINNED, not endorsed: save_instructions_dict(param_dtype=None) writes no key, so a float32 model loses its dtype
    from tensorpotential.instructions.base import save_instructions_dict
    from tensorpotential.potentials import presets

    instructions = presets.LINEAR(
        element_map={"Cu": 0}, lmax=1, n_rad_max=4, embedding_size=4
    ).get_instructions()
    path = str(tmp_path / "m.yaml")

    save_instructions_dict(path, instructions)

    assert resolve_param_dtype(path) is tf.float64
