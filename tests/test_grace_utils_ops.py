"""Characterization of the model operations of ``grace_utils`` that CLI2 moves into ``model_ops``.

Covers ``update_model``, ``resave_checkpoint``, ``reduce_elements``, ``cast_model_param``, ``model_summary``,
``export_model`` (FS yaml branch) and ``preprocess_args`` as the command runs them (``grace_utils.main`` on the
test yamls, in process), before they get explicit keyword arguments. The physical check is the energy of a
dimer: converting, resaving or reducing a model must not change what it predicts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml

from tensorpotential.instructions.base import load_instructions
from tensorpotential.scripts import grace_utils
from tests.saved_models import (
    FS_CONFIG,
    dimer_energy,
    run_grace_utils,
    save_seeded_model,
)

_run = run_grace_utils


@pytest.fixture(scope="module")
def two_elements(tmp_path_factory):
    return save_seeded_model(tmp_path_factory.mktemp("two"), {"H": 0, "He": 1})


@pytest.fixture
def one_element(tmp_path):
    return save_seeded_model(tmp_path, {"H": 0})


@pytest.fixture
def list_model(tmp_path):
    return save_seeded_model(tmp_path, {"H": 0}, as_list=True)


_FLOAT32 = ("--param_dtype", "float32")


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


@pytest.mark.parametrize(
    ("suffix", "checkpoint", "expected_suffix", "expected_checkpoint"),
    [
        ("converted", "c/ckpt.index", "-converted", "c/ckpt"),
        ("-already", "c/ckpt", "-already", "c/ckpt"),
        ("x", None, "-x", None),
        ("x", "", "-x", ""),
    ],
)
def test_preprocess_args_prefixes_the_suffix_and_strips_the_index_extension(
    suffix, checkpoint, expected_suffix, expected_checkpoint
):
    args = _args(output_suffix=suffix, checkpoint_path=checkpoint)
    grace_utils.preprocess_args(args)
    assert (args.output_suffix, args.checkpoint_path) == (
        expected_suffix,
        expected_checkpoint,
    )


def test_update_model_converts_a_list_model_and_keeps_its_prediction(
    list_model, tmp_path
):
    model, checkpoint = list_model
    assert isinstance(load_instructions(str(model)), list)
    before = dimer_energy(model, checkpoint, "H")
    # a list model has no metadata, so the precision of its float32 checkpoint is given
    assert _run(model, checkpoint, "update_model", options=_FLOAT32) == 0
    new_model = tmp_path / "model-converted.yaml"
    new_checkpoint = tmp_path / "checkpoint-converted"
    assert isinstance(load_instructions(str(new_model)), dict)
    assert Path(f"{new_checkpoint}.index").exists()
    assert np.isclose(
        dimer_energy(new_model, new_checkpoint, "H"), before, rtol=1e-5, atol=1e-6
    )


def test_update_model_stops_with_status_zero_on_a_model_that_is_already_a_dict(
    one_element, tmp_path
):
    model, checkpoint = one_element
    with pytest.raises(SystemExit) as stop:
        _run(model, checkpoint, "update_model")
    assert stop.value.code == 0
    assert not (tmp_path / "model-converted.yaml").exists()


def test_update_model_accepts_the_index_file_as_checkpoint_and_a_custom_suffix(
    list_model, tmp_path
):
    model, checkpoint = list_model
    index = Path(f"{checkpoint}.index")
    assert _run(model, index, "update_model", options=(*_FLOAT32, "-os", "v2")) == 0
    assert (tmp_path / "model-v2.yaml").exists()
    assert Path(f"{tmp_path / 'checkpoint-v2'}.index").exists()


def test_resave_checkpoint_writes_a_checkpoint_with_the_same_weights(
    one_element, tmp_path
):
    model, checkpoint = one_element
    before = dimer_energy(model, checkpoint, "H")
    assert _run(model, checkpoint, "resave_checkpoint") == 0
    new_checkpoint = tmp_path / "checkpoint-converted"
    assert Path(f"{new_checkpoint}.index").exists()
    assert np.isclose(
        dimer_energy(model, new_checkpoint, "H"), before, rtol=1e-5, atol=1e-6
    )


def _element_maps(model: Path) -> list[dict]:
    instructions = yaml.safe_load(model.read_text())["instructions"].values()
    return [ins["element_map"] for ins in instructions if "element_map" in ins]


def test_reduce_elements_keeps_the_energy_of_the_kept_element_and_drops_the_other(
    two_elements, tmp_path
):
    model, checkpoint = two_elements
    work = tmp_path / "w"
    work.mkdir()
    (work / "model.yaml").write_text(model.read_text())
    for part in checkpoint.parent.glob("checkpoint*"):
        (work / part.name).write_bytes(part.read_bytes())
    before = dimer_energy(work / "model.yaml", work / "checkpoint", "He")
    assert (
        _run(work / "model.yaml", work / "checkpoint", "reduce_elements", "-e", "He")
        == 0
    )
    reduced = work / "model-converted.yaml"
    maps = _element_maps(reduced)
    assert maps
    assert all(list(m) == ["He"] for m in maps)
    after = dimer_energy(reduced, work / "checkpoint-converted", "He")
    assert np.isclose(after, before, rtol=1e-5, atol=1e-6)


def test_reduce_elements_is_refused_for_a_list_model_with_status_zero(
    list_model, tmp_path
):
    model, checkpoint = list_model
    with pytest.raises(SystemExit) as stop:
        grace_utils.main([
            "-p",
            str(model),
            "-c",
            str(checkpoint),
            "reduce_elements",
            "-e",
            "H",
        ])
    assert stop.value.code == 0
    assert not (tmp_path / "model-converted.yaml").exists()


def test_cast_model_param_writes_a_float64_model_directory_and_a_saved_model(
    one_element, tmp_path, monkeypatch
):
    model, checkpoint = one_element
    monkeypatch.chdir(tmp_path)
    assert (
        _run(model, checkpoint, "cast_model_param", "-curr", "fp32", "-to", "fp64") == 0
    )
    casted = tmp_path / "casted-converted"
    meta = yaml.safe_load((casted / "model.yaml").read_text())["metadata"]
    assert meta["param_dtype"] == "float64"
    assert (casted / "checkpoint.index").exists()
    assert "compute" in tf.saved_model.load(str(tmp_path / "casted_model")).signatures


def test_cast_model_param_refuses_an_unknown_precision(one_element):
    model, checkpoint = one_element
    args = _args(
        potential=str(model),
        checkpoint_path=str(checkpoint),
        output_suffix="c",
        curr="fp16",
        to="fp64",
    )
    with pytest.raises(AssertionError, match="Invalid dtype"):
        grace_utils.cast_model_param(args)


def test_summary_logs_the_model_summary_at_the_requested_verbosity(one_element, caplog):
    model, _ = one_element
    with caplog.at_level(logging.INFO):
        assert _run(model, None, "summary", "-v", "1") == 0
    assert any(
        "Summary (verbose=1) is:" in record.getMessage() for record in caplog.records
    )


def test_export_to_fs_yaml_appends_the_extension_and_writes_the_model(tmp_path):
    model, checkpoint = save_seeded_model(
        tmp_path, {"Al": 0, "Li": 1}, preset="FS", config=FS_CONFIG
    )
    target = tmp_path / "fs_model"
    assert _run(model, checkpoint, "export", "-sf", "-n", str(target)) == 0
    exported = Path(f"{target}.yaml")
    assert exported.exists()
    assert yaml.safe_load(exported.read_text())


def test_export_to_fs_yaml_keeps_an_explicit_yml_name(tmp_path):
    model, checkpoint = save_seeded_model(
        tmp_path, {"Al": 0, "Li": 1}, preset="FS", config=FS_CONFIG
    )
    target = tmp_path / "fs_model.yml"
    assert _run(model, checkpoint, "export", "-sf", "-n", str(target)) == 0
    assert target.exists()
    assert not Path(f"{target}.yaml").exists()


def test_export_writes_a_saved_model_with_the_compute_signatures(one_element, tmp_path):
    model, checkpoint = one_element
    target = tmp_path / "exported_model"
    assert _run(model, checkpoint, "export", "-n", str(target)) == 0
    signatures = tf.saved_model.load(str(target)).signatures
    assert "compute" in signatures
    assert "compute_energy" in signatures


def test_reduce_elements_keeps_every_requested_element(tmp_path):
    model, checkpoint = save_seeded_model(tmp_path, {"H": 0, "He": 1, "Li": 2})
    assert _run(model, checkpoint, "reduce_elements", "-e", "He", "Li") == 0
    maps = _element_maps(tmp_path / "model-converted.yaml")
    assert maps
    assert all(list(m) == ["He", "Li"] for m in maps)
