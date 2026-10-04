"""Tests of the ``grace_utils aux_model`` command of ``tensorpotential/scripts/grace_utils.py``.

``aux_model`` adds the ``compute_energy`` function to the saved model and, for a GRACE-2L model
(one with a ``SingleParticleBasisFunctionEquivariantInd``), splits it at the communicated keys
(``-ck``) into ``forward_layer_1``, ``backward_layer_2`` and ``backward_layer_1``. It has no
``--aux`` argument; the tests pin that behaviour, which the documentation describes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import tensorflow as tf

from tensorpotential import TensorPotential
from tensorpotential.instructions.base import save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.scripts import grace_utils

_COMMUNICATED_KEYS_2L = ["I", "I_out_0_LN"]
_DOCS = Path(__file__).resolve().parent.parent / "docs" / "gracemaker" / "utilities.md"


def _save_model(directory: Path, preset_name: str, **config) -> tuple[Path, Path]:
    instructions = get_preset(preset_name)(**config).get_instructions()
    model = directory / "model.yaml"
    save_instructions_dict(str(model), instructions, param_dtype=tf.float32)
    checkpoint = directory / "checkpoint"
    TensorPotential(potential=str(model), param_dtype=tf.float32).save_checkpoint(
        checkpoint_name=str(checkpoint)
    )
    return model, checkpoint


def _run_aux_model(model: Path, checkpoint: Path, output: Path, *extra: str) -> None:
    argv = ["grace_utils", "-p", str(model), "-c", str(checkpoint)]
    argv += ["aux_model", "-o", str(output), *extra]
    with patch.object(sys, "argv", argv):
        grace_utils.main()


def _signatures(path: Path) -> set[str]:
    return set(tf.saved_model.load(str(path)).signatures)


@pytest.fixture(scope="module")
def model_1l(tmp_path_factory):
    return _save_model(
        tmp_path_factory.mktemp("aux_1l"),
        "GRACE_1LAYER_latest",
        element_map={"H": 0},
        rcut=4.5,
        n_rad_max=4,
        lmax=2,
    )


@pytest.fixture(scope="module")
def model_2l(tmp_path_factory):
    return _save_model(
        tmp_path_factory.mktemp("aux_2l"),
        "GRACE_2LAYER_latest",
        element_map={"Na": 0, "Cl": 1},
        rcut=6.0,
        max_order=2,
        lmax=[2, 1],
    )


def test_aux_model_on_a_1l_model_adds_compute_energy_and_nothing_else(
    tmp_path, model_1l
):
    output = tmp_path / "upgraded"
    _run_aux_model(*model_1l, output)
    assert _signatures(output) == {"compute", "compute_energy"}


def test_aux_model_on_a_2l_model_splits_it_at_the_communicated_keys(tmp_path, model_2l):
    output = tmp_path / "upgraded"
    _run_aux_model(*model_2l, output, "-ck", *_COMMUNICATED_KEYS_2L)
    assert {
        "compute",
        "compute_energy",
        "forward_layer_1",
        "backward_layer_2",
        "backward_layer_1",
    } <= _signatures(output)


def test_aux_model_on_a_2l_model_names_a_communicated_key_the_model_lacks(
    tmp_path, model_2l
):
    # The defaults of -ck (I_nl_LN, I) belong to some models only; a missing key stops the command.
    with pytest.raises(AssertionError, match="I_nl_LN not found in dependency graph"):
        _run_aux_model(*model_2l, tmp_path / "upgraded")


def test_aux_model_has_no_aux_argument(tmp_path, model_1l, capsys):
    with pytest.raises(SystemExit) as stop:
        _run_aux_model(*model_1l, tmp_path / "upgraded", "--aux", "energy_only")
    assert stop.value.code == 2
    assert "unrecognized arguments: --aux" in capsys.readouterr().err


# ------------------------------------------------------------------ documentation against the parser


def _help(monkeypatch, capsys, *words: str) -> str:
    """The ``--help`` text of ``grace_utils`` (or a subcommand) at the width of the documentation."""
    monkeypatch.setenv("COLUMNS", "80")
    with patch.object(sys, "argv", ["grace_utils", "-p", "model.yaml", *words, "-h"]):
        with pytest.raises(SystemExit) as stop:
            grace_utils.main()
    assert stop.value.code == 0
    return capsys.readouterr().out


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _docs_text() -> str:
    return _DOCS.read_text()


def test_documented_subcommand_line_is_the_help_of_the_parser(monkeypatch, capsys):
    listed = next(
        line.strip()
        for line in _help(monkeypatch, capsys).splitlines()
        if line.strip().startswith("aux_model ")
    )
    documented = [
        line.strip()
        for line in _docs_text().splitlines()
        if line.strip().startswith("aux_model ")
    ]
    assert documented == [listed]


def test_documented_aux_model_options_are_those_of_the_parser(monkeypatch, capsys):
    options = _help(monkeypatch, capsys, "aux_model").split("options:\n", 1)[1]
    # The documentation lists the options of each subcommand without -h.
    parser_options = options.split("  -o ", 1)[1]
    block = _docs_text().split("aux_model:\n", 1)[1].split("```", 1)[0]
    assert _normalised(block) == _normalised("-o " + parser_options)


def test_documentation_does_not_offer_the_removed_aux_argument():
    assert "--aux" not in _docs_text()
    assert "energy_only" not in _docs_text()


def test_documented_aux_model_command_is_accepted_by_the_parser():
    example = next(
        line
        for line in _docs_text().splitlines()
        if line.startswith("grace_utils") and " aux_model " in line
    )
    argv = example.split()
    # aux_model itself is replaced: only the parsing of the documented command is under test.
    with (
        patch.object(sys, "argv", argv),
        patch.object(grace_utils, "aux_model") as called,
    ):
        grace_utils.main()
    args = called.call_args.args[0]
    assert args.output_path == "/path/to/upgraded_model"
    assert args.communicated_keys == ["I", "I_out_0_LN"]
