"""``tensorpotential.scripts.grace_models``: the ``grace_models`` command line (``main``) and its dispatch.

Logic layer: no sub-command prints the help, each sub-command reaches its function with the parsed options, an unknown
model name is reported, usage errors exit with 2. The downloads are replaced at the network boundary by recorders
(``get_or_download_model``, ``get_or_download_kokkos``, ``get_or_download_checkpoint``); nothing is read from or written to
the user's cache by these tests (``list`` only reads the registry and checks which files exist).
"""

from __future__ import annotations

import pytest

from tensorpotential.calculator import foundation_models
from tensorpotential.scripts import grace_models
from tests import cli_help
from tests.cli_help import CASES, normalise


def run_models(*argv: str) -> tuple[int, str, str]:
    return cli_help.run_main(grace_models.main, "grace_models", list(argv))


@pytest.fixture
def downloads(monkeypatch) -> list[tuple[str, str]]:
    """Record the download calls instead of fetching from the network."""
    calls: list[tuple[str, str]] = []
    for kind, attr in (
        ("model", "get_or_download_model"),
        ("kokkos", "get_or_download_kokkos"),
        ("checkpoint", "get_or_download_checkpoint"),
    ):
        monkeypatch.setattr(
            foundation_models, attr, lambda name, kind=kind: calls.append((kind, name))
        )
    return calls


KNOWN = foundation_models.MODELS_NAME_LIST[0]


def test_without_a_command_the_help_is_printed():
    code, out, _ = run_models()
    assert code == 0
    reference = next(case for case in CASES if case.case_id == "grace_models").reference
    assert normalise(out) == normalise(reference.read_text(encoding="utf-8"))


def test_an_unknown_command_is_a_usage_error():
    code, _, err = run_models("frobnicate")
    assert code == 2
    assert "invalid choice: 'frobnicate'" in err


def test_info_needs_a_model_name():
    code, _, err = run_models("info")
    assert code == 2
    assert "the following arguments are required: model_name" in err


def test_download_of_an_unknown_model_is_reported_and_downloads_nothing(downloads):
    code, out, _ = run_models("download", "no_such_model")
    assert code == 0
    assert out == "Model no_such_model not found in available models.\n"
    assert downloads == []


def test_download_of_a_known_model(downloads):
    code, out, _ = run_models("download", KNOWN)
    assert code == 0
    assert downloads == [("model", KNOWN)]
    assert f"Downloading model: {KNOWN}" in out
    assert f"Model {KNOWN} downloaded successfully." in out


def test_download_with_kokkos_also_fetches_the_kokkos_file(downloads):
    run_models("download", KNOWN, "--kokkos")
    assert downloads == [("model", KNOWN), ("kokkos", KNOWN)]


def test_download_all_fetches_every_registered_model_in_order(downloads):
    run_models("download", "all")
    assert downloads == [("model", name) for name in foundation_models.MODELS_NAME_LIST]


def test_checkpoint_of_a_known_model(downloads):
    _, out, _ = run_models("checkpoint", KNOWN)
    assert downloads == [("checkpoint", KNOWN)]
    assert f"Downloading checkpoint for {KNOWN}" in out
    assert f"Checkpoint for {KNOWN} downloaded successfully." in out


def test_checkpoint_all_fetches_every_registered_checkpoint(downloads):
    run_models("checkpoint", "all")
    assert downloads == [
        ("checkpoint", name) for name in foundation_models.MODELS_NAME_LIST
    ]


def test_checkpoint_of_an_unknown_model_is_reported(downloads):
    _, out, _ = run_models("checkpoint", "no_such_model")
    assert out == "Model no_such_model not found in available models.\n"
    assert downloads == []


def test_list_names_every_registered_model_and_their_number():
    code, out, _ = run_models("list")
    assert code == 0
    assert f"Available models ({len(foundation_models.MODELS_METADATA)}):" in out
    for name in foundation_models.MODELS_METADATA:
        assert name in out


def test_list_ascii_uses_plain_glyphs_only():
    _, out, _ = run_models("list", "--ascii")
    table = out.split("\n", 2)[2]
    assert "✓" not in table
    assert "✗" not in table


def test_list_verbose_prints_one_block_per_model():
    code, out, _ = run_models("list", "-v")
    assert code == 0
    assert out.startswith("Available models:\n" + "=" * 80)
    for name in foundation_models.MODELS_METADATA:
        assert name in out


def test_the_verbose_list_differs_from_the_table():
    code, out, _ = run_models("list")
    code_v, out_v, _ = run_models("list", "--verbose")
    assert code == code_v == 0
    assert out != out_v


def test_main_returns_zero_with_and_without_a_command_and_does_not_read_sys_argv(monkeypatch, capsys):
    cli_help.poison_sys_argv(monkeypatch)
    assert grace_models.main([]) == 0
    assert grace_models.main(["download", "no_such_model"]) == 0
    assert "usage: grace_models" in capsys.readouterr().out
