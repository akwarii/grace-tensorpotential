"""``tensorpotential.scripts.grace_uq``: the ``grace_uq <subcommand>`` entry point, driven through ``main``.

Logic layer: the top-level help, an unknown subcommand, the routing of the remaining arguments to the sub-command and
the exit code that ``main`` raises. The text of the help is compared with the reference in ``tests/test_cli_help.py``.
The sub-commands themselves are tested in ``tests/test_uq_cli.py``.
"""

from __future__ import annotations

import pytest

from tensorpotential.scripts import grace_uq
from tests import cli_help

from .test_uq_cli import _write_synthetic_info_artifact


def run_grace_uq(*argv: str) -> tuple[int, str, str]:
    return cli_help.run_main(grace_uq.main, "grace_uq", list(argv))


@pytest.mark.parametrize("argv", [(), ("-h",), ("--help",)])
def test_top_level_help_lists_the_four_subcommands_and_exits_0(argv):
    code, out, err = run_grace_uq(*argv)
    assert (code, err) == (0, "")
    for sub in ("build", "info", "predict", "select"):
        assert f"  {sub}" in out
    assert "grace_uq <subcommand> [options]" in out


def test_help_is_plain_text_when_the_output_is_not_a_terminal():
    _, out, _ = run_grace_uq("--help")
    assert "\033[" not in out


def test_unknown_subcommand_exits_2_and_names_the_choices():
    code, out, err = run_grace_uq("frobnicate")
    assert code == 2
    assert out == ""
    assert err == "grace_uq: unknown subcommand 'frobnicate'. Use one of: build, info, predict, select\n"


def test_options_before_a_subcommand_are_an_unknown_subcommand():
    code, _, err = run_grace_uq("--n-workers", "2", "build")
    assert code == 2
    assert "unknown subcommand '--n-workers'" in err


def test_the_arguments_after_the_subcommand_reach_it(tmp_path):
    artifact = str(tmp_path / "art.npz")
    _write_synthetic_info_artifact(artifact, n_elements=2, K=3, D=4)
    code, out, err = run_grace_uq("info", artifact)
    assert (code, err) == (0, "")
    assert "feature_dim D" in out


def test_a_subcommand_error_is_the_exit_code_with_its_message(tmp_path):
    code, _, err = run_grace_uq("info", str(tmp_path / "absent.npz"))
    assert code == 2
    assert f"grace_uq info: error: Artifact not found: {tmp_path / 'absent.npz'}" in err


def test_a_subcommand_usage_error_exits_2():
    code, _, err = run_grace_uq("info", "--no-such-option")
    assert code == 2
    assert "usage: grace_uq info" in err
