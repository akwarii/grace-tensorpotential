"""``tests/cli_help.py``: the reference texts of the console commands and the comparison that guards them.

Logic layer: the normaliser folds exactly the two differences between Python versions and nothing else, the case
table matches the committed files, the capture reports a failing ``--help``. The text of every real parser is
compared with its reference by ``test_help_text_is_unchanged``, one case per command and sub-command.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import pytest

from tests import cli_help
from tests.cli_help import CASES, REFERENCE_DIR, HelpCase, normalise


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_help_text_is_unchanged(case: HelpCase) -> None:
    """The text a command prints for ``--help`` equals the reference written before CLI1 (no option, default, choice
    or help string moved)."""
    assert normalise(cli_help.capture_help(case)) == normalise(
        case.reference.read_text(encoding="utf-8")
    )


def _help_of(parser: argparse.ArgumentParser) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        parser.print_help()
    return out.getvalue()


OLD_FORM = """\
usage: toy [-h] [-o OUTPUT] [-e ELEMENTS [ELEMENTS ...]] [-c {a,b}]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file
  -e ELEMENTS [ELEMENTS ...], --elements ELEMENTS [ELEMENTS ...]
                        elements
  -c {a,b}, --choice {a,b}
                        which
"""

NEW_FORM = """\
usage: toy [-h] [-o OUTPUT] [-e ELEMENTS [ELEMENTS ...]] [-c {a,b}]

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   output file
  -e, --elements ELEMENTS [ELEMENTS ...]
                        elements
  -c, --choice {a,b}    which
"""


def test_normalise_folds_the_python_313_option_form_and_the_wrapping():
    assert normalise(OLD_FORM) == normalise(NEW_FORM)


def test_normalise_keeps_every_word():
    changed = NEW_FORM.replace("output file", "output files")
    assert normalise(changed) != normalise(NEW_FORM)
    renamed = NEW_FORM.replace("--output", "--out")
    assert normalise(renamed) != normalise(NEW_FORM)


def test_normalise_does_not_merge_two_different_metavars():
    differing = "-o OUTPUT, --output FILE  the output"
    assert normalise(differing) == "-o OUTPUT, --output FILE the output"


def test_normalise_is_idempotent():
    once = normalise(OLD_FORM)
    assert normalise(once) == once


def test_a_real_argparse_help_normalises_to_the_same_text_whatever_the_width(
    monkeypatch,
):
    parser = argparse.ArgumentParser(prog="toy", description="d")
    parser.add_argument("-o", "--output", help="output file name " * 6)
    parser.add_argument("-e", "--elements", nargs="+", help="elements")
    texts = []
    for width in ("60", "100", "200"):
        monkeypatch.setenv("COLUMNS", width)
        texts.append(normalise(_help_of(parser)))
    assert len(set(texts)) == 1


def test_case_ids_are_unique_and_name_a_reference_file():
    ids = [case.case_id for case in CASES]
    assert len(ids) == len(set(ids))
    assert all(case.reference.is_file() for case in CASES)


def test_every_committed_reference_belongs_to_a_case():
    committed = {path.name for path in REFERENCE_DIR.glob("*.txt")}
    assert committed == {case.reference.name for case in CASES}


def test_every_reference_names_its_command_in_the_first_line():
    for case in CASES:
        first = case.reference.read_text(encoding="utf-8").splitlines()[0]
        assert case.command in first, case.case_id


def test_the_cases_cover_the_ten_console_scripts():
    commands = {case.command for case in CASES}
    assert commands == {
        "gracemaker",
        "grace_models",
        "grace_collect",
        "extxyz2df",
        "df2extxyz",
        "grace_preprocess",
        "grace_predict",
        "grace_utils",
        "grace_uq",
        "grace_dashboard",
    }


def test_run_main_reports_exit_code_output_and_the_command_name():
    def main(argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--n", type=int)
        parser.parse_args(argv)
        return 3

    code, out, err = cli_help.run_main(main, "some_command", ["--help"])
    assert (code, err) == (0, "")
    assert out.startswith("usage: some_command")
    code, out, err = cli_help.run_main(main, "some_command", ["--n", "x"])
    assert code == 2
    assert "invalid int value" in err
    assert cli_help.run_main(main, "some_command", ["--n", "1"])[0] == 3


def test_run_main_reads_sys_argv_for_a_main_without_arguments_and_restores_the_environment():
    seen = {}

    def main():
        seen["argv"] = list(sys.argv)
        seen["columns"] = os.environ["COLUMNS"]

    before_argv = list(sys.argv)
    before_columns = os.environ.get("COLUMNS")
    cli_help.run_main(main, "cmd", ["a", "b"])
    assert seen == {"argv": ["cmd", "a", "b"], "columns": cli_help.COLUMNS}
    assert sys.argv == before_argv
    assert os.environ.get("COLUMNS") == before_columns


def test_capture_help_raises_when_help_does_not_exit_with_zero(monkeypatch):
    def broken(argv=None):
        raise SystemExit(2)

    case = HelpCase("broken", (), "tests.test_cli_help", "broken_main")
    monkeypatch.setattr(HelpCase, "main", lambda self: broken)
    with pytest.raises(RuntimeError, match="exited with 2"):
        cli_help.capture_help(case)


def test_run_main_turns_a_system_exit_message_into_stderr_and_exit_code_1():
    def main(argv=None):
        raise SystemExit("something is wrong")

    assert cli_help.run_main(main, "cmd", []) == (1, "", "something is wrong\n")
