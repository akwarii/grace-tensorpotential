"""The messages channel of the library: bare text on stdout and stderr when a command shows it, nothing otherwise."""

from __future__ import annotations

import logging

import pytest

from tensorpotential.core.messages import (
    MESSAGES_LOGGER,
    get_messages_logger,
    show_messages,
)
from tests.fresh_python import run_fresh_python

pytestmark = pytest.mark.usefixtures("restore_messages")


def test_the_logger_has_the_documented_name():
    assert get_messages_logger().name == MESSAGES_LOGGER == "tensorpotential.messages"


def test_info_goes_to_stdout_and_warning_to_stderr_as_bare_text(capsys):
    show_messages()
    log = get_messages_logger()
    log.info("progress %d", 3)
    log.warning("careful")
    captured = capsys.readouterr()
    assert captured.out == "progress 3\n"
    assert captured.err == "careful\n"


def test_debug_is_not_shown(capsys):
    show_messages()
    get_messages_logger().debug("hidden")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_a_second_call_does_not_print_twice(capsys):
    show_messages()
    show_messages()
    get_messages_logger().info("once")
    assert capsys.readouterr().out == "once\n"


def test_the_messages_do_not_reach_the_root_logger(caplog):
    show_messages()
    with caplog.at_level(logging.INFO):
        get_messages_logger().info("not on the root")
    assert caplog.records == []


def test_the_root_logger_is_left_alone():
    root_level, root_handlers = (
        logging.getLogger().level,
        list(logging.getLogger().handlers),
    )
    show_messages()
    assert logging.getLogger().level == root_level
    assert logging.getLogger().handlers == root_handlers


def test_a_redirection_of_stdout_is_followed(capsys):
    show_messages()
    capsys.readouterr()
    get_messages_logger().info("after the redirection")
    assert capsys.readouterr().out == "after the redirection\n"


def test_without_show_messages_the_messages_reach_the_root_logger(caplog, capsys):
    with caplog.at_level(logging.INFO, logger=MESSAGES_LOGGER):
        get_messages_logger().info("to the root")
    assert [r.getMessage() for r in caplog.records] == ["to the root"]
    assert capsys.readouterr().out == ""


def test_without_show_messages_info_is_silent_and_warning_reaches_stderr(tmp_path):
    code = (
        "from tensorpotential.core.messages import get_messages_logger\n"
        "log = get_messages_logger()\nlog.info('quiet')\nlog.warning('loud')\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "quiet" not in result.stdout + result.stderr
    assert result.stderr.splitlines()[-1] == "loud"


def test_the_module_imports_without_tensorflow(tmp_path):
    code = "import sys\nimport tensorpotential.core.messages\nprint('tensorflow' in sys.modules)\n"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "False"


def test_configure_logging_of_the_scripts_shows_the_messages_as_bare_text(tmp_path):
    code = (
        "import logging\n"
        "from tensorpotential.scripts._script_logging import configure_logging\n"
        "from tensorpotential.core.messages import get_messages_logger\n"
        "configure_logging()\n"
        "get_messages_logger().info('bare')\n"
        "logging.getLogger('other').info('prefixed')\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "bare"
    assert "bare" not in result.stderr
    assert result.stderr.rstrip().endswith("I - prefixed")


def test_info_is_shown_whatever_the_level_of_the_root_logger(tmp_path):
    shown = run_fresh_python(
        "import logging; from tensorpotential.core import messages as m; m.show_messages(); m.get_messages_logger().info('shown'); print(logging.getLogger().level)",
        tmp_path,
    )
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout.splitlines()[-2:] == ["shown", str(logging.WARNING)]
