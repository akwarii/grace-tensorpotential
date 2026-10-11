"""``tensorpotential.scripts._script_logging``: the logging set-up shared by the console scripts."""

from __future__ import annotations

import logging
from pathlib import Path

from tensorpotential.scripts import _script_logging


def test_the_root_logger_is_configured_with_the_format_of_the_scripts(monkeypatch):
    # logging.basicConfig does nothing while a handler exists, and pytest keeps its own on the root logger, so the call
    # is recorded instead
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))
    assert _script_logging.configure_logging() is logging.getLogger()
    assert calls == [
        {
            "level": logging.INFO,
            "format": "%(asctime)s %(levelname).1s - %(message)s",
            "datefmt": "%Y/%m/%d %H:%M:%S",
        }
    ]


def test_the_format_prints_time_a_one_letter_level_and_the_message():
    formatter = logging.Formatter(
        _script_logging.LOG_FMT, datefmt=_script_logging.LOG_DATEFMT
    )
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "message", None, None)
    line = formatter.format(record)
    assert line.endswith(" W - message")
    assert (
        line[4] == "/" and line[7] == "/" and line[13] == ":"
    )  # 2026/10/11 12:00:00 W - message


def test_every_script_that_used_to_repeat_the_set_up_uses_the_helper():
    root = Path(_script_logging.__file__).parent
    for name in (
        "grace_predict",
        "grace_utils",
        "grace_collect",
        "df2extxyz",
        "extxyz2df",
    ):
        source = (root / f"{name}.py").read_text()
        assert "logger = configure_logging()" in source, name
        assert "logging.basicConfig" not in source, name
