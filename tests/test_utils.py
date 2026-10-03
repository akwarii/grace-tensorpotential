"""Tests of ``utils.py``: ``get_param_dtype_from_config``, the ``param_dtype`` default of a training config.

This is one of the code defaults of ``param_dtype`` that disagree (see ``test_metadata_utils.py``): a config
without ``param_dtype`` gives float32, a yaml without metadata gives float64. Pinned, not endorsed.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

import pytest
import tensorflow as tf

from tensorpotential.utils import get_param_dtype_from_config


class RecordingLog:
    """The two methods of a logger that ``get_param_dtype_from_config`` calls, remembering the arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def info(self, *args, **kwargs):
        self.calls.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args, kwargs))


@pytest.mark.parametrize(
    ("name", "dtype"), [("float32", tf.float32), ("float64", tf.float64)]
)
def test_param_dtype_key_is_read(name, dtype):
    assert get_param_dtype_from_config({"param_dtype": name}) == (dtype, name)


def test_default_without_a_key_is_float32():
    # PINNED, not endorsed: float32 here; metadata_utils.resolve_param_dtype gives float64 for a yaml without the key
    assert get_param_dtype_from_config({}) == (tf.float32, "float32")


def test_unknown_name_is_a_key_error():
    with pytest.raises(KeyError, match="Unknown dtype name float16"):
        get_param_dtype_from_config({"param_dtype": "float16"})


def test_deprecated_float_dtype_is_used_when_param_dtype_is_absent():
    assert get_param_dtype_from_config({"float_dtype": "float64"}) == (
        tf.float64,
        "float64",
    )


def test_param_dtype_wins_over_the_deprecated_float_dtype():
    config = {"param_dtype": "float32", "float_dtype": "float64"}

    assert get_param_dtype_from_config(config) == (tf.float32, "float32")


def test_log_reports_the_chosen_dtype():
    log = RecordingLog()

    get_param_dtype_from_config({"param_dtype": "float64"}, log=log)

    assert log.calls == [("info", ("Model parameter dtype: float64",), {})]


def test_log_reports_the_default():
    log = RecordingLog()

    get_param_dtype_from_config({}, log=log)

    assert log.calls == [
        ("info", ("Using default model parameter dtype: float32",), {})
    ]


def test_deprecated_key_warns_with_the_stacklevel_and_class_as_extra_arguments():
    log = RecordingLog()

    get_param_dtype_from_config({"float_dtype": "float64"}, log=log)

    kind, args, kwargs = log.calls[0]
    assert kind == "warning"
    assert "'float_dtype' is deprecated" in args[0]
    assert args[1] is DeprecationWarning
    assert kwargs == {"stacklevel": 2}
    assert log.calls[1] == ("info", ("Model parameter dtype: float64",), {})


def test_deprecated_key_warning_cannot_be_formatted_by_a_real_logger():
    # PINNED, not endorsed (TEST8 finding): logging.Logger.warning(msg, DeprecationWarning, stacklevel=2) passes the
    # class as a %-argument of a message without placeholders, so formatting the record raises TypeError (logging
    # prints a "Logging error" to stderr instead of failing) and the text of the warning is lost.
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("test_utils.param_dtype")
    logger.setLevel(logging.DEBUG)
    logger.propagate = (
        False  # pytest's own handler formats records and re-raises the error
    )
    logger.addHandler(Collect())
    try:
        result = get_param_dtype_from_config({"float_dtype": "float64"}, log=logger)
    finally:
        logger.handlers.clear()

    assert result == (tf.float64, "float64")
    warning = next(r for r in records if r.levelno == logging.WARNING)
    with pytest.raises(TypeError, match="not all arguments converted"):
        warning.getMessage()
