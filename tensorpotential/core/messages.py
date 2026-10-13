"""The messages the library shows to the person at the terminal, as a logging channel.

Library code reports progress and warnings through the logger named :data:`MESSAGES_LOGGER` instead of ``print``.
A program decides what happens to them: nothing is shown for ``INFO`` until it configures logging; the command-line tools
call :func:`show_messages`, which writes ``INFO`` to the standard output and ``WARNING`` and above to the standard
error as the bare message text, so that their output is the text the ``print`` calls produced before.

The module imports no TensorFlow and is on the import contract of the TensorFlow-free modules.
"""

from __future__ import annotations

import logging
import sys

MESSAGES_LOGGER = "tensorpotential.messages"
_HANDLER_ATTRIBUTE = "_tensorpotential_message_handler"


class _BelowWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class _CurrentStream(logging.Handler):
    """Write the message to ``sys.stdout`` or ``sys.stderr``, looked up at each record so that a redirection is followed."""

    def __init__(self, stream_name: str) -> None:
        super().__init__()
        self._stream_name = stream_name

    def emit(self, record: logging.LogRecord) -> None:
        stream = getattr(sys, self._stream_name)
        stream.write(self.format(record) + "\n")
        stream.flush()


def get_messages_logger() -> logging.Logger:
    """Return the logger of the messages shown to the person at the terminal."""
    return logging.getLogger(MESSAGES_LOGGER)


def show_messages() -> logging.Logger:
    """Show the messages of the library as bare text: ``INFO`` on the standard output, ``WARNING`` and above on the standard error.

    The handlers are installed once; a second call changes nothing. The messages do not reach the handlers of the root
    logger, so a program that configures logging with a timestamp does not print them twice, and nothing else of the
    root logger is touched.

    Returns
    -------
    logging.Logger
        The logger of the messages.
    """
    logger = get_messages_logger()
    if getattr(logger, _HANDLER_ATTRIBUTE, False):
        return logger
    out = _CurrentStream("stdout")
    out.addFilter(_BelowWarning())
    err = _CurrentStream("stderr")
    err.setLevel(logging.WARNING)
    for handler in (out, err):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, _HANDLER_ATTRIBUTE, True)
    return logger


def hide_messages() -> None:
    """Undo :func:`show_messages`: remove its handlers and let the messages reach the root logger again."""
    logger = get_messages_logger()
    for handler in [h for h in logger.handlers if isinstance(h, _CurrentStream)]:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    if hasattr(logger, _HANDLER_ATTRIBUTE):
        delattr(logger, _HANDLER_ATTRIBUTE)
