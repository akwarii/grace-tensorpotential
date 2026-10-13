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


class _MessageHandler(logging.Handler):
    """Write the bare message to ``sys.stderr`` from ``WARNING`` up and to ``sys.stdout`` below.

    The stream is looked up at each record, so that a redirection of the standard streams is followed.
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
        stream.write(self.format(record) + "\n")
        stream.flush()


def get_messages_logger() -> logging.Logger:
    """Return the logger of the messages shown to the person at the terminal."""
    return logging.getLogger(MESSAGES_LOGGER)


def show_messages() -> logging.Logger:
    """Show the messages of the library as bare text: ``INFO`` on the standard output, ``WARNING`` and above on the standard error.

    The handler is installed once; a second call changes nothing. The messages do not reach the handlers of the root
    logger, so a program that configures logging with a timestamp does not print them twice, and nothing else of the
    root logger is touched.

    Returns
    -------
    logging.Logger
        The logger of the messages.
    """
    logger = get_messages_logger()
    if not any(isinstance(handler, _MessageHandler) for handler in logger.handlers):
        handler = _MessageHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
