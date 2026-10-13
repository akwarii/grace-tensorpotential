"""The logging set-up shared by the console scripts: one format, one place."""

from __future__ import annotations

import logging

from tensorpotential.core.messages import show_messages

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
LOG_DATEFMT = "%Y/%m/%d %H:%M:%S"


def configure_logging() -> logging.Logger:
    """Configure the root logger as the scripts do (INFO, time, one-letter level) and return it.

    The messages of the library (``tensorpotential.messages``) are shown as bare text, as the ``print`` calls they replace.

    Returns
    -------
    logging.Logger
        The root logger.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=LOG_DATEFMT)
    show_messages()
    return logging.getLogger()
