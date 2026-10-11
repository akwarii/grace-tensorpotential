"""The shared console of the command-line tools.

One :class:`rich.console.Console` for the standard output and one for the standard error, and the few helpers that print
through them: a line of text, an error line, a status line, styled segments, and a table of commands.

The rule of this module is that **rich only styles**. A message is written as the text it is (no markup is read, nothing is
highlighted, wrapped or replaced), so that when the output is not a terminal (a pipe, a file, a SLURM log) every message is
the same text, character for character, as the ``print`` it replaces. On a terminal the same text is styled; ``NO_COLOR``
(read by rich when the console is made) turns the colour off. Only new text, such as the top-level help of ``grace``, may
use a table.

The module imports no TensorFlow and is on the import contract of the TensorFlow-free modules.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from typing import TextIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

ERROR_STYLE = "bold red"
STATUS_STYLE = "dim"
SUCCESS_STYLE = "bold green"
HEADING_STYLE = "bold"
COMMAND_STYLE = "bold cyan"


def make_console(
    *,
    stderr: bool = False,
    file: TextIO | None = None,
    force_terminal: bool | None = None,
    width: int | None = None,
) -> Console:
    """Make a console with the settings of the tools.

    Parameters
    ----------
    stderr
        Write to the standard error (looked up at each write, so that a redirected stream is followed).
    file
        Write to this stream instead (for tests).
    force_terminal
        Treat the stream as a terminal (``True``) or not (``False``); ``None`` lets rich decide.
    width
        Width of the terminal, or ``None`` to let rich decide.

    Returns
    -------
    rich.console.Console
        A console that reads ``NO_COLOR`` from the environment when it is made.
    """
    return Console(stderr=stderr, file=file, force_terminal=force_terminal, width=width)


@functools.cache
def get_console(*, stderr: bool = False) -> Console:
    """Return the shared console of the standard output (or of the standard error).

    Parameters
    ----------
    stderr
        The console of the standard error instead of the standard output.

    Returns
    -------
    rich.console.Console
        The same object at each call, until :func:`reset_consoles` is called.
    """
    return make_console(stderr=stderr)


def reset_consoles() -> None:
    """Forget the shared consoles, so that the next call makes new ones (reads ``NO_COLOR`` again)."""
    get_console.cache_clear()


def say(
    message: str, *, style: str | None = None, stderr: bool = False, end: str = "\n"
) -> None:
    """Write ``message`` as it is, styled on a terminal.

    Parameters
    ----------
    message
        The text. It is not read as markup, not highlighted and not wrapped.
    style
        A rich style (``"bold red"``), applied on a terminal only.
    stderr
        Write to the standard error.
    end
        What follows the text.
    """
    get_console(stderr=stderr).print(
        Text(message, style=style or ""), soft_wrap=True, end=end
    )


def print_error(message: str) -> None:
    """Write an error line to the standard error, in red on a terminal."""
    say(message, style=ERROR_STYLE, stderr=True)


def print_status(message: str) -> None:
    """Write a status line to the standard output, dimmed on a terminal."""
    say(message, style=STATUS_STYLE)


def print_success(message: str) -> None:
    """Write a success line to the standard output, in green on a terminal."""
    say(message, style=SUCCESS_STYLE)


def print_segments(segments: Iterable[tuple[str, str | None]]) -> None:
    """Write a text made of styled segments, each ``(text, style)``, followed by a newline.

    The unstyled text is the concatenation of the segments, so a text written as segments reads the same on a pipe as
    the string it replaces.
    """
    parts = [(text, style) if style else text for text, style in segments]
    get_console().print(Text.assemble(*parts), soft_wrap=True)


def command_table_text(
    usage: str, rows: Sequence[tuple[str, str]], heading: str, footer: str
) -> str:
    """Return the plain text of a table of commands: usage, a heading, one aligned row per command, a footer.

    Parameters
    ----------
    usage
        The usage line.
    rows
        ``(name, one-line help)`` per command.
    heading
        The title above the rows, without the colon.
    footer
        The last line.
    """
    width = max(len(name) for name, _ in rows)
    body = "\n".join(f"  {name.ljust(width)}  {text}" for name, text in rows)
    return f"{usage}\n\n{heading}:\n{body}\n\n{footer}\n"


def print_command_table(
    usage: str, rows: Sequence[tuple[str, str]], heading: str, footer: str
) -> None:
    """Write a table of commands: a rich table on a terminal, :func:`command_table_text` otherwise.

    Parameters
    ----------
    usage, rows, heading, footer
        As for :func:`command_table_text`.
    """
    console = get_console()
    if not console.is_terminal:
        console.file.write(command_table_text(usage, rows, heading, footer))
        return
    table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    table.add_column(style=COMMAND_STYLE, no_wrap=True)
    table.add_column()
    for name, text in rows:
        table.add_row(name, text)
    console.print(Text(usage, style=HEADING_STYLE))
    console.print()
    console.print(Text(f"{heading}:", style=HEADING_STYLE))
    console.print(table)
    console.print()
    console.print(Text(footer), soft_wrap=True)
