"""The prompt and message helpers of the interactive input wizards, on top of the shared console.

``gracemaker -t`` (``cli/wizard.py``) and the generic-tensor wizard (``extra/gen_tensor/wizard.py``) ask their questions
and print their messages through these helpers, and neither imports the other: this module is what they share. A question
is asked with ``questionary`` when it is installed (a terminal is needed), with ``input()`` otherwise; a message is printed
through the shared rich console (:mod:`tensorpotential.cli.console`) when rich is installed, as plain text otherwise.

``_HAS_QUESTIONARY`` and ``_HAS_RICH`` say which of the two worlds is active; tests switch them. The names and behaviour
are those of the helpers when they lived in ``cli/wizard.py``, which still re-exports them.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

# questionary and rich are dependencies of the package (pyproject.toml); the two flags say which world is active, and tests
# switch them.
import questionary
from questionary import Style as QStyle
from rich import box as rich_box
from rich.panel import Panel
from rich.text import Text as RichText

from tensorpotential.cli.console import get_console

_QUESTIONARY_STYLE = QStyle([
    ("qmark", "fg:#00d7ff bold"),
    ("question", "bold"),
    ("answer", "fg:#00d7ff bold"),
    ("pointer", "fg:#00d7ff bold"),
    ("highlighted", "fg:#00d7ff bold"),
    ("selected", "fg:#00d7ff"),
    ("separator", "fg:#6c6c6c"),
    ("instruction", "fg:#6c6c6c"),
    ("text", ""),
    ("disabled", "fg:#858585 italic"),
])
_HAS_QUESTIONARY = True
_rich_console = get_console()
_HAS_RICH = True

try:
    import readline  # noqa: F401

    no_readline = False
except ImportError:
    no_readline = True


class DatasetState(Protocol):
    """What the dataset question reads and writes: the state of either wizard."""

    train_filename: str
    use_separate_test: bool
    test_filename: str
    test_size: float


def _write(text: str = "") -> None:
    """Write a line of plain text to the standard output."""
    sys.stdout.write(text + "\n")


# ---------------------------------------------------------------------------
# Plain input fallbacks
# ---------------------------------------------------------------------------


def input_choice(query, choices, default_choice):
    """Input from stdin with list of available choices and default choice."""
    choices = list(choices)
    if default_choice not in choices:
        raise AssertionError(default_choice)
    while True:
        choice = (
            input(
                query
                + f", available options: {', '.join(choices)} (default = {default_choice}): "
            )
            or default_choice
        )
        if choice in choices:
            return choice


def input_with_default(query, default_choice: Any = None):
    """Input from stdin with default value."""
    return input(query + ": ") or default_choice


def input_no_default(query):
    """Input from stdin without default."""
    while True:
        choice = input(query)
        if choice:
            return choice


# ---------------------------------------------------------------------------
# Rich / Questionary helpers
# ---------------------------------------------------------------------------


def _print_header():
    if _HAS_RICH:
        _rich_console.print()
        _rich_console.print(
            Panel(
                RichText.assemble(
                    ("GRACEmaker ", "bold cyan"),
                    ("input.yaml", "bold white"),
                    (" wizard", "bold cyan"),
                ),
                subtitle="[dim]navigate with arrow keys · confirm with Enter[/dim]",
                border_style="cyan",
                box=rich_box.ROUNDED,
                expand=False,
            )
        )
        _rich_console.print()
    else:
        _write("=" * 50)
        _write("  GRACEmaker input.yaml wizard")
        _write("=" * 50)


def _info(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [dim]→[/dim] {msg}")
    else:
        _write(f"  -> {msg}")


def _success(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [bold green]✓[/bold green] {msg}")
    else:
        _write(f"  OK: {msg}")


def _warning(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [bold orange1]![/bold orange1] {msg}")
    else:
        _write(f"  WARNING: {msg}")


def _error(msg: str):
    if _HAS_RICH:
        _rich_console.print(f"  [bold red]✗[/bold red] {msg}")
    else:
        _write(f"  ERROR: {msg}")


def _section(title: str):
    if _HAS_RICH:
        _rich_console.print(f"\n[bold cyan]── {title}[/bold cyan]")
    else:
        _write(f"\n-- {title}")


def _tip_path_input():
    """Print a one-line hint explaining how to enter file paths."""
    if _HAS_QUESTIONARY:
        tip = "Tab ↹ autocompletes path  ·  ↑↓ navigates history"
    elif not no_readline:
        tip = "↑↓ for command history  ·  Tab may autocomplete (readline)"
    else:
        tip = "Type or paste the full file path"
    if _HAS_RICH:
        _rich_console.print(f"  [dim italic]{tip}[/dim italic]")
    else:
        _write(f"  ({tip})")


def _ask_select(message, choices, default=None):
    if _HAS_QUESTIONARY:
        choice_list = list(choices)
        # choices may be questionary.Choice objects; find matching default
        def_val = default
        if def_val not in choice_list:
            # try matching by .value for Choice objects
            def_val = next(
                (c for c in choice_list if getattr(c, "value", c) == default),
                choice_list[0],
            )
        result = questionary.select(
            message, choices=choice_list, default=def_val, style=_QUESTIONARY_STYLE
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    # extract string values from Choice objects if needed
    str_choices = [getattr(c, "value", c) for c in choices]
    return input_choice(
        message, str_choices, default if default in str_choices else str_choices[0]
    )


def _ask_text(message, default=None):
    if _HAS_QUESTIONARY:
        result = questionary.text(
            message,
            default=str(default) if default is not None else "",
            style=_QUESTIONARY_STYLE,
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result if result else default
    return input_with_default(message, default)


def _ask_path(message):
    if _HAS_QUESTIONARY:
        result = questionary.path(message, style=_QUESTIONARY_STYLE).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    return input_no_default(message + " ")


def _ask_confirm(message, default=True):
    if _HAS_QUESTIONARY:
        result = questionary.confirm(
            message, default=default, style=_QUESTIONARY_STYLE
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    ans = input_choice(message, ["yes", "no"], "yes" if default else "no")
    return ans == "yes"


# ---------------------------------------------------------------------------
# Dataset question (shared by both wizards)
# ---------------------------------------------------------------------------


def _section_dataset(s: DatasetState) -> DatasetState:
    """Ask for the training and test data of the fit."""
    _section("Dataset")
    _tip_path_input()
    s.train_filename = _ask_path("Training dataset file (e.g. data.pkl.gz):")
    if not s.train_filename:
        _info("No filename entered.")
        sys.exit(1)
    _success(f"Train file: {s.train_filename}")

    s.use_separate_test = _ask_confirm(
        "Use a separate test dataset file?", default=False
    )
    if s.use_separate_test:
        s.test_filename = _ask_path("Test dataset file:")
        _success(f"Test file: {s.test_filename}")
    else:
        s.test_size = float(
            _ask_text("Test set fraction (split from train)", default=0.05)
        )
    _success(f"Test fraction: {s.test_size}")
    return s
