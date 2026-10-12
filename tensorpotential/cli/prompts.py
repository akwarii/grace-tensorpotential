"""The prompt and message helpers of the interactive input wizards, on top of the shared console.

``gracemaker -t`` (``cli/wizard.py``) and the generic-tensor wizard (``extra/gen_tensor/wizard.py``) ask their questions
and print their messages through these helpers, and neither imports the other: this module is what they share. A question
is asked with ``questionary`` (a terminal is needed) and a message is printed through the shared rich console
(:mod:`tensorpotential.cli.console`); both are dependencies of the package, so there is no plain-text fallback. The names
are those of the helpers when they lived in ``cli/wizard.py``, which still re-exports them.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

# questionary and rich are dependencies of the package (pyproject.toml).
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
_rich_console = get_console()


class DatasetState(Protocol):
    """What the dataset question reads and writes: the state of either wizard."""

    train_filename: str
    use_separate_test: bool
    test_filename: str
    test_size: float


# ---------------------------------------------------------------------------
# Rich / Questionary helpers
# ---------------------------------------------------------------------------


def _print_header():
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


def _info(msg: str):
    _rich_console.print(f"  [dim]→[/dim] {msg}")


def _success(msg: str):
    _rich_console.print(f"  [bold green]✓[/bold green] {msg}")


def _warning(msg: str):
    _rich_console.print(f"  [bold orange1]![/bold orange1] {msg}")


def _error(msg: str):
    _rich_console.print(f"  [bold red]✗[/bold red] {msg}")


def _section(title: str):
    _rich_console.print(f"\n[bold cyan]── {title}[/bold cyan]")


def _tip_path_input():
    """Print a one-line hint explaining how to enter file paths."""
    _rich_console.print(
        "  [dim italic]Tab ↹ autocompletes path  ·  ↑↓ navigates history[/dim italic]"
    )


def _answered(question: Any) -> Any:
    """Ask a questionary question; Ctrl-C (``None``) is a ``KeyboardInterrupt`` for the wizard to handle."""
    result = question.ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def _ask_select(message, choices, default=None):
    choice_list = list(choices)
    # choices may be questionary.Choice objects; find matching default
    def_val = default
    if def_val not in choice_list:
        # try matching by .value for Choice objects
        def_val = next(
            (c for c in choice_list if getattr(c, "value", c) == default),
            choice_list[0],
        )
    return _answered(
        questionary.select(
            message, choices=choice_list, default=def_val, style=_QUESTIONARY_STYLE
        )
    )


def _ask_text(message, default=None):
    result = _answered(
        questionary.text(
            message,
            default=str(default) if default is not None else "",
            style=_QUESTIONARY_STYLE,
        )
    )
    return result if result else default


def _ask_path(message):
    return _answered(questionary.path(message, style=_QUESTIONARY_STYLE))


def _ask_confirm(message, default=True):
    return _answered(
        questionary.confirm(message, default=default, style=_QUESTIONARY_STYLE)
    )


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
