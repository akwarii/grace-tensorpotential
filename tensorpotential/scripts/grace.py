"""The ``grace`` command: one entry point for the console commands of the package.

``grace <subcommand> [options]`` runs the same function as the console script of that name, with the remaining
arguments: ``grace train`` is ``gracemaker``, ``grace utils`` is ``grace_utils``, ``grace uq build`` is
``grace_uq build``, and so on. The old command names keep working and print the same output; only the program name
in usage lines differs (``grace uq build`` when started through ``grace``).

The table of commands (:data:`COMMANDS`, and :data:`UQ_COMMANDS` for the sub-commands of ``grace uq``) is the one place
that lists them. This module imports nothing heavy: ``grace``, ``grace -h``, an unknown sub-command and a command line
that only asks for the help never import TensorFlow. A command is imported when it is run, and the backend it needs
(:attr:`Command.backend`) is checked first, so that a missing TensorFlow gives the ``pip install`` hint before any import.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tensorpotential.core.backends import BackendNotInstalledError, require_backend

_SCRIPTS = "tensorpotential.scripts"
_UQ = "tensorpotential.uq.cli"


@dataclass(frozen=True)
class Command:
    """One row of the table: the name typed after ``grace`` and the function that runs it.

    Attributes
    ----------
    name
        The sub-command word, e.g. ``"train"``.
    module, function
        Where the entry function lives. It is called as ``function(argv, prog="grace <name>")`` and returns the
        exit code.
    help
        One line for the top-level help.
    backend
        The backend (``"tf"``) that has to be installed to run the command, or ``None``. It is checked with
        :func:`tensorpotential.core.backends.require_backend` before the module is imported.
    """

    name: str
    module: str
    function: str
    help: str
    backend: str | None = None


COMMANDS: tuple[Command, ...] = (
    Command(
        "train",
        f"{_SCRIPTS}.gracemaker",
        "main",
        "Fit a potential from an input.yaml (gracemaker)",
        "tf",
    ),
    Command(
        "preprocess",
        f"{_SCRIPTS}.grace_preprocess",
        "main",
        "Precompute a dataset for fitting (grace_preprocess)",
        "tf",
    ),
    Command(
        "predict",
        f"{_SCRIPTS}.grace_predict",
        "main",
        "Predict energies, forces and stresses of a dataset (grace_predict)",
        "tf",
    ),
    Command(
        "utils",
        f"{_SCRIPTS}.grace_utils",
        "main",
        "Convert, reduce, summarize and export models (grace_utils)",
        "tf",
    ),
    Command(
        "models",
        f"{_SCRIPTS}.grace_models",
        "main",
        "List and download foundation models (grace_models)",
    ),
    Command(
        "uq",
        f"{_SCRIPTS}.grace_uq",
        "main",
        "Uncertainty quantification: build, info, predict, select (grace_uq)",
    ),
    Command(
        "collect",
        f"{_SCRIPTS}.grace_collect",
        "main",
        "Collect VASP calculations into a dataframe (grace_collect)",
    ),
    Command(
        "extxyz2df",
        f"{_SCRIPTS}.extxyz2df",
        "main",
        "Convert an extxyz file into a dataframe (extxyz2df)",
        "tf",
    ),
    Command(
        "df2extxyz",
        f"{_SCRIPTS}.df2extxyz",
        "main",
        "Convert a dataframe into an extxyz file (df2extxyz)",
    ),
    Command(
        "dashboard",
        f"{_SCRIPTS}.grace_dashboard",
        "main",
        "Browse the results of fits in a web dashboard (grace_dashboard)",
    ),
)

UQ_COMMANDS: tuple[Command, ...] = (
    Command(
        "build",
        f"{_UQ}.build",
        "build_main",
        "Parallel GMM-UQ artifact generation",
        "tf",
    ),
    Command("info", f"{_UQ}.info", "info_main", "Summarize a UQ artifact", "tf"),
    Command(
        "predict",
        f"{_UQ}.predict",
        "predict_main",
        "Run E/F/S and per-atom gamma over a dataset",
        "tf",
    ),
    Command(
        "select",
        f"{_UQ}.select",
        "select_main",
        "Pick structures from a candidate pool",
        "tf",
    ),
)


def find_command(commands: Sequence[Command], name: str) -> Command | None:
    """Return the row called ``name``, or ``None``."""
    return next((command for command in commands if command.name == name), None)


def resolve(command: Command) -> Callable[..., object]:
    """Import the module of ``command`` and return its entry function."""
    return getattr(importlib.import_module(command.module), command.function)


def run_command(command: Command, argv: Sequence[str], *, prog: str) -> int:
    """Check the backend of ``command``, import it and run it with ``argv``; return its exit code.

    Parameters
    ----------
    command
        The row to run.
    argv
        The arguments after the sub-command word.
    prog
        The program name for usage lines, e.g. ``"grace uq build"``.

    A backend that is not installed is reported on the standard error (the message names the extra to install) and
    gives exit code 1, before the command is imported.
    """
    if command.backend is not None:
        try:
            require_backend(command.backend, needed_by=prog)
        except BackendNotInstalledError as error:
            sys.stderr.write(f"{error}\n")
            return 1
    result = resolve(command)(list(argv), prog=prog)
    return result if isinstance(result, int) else 0


def dispatch(
    commands: Sequence[Command],
    argv: Sequence[str],
    *,
    prog: str,
    print_help: Callable[[], None],
) -> int:
    """Run the sub-command named by the first word of ``argv``.

    No word, ``-h`` and ``--help`` print the help and return 0; an unknown word is reported on the standard error
    and returns 2.
    """
    if not argv or argv[0] in {"-h", "--help"}:
        print_help()
        return 0
    command = find_command(commands, argv[0])
    if command is None:
        known = ", ".join(sorted(c.name for c in commands))
        sys.stderr.write(
            f"{prog}: unknown subcommand {argv[0]!r}. Use one of: {known}\n"
        )
        return 2
    return run_command(command, argv[1:], prog=f"{prog} {command.name}")


def render_help(commands: Sequence[Command], prog: str = "grace") -> str:
    """Return the top-level help text: usage, one line per sub-command, and how to get theirs."""
    width = max(len(command.name) for command in commands)
    rows = "\n".join(
        f"  {command.name.ljust(width)}  {command.help}" for command in commands
    )
    return (
        f"usage: {prog} <subcommand> [options]\n\n"
        f"Subcommands:\n{rows}\n\n"
        f"Run `{prog} <subcommand> --help` for the options of a subcommand.\n"
    )


def main(argv: Sequence[str] | None = None, *, prog: str = "grace") -> int:
    """Entry point of the ``grace`` console script; returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    return dispatch(
        COMMANDS,
        args,
        prog=prog,
        print_help=lambda: sys.stdout.write(render_help(COMMANDS, prog)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
