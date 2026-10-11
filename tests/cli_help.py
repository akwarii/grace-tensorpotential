"""The ``--help`` text of every console command, captured in process, against committed references.

CLI1 puts a ``grace`` command next to the ten console scripts and tidies their entry code. The contract is that no option,
default, choice or help string moves, so the text each parser prints is committed in ``tests/cli_help/`` (one file per
command or sub-command, written from the tree before the change) and every later change is compared with it.

The text is captured the way a user sees it: ``sys.argv[0]`` is the console-script name (parsers without a ``prog`` read
it), ``COLUMNS`` is fixed, and ``main`` is called in process, so the coverage run sees the parser code.

Python 3.13 prints an option as ``-o, --output OUTPUT`` where older versions print ``-o OUTPUT, --output OUTPUT``, and the
width of the option column follows from it, so the comparison goes through :func:`normalise`, which folds both forms and
the line wrapping. A word that changes still fails.

Regenerate the references (only from an unmodified tree, never to make a test pass)::

    python -m tests.cli_help write
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import os
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REFERENCE_DIR = Path(__file__).parent / "cli_help"
COLUMNS = "100"


@dataclass(frozen=True)
class HelpCase:
    """One parser: the console name it is started under, the sub-command words, the function that runs it."""

    command: str
    words: tuple[str, ...]
    module: str
    function: str

    @property
    def case_id(self) -> str:
        """The file stem of the reference: ``grace_utils.update_model`` or ``extxyz2df``."""
        return ".".join((self.command, *self.words))

    @property
    def reference(self) -> Path:
        """Path of the committed reference text."""
        return REFERENCE_DIR / f"{self.case_id}.txt"

    def main(self) -> Callable[..., object]:
        """The function the console script (or the dispatcher) runs."""
        return getattr(importlib.import_module(self.module), self.function)


def _cases(command: str, module: str, function: str, *subs: str) -> list[HelpCase]:
    return [HelpCase(command, (), module, function)] + [
        HelpCase(command, (sub,), module, function) for sub in subs
    ]


_SCRIPTS = "tensorpotential.scripts"
CASES: tuple[HelpCase, ...] = (
    *_cases(
        "grace_utils",
        f"{_SCRIPTS}.grace_utils",
        "main",
        "update_model",
        "resave_checkpoint",
        "reduce_elements",
        "cast_model_param",
        "export",
        "export_kokkos",
        "summary",
        "aux_model",
    ),
    *_cases(
        "grace_models",
        f"{_SCRIPTS}.grace_models",
        "main",
        "list",
        "info",
        "download",
        "checkpoint",
        "update",
    ),
    *_cases(
        "grace_uq", f"{_SCRIPTS}.grace_uq", "main", "build", "info", "predict", "select"
    ),
    *_cases("gracemaker", f"{_SCRIPTS}.gracemaker", "main"),
    *_cases("grace_preprocess", f"{_SCRIPTS}.grace_preprocess", "main"),
    *_cases("grace_predict", f"{_SCRIPTS}.grace_predict", "main"),
    *_cases("grace_collect", f"{_SCRIPTS}.grace_collect", "main"),
    *_cases("extxyz2df", f"{_SCRIPTS}.extxyz2df", "main"),
    *_cases("df2extxyz", f"{_SCRIPTS}.df2extxyz", "main"),
    *_cases("grace_dashboard", f"{_SCRIPTS}.grace_dashboard", "main"),
)


@contextlib.contextmanager
def _environment(command: str, argv: Sequence[str]) -> Iterator[None]:
    saved_argv, saved_columns = sys.argv, os.environ.get("COLUMNS")
    sys.argv = [command, *argv]
    os.environ["COLUMNS"] = COLUMNS
    try:
        yield
    finally:
        sys.argv = saved_argv
        if saved_columns is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = saved_columns


def run_main(
    main: Callable[..., object], command: str, argv: Sequence[str]
) -> tuple[int, str, str]:
    """Run ``main`` as the console script ``command`` would, in process.

    A ``main`` that reads ``sys.argv`` (the scripts before CLI1 task 2) and one that takes ``argv`` both see ``argv``.
    Returns the exit code (``SystemExit`` included: ``None`` is 0, a message is written to the error output and is 1), the standard output and the standard error.
    """
    out, err = io.StringIO(), io.StringIO()
    takes_argv = bool(inspect.signature(main).parameters)
    with (
        _environment(command, argv),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            result = main(list(argv)) if takes_argv else main()
        except SystemExit as stop:
            result = stop.code
            if isinstance(
                result, str
            ):  # SystemExit("message"): Python prints it to stderr and exits with 1
                err.write(f"{result}\n")
                result = 1
            elif result is None:
                result = 0
    return (result if isinstance(result, int) else 0), out.getvalue(), err.getvalue()


def capture_help(case: HelpCase) -> str:
    """The text ``<command> <words> --help`` prints, with the exit code checked to be 0."""
    code, out, err = run_main(case.main(), case.command, [*case.words, "--help"])
    if code != 0:
        raise RuntimeError(f"{case.case_id} --help exited with {code}: {err}")
    return out


_OLD_OPTION_PAIR = re.compile(r"(-\w) ((?:(?!, -).){1,60}?), (--[\w-]+) \2(?=\s|$)")


def normalise(text: str) -> str:
    """Fold the Python 3.13 option form and the line wrapping, keep every word.

    ``-o, --output OUTPUT`` and ``-o OUTPUT, --output OUTPUT`` become the first form, and runs of white space one blank.
    """
    return _OLD_OPTION_PAIR.sub(r"\1, \3 \2", " ".join(text.split()))


def write_references() -> list[Path]:
    """Write one reference file per case from the code on ``PYTHONPATH`` (an unmodified tree only)."""
    REFERENCE_DIR.mkdir(exist_ok=True)
    written = []
    for case in CASES:
        case.reference.write_text(capture_help(case), encoding="utf-8")
        written.append(case.reference)
    return written


if __name__ == "__main__":
    if sys.argv[1:] != ["write"]:
        raise SystemExit(__doc__)
    for path in write_references():
        sys.stdout.write(f"{path}\n")
