"""``tensorpotential.cli.console``: the shared rich console and the helpers that print through it.

Logic layer: a message printed when the output is not a terminal is the text it is, character for character (markup,
brackets, emoji codes, tabs of long lines are not interpreted or wrapped); on a terminal the same text is styled; ``NO_COLOR``
turns the colour off; error lines go to the standard error; the table of commands reads the same on a pipe as its plain
text. There are no physical values: the module only prints.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

from tensorpotential.cli import console

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CODES = re.compile(r"\x1b\[([0-9;]*)m")


def has_colour(text: str) -> bool:
    """Whether ``text`` holds an ANSI code that sets a foreground or background colour (bold and reset do not)."""
    for params in CODES.findall(text):
        for code in params.split(";"):
            if code and (
                30 <= int(code) <= 38 or 40 <= int(code) <= 48 or 90 <= int(code) <= 107
            ):
                return True
    return False


AWKWARD = [
    "plain text",
    "[bold]not markup[/bold] and [0, 2] and [red]",
    "emoji code :smile: stays",
    "numbers 1.5 and 42 and True are not highlighted",
    "  indented, with  double  spaces  ",
    "x" * 300,
    "",
]


@pytest.fixture(autouse=True)
def fresh_consoles():
    console.reset_consoles()
    yield
    console.reset_consoles()


def terminal(**kwargs) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    kwargs.setdefault("width", 100)
    return console.make_console(file=buffer, force_terminal=True, **kwargs), buffer


@pytest.fixture
def on_terminal(monkeypatch):
    """Replace the shared consoles by forced-terminal consoles writing into buffers (a terminal is not available)."""
    out_console, out = terminal()
    err_console, err = terminal()
    monkeypatch.setattr(
        console,
        "get_console",
        lambda *, stderr=False: err_console if stderr else out_console,
    )
    return out, err


@pytest.mark.parametrize("message", AWKWARD)
def test_a_message_is_printed_as_the_text_it_is_when_the_output_is_not_a_terminal(
    message, capsys
):
    console.say(message)
    assert capsys.readouterr().out == message + "\n"


def test_say_without_a_newline_and_to_the_standard_error(capsys):
    console.say("a", end="")
    console.say("b", stderr=True, end="!")
    out, err = capsys.readouterr()
    assert (out, err) == ("a", "b!")


def test_error_lines_go_to_the_standard_error_status_and_success_to_the_output(capsys):
    console.print_error("grace: unknown subcommand 'x'. Use one of: a, b")
    console.print_status("working [1/3]")
    console.print_success("done")
    out, err = capsys.readouterr()
    assert err == "grace: unknown subcommand 'x'. Use one of: a, b\n"
    assert out == "working [1/3]\ndone\n"


def test_segments_read_as_their_concatenation_without_styles(capsys):
    console.print_segments([
        ("grace", "bold"),
        (" <subcommand> [options]\n", None),
        ("build", "cyan"),
    ])
    assert capsys.readouterr().out == "grace <subcommand> [options]\nbuild\n"


def test_a_message_is_styled_on_a_terminal_and_its_text_is_unchanged(on_terminal):
    out, err = on_terminal
    console.say("[not markup] text", style="bold red")
    console.print_error("failed")
    assert "\x1b[1;31m[not markup] text\x1b[0m\n" == out.getvalue()
    assert ANSI.sub("", err.getvalue()) == "failed\n"
    assert has_colour(err.getvalue())


def test_segments_are_styled_on_a_terminal(on_terminal):
    out, _ = on_terminal
    console.print_segments([("a", "bold"), (" b ", None), ("c", "green")])
    assert ANSI.sub("", out.getvalue()) == "a b c\n"
    assert "\x1b[1ma" in out.getvalue()
    assert has_colour(out.getvalue())


def test_no_color_turns_the_colour_off_but_not_the_text(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    no_colour, buffer = terminal()
    no_colour.print(console.Text("hello", style="bold red"))
    assert ANSI.sub("", buffer.getvalue()) == "hello\n"
    assert not has_colour(buffer.getvalue())
    monkeypatch.delenv("NO_COLOR")
    coloured, buffer = terminal()
    coloured.print(console.Text("hello", style="bold red"))
    assert has_colour(buffer.getvalue())


def test_the_shared_consoles_read_no_color_when_they_are_made(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    console.reset_consoles()
    assert console.get_console().no_color is True
    monkeypatch.delenv("NO_COLOR")
    assert console.get_console().no_color is True  # still the console made before
    console.reset_consoles()
    assert console.get_console().no_color is False


def test_the_shared_console_is_one_object_per_stream_until_it_is_reset():
    assert console.get_console() is console.get_console()
    assert console.get_console(stderr=True) is console.get_console(stderr=True)
    assert console.get_console() is not console.get_console(stderr=True)
    first = console.get_console()
    console.reset_consoles()
    assert console.get_console() is not first


def test_the_consoles_follow_a_redirected_stream(capsys):
    console.say("one")
    console.print_error("two")
    out, err = capsys.readouterr()
    assert (out, err) == ("one\n", "two\n")


ROWS = [("train", "Fit a potential"), ("uq", "Uncertainty"), ("extxyz2df", "Convert")]


def test_the_plain_text_of_the_command_table_is_aligned_and_complete():
    text = console.command_table_text(
        "usage: grace <subcommand>",
        ROWS,
        "Subcommands",
        "Run `grace <subcommand> --help`.",
    )
    assert text == (
        "usage: grace <subcommand>\n\n"
        "Subcommands:\n"
        "  train      Fit a potential\n"
        "  uq         Uncertainty\n"
        "  extxyz2df  Convert\n\n"
        "Run `grace <subcommand> --help`.\n"
    )


def test_the_command_table_is_the_plain_text_when_the_output_is_not_a_terminal(capsys):
    console.print_command_table("usage: grace", ROWS, "Subcommands", "footer")
    assert capsys.readouterr().out == console.command_table_text(
        "usage: grace", ROWS, "Subcommands", "footer"
    )


def test_the_command_table_is_a_styled_table_on_a_terminal(on_terminal):
    out, _ = on_terminal
    console.print_command_table("usage: grace", ROWS, "Subcommands", "footer")
    plain = ANSI.sub("", out.getvalue())
    for name, text in ROWS:
        assert re.search(rf"^{name}\s+{text}\s*$", plain, re.MULTILINE), name
    assert plain.splitlines()[0] == "usage: grace"
    assert plain.splitlines()[-1] == "footer"
    assert has_colour(out.getvalue())


def test_the_command_table_of_a_narrow_terminal_still_holds_every_word(monkeypatch):
    narrow, buffer = terminal(width=30)
    monkeypatch.setattr(console, "get_console", lambda *, stderr=False: narrow)
    console.print_command_table(
        "usage: grace",
        [("a", "one two three four five six seven eight")],
        "Subcommands",
        "footer",
    )
    words = ANSI.sub("", buffer.getvalue()).split()
    for word in "one two three four five six seven eight".split():
        assert word in words
