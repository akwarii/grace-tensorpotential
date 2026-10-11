"""Tests for tensorpotential.cli.prompts: the prompt and message helpers of the input wizards.

Both worlds of each helper are exercised by switching the module flags: with or without ``questionary`` (a fake replaces its
interactive prompts, an external boundary that needs a terminal) and with or without ``rich`` (the plain text is checked
character for character, the rich text by its words and glyph).
"""

from __future__ import annotations

import ast
import builtins
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

import tensorpotential.cli.prompts as P


@dataclass
class State:
    """The dataset part of a wizard state."""

    train_filename: str = ""
    use_separate_test: bool = False
    test_filename: str = ""
    test_size: float = 0.05


@pytest.fixture
def silence_output(monkeypatch):
    monkeypatch.setattr(P, "_info", lambda msg: None)
    monkeypatch.setattr(P, "_success", lambda msg: None)
    monkeypatch.setattr(P, "_section", lambda title: None)


# ---------------------------------------------------------------------------
# Prompt and output helpers (the layer CLI1 moves to ``cli/P.py`` on top of a shared console).
# Both worlds of each helper are exercised by switching the module flags: with or without ``questionary`` (a fake
# replaces its interactive prompts, an external boundary that needs a terminal) and with or without ``rich``
# (the plain text is checked character for character, the rich text by its words and glyph).
# ---------------------------------------------------------------------------


class FakeQuestionary:
    """Stands in for ``questionary``: each prompt returns the next scripted answer (``None`` is a cancelled prompt)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def _prompt(self, kind):
        def build(message, **kwargs):
            self.calls.append((kind, message, kwargs))
            return types.SimpleNamespace(ask=lambda: self.answers.pop(0))

        return build

    def __getattr__(self, name):
        if name in {"select", "text", "path", "confirm"}:
            return self._prompt(name)
        raise AttributeError(name)


@pytest.fixture
def plain(monkeypatch):
    """No rich, no questionary: the plain-text world."""
    monkeypatch.setattr(P, "_HAS_RICH", False)
    monkeypatch.setattr(P, "_HAS_QUESTIONARY", False)


@pytest.fixture
def scripted_input(monkeypatch):
    """``input()`` returns the scripted lines in order; the prompts shown are recorded."""

    def install(*lines):
        prompts = []
        remaining = list(lines)

        def fake_input(prompt=""):
            prompts.append(prompt)
            return remaining.pop(0)

        monkeypatch.setattr(builtins, "input", fake_input)
        return prompts

    return install


@pytest.mark.parametrize(
    ("helper", "plain_text"),
    [
        (P._info, "  -> hello\n"),
        (P._success, "  OK: hello\n"),
        (P._warning, "  WARNING: hello\n"),
        (P._error, "  ERROR: hello\n"),
        (P._section, "\n-- hello\n"),
    ],
)
def test_message_helpers_print_plain_text_without_rich(
    plain, capsys, helper, plain_text
):
    helper("hello")
    assert capsys.readouterr().out == plain_text


@pytest.mark.parametrize(
    ("helper", "glyph"),
    [
        (P._info, "→"),
        (P._success, "✓"),
        (P._warning, "!"),
        (P._error, "✗"),
        (P._section, "──"),
    ],
)
def test_message_helpers_print_the_message_with_a_glyph_with_rich(
    monkeypatch, capsys, helper, glyph
):
    monkeypatch.setattr(P, "_HAS_RICH", True)
    helper("hello")
    out = capsys.readouterr().out
    assert "hello" in out
    assert glyph in out


def test_header_is_a_ruled_title_without_rich_and_a_panel_with_rich(
    monkeypatch, capsys
):
    monkeypatch.setattr(P, "_HAS_RICH", False)
    P._print_header()
    assert (
        capsys.readouterr().out
        == "=" * 50 + "\n  GRACEmaker input.yaml wizard\n" + "=" * 50 + "\n"
    )
    monkeypatch.setattr(P, "_HAS_RICH", True)
    P._print_header()
    out = capsys.readouterr().out
    assert "GRACEmaker input.yaml wizard" in out
    assert "navigate with arrow keys" in out


@pytest.mark.parametrize(
    ("has_questionary", "has_readline", "tip"),
    [
        (True, True, "Tab ↹ autocompletes path"),
        (False, True, "↑↓ for command history"),
        (False, False, "Type or paste the full file path"),
    ],
)
def test_path_tip_depends_on_what_the_terminal_offers(
    monkeypatch, capsys, has_questionary, has_readline, tip
):
    monkeypatch.setattr(P, "_HAS_RICH", False)
    monkeypatch.setattr(P, "_HAS_QUESTIONARY", has_questionary)
    monkeypatch.setattr(P, "no_readline", not has_readline)
    P._tip_path_input()
    out = capsys.readouterr().out
    assert tip in out
    assert out.startswith("  (") and out.endswith(")\n")


def test_path_tip_with_rich_is_printed_by_the_console(monkeypatch, capsys):
    monkeypatch.setattr(P, "_HAS_RICH", True)
    monkeypatch.setattr(P, "_HAS_QUESTIONARY", False)
    monkeypatch.setattr(P, "no_readline", True)
    P._tip_path_input()
    assert "Type or paste the full file path" in capsys.readouterr().out


# --- plain input fallbacks ---------------------------------------------------------------------------------------------


def test_input_choice_repeats_until_a_choice_is_valid_and_empty_means_the_default(
    scripted_input,
):
    prompts = scripted_input(
        "maybe",
        "",
    )
    assert P.input_choice("Pick", ["a", "b"], "b") == "b"
    assert prompts == ["Pick, available options: a, b (default = b): "] * 2


def test_input_choice_rejects_a_default_that_is_not_a_choice():
    with pytest.raises(AssertionError):
        P.input_choice("Pick", ["a", "b"], "c")


def test_input_with_default_returns_the_typed_text_or_the_default(scripted_input):
    prompts = scripted_input("typed", "")
    assert P.input_with_default("Name", "dflt") == "typed"
    assert P.input_with_default("Name", "dflt") == "dflt"
    assert prompts == ["Name: ", "Name: "]


def test_input_no_default_asks_again_until_something_is_typed(scripted_input):
    prompts = scripted_input("", "", "value")
    assert P.input_no_default("Path ") == "value"
    assert prompts == ["Path "] * 3


# --- ask helpers without questionary -------------------------------------------------------------------------------------


def test_ask_select_without_questionary_takes_the_values_of_choice_objects(
    plain, scripted_input
):
    choices = [
        types.SimpleNamespace(value="x", title="X title"),
        types.SimpleNamespace(value="y", title="Y title"),
    ]
    prompts = scripted_input("y")
    assert P._ask_select("Which", choices, default="x") == "y"
    assert prompts == ["Which, available options: x, y (default = x): "]


def test_ask_select_without_questionary_falls_back_to_the_first_choice_for_an_unknown_default(
    plain, scripted_input
):
    scripted_input("")
    assert P._ask_select("Which", ["p", "q"], default="zzz") == "p"


def test_ask_text_and_path_and_confirm_without_questionary(plain, scripted_input):
    scripted_input("typed", "", "some/file", "no", "", "yes")
    assert P._ask_text("Name", "dflt") == "typed"
    assert P._ask_text("Name", "dflt") == "dflt"
    assert P._ask_path("File:") == "some/file"
    assert P._ask_confirm("Sure?", default=True) is False
    assert P._ask_confirm("Sure?", default=False) is False
    assert P._ask_confirm("Sure?", default=False) is True


# --- ask helpers with questionary ----------------------------------------------------------------------------------------


@pytest.fixture
def with_questionary(monkeypatch):
    monkeypatch.setattr(P, "_HAS_QUESTIONARY", True)

    def install(*answers):
        fake = FakeQuestionary(*answers)
        monkeypatch.setattr(P, "questionary", fake, raising=False)
        return fake

    return install


def test_ask_select_with_questionary_passes_the_default_and_returns_the_answer(
    with_questionary,
):
    fake = with_questionary("b")
    assert P._ask_select("Which", ["a", "b"], default="a") == "b"
    kind, message, kwargs = fake.calls[0]
    assert (kind, message, kwargs["choices"], kwargs["default"]) == (
        "select",
        "Which",
        ["a", "b"],
        "a",
    )
    assert kwargs["style"] is P._QUESTIONARY_STYLE


def test_ask_select_with_questionary_finds_the_default_among_choice_objects_or_uses_the_first(
    with_questionary,
):
    first, second = types.SimpleNamespace(value="x"), types.SimpleNamespace(value="y")
    fake = with_questionary("y", "x")
    P._ask_select("Which", [first, second], default="y")
    P._ask_select("Which", [first, second], default="unknown")
    assert fake.calls[0][2]["default"] is second
    assert fake.calls[1][2]["default"] is first


def test_ask_text_with_questionary_returns_the_default_for_an_empty_answer(
    with_questionary,
):
    fake = with_questionary("typed", "", "")
    assert P._ask_text("Name", 5) == "typed"
    assert P._ask_text("Name", 5) == 5
    assert P._ask_text("Name") is None
    assert [c[2]["default"] for c in fake.calls] == ["5", "5", ""]


def test_ask_path_and_confirm_with_questionary(with_questionary):
    fake = with_questionary("a/b.pkl.gz", True, False)
    assert P._ask_path("File:") == "a/b.pkl.gz"
    assert P._ask_confirm("Sure?", default=False) is True
    assert P._ask_confirm("Sure?") is False
    assert [c[0] for c in fake.calls] == ["path", "confirm", "confirm"]
    assert fake.calls[1][2]["default"] is False
    assert fake.calls[2][2]["default"] is True


@pytest.mark.parametrize(
    "call",
    [
        lambda: P._ask_select("m", ["a"]),
        lambda: P._ask_text("m"),
        lambda: P._ask_path("m"),
        lambda: P._ask_confirm("m"),
    ],
)
def test_a_cancelled_questionary_prompt_raises_keyboard_interrupt(
    with_questionary, call
):
    with_questionary(None)
    with pytest.raises(KeyboardInterrupt):
        call()


# --- the dataset section -------------------------------------------------------------------------------------------------


def _scripted_prompts(
    monkeypatch,
    path="train.pkl.gz",
    separate=False,
    test_path="test.pkl.gz",
    fraction=0.1,
):
    monkeypatch.setattr(
        P, "_ask_path", lambda msg: test_path if "Test" in msg else path
    )
    monkeypatch.setattr(P, "_ask_confirm", lambda msg, default=True: separate)
    monkeypatch.setattr(P, "_ask_text", lambda msg, default=None: str(fraction))


def test_section_dataset_with_a_split_from_the_training_file(
    monkeypatch, silence_output
):
    _scripted_prompts(monkeypatch, fraction=0.2)
    s = P._section_dataset(State())
    assert (s.train_filename, s.use_separate_test, s.test_size) == (
        "train.pkl.gz",
        False,
        0.2,
    )


def test_section_dataset_with_a_separate_test_file(monkeypatch, silence_output):
    _scripted_prompts(monkeypatch, separate=True)
    s = P._section_dataset(State())
    assert (s.train_filename, s.use_separate_test, s.test_filename) == (
        "train.pkl.gz",
        True,
        "test.pkl.gz",
    )
    assert s.test_size == State().test_size


def test_section_dataset_stops_when_no_file_name_is_entered(
    monkeypatch, silence_output
):
    _scripted_prompts(monkeypatch, path="")
    with pytest.raises(SystemExit) as stop:
        P._section_dataset(State())
    assert stop.value.code == 1


# --- the two wizards no longer import each other ------------------------------------------------------------------------


def _imports_of(path: Path) -> set[str]:
    """Every module a file imports, at the top of the file or inside a function."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


MODULES = {
    "tensorpotential.cli.wizard": "cli/wizard.py",
    "tensorpotential.cli.prompts": "cli/prompts.py",
    "tensorpotential.extra.gen_tensor.wizard": "extra/gen_tensor/wizard.py",
}


def _graph() -> dict[str, set[str]]:
    root = Path(P.__file__).resolve().parent.parent
    return {
        name: _imports_of(root / path) & set(MODULES) for name, path in MODULES.items()
    }


def test_the_static_import_graph_of_the_two_wizards_and_their_helpers_has_no_cycle():
    graph = _graph()
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        assert name not in trail, " -> ".join([*trail, name])
        if name in done:
            return
        for target in graph[name]:
            visit(target, (*trail, name))
        done.add(name)

    for name in graph:
        visit(name, ())


def test_the_generic_tensor_wizard_imports_the_prompts_and_not_the_input_wizard():
    graph = _graph()
    assert graph["tensorpotential.extra.gen_tensor.wizard"] == {
        "tensorpotential.cli.prompts"
    }
    assert graph["tensorpotential.cli.prompts"] == set()
    assert "tensorpotential.cli.prompts" in graph["tensorpotential.cli.wizard"]
