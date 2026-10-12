"""Tests for tensorpotential.cli.prompts: the prompt and message helpers of the input wizards.

A fake replaces the interactive prompts of ``questionary`` (an external boundary that needs a terminal); the messages go
through the real rich console and are checked by their words and glyph.
"""

from __future__ import annotations

import ast
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
# Prompt and output helpers (on top of the shared console)
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
def test_message_helpers_print_the_message_with_a_glyph(capsys, helper, glyph):
    helper("hello")
    out = capsys.readouterr().out
    assert "hello" in out
    assert glyph in out


def test_header_is_a_panel_with_the_title_and_the_navigation_hint(capsys):
    P._print_header()
    out = capsys.readouterr().out
    assert "GRACEmaker input.yaml wizard" in out
    assert "navigate with arrow keys" in out


def test_path_tip_says_what_the_questionary_path_prompt_offers(capsys):
    P._tip_path_input()
    assert "Tab ↹ autocompletes path" in capsys.readouterr().out


# --- ask helpers ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def with_questionary(monkeypatch):
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
