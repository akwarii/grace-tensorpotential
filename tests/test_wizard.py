"""Tests for tensorpotential.cli.wizard."""
from __future__ import annotations

import pytest

import tensorpotential.cli.wizard as wizard
from tensorpotential.cli.wizard import (
    _WizardState,
    _is_fs_model,
    _apply_state,
    _section_optimizer,
    _section_loss,
    _section_weighting,
)


# ---------------------------------------------------------------------------
# Minimal template for _apply_state tests
# ---------------------------------------------------------------------------

_MINIMAL_TEMPLATE = """\
cutoff: {{CUTOFF}}
data:
  filename: {{TRAIN_FILENAME}}
  {{TEST_DATA}}
  save_dataset: {{SAVE_DATASET}}
potential:
  {{POTENTIAL_SETTINGS}}
  {{TRAINABLE_VARIABLE_NAMES}}
fit:
  loss: {
    energy: { weight: {{ENERGY_LOSS_WEIGHT}}, type: {{LOSS_TYPE}} {{EXTRA_E_ARGS}} },
    forces: { weight: {{FORCE_LOSS_WEIGHT}}, type: {{LOSS_TYPE}} {{EXTRA_E_ARGS}} },
    {{STRESS_LOSS}}
    {{SWITCH_LOSS}}
  }
  {{WEIGHTING_SCHEME}}
  target_total_updates: {{TARGET_TOTAL_UPDATES}}
{{OPTIMIZER_BLOCK}}
  compute_convex_hull: {{COMPUTE_CONVEX_HULL}}
  batch_size: {{BATCH_SIZE}}
  test_batch_size: {{TEST_BATCH_SIZE}}
  eval_init_stats: {{eval_init_stats}}
  reset_epoch_and_step: {{RESET_EPOCH}}
"""


def _make_state(**kwargs) -> _WizardState:
    """Create a _WizardState with sensible defaults for testing."""
    s = _WizardState()
    s.train_filename = "train.pkl.gz"
    s.preset_name = "GRACE_1LAYER_latest"
    s.cutoff = 6.0
    s.preset_kwargs_str = "{}"
    s.fit_type = "fit from scratch"
    s.finetune_model = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Unit tests: _is_fs_model
# ---------------------------------------------------------------------------


def test_is_fs_model_preset():
    s = _make_state(preset_name="FS")
    assert _is_fs_model(s) is True


def test_is_fs_model_foundation():
    s = _make_state(preset_name="", foundation_model_name="GRACE-FS-OAM")
    assert _is_fs_model(s) is True


def test_is_fs_model_non_fs():
    s = _make_state(preset_name="GRACE_1LAYER_latest")
    assert _is_fs_model(s) is False


def test_is_fs_model_non_fs_foundation():
    s = _make_state(preset_name="", foundation_model_name="GRACE-2L-OAM")
    assert _is_fs_model(s) is False


# ---------------------------------------------------------------------------
# Unit tests: _apply_state
# ---------------------------------------------------------------------------


def test_apply_state_adam():
    s = _make_state(
        optimizer="Adam",
        learning_rate=0.008,
        use_switch=False,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "optimizer: Adam" in result
    assert "scheduler: cosine_decay" in result
    assert "learning_rate: 0.008" in result
    # No L-BFGS-B content
    assert "maxcor" not in result


def _assert_apply_state_bfgs_family(optimizer):
    """A (L-)BFGS state writes its optimizer and ``maxcor`` and no Adam-specific content."""
    s = _make_state(
        optimizer=optimizer,
        bfgs_maxcor=100,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert f"optimizer: {optimizer}" in result
    assert "maxcor" in result
    assert "scheduler:" not in result
    assert "learning_rate:" not in result


def test_apply_state_lbfgsb():
    _assert_apply_state_bfgs_family("L-BFGS-B")


def test_apply_state_bfgs():
    _assert_apply_state_bfgs_family("BFGS")


def test_apply_state_adam_with_switch():
    s = _make_state(
        optimizer="Adam",
        learning_rate=0.001,
        use_switch=True,
        switch_after=0.75,
        lr_reduction_factor=0.1,
        new_energy_w=128,
        new_force_w=32,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "scheduler: reduce_on_plateau" in result
    assert "switch:" in result


def test_apply_state_no_switch():
    s = _make_state(optimizer="Adam", use_switch=False)
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    # The {{SWITCH_LOSS}} placeholder should be replaced with empty string
    assert "{{SWITCH_LOSS}}" not in result
    assert "switch:" not in result


def test_apply_state_no_dangling_placeholders():
    """After applying state, no {{...}} placeholders should remain."""
    s = _make_state(optimizer="L-BFGS-B")
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "{{" not in result, f"Found unresolved placeholders in:\n{result}"


def test_apply_state_adam_no_dangling_placeholders():
    s = _make_state(optimizer="Adam", use_switch=False)
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "{{" not in result, f"Found unresolved placeholders in:\n{result}"


# ---------------------------------------------------------------------------
# Section-level tests (monkeypatched _ask_*)
# ---------------------------------------------------------------------------


@pytest.fixture
def silence_output(monkeypatch):
    """Suppress print-based output helpers."""
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)


def _patch_ask_defaults(monkeypatch):
    """Patch all _ask_* to return their defaults."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "data.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)


def test_section_optimizer_fs_scratch_defaults_bfgs(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="fit from scratch")
    s = _section_optimizer(s)
    assert s.optimizer == "BFGS"


def test_section_optimizer_fs_finetune_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="finetune foundation model", finetune_model=True)
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_fs_continue_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="continue fit", finetune_model=True)
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_fs_foundation_finetune_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(
        preset_name="",
        foundation_model_name="GRACE-FS-OAM",
        fit_type="finetune foundation model",
        finetune_model=True,
    )
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_grace_defaults_adam(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="GRACE_1LAYER_latest", fit_type="fit from scratch")
    s = _section_optimizer(s)
    assert s.optimizer == "Adam"


def _use_switch_after_section_loss(monkeypatch, optimizer):
    """``use_switch`` after ``_section_loss`` when every prompt takes its default and the switch confirm says yes."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: True)
    s = _make_state(optimizer=optimizer)
    s = _section_loss(s)
    return s.use_switch


def test_section_loss_no_switch_for_bfgs(monkeypatch, silence_output):
    """For BFGS optimizer, switch prompt is skipped and use_switch forced False."""
    # _ask_confirm always returns True — if the switch prompt is NOT skipped,
    # use_switch would be True; we verify it stays False for BFGS.
    assert _use_switch_after_section_loss(monkeypatch, "BFGS") is False


def test_section_loss_no_switch_for_lbfgsb(monkeypatch, silence_output):
    assert _use_switch_after_section_loss(monkeypatch, "L-BFGS-B") is False


def test_section_loss_switch_allowed_for_adam(monkeypatch, silence_output):
    """For Adam, the switch prompt is asked and respected."""
    assert _use_switch_after_section_loss(monkeypatch, "Adam") is True


def test_section_weighting_bfgs_scratch_defaults(monkeypatch, silence_output):
    """For BFGS + from scratch, default iterations should be 500."""
    captured = {}

    def fake_ask_text(msg, default=None):
        captured[msg] = default
        return default

    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", fake_ask_text)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="BFGS", fit_type="fit from scratch")
    s = _section_weighting(s)
    assert s.target_total_updates == 500


def test_section_weighting_bfgs_finetune_defaults(monkeypatch, silence_output):
    """For BFGS + finetune, default iterations should be 100."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="BFGS", fit_type="finetune foundation model", finetune_model=True)
    s = _section_weighting(s)
    assert s.target_total_updates == 100


def test_section_weighting_adam_scratch_defaults(monkeypatch, silence_output):
    """For Adam + from scratch, default total updates should be 50000."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="Adam", fit_type="fit from scratch")
    s = _section_weighting(s)
    assert s.target_total_updates == 50000


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


def test_e2e_fs_lbfgsb(monkeypatch, tmp_path):
    """FS from scratch with L-BFGS-B chosen → valid input.yaml without scheduler/LR."""

    def fake_select(msg, choices, default=None):
        if "Optimizer" in msg:
            return "L-BFGS-B"
        if "Fit type" in msg:
            return "fit from scratch"
        if "preset" in msg.lower() or "Model preset" in msg:
            return "FS"
        # For complexity/size selections, return default
        return default

    monkeypatch.setattr(wizard, "_ask_select", fake_select)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "train.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)
    monkeypatch.setattr(wizard, "_print_header", lambda: None)
    monkeypatch.setattr(wizard, "_show_review", lambda s: None)
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)

    # Force non-questionary path for the review loop
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)
    # Provide "" to the review loop input → triggers "ok" (confirm)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        wizard.generate_template_input()

    assert exc_info.value.code == 0
    output_file = tmp_path / "input.yaml"
    assert output_file.exists(), "input.yaml was not written"
    content = output_file.read_text()
    assert "optimizer: L-BFGS-B" in content, "Expected L-BFGS-B optimizer in output"
    assert "scheduler:" not in content, "Unexpected scheduler key for L-BFGS-B"
    assert "learning_rate:" not in content, "Unexpected learning_rate key for L-BFGS-B"


def test_e2e_adam_has_scheduler(monkeypatch, tmp_path):
    """Adam optimizer → input.yaml should contain scheduler and learning_rate."""

    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "train.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)
    monkeypatch.setattr(wizard, "_print_header", lambda: None)
    monkeypatch.setattr(wizard, "_show_review", lambda s: None)
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)

    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        wizard.generate_template_input()

    assert exc_info.value.code == 0
    content = (tmp_path / "input.yaml").read_text()
    assert "optimizer: Adam" in content
    assert "scheduler:" in content
    assert "learning_rate:" in content


# ---------------------------------------------------------------------------
# SMAX-OMAT model tree + selection tests
# ---------------------------------------------------------------------------


_SMAX_OMAT_MODELS = [
    "GRACE-1L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-OMAT-medium",
]


def test_smax_omat_models_in_tree():
    """All three SMAX-OMAT models must appear in _FOUNDATION_MODEL_TREE."""
    from tensorpotential.cli.wizard import _FOUNDATION_MODEL_TREE

    all_names = []
    for tier_data in _FOUNDATION_MODEL_TREE.values():
        for ds_data in tier_data.values():
            for size_list in ds_data.values():
                all_names.extend(size_list)

    for model in _SMAX_OMAT_MODELS:
        assert model in all_names, f"{model!r} missing from _FOUNDATION_MODEL_TREE"


def test_pure_smax_models_not_in_tree():
    """Pure SMAX (non-OMAT) models must NOT appear in the wizard tree."""
    from tensorpotential.cli.wizard import _FOUNDATION_MODEL_TREE

    all_names = []
    for tier_data in _FOUNDATION_MODEL_TREE.values():
        for ds_data in tier_data.values():
            for size_list in ds_data.values():
                all_names.extend(size_list)

    for name in all_names:
        assert "SMAX" not in name or "SMAX-OMAT" in name, (
            f"Pure SMAX model {name!r} should not be in the wizard tree"
        )


def _ask_foundation_model_selecting(monkeypatch, *answers):
    """Result of ``_ask_foundation_model`` when the successive prompts are answered with ``answers``."""
    selections = iter(answers)
    monkeypatch.setattr(
        wizard,
        "_ask_select",
        lambda msg, choices, default=None: next(selections),
    )
    return wizard._ask_foundation_model()


def test_ask_foundation_model_smax_1l(monkeypatch, silence_output):
    """Selecting 1L → SMAX-OMAT → large returns the expected model."""
    result = _ask_foundation_model_selecting(monkeypatch, "1L", "SMAX-OMAT", "large")
    assert result == "GRACE-1L-SMAX-OMAT-large"


def test_ask_foundation_model_smax_2l_medium(monkeypatch, silence_output):
    """Selecting 2L → SMAX-OMAT → medium returns GRACE-2L-SMAX-OMAT-medium."""
    result = _ask_foundation_model_selecting(monkeypatch, "2L", "SMAX-OMAT", "medium")
    assert result == "GRACE-2L-SMAX-OMAT-medium"


# ---------------------------------------------------------------------------
# 3L model tree + selection tests
# ---------------------------------------------------------------------------

_3L_MODELS = ["GRACE-3L-OMAT-large", "GRACE-3L-OMAT-large-ft-AM"]


def test_3l_models_in_tree():
    """Both 3L models must appear in _FOUNDATION_MODEL_TREE under a '3L' tier."""
    from tensorpotential.cli.wizard import _FOUNDATION_MODEL_TREE, _TIER_DESC

    assert "3L" in _FOUNDATION_MODEL_TREE
    assert "3L" in _TIER_DESC

    all_names = []
    for tier_data in _FOUNDATION_MODEL_TREE.values():
        for ds_data in tier_data.values():
            for size_list in ds_data.values():
                all_names.extend(size_list)

    for model in _3L_MODELS:
        assert model in all_names, f"{model!r} missing from _FOUNDATION_MODEL_TREE"


def test_ask_foundation_model_3l_oam(monkeypatch, silence_output):
    """Selecting 3L -> OAM returns the ft-AM model (single size/variant, auto-picked)."""
    result = _ask_foundation_model_selecting(monkeypatch, "3L", "OAM")
    assert result == "GRACE-3L-OMAT-large-ft-AM"


def test_ask_foundation_model_3l_omat(monkeypatch, silence_output):
    """Selecting 3L -> OMAT returns the base model."""
    result = _ask_foundation_model_selecting(monkeypatch, "3L", "OMAT")
    assert result == "GRACE-3L-OMAT-large"


def test_apply_state_frozen_smax_2l():
    """Frozen finetuning of a 2L-SMAX-OMAT model uses the 2L trainable-variable pattern."""
    s = _make_state(
        finetune_model=True,
        fit_type="finetune foundation model",
        foundation_model_name="GRACE-2L-SMAX-OMAT-large",
        finetune_mode="frozen",
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "I2/reducing_" in result, "Expected 2L frozen-weight pattern in output"
    assert "rho/reducing_" in result


def test_apply_state_frozen_3l():
    """Frozen finetuning of a 3L model emits densities + inter-layer messages."""
    s = _make_state(
        finetune_model=True,
        fit_type="finetune foundation model",
        foundation_model_name="GRACE-3L-OMAT-large",
        finetune_mode="frozen",
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    for name in [
        "rho1/reducing_",
        "rho2/reducing_",
        "rho3/reducing_",
        "eq1/reducing_",
        "eq2/reducing_",
    ]:
        assert name in result, f"Expected {name!r} in 3L frozen output"
    # must NOT fall back to the generic 1L pattern
    assert "rho/reducing_" not in result


def test_apply_state_frozen_3l_ft_am():
    """The ft-AM 3L name also matches the 3L branch."""
    s = _make_state(
        finetune_model=True,
        fit_type="finetune foundation model",
        foundation_model_name="GRACE-3L-OMAT-large-ft-AM",
        finetune_mode="frozen",
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "rho1/reducing_" in result and "eq2/reducing_" in result


def test_apply_state_frozen_fs_raises():
    """Frozen finetuning of an FS model is unsupported and must raise."""
    s = _make_state(
        finetune_model=True,
        fit_type="finetune foundation model",
        foundation_model_name="GRACE-FS-OMAT",
        finetune_mode="frozen",
    )
    with pytest.raises(ValueError, match="1L/2L/3L"):
        _apply_state(s, _MINIMAL_TEMPLATE)


# ---------------------------------------------------------------------------
# Finetuning precision (fp32 default / fp64) tests
# ---------------------------------------------------------------------------


def _maybe_ask_fp64_variant_answering(monkeypatch, answer):
    """Result of ``_maybe_ask_fp64_variant`` for a model with a -fp64 sibling when the prompt gets ``answer``."""
    monkeypatch.setattr(
        wizard,
        "_ask_select",
        lambda msg, choices, default=None: answer,
    )
    return wizard._maybe_ask_fp64_variant("GRACE-2L-OMAT-large-base")


def test_maybe_ask_fp64_variant_chooses_fp64(monkeypatch):
    """A model with a -fp64 sibling: choosing FP64 appends the suffix."""
    out = _maybe_ask_fp64_variant_answering(
        monkeypatch, "FP64 (full precision — for older/full-precision workflows)"
    )
    assert out == "GRACE-2L-OMAT-large-base-fp64"


def test_maybe_ask_fp64_variant_chooses_fp32(monkeypatch):
    """A model with a -fp64 sibling: choosing FP32 keeps the bare name."""
    out = _maybe_ask_fp64_variant_answering(
        monkeypatch, "FP32 (default, recommended — faster, half memory)"
    )
    assert out == "GRACE-2L-OMAT-large-base"


def test_maybe_ask_fp64_variant_no_variant_does_not_ask(monkeypatch):
    """A model without a -fp64 sibling (3L): returns unchanged, asks nothing."""
    def _boom(*a, **k):
        raise AssertionError("_ask_select must not be called when no fp64 variant exists")

    monkeypatch.setattr(wizard, "_ask_select", _boom)
    out = wizard._maybe_ask_fp64_variant("GRACE-3L-OMAT-large")
    assert out == "GRACE-3L-OMAT-large"


# ---------------------------------------------------------------------------
# Prompt and output helpers (the layer CLI1 moves to ``cli/prompts.py`` on top of a shared console).
# Both worlds of each helper are exercised by switching the module flags: with or without ``questionary`` (a fake
# replaces its interactive prompts, an external boundary that needs a terminal) and with or without ``rich``
# (the plain text is checked character for character, the rich text by its words and glyph).
# ---------------------------------------------------------------------------

import builtins  # noqa: E402
import types  # noqa: E402


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
    monkeypatch.setattr(wizard, "_HAS_RICH", False)
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)


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
        (wizard._info, "  -> hello\n"),
        (wizard._success, "  OK: hello\n"),
        (wizard._warning, "  WARNING: hello\n"),
        (wizard._error, "  ERROR: hello\n"),
        (wizard._section, "\n-- hello\n"),
    ],
)
def test_message_helpers_print_plain_text_without_rich(plain, capsys, helper, plain_text):
    helper("hello")
    assert capsys.readouterr().out == plain_text


@pytest.mark.parametrize(
    ("helper", "glyph"),
    [(wizard._info, "→"), (wizard._success, "✓"), (wizard._warning, "!"), (wizard._error, "✗"), (wizard._section, "──")],
)
def test_message_helpers_print_the_message_with_a_glyph_with_rich(monkeypatch, capsys, helper, glyph):
    monkeypatch.setattr(wizard, "_HAS_RICH", True)
    helper("hello")
    out = capsys.readouterr().out
    assert "hello" in out
    assert glyph in out


def test_header_is_a_ruled_title_without_rich_and_a_panel_with_rich(monkeypatch, capsys):
    monkeypatch.setattr(wizard, "_HAS_RICH", False)
    wizard._print_header()
    assert capsys.readouterr().out == "=" * 50 + "\n  GRACEmaker input.yaml wizard\n" + "=" * 50 + "\n"
    monkeypatch.setattr(wizard, "_HAS_RICH", True)
    wizard._print_header()
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
def test_path_tip_depends_on_what_the_terminal_offers(monkeypatch, capsys, has_questionary, has_readline, tip):
    monkeypatch.setattr(wizard, "_HAS_RICH", False)
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", has_questionary)
    monkeypatch.setattr(wizard, "no_readline", not has_readline)
    wizard._tip_path_input()
    out = capsys.readouterr().out
    assert tip in out
    assert out.startswith("  (") and out.endswith(")\n")


def test_path_tip_with_rich_is_printed_by_the_console(monkeypatch, capsys):
    monkeypatch.setattr(wizard, "_HAS_RICH", True)
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)
    monkeypatch.setattr(wizard, "no_readline", True)
    wizard._tip_path_input()
    assert "Type or paste the full file path" in capsys.readouterr().out


# --- plain input fallbacks ---------------------------------------------------------------------------------------------


def test_input_choice_repeats_until_a_choice_is_valid_and_empty_means_the_default(scripted_input):
    prompts = scripted_input("maybe", "", )
    assert wizard.input_choice("Pick", ["a", "b"], "b") == "b"
    assert prompts == ["Pick, available options: a, b (default = b): "] * 2


def test_input_choice_rejects_a_default_that_is_not_a_choice():
    with pytest.raises(AssertionError):
        wizard.input_choice("Pick", ["a", "b"], "c")


def test_input_with_default_returns_the_typed_text_or_the_default(scripted_input):
    prompts = scripted_input("typed", "")
    assert wizard.input_with_default("Name", "dflt") == "typed"
    assert wizard.input_with_default("Name", "dflt") == "dflt"
    assert prompts == ["Name: ", "Name: "]


def test_input_no_default_asks_again_until_something_is_typed(scripted_input):
    prompts = scripted_input("", "", "value")
    assert wizard.input_no_default("Path ") == "value"
    assert prompts == ["Path "] * 3


# --- ask helpers without questionary -------------------------------------------------------------------------------------


def test_ask_select_without_questionary_takes_the_values_of_choice_objects(plain, scripted_input):
    choices = [types.SimpleNamespace(value="x", title="X title"), types.SimpleNamespace(value="y", title="Y title")]
    prompts = scripted_input("y")
    assert wizard._ask_select("Which", choices, default="x") == "y"
    assert prompts == ["Which, available options: x, y (default = x): "]


def test_ask_select_without_questionary_falls_back_to_the_first_choice_for_an_unknown_default(plain, scripted_input):
    scripted_input("")
    assert wizard._ask_select("Which", ["p", "q"], default="zzz") == "p"


def test_ask_text_and_path_and_confirm_without_questionary(plain, scripted_input):
    scripted_input("typed", "", "some/file", "no", "", "yes")
    assert wizard._ask_text("Name", "dflt") == "typed"
    assert wizard._ask_text("Name", "dflt") == "dflt"
    assert wizard._ask_path("File:") == "some/file"
    assert wizard._ask_confirm("Sure?", default=True) is False
    assert wizard._ask_confirm("Sure?", default=False) is False
    assert wizard._ask_confirm("Sure?", default=False) is True


# --- ask helpers with questionary ----------------------------------------------------------------------------------------


@pytest.fixture
def with_questionary(monkeypatch):
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", True)

    def install(*answers):
        fake = FakeQuestionary(*answers)
        monkeypatch.setattr(wizard, "questionary", fake, raising=False)
        return fake

    return install


def test_ask_select_with_questionary_passes_the_default_and_returns_the_answer(with_questionary):
    fake = with_questionary("b")
    assert wizard._ask_select("Which", ["a", "b"], default="a") == "b"
    kind, message, kwargs = fake.calls[0]
    assert (kind, message, kwargs["choices"], kwargs["default"]) == ("select", "Which", ["a", "b"], "a")
    assert kwargs["style"] is wizard._QUESTIONARY_STYLE


def test_ask_select_with_questionary_finds_the_default_among_choice_objects_or_uses_the_first(with_questionary):
    first, second = types.SimpleNamespace(value="x"), types.SimpleNamespace(value="y")
    fake = with_questionary("y", "x")
    wizard._ask_select("Which", [first, second], default="y")
    wizard._ask_select("Which", [first, second], default="unknown")
    assert fake.calls[0][2]["default"] is second
    assert fake.calls[1][2]["default"] is first


def test_ask_text_with_questionary_returns_the_default_for_an_empty_answer(with_questionary):
    fake = with_questionary("typed", "", "")
    assert wizard._ask_text("Name", 5) == "typed"
    assert wizard._ask_text("Name", 5) == 5
    assert wizard._ask_text("Name") is None
    assert [c[2]["default"] for c in fake.calls] == ["5", "5", ""]


def test_ask_path_and_confirm_with_questionary(with_questionary):
    fake = with_questionary("a/b.pkl.gz", True, False)
    assert wizard._ask_path("File:") == "a/b.pkl.gz"
    assert wizard._ask_confirm("Sure?", default=False) is True
    assert wizard._ask_confirm("Sure?") is False
    assert [c[0] for c in fake.calls] == ["path", "confirm", "confirm"]
    assert fake.calls[1][2]["default"] is False
    assert fake.calls[2][2]["default"] is True


@pytest.mark.parametrize(
    "call",
    [
        lambda: wizard._ask_select("m", ["a"]),
        lambda: wizard._ask_text("m"),
        lambda: wizard._ask_path("m"),
        lambda: wizard._ask_confirm("m"),
    ],
)
def test_a_cancelled_questionary_prompt_raises_keyboard_interrupt(with_questionary, call):
    with_questionary(None)
    with pytest.raises(KeyboardInterrupt):
        call()


# --- the dataset section -------------------------------------------------------------------------------------------------


def _scripted_prompts(monkeypatch, path="train.pkl.gz", separate=False, test_path="test.pkl.gz", fraction=0.1):
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: test_path if "Test" in msg else path)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: separate)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: str(fraction))


def test_section_dataset_with_a_split_from_the_training_file(monkeypatch, silence_output):
    _scripted_prompts(monkeypatch, fraction=0.2)
    s = wizard._section_dataset(_WizardState())
    assert (s.train_filename, s.use_separate_test, s.test_size) == ("train.pkl.gz", False, 0.2)


def test_section_dataset_with_a_separate_test_file(monkeypatch, silence_output):
    _scripted_prompts(monkeypatch, separate=True)
    s = wizard._section_dataset(_WizardState())
    assert (s.train_filename, s.use_separate_test, s.test_filename) == ("train.pkl.gz", True, "test.pkl.gz")
    assert s.test_size == _WizardState().test_size


def test_section_dataset_stops_when_no_file_name_is_entered(monkeypatch, silence_output):
    _scripted_prompts(monkeypatch, path="")
    with pytest.raises(SystemExit) as stop:
        wizard._section_dataset(_WizardState())
    assert stop.value.code == 1
