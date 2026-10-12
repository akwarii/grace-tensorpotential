"""Tests for tools/lint_ratchet.py.

The end-to-end tests run the real ruff and ty on a scratch project that carries the ruff tables
of the repository's ``pyproject.toml``, so they also check the two mechanisms of TOOL1: the strict
set flags a module in a new package and not the same code in legacy, and so do the ty overrides.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lint_ratchet as lr

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"

# Violates the strict set (I001, T201, ERA001, UP006, D103, PTH118, ...) and ty (invalid-parameter-default,
# unresolved-attribute). In legacy only the commented-out code (ERA001) and os.path (PTH118) are counted.
PROBE = """import sys
import os
from typing import List

# x = compute(1)
def f(a: int = None) -> int:
    values: List[int] = []
    print(os.path.join("a", sys.argv[0]), values)
    return a.missing
"""
CLEAN = "def f(a: int) -> int:\n    return a\n"
LEGACY = ["legacy/mod.py"]
# The legacy list of TOOL3 had this many files; it shrinks, so a longer one is a hand edit.
LEGACY_CEILING = 272
ERA = "# x = compute(1)\nVALUE = 1\n"


def _repo_ruff_tables() -> str:
    """The ``[tool.ruff*]`` tables of the repository, as text."""
    text = PYPROJECT.read_text()
    return text[text.index("[tool.ruff]") : text.index("[dependency-groups]")]


def _pyproject(ruff: str = "0.16.7", ty: str = "0.0.84") -> str:
    return f"""[project]
name = "scratch"
version = "0"

[dependency-groups]
dev = ["ruff=={ruff}", "ty=={ty}"]

{_repo_ruff_tables()}
[tool.ty.environment]
python = "{sys.prefix}"

[[tool.ty.overrides]]
include = ["legacy/**"]

[tool.ty.overrides.rules]
invalid-parameter-default = "ignore"
unresolved-attribute = "ignore"
"""


def _installed(tool: str) -> str:
    return lr.tool_version(tool, REPO)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Scratch project: a legacy module, a strict package and the ruff tables of the repository."""
    (tmp_path / "legacy").mkdir()
    (tmp_path / "tests_torch").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        _pyproject(_installed("ruff"), _installed("ty"))
    )
    (tmp_path / "legacy" / "mod.py").write_text(PROBE)
    (tmp_path / "tests_torch" / "test_ok.py").write_text(CLEAN)
    return tmp_path


def _baseline(project: Path) -> Path:
    return project / "baselines" / "lint_ratchet.json"


def _run(project: Path, *args: str) -> int:
    return lr.main([*args, "--root", str(project)])


# --- logic: pure functions ------------------------------------------------------------------


def test_in_new_package_matches_whole_path_components() -> None:
    assert lr.in_new_package("tests_torch/a.py")
    assert lr.in_new_package("tensorpotential/core/x/y.py")
    assert not lr.in_new_package("tensorpotential/core_extra/a.py")
    assert not lr.in_new_package("tests/a.py")
    assert not lr.in_new_package("tensorpotential/torch_backend_old/a.py")


def test_tally_counts_per_file_and_rule_and_separates_new_packages() -> None:
    findings = [
        ("a.py", "F401"),
        ("a.py", "F401"),
        ("a.py", "E722"),
        ("tests_torch/t.py", "I001"),
    ]
    legacy, strict = lr.tally(findings, lr.NEW_PACKAGES, ["a.py"], lambda _: True)
    assert legacy == {"a.py": {"E722": 1, "F401": 2}}
    assert strict == ["tests_torch/t.py: I001"]


def test_tally_keeps_only_the_wanted_rules_of_a_legacy_file() -> None:
    findings = [("a.py", "F401"), ("a.py", "D103"), ("a.py", "T201")]
    legacy, strict = lr.tally(findings, lr.NEW_PACKAGES, ["a.py"], lr.is_counted)
    assert legacy == {"a.py": {"F401": 1}}
    assert strict == []


def test_tally_makes_a_file_in_neither_list_strict_for_every_rule() -> None:
    findings = [("data/new.py", "PTH118"), ("data/new.py", "D103"), ("old.py", "D103")]
    legacy, strict = lr.tally(findings, lr.NEW_PACKAGES, ["old.py"], lr.is_counted)
    assert legacy == {}
    assert [line.split(" (")[0] for line in strict] == [
        "data/new.py: D103",
        "data/new.py: PTH118",
    ]
    assert all("baselines/legacy_files.txt" in line for line in strict)


COUNTED = (
    "E722", "F401", "F841", "ERA001", "C901", "PLR0912", "PLR0913", "PLR0915", "BLE001", "B905",
    "B007", "RUF012", "NPY002", "SIM102", "RET504", "PLW2901", "PTH118", "S301", "S603",
    "invalid-syntax", "syntax-error",
)  # fmt: skip
NOT_COUNTED = (
    "S101", "D103", "D401", "T201", "PLR0911", "RUF100", "RUF059", "I001", "UP006", "C408",
    "NPY001", "TRY003", "ARG001", "PERF401", "W293", "PD011", "A002", "TD002", "FIX002", "PIE808",
    "PLE2515", "ERA", "unresolved-attribute",
)  # fmt: skip


@pytest.mark.parametrize("rule", COUNTED)
def test_the_counted_rules_are_counted_for_a_legacy_file(rule: str) -> None:
    assert lr.is_counted(rule)


@pytest.mark.parametrize("rule", NOT_COUNTED)
def test_the_other_rules_are_left_out_for_a_legacy_file(rule: str) -> None:
    assert not lr.is_counted(rule)


def test_compare_reports_rises_and_drops_by_file_and_rule() -> None:
    base = {"a.py": {"F401": 2, "E722": 1}, "b.py": {"F841": 1}}
    new = {"a.py": {"F401": 3}, "c.py": {"F841": 1}}
    rises, drops = lr.compare(base, new)
    assert rises == ["a.py F401: 2 -> 3", "c.py F841: 0 -> 1"]
    assert drops == ["a.py E722: 1 -> 0", "b.py F841: 1 -> 0"]


def test_compare_ignores_equal_counts_and_line_moves() -> None:
    same = {"a.py": {"F401": 2}}
    assert lr.compare(same, {"a.py": {"F401": 2}}) == ([], [])


def test_json_list_rejects_garbage_and_non_lists() -> None:
    assert lr._json_list("[]", "x") == []
    with pytest.raises(lr.ToolError, match="unreadable JSON"):
        lr._json_list("not json", "x")
    with pytest.raises(lr.ToolError, match="expected a list"):
        lr._json_list("{}", "x")


def test_parse_ruff_keys_a_finding_without_a_code_as_a_syntax_error(
    tmp_path: Path,
) -> None:
    text = json.dumps([
        {"filename": str(tmp_path / "a.py"), "code": "F401"},
        {"filename": str(tmp_path / "sub" / "b.py"), "code": None},
    ])
    assert lr.parse_ruff(text, tmp_path) == [
        ("a.py", "F401"),
        ("sub/b.py", "syntax-error"),
    ]


def test_run_raises_for_an_exit_code_that_is_not_accepted(tmp_path: Path) -> None:
    with pytest.raises(lr.ToolError, match="exited with 3"):
        lr._run([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, (0, 1))


def test_exe_reports_a_missing_tool() -> None:
    with pytest.raises(lr.ToolError, match="not installed"):
        lr._exe("no-such-linter-zz")


def test_declared_versions_reads_pins_hooks_and_docs(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('dev = ["ruff==1.2.3", "ty==0.0.9"]')
    (tmp_path / "CLAUDE.md").write_text("run uvx ruff@1.2.3 and ruff 1.2.4 here")
    skills = tmp_path / ".claude" / "skills" / "s"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("ty==0.0.9")
    assert lr.declared_versions(tmp_path) == {
        "ruff": {"1.2.3", "1.2.4"},
        "ty": {"0.0.9"},
    }


def test_version_problems_flag_a_stale_pin_and_a_stale_recording(
    tmp_path: Path,
) -> None:
    ruff, ty = _installed("ruff"), _installed("ty")
    (tmp_path / "pyproject.toml").write_text(f'dev = ["ruff==0.0.1", "ty=={ty}"]')
    problems = lr.version_problems(tmp_path, {"ruff": ruff, "ty": "9.9.9"})
    assert len(problems) == 2
    assert problems[0].startswith("ruff:") and "0.0.1" in problems[0]
    assert problems[1].startswith("ty:") and "9.9.9" in problems[1]


def test_repository_names_the_installed_versions_of_ruff_and_ty() -> None:
    assert lr.version_problems(REPO) == []


# --- the two mechanisms of M0.7, with the real tools -------------------------------------


def test_strict_set_flags_a_new_package_and_not_legacy(project: Path) -> None:
    shutil.copy(project / "legacy" / "mod.py", project / "tests_torch" / "mod.py")
    now = lr.measure(project, LEGACY)
    assert now["ruff"] == {
        "legacy/mod.py": {"ERA001": 1, "PTH118": 1}
    }  # legacy: only the counted rules (not I001, T201, UP006, D103, ...)
    assert now["ty"] == {}  # ... and the ty overrides silence the legacy rules
    strict = " ".join(now["strict"])
    for rule in (
        "I001",
        "T201",
        "ERA001",
        "UP006",
        "invalid-parameter-default",
        "unresolved-attribute",
    ):
        assert f"tests_torch/mod.py: {rule}" in strict


def test_legacy_keeps_the_ratchet_rules_and_the_unsilenced_ty_rules(
    project: Path,
) -> None:
    (project / "legacy" / "mod.py").write_text(
        ERA + "def g(a: int) -> int:\n    return a + 'x'\n"
    )
    now = lr.measure(project, LEGACY)
    assert now["ruff"] == {"legacy/mod.py": {"ERA001": 1}}
    assert now["ty"] == {"legacy/mod.py": {"unsupported-operator": 1}}
    assert now["strict"] == []


def test_unresolved_import_is_not_counted(project: Path) -> None:
    (project / "legacy" / "mod.py").write_text(
        "import not_installed_zz\n\nVALUE = not_installed_zz\n"
    )
    assert lr.measure(project, LEGACY)["ty"] == {}


def test_a_syntax_error_counts_as_a_finding_not_as_zero(project: Path) -> None:
    (project / "legacy" / "mod.py").write_text("x = (\n")
    assert lr.measure(project, LEGACY)["ruff"] == {
        "legacy/mod.py": {"invalid-syntax": 1}
    }


# --- the ratchet as a command ----------------------------------------------------------------


def test_record_then_check_is_ok(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    data = json.loads(_baseline(project).read_text())
    assert data["versions"] == {"ruff": _installed("ruff"), "ty": _installed("ty")}
    assert _run(project, "check") == 0
    assert "lint ratchet: ok" in capsys.readouterr().out


def test_a_rise_in_legacy_fails_and_names_file_and_rule(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    (project / "legacy" / "mod.py").write_text(PROBE + ERA)
    capsys.readouterr()
    assert _run(project, "check") == 1
    assert "rose: ruff legacy/mod.py ERA001: 1 -> 2" in capsys.readouterr().out


def test_a_finding_in_a_new_package_fails_without_a_baseline_entry(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    (project / "tests_torch" / "test_ok.py").write_text("import os\n")
    capsys.readouterr()
    assert _run(project, "check") == 1
    assert "strict: tests_torch/test_ok.py: F401" in capsys.readouterr().out


def test_a_drop_prints_the_record_command_and_passes_unless_asked(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "legacy" / "mod.py").write_text(ERA)
    assert _run(project, "record") == 0
    (project / "legacy" / "mod.py").write_text("VALUE = 1\n")
    capsys.readouterr()
    assert _run(project, "check") == 0
    out = capsys.readouterr().out
    assert "dropped: ruff legacy/mod.py ERA001: 1 -> 0" in out
    assert (
        "python tools/lint_ratchet.py record --baseline baselines/lint_ratchet.json"
        in out
    )
    assert _run(project, "check", "--fail-on-drop") == 1


def test_record_refuses_a_rise_unless_allowed(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    before = _baseline(project).read_text()
    (project / "legacy" / "mod.py").write_text(PROBE + ERA)
    assert _run(project, "record") == 1
    assert "refusing to record a rise" in capsys.readouterr().out
    assert _baseline(project).read_text() == before
    assert _run(project, "record", "--allow-rise") == 0
    assert _baseline(project).read_text() != before


def test_record_refuses_findings_in_a_new_package(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "tests_torch" / "test_ok.py").write_text("import os\n")
    assert _run(project, "record") == 1
    assert "cannot be baselined" in capsys.readouterr().out
    assert not _baseline(project).exists()


def test_check_without_a_baseline_is_an_error_not_a_pass(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "check") == 2
    assert "no baseline" in capsys.readouterr().err


def test_a_tool_that_stops_early_is_an_error_not_zero_findings(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    (project / "pyproject.toml").write_text(
        _pyproject(_installed("ruff"), _installed("ty")) + "\n[tool.ty.zz]\nx = 1\n"
    )
    capsys.readouterr()
    assert _run(project, "check") == 2
    assert "ty exited with 2" in capsys.readouterr().err


def test_a_version_mismatch_fails_the_check(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "record") == 0
    (project / "pyproject.toml").write_text(_pyproject("0.0.1", _installed("ty")))
    capsys.readouterr()
    assert _run(project, "check") == 1
    assert "ruff: installed" in capsys.readouterr().out
    assert _run(project, "versions") == 1


def test_versions_command_passes_when_everything_agrees(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(project, "versions") == 0
    assert "versions: ok" in capsys.readouterr().out


# --- TOOL3: more families, strict by default, moved files -------------------------------------

EXCEPT = "def g():\n    try:\n        return 1\n    except Exception:\n        return 2\n"
COMPLEX = (
    "def g(x):\n"
    + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(11))
    + "    return -1\n"
)
OS_PATH = 'import os\n\nPATH = os.path.join("a", "b")\n'
STRICT_OK = '"""Module."""\n\n\ndef f(a: int) -> int:\n    """Return a."""\n    return a\n'
TY_BAD = "def f(a: int = None) -> int:\n    return a.missing\n"


def _legacy(project: Path) -> Path:
    return project / "baselines" / "legacy_files.txt"


def _listed(project: Path) -> list[str]:
    return lr.read_legacy(_legacy(project))


def _recorded(project: Path) -> Path:
    """Record the baseline of ``project`` (``legacy/mod.py`` is its only legacy file)."""
    assert _run(project, "record") == 0
    return project


def _check(project: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    capsys.readouterr()
    status = _run(project, "check", *args)
    return status, capsys.readouterr().out


def test_the_first_record_lists_the_files_outside_the_new_packages(project: Path) -> None:
    _recorded(project)
    assert _listed(project) == ["legacy/mod.py"]
    digest = json.loads(_baseline(project).read_text())["legacy"]
    assert digest == lr.legacy_digest(["legacy/mod.py"])
    assert digest["count"] == 1


def test_lintable_files_are_python_and_notebooks_in_sorted_relative_form(project: Path) -> None:
    (project / "legacy" / "nb.ipynb").write_text("{}")
    (project / "legacy" / "notes.md").write_text("x")
    assert lr.lintable_files(project) == [
        "legacy/mod.py",
        "legacy/nb.ipynb",
        "tests_torch/test_ok.py",
    ]


def test_the_strict_ignores_are_those_of_pyproject_without_the_legacy_entry(project: Path) -> None:
    table = lr._strict_per_file_ignores(project)
    assert table == '{"{tests_torch,tests,tools/tests}/**" = ["D", "S101"]}'
    (project / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert lr._strict_per_file_ignores(project) == "{}"


def test_the_strict_ty_configuration_drops_the_overrides_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ty.environment]\npython = "./.venv"\n\n[tool.ty.rules]\nx = "ignore"\n\n'
        '[[tool.ty.overrides]]\ninclude = ["a/**"]\n\n[tool.ty.overrides.rules]\ny = "ignore"\n'
    )
    text = lr._strict_ty_config(tmp_path)
    assert f'python = "{(tmp_path / ".venv").resolve()}"' in text
    assert '[rules]\nx = "ignore"' in text
    assert "overrides" not in text
    assert "y = " not in text
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert lr._strict_ty_config(tmp_path) == "\n"


def test_one_more_broad_except_in_a_legacy_file_fails(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "legacy" / "mod.py").write_text(PROBE + EXCEPT)
    status, out = _check(project, capsys)
    assert status == 1
    assert "rose: ruff legacy/mod.py BLE001: 0 -> 1" in out


def test_a_function_over_complexity_ten_in_a_legacy_file_fails(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "legacy" / "mod.py").write_text(PROBE + COMPLEX)
    status, out = _check(project, capsys)
    assert status == 1
    assert "rose: ruff legacy/mod.py C901: 0 -> 1" in out


def test_a_new_file_in_a_legacy_directory_is_strict_for_ruff(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "data").mkdir()
    (project / "data" / "new.py").write_text(OS_PATH)
    status, out = _check(project, capsys)
    assert status == 1
    assert "strict: data/new.py: PTH118" in out
    assert "baselines/legacy_files.txt" in out


def test_a_new_file_is_checked_by_ty_without_the_legacy_overrides(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "legacy" / "new.py").write_text(TY_BAD)  # in the overridden directory
    status, out = _check(project, capsys)
    assert status == 1
    assert "strict: legacy/new.py: invalid-parameter-default" in out
    assert "strict: legacy/new.py: unresolved-attribute" in out


def test_a_new_file_that_is_clean_passes(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "legacy" / "clean.py").write_text(STRICT_OK)
    status, out = _check(project, capsys)
    assert (status, out) == (0, "lint ratchet: ok\n")


def test_the_docstring_and_assert_rules_are_off_for_a_new_test_file(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "tests").mkdir()
    (project / "tests" / "test_new.py").write_text(
        "def test_it() -> None:\n    assert 1 + 1 == 2\n"
    )
    status, out = _check(project, capsys)
    assert (status, out) == (0, "lint ratchet: ok\n")


def test_a_file_added_to_the_legacy_list_by_hand_fails_the_check(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    (project / "data").mkdir()
    (project / "data" / "new.py").write_text(OS_PATH)
    with _legacy(project).open("a", encoding="utf-8") as handle:
        handle.write("data/new.py\n")
    status, out = _check(project, capsys)
    assert status == 1
    assert "legacy_files.txt has 2 entries against 1 recorded" in out
    assert "only `record` may change it" in out


def test_the_failure_names_the_entry_that_is_not_in_the_committed_list(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run([*git, *args], cwd=project, check=True, capture_output=True)  # noqa: S603
    (project / "legacy" / "new.py").write_text(STRICT_OK)
    with _legacy(project).open("a", encoding="utf-8") as handle:
        handle.write("legacy/new.py\n")
    status, out = _check(project, capsys)
    assert status == 1
    assert "(not in the committed list: legacy/new.py)" in out


def test_record_never_adds_a_file_to_the_legacy_list(project: Path) -> None:
    _recorded(project)
    (project / "legacy" / "clean.py").write_text(STRICT_OK)
    assert _run(project, "record") == 0
    assert _listed(project) == ["legacy/mod.py"]


def test_record_refuses_a_new_file_with_a_finding(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    before = _legacy(project).read_text()
    (project / "legacy" / "new.py").write_text(OS_PATH)
    capsys.readouterr()
    assert _run(project, "record") == 1
    assert "legacy/new.py: PTH118" in capsys.readouterr().out
    assert _legacy(project).read_text() == before


def test_record_drops_a_deleted_file_from_the_list(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "legacy" / "gone.py").write_text(ERA)
    _recorded(project)
    assert _listed(project) == ["legacy/gone.py", "legacy/mod.py"]
    (project / "legacy" / "gone.py").unlink()
    status, out = _check(project, capsys)
    assert status == 1
    assert "legacy list: legacy/gone.py does not exist" in out
    assert _run(project, "record") == 0
    assert _listed(project) == ["legacy/mod.py"]
    assert _check(project, capsys)[0] == 0


def test_a_listed_file_in_a_new_package_or_listed_twice_is_reported(
    project: Path,
) -> None:
    _recorded(project)
    problems = lr.legacy_problems(
        project,
        ["legacy/mod.py", "legacy/mod.py", "tests_torch/test_ok.py"],
        ["legacy/mod.py", "legacy/mod.py", "tests_torch/test_ok.py"],
        lr.legacy_digest(["legacy/mod.py"]),
        _legacy(project),
    )
    assert "legacy list: legacy/mod.py is listed twice" in problems
    assert "legacy list: tests_torch/test_ok.py is in a new package" in problems


def test_a_baseline_without_a_legacy_digest_asks_for_a_record(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    data = json.loads(_baseline(project).read_text())
    del data["legacy"]
    _baseline(project).write_text(json.dumps(data))
    status, out = _check(project, capsys)
    assert status == 1
    assert "the baseline has no legacy digest" in out


def test_check_without_a_legacy_list_is_an_error_not_a_pass(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    _legacy(project).unlink()
    capsys.readouterr()
    assert _run(project, "check") == 2
    assert "no legacy list" in capsys.readouterr().err


def _move(project: Path) -> None:
    (project / "legacy" / "mod.py").rename(project / "legacy" / "moved.py")


def test_a_moved_file_with_unchanged_counts_passes_with_moved(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    _move(project)
    assert _check(project, capsys)[0] == 1  # without --moved: a missing file and a new one
    status, out = _check(project, capsys, "--moved", "legacy/mod.py", "legacy/moved.py")
    assert (status, out) == (0, "lint ratchet: ok\n")


def test_a_rise_in_a_moved_file_is_still_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    _move(project)
    (project / "legacy" / "moved.py").write_text(PROBE + ERA)
    status, out = _check(project, capsys, "--moved", "legacy/mod.py", "legacy/moved.py")
    assert status == 1
    assert "rose: ruff legacy/moved.py ERA001: 1 -> 2" in out


def test_record_with_moved_rewrites_the_baseline_and_the_list(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    _move(project)
    assert _run(project, "record", "--moved", "legacy/mod.py", "legacy/moved.py") == 0
    assert _listed(project) == ["legacy/moved.py"]
    assert set(json.loads(_baseline(project).read_text())["ruff"]) == {"legacy/moved.py"}
    assert _check(project, capsys)[0] == 0


def test_record_with_moved_refuses_a_rise_of_the_moved_file(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(project)
    _move(project)
    (project / "legacy" / "moved.py").write_text(PROBE + ERA)
    capsys.readouterr()
    assert _run(project, "record", "--moved", "legacy/mod.py", "legacy/moved.py") == 1
    assert "legacy/moved.py ERA001: 1 -> 2" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["check", "record"])
def test_a_move_onto_itself_is_an_error(
    project: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    _recorded(project)
    capsys.readouterr()
    assert _run(project, command, "--moved", "legacy/mod.py", "legacy/mod.py") == 2
    assert "the same" in capsys.readouterr().err


def test_the_committed_legacy_list_is_the_recorded_one_and_names_tracked_files() -> None:
    entries = lr.read_legacy(REPO / lr.LEGACY_LIST)
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],  # noqa: S607 - git on PATH, the repository is a git checkout
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split()
    )  # fmt: skip
    recorded = json.loads((REPO / lr.BASELINE).read_text())["legacy"]
    assert lr.legacy_digest(entries) == recorded
    assert entries == sorted(entries)
    assert set(entries) <= tracked
    assert not [e for e in entries if lr.in_new_package(e)]
    assert len(entries) <= LEGACY_CEILING


# --- consistency of the committed configuration ------------------------------------------------


def _tomllib() -> ModuleType:
    return pytest.importorskip(
        "tomllib"
    )  # Python 3.11+; the supported floor of the tools is 3.10


def _ruff() -> dict:
    return _tomllib().loads(PYPROJECT.read_text())["tool"]["ruff"]


STRICT_FAMILIES = (
    "I", "UP", "B", "SIM", "C4", "RUF", "PD", "NPY", "PIE", "PLE", "PLW", "PERF", "RET", "PTH",
    "T20", "ERA", "W", "C90", "PLR0911", "PLR0912", "PLR0913", "PLR0915", "D", "BLE", "TRY", "S",
    "ARG", "A", "TD", "FIX",
)  # fmt: skip
NEW_GLOB = "{" + ",".join(lr.NEW_PACKAGES) + "}/**"
TEST_GLOB = "{tests_torch,tests,tools/tests}/**"  # new tests: no docstrings, plain asserts


def test_ruff_is_configured_once_in_pyproject() -> None:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*ruff.toml"],  # noqa: S607 - git on PATH, the repository is a git checkout
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert [name for name in listed if (REPO / name).exists()] == []


def test_strict_families_and_engineering_limits_are_configured() -> None:
    lint = _ruff()["lint"]
    assert set(STRICT_FAMILIES) | {"E", "F"} == set(lint["select"])
    assert lint["mccabe"]["max-complexity"] == 10
    assert lint["pylint"] == {"max-args": 6, "max-branches": 12, "max-statements": 50}
    assert lint["pydocstyle"]["convention"] == "numpy"
    assert {"E741", "N806", "N803", "PLR2004"} <= set(
        lint["ignore"]
    )  # physics names, constants


def test_legacy_ignores_every_strict_family_except_the_commented_out_code() -> None:
    ignores = _ruff()["lint"]["per-file-ignores"]
    assert set(ignores[f"!{NEW_GLOB}"]) == set(STRICT_FAMILIES) - {"ERA"}
    assert set(ignores[TEST_GLOB]) == {"D", "S101"}
    assert set(ignores) == {f"!{NEW_GLOB}", TEST_GLOB}


def test_new_packages_have_python_312_as_target() -> None:
    assert _ruff()["per-file-target-version"] == {NEW_GLOB: "py312"}


def test_pyproject_ty_overrides_cover_legacy_and_exclude_the_new_packages() -> None:
    cfg = _tomllib().loads(PYPROJECT.read_text())["tool"]["ty"]["overrides"][0]
    assert set(cfg["rules"].values()) == {"ignore"}
    assert set(cfg["rules"]) == {
        "invalid-parameter-default", "unresolved-attribute", "invalid-argument-type", "unsupported-operator",
    }  # fmt: skip
    for package in lr.NEW_PACKAGES:
        if package.startswith("tensorpotential/"):
            assert f"{package}/**" in cfg["exclude"]
