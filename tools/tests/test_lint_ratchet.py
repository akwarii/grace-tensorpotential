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

# Violates the strict set (I001, T201, ERA001, UP006, D103, ...) and ty (invalid-parameter-default,
# unresolved-attribute). In legacy only the commented-out code is counted (ERA001).
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
    legacy, strict = lr.tally(findings, lr.NEW_PACKAGES)
    assert legacy == {"a.py": {"E722": 1, "F401": 2}}
    assert strict == ["tests_torch/t.py: I001"]


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
    now = lr.measure(project)
    assert now["ruff"] == {
        "legacy/mod.py": {"ERA001": 1}
    }  # legacy: E, F and the ratchet rules only
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
    now = lr.measure(project)
    assert now["ruff"] == {"legacy/mod.py": {"ERA001": 1}}
    assert now["ty"] == {"legacy/mod.py": {"unsupported-operator": 1}}
    assert now["strict"] == []


def test_unresolved_import_is_not_counted(project: Path) -> None:
    (project / "legacy" / "mod.py").write_text(
        "import not_installed_zz\n\nVALUE = not_installed_zz\n"
    )
    assert lr.measure(project)["ty"] == {}


def test_a_syntax_error_counts_as_a_finding_not_as_zero(project: Path) -> None:
    (project / "legacy" / "mod.py").write_text("x = (\n")
    assert lr.measure(project)["ruff"] == {"legacy/mod.py": {"invalid-syntax": 1}}


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
    assert set(ignores["tests_torch/**"]) == {"D", "S101"}
    assert set(ignores) == {f"!{NEW_GLOB}", "tests_torch/**"}


def test_new_packages_have_python_311_as_target() -> None:
    assert _ruff()["per-file-target-version"] == {NEW_GLOB: "py311"}


def test_pyproject_ty_overrides_cover_legacy_and_exclude_the_new_packages() -> None:
    cfg = _tomllib().loads(PYPROJECT.read_text())["tool"]["ty"]["overrides"][0]
    assert set(cfg["rules"].values()) == {"ignore"}
    assert set(cfg["rules"]) == {
        "invalid-parameter-default", "unresolved-attribute", "invalid-argument-type", "unsupported-operator",
    }  # fmt: skip
    for package in lr.NEW_PACKAGES:
        if package.startswith("tensorpotential/"):
            assert f"{package}/**" in cfg["exclude"]
