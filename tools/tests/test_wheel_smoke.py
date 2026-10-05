"""Tests for tools/wheel_smoke.py.

``check_script`` is exercised on small executable scripts written into ``tmp_path`` that behave like the
console scripts in the three situations of the tool (usage text, install hint, missing flask); the
functions that ask a real environment (``installed_packages``, ``console_scripts``, ``check_core``) run on
the interpreter of the test environment, which has ``tensorpotential`` installed.
"""

import stat
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wheel_smoke as ws

HINT = "pip install 'tensorpotential[tf]'"


def fake_env(tmp_path: Path, scripts: dict[str, str]) -> Path:
    """A ``bin`` directory with one shell script per name; returns the path of its ``python``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir / "python"


USAGE = "echo 'usage: {name} [-h]'\n"
HINTED = f'echo "ImportError: needs TensorFlow. {HINT}" >&2\nexit 1\n'
BARE = 'echo "ModuleNotFoundError: No module named tensorflow" >&2\nexit 1\n'
NO_FLASK = 'echo "ModuleNotFoundError: No module named flask" >&2\nexit 1\n'


@pytest.mark.parametrize(
    ("name", "body", "have", "problem"),
    [
        ("grace_models", USAGE, set(), False),
        ("grace_models", "exit 1\n", set(), True),  # usage expected, exit 1
        ("grace_models", "echo done\n", set(), True),  # exit 0 but no usage text
        ("grace_uq", "echo 'grace_uq <subcommand>'\n", set(), False),  # own help text
        ("grace_utils", HINTED, set(), False),  # no TF: the hint is expected
        (
            "grace_utils",
            BARE,
            set(),
            True,
        ),  # no TF: a bare ModuleNotFoundError is the old failure
        ("grace_utils", USAGE, set(), True),  # no TF but it ran: the guard is missing
        ("grace_utils", USAGE, {"tensorflow"}, False),  # TF installed: usage expected
        (
            "grace_utils",
            HINTED,
            {"tensorflow"},
            True,
        ),  # TF installed: the hint is a failure
        (
            "grace_dashboard",
            NO_FLASK,
            set(),
            False,
        ),  # flask missing: named, as in CLEAN3
        ("grace_dashboard", USAGE, set(), True),
        ("grace_dashboard", USAGE, {"flask"}, False),  # flask installed: usage expected
    ],
)
def test_check_script_classifies_each_behaviour(
    tmp_path: Path, name: str, body: str, have: set[str], problem: bool
) -> None:
    python = fake_env(tmp_path, {name: body.replace("{name}", name)})
    result = ws.check_script(python, name, have)
    assert (result is not None) is problem, result


def test_the_scripts_that_need_tensorflow_are_the_five_found_by_running_them() -> None:
    assert ws.NEED_TF == {
        "gracemaker",
        "extxyz2df",
        "grace_preprocess",
        "grace_predict",
        "grace_utils",
    }


def test_the_real_environment_registers_ten_console_scripts() -> None:
    scripts = ws.console_scripts(Path(sys.executable))
    assert len(scripts) == 10
    assert {"gracemaker", "grace_uq", "grace_dashboard"} <= set(scripts)


def test_importing_the_core_in_the_real_environment_loads_no_tensorflow() -> None:
    assert ws.check_core(Path(sys.executable)) == []


def test_installed_packages_reports_what_the_environment_can_import() -> None:
    have = ws.installed_packages(Path(sys.executable))
    assert have <= {"tensorflow", "tf_keras", "torch", "flask"}
    assert "tensorflow" in have  # the test environment is the TensorFlow one


def test_main_fails_when_the_environment_is_not_the_expected_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = ws.main([sys.executable, "--expect", "no-tf"])
    assert status == 1
    assert "--expect no-tf, but the environment has" in capsys.readouterr().err


SCRIPT_NAMES = [
    "df2extxyz",
    "extxyz2df",
    "grace_collect",
    "grace_dashboard",
    "grace_models",
    "grace_predict",
    "grace_preprocess",
    "grace_uq",
    "grace_utils",
    "gracemaker",
]


def scripted_env(tmp_path: Path, *, packages: str, core: str, names: list[str]) -> Path:
    """A fake venv whose ``python`` answers the three probes of the tool with canned text.

    The probe is recognised by what its code mentions: ``find_spec`` (installed packages),
    ``entry_points`` (console scripts), otherwise the import of the core.
    """
    python_body = f"""
        case "$2" in
          *find_spec*) echo '{packages}' ;;
          *entry_points*) echo '{" ".join(names)}' ;;
          *) {core} ;;
        esac
    """
    scripts = {name: USAGE.replace("{name}", name) for name in names}
    return fake_env(tmp_path, {"python": python_body, **scripts})


def test_main_passes_on_a_environment_where_everything_behaves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python = scripted_env(
        tmp_path,
        packages="tensorflow,tf_keras,flask",
        core="echo 'LOADED '",
        names=SCRIPT_NAMES,
    )
    assert ws.main([str(python), "--expect", "tf"]) == 0
    out = capsys.readouterr().out
    assert "ok   gracemaker --help" in out
    assert "FAIL" not in out


def test_main_reports_a_core_that_does_not_import_and_a_wrong_script_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python = scripted_env(
        tmp_path,
        packages="tensorflow,tf_keras",
        core="echo 'ImportError: boom' >&2; exit 1",
        names=SCRIPT_NAMES[:9],
    )
    assert ws.main([str(python)]) == 1
    err = capsys.readouterr().err
    assert "the TensorFlow-free core does not import" in err
    assert "expected 10 console scripts, found 9" in err


def test_main_reports_a_core_that_loads_tensorflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python = scripted_env(
        tmp_path,
        packages="tensorflow,tf_keras",
        core="echo 'LOADED tensorflow'",
        names=SCRIPT_NAMES,
    )
    assert ws.main([str(python)]) == 1
    assert "importing the core loaded tensorflow" in capsys.readouterr().err


def test_main_accepts_a_relative_path_to_the_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    python = scripted_env(
        tmp_path,
        packages="tensorflow,tf_keras,flask",
        core="echo 'LOADED '",
        names=SCRIPT_NAMES,
    )
    monkeypatch.chdir(tmp_path)
    assert ws.main([str(python.relative_to(tmp_path)), "--expect", "tf"]) == 0
    assert "FAIL" not in capsys.readouterr().out
