"""Tests for tools/resolve_extras.py.

Logic layer: name normalisation, the expansion of self-references (``all`` names the other extras), empty and
undeclared extras, the problems reported. Behaviour layer: ``uv pip compile`` on a throw-away project whose extras
are a real package (``six``), an empty list and a requirement whose marker never matches, and on the real
``pyproject.toml`` of this repository.
"""

import shutil
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import resolve_extras as re_

REPO = Path(__file__).resolve().parents[2]
needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")

PYPROJECT = {
    "project": {
        "name": "Demo_Pkg",
        "optional-dependencies": {
            "a": ["foo>=1", "Bar_Baz[x]; sys_platform == 'linux'"],
            "b": ["qux"],
            "empty": [],
            "selfish": ["demo-pkg"],
            "all": ["demo-pkg[a,b]"],
            "loop": ["demo_pkg[loop]"],
        },
    }
}


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("tf_keras", "tf-keras"),
        ("tensorflow[and-cuda]<=2.20; sys_platform == 'linux'", "tensorflow"),
        ("Torch.Sim~=1.0", "torch-sim"),
        ("  numpy ", "numpy"),
        ("pyyaml>=6.0.2", "pyyaml"),
    ],
)
def test_requirement_name_is_the_normalised_project_name(text: str, name: str) -> None:
    assert re_.requirement_name(text) == name


def test_an_extra_lists_the_names_it_asks_for() -> None:
    assert re_.extra_requirements(PYPROJECT, "a") == {"foo", "bar-baz"}


def test_a_self_reference_is_expanded_to_the_extras_it_names() -> None:
    assert re_.extra_requirements(PYPROJECT, "all") == {"foo", "bar-baz", "qux"}


def test_an_empty_extra_and_one_that_only_names_the_project_list_nothing() -> None:
    assert re_.extra_requirements(PYPROJECT, "empty") == set()
    assert re_.extra_requirements(PYPROJECT, "selfish") == set()


def test_a_self_reference_loop_terminates() -> None:
    assert re_.extra_requirements(PYPROJECT, "loop") == set()


def test_an_undeclared_extra_is_a_key_error_naming_the_declared_ones() -> None:
    with pytest.raises(KeyError, match=r"'nope' is not declared.*'all'"):
        re_.extra_requirements(PYPROJECT, "nope")


def test_resolution_problems() -> None:
    assert re_.resolution_problems("a", {"foo"}, {"foo", "x"}) == []
    assert re_.resolution_problems("a", set(), {"foo"}) == [
        "extra 'a' lists no requirement"
    ]
    assert re_.resolution_problems("a", {"foo", "bar"}, {"foo"}) == [
        "extra 'a': bar missing from the resolved set"
    ]


def write_project(tmp_path: Path, extras: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "0.1"\nrequires-python = ">=3.11"\n'
        f"[project.optional-dependencies]\n{extras}\n"
    )
    return path


@needs_uv
def test_a_real_extra_resolves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_project(tmp_path, 'ok = ["six"]')
    assert re_.main(["--pyproject", str(path)]) == 0
    assert "ok   ok: 1 requirement(s)" in capsys.readouterr().out


@needs_uv
def test_an_empty_extra_fails_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_project(tmp_path, 'ok = ["six"]\nempty = []')
    assert re_.main(["--pyproject", str(path)]) == 1
    assert "extra 'empty' lists no requirement" in capsys.readouterr().err


@needs_uv
def test_a_requirement_that_the_markers_exclude_fails_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_project(tmp_path, "odd = [\"six; python_version < '3'\"]")
    assert re_.main(["--pyproject", str(path)]) == 1
    assert "extra 'odd': six missing from the resolved set" in capsys.readouterr().err


@needs_uv
def test_an_unresolvable_extra_is_an_error(tmp_path: Path) -> None:
    path = write_project(tmp_path, 'bad = ["no-such-package-for-resolve-extras-tests"]')
    with pytest.raises(RuntimeError, match="could not resolve extra 'bad'"):
        re_.main(["--pyproject", str(path)])


def test_a_project_without_extras_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "demo"\nversion = "0.1"\n')
    assert re_.main(["--pyproject", str(path)]) == 1
    assert "declares no extras" in capsys.readouterr().err


def test_the_repository_declares_the_four_extras_with_their_packages() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert re_.extra_requirements(pyproject, "tf") == {"tensorflow", "tf-keras"}
    assert re_.extra_requirements(pyproject, "torch") == {"torch"}
    assert re_.extra_requirements(pyproject, "torch-sim") == {
        "torch",
        "torch-sim-atomistic",
        "vesin",
    }
    assert re_.extra_requirements(pyproject, "all") == {
        "tensorflow", "tf-keras", "torch", "torch-sim-atomistic", "vesin"
    }  # fmt: skip
