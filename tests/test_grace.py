"""``tensorpotential.scripts.grace``: the ``grace`` command, its table of sub-commands and the dispatch.

Logic layer: the table (names, functions, backends) against the console scripts of ``pyproject.toml``; the dispatch with a table of
stand-in commands (route, remaining arguments, exit code, unknown sub-command, missing backend); the help of every real
command through ``grace`` against the committed reference of the old name with the program name replaced; the same output
through the old name and through ``grace`` for commands that can run on small inputs; a fresh interpreter that refuses
TensorFlow runs ``grace``, ``grace -h``, an unknown sub-command and the help of the commands that need no TensorFlow; and
``grace train`` hands over to the multi-GPU setup exactly like ``gracemaker``.

There are no physical values here: the module only routes arguments.
"""

from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from tensorpotential.core.backends import BackendNotInstalledError
from tensorpotential.scripts import df2extxyz, grace, grace_models, grace_uq
from tests import cli_help
from tests.cli_help import CASES, normalise
from tests.fresh_python import run_fresh_python

from .test_uq_cli import _write_synthetic_info_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The old console script of each sub-command of ``grace``.
OLD_NAME = {
    "train": "gracemaker",
    "preprocess": "grace_preprocess",
    "predict": "grace_predict",
    "utils": "grace_utils",
    "models": "grace_models",
    "uq": "grace_uq",
    "collect": "grace_collect",
    "extxyz2df": "extxyz2df",
    "df2extxyz": "df2extxyz",
    "dashboard": "grace_dashboard",
}
NEW_NAME = {old: new for new, old in OLD_NAME.items()}

#: The table of ``grace_uq`` before it moved here: the oracle for the sub-commands of ``grace uq``.
OLD_UQ_TABLE = {
    "build": ("tensorpotential.uq.cli.build", "build_main"),
    "info": ("tensorpotential.uq.cli.info", "info_main"),
    "predict": ("tensorpotential.uq.cli.predict", "predict_main"),
    "select": ("tensorpotential.uq.cli.select", "select_main"),
}


def project_scripts() -> dict[str, str]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "scripts"
    ]


# --- the table ---------------------------------------------------------------------------------------------------------


def test_every_row_runs_the_function_of_its_old_console_script():
    scripts = project_scripts()
    assert {c.name for c in grace.COMMANDS} == set(OLD_NAME)
    for command in grace.COMMANDS:
        assert (
            scripts[OLD_NAME[command.name]] == f"{command.module}:{command.function}"
        ), command.name


def test_the_old_console_scripts_are_still_installed_next_to_grace():
    scripts = project_scripts()
    assert scripts["grace"] == "tensorpotential.scripts.grace:main"
    assert set(OLD_NAME.values()) <= set(scripts)
    assert len(scripts) == len(OLD_NAME) + 1


def test_the_uq_rows_are_the_old_table_of_grace_uq():
    assert {c.name: (c.module, c.function) for c in grace.UQ_COMMANDS} == OLD_UQ_TABLE


def test_names_are_unique_and_every_function_takes_argv_and_a_prog():
    for table in (grace.COMMANDS, grace.UQ_COMMANDS):
        assert len({c.name for c in table}) == len(table)
        for command in table:
            parameters = inspect.signature(grace.resolve(command)).parameters
            assert list(parameters)[0] == "argv", command.name
            assert parameters["prog"].kind is inspect.Parameter.KEYWORD_ONLY, (
                command.name
            )
            assert command.help and "\n" not in command.help


def test_the_commands_that_need_tensorflow_are_the_five_of_the_install_check():
    needing = {OLD_NAME[c.name] for c in grace.COMMANDS if c.backend == "tf"}
    assert needing == {
        "gracemaker",
        "extxyz2df",
        "grace_preprocess",
        "grace_predict",
        "grace_utils",
    }
    assert all(c.backend == "tf" for c in grace.UQ_COMMANDS)
    assert {c.backend for c in grace.COMMANDS} <= {None, "tf"}


# --- the dispatch, with stand-in commands -----------------------------------------------------------------------------------

CALLS: list[tuple[list[str], str | None]] = []
RESULT: list[object] = [0]


def stand_in(argv=None, *, prog=None):
    """The function behind the stand-in commands: records what it is given and returns what the test says."""
    CALLS.append((list(argv), prog))
    return RESULT[0]


def other_stand_in(argv=None, *, prog=None):
    CALLS.append((["other", *(argv or [])], prog))
    return 0


@pytest.fixture
def table(monkeypatch):
    """A table of two stand-ins (one needing a backend) in place of the real one."""
    CALLS.clear()
    RESULT[0] = 0
    rows = (
        grace.Command("one", __name__, "stand_in", "The first command"),
        grace.Command("two", __name__, "other_stand_in", "The second command", "tf"),
    )
    monkeypatch.setattr(grace, "COMMANDS", rows)
    return rows


def run_grace(*argv: str) -> tuple[int, str, str]:
    return cli_help.run_main(grace.main, "grace", list(argv))


def test_the_sub_command_gets_the_remaining_arguments_and_its_prog_name(table):
    code, out, err = run_grace("one", "a.pkl.gz", "--flag", "x", "-v")
    assert (code, out, err) == (0, "", "")
    assert CALLS == [(["a.pkl.gz", "--flag", "x", "-v"], "grace one")]


def test_no_argument_is_dropped_between_grace_and_the_function(table):
    argv = ["--", "-h", "--help", "one", "", "a b"]
    run_grace("one", *argv)
    assert CALLS[0][0] == argv


def test_each_row_routes_to_its_own_function(table, monkeypatch):
    monkeypatch.setattr(grace, "require_backend", lambda *a, **k: None)
    run_grace("two", "x")
    run_grace("one", "y")
    assert CALLS == [(["other", "x"], "grace two"), (["y"], "grace one")]


@pytest.mark.parametrize("result", [0, 1, 2, 7])
def test_the_exit_code_of_the_function_is_the_exit_code_of_grace(table, result):
    RESULT[0] = result
    assert run_grace("one")[0] == result


def test_a_function_that_returns_nothing_is_a_success(table):
    RESULT[0] = None
    assert run_grace("one")[0] == 0


def test_an_unknown_sub_command_is_reported_with_the_known_ones_and_exits_2(table):
    code, out, err = run_grace("three", "x")
    assert (code, out) == (2, "")
    assert err == "grace: unknown subcommand 'three'. Use one of: one, two\n"
    assert CALLS == []


@pytest.mark.parametrize("argv", [(), ("-h",), ("--help",)])
def test_without_a_sub_command_or_with_help_the_table_is_listed_and_the_exit_code_is_0(
    table, argv
):
    code, out, err = run_grace(*argv)
    assert (code, err) == (0, "")
    assert out.startswith("usage: grace <subcommand> [options]\n")
    assert "  one  The first command\n" in out
    assert "  two  The second command\n" in out
    assert "Run `grace <subcommand> --help`" in out
    assert CALLS == []


def test_an_option_before_the_sub_command_is_an_unknown_sub_command(table):
    code, _, err = run_grace("--verbose", "one")
    assert code == 2
    assert "unknown subcommand '--verbose'" in err


def test_the_sub_command_is_not_imported_for_a_help_request(table, monkeypatch):
    monkeypatch.setattr(
        grace, "resolve", lambda command: pytest.fail("imported for the help")
    )
    assert run_grace("--help")[0] == 0


def test_a_missing_backend_is_reported_before_the_command_is_imported(
    table, monkeypatch
):
    def refuse(backend, needed_by=None):
        raise BackendNotInstalledError(
            backend,
            ("tensorflow",),
            f"{needed_by} needs TensorFlow: pip install 'tensorpotential[tf]'",
        )

    monkeypatch.setattr(grace, "require_backend", refuse)
    monkeypatch.setattr(
        grace, "resolve", lambda command: pytest.fail("imported without the backend")
    )
    code, out, err = run_grace("two")
    assert code == 1
    assert out == ""
    assert err == "grace two needs TensorFlow: pip install 'tensorpotential[tf]'\n"
    assert CALLS == []


def test_a_command_without_a_backend_does_not_ask_for_one(table, monkeypatch):
    monkeypatch.setattr(
        grace, "require_backend", lambda *a, **k: pytest.fail("a backend was required")
    )
    assert run_grace("one")[0] == 0


def test_the_real_table_lists_every_sub_command_in_the_top_level_help():
    code, out, _ = run_grace("--help")
    assert code == 0
    for command in grace.COMMANDS:
        assert f"  {command.name}" in out
        assert command.help in out


# --- the help of every real command ---------------------------------------------------------------------------------------


def expected_through_grace(case) -> tuple[list[str], str]:
    """The arguments and the reference text for one case of ``tests/cli_help.py`` when run through ``grace``."""
    reference = case.reference.read_text(encoding="utf-8")
    new = NEW_NAME[case.command]
    if case.command == "grace_uq":
        argv = ["uq", *case.words, "--help"]
        if not case.words:
            return argv, reference.replace("grace_uq", "grace uq")
        return argv, reference.replace(
            f"usage: grace_uq {case.words[0]}", f"usage: grace uq {case.words[0]}"
        )
    return [new, *case.words, "--help"], reference.replace(
        f"usage: {case.command}", f"usage: grace {new}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_the_help_through_grace_is_the_old_help_with_the_program_name_replaced(case):
    argv, expected = expected_through_grace(case)
    code, out, err = run_grace(*argv)
    assert code == 0, err
    assert normalise(out) == normalise(expected)


def test_the_old_name_still_prints_its_own_name_in_the_usage_of_the_uq_sub_commands():
    code, out, _ = cli_help.run_main(grace_uq.main, "grace_uq", ["select", "--help"])
    assert code == 0
    assert out.startswith("usage: grace_uq select ")


# --- the same output through the old name and through grace -------------------------------------------------------------------


def same_through_both(old_main, old_argv, new_argv, replace=()):
    old = cli_help.run_main(old_main, OLD_NAME_OF_MAIN[old_main], old_argv)
    new = run_grace(*new_argv)
    texts = []
    for result in (old, new):
        code, out, err = result
        for before, after in replace:
            out, err = out.replace(before, after), err.replace(before, after)
        texts.append((code, out, err))
    return texts


OLD_NAME_OF_MAIN = {
    grace_models.main: "grace_models",
    grace_uq.main: "grace_uq",
    df2extxyz.main: "df2extxyz",
}


def test_models_list_prints_the_same_through_both_names():
    old, new = same_through_both(
        grace_models.main, ["list", "--ascii"], ["models", "list", "--ascii"]
    )
    assert old == new
    assert old[0] == 0


def test_models_download_of_an_unknown_model_prints_the_same_through_both_names():
    old, new = same_through_both(
        grace_models.main,
        ["download", "no_such_model"],
        ["models", "download", "no_such_model"],
    )
    assert old == new
    assert "not found in available models" in old[1]


def test_uq_info_prints_the_same_through_both_names(tmp_path):
    artifact = str(tmp_path / "art.npz")
    _write_synthetic_info_artifact(artifact, n_elements=2, K=3, D=4)
    old, new = same_through_both(
        grace_uq.main, ["info", artifact], ["uq", "info", artifact]
    )
    assert old == new
    assert old[0] == 0
    assert "feature_dim D" in old[1]


def test_a_usage_error_has_the_same_text_and_code_through_both_names_up_to_the_program_name(
    tmp_path,
):
    missing = str(tmp_path / "absent.npz")
    old, new = same_through_both(
        grace_uq.main,
        ["info", missing],
        ["uq", "info", missing],
        replace=[("grace uq info", "grace_uq info")],
    )
    assert old == new
    assert old[0] == 2
    assert "Artifact not found" in old[2]


def test_df2extxyz_writes_the_same_file_through_both_names(tmp_path):
    frame = pd.read_pickle(
        REPO_ROOT / "tests" / "data" / "MoNbTaW_train50.pkl.gz"
    ).head(2)
    frame = frame[["ase_atoms", "energy_corrected", "forces", "stress"]]
    outputs = []
    for label, argv in (("old", None), ("new", "grace")):
        source = tmp_path / f"{label}.pkl.gz"
        frame.to_pickle(source, compression="gzip")
        target = tmp_path / f"{label}.xyz"
        args = [str(source), "-o", str(target)]
        code = grace.main(["df2extxyz", *args]) if argv else df2extxyz.main(args)
        assert code == 0
        outputs.append(target.read_text())
    assert outputs[0] == outputs[1]
    assert outputs[0].count("Lattice=") == 2


def test_extxyz2df_stores_the_same_dataframe_through_both_names(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-14.0, forces=np.ones((4, 3)), stress=np.arange(6.0)
    )
    source = tmp_path / "frames.xyz"
    write(source, [atoms], format="extxyz")
    from tensorpotential.scripts import extxyz2df

    frames = []
    for label, runner in (
        ("old", extxyz2df.main),
        ("new", lambda argv: grace.main(["extxyz2df", *argv])),
    ):
        target = tmp_path / f"{label}.pkl.gz"
        assert runner([str(source), "--output-dataset-filename", str(target)]) == 0
        frames.append(pd.read_pickle(target, compression="gzip"))
    for column in ("energy", "forces", "stress"):
        np.testing.assert_array_equal(
            np.stack(frames[0][column].to_list()), np.stack(frames[1][column].to_list())
        )


# --- a fresh interpreter without TensorFlow --------------------------------------------------------------------------------

_BLOCKED = """
import contextlib, io, json, sys

class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("tensorflow", "keras", "tf_keras"):
            raise ImportError(f"No module named {name!r} (refused by the test)", name=name)

sys.meta_path.insert(0, Refuse())
from tensorpotential.scripts import grace

results = {}
for argv in json.load(sys.stdin):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = grace.main(argv)
        except SystemExit as stop:
            code = stop.code
    results[" ".join(argv)] = {"code": code, "out": out.getvalue(), "err": err.getvalue()}
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("tensorflow", "keras", "tf_keras"))
print(json.dumps({"results": results, "loaded": loaded}))
"""

NO_TF_ARGVS = [
    [],
    ["-h"],
    ["--help"],
    ["frobnicate"],
    ["uq"],
    ["uq", "-h"],
    ["uq", "frobnicate"],
    ["collect", "--help"],
    ["models", "--help"],
    ["df2extxyz", "--help"],
]
NEEDS_TF_ARGVS = [
    ["train", "--help"],
    ["utils", "-p", "x", "summary"],
    ["uq", "info", "x.npz"],
    ["extxyz2df", "--help"],
]


@pytest.fixture(scope="module")
def without_tensorflow(tmp_path_factory):
    result = run_fresh_python(
        _BLOCKED,
        tmp_path_factory.mktemp("grace_no_tf"),
        stdin=json.dumps(NO_TF_ARGVS + NEEDS_TF_ARGVS),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_grace_and_its_help_and_an_unknown_sub_command_run_without_tensorflow(
    without_tensorflow,
):
    results = without_tensorflow["results"]
    assert [results[key]["code"] for key in ("", "-h", "--help")] == [0, 0, 0]
    assert results[""]["out"].startswith("usage: grace <subcommand> [options]")
    assert results["frobnicate"]["code"] == 2
    assert results["frobnicate"]["err"].startswith(
        "grace: unknown subcommand 'frobnicate'. Use one of: collect, dashboard,"
    )


def test_the_uq_group_and_the_commands_without_a_backend_run_without_tensorflow(
    without_tensorflow,
):
    results = without_tensorflow["results"]
    assert (
        results["uq"]["code"] == 0
        and "grace uq <subcommand> [options]" in results["uq"]["out"]
    )
    assert results["uq -h"]["out"] == results["uq"]["out"]
    assert results["uq frobnicate"]["code"] == 2
    assert (
        results["uq frobnicate"]["err"]
        == "grace uq: unknown subcommand 'frobnicate'. Use one of: build, info, predict, select\n"
    )
    for name in ("collect --help", "models --help", "df2extxyz --help"):
        assert results[name]["code"] == 0, name
        assert results[name]["out"].startswith("usage: grace "), name


def test_a_command_that_needs_tensorflow_stops_with_the_install_hint_and_exit_1(
    without_tensorflow,
):
    for argv in NEEDS_TF_ARGVS:
        result = without_tensorflow["results"][" ".join(argv)]
        assert result["code"] == 1, argv
        assert "pip install 'tensorpotential[tf]'" in result["err"], argv
        assert result["out"] == ""
        assert "needs TensorFlow" in result["err"], argv


def test_nothing_of_the_above_imported_tensorflow(without_tensorflow):
    assert without_tensorflow["loaded"] == []


# --- grace train = gracemaker, multi-GPU included ---------------------------------------------------------------------------


@pytest.fixture
def train_env(monkeypatch, tmp_path):
    """The replacements of the gracemaker script tests: the library run and the MWMS strategy are recorded, not run."""
    from tensorpotential.cli import distribute, gracemaker

    for name in ("TF_CONFIG", "NUM_VIRTUAL_DEVICES"):
        monkeypatch.setenv(name, "unset-by-test")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    record = {"runs": [], "mwms": 0}
    stand_in_strategy = object()

    def fake_main(argv=None, strategy=None, strategy_desc="", prog=None):
        record["runs"].append({
            "argv": list(argv),
            "strategy": strategy,
            "desc": strategy_desc,
            "prog": prog,
        })

    def fake_mwms():
        record["mwms"] += 1
        return stand_in_strategy

    monkeypatch.setattr(gracemaker, "main", fake_main)
    monkeypatch.setattr(distribute, "make_mwms_strategy", fake_mwms)
    record["strategy"] = stand_in_strategy
    return record


def test_train_hands_its_arguments_and_prog_to_the_library(train_env):
    assert run_grace("train", "input.yaml", "--seed", "3")[0] == 0
    (run,) = train_env["runs"]
    assert run == {
        "argv": ["input.yaml", "--seed", "3"],
        "strategy": None,
        "desc": "",
        "prog": "grace train",
    }


@pytest.mark.parametrize("flag", ["-m", "--multigpu"])
def test_the_multigpu_flag_is_seen_through_grace_train_like_through_gracemaker(
    train_env, flag
):
    run_grace("train", flag, "input.yaml")
    (run,) = train_env["runs"]
    assert run["strategy"] is train_env["strategy"]
    assert run["desc"] == "Single host/multi GPU (MWMS auto-TF_CONFIG)"
    assert run["argv"] == [flag, "input.yaml"]


def test_a_mirrored_strategy_in_the_input_yaml_is_seen_through_grace_train(
    train_env, tmp_path
):
    (tmp_path / "input.yaml").write_text("fit:\n  strategy: mirrored\n")
    run_grace("train", "--seed", "1", "input.yaml")
    assert train_env["mwms"] == 1
    assert train_env["runs"][0]["argv"] == ["--seed", "1", "input.yaml"]


def test_the_top_level_help_is_a_styled_table_on_a_terminal(monkeypatch):
    import io
    import re

    from tensorpotential.cli import console

    buffer = io.StringIO()
    terminal = console.make_console(file=buffer, force_terminal=True, width=120)
    monkeypatch.setattr(console, "get_console", lambda *, stderr=False: terminal)
    assert grace.main(["--help"]) == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    assert plain.startswith("usage: grace <subcommand> [options]\n")
    for command in grace.COMMANDS:
        assert re.search(
            rf"^{command.name}\s+{re.escape(command.help)}\s*$", plain, re.MULTILINE
        )
    assert buffer.getvalue() != plain


def test_an_error_of_grace_is_red_on_a_terminal(monkeypatch):
    import io
    import re

    from tensorpotential.cli import console

    buffer = io.StringIO()
    terminal = console.make_console(file=buffer, force_terminal=True, width=120)
    monkeypatch.setattr(console, "get_console", lambda *, stderr=False: terminal)
    assert grace.main(["nonsense"]) == 2
    assert re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue()).startswith(
        "grace: unknown subcommand 'nonsense'"
    )
    assert "\x1b[1;31m" in buffer.getvalue()
