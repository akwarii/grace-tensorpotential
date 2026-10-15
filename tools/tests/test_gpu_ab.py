"""Tests for tools/gpu_ab.py and tools/slurm/gpu_ab.sh.

Logic layer: step names, the runs of each step and their order, the checkpoint files (a step started
again takes only the missing runs; a failed size stops the scan of that commit; an option that does not
apply ends the step), the saved config, the summary arithmetic, the array-task selection of the shell
wrapper. The measurements are replaced by a fake executor that writes result files.
Value layer: the extraction of a real commit with ``git archive`` and the check that ``tensorpotential``
is then imported from it, against the location the interpreter reports.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import gpu_ab as ga  # noqa: E402

ARCH = "GRACE-FS-OMAT"


def _config(**changes: Any) -> ga.Config:
    config = ga.Config(
        base="a" * 40,
        head="b" * 40,
        device="cpu",
        pairs=2,
        repeats=1,
        cell=1,
        archs=(ARCH,),
        dtypes=("float64",),
        options=("none", "dense_nbr"),
        scan_cells=(1, 2, 3),
    )
    return dataclasses.replace(config, **changes)


class FakeExecutor:
    """Writes a result file for every command; ``fail`` maps (tree, cell) to an exit status."""

    def __init__(self, fail=None, ms=None):
        self.calls = []
        self.fail = fail or {}
        self.ms = ms or {"base": 10.0, "head": 8.0}

    def __call__(self, cmd, env):
        side = "base" if env["PYTHONPATH"].endswith("base") else "head"
        cell = int(cmd[cmd.index("--supercell") + 1])
        self.calls.append((side, cell, tuple(cmd)))
        status = self.fail.get((side, cell), 0)
        if status == 0:
            profile = {
                "n_bonds": 100 * cell**3,
                "gpu_peak_memory_mb": 10.0 * cell**3,
                "train_step_ms" if "--train-step" in cmd else "whole_ms": self.ms[side],
            }
            Path(cmd[cmd.index("--json") + 1]).write_text(
                json.dumps({"profiles": [profile]})
            )
            if "--outputs" in cmd:
                values = (
                    np.array([1.0, 2.0])
                    if side == "base"
                    else np.array([1.0, 2.0 + 1e-12])
                )
                np.savez(cmd[cmd.index("--outputs") + 1], energy=values)
        return subprocess.CompletedProcess(cmd, status, "", "boom" if status else "")


@pytest.fixture
def trees(tmp_path):
    return {"base": tmp_path / "trees" / "base", "head": tmp_path / "trees" / "head"}


# ------------------------------------------------------------------ logic


def test_steps_cover_every_combination_and_parse_back():
    config = _config(dtypes=("float64", "float32"))

    names = ga.step_names(config)

    assert len(names) == 4 + 2 + 4 + 2
    assert names[0] == f"infer-{ARCH}-float64-none"
    for name in names:
        kind, arch, dtype, option = ga.parse_step(name, config)
        assert (kind, arch) == (name.split("-")[0], ARCH)
        assert dtype in name
        assert option in {"none", "dense_nbr"}
    with pytest.raises(ValueError, match="unknown step"):
        ga.parse_step("infer-nonsense", config)


def test_the_two_commits_alternate_in_every_timed_step():
    config = _config()

    runs = ga.runs_of(f"infer-{ARCH}-float64-none", config)

    assert [(r.side, r.index) for r in runs] == [
        ("base", 0),
        ("head", 0),
        ("base", 1),
        ("head", 1),
    ]
    train = ga.runs_of(f"train-{ARCH}-float64", config)
    assert all(r.extra == ("--train-step",) for r in train)
    scan = ga.runs_of(f"scan-{ARCH}-float64-dense_nbr", config)
    assert [(r.side, r.cell) for r in scan][:3] == [
        ("base", 1),
        ("head", 1),
        ("base", 2),
    ]
    numerics = ga.runs_of(f"numerics-{ARCH}-float64", config)
    assert [r.side for r in numerics] == ["base", "head"]
    assert numerics[0].extra == ("--outputs",)


def test_an_unknown_step_kind_is_refused():
    config = _config()

    with pytest.raises(ValueError, match="unknown step kind"):
        ga.runs_of(f"bogus-{ARCH}-float64", config)


def test_the_command_runs_the_profile_tool_of_this_tree_with_the_options(tmp_path):
    run = ga.Run(
        "s", "base", 0, "m.yaml", "float32", "dense_nbr", 2, 3, ("--train-step",)
    )

    cmd = ga.command(run, _config(), tmp_path / "r.json", "py")

    assert cmd[:3] == ["py", str(ga.PROFILE), "m.yaml"]
    assert cmd[cmd.index("--dtype") + 1] == "float32"
    assert cmd[cmd.index("--option") + 1] == "dense_nbr"
    assert cmd[cmd.index("--supercell") + 1] == "2"
    assert "--whole-only" in cmd
    assert "--train-step" in cmd
    outputs = ga.command(
        ga.Run("s", "head", 0, "m.yaml", "float64", "none", 1, 1, ("--outputs",)),
        _config(),
        tmp_path / "r.json",
        "py",
    )
    assert outputs[-2:] == ["--outputs", str(tmp_path / "r.npz")]


def test_a_step_records_each_run_and_marks_itself_done(tmp_path, trees):
    config = _config()
    fake = FakeExecutor()
    name = f"infer-{ARCH}-float64-none"

    ga.run_step(name, config, tmp_path, trees, fake)

    assert len(fake.calls) == 4
    assert sorted(p.name for p in (tmp_path / "runs" / name).iterdir()) == [
        "base-0.json",
        "base-1.json",
        "head-0.json",
        "head-1.json",
    ]
    assert (tmp_path / "done" / name).exists()
    ga.run_step(name, config, tmp_path, trees, fake)
    assert len(fake.calls) == 4  # a done step is skipped


def test_an_interrupted_step_resumes_at_the_first_missing_run(tmp_path, trees):
    config = _config()
    name = f"infer-{ARCH}-float64-none"
    fake = FakeExecutor()

    class Interrupted(Exception):
        pass

    def interrupt_after_two(cmd, env):
        if len(fake.calls) == 2:
            raise Interrupted
        return fake(cmd, env)

    with pytest.raises(Interrupted):
        ga.run_step(name, config, tmp_path, trees, interrupt_after_two)
    assert not (tmp_path / "done" / name).exists()

    ga.run_step(name, config, tmp_path, trees, fake)

    assert [c[0] for c in fake.calls] == ["base", "head", "base", "head"]
    assert (tmp_path / "done" / name).exists()


def test_a_failed_size_stops_the_scan_of_that_commit_only(tmp_path, trees):
    config = _config()
    fake = FakeExecutor(fail={("head", 2): 1})
    name = f"scan-{ARCH}-float64-none"

    ga.run_step(name, config, tmp_path, trees, fake)

    assert [(side, cell) for side, cell, _ in fake.calls] == [
        ("base", 1),
        ("head", 1),
        ("base", 2),
        ("head", 2),
        ("base", 3),
    ]
    assert "exit 1" in (tmp_path / "runs" / name / "head-2.failed").read_text()


def test_an_option_that_does_not_apply_ends_the_step(tmp_path, trees):
    config = _config()
    fake = FakeExecutor(fail={("base", 1): ga.NOT_APPLICABLE})
    name = f"infer-{ARCH}-float64-dense_nbr"

    ga.run_step(name, config, tmp_path, trees, fake)

    assert len(fake.calls) == 1
    assert (tmp_path / "runs" / name / "base-0.na").exists()


def test_a_failed_timed_run_is_recorded_and_the_others_still_run(tmp_path, trees):
    fake = FakeExecutor(fail={("head", 1): 1})
    name = f"infer-{ARCH}-float64-none"

    ga.run_step(name, _config(), tmp_path, trees, fake)

    assert len(fake.calls) == 4
    assert ga.recorded(tmp_path, ga.runs_of(name, _config())[1]) == "failed"


def test_the_first_config_is_saved_and_later_requests_use_it(tmp_path, capsys):
    first = _config()

    assert ga.load_config(tmp_path, first) == first
    again = ga.load_config(tmp_path, _config(pairs=9))

    assert again == first
    assert "using the settings saved" in capsys.readouterr().err
    assert ga.load_config(tmp_path, first) == first


def test_summary_medians_changes_and_flags(tmp_path, trees):
    config = _config()
    fake = FakeExecutor(ms={"base": 10.0, "head": 10.8})
    for name in ga.step_names(config):
        ga.run_step(name, config, tmp_path, trees, fake)

    rows = ga.write_summary(config, tmp_path)

    infer = rows["times"][0]
    assert (infer["base_ms"], infer["head_ms"], infer["base_n"]) == (10.0, 10.8, 2)
    assert infer["change"] == pytest.approx(0.08)
    assert infer["worse"]
    train = next(r for r in rows["times"] if r["step"].startswith("train-"))
    assert train["base_ms"] == 10.0
    scan = rows["scans"][0]
    assert (
        scan["base_largest_cell"]
        == scan["head_largest_cell"]
        == scan["common_cell"]
        == 3
    )
    assert scan["base_kb_per_bond"] == pytest.approx(1024 * 270.0 / 2700)
    assert not scan["worse"]
    numerics = rows["numerics"][0]
    assert numerics["max_abs_diff"] == pytest.approx(1e-12, rel=1e-3)
    assert numerics["key"] == "energy"
    text = (tmp_path / "summary.md").read_text()
    assert "WORSE" in text
    assert "base `aaaaaaaaaa` (A)" in text
    assert (
        json.loads((tmp_path / "summary.json").read_text())["scans"][0]["common_cell"]
        == 3
    )
    assert "base: " + "a" * 40 in (tmp_path / "environment.txt").read_text()


def test_the_summary_of_missing_results_has_empty_cells(tmp_path):
    rows = ga.write_summary(_config(), tmp_path)

    assert rows["times"][0]["base_ms"] is None
    assert rows["times"][0]["change"] is None
    assert "common_cell" not in rows["scans"][0]
    assert "max_abs_diff" not in rows["numerics"][0]
    assert "| n/a |" in (tmp_path / "summary.md").read_text()


def test_main_lists_the_steps_with_their_state(tmp_path, capsys):
    out = tmp_path / "out"

    assert (
        ga.main([
            str(out),
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--archs",
            ARCH,
            "--list",
        ])
        == 0
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"infer-{ARCH}-float64-none\ttodo"
    assert len(lines) == 4 + 2 + 4 + 2
    assert json.loads((out / "config.json").read_text())["head"] == ga.resolve("HEAD")


def test_main_runs_the_requested_step_then_writes_the_summary(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "out"
    fake = FakeExecutor()
    monkeypatch.setattr(
        ga,
        "ensure_tree",
        lambda _out, sha: tmp_path / ("base" if sha == "x" else "head"),
    )
    monkeypatch.setattr(ga, "resolve", lambda ref: "x" if ref == "A" else "y")
    monkeypatch.setattr(ga, "execute", fake)
    name = f"infer-{ARCH}-float64-none"

    status = ga.main([
        str(out),
        "--base",
        "A",
        "--head",
        "B",
        "--archs",
        ARCH,
        "--pairs",
        "1",
        "--step",
        name,
    ])

    assert status == 0
    assert len(fake.calls) == 2
    assert "| infer-" in capsys.readouterr().out
    with pytest.raises(ValueError, match="unknown step"):
        ga.main([str(out), "--step", "infer-bogus"])


def test_main_summary_only_takes_no_run(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(ga, "ensure_tree", lambda *_a: pytest.fail("no tree needed"))

    assert (
        ga.main([
            str(out),
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--archs",
            ARCH,
            "--summary",
        ])
        == 0
    )
    assert (out / "summary.md").exists()


def test_a_tree_whose_package_is_not_the_imported_one_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ga, "package_location", lambda _tree: "/elsewhere/tensorpotential/__init__.py"
    )

    with pytest.raises(RuntimeError, match="takes precedence"):
        ga.ensure_tree(tmp_path, ga.resolve("HEAD"))
    assert not (tmp_path / "trees" / ga.resolve("HEAD") / ".complete").exists()


# ------------------------------------------------------------------ values


def test_a_commit_is_extracted_once_and_imported_from_its_tree(tmp_path):
    sha = ga.resolve("HEAD")

    tree = ga.ensure_tree(tmp_path, sha)

    assert (tree / "tensorpotential" / "__init__.py").read_bytes() == (
        ROOT / "tensorpotential" / "__init__.py"
    ).read_bytes()
    assert ga.package_location(tree) == str(tree / "tensorpotential" / "__init__.py")
    stamp = (tree / ".complete").stat().st_mtime_ns
    assert ga.ensure_tree(tmp_path, sha) == tree
    assert (tree / ".complete").stat().st_mtime_ns == stamp  # not extracted again


def test_the_wrapper_runs_the_step_of_its_array_task(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" --list "* ]]; then printf "first\\ttodo\\nsecond\\ttodo\\n"; '
        'else echo "ARGS $*"; fi\n'
    )
    fake_python.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake_python), "SLURM_ARRAY_TASK_ID": "1"}

    done = subprocess.run(  # noqa: S603 - the wrapper under test
        ["bash", str(ROOT / "tools" / "slurm" / "gpu_ab.sh"), "OUT", "--base", "A"],  # noqa: S607
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert done.stdout.strip() == "ARGS tools/gpu_ab.py OUT --base A --step second"
    plain = {k: v for k, v in env.items() if k != "SLURM_ARRAY_TASK_ID"}
    direct = subprocess.run(  # noqa: S603 - the wrapper under test
        ["bash", str(ROOT / "tools" / "slurm" / "gpu_ab.sh"), "OUT", "--summary"],  # noqa: S607
        capture_output=True,
        text=True,
        env=plain,
        check=True,
    )
    assert direct.stdout.strip() == "ARGS tools/gpu_ab.py OUT --summary"


def test_a_relative_output_directory_still_imports_from_the_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = ga.resolve("HEAD")

    tree = ga.ensure_tree(Path("relative_out"), sha)

    assert tree.is_absolute()
    assert ga.package_location(tree) == str(tree / "tensorpotential" / "__init__.py")
