"""Tests for tools/bench_cpu.py.

Logic tests drive the statistics, the table, the idle check and the command line with hand-made samples
and a fake runner. Value tests run the real cases at a tiny size and check their numbers against oracles
that do not use the tool: the atom counts of the supercells, the number of timed calls, the sum of the
phases and a fresh interpreter's own view of the import.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import bench_cpu as bc  # noqa: E402

pytest.importorskip("tensorpotential")
tf = pytest.importorskip("tensorflow")

import grace_probe as gp  # noqa: E402

gp.osn.configure_tensorflow()  # one TensorFlow configuration for every test module of tools/


def _sample(wall, rss=100.0, foreign=0.0, **phases):
    return {
        "wall_ms": wall,
        "peak_rss_mb": rss,
        "foreign_cores": foreign,
        "phases_ms": phases,
    }


# ------------------------------------------------------------------ logic


def test_console_scripts_are_the_table_of_pyproject():
    scripts = bc.console_scripts()

    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "scripts"
    ]
    assert scripts == expected
    assert scripts["grace"] == "tensorpotential.scripts.grace:main"


def test_default_cases_cover_every_script_yaml_and_size():
    cases = bc.default_cases()

    assert {f"script:{name}" for name in bc.console_scripts()} <= set(cases)
    assert {f"calc_init:{name}" for name in bc.YAMLS} <= set(cases)
    assert {f"builder:{size}" for size in bc.BUILDER_SIZES} <= set(cases)
    assert cases[:1] == ["import:tensorflow"]
    assert cases[-3:] == ["train_step", "train_run", "calc_eval"]
    assert len(cases) == len(set(cases))


def test_busy_processes_keeps_only_test_and_training_lines():
    ps = "1 10 /sbin/init\n2 5 python -m pytest tests -q\n3 7 gracemaker input.yaml\n4 9 vim notes"

    assert bc.busy_processes(ps) == [
        "2 5 python -m pytest tests -q",
        "3 7 gracemaker input.yaml",
    ]


@pytest.mark.parametrize(
    ("load", "ps", "expected"),
    [
        (0.5, "1 1 /sbin/init", None),
        (4.5, "1 1 /sbin/init", "load"),
        (0.5, "1 1 python -m pytest x", "test or training"),
        # this process's own pytest line is not another job
        (0.5, f"{os.getpid()} 1 python -m pytest tools/tests", None),
    ],
)
def test_idle_problem_names_the_reason(monkeypatch, load, ps, expected):
    monkeypatch.setattr(bc.os, "getloadavg", lambda: (load, 0.0, 0.0))
    monkeypatch.setattr(
        bc.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=ps)
    )

    problem = bc.idle_problem()

    if expected is None:
        assert problem is None
    else:
        assert problem is not None
        assert expected in problem


def test_idle_problem_ignores_the_load_when_asked_not_to_read_it(monkeypatch):
    monkeypatch.setattr(bc.os, "getloadavg", lambda: (9.0, 0.0, 0.0))
    monkeypatch.setattr(
        bc.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout="")
    )

    assert bc.idle_problem(check_load=False) is None
    assert "load" in (bc.idle_problem() or "")


def test_only_the_first_check_of_a_run_reads_the_load(monkeypatch):
    checks = []
    monkeypatch.setattr(
        bc, "idle_problem", lambda check_load=True: checks.append(check_load)
    )
    monkeypatch.setattr(bc, "run_sample", lambda *_a: _sample(1, a_ms=1))

    bc.run(["a", "b", "c"], repeats=1, steps=1, structures=1)

    assert checks == [True, False, False, False]  # the last is the check at the end


def test_summarise_reports_median_not_mean_and_the_spread():
    samples = [
        _sample(w, rss=r, foreign=f, a_ms=a)
        for w, r, f, a in [(100, 10, 0.0, 1), (110, 20, 0.2, 2), (400, 30, 0.1, 9)]
    ]

    row = bc.summarise("case", samples)

    assert row["wall_ms"] == {"median": 110, "min": 100, "max": 400}
    assert row["peak_rss_mb"]["median"] == 20
    assert row["phases_ms"] == {"a_ms": 2}
    assert row["foreign_cores_max"] == pytest.approx(0.2)
    assert (row["runs"], row["failed"]) == (3, 0)


def test_summarise_counts_failures_and_keeps_the_last_error():
    samples = [
        _sample(100, a_ms=1),
        {"wall_ms": 5, "foreign_cores": None, "error": "ModuleNotFoundError: flask"},
    ]

    row = bc.summarise("script:grace_dashboard", samples)

    assert (row["runs"], row["failed"]) == (2, 1)
    assert row["wall_ms"]["median"] == 100
    only_failures = bc.summarise("x", samples[1:])
    assert only_failures["error"] == "ModuleNotFoundError: flask"
    assert "wall_ms" not in only_failures


def test_summarise_without_a_measurement_of_other_processes():
    row = bc.summarise("x", [_sample(1, foreign=None, a_ms=1)])

    assert row["foreign_cores_max"] is None


def test_render_table_has_a_row_per_case_and_shows_failures():
    rows = [
        bc.summarise(
            "import:json", [_sample(12.4, rss=30.2, foreign=0.0, import_ms=11.5)]
        ),
        {"case": "script:x", "runs": 5, "failed": 5, "error": "boom"},
    ]

    lines = bc.render_table(rows).splitlines()

    assert lines[2] == "| import:json | 1 | 12 [12, 12] | 30 | import 11.5 | 0.00 |"
    assert lines[3] == "| script:x | 5 | failed: boom | | | |"
    assert len(lines) == 4


def test_render_table_marks_an_unmeasured_load():
    row = bc.summarise("x", [_sample(1, foreign=None, a_ms=1)])

    assert bc.render_table([row]).splitlines()[2].endswith("| n/a |")


def test_run_repeats_every_case_and_records_the_load(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "idle_problem", lambda **_kw: None)
    monkeypatch.setattr(
        bc,
        "run_sample",
        lambda name, steps, structures: (
            calls.append((name, steps, structures)) or _sample(1, a_ms=1)
        ),
    )

    result = bc.run(["a", "b"], repeats=3, steps=7, structures=9)

    assert calls == [("a", 7, 9)] * 3 + [("b", 7, 9)] * 3
    assert [row["case"] for row in result["rows"]] == ["a", "b"]
    assert result["idle_at_end"] is True
    assert result["load_average_1min_before"] >= 0


def test_wait_until_idle_polls_until_the_problem_goes_away(monkeypatch):
    answers = iter(["busy", "busy", None])
    sleeps = []
    monkeypatch.setattr(bc, "idle_problem", lambda **_kw: next(answers))
    monkeypatch.setattr(bc.time, "sleep", sleeps.append)

    assert bc.wait_until_idle(False, wait_s=1e6, poll_s=3) is None
    assert sleeps == [3, 3]


def test_wait_until_idle_gives_up_after_the_wait(monkeypatch):
    monkeypatch.setattr(bc, "idle_problem", lambda **_kw: "busy")
    monkeypatch.setattr(bc.time, "sleep", lambda _s: None)

    assert bc.wait_until_idle(True, wait_s=0) == "busy"


def test_run_refuses_a_busy_machine_unless_allowed(monkeypatch):
    monkeypatch.setattr(
        bc, "idle_problem", lambda **_kw: "one-minute load 9 is above 4.0"
    )
    monkeypatch.setattr(bc, "run_sample", lambda *_a: _sample(1, a_ms=1))

    with pytest.raises(SystemExit, match="not idle before a"):
        bc.run(["a"], repeats=1, steps=1, structures=1, wait_idle_s=0)

    result = bc.run(["a"], repeats=1, steps=1, structures=1, allow_busy=True)
    assert result["idle_at_end"] is False


def test_peak_rss_reads_the_high_water_mark_and_falls_back_without_proc(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tx\nVmHWM:\t   2048 kB\nVmRSS:\t 1000 kB\n")

    assert bc.peak_rss_mb(status) == 2.0
    assert bc.peak_rss_mb(tmp_path / "missing") > 5  # ru_maxrss of this process
    assert bc.peak_rss_mb() > 5


def test_run_case_rejects_an_unknown_case():
    with pytest.raises(SystemExit, match="unknown case 'nope'"):
        bc.run_case("nope")


def test_main_case_prints_one_json_line(monkeypatch, capsys):
    monkeypatch.setattr(
        bc,
        "run_case",
        lambda name, steps, structures: {
            "name": name,
            "steps": steps,
            "structures": structures,
        },
    )

    assert bc.main(["case", "train_run", "--steps", "5", "--structures", "3"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "name": "train_run",
        "steps": 5,
        "structures": 3,
    }


def test_main_run_prints_the_table_and_writes_the_summary(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(bc, "idle_problem", lambda **_kw: None)
    monkeypatch.setattr(bc, "run_sample", lambda *_a: _sample(10, import_ms=9))
    out = tmp_path / "bench.json"

    assert (
        bc.main(["run", "--cases", "import:json", "--repeats", "2", "--out", str(out)])
        == 0
    )

    text = capsys.readouterr().out
    assert "| import:json | 2 | 10 [10, 10] |" in text
    assert "load average before" in text
    assert json.loads(out.read_text())["rows"][0]["case"] == "import:json"


# ------------------------------------------------------------------ values


def test_a_sample_of_a_fresh_interpreter_has_wall_time_memory_and_an_idle_reading():
    sample = bc.run_sample("import:json", steps=1, structures=1)

    assert "error" not in sample
    # the child's own import is part of its wall time, which also holds interpreter start-up
    assert 0 < sample["phases_ms"]["import_ms"] < sample["wall_ms"]
    # a Python interpreter holds between 5 and 200 MB resident, whatever the platform
    assert 5 < sample["peak_rss_mb"] < 200
    assert sample["foreign_cores"] is not None


def test_a_failing_child_becomes_an_error_sample():
    sample = bc.run_sample("import:no_such_module_for_bench", steps=1, structures=1)

    assert "ModuleNotFoundError" in sample["error"]
    assert "phases_ms" not in sample


def test_a_script_case_times_the_help_of_a_real_console_script():
    phases = bc._script_case("extxyz2df")

    assert phases["import_ms"] > 0
    assert phases["help_ms"] > 0


def test_the_builder_case_measures_the_supercell_it_names():
    largest = max(len(a) for a in bc._atoms_list())

    for size, edge in bc.BUILDER_SIZES.items():
        phases = bc._builder_case(size, repeats=2)

        assert phases["n_atoms"] == largest * edge**3, size
        assert phases["neighbour_list_ms"] > 0


def test_the_calculator_init_case_builds_the_model_and_the_calculator():
    phases = bc._calc_init_case(bc.YAMLS[0])

    assert phases["build_model_ms"] > 0
    assert phases["construct_calculator_ms"] > 0


def test_the_training_case_runs_the_steps_it_is_asked_for(monkeypatch):
    steps = []
    real = bc._timed

    def counting(phases, name, fn):
        steps.append(name)
        return real(phases, name, fn)

    monkeypatch.setattr(bc, "_timed", counting)

    phases = bc._train_case(4, structures=8)

    assert steps.count("first_step_ms") == 1
    assert steps.count("steady_steps_ms") == 3
    assert phases["steady_steps_ms"] > 0
    assert phases["dataset_ms"] > 0


def test_the_calculator_evaluation_case_splits_its_time_into_data_model_and_the_rest():
    structures = 5

    phases = bc._calc_eval_case(structures)

    assert phases["data_ms"] > 0
    assert phases["model_ms"] > 0
    assert phases["other_ms"] >= 0
    parts = phases["data_ms"] + phases["model_ms"] + phases["other_ms"]
    assert parts == pytest.approx(phases["evaluate_total_ms"])
    # get_potential_energy and get_forces of one structure share one calculation: one call each
    assert phases["evaluate_total_ms"] > phases["model_ms"]


def test_run_case_reports_peak_memory_of_this_process():
    result = bc.run_case("import:json")

    assert result["phases_ms"]["import_ms"] >= 0
    assert result["peak_rss_mb"] > 5


# ------------------------------------------------------------------ device


def test_the_device_is_the_cpu_unless_the_environment_says_gpu(monkeypatch):
    monkeypatch.delenv(bc.DEVICE_ENV, raising=False)
    assert bc.bench_device() == "cpu"

    monkeypatch.setenv(bc.DEVICE_ENV, "gpu")
    assert bc.bench_device() == "gpu"


def test_a_gpu_run_without_a_gpu_is_refused(monkeypatch):
    monkeypatch.setenv(bc.DEVICE_ENV, "gpu")
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda _kind=None: [])

    with pytest.raises(RuntimeError, match="sees no GPU"):
        bc._tensorflow_phases({})


def test_a_cpu_run_hides_the_gpus_and_a_gpu_run_does_not(monkeypatch):
    seen = []

    def fake_run(_command, **kwargs):
        seen.append(kwargs["env"].get("CUDA_VISIBLE_DEVICES"))
        return SimpleNamespace(
            returncode=0, stdout='{"phases_ms": {}, "peak_rss_mb": 1}\n', stderr=""
        )

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(bc.DEVICE_ENV, "cpu")
    bc.run_sample("import:json", steps=1, structures=1)
    monkeypatch.setenv(bc.DEVICE_ENV, "gpu")
    bc.run_sample("import:json", steps=1, structures=1)

    assert seen == ["-1", None]


def test_a_gpu_case_reports_the_peak_gpu_memory_in_mb(monkeypatch):
    monkeypatch.setenv(bc.DEVICE_ENV, "gpu")
    monkeypatch.setattr(
        tf.config.experimental, "get_memory_info", lambda _device: {"peak": 3 * 2**20}
    )

    result = bc.run_case("import:json")

    assert result["phases_ms"]["gpu_peak_mb"] == 3.0


def test_main_run_sets_the_device_and_the_summary_names_it(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv(bc.DEVICE_ENV, "cpu")
    monkeypatch.setattr(bc, "idle_problem", lambda **_kw: None)
    monkeypatch.setattr(bc, "run_sample", lambda *_a: _sample(10, import_ms=9))
    out = tmp_path / "bench.json"

    bc.main([
        "run",
        "--device",
        "gpu",
        "--cases",
        "import:json",
        "--repeats",
        "1",
        "--out",
        str(out),
    ])

    assert "device gpu" in capsys.readouterr().out
    assert json.loads(out.read_text())["device"] == "gpu"
