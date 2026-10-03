"""Logic tests for tools/coverage_ratchet.py (no physics in this tool).

Reports are built the way coverage.py writes them (format 3) and, in one test, by a real
``coverage run``, so the field names are checked against the library that produces them.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coverage_ratchet as cr


def report(files: dict[str, tuple[int, int, int, int]], branch: bool = True) -> dict:
    """A coverage.py report; each file is (covered lines, statements, covered branches, branches)."""
    return {
        "meta": {"version": "7.0.0", "branch_coverage": branch},
        "files": {
            path: {
                "summary": {
                    "covered_lines": cl,
                    "num_statements": ns,
                    "covered_branches": cb,
                    "num_branches": nb,
                }
            }
            for path, (cl, ns, cb, nb) in files.items()
        },
    }


BASE = report({"a.py": (8, 10, 3, 4), "b.py": (5, 5, 0, 0)})  # a: 11/14, b: 5/5


def write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_counts_add_statements_and_branches():
    assert cr.counts_from_report(BASE) == {
        "a.py": {"covered": 11, "total": 14},
        "b.py": {"covered": 5, "total": 5},
    }


def test_report_without_branch_data_is_refused():
    with pytest.raises(cr.RatchetError, match="--cov-branch"):
        cr.counts_from_report(report({"a.py": (1, 1, 0, 0)}, branch=False))


def test_percent_and_empty_file():
    assert cr.percent({"covered": 11, "total": 14}) == pytest.approx(100 * 11 / 14)
    assert cr.percent({"covered": 0, "total": 0}) == 100.0


def test_falls_lists_only_files_that_lost_share():
    base = cr.counts_from_report(BASE)
    worse = cr.counts_from_report(report({"a.py": (7, 10, 3, 4), "b.py": (5, 5, 0, 0)}))
    better = cr.counts_from_report(
        report({"a.py": (9, 10, 3, 4), "b.py": (5, 5, 0, 0)})
    )
    assert cr.falls(base, worse) == ["fell: a.py 78.57% -> 71.43%"]
    assert cr.falls(base, better) == []
    assert cr.falls(base, base) == []


def test_tolerance_in_percentage_points():
    base = {"a.py": {"covered": 100, "total": 100}}
    now = {"a.py": {"covered": 99, "total": 100}}
    assert cr.falls(base, now, tolerance=0.5)
    assert cr.falls(base, now, tolerance=1.0) == []


def test_share_not_count_decides():
    # removing covered lines while the rest stays equally covered is not a fall
    base = {"a.py": {"covered": 80, "total": 100}}
    assert cr.falls(base, {"a.py": {"covered": 40, "total": 50}}) == []
    assert cr.falls(base, {"a.py": {"covered": 30, "total": 50}})


def test_a_fall_that_adds_uncovered_code_is_listed():
    base = {"a.py": {"covered": 80, "total": 100}}
    assert cr.falls(base, {"a.py": {"covered": 70, "total": 100}})


def test_a_rise_that_adds_uncovered_code_is_not_listed():
    # 85% of 200 leaves 30 uncovered against 20, but the share went up
    base = {"a.py": {"covered": 80, "total": 100}}
    assert cr.falls(base, {"a.py": {"covered": 170, "total": 200}}) == []


def test_unchanged_counts_are_not_listed():
    base = {"a.py": {"covered": 80, "total": 100}, "e.py": {"covered": 0, "total": 0}}
    assert cr.falls(base, dict(base)) == []


def test_record_then_check_round_trip(tmp_path, capsys):
    cov = write(tmp_path, "cov.json", BASE)
    baseline = tmp_path / "base.json"
    assert cr.main(["record", str(cov), "--baseline", str(baseline)]) == 0
    assert "recorded 2 files" in capsys.readouterr().out
    data = json.loads(baseline.read_text())
    assert data["tool"] == "coverage.py 7.0.0"
    assert cr.main(["check", str(cov), "--baseline", str(baseline)]) == 0
    assert "0 fell, 0 gone, 0 new" in capsys.readouterr().out


def test_check_fails_on_a_fall_and_names_the_file(tmp_path, capsys):
    baseline = tmp_path / "base.json"
    cr.main([
        "record",
        str(write(tmp_path, "c0.json", BASE)),
        "--baseline",
        str(baseline),
    ])
    capsys.readouterr()
    lower = write(
        tmp_path, "c1.json", report({"a.py": (6, 10, 3, 4), "b.py": (5, 5, 0, 0)})
    )
    assert cr.main(["check", str(lower), "--baseline", str(baseline)]) == 1
    assert "fell: a.py" in capsys.readouterr().out


def test_gone_and_new_files_are_reported_not_failed(tmp_path, capsys):
    baseline = tmp_path / "base.json"
    cr.main([
        "record",
        str(write(tmp_path, "c0.json", BASE)),
        "--baseline",
        str(baseline),
    ])
    capsys.readouterr()
    now = write(
        tmp_path, "c1.json", report({"a.py": (8, 10, 3, 4), "c.py": (0, 4, 0, 0)})
    )
    assert cr.main(["check", str(now), "--baseline", str(baseline)]) == 0
    out = capsys.readouterr().out
    assert "gone: b.py" in out
    assert "1 gone, 1 new" in out


def test_record_refuses_a_fall_unless_allowed(tmp_path, capsys):
    baseline = tmp_path / "base.json"
    cr.main([
        "record",
        str(write(tmp_path, "c0.json", BASE)),
        "--baseline",
        str(baseline),
    ])
    capsys.readouterr()
    lower = write(
        tmp_path, "c1.json", report({"a.py": (6, 10, 3, 4), "b.py": (5, 5, 0, 0)})
    )
    before = baseline.read_text()
    assert cr.main(["record", str(lower), "--baseline", str(baseline)]) == 1
    assert "refusing to record a fall" in capsys.readouterr().out
    assert baseline.read_text() == before
    assert (
        cr.main(["record", str(lower), "--baseline", str(baseline), "--allow-fall"])
        == 0
    )
    assert baseline.read_text() != before


def test_record_accepts_a_rise_and_new_files(tmp_path):
    baseline = tmp_path / "base.json"
    cr.main([
        "record",
        str(write(tmp_path, "c0.json", BASE)),
        "--baseline",
        str(baseline),
    ])
    higher = write(
        tmp_path, "c1.json", report({"a.py": (10, 10, 4, 4), "n.py": (1, 2, 0, 0)})
    )
    assert cr.main(["record", str(higher), "--baseline", str(baseline)]) == 0
    assert set(json.loads(baseline.read_text())["files"]) == {"a.py", "n.py"}


@pytest.mark.parametrize("command", ["check", "record"])
def test_unreadable_inputs_exit_with_status_2(tmp_path, capsys, command):
    assert cr.main([command, str(tmp_path / "missing.json")]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert cr.main([command, str(broken)]) == 2
    capsys.readouterr()


def test_check_without_a_baseline_says_how_to_make_one(tmp_path, capsys):
    cov = write(tmp_path, "cov.json", BASE)
    assert cr.main(["check", str(cov), "--baseline", str(tmp_path / "none.json")]) == 2
    assert "coverage_ratchet.py record" in capsys.readouterr().err


def test_field_names_match_what_coverage_py_writes(tmp_path):
    (tmp_path / "m.py").write_text(
        "def f(x):\n    if x:\n        return 1\n    return 2\n"
    )
    (tmp_path / "d.py").write_text("import m\nm.f(1)\n")
    for args in (
        ["run", "--branch", "--source=.", "d.py"],
        ["json", "-q", "-o", "cov.json"],
    ):
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "coverage", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    counts = cr.counts_from_report(json.loads((tmp_path / "cov.json").read_text()))
    # m.py: def, if, return 1, return 2 = 4 statements, 2 branches; run with f(1): 3 + 1 executed
    assert counts["m.py"] == {"covered": 4, "total": 6}
