"""Logic tests for tools/junit_outcomes.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import junit_outcomes as jo

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="8">
  <testcase classname="test_a" name="test_ok"/>
  <testcase classname="test_a" name="test_bad"><failure message="boom"/></testcase>
  <testcase classname="test_a" name="test_err"><error message="setup"/></testcase>
  <testcase classname="test_a" name="test_skip">
    <skipped type="pytest.skip" message="no model"/></testcase>
  <testcase classname="test_b" name="test_xfail">
    <skipped type="pytest.xfail" message="known"/></testcase>
  <testcase classname="test_b" name="test_xpass"/>
  <testcase classname="test_b" name="test_teardown"/>
  <testcase classname="test_b" name="test_teardown">
    <error message="teardown"/></testcase>
  <testcase classname="test_b" name="test_fail_then_teardown">
    <failure message="call"/></testcase>
  <testcase classname="test_b" name="test_fail_then_teardown">
    <error message="teardown"/></testcase>
</testsuite></testsuites>
"""

LOG = """some output
=========================== short test summary info ============================
XFAIL test_b.py::test_xfail
XPASS test_b.py::test_xpass
XPASS tests/sub/test_c.py::test_never_ran
= 1 failed in 1s =
"""


@pytest.fixture
def junit(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT)
    return path


def test_each_outcome_is_classified(junit):
    tests = jo.summarize(junit)["tests"]
    assert tests["test_a::test_ok"] == "passed"
    assert tests["test_a::test_bad"] == "failed"
    assert tests["test_a::test_err"] == "error"
    assert tests["test_a::test_skip"] == "skipped"
    assert tests["test_b::test_xfail"] == "xfailed"
    assert tests["test_b::test_xpass"] == "passed"  # junit cannot tell without a log


def test_teardown_error_overrides_pass_but_not_a_failure(junit):
    tests = jo.summarize(junit)["tests"]
    assert tests["test_b::test_teardown"] == "error"
    assert tests["test_b::test_fail_then_teardown"] == "failed"


def test_log_marks_xpass_only_for_tests_that_passed(junit, tmp_path):
    log = tmp_path / "pytest.log"
    log.write_text(LOG.replace("XPASS tests/sub/test_c.py::test_never_ran\n", ""))
    result = jo.summarize(junit, log)
    assert result["tests"]["test_b::test_xpass"] == "xpassed"
    assert result["tests"]["test_b::test_xfail"] == "xfailed"
    assert result["counts"] == {
        "passed": 1,
        "failed": 2,
        "error": 2,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
    }


def test_log_ids_match_junit_class_names_with_a_rootdir_prefix(tmp_path):
    junit = tmp_path / "j.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_b" name="test_xpass"/>'
        '<testcase classname="tests.test_b" name="test_xpass_other"/>'
        "</testsuite></testsuites>"
    )
    log = tmp_path / "p.log"
    log.write_text("XPASS test_b.py::test_xpass\n")
    tests = jo.summarize(junit, log)["tests"]
    assert tests["tests.test_b::test_xpass"] == "xpassed"
    assert tests["tests.test_b::test_xpass_other"] == "passed"


def test_xpass_does_not_hide_a_teardown_error(tmp_path):
    junit = tmp_path / "j.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_b" name="test_xpass"/>'
        '<testcase classname="tests.test_b" name="test_xpass">'
        '<error message="teardown"/></testcase>'
        "</testsuite></testsuites>"
    )
    log = tmp_path / "p.log"
    log.write_text("XPASS test_b.py::test_xpass\n")
    assert jo.summarize(junit, log)["tests"]["tests.test_b::test_xpass"] == "error"


@pytest.mark.parametrize(
    "xpass_line",
    [
        "XPASS test_b.py::test_unknown",  # not in the junit file
        "XPASS test_b.py::test_xpass",  # ambiguous: two modules named test_b
    ],
)
def test_xpass_line_without_a_unique_junit_match_is_an_error(tmp_path, xpass_line):
    junit = tmp_path / "j.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_b" name="test_xpass"/>'
        '<testcase classname="tests.sub.test_b" name="test_xpass"/>'
        "</testsuite></testsuites>"
    )
    log = tmp_path / "p.log"
    log.write_text(xpass_line + "\n")
    with pytest.raises(ValueError, match="matches"):
        jo.summarize(junit, log)


def test_counts_cover_every_outcome_name_even_when_zero(junit):
    assert set(jo.summarize(junit)["counts"]) == set(jo.OUTCOMES)


def test_xpass_ids_and_nodeid_conversion():
    assert jo.xpass_ids(LOG) == {
        "test_b.py::test_xpass",
        "tests/sub/test_c.py::test_never_ran",
    }
    assert (
        jo._junit_id("tests/sub/test_c.py::test_x[a-1]")
        == "tests.sub.test_c::test_x[a-1]"
    )
    assert jo._junit_id("test_b.py::test_y") == "test_b::test_y"


def test_compare_reports_added_removed_and_changed():
    old = {"tests": {"a": "passed", "b": "failed", "c": "passed"}}
    new = {"tests": {"a": "passed", "b": "passed", "d": "passed"}}
    assert jo.compare(old, new) == {
        "added": ["d"],
        "removed": ["c"],
        "changed": [["b", "failed", "passed"]],
    }
    assert not any(jo.compare(old, old).values())


def test_cli_roundtrip_and_exit_status(junit, tmp_path, capsys):
    base = tmp_path / "base.json"
    assert jo.main(["summarize", str(junit), str(base)]) == 0
    assert "failed" in capsys.readouterr().out
    assert jo.main(["compare", str(base), str(base)]) == 0

    changed = json.loads(base.read_text())
    changed["tests"]["test_a::test_ok"] = "failed"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(changed))
    assert jo.main(["compare", str(base), str(other)]) == 1
    assert "changed: ['test_a::test_ok', 'passed', 'failed']" in capsys.readouterr().out
