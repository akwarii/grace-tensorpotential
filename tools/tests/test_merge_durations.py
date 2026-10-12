"""Logic tests for tools/merge_durations.py (no physics in this tool)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merge_durations as md

REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


# --- load_durations -----------------------------------------------------------------------------------------------


def test_load_reads_ids_to_floats(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"a.py::t": 1, "b.py::u": 2.5})
    loaded = md.load_durations(path)
    assert loaded == {"a.py::t": 1.0, "b.py::u": 2.5}
    assert all(isinstance(v, float) for v in loaded.values())


@pytest.mark.parametrize(
    "content, match",
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"a.py::t": "slow"}', "expected a finite non-negative number"),
        ('{"a.py::t": true}', "expected a finite non-negative number"),
        ('{"a.py::t": -0.1}', "expected a finite non-negative number"),
        ('{"a.py::t": NaN}', "expected a finite non-negative number"),
        ('{"a.py::t": Infinity}', "expected a finite non-negative number"),
        ('{"a.py::t": null}', "expected a finite non-negative number"),
    ],
)
def test_load_refuses_a_malformed_file(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / "d.json"
    path.write_text(content)
    with pytest.raises(md.DurationsError, match=match):
        md.load_durations(path)


def test_load_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(md.DurationsError, match="cannot be read"):
        md.load_durations(tmp_path / "absent.json")


# --- format ------------------------------------------------------------------------------------------------------


def test_format_sorts_keys_indents_four_and_ends_with_a_newline() -> None:
    text = md.format_durations({"b.py::t": 2.0, "a.py::t": 1.5})
    assert text == '{\n    "a.py::t": 1.5,\n    "b.py::t": 2.0\n}\n'


def test_format_reproduces_the_committed_file_byte_for_byte() -> None:
    committed = REPO / "tests" / ".test_durations"
    assert md.format_durations(md.load_durations(committed)) == committed.read_text()


# --- merge -------------------------------------------------------------------------------------------------------


def test_merge_takes_the_shard_value_and_reports_new_and_dropped() -> None:
    committed = {"a.py::keep": 1.0, "a.py::gone": 5.0, "a.py::moved": 2.0}
    shards = [{"a.py::keep": 1.5, "a.py::fresh": 0.5}, {"a.py::moved": 4.0}]
    result = md.merge(committed, shards)
    assert result.durations == {
        "a.py::keep": 1.5,
        "a.py::fresh": 0.5,
        "a.py::moved": 4.0,
    }
    assert result.new == {"a.py::fresh"}
    assert result.dropped == {"a.py::gone"}


def test_merge_is_independent_of_the_order_of_the_shards() -> None:
    committed = {"a.py::t": 1.0}
    shards = [{"a.py::t": 1.0}, {"b.py::t": 2.0}, {"c.py::t": 3.0}]
    forward = md.merge(committed, shards)
    backward = md.merge(committed, shards[::-1])
    assert forward == backward
    assert md.format_durations(forward.durations) == md.format_durations(
        backward.durations
    )


def test_merge_without_a_committed_file_treats_every_test_as_new() -> None:
    result = md.merge({}, [{"a.py::t": 1.0}])
    assert result.new == {"a.py::t"}
    assert result.dropped == frozenset()
    assert result.changed_share == 0.0


def test_merge_refuses_a_test_in_two_shards() -> None:
    with pytest.raises(md.DurationsError, match=r"a.py::t is in shard files 1 and 2"):
        md.merge({}, [{"a.py::t": 1.0}, {"a.py::t": 2.0}])


def test_merge_refuses_no_shard_and_an_empty_shard() -> None:
    with pytest.raises(md.DurationsError, match="no shard file"):
        md.merge({"a.py::t": 1.0}, [])
    with pytest.raises(md.DurationsError, match="shard file 2 holds no test"):
        md.merge({"a.py::t": 1.0}, [{"a.py::t": 1.0}, {}])


def test_changed_share_weighs_the_tests_that_moved_more_than_twenty_percent() -> None:
    committed = {
        "a::stable": 10.0,
        "a::slower": 10.0,
        "a::faster": 10.0,
        "a::edge": 10.0,
    }
    merged = {
        "a::stable": 11.0,
        "a::slower": 13.0,
        "a::faster": 5.0,
        "a::edge": 12.0,
        "a::new": 40.0,
    }
    # slower (+30%) and faster (-50%) changed; stable (+10%) and edge (+20%, not more) did not; a new test is not counted
    assert md._changed_share(committed, merged) == pytest.approx((13.0 + 5.0) / 81.0)


def test_changed_share_of_a_test_that_was_free_and_is_not_counts_as_changed() -> None:
    assert md._changed_share({"a::t": 0.0}, {"a::t": 2.0}) == 1.0
    assert md._changed_share({"a::t": 0.0}, {"a::t": 0.0}) == 0.0


# --- summary -----------------------------------------------------------------------------------------------------


def test_summary_lists_the_counts_the_workflow_reports() -> None:
    result = md.merge({"a::gone": 1.0, "a::t": 1.0}, [{"a::t": 2.0, "a::fresh": 2.0}])
    text = md.summary_markdown(result)
    assert "| tests in the file | 2 |" in text
    assert "| new (not in the committed file) | 1 |" in text
    assert "| dropped (no longer run) | 1 |" in text
    assert "| total duration | 4 s |" in text
    assert "changed by more than 20% | 50.0% |" in text


# --- collected ids and staleness --------------------------------------------------------------------------------


COLLECTED = """tests/test_a.py::test_one
tests/test_a.py::test_two[param with space]
tests/core/test_b.py::TestK::test_three

=============================== warnings summary ===============================
  /x/y.py:12: UserWarning: a::b
3 tests collected in 1.20s
"""


def test_parse_collected_keeps_ids_and_drops_summary_and_warning_lines() -> None:
    assert md.parse_collected(COLLECTED) == {
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two[param with space]",
        "tests/core/test_b.py::TestK::test_three",
    }


def test_staleness_is_silent_up_to_the_threshold_and_warns_above_it() -> None:
    collected = {f"t.py::{i}" for i in range(10)}
    committed = {f"t.py::{i}": 1.0 for i in range(9)}  # one of ten unknown: exactly 10%
    assert md.staleness_message(committed, collected) is None
    message = md.staleness_message({f"t.py::{i}": 1.0 for i in range(8)}, collected)
    assert message is not None
    assert message.startswith(
        "::warning title=Stale test durations::2 of 10 collected tests (20%)"
    )


def test_staleness_refuses_an_empty_collection() -> None:
    with pytest.raises(md.DurationsError, match="no collected test id"):
        md.staleness_message({"a::t": 1.0}, set())


def test_staleness_threshold_is_a_parameter() -> None:
    assert (
        md.staleness_message({"t.py::0": 1.0}, {"t.py::0", "t.py::1"}, threshold=0.6)
        is None
    )
    assert (
        md.staleness_message({"t.py::0": 1.0}, {"t.py::0", "t.py::1"}, threshold=0.4)
        is not None
    )


# --- command line ------------------------------------------------------------------------------------------------


def test_cli_merge_writes_the_file_and_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _write(tmp_path / "old.json", {"a.py::gone": 1.0, "a.py::t": 1.0})
    one = _write(tmp_path / "s1.json", {"a.py::t": 3.0})
    two = _write(tmp_path / "s2.json", {"b.py::u": 1.0})
    out, summary = tmp_path / "new.json", tmp_path / "summary.md"
    status = md.main([
        "merge",
        "--committed",
        str(committed),
        "--out",
        str(out),
        "--summary",
        str(summary),
        str(one),
        str(two),
    ])
    assert status == 0
    assert out.read_text() == '{\n    "a.py::t": 3.0,\n    "b.py::u": 1.0\n}\n'
    assert "| dropped (no longer run) | 1 |" in summary.read_text()
    assert "| tests in the file | 2 |" in capsys.readouterr().out


def test_cli_merge_error_is_status_two_names_the_file_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _write(tmp_path / "old.json", {"a.py::t": 1.0})
    broken = tmp_path / "s1.json"
    broken.write_text("oops")
    out = tmp_path / "new.json"
    status = md.main([
        "merge",
        "--committed",
        str(committed),
        "--out",
        str(out),
        str(broken),
    ])
    assert status == 2
    assert "s1.json" in capsys.readouterr().err
    assert not out.exists()


def test_cli_stale_warns_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _write(tmp_path / "old.json", {"t.py::0": 1.0})
    collected = tmp_path / "collected.txt"
    collected.write_text("t.py::0\nt.py::1\nt.py::2\n3 tests collected in 1s\n")
    assert (
        md.main(["stale", "--committed", str(committed), "--collected", str(collected)])
        == 0
    )
    assert "::warning title=Stale test durations::2 of 3" in capsys.readouterr().out


def test_cli_stale_says_so_when_the_file_is_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _write(tmp_path / "old.json", {"t.py::0": 1.0})
    collected = tmp_path / "collected.txt"
    collected.write_text("t.py::0\n")
    assert (
        md.main(["stale", "--committed", str(committed), "--collected", str(collected)])
        == 0
    )
    assert "up to date" in capsys.readouterr().out


def test_cli_stale_refuses_a_missing_or_empty_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed = _write(tmp_path / "old.json", {"t.py::0": 1.0})
    assert (
        md.main([
            "stale",
            "--committed",
            str(committed),
            "--collected",
            str(tmp_path / "none.txt"),
        ])
        == 2
    )
    empty = tmp_path / "collected.txt"
    empty.write_text("no tests ran\n")
    assert (
        md.main(["stale", "--committed", str(committed), "--collected", str(empty)])
        == 2
    )
    assert capsys.readouterr().err.count("merge_durations:") == 2
