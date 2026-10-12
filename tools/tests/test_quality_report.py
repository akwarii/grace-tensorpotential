"""Logic tests for tools/quality_report.py (no physics in this tool).

Trees are written to ``tmp_path`` and the numbers expected are counted by hand from the sources below. The
lint measurement is replaced by a fixed one in the end-to-end tests (its real run is the job of
``test_lint_ratchet.py``); one test runs it for real on a scratch project to check the field names this tool
reads. Coverage reports are built the way coverage.py writes them.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quality_report as qr

REPO = Path(__file__).resolve().parents[2]

API = '''\
def public(a) -> int:
    """Documented."""
    return a


def no_return(a):
    """Documented."""
    return a


def no_doc(a) -> int:
    return a


def _private(a) -> int:
    return a


async def coro(a) -> int:
    """Documented."""
    return a


def outer(a) -> int:
    """Documented."""

    def nested(b):
        return b

    return nested(a)


class Shown:
    def method(self) -> int:
        return 1

    def other(self):
        """Documented."""

    def _hidden(self):
        return 1


class _Hidden:
    def method(self):
        return 1
'''
# public: public, no_return, no_doc, coro, outer, Shown.method, Shown.other = 7
# with a return annotation: public, no_doc, coro, outer, Shown.method = 5
# with a docstring: public, no_return, coro, outer, Shown.other = 5

PACE = "def old(a):\n    return a  # TODO: remove\n"


def func(name: str, *, var: str = "x") -> str:
    """A function of eight lines, so that two of them form a clone group."""
    return f"def {name}(a, b=1):\n" + f"    {var} = a + b\n" * 6 + f"    return {var}\n"


def report(files: dict[str, tuple[int, int, int, int]], *, branch: bool = True) -> dict:
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


COVERAGE = {
    "tensorpotential/data/a.py": (8, 10, 3, 4),
    "tensorpotential/data/b.py": (5, 5, 0, 0),
    "tensorpotential/loss.py": (6, 10, 0, 0),
    "tensorpotential/compat/pace/p.py": (0, 10, 0, 0),
    "tests/test_a.py": (10, 10, 0, 0),
}


def write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path``, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small repository: a library, its tests, a ledger of two rows and the legacy list."""
    write(tmp_path / "tensorpotential" / "api.py", API)
    write(
        tmp_path / "tensorpotential" / "data" / "d.py",
        "def f(a) -> int:\n    return a  # TODO(#1) later\n",
    )
    write(tmp_path / "tensorpotential" / "compat" / "pace" / "p.py", PACE)
    write(tmp_path / "tests" / "test_x.py", "def test_x():\n    assert True\n")
    write(
        tmp_path / "tools" / "divergence.yaml",
        "- {path: a.py, reason: r, milestone: m, upstream: pending}\n- {path: b.py, reason: r, milestone: m, upstream: never}\n",
    )
    write(tmp_path / "baselines" / "legacy_files.txt", "")
    return tmp_path


@pytest.fixture
def lint(monkeypatch: pytest.MonkeyPatch) -> dict:
    """The lint measurement, replaced by a state the test can change."""
    state: dict = {
        "ruff": {"a.py": {"C901": 2, "F401": 1}, "b.py": {"C901": 1}},
        "ty": {"a.py": {"not-iterable": 4}},
        "strict": [],
    }
    monkeypatch.setattr(qr.lint_ratchet, "read_legacy", lambda _path: [])
    monkeypatch.setattr(qr.lint_ratchet, "measure", lambda _root, _legacy: state)
    return state


def run(root: Path, *args: str) -> int:
    """Run the command line on ``root`` and return the exit status."""
    return qr.main([*args, "--root", str(root)])


def append(path: Path, text: str) -> None:
    """Append ``text`` to ``path``."""
    with path.open("a") as out:
        out.write(text)


OK = "quality report: ok\n"
JUNIT = '<testsuite><testcase name="a" time="2"/></testsuite>'


def save(root: Path, cov: dict | None = None) -> Path:
    """Write a coverage report below ``root``."""
    return write(root / "cov.json", json.dumps(report(cov or COVERAGE)))


# --- logic: the collectors ------------------------------------------------------------------


def test_public_functions_are_counted_through_public_names_only(tree: Path) -> None:
    metrics = qr.api_metrics(
        tree
    )  # api.py has 7 public functions, d.py one without a docstring
    assert metrics["api.return_annotations"] == qr.ratio(5 + 1, 8)
    assert metrics["api.docstrings"] == qr.ratio(5, 8)


def test_the_excluded_package_is_left_out_of_every_library_metric(tree: Path) -> None:
    names = [p.relative_to(tree).as_posix() for p in qr.library_files(tree)]
    assert names == ["tensorpotential/api.py", "tensorpotential/data/d.py"]
    assert qr.todo_metrics(tree) == {
        "todo.markers": qr.count(1)
    }  # the marker of pace is not counted


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("# TODO(#12) later", 1),
        ("# FIXME: x", 1),
        ("x = 1  # XXX", 1),
        ("# TODOS and XXXX and fixme", 0),
        ("# nothing", 0),
    ],
)
def test_a_marker_is_a_whole_word_in_capitals(
    tree: Path, line: str, expected: int
) -> None:
    write(tree / "tensorpotential" / "api.py", line + "\n")
    write(tree / "tensorpotential" / "data" / "d.py", "")
    assert qr.todo_metrics(tree)["todo.markers"].value == expected


def test_the_divergence_metric_counts_the_rows_of_the_ledger(tree: Path) -> None:
    assert qr.divergence_metrics(tree) == {"divergence.upstream_files": qr.count(2)}


def test_clones_of_a_tree_are_counted_and_an_empty_tree_is_zero(tree: Path) -> None:
    write(tree / "tensorpotential" / "c.py", func("a") + "\n\n" + func("b"))
    metrics = qr.clone_metrics(tree)
    assert metrics["clones.library.groups"] == qr.count(1)
    assert metrics["clones.library.functions"] == qr.count(2)
    assert metrics["clones.tests.groups"] == qr.count(0)
    assert set(metrics) == {
        f"clones.{t}.{m}"
        for t in ("library", "tests")
        for m in ("groups", "functions", "redundant_lines")
    }


def test_clone_kinds_are_summed(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kinds = {
        "identical": {"groups": 1, "functions": 3, "redundant_lines": 10},
        "renamed": {"groups": 2, "functions": 4, "redundant_lines": 5},
    }
    monkeypatch.setattr(qr.check_clones, "measure", lambda _root: {"library": kinds})
    assert qr.clone_metrics(tree) == {
        "clones.library.groups": qr.count(3),
        "clones.library.functions": qr.count(7),
        "clones.library.redundant_lines": qr.count(15),
    }


def test_coverage_is_summed_per_package_and_in_total_without_pace_and_tests() -> None:
    metrics = qr.coverage_metrics(
        qr.coverage_ratchet.counts_from_report(report(COVERAGE))
    )
    assert metrics == {
        "coverage.data": qr.ratio(11 + 5, 14 + 5),
        "coverage.toplevel": qr.ratio(6, 10),
        "coverage.total": qr.ratio(11 + 5 + 6, 14 + 5 + 10),
    }


def args_of(**kwargs: object) -> argparse.Namespace:
    """The coverage options of the command line."""
    return argparse.Namespace(**{"coverage": None, "ratchet_coverage": False, **kwargs})


def test_a_coverage_report_without_branches_is_refused(tree: Path) -> None:
    cov = write(tree / "c.json", json.dumps(report(COVERAGE, branch=False)))
    with pytest.raises(qr.coverage_ratchet.RatchetError, match="branch"):
        qr.coverage_counts(tree, args_of(coverage=cov))


def test_the_counts_come_from_a_report_or_from_the_coverage_baseline_or_not_at_all(
    tree: Path,
) -> None:
    cov = write(
        tree / "c.json", json.dumps(report({"tensorpotential/a.py": (8, 10, 3, 4)}))
    )
    assert qr.coverage_counts(tree, args_of(coverage=cov)) == {
        "tensorpotential/a.py": {"covered": 11, "total": 14}
    }
    counts = {"tensorpotential/b.py": {"covered": 1, "total": 2}}
    write(tree / "baselines" / "coverage_baseline.json", json.dumps({"files": counts}))
    assert qr.coverage_counts(tree, args_of(ratchet_coverage=True)) == counts
    assert qr.coverage_counts(tree, args_of()) is None


def test_lint_findings_are_summed_per_tool_and_rule(lint: dict) -> None:
    lint["strict"] = ["x.py: D100", "y.py: D100"]
    assert qr.lint_metrics(Path()) == {
        "lint.strict": qr.count(2),
        "lint.ruff.C901": qr.count(3),
        "lint.ruff.F401": qr.count(1),
        "lint.ty.not-iterable": qr.count(4),
    }


def test_junit_metrics_sum_cases_and_time_over_the_files(tmp_path: Path) -> None:
    one = write(
        tmp_path / "1.xml",
        '<testsuites><testsuite><testcase name="a" time="1.5"/><testcase name="b" time="2"/></testsuite></testsuites>',
    )
    two = write(tmp_path / "2.xml", '<testsuite><testcase name="c"/></testsuite>')
    assert qr.junit_metrics([one, two]) == {
        "tests.count": qr.Metric("info", 3),
        "tests.seconds": qr.Metric("info", 3.5),
    }


def test_mutation_scores_are_read_as_numbers_or_objects(tmp_path: Path) -> None:
    text = '{"modules": {"loss.py": 0.8, "tpmodel.py": {"score": 0.6}}}'
    metrics = qr.mutation_metrics(write(tmp_path / "m.json", text))
    assert metrics == {
        "mutation.loss.py": qr.Metric("score", 0.8),
        "mutation.tpmodel.py": qr.Metric("score", 0.6),
    }


@pytest.mark.parametrize(
    "text",
    ["{}", '{"modules": 3}', '{"modules": {"a": "x"}}', '{"modules": {"a": {}}}'],
)
def test_a_mutation_file_that_is_not_a_score_per_module_is_an_error(
    tmp_path: Path, text: str
) -> None:
    with pytest.raises(qr.ReportError, match="mutation score"):
        qr.mutation_metrics(write(tmp_path / "m.json", text))


def test_lint_metrics_read_the_real_measure_on_a_scratch_project(
    tmp_path: Path,
) -> None:
    text = (REPO / "pyproject.toml").read_text()
    write(
        tmp_path / "pyproject.toml",
        text[text.index("[tool.ruff]") : text.index("[dependency-groups]")]
        + f'[tool.ty.environment]\npython = "{sys.prefix}"\n',
    )
    write(
        tmp_path / "legacy" / "mod.py",
        "def f(a: int = None):\n    try:\n        pass\n    except:\n        pass\n",
    )
    write(tmp_path / "baselines" / "legacy_files.txt", "legacy/mod.py\n")
    write(
        tmp_path / "tests_torch" / "test_bad.py",
        "def f(a):\n    try:\n        pass\n    except:\n        pass\n",
    )
    metrics = qr.lint_metrics(tmp_path)
    # the legacy file is counted by rule, the file of the strict package apart
    assert metrics["lint.ruff.E722"] == qr.count(1)
    assert metrics["lint.ty.invalid-parameter-default"] == qr.count(1)
    assert metrics["lint.strict"].value >= 1


# --- logic: the metric, the comparison and the table ----------------------------------------


@pytest.mark.parametrize(
    "metric",
    [qr.count(3), qr.ratio(2, 5), qr.Metric("score", 0.5), qr.Metric("info", 4)],
)
def test_a_metric_survives_its_json_form(metric: qr.Metric) -> None:
    assert qr.Metric.from_json(metric.to_json()) == metric


def test_a_ratio_stores_its_parts_and_shows_a_share() -> None:
    assert "value" not in qr.ratio(2, 5).to_json()
    assert qr.ratio(1, 4).shown() == "25.00% (1/4)"
    assert qr.count(3).shown() == "3"
    assert qr.ratio(0, 0).share == 1.0


SCORE = qr.Metric("score", 0.8)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (qr.count(3), qr.count(4), True),
        (qr.count(3), qr.count(3), False),
        (qr.count(3), qr.count(2), False),
        (None, qr.count(1), True),  # a rule that was absent counted zero
        (None, qr.count(0), False),
        (SCORE, qr.Metric("score", 0.7), True),
        (SCORE, SCORE, False),
        (None, SCORE, False),
        (qr.ratio(8, 10), qr.ratio(8, 11), True),  # share fell, one more missing
        (qr.ratio(8, 10), qr.ratio(7, 8), False),  # the share rose
        (
            qr.ratio(8, 10),
            qr.ratio(7, 9),
            False,
        ),  # share fell, the missing part did not grow
        (qr.ratio(8, 10), qr.ratio(80, 100), False),
        (qr.ratio(8, 10), qr.ratio(79, 100), True),  # 79% and 21 missing against 2
        (None, qr.ratio(1, 2), False),
        (
            qr.Metric("info", 1),
            qr.Metric("info", 9),
            False,
        ),  # reported, never ratcheted
        (None, qr.Metric("info", 9), False),
    ],
)
def test_worse_follows_the_direction_of_each_kind(
    old: qr.Metric | None, new: qr.Metric, expected: bool
) -> None:
    assert qr.worse(old, new) is expected


def test_regressions_skip_info_and_the_groups_that_were_not_measured() -> None:
    base = {
        "todo.markers": qr.count(1),
        "coverage.data": qr.ratio(9, 10),
        "lint.strict": qr.count(0),
        "tests.count": qr.Metric("info", 1),
    }
    now = {
        "todo.markers": qr.count(2),
        "coverage.data": qr.ratio(1, 10),
        "tests.count": qr.Metric("info", 0),
    }
    assert qr.regressions(base, now, {"todo"}) == ["todo.markers: 1 -> 2"]
    assert qr.regressions(base, now, {"coverage"}) == [
        "coverage.data: 90.00% (9/10) -> 10.00% (1/10)"
    ]
    assert qr.regressions(base, now, {"tests"}) == []
    assert qr.regressions(base, now, set()) == []


def test_a_new_count_above_zero_is_a_regression_and_a_gone_count_is_not() -> None:
    assert qr.regressions({}, {"lint.ruff.B1": qr.count(2)}, {"lint"}) == [
        "lint.ruff.B1: 0 -> 2"
    ]
    assert qr.regressions({"lint.ruff.B1": qr.count(2)}, {}, {"lint"}) == []


def test_the_table_has_a_section_per_group_and_the_baseline_value() -> None:
    now = {
        "todo.markers": qr.count(2),
        "api.docstrings": qr.ratio(1, 2),
        "api.x": qr.count(0),
    }
    text = qr.table(now, {"todo.markers": qr.count(1)})
    assert text.splitlines() == [
        "| metric | now | baseline |",
        "|---|---|---|",
        "| **api** | | |",
        "| api.docstrings | 50.00% (1/2) |  |",
        "| api.x | 0 |  |",
        "| **todo** | | |",
        "| todo.markers | 2 | 1 |",
    ]


# --- logic: the command line, with the planted regressions of the issue ---------------------


def recorded(root: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> Path:
    """Record the baseline of ``root`` and return the coverage report it was made from."""
    cov = save(root)
    assert run(root, "record", "--coverage", str(cov), *extra) == 0
    capsys.readouterr()
    return cov


def check(
    root: Path, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    """Run ``check`` and return its exit status and what it printed."""
    status = run(root, "check", *args)
    return status, capsys.readouterr().out


@pytest.mark.usefixtures("lint")
def test_the_tree_passes_its_own_baseline(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    assert check(tree, capsys, "--coverage", str(cov)) == (0, OK)
    stored = json.loads((tree / "baselines" / "quality.json").read_text())["metrics"]
    assert stored["todo.markers"] == {"kind": "count", "value": 1.0}
    assert stored["coverage.total"] == {"kind": "ratio", "have": 22, "total": 29}


@pytest.mark.usefixtures("lint")
def test_one_more_marker_fails_and_names_the_metric(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    append(tree / "tensorpotential" / "api.py", "# TODO(#2) again\n")
    assert check(tree, capsys, "--coverage", str(cov)) == (
        1,
        "worse: todo.markers: 1 -> 2\n",
    )


@pytest.mark.usefixtures("lint")
def test_one_lower_annotation_share_fails_and_names_the_metric(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    append(tree / "tensorpotential" / "api.py", "\n\ndef added(a):\n    return a\n")
    status, out = check(tree, capsys, "--coverage", str(cov))
    assert status == 1
    assert "worse: api.return_annotations: 75.00% (6/8) -> 66.67% (6/9)" in out
    assert "worse: api.docstrings" in out


@pytest.mark.usefixtures("lint")
def test_one_more_clone_group_fails_and_names_the_metric(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    write(tree / "tensorpotential" / "c.py", func("a") + "\n\n" + func("b"))
    status, out = check(tree, capsys, "--coverage", str(cov))
    assert status == 1
    assert "worse: clones.library.groups: 0 -> 1" in out
    assert "worse: clones.library.functions: 0 -> 2" in out


@pytest.mark.usefixtures("lint")
def test_one_lower_coverage_file_fails_and_names_the_package(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded(tree, capsys)
    lower = dict(COVERAGE, **{"tensorpotential/data/b.py": (2, 5, 0, 0)})
    status, out = check(tree, capsys, "--coverage", str(save(tree, lower)))
    assert status == 1
    assert "worse: coverage.data: 84.21% (16/19) -> 68.42% (13/19)" in out
    assert "worse: coverage.total" in out


@pytest.mark.usefixtures("lint")
def test_a_removed_covered_file_is_not_a_coverage_regression(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded(tree, capsys)
    smaller = {k: v for k, v in COVERAGE.items() if k != "tensorpotential/data/b.py"}
    assert (
        check(tree, capsys, "--coverage", str(save(tree, smaller)))[0] == 0
    )  # 11/14 = 78.6% < 84.2%, but the same 3 missing


def test_a_lint_rise_and_a_strict_finding_fail(
    tree: Path, lint: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    lint["ruff"]["c.py"] = {"BLE001": 1}
    lint["strict"] = ["d.py: D100"]
    status, out = check(tree, capsys, "--coverage", str(cov))
    assert status == 1
    assert "worse: lint.ruff.BLE001: 0 -> 1" in out
    assert "worse: lint.strict: 0 -> 1" in out


def test_a_lint_drop_passes_and_lowers_the_baseline_at_record(
    tree: Path, lint: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    lint["ruff"]["a.py"]["C901"] = 0
    assert check(tree, capsys, "--coverage", str(cov))[0] == 0
    assert run(tree, "record", "--coverage", str(cov)) == 0
    stored = json.loads((tree / "baselines" / "quality.json").read_text())["metrics"]
    assert stored["lint.ruff.C901"]["value"] == 1.0


@pytest.mark.usefixtures("lint")
def test_one_more_upstream_file_fails(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    append(
        tree / "tools" / "divergence.yaml",
        "- {path: c.py, reason: r, milestone: m, upstream: pending}\n",
    )
    assert check(tree, capsys, "--coverage", str(cov)) == (
        1,
        "worse: divergence.upstream_files: 2 -> 3\n",
    )


@pytest.mark.usefixtures("lint")
def test_a_lower_mutation_score_fails_and_a_missing_file_is_not_measured(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mutation = tree / "baselines" / "mutation_score.json"
    write(mutation, '{"modules": {"loss.py": 0.8}}')
    cov = recorded(tree, capsys)
    write(mutation, '{"modules": {"loss.py": 0.7}}')
    assert check(tree, capsys, "--coverage", str(cov)) == (
        1,
        "worse: mutation.loss.py: 0.8 -> 0.7\n",
    )
    mutation.unlink()
    assert check(tree, capsys, "--coverage", str(cov))[0] == 0


def test_record_refuses_a_worse_value_unless_allowed(
    tree: Path, lint: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    cov = recorded(tree, capsys)
    lint["ruff"]["a.py"]["C901"] = 5
    assert run(tree, "record", "--coverage", str(cov)) == 1
    assert (
        "refusing to record a worse value (use --allow-worse):\nlint.ruff.C901: 3 -> 6"
        in capsys.readouterr().out
    )
    assert run(tree, "record", "--coverage", str(cov), "--allow-worse") == 0
    assert qr.load_baseline(tree / "baselines" / "quality.json")[
        "lint.ruff.C901"
    ] == qr.count(6)


def test_record_without_coverage_keeps_the_coverage_entries(
    tree: Path, lint: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded(tree, capsys)
    before = qr.load_baseline(tree / "baselines" / "quality.json")
    lint["ruff"]["a.py"]["C901"] = 0
    assert run(tree, "record", "--no-coverage") == 0
    capsys.readouterr()
    after = qr.load_baseline(tree / "baselines" / "quality.json")
    kept = {n: m for n, m in before.items() if n.startswith("coverage.")}
    assert kept
    assert {n: m for n, m in after.items() if n.startswith("coverage.")} == kept
    assert after["lint.ruff.C901"] == qr.count(1)
    assert check(tree, capsys, "--no-coverage") == (0, OK)


@pytest.mark.usefixtures("lint")
def test_the_first_record_without_a_baseline_and_a_check_without_one(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tree, "check", "--no-coverage") == 2
    assert "no baseline" in capsys.readouterr().err
    assert run(tree, "record", "--no-coverage") == 0
    assert not [
        n
        for n in qr.load_baseline(tree / "baselines" / "quality.json")
        if n.startswith("coverage.")
    ]


@pytest.mark.usefixtures("lint")
def test_report_prints_the_table_and_writes_json_and_markdown(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit = write(tree / "j.xml", JUNIT)
    assert run(tree, "report", "--no-coverage", "--junit", str(junit)) == 0
    out = capsys.readouterr().out
    assert "| todo.markers | 1 |  |" in out
    assert "| tests.count | 1 |  |" in out
    summary, data = tree / "summary.md", tree / "q.json"
    summary.write_text("earlier\n")
    assert (
        run(
            tree,
            "report",
            "--no-coverage",
            "--markdown",
            str(summary),
            "--json",
            str(data),
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert summary.read_text().startswith("earlier\n## Quality report\n\n| metric |")
    assert json.loads(data.read_text())["metrics"]["todo.markers"] == {
        "kind": "count",
        "value": 1.0,
    }


@pytest.mark.usefixtures("lint")
def test_info_metrics_are_reported_and_never_recorded_or_compared(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junit = write(tree / "j.xml", JUNIT)
    assert run(tree, "record", "--no-coverage", "--junit", str(junit)) == 0
    capsys.readouterr()
    write(
        junit,
        '<testsuite><testcase name="a" time="9"/><testcase name="b"/></testsuite>',
    )
    assert check(tree, capsys, "--no-coverage", "--junit", str(junit)) == (0, OK)
    assert "tests.count" not in qr.load_baseline(tree / "baselines" / "quality.json")


@pytest.mark.usefixtures("lint")
@pytest.mark.usefixtures("lint")
@pytest.mark.parametrize(
    ("path", "text", "message"),
    [
        ("absent.json", None, "cannot read"),
        ("nobranch.json", json.dumps(report(COVERAGE, branch=False)), "branch"),
    ],
)
def test_a_coverage_report_that_cannot_be_used_is_an_error_not_a_pass(
    tree: Path,
    capsys: pytest.CaptureFixture[str],
    path: str,
    text: str | None,
    message: str,
) -> None:
    if text is not None:
        write(tree / path, text)
    assert run(tree, "report", "--coverage", str(tree / path)) == 2
    assert message in capsys.readouterr().err


@pytest.mark.usefixtures("lint")
def test_a_mutation_file_that_cannot_be_used_is_an_error_not_a_skip(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tree / "baselines" / "mutation_score.json", "[]")
    assert run(tree, "report", "--no-coverage") == 2
    assert "mutation score" in capsys.readouterr().err


def test_a_coverage_choice_is_required(tree: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        run(tree, "check")
    assert exc.value.code == 2


@pytest.mark.usefixtures("lint")
def test_the_baseline_can_be_recorded_from_the_counts_of_the_coverage_ratchet(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    counts = qr.coverage_ratchet.counts_from_report(report(COVERAGE))
    write(tree / "baselines" / "coverage_baseline.json", json.dumps({"files": counts}))
    assert run(tree, "record", "--ratchet-coverage") == 0
    capsys.readouterr()
    assert qr.load_baseline(tree / "baselines" / "quality.json")[
        "coverage.total"
    ] == qr.ratio(22, 29)
    assert check(tree, capsys, "--coverage", str(save(tree))) == (0, OK)


@pytest.mark.usefixtures("lint")
@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("tools/divergence.yaml", "not a list\n"),
        ("tools/divergence.yaml", "- [unclosed\n"),
        ("baselines/quality.json", "{}"),
        ("baselines/quality.json", '{"metrics": {"a.b": {"value": 1}}}'),
    ],
)
def test_a_ledger_or_baseline_that_cannot_be_read_is_an_error_not_a_regression(
    tree: Path, capsys: pytest.CaptureFixture[str], path: str, text: str
) -> None:
    write(tree / path, text)
    assert run(tree, "check", "--no-coverage") == 2
    assert capsys.readouterr().err.startswith("quality_report: ")
