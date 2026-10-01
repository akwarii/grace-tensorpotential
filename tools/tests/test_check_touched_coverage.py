"""Tests for tools/check_touched_coverage.py (logic layer; the numbers are counted by hand).

The physics layer does not apply to a coverage tool. Its oracle is a hand count: each
module below is small enough to list its statements and branches, and coverage.py is run
for real (``coverage run --branch``) on throwaway git repositories, so no report is mocked.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_touched_coverage as ctc

CLASSIFY = """\
def classify(x):
    if x > 0:
        return "pos"
    elif x < 0:
        return "neg"
    return "zero"
"""
# classify: 6 statements (def, if, 3 returns, elif) and 4 branches (2 per test) = 10 items.
# Called with 1 only: statements def, if, return pos = 3; branch if->pos = 1: 4/10.
# Called with 1 and -1: statements 5, branches if->pos, if->elif, elif->neg = 3: 8/10.
# Called with 1, -1, 0: 10/10.

IDENT = "def ident(x):\n    return x\n"


def run(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(  # noqa: S603
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


class Repo:
    """A throwaway git repository with a ``pkg`` package and a coverage runner."""

    def __init__(self, root: Path) -> None:
        self.root = root
        run(["git", "init", "-q", "-b", "main"], root)
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("")

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text) if text.startswith("\n") else text)

    def commit(self) -> None:
        run(["git", "add", "-A"], self.root)
        run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "commit",
                "-qm",
                "c",
            ],
            self.root,
        )

    def coverage(self, driver: str) -> dict:
        """Run ``driver`` under coverage with branches; return the JSON report."""
        (self.root / "driver.py").write_text(driver)
        for args in (
            ["run", "--branch", "--source=pkg", "driver.py"],
            ["json", "-q", "-o", "cov.json"],
        ):
            run([sys.executable, "-m", "coverage", *args], self.root)
        return json.loads((self.root / "cov.json").read_text())

    def check(self, driver: str, *extra: str, capsys=None) -> tuple[int, str]:
        self.coverage(driver)
        code = ctc.main([
            "--coverage",
            str(self.root / "cov.json"),
            "--base",
            "HEAD",
            "--root",
            str(self.root),
            *extra,
        ])
        return code, capsys.readouterr().out if capsys else ""


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.write("pkg/mod.py", CLASSIFY + "\n\n" + IDENT)
    r.commit()
    return r


def percent_of(out: str, qualname: str) -> float:
    for line in out.splitlines():
        if f"::{qualname} " in line:
            return float(line.split("%")[0].split()[-1])
    raise AssertionError(f"{qualname} not in output:\n{out}")


@pytest.mark.parametrize(
    ("calls", "expected"),
    [("1", 40.0), ("1, -1", 80.0), ("1, -1, 0", 100.0)],
)
def test_unit_percentages_match_the_hand_count(repo, capsys, calls, expected):
    driver = "from pkg.mod import classify\n" + "\n".join(
        f"classify({c})" for c in calls.split(", ")
    )
    code, out = repo.check(driver, "--unit", "pkg/mod.py::classify", capsys=capsys)
    assert percent_of(out, "classify") == pytest.approx(expected)
    assert code == (0 if expected >= 90 else 1)


def test_threshold_option_moves_the_pass_line(repo, capsys):
    driver = "from pkg.mod import classify\nclassify(1)\nclassify(-1)\n"
    code, _ = repo.check(driver, "--unit", "pkg/mod.py::classify", capsys=capsys)
    assert code == 1
    code, out = repo.check(
        driver, "--unit", "pkg/mod.py::classify", "--threshold", "80", capsys=capsys
    )
    assert code == 0
    assert out.startswith("ok  ")


def test_only_the_edited_function_is_checked(repo, capsys):
    repo.write(
        "pkg/mod.py",
        (repo.root / "pkg/mod.py").read_text().replace("return x", "return -x"),
    )
    code, out = repo.check("from pkg.mod import ident\nident(2)\n", capsys=capsys)
    assert code == 0
    assert "::ident " in out
    assert "classify" not in out
    assert "1 unit(s) checked" in out


def test_untested_edited_function_turns_red(repo, capsys):
    repo.write(
        "pkg/mod.py",
        (repo.root / "pkg/mod.py").read_text().replace("return x", "return -x"),
    )
    code, out = repo.check("import pkg.mod\n", capsys=capsys)  # imported, never called
    assert code == 1
    assert out.startswith("FAIL")
    assert percent_of(out, "ident") == pytest.approx(50.0)  # only the def line ran


def test_file_unknown_to_coverage_counts_as_zero(repo, capsys):
    repo.write(
        "pkg/mod.py",
        (repo.root / "pkg/mod.py").read_text().replace("return x", "return -x"),
    )
    repo.coverage("pass\n")  # pkg.mod never imported
    report = {"files": {}}
    results, problems = ctc.collect(repo.root, "HEAD", report)
    assert [r.percent for r in results] == [0.0]
    assert [r.total for r in results] == [2]  # the statements of ident: def and return
    assert problems == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace("def ident(x):", "def ident(x):  # now commented"),
        lambda s: s.replace("def ident(x):\n", 'def ident(x):\n    """Docstring."""\n'),
        lambda s: s.replace("def ident(x):", "def ident(x: int) -> int:"),
        lambda s: s + "\n\n# trailing comment\n",
    ],
    ids=["comment", "docstring", "annotation", "trailing-comment"],
)
def test_comment_docstring_annotation_edits_are_exempt(repo, capsys, edit):
    repo.write("pkg/mod.py", edit((repo.root / "pkg/mod.py").read_text()))
    code, out = repo.check("import pkg.mod\n", capsys=capsys)
    assert code == 0
    assert "0 unit(s) checked" in out


def test_new_function_is_gated(repo, capsys):
    text = (
        repo.root / "pkg/mod.py"
    ).read_text() + "\n\ndef fresh(x):\n    return x + 1\n"
    repo.write("pkg/mod.py", text)
    code, out = repo.check("import pkg.mod\n", capsys=capsys)
    assert code == 1
    assert "::fresh " in out
    assert "::ident " not in out


def test_new_untracked_file_is_gated(repo, capsys):
    repo.write("pkg/new.py", "def one():\n    return 1\n")
    code, out = repo.check("from pkg.new import one\none()\n", capsys=capsys)
    assert code == 0
    assert percent_of(out, "one") == pytest.approx(100.0)


def test_deleted_function_is_not_checked(repo, capsys):
    repo.write("pkg/mod.py", CLASSIFY)
    code, out = repo.check("import pkg.mod\n", capsys=capsys)
    assert code == 0
    assert "0 unit(s) checked" in out


NESTED = """\
def outer(x):
    def inner(y):
        if y:
            return 1
        return 2
    return inner(x)
"""
# outer owns lines 1 (def), 2 (the header of inner, executed whenever outer runs) and 6:
# 3 statements, no branch. inner owns its header (line 2) and its body (lines 3-5):
# def, if, return 1, return 2 = 4 statements, plus 2 branches of the if = 6 items.
# outer(1) runs def, def inner, return and inner's def, if, return 1 and branch if->return 1.


def test_nested_function_is_its_own_unit(tmp_path, capsys):
    r = Repo(tmp_path)
    r.write("pkg/n.py", NESTED)
    r.commit()
    r.write("pkg/n.py", NESTED.replace("return 2", "return 3"))
    code, out = r.check("from pkg.n import outer\nouter(1)\n", capsys=capsys)
    assert "::outer.inner " in out
    assert "::outer " not in out  # outer's own code did not change
    assert percent_of(out, "outer.inner") == pytest.approx(
        100.0 * 4 / 6, abs=0.05
    )  # y=1 only: 3 statements, 1 branch
    assert code == 1


def test_enclosing_function_excludes_nested_body(tmp_path, capsys):
    r = Repo(tmp_path)
    r.write("pkg/n.py", NESTED)
    r.commit()
    r.write("pkg/n.py", NESTED.replace("return inner(x)", "return inner(x) + 0"))
    code, out = r.check("from pkg.n import outer\nouter(1)\n", capsys=capsys)
    assert code == 0
    assert "::outer " in out
    assert "::outer.inner " not in out
    assert percent_of(out, "outer") == pytest.approx(100.0)  # 3 of 3, no branch


CLS = """\
class Box:
    size = 2

    def area(self):
        return self.size**2

    def label(self):
        return "box"
"""


def test_method_edit_checks_only_the_method(tmp_path, capsys):
    r = Repo(tmp_path)
    r.write("pkg/c.py", CLS)
    r.commit()
    r.write("pkg/c.py", CLS.replace("**2", "**3"))
    code, out = r.check("from pkg.c import Box\nBox().area()\n", capsys=capsys)
    assert code == 0
    assert "::Box.area " in out
    assert "::Box " not in out
    assert "::Box.label " not in out


def test_class_attribute_edit_checks_the_class_body(tmp_path, capsys):
    r = Repo(tmp_path)
    r.write("pkg/c.py", CLS)
    r.commit()
    r.write("pkg/c.py", CLS.replace("size = 2", "size = 3"))
    code, out = r.check("import pkg.c\n", capsys=capsys)
    assert code == 0  # class line, attribute and two method headers all run at import
    assert "::Box " in out
    assert "::Box.area " not in out
    assert percent_of(out, "Box") == pytest.approx(100.0)


PROPERTY = """\
class P:
    @property
    def v(self):
        return 1

    @v.setter
    def v(self, value):
        pass
"""


def test_repeated_names_get_numbered_keys():
    units = ctc.units_of(PROPERTY)
    assert sorted(units) == ["P", "P.v", "P.v#2"]
    assert units["P.v"].fingerprint != units["P.v#2"].fingerprint


def test_decorator_line_belongs_to_the_unit():
    units = ctc.units_of(PROPERTY)
    assert units["P.v"].first == 2  # the @property line, not the def line
    assert 2 in units["P.v"].lines


def test_pragma_no_cover_lines_leave_the_denominator(tmp_path, capsys):
    r = Repo(tmp_path)
    r.write("pkg/p.py", "def f(x):\n    return x\n")
    r.commit()
    r.write(
        "pkg/p.py",
        "def f(x):\n    if x < 0:  # pragma: no cover\n        return 0\n    return x\n",
    )
    code, out = r.check("from pkg.p import f\nf(1)\n", capsys=capsys)
    assert code == 0
    assert "excluded" in out
    assert percent_of(out, "f") == pytest.approx(100.0)


def test_async_function_is_a_unit():
    units = ctc.units_of("async def g(x):\n    return x\n")
    assert list(units) == ["g"]


def test_unit_without_statements_counts_as_covered():
    unit = ctc.units_of("def g():\n    pass\n")["g"]
    result = ctc.unit_coverage("p.py", unit, {"excluded_lines": [1, 2]})
    assert (result.hit, result.total, result.excluded) == (0, 0, 2)
    assert result.percent == 100.0


def test_absolute_report_keys_are_found(repo, capsys):
    repo.write(
        "pkg/mod.py",
        (repo.root / "pkg/mod.py").read_text().replace("return x", "return -x"),
    )
    report = repo.coverage("from pkg.mod import ident\nident(1)\n")
    absolute = {"files": {str(repo.root / k): v for k, v in report["files"].items()}}
    results, problems = ctc.collect(repo.root, "HEAD", absolute)
    assert [r.percent for r in results] == [100.0]
    assert problems == []


def test_unknown_unit_is_an_error(repo, capsys):
    code, out = repo.check(
        "import pkg.mod\n", "--unit", "pkg/mod.py::nope", capsys=capsys
    )
    assert code == 1
    assert "ERROR pkg/mod.py::nope: no such unit" in out


def test_unparsable_file_is_an_error(repo, capsys):
    repo.coverage("pass\n")  # coverage itself cannot report a file that does not parse
    repo.write("pkg/mod.py", "def broken(:\n")
    code = ctc.main([
        "--coverage",
        str(repo.root / "cov.json"),
        "--base",
        "HEAD",
        "--root",
        str(repo.root),
    ])
    out = capsys.readouterr().out
    assert code == 1
    assert "ERROR pkg/mod.py: cannot parse" in out


def test_missing_file_named_by_unit_is_an_error(repo, capsys):
    code, out = repo.check("pass\n", "--unit", "pkg/absent.py::f", capsys=capsys)
    assert code == 1
    assert "pkg/absent.py: cannot parse" in out


@pytest.mark.parametrize(
    "path", ["tests/test_x.py", "tools/helper.py", "baselines/x.py"]
)
def test_tests_and_fork_only_paths_are_not_examined(repo, capsys, path):
    repo.write(path, "def helper():\n    return 1\n")
    code, out = repo.check("pass\n", capsys=capsys)
    assert code == 0
    assert "0 unit(s) checked" in out


def test_rename_counts_as_new_code(repo, capsys):
    run(["git", "mv", "pkg/mod.py", "pkg/moved.py"], repo.root)
    code, out = repo.check("import pkg.moved\n", capsys=capsys)
    assert "pkg/moved.py::classify" in out
    assert code == 1


def test_format_line_shape():
    unit = ctc.units_of("def g():\n    return 1\n")["g"]
    line = ctc.format_line(ctc.UnitCoverage("p.py", unit, 1, 2), 90.0)
    assert line == "FAIL  50.0%  p.py::g  (1/2, L1-L2)"
    line = ctc.format_line(ctc.UnitCoverage("p.py", unit, 2, 2, excluded=3), 90.0)
    assert line == "ok   100.0%  p.py::g  (2/2, L1-L2, 3 excluded)"


def test_units_of_rejects_invalid_source():
    with pytest.raises(SyntaxError):
        ctc.units_of("def (:\n")


def test_touched_units_none_means_all():
    new = ctc.units_of(IDENT)
    assert [u.qualname for u in ctc.touched_units(None, new)] == ["ident"]
    assert ctc.touched_units(new, new) == []


EXIT_BRANCH = "def g(x, acc):\n    if x:\n        acc.append(x)\n"
# g: statements def, if, append = 3; branches if->append and if->(leave the function) = 2.
# g(1): 3 statements and 1 branch run: 4/5. Adding g(0) takes the second branch: 5/5.


@pytest.mark.parametrize(
    ("calls", "expected"), [("g(1, [])", 80.0), ("g(1, [])\ng(0, [])", 100.0)]
)
def test_branch_leaving_the_function_counts(tmp_path, capsys, calls, expected):
    r = Repo(tmp_path)
    r.write("pkg/e.py", EXIT_BRANCH)
    r.commit()
    code, out = r.check(
        f"from pkg.e import g\n{calls}\n", "--unit", "pkg/e.py::g", capsys=capsys
    )
    assert percent_of(out, "g") == pytest.approx(expected)
    assert code == (0 if expected >= 90 else 1)
