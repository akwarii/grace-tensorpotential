"""Logic tests for tools/check_clones.py (no physics in this tool).

Projects are written to ``tmp_path`` with the two trees the tool scans (``tensorpotential`` and
``tests``); the expected group numbers are counted by hand from the sources below.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_clones as cc


def func(
    name: str, lines: int = 8, *, var: str = "x", const: int = 1, doc: str = ""
) -> str:
    """A function of ``lines`` lines in total (def line, optional docstring, body)."""
    head = f"def {name}(a, b={const}):\n" + (f'    """{doc}"""\n' if doc else "")
    used = 1 + (1 if doc else 0)
    body = f"    {var} = a + b\n" + f"    {var} = {var} * 2\n" * (lines - used - 2)
    return head + body + f"    return {var}\n"


def project(root: Path, library: dict[str, str], tests: dict[str, str]) -> Path:
    """Write ``library`` below ``tensorpotential/`` and ``tests`` below ``tests/``."""
    for tree, files in (("tensorpotential", library), ("tests", tests)):
        base = root / tree
        base.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            (base / name).write_text(text)
    return root


def lib_groups(root: Path) -> list[cc.Group]:
    return cc.groups(cc.functions(root, cc.TREES[0]))


def test_function_length_is_counted_from_def_to_last_line(tmp_path):
    src = func("seven", 7) + "\n" + func("eight", 8) + "\n" + func("nine", 9)
    root = project(tmp_path, {"m.py": src}, {})
    found = cc.functions(root, cc.TREES[0])
    assert [(f.where.split(":")[-1], f.lines) for f in found] == [
        ("eight", 8),
        ("nine", 9),
    ]


@pytest.mark.parametrize("stub", ["pass", "...", "pass\n    ..."])
def test_a_stub_with_a_long_signature_is_not_a_clone(tmp_path, stub):
    signature = "def f(\n" + "    a,\n" * 8 + "):\n"
    src = signature + f"    {stub}\n"
    root = project(tmp_path, {"a.py": src, "b.py": src.replace("def f", "def g")}, {})
    assert cc.functions(root, cc.TREES[0]) == []


def test_a_body_that_only_starts_with_pass_is_still_a_clone(tmp_path):
    src = "def f(a):\n    pass\n" + "    a = a + 1\n" * 6
    root = project(tmp_path, {"a.py": src, "b.py": src.replace("def f", "def g")}, {})
    (group,) = lib_groups(root)
    assert (group.kind, len(group.members)) == ("identical", 2)


def test_a_docstring_only_function_is_not_a_clone(tmp_path):
    src = "def f(\n" + "    a,\n" * 8 + '):\n    """Only a docstring."""\n'
    root = project(tmp_path, {"a.py": src, "b.py": src.replace("def f", "def g")}, {})
    assert cc.functions(root, cc.TREES[0]) == []


def test_docstring_counts_as_a_line_but_not_as_body(tmp_path):
    root = project(
        tmp_path, {"a.py": func("f", 8, doc="one"), "b.py": func("g", 8, doc="two")}, {}
    )
    (group,) = lib_groups(root)
    assert group.kind == "identical"
    assert [f.lines for f in group.members] == [8, 8]


def test_identical_group_ignores_function_name_and_signature(tmp_path):
    other = func("renamed", 8, const=5)  # a different default in the signature
    root = project(tmp_path, {"a.py": func("f"), "b.py": other}, {})
    (group,) = lib_groups(root)
    assert (group.kind, len(group.members)) == ("identical", 2)


def test_renamed_group_when_only_identifiers_differ(tmp_path):
    root = project(
        tmp_path, {"a.py": func("f", var="x"), "b.py": func("g", var="y")}, {}
    )
    (group,) = lib_groups(root)
    assert (group.kind, len(group.members)) == ("renamed", 2)


def test_constants_matter_in_the_library_but_not_in_the_test_tree(tmp_path):
    body_a = "def f(a):\n" + "    a = a + 1\n" * 7
    body_b = "def g(a):\n" + "    a = a + 2\n" * 7
    root = project(
        tmp_path,
        {"m.py": body_a + "\n" + body_b},
        {"test_m.py": body_a + "\n" + body_b},
    )
    assert lib_groups(root) == []
    (group,) = cc.groups(cc.functions(root, cc.TREES[1]))
    assert group.kind == "renamed"


def test_attributes_arguments_and_keywords_are_abstracted(tmp_path):
    a = "def f(self, p, **kw):\n" + "    self.alpha = g(p, mode=1, **kw)\n" * 7
    b = "def h(me, q, **opts):\n" + "    me.beta = k(q, kind=1, **opts)\n" * 7
    (group,) = lib_groups(project(tmp_path, {"a.py": a, "b.py": b}, {}))
    assert (group.kind, len(group.members)) == ("renamed", 2)


def test_arguments_of_nested_functions_are_abstracted(tmp_path):
    a = "def f(p):\n" + "    p = p + 1\n" * 6 + "    return lambda u: u + p\n"
    b = "def h(q):\n" + "    q = q + 1\n" * 6 + "    return lambda v: v + q\n"
    (group,) = lib_groups(project(tmp_path, {"a.py": a, "b.py": b}, {}))
    assert (group.kind, len(group.members)) == ("renamed", 2)


def test_a_different_operator_is_not_a_clone(tmp_path):
    other = func("g").replace("a + b", "a - b")
    root = project(tmp_path, {"a.py": func("f"), "b.py": other}, {})
    assert lib_groups(root) == []


def test_three_identical_and_one_renamed_form_one_renamed_group(tmp_path):
    files = {f"{n}.py": func(n) for n in "abc"} | {"d.py": func("d", var="y")}
    (group,) = lib_groups(project(tmp_path, files, {}))
    assert (group.kind, len(group.members)) == ("renamed", 4)


def test_redundant_lines_are_all_members_but_the_largest(tmp_path):
    files = {"a.py": func("f", 8), "b.py": func("g", 8), "c.py": func("h", 8)}
    (group,) = lib_groups(project(tmp_path, files, {}))
    assert group.redundant_lines == 16
    sizes = cc.Group("identical", (cc.Func("a", 9, "", ""), cc.Func("b", 20, "", "")))
    assert sizes.redundant_lines == 9


def test_compat_pace_is_excluded_from_the_library(tmp_path):
    files = {"a.py": func("f"), "compat/pace/b.py": func("g")}
    assert lib_groups(project(tmp_path, files, {})) == []


def test_methods_nested_functions_and_async_get_qualified_names(tmp_path):
    cls = (
        "class K:\n"
        + "".join("    " + line + "\n" for line in func("m").splitlines())
        + "    async def am(self, a, b=1):\n"
        + "        x = a\n" * 7
    )
    nested = (
        "def outer():\n    def inner(a):\n"
        + "        a = a + 1\n" * 7
        + "    return inner\n"
    )
    root = project(tmp_path, {"m.py": cls + nested}, {})
    names = [f.where.split(":")[-1] for f in cc.functions(root, cc.TREES[0])]
    assert names == ["K.m", "K.am", "outer", "outer.inner"]


def test_count_per_kind(tmp_path):
    files = {
        "a.py": func("f", 8),
        "b.py": func("g", 8),
        "c.py": func("h", 10, var="p"),
        "d.py": func("i", 10, var="q"),
        "e.py": func("j", 10, var="r"),
    }
    counts = cc.count(lib_groups(project(tmp_path, files, {})))
    assert counts == {
        "identical": {"groups": 1, "functions": 2, "redundant_lines": 8},
        "renamed": {"groups": 1, "functions": 3, "redundant_lines": 20},
    }


def test_unparsable_file_is_an_error_not_zero_clones(tmp_path):
    root = project(tmp_path, {"bad.py": "def f(:\n"}, {})
    with pytest.raises(cc.ToolError, match=r"cannot parse tensorpotential/bad\.py"):
        cc.functions(root, cc.TREES[0])


def test_missing_tree_is_an_error(tmp_path):
    with pytest.raises(cc.ToolError, match="does not exist"):
        cc.measure(tmp_path)


def test_compare_reports_rises_and_drops():
    base = {
        "library": {"identical": {"groups": 2, "functions": 4, "redundant_lines": 30}}
    }
    new = {
        "library": {"identical": {"groups": 1, "functions": 5, "redundant_lines": 30}}
    }
    rises, drops = cc.compare(base, new)
    assert rises == ["library functions: 4 -> 5"]
    assert drops == ["library groups: 2 -> 1"]


def test_compare_sums_the_kinds_so_a_renamed_clone_is_not_a_rise():
    counts = {"groups": 1, "functions": 2, "redundant_lines": 8}
    zero = dict.fromkeys(counts, 0)
    base = {"library": {"identical": counts, "renamed": zero}}
    new = {"library": {"identical": zero, "renamed": counts}}
    assert cc.compare(base, new) == ([], [])


@pytest.fixture
def repo(tmp_path):
    """A project with one identical library group and a baseline recorded from it."""
    root = project(tmp_path, {"a.py": func("f"), "b.py": func("g")}, {"t.py": ""})
    assert cc.main(["record", "--root", str(root)]) == 0
    return root


def test_record_writes_counts_and_min_lines(repo):
    data = json.loads((repo / cc.BASELINE).read_text())
    assert data["min_lines"] == cc.MIN_LINES
    assert data["counts"]["library"]["identical"] == {
        "groups": 1,
        "functions": 2,
        "redundant_lines": 8,
    }
    assert data["counts"]["tests"]["renamed"]["groups"] == 0


def test_check_passes_on_the_recorded_tree(repo, capsys):
    assert cc.main(["check", "--root", str(repo)]) == 0
    assert "clone ratchet: ok" in capsys.readouterr().out


def test_check_fails_when_a_clone_is_added(repo, capsys):
    (repo / "tensorpotential" / "c.py").write_text(func("h"))
    assert cc.main(["check", "--root", str(repo)]) == 1
    assert "rose: library functions: 2 -> 3" in capsys.readouterr().out


def test_check_reports_a_new_group_as_a_rise(repo, capsys):
    (repo / "tensorpotential" / "c.py").write_text(func("h", 9, var="z"))
    (repo / "tensorpotential" / "d.py").write_text(func("i", 9, var="z"))
    assert cc.main(["check", "--root", str(repo)]) == 1
    assert "rose: library groups: 1 -> 2" in capsys.readouterr().out


def test_check_notes_a_drop_and_fails_only_on_request(repo, capsys):
    (repo / "tensorpotential" / "b.py").write_text(func("g").replace("a + b", "a - b"))
    assert cc.main(["check", "--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "dropped: library groups: 1 -> 0" in out
    assert "record the lower baseline" in out
    assert cc.main(["check", "--root", str(repo), "--fail-on-drop"]) == 1


def test_record_refuses_a_rise_unless_allowed(repo, capsys):
    (repo / "tensorpotential" / "c.py").write_text(func("h"))
    assert cc.main(["record", "--root", str(repo)]) == 1
    assert "refusing to record a rise" in capsys.readouterr().out
    assert cc.main(["record", "--root", str(repo), "--allow-rise"]) == 0
    assert cc.main(["check", "--root", str(repo)]) == 0


def test_record_accepts_a_drop(repo):
    (repo / "tensorpotential" / "b.py").write_text("")
    assert cc.main(["record", "--root", str(repo)]) == 0
    data = json.loads((repo / cc.BASELINE).read_text())
    assert data["counts"]["library"]["identical"]["groups"] == 0


def test_missing_baseline_is_exit_2(tmp_path, capsys):
    root = project(tmp_path, {}, {})
    assert cc.main(["check", "--root", str(root)]) == 2
    assert "no baseline" in capsys.readouterr().err


def test_baseline_with_another_min_lines_is_exit_2(repo, capsys):
    path = repo / cc.BASELINE
    data = json.loads(path.read_text())
    data["min_lines"] = 5
    path.write_text(json.dumps(data))
    assert cc.main(["check", "--root", str(repo)]) == 2
    assert "min_lines=5" in capsys.readouterr().err


def test_unparsable_source_is_exit_2(repo, capsys):
    (repo / "tests" / "bad.py").write_text("def f(:\n")
    assert cc.main(["check", "--root", str(repo)]) == 2
    assert "cannot parse tests/bad.py" in capsys.readouterr().err


def test_list_prints_each_group_with_line_counts(repo, capsys):
    assert cc.main(["list", "--root", str(repo), "--tree", "library"]) == 0
    line = capsys.readouterr().out.strip()
    assert line == (
        "library identical: tensorpotential/a.py:1:f (8), tensorpotential/b.py:1:g (8)"
    )


def test_list_can_be_restricted_to_the_test_tree(repo, capsys):
    assert cc.main(["list", "--root", str(repo), "--tree", "tests"]) == 0
    assert capsys.readouterr().out == ""
