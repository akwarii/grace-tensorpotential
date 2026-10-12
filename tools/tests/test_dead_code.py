"""Logic tests for tools/dead_code.py (no physics in this tool).

The scanned trees are small packages written into ``tmp_path`` with code vulture is known to flag at 80% confidence (an unused
import, 90%, and an unused argument, 100%), so the tool is checked against the library it wraps and not against a stub.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dead_code as dc

REPO = Path(__file__).resolve().parents[2]

CLEAN = "def used(x):\n    return x\n"
UNUSED_IMPORT = "import os\n\ndef used(x):\n    return x\n"
TWO_UNUSED_IMPORTS = "import os\nimport json\n\ndef used(x):\n    return x\n"
UNUSED_ARG = "def used(x, spare):\n    return x\n"


def make_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``files`` under ``root/tensorpotential`` and an empty whitelist; return the whitelist path."""
    for name, text in files.items():
        path = root / "tensorpotential" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    whitelist = root / "whitelist.txt"
    whitelist.write_text("")
    return whitelist


def run(root: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.chdir(root)
    return dc.main([
        *args,
        "--whitelist",
        str(root / "whitelist.txt"),
        "--baseline",
        str(root / "base.json"),
    ])


def test_scan_finds_unused_import_and_argument_with_relative_paths(tmp_path):
    wl = make_tree(tmp_path, {"a.py": UNUSED_IMPORT, "b.py": UNUSED_ARG, "c.py": CLEAN})
    found = dc.scan(tmp_path, wl)
    assert [(f.path, f.kind, f.name) for f in found] == [
        ("tensorpotential/a.py", "import", "os"),
        ("tensorpotential/b.py", "variable", "spare"),
    ]
    assert found[0].confidence == 90
    assert (
        str(found[1])
        == "tensorpotential/b.py:1: unused variable 'spare' (100% confidence)"
    )


def test_scan_respects_the_confidence_floor(tmp_path):
    wl = make_tree(tmp_path, {"a.py": UNUSED_IMPORT, "b.py": UNUSED_ARG})
    assert [f.name for f in dc.scan(tmp_path, wl, min_confidence=95)] == ["spare"]


def test_scan_skips_compat_pace(tmp_path):
    wl = make_tree(tmp_path, {"compat/pace/old.py": UNUSED_IMPORT, "a.py": CLEAN})
    assert dc.scan(tmp_path, wl) == []


def test_whitelist_names_and_decorators_silence_findings(tmp_path):
    routed = "@app.route('/x')\ndef handler():\n    return 1\n"
    wl = make_tree(tmp_path, {"b.py": UNUSED_ARG, "r.py": routed})
    before = {f.name for f in dc.scan(tmp_path, wl, min_confidence=60)}
    assert {"spare", "handler"} <= before  # both are found without a whitelist
    wl.write_text(
        "spare  # kept for a reason\n@app.route*  # a web route\n\n# a comment line\n"
    )
    after = {f.name for f in dc.scan(tmp_path, wl, min_confidence=60)}
    assert after == before - {"spare", "handler"}


def test_load_whitelist_splits_names_from_decorators(tmp_path):
    wl = tmp_path / "w.txt"
    wl.write_text("# header\nfoo  # why\n@app.route*\n\n  bar_*  \n")
    assert dc.load_whitelist(wl) == (["foo", "bar_*"], ["@app.route*"])


def test_missing_whitelist_is_an_error(tmp_path):
    with pytest.raises(dc.DeadCodeError, match="no whitelist"):
        dc.load_whitelist(tmp_path / "absent.txt")


def test_counts_and_rises():
    findings = [
        dc.Finding("a.py", 1, 90, "import", "os"),
        dc.Finding("a.py", 2, 90, "import", "sys"),
    ]
    counts = dc.counts_of(findings)
    assert counts == {"a.py": 2}
    assert dc.rises({"a.py": 2}, counts) == []
    assert dc.rises({"a.py": 3}, counts) == []
    assert dc.rises({"a.py": 1}, counts) == ["rose: a.py 1 -> 2"]
    assert dc.rises({}, counts) == [
        "rose: a.py 0 -> 2"
    ]  # a file without an entry starts at zero


def test_check_passes_on_the_recorded_tree(tmp_path, monkeypatch):
    make_tree(tmp_path, {"a.py": UNUSED_IMPORT})
    assert run(tmp_path, monkeypatch, "record") == 0
    assert run(tmp_path, monkeypatch, "check") == 0


def test_planted_dead_code_in_a_new_file_fails_check(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"a.py": CLEAN})
    run(tmp_path, monkeypatch, "record")
    (tmp_path / "tensorpotential" / "new.py").write_text(UNUSED_IMPORT)
    assert run(tmp_path, monkeypatch, "check") == 1
    out = capsys.readouterr().out
    assert "rose: tensorpotential/new.py 0 -> 1" in out
    assert "unused import 'os'" in out


def test_planted_dead_code_in_a_known_file_fails_check(tmp_path, monkeypatch):
    make_tree(tmp_path, {"a.py": UNUSED_IMPORT})
    run(tmp_path, monkeypatch, "record")
    (tmp_path / "tensorpotential" / "a.py").write_text(TWO_UNUSED_IMPORTS)
    assert run(tmp_path, monkeypatch, "check") == 1


def test_a_fall_passes_check_and_is_reported(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"a.py": TWO_UNUSED_IMPORTS})
    run(tmp_path, monkeypatch, "record")
    (tmp_path / "tensorpotential" / "a.py").write_text(UNUSED_IMPORT)
    capsys.readouterr()
    assert run(tmp_path, monkeypatch, "check") == 0
    assert "1 fell" in capsys.readouterr().out
    assert run(tmp_path, monkeypatch, "record") == 0
    assert json.loads((tmp_path / "base.json").read_text())["files"] == {
        "tensorpotential/a.py": 1
    }


def test_record_refuses_a_rise_unless_allowed(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"a.py": CLEAN})
    run(tmp_path, monkeypatch, "record")
    (tmp_path / "tensorpotential" / "a.py").write_text(UNUSED_IMPORT)
    assert run(tmp_path, monkeypatch, "record") == 1
    assert "rose: tensorpotential/a.py 0 -> 1" in capsys.readouterr().out
    assert json.loads((tmp_path / "base.json").read_text())["files"] == {}
    assert run(tmp_path, monkeypatch, "record", "--allow-rise") == 0
    assert json.loads((tmp_path / "base.json").read_text())["files"] == {
        "tensorpotential/a.py": 1
    }


def test_record_refuses_to_change_the_confidence_silently(
    tmp_path, monkeypatch, capsys
):
    make_tree(tmp_path, {"a.py": UNUSED_IMPORT})
    run(tmp_path, monkeypatch, "record")
    assert run(tmp_path, monkeypatch, "record", "--min-confidence", "60") == 1
    assert "confidence 80 -> 60" in capsys.readouterr().out


def test_check_uses_the_confidence_of_the_baseline(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"b.py": UNUSED_ARG})
    run(tmp_path, monkeypatch, "record", "--min-confidence", "100")
    assert run(tmp_path, monkeypatch, "check") == 0
    assert "at 100% confidence" in capsys.readouterr().out


def test_moved_file_keeps_its_baseline_entry(tmp_path, monkeypatch):
    make_tree(tmp_path, {"old.py": UNUSED_IMPORT})
    run(tmp_path, monkeypatch, "record")
    (tmp_path / "tensorpotential" / "old.py").rename(
        tmp_path / "tensorpotential" / "new.py"
    )
    assert run(tmp_path, monkeypatch, "check") == 1
    moved = ["--moved", "tensorpotential/old.py", "tensorpotential/new.py"]
    assert run(tmp_path, monkeypatch, "check", *moved) == 0


def test_list_prints_every_finding(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"a.py": UNUSED_IMPORT})
    assert run(tmp_path, monkeypatch, "list") == 0
    assert (
        capsys.readouterr().out
        == "tensorpotential/a.py:1: unused import 'os' (90% confidence)\n"
    )


def test_missing_baseline_and_bad_inputs_exit_2(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, {"a.py": CLEAN})
    assert run(tmp_path, monkeypatch, "check") == 2
    assert "no baseline" in capsys.readouterr().err
    (tmp_path / "base.json").write_text("not json")
    assert run(tmp_path, monkeypatch, "check") == 2
    (tmp_path / "base.json").write_text("{}")
    assert run(tmp_path, monkeypatch, "check") == 2  # no "files" key
    assert run(tmp_path, monkeypatch, "check", "--moved", "x.py", "x.py") == 2


def test_the_committed_baseline_matches_the_tree():
    """The exit criterion: ``check`` exits 0 on the repository, with the committed baseline and whitelist."""
    done = subprocess.run(  # noqa: S603 - the interpreter and a script of this repository
        [sys.executable, str(REPO / "tools" / "dead_code.py"), "check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
