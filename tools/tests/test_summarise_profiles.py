"""Tests for tools/summarise_profiles.py: hand-made profile files in, the table cells checked by hand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import summarise_profiles as sp  # noqa: E402


def _row(name, cls, fwd_bwd):
    return {
        "name": name,
        "class": cls,
        "fwd_ms": 1.0,
        "fwd_bwd_ms": fwd_bwd,
        "error": None,
    }


def _profile(yaml="m/GRACE-2L.yaml", atoms=54, bonds=3000, whole=300.0, **extra):
    rows = [
        _row("a", "SPBF", 60.0),
        _row("b", "SPBF", 20.0),
        _row("c", "Product", 20.0),
        _row("d", "Reduce", None),
    ]
    return {
        "yaml": yaml,
        "dtype": "float64",
        "option": "none",
        "grads": "inputs",
        "n_atoms": atoms,
        "n_bonds": bonds,
        "whole_ms": whole,
        "rows": rows,
        **extra,
    }


def _write(directory: Path, name: str, profiles, env=None):
    (directory / name).write_text(
        json.dumps({"environment": env or {"device": "cpu"}, "profiles": profiles})
    )


def test_class_shares_add_the_instructions_of_a_class_and_skip_untimed_rows():
    shares = sp.class_shares(_profile()["rows"])

    assert shares == [("SPBF", 80.0), ("Product", 20.0)]


def test_class_shares_of_rows_without_time_are_empty():
    assert sp.class_shares([_row("a", "X", None)]) == []
    assert sp.class_shares([]) == []


def test_a_cpu_row_has_time_per_atom_the_model_stem_and_the_top_classes():
    cells = sp.row_of({
        **_profile(),
        "environment": {"device": "cpu"},
        "peak_rss_mb": 1500.4,
    })

    assert cells[:9] == [
        "GRACE-2L",
        "cpu",
        "float64",
        "none",
        "inputs",
        "54",
        "3000",
        "300",
        "5.6",
    ]
    assert cells[9:12] == ["", "", "1500"]
    assert cells[12] == "SPBF 80%, Product 20%"


def test_a_gpu_row_names_the_gpu_and_the_memory_per_bond():
    profile = {
        **_profile(bonds=2048),
        "environment": {"device": "gpu", "gpu_name": "A100"},
        "gpu_peak_memory_mb": 4.0,
    }

    cells = sp.row_of(profile)

    assert cells[1] == "A100"
    assert cells[9:11] == ["4", "2"]  # 4 MB over 2048 bonds = 2 KB each


def test_a_whole_only_profile_has_no_top_classes_and_no_gradient_label():
    profile = {**_profile(), "rows": [], "environment": {}}

    cells = sp.row_of(profile)

    assert cells[12] == ""
    assert cells[4] == ""


def test_load_profiles_reads_nested_files_and_skips_other_json(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(tmp_path, "a.json", [_profile(), _profile(atoms=432)])
    _write(
        tmp_path / "sub",
        "b.json",
        [_profile(yaml="x/FS.yaml")],
        env={"device": "gpu", "gpu_name": "A100"},
    )
    (tmp_path / "other.json").write_text(json.dumps({"repeats": 5, "rows": []}))
    (tmp_path / "list.json").write_text("[1, 2]")

    profiles = sp.load_profiles([tmp_path])

    assert len(profiles) == 3
    assert {p["environment"]["device"] for p in profiles} == {"cpu", "gpu"}


def test_render_sorts_by_model_then_size():
    profiles = [
        {**_profile(yaml="b/Z.yaml", atoms=54), "environment": {}},
        {**_profile(yaml="a/Y.yaml", atoms=432), "environment": {}},
        {**_profile(yaml="a/Y.yaml", atoms=54), "environment": {}},
    ]

    lines = sp.render(profiles).splitlines()

    assert lines[0].startswith("| model |")
    assert [
        (ln.split("|")[1].strip(), ln.split("|")[6].strip()) for ln in lines[2:]
    ] == [("Y", "54"), ("Y", "432"), ("Z", "54")]


def test_main_prints_and_writes_the_table(tmp_path, capsys):
    _write(tmp_path, "a.json", [_profile()])
    out = tmp_path / "t.md"

    assert sp.main([str(tmp_path), "--out", str(out)]) == 0

    assert "GRACE-2L" in capsys.readouterr().out
    assert out.read_text().startswith("| model |")


def test_main_without_any_profile_fails(tmp_path, capsys):
    assert sp.main([str(tmp_path)]) == 1
    assert "no profile JSON" in capsys.readouterr().err


def test_the_header_and_separator_have_one_cell_per_column():
    lines = sp.render([{**_profile(), "environment": {}}]).splitlines()

    assert lines[0].count("|") == lines[1].count("|") == lines[2].count("|")
