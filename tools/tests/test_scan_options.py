"""Logic tests for tools/scan_options.py: the yaml formats, the matrix rows and the status of a row."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scan_options as so

ROOT = Path(__file__).resolve().parents[2]
CLS = "tensorpotential.instructions.compute.FCRight2Left"


def write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


WRAPPED = """\
    metadata:
      param_dtype: float32
      saved_at: '2026-01-01T00:00:00Z'
    instructions:
      A:
        __cls__: tensorpotential.instructions.compute.FCRight2Left
        name: A
        n_out: 32
        normalize: true
        left: {_instruction_: true, name: B}
        lora_config: null
      B:
        __cls__: tensorpotential.instructions.compute.FCRight2Left
        name: B
        n_out: [1, 2]
        normalize: false
"""


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, "null"),
        (True, "bool"),
        (3, "int"),
        (3.0, "float"),
        ("x", "str"),
        ([], "list[empty]"),
        ([1, 2], "list[int]"),
        ([[1], [2, 3]], "list[list[int]]"),
        ([1, 2.5], "list[float|int]"),
        ({"Cu": 0}, "dict[str,int]"),
        ({0: 1.5}, "dict[int,float]"),
        ({"_instruction_": True, "name": "B"}, "ref"),
        ([{"_instruction_": True, "name": "B"}], "list[ref]"),
    ],
)
def test_value_kind(value, kind):
    assert so.value_kind(value) == kind


def test_bool_is_not_an_int():
    # bool is a subclass of int in Python; the matrix must keep them apart
    assert so.value_kind(True) != so.value_kind(1)


def test_wrapped_format_splits_metadata_and_instructions(tmp_path):
    rows = so.scan([write(tmp_path / "m.yaml", WRAPPED)])
    keyed = {(r.cls, r.option, r.kind): r for r in rows}
    assert keyed[so.METADATA_CLASS, "param_dtype", "str"].values == ("float32",)
    assert (
        so.METADATA_CLASS,
        "saved_at",
        "str",
    ) not in keyed  # a timestamp, not an option
    assert keyed[CLS, "n_out", "int"].low == keyed[CLS, "n_out", "int"].high == 32
    assert keyed[CLS, "n_out", "list[int]"].count == 1  # one yaml, two kinds, two rows
    assert keyed[CLS, "normalize", "bool"].values == (False, True)
    assert keyed[CLS, "normalize", "bool"].count == 2
    assert keyed[CLS, "left", "ref"].values == ()
    assert (CLS, "name", "str") not in keyed


def test_flat_dict_with_metadata_sentinel_and_list_formats(tmp_path):
    flat = write(
        tmp_path / "flat.yaml",
        """\
        __metadata__: {param_dtype: float64}
        A: {__cls__: tensorpotential.instructions.compute.BondLength, name: A, instruction_with_bonds: null}
        """,
    )
    old = write(
        tmp_path / "old.yaml",
        """\
        - {__cls__: tensorpotential.instructions.compute.BondLength, name: A, instruction_with_bonds: null}
        """,
    )
    flat_keys = {(r.cls, r.option) for r in so.scan([flat])}
    assert (so.METADATA_CLASS, "param_dtype") in flat_keys
    assert (
        "tensorpotential.instructions.compute.BondLength",
        "instruction_with_bonds",
    ) in flat_keys
    assert not any(option == "__metadata__" for _, option in flat_keys)
    old_keys = {(r.cls, r.option) for r in so.scan([old])}
    assert not any(cls == so.METADATA_CLASS for cls, _ in old_keys)
    assert (
        "tensorpotential.instructions.compute.BondLength",
        "instruction_with_bonds",
    ) in old_keys


def test_list_of_strings_reports_the_strings(tmp_path):
    path = write(
        tmp_path / "m.yaml",
        """\
        A: {__cls__: tensorpotential.instructions.compute.MLPRadialFunction_v2, name: A, activation: [silu, tanh]}
        B: {__cls__: tensorpotential.instructions.compute.MLPRadialFunction_v2, name: B, activation: [silu, silu]}
        """,
    )
    (row,) = [r for r in so.scan([path]) if r.option == "activation"]
    assert row.kind == "list[str]"
    assert row.values == ("silu", "tanh")


def test_rows_record_the_files_holding_a_cell(tmp_path):
    first = write(tmp_path / "a.yaml", WRAPPED)
    second = write(tmp_path / "b.yaml", WRAPPED)
    row = next(
        r
        for r in so.scan([first, second])
        if (r.option, r.kind) == ("normalize", "bool")
    )
    assert row.files == ("a.yaml", "b.yaml")
    assert row.count == 4


def test_expand_takes_the_yaml_files_of_a_directory_sorted_and_once(tmp_path):
    (tmp_path / "x.yaml").write_text("[]")
    (tmp_path / "y.txt").write_text("")
    single = tmp_path / "w.yaml"
    single.write_text("[]")
    assert [p.name for p in so.expand([tmp_path, single, single])] == [
        "w.yaml",
        "x.yaml",
    ]


@pytest.fixture(scope="module")
def options():
    return so.load_options()


def row(cls, option, kind, values=(), low=None, high=None):
    return so.Row(cls, option, kind, tuple(values), low, high, ("f.yaml",), 1)


@pytest.mark.parametrize(
    ("the_row", "status"),
    [
        (row(CLS, "__cls__", "str", [CLS]), "supported"),
        (row(CLS, "left", "ref"), "supported"),
        (row(CLS, "normalize", "bool", [True]), "supported"),
        (
            row(CLS, "normalize", "bool", [False]),
            "unclassified",
        ),  # a value no scanned model uses
        (
            row(CLS, "normalize", "bool", [False, True]),
            "unclassified",
        ),  # one accepted value is not enough
        (row(CLS, "n_out", "int", low=32, high=64), "supported"),
        (
            row(CLS, "n_out", "null", [None]),
            "rejected",
        ),  # explicit rejection with a reason
        (
            row(CLS, "n_out", "str", ["x"]),
            "unclassified",
        ),  # a kind the rule does not list
        (row(CLS, "no_such_option", "int", low=1, high=1), "unclassified"),
        (row(CLS, "lora_config", "null", [None]), "supported"),
        (row(so.METADATA_CLASS, "param_dtype", "str", ["float32"]), "supported"),
        (row(so.METADATA_CLASS, "param_dtype", "str", ["float16"]), "unclassified"),
        (
            row(so.METADATA_CLASS, "tensorpotential_version", "str", ["0.5.10"]),
            "ignored",
        ),
    ],
)
def test_classify(options, the_row, status):
    assert so.classify(the_row, options)[0] == status


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (
            "tensorpotential.instructions.compute.SPBF",
            "rejected",
        ),  # every option of a rejected class
        ("tensorpotential.instructions.compute.NoSuchClass", "unclassified"),
    ],
)
@pytest.mark.parametrize("option", ["__cls__", "anything"])
def test_classify_by_class(options, cls, status, option):
    assert so.classify(row(cls, option, "int", low=1, high=1), options)[0] == status


def test_classify_gives_a_reason_for_a_rejection(options):
    status, reason = so.classify(row(CLS, "n_out", "null", [None]), options)
    assert status == "rejected"
    assert "3L" in reason


def test_numeric_values_are_checked_against_the_allowed_values(options):
    target = "tensorpotential.instructions.output.CreateOutputTarget"
    assert (
        so.classify(row(target, "l", "int", low=0, high=0), options)[0] == "supported"
    )
    assert (
        so.classify(row(target, "l", "int", low=0, high=2), options)[0]
        == "unclassified"
    )


def test_blockers_name_rejected_classes_and_values(tmp_path, options):
    path = write(
        tmp_path / "m.yaml",
        """\
        A: {__cls__: tensorpotential.instructions.compute.SPBF, name: A, lmax: 1}
        B: {__cls__: tensorpotential.instructions.compute.FCRight2Left, name: B, n_out: null}
        C: {__cls__: tensorpotential.instructions.compute.BondLength, name: C}
        """,
    )
    assert so.blockers(path, options) == ["compute.FCRight2Left.n_out", "compute.SPBF"]


def test_markdown_collapses_the_options_of_a_rejected_class(tmp_path, options):
    path = write(
        tmp_path / "m.yaml",
        """\
        A: {__cls__: tensorpotential.instructions.compute.SPBF, name: A, lmax: 1, p: 5}
        B: {__cls__: tensorpotential.instructions.compute.BondLength, name: B}
        """,
    )
    table = so.to_markdown([path], options)
    assert table.startswith("# Option matrix")
    assert "| m.yaml | compute.SPBF |" in table
    assert "| compute.SPBF | all options |" in table
    assert "| compute.SPBF | lmax" not in table
    assert "| compute.BondLength | __cls__ |" in table


def test_main_exit_status_and_outputs(tmp_path, capsys):
    good = write(
        tmp_path / "good.yaml",
        "A: {__cls__: tensorpotential.instructions.compute.BondLength, name: A, instruction_with_bonds: null}\n",
    )
    bad = write(
        tmp_path / "bad.yaml",
        "A: {__cls__: tensorpotential.instructions.compute.BondLength, name: A, surprise: 1}\n",
    )
    out = tmp_path / "m.md"
    js = tmp_path / "m.json"
    assert so.main([str(good), "--markdown", str(out), "--json", str(js)]) == 0
    assert "compute.BondLength" in out.read_text()
    assert '"option": "instruction_with_bonds"' in js.read_text()
    assert so.main([str(bad)]) == 1
    captured = capsys.readouterr()
    assert "unclassified" in captured.err
    assert "surprise" in captured.err
    assert "| compute.BondLength | surprise" in captured.out


def test_the_tool_does_not_import_tensorflow():
    code = (
        "import sys; sys.path.insert(0, 'tools'); import scan_options as so; so.load_options(); "
        "assert 'tensorflow' not in sys.modules and 'tensorpotential' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
