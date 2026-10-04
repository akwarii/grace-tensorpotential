"""Tests of ``torch_backend/spec/registry.py``: the class registry, the strict check, and the completeness tests.

Two layers:

- logic: the registry entries, the value kinds and the option rules, and every kind of finding of
  ``check_supported`` (unknown class, rejected class, unknown option with the nearest valid key, missing option,
  unsupported value, rejected value), that all findings come in one error, and that valid models pass;
- physical values (completeness): the oracle is the TF **source**, read with ``ast`` by ``tools/instruction_ast.py``
  (which never calls the registry). Every instruction class the source defines has an entry, the pinned defaults
  equal the source's constructor defaults, every supported class has a spec sheet, and the classes without a golden
  test are listed in ``EXPECTED_WITHOUT_GOLDEN``, which each twin issue shrinks. Planted changes of a copy of the
  source (a new class, a changed default, a removed class, a new parameter) make the same check fail.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import instruction_ast as ia  # noqa: E402
import scan_options as so  # noqa: E402

from tensorpotential.torch_backend.spec import options as opt  # noqa: E402
from tensorpotential.torch_backend.spec import registry as reg  # noqa: E402
from tensorpotential.torch_backend.spec.errors import (  # noqa: E402
    Problem,
    SpecError,
    UnsupportedModelError,
)
from tensorpotential.torch_backend.spec.model import (  # noqa: E402
    InstructionRef,
    InstructionSpec,
)

C = opt.COMPUTE
O = opt.OUTPUT  # noqa: E741

# The supported classes that have no golden test yet (their golden fixtures come from FIX2, the twin tests from
# Stage 3). A twin issue that adds a golden test sets ``GOLDEN_TESTS`` in registry.py and removes the class here:
# the test below fails in both directions, so the list can only shrink.
EXPECTED_WITHOUT_GOLDEN = {
    "BondLength",
    "ScaledBondVector",
    "RadialBasis",
    "SphericalHarmonic",
    "MLPRadialFunction",
    "MLPRadialFunction_v2",
    "ScalarChemicalEmbedding",
    "SingleParticleBasisFunctionScalarInd",
    "SingleParticleBasisFunctionEquivariantInd",
    "ProductFunction",
    "FCRight2Left",
    "FunctionReduceN",
    "InvariantLayerRMSNorm",
    "CreateOutputTarget",
    "LinMLPOut2ScalarTarget",
    "ConstantScaleShiftTarget",
    "TrainableShiftTarget",
}

SOURCE_COPY_DIRS = ("instructions", "functions", "uq")


def ref(name: str) -> InstructionRef:
    return InstructionRef(name)


def make(
    name: str, cls: str, defaulted: tuple[str, ...] = (), **options: Any
) -> InstructionSpec:
    """An instruction with the pinned defaults filled in, as the loader makes it (``defaulted`` lists none unless asked)."""
    filled = {k: v for k, v in opt.DEFAULTS.get(cls, {}).items() if k not in options}
    return InstructionSpec(
        name=name,
        cls=cls,
        options={**options, **filled},
        depends_on=(),
        index=0,
        defaulted=frozenset(filled) if defaulted == ("*",) else frozenset(defaulted),
    )


def scaled(name: str = "S", **options: Any) -> InstructionSpec:
    return make(name, C + "ScaledBondVector", bond_length=ref("L"), **options)


def problems_of(*instructions: InstructionSpec) -> list[Problem]:
    return reg.find_problems(instructions)


def kinds_of(*instructions: InstructionSpec) -> list[tuple[str, str | None]]:
    return [(p.kind, p.option) for p in problems_of(*instructions)]


# --------------------------------------------------------------------------------------------- the registry


def test_every_class_of_the_option_tables_has_one_entry():
    assert set(reg.REGISTRY) == set(opt.SUPPORTED_OPTIONS) | set(opt.REJECTED_CLASSES)
    assert list(reg.REGISTRY) == sorted(reg.REGISTRY)
    assert len(reg.supported_classes()) == 17
    assert set(reg.supported_classes()) == set(opt.SUPPORTED_OPTIONS)


def test_a_supported_entry_has_a_sheet_and_no_reason_a_rejected_one_has_a_reason_and_no_sheet():
    for path, entry in reg.REGISTRY.items():
        assert entry.path == path
        assert entry.name == path.rsplit(".", 1)[-1]
        if entry.status == "supported":
            assert entry.reason == ""
            assert entry.sheet == f"docs/torch_backend/spec/{entry.name}.md"
        else:
            assert entry.reason.strip()
            assert entry.sheet is None
            assert entry.golden_test is None


def test_the_rejection_reasons_are_distinct_enough_to_act_on():
    assert "CollectInvarBasis" in opt.REJECTED_CLASSES[C + "CollectInvarBasis"] or (
        "abstract" in opt.REJECTED_CLASSES[C + "CollectInvarBasis"]
    )
    assert "base class" in opt.REJECTED_CLASSES[opt.BASE + "TPInstruction"]
    assert (
        "uncertainty" in opt.REJECTED_CLASSES[opt.UQ + "RandomProjectedBasisFeatures"]
    )


# ----------------------------------------------------------------------------------- value kinds and rules


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, "null"),
        (True, "bool"),
        (3, "int"),
        (2.5, "float"),
        ("a", "str"),
        (InstructionRef("x"), "ref"),
        ([1, 2], "list[int]"),
        ([[1, 2], [0, 1]], "list[list[int]]"),
        ([InstructionRef("a")], "list[ref]"),
        (["silu"], "list[str]"),
        ([], "list[empty]"),
        ([1, 2.0], "list[float|int]"),
        ({"Cu": 0}, "dict[str,int]"),
        ({0: 1.5}, "dict[int,float]"),
        ({}, "dict[empty,empty]"),
    ],
)
def test_value_kinds(value, kind):
    assert reg.value_kind(value) == kind


def test_value_kinds_equal_those_of_the_scanner_on_every_scanned_value():
    """Oracle: ``tools/scan_options.value_kind`` reads the raw yaml, where a reference is a mapping."""
    checked = 0
    for path in so.expand([
        ROOT / "tests/data/model_yamls",
        *sorted((ROOT / "tests").glob("model_grace*.yaml")),
    ]):
        _, instructions = so.instruction_dicts(yaml.safe_load(path.read_text()))
        for ins in instructions:
            for key, value in ins.items():
                if key not in ("__cls__", "name"):
                    assert reg.value_kind(_as_refs(value)) == so.value_kind(value), (
                        path.name,
                        key,
                    )
                    checked += 1
    assert checked > 1000


def _as_refs(value: Any) -> Any:
    if isinstance(value, dict) and "_instruction_" in value:
        return InstructionRef(value["name"])
    if isinstance(value, list):
        return [_as_refs(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_refs(v) for k, v in value.items()}
    return value


def test_accepts_checks_the_kind_then_the_values():
    rule = opt.OptionRule(("int",), (1,))
    assert reg.accepts(rule, 1)
    assert not reg.accepts(rule, 2)
    assert not reg.accepts(rule, True)  # True == 1, but a bool is not an int
    assert not reg.accepts(rule, 1.0)
    assert reg.accepts(opt.OptionRule(("int", "null")), None)
    assert not reg.accepts(opt.OptionRule(("int",)), None)


def test_accepts_applies_the_values_of_a_list_of_strings_to_each_element():
    rule = opt.OptionRule(("list[str]",), ("silu",))
    assert reg.accepts(rule, ["silu", "silu"])
    assert not reg.accepts(rule, ["silu", "tanh"])
    assert not reg.accepts(rule, [])  # list[empty] is not list[str]: closed world


# ----------------------------------------------------------------------------------------- check_supported


def test_a_valid_instruction_has_no_problem():
    assert problems_of(make("L", C + "BondLength"), scaled()) == []
    reg.check_supported([make("L", C + "BondLength"), scaled()])


def test_an_unknown_class_is_reported_with_the_exact_name_rule_and_the_nearest_class():
    (problem,) = problems_of(make("X", C + "BondLengthh"))
    assert (problem.kind, problem.instruction, problem.option) == (
        "unknown_class",
        "X",
        None,
    )
    assert problem.cls == C + "BondLengthh"
    assert "unknown to the torch backend" in problem.message
    assert "exact name" in problem.message
    assert f"did you mean {C}BondLength?" in problem.message


def test_a_subclass_of_a_supported_class_is_not_supported():
    (problem,) = problems_of(make("X", "mypackage.MyBondLength"))
    assert problem.kind == "unknown_class"
    assert "a subclass of a supported class is not supported" in problem.message
    assert "did you mean" not in problem.message


def test_a_rejected_class_is_reported_with_its_reason():
    (problem,) = problems_of(make("X", C + "SPBF"))
    assert problem.kind == "rejected_class"
    assert problem.message.endswith(opt.REJECTED_CLASSES[C + "SPBF"])
    assert "instruction 'X'" in problem.message


def test_an_unknown_option_names_the_nearest_valid_key():
    ins = make(
        "R",
        C + "MLPRadialFunction",
        basis=ref("B"),
        n_rad_max=3,
        lmax=2,
        hidden_layer=[32],
    )
    found = [p for p in problems_of(ins) if p.kind == "unknown_option"]
    (problem,) = found
    assert problem.option == "hidden_layer"
    assert "did you mean 'hidden_layers'" in problem.message
    assert "instruction 'R' (MLPRadialFunction)" in problem.message


def test_an_unknown_option_without_a_near_key_gets_no_suggestion():
    (problem,) = [
        p
        for p in problems_of(make("L", C + "BondLength", zzzzzz=1))
        if p.kind == "unknown_option"
    ]
    assert "did you mean" not in problem.message


def test_a_key_that_tf_ignores_is_accepted_but_never_suggested():
    base = {"instruction_with_bonds": None}
    ignored = [
        k for k, r in opt.SUPPORTED_OPTIONS[C + "ProductFunction"].items() if r.ignored
    ]
    assert ignored
    pf = make(
        "P",
        C + "ProductFunction",
        left=ref("a"),
        right=ref("b"),
        lmax=2,
        Lmax=2,
        normalize=True,
        keep_parity=[[0, 1]],
    )
    assert problems_of(pf) == []
    with_ignored = InstructionSpec("P", pf.cls, {**pf.options, "n_out": None}, (), 0)
    assert problems_of(with_ignored) == []
    typo = InstructionSpec("P", pf.cls, {**pf.options, "n_ou": None}, (), 0)
    (problem,) = problems_of(typo)
    assert problem.kind == "unknown_option"
    assert (
        "n_out" not in problem.message
    )  # an ignored key is not a valid key to suggest
    assert base


def test_a_required_option_that_is_missing_is_reported():
    ins = InstructionSpec("S", C + "ScaledBondVector", {"bonds": None}, (), 0)
    (problem,) = problems_of(ins)
    assert (problem.kind, problem.option) == ("missing_option", "bond_length")
    assert "required option 'bond_length' is missing" in problem.message


def test_an_option_with_a_default_is_not_required():
    assert problems_of(make("L", C + "BondLength")) == []
    assert "instruction_with_bonds" not in make("L", C + "BondLength").defaulted


def test_a_value_of_the_wrong_kind_says_what_kinds_are_supported():
    (problem,) = problems_of(scaled(bonds="x"))
    assert (problem.kind, problem.option) == ("unsupported_value", "bonds")
    assert "'x' (str)" in problem.message
    assert "supported values: None" in problem.message


def test_a_value_outside_the_listed_values_says_what_values_are_supported():
    ins = make(
        "R", C + "RadialBasis", bonds=ref("L"), basis_type="Gaussian", nfunc=4, rcut=5.0
    )
    (problem,) = problems_of(ins)
    assert (problem.kind, problem.option) == ("unsupported_value", "basis_type")
    assert "'Gaussian'" in problem.message
    assert "'Cheb', 'SBessel'" in problem.message


def test_a_kind_rule_without_values_lists_the_kinds():
    ins = make(
        "R",
        C + "RadialBasis",
        bonds=ref("L"),
        basis_type="Cheb",
        nfunc="four",
        rcut=5.0,
    )
    (problem,) = problems_of(ins)
    assert problem.option == "nfunc"
    assert "supported kinds: int" in problem.message


def test_a_value_seen_in_a_shipped_yaml_and_rejected_gives_its_reason():
    ins = make(
        "A",
        C + "FCRight2Left",
        left=ref("a"),
        right=ref("b"),
        n_out=None,
        norm_out=True,
    )
    (problem,) = problems_of(ins)
    assert (problem.kind, problem.option) == ("rejected_value", "n_out")
    assert "GRACE-3L-OMAT-large" in problem.message
    assert "(the default)" not in problem.message


def test_a_default_that_is_not_supported_says_so_and_asks_for_the_option():
    ins = make("A", C + "FCRight2Left", ("*",), left=ref("a"), right=ref("b"), n_out=3)
    found = {p.option: p for p in problems_of(ins)}
    assert set(found) == {"norm_out"}
    assert (
        "absent from the yaml, so the default False applies"
        in found["norm_out"].message
    )
    assert "write the option explicitly" in found["norm_out"].message


def test_a_rejected_default_is_marked_as_the_default():
    ins = make(
        "A", C + "FCRight2Left", ("*",), left=ref("a"), right=ref("b"), norm_out=True
    )
    (problem,) = problems_of(ins)
    assert (problem.kind, problem.option) == ("rejected_value", "n_out")
    assert "= None (the default)" in problem.message


def test_a_long_value_is_cut_in_the_message():
    (problem,) = problems_of(scaled(bonds="x" * 500))
    assert "..." in problem.message
    assert len(problem.message) < 250


def test_every_finding_comes_in_one_error_in_file_order():
    instructions = [
        make("A", C + "SPBF"),
        make("B", "mypkg.Mine"),
        scaled("C", bonds="x"),
        InstructionSpec("D", C + "ScaledBondVector", {"bonds": None, "oops": 1}, (), 3),
    ]
    with pytest.raises(UnsupportedModelError) as caught:
        reg.check_supported(instructions)
    error = caught.value
    assert [(p.instruction, p.kind) for p in error.problems] == [
        ("A", "rejected_class"),
        ("B", "unknown_class"),
        ("C", "unsupported_value"),
        ("D", "unknown_option"),
        ("D", "missing_option"),
    ]
    lines = str(error).splitlines()
    assert lines[0] == "the model cannot be loaded by the torch backend:"
    assert [line.strip() for line in lines[1:]] == [p.message for p in error.problems]


def test_the_error_is_a_spec_error_and_carries_no_problems_for_a_model_level_value():
    assert issubclass(UnsupportedModelError, SpecError)
    assert UnsupportedModelError("x").problems == ()


def test_the_check_reads_the_options_the_loader_made_not_the_yaml_alone():
    """A key the yaml omits and the default fills in is checked like a written one (and marked as defaulted)."""
    ins = make(
        "P",
        C + "ProductFunction",
        ("*",),
        left=ref("a"),
        right=ref("b"),
        lmax=2,
        Lmax=2,
        keep_parity=[[0, 1]],
    )
    (problem,) = problems_of(ins)
    assert problem.option == "normalize"
    assert "absent from the yaml" in problem.message


# ----------------------------------------------------------------- completeness (the oracle is the source)


@pytest.fixture(scope="module")
def source_table():
    return ia.extract(ROOT)


def test_every_instruction_class_of_the_source_has_an_entry_and_no_entry_is_stale(
    source_table,
):
    assert set(source_table) == set(reg.REGISTRY)
    assert len(source_table) == 39


def test_each_class_is_supported_or_rejected_with_a_reason(source_table):
    for path in source_table:
        entry = reg.REGISTRY[path]
        assert entry.status in ("supported", "rejected"), path
        assert entry.status == "supported" or entry.reason, path
    assert sum(e.status == "supported" for e in reg.REGISTRY.values()) == 17
    assert sum(e.status == "rejected" for e in reg.REGISTRY.values()) == 22


def test_the_pinned_defaults_equal_the_defaults_of_the_source(source_table):
    registry, options = ia._import_registry()
    assert ia.default_problems(ROOT, source_table, registry, options) == []


def test_the_whole_check_passes_on_the_current_tree():
    assert ia.problems(ROOT) == []


def test_every_supported_class_has_a_spec_sheet_that_names_it():
    for path in reg.supported_classes():
        entry = reg.REGISTRY[path]
        assert entry.sheet is not None
        sheet = ROOT / entry.sheet
        assert sheet.is_file(), f"{entry.name}: no spec sheet at {entry.sheet}"
        assert entry.name in sheet.read_text().splitlines()[0], entry.sheet
    readme = (ROOT / "docs/torch_backend/spec/README.md").read_text()
    for path in reg.supported_classes():
        assert f"[{reg.REGISTRY[path].name}]" in readme


def test_there_is_no_sheet_for_a_class_the_registry_does_not_support():
    sheets = {p.stem for p in (ROOT / "docs/torch_backend/spec").glob("*.md")}
    supported = {e.name for e in reg.REGISTRY.values() if e.status == "supported"}
    assert sheets - supported == {"README", "option_matrix"}


def test_the_classes_without_a_golden_test_are_the_expected_list():
    """Strict expected failure: a twin issue that adds a golden test removes the class from the list, and the list can only shrink."""
    lacking = {
        e.name
        for e in reg.REGISTRY.values()
        if e.status == "supported" and e.golden_test is None
    }
    assert lacking == EXPECTED_WITHOUT_GOLDEN
    assert EXPECTED_WITHOUT_GOLDEN <= {e.name for e in reg.REGISTRY.values()}


def test_a_registered_golden_test_exists_and_names_its_class():
    for path, test_file in reg.GOLDEN_TESTS.items():
        assert path in opt.SUPPORTED_OPTIONS
        text = (ROOT / test_file).read_text()
        assert path.rsplit(".", 1)[-1] in text


# ----------------------------------------------------------------------------------------- planted changes


@pytest.fixture
def copy(tmp_path):
    """A copy of the TF source (without ``compat``) that a test may edit."""
    package = tmp_path / "tensorpotential"
    for name in SOURCE_COPY_DIRS:
        shutil.copytree(
            ROOT / "tensorpotential" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copy(ROOT / "tensorpotential/__init__.py", package / "__init__.py")
    return tmp_path


def edit(root: Path, relative: str, old: str, new: str, in_class: str = "") -> None:
    """Replace the first ``old`` after the header of ``in_class`` (from the top of the file when it is empty)."""
    path = root / "tensorpotential" / relative
    text = path.read_text()
    start = text.index(f"class {in_class}(") if in_class else 0
    at = text.index(old, start)
    path.write_text(text[:at] + new + text[at + len(old) :])


def test_the_copy_alone_is_clean(copy):
    assert ia.problems(copy) == []


def test_a_class_added_to_a_copy_of_the_source_is_reported(copy):
    with (copy / "tensorpotential/instructions/compute.py").open("a") as handle:
        handle.write(
            "\n\n@capture_init_args\nclass BrandNew(TPInstruction):\n"
            "    def __init__(self, a, name='BrandNew'):\n        super().__init__(name=name)\n"
        )
    found = [p for p in ia.problems(copy) if "BrandNew" in p]
    assert len(found) == 1
    assert "without an entry in the registry" in found[0]
    assert "compute.py" in found[0]


def test_a_class_added_in_another_module_through_a_reexport_is_reported(copy):
    with (copy / "tensorpotential/uq/instructions.py").open("a") as handle:
        handle.write("\n\nclass Extra(TPInstruction):\n    pass\n")
    assert any("uq.instructions.Extra" in p for p in ia.problems(copy))


def test_a_changed_default_in_a_copy_is_reported(copy):
    edit(
        copy,
        "instructions/compute.py",
        "lm_first: bool = False",
        "lm_first: bool = True",
        "ProductFunction",
    )
    found = [p for p in ia.problems(copy) if "lm_first" in p]
    assert found == ["ProductFunction.lm_first: source default True, pinned False"]


def test_a_changed_default_of_the_basis_function_that_radial_basis_forwards_to_is_reported(
    copy,
):
    edit(copy, "functions/radial.py", "p: int = 5,", "p: int = 6,")
    assert "RadialBasis.p: source default 6, pinned 5" in ia.problems(copy)


SPBF_SCALAR = "SingleParticleBasisFunctionScalarInd"
CONSTRUCTOR_CHANGES = [
    pytest.param(
        "avg_n_neigh: float | dict = 1.0,",
        "avg_n_neigh: float | dict = 1.0,\n        extra_knob: int = 3,",
        [
            "constructor parameter 'extra_knob' has no option rule",
            "extra_knob: source default 3, pinned <absent>",
        ],
        id="new-parameter",
    ),
    pytest.param(
        "        indicator_l_depend: bool = False,\n",
        "",
        ["option 'indicator_l_depend' is no constructor parameter any more"],
        id="removed-parameter",
    ),
    pytest.param(
        "avg_n_neigh: float | dict = 1.0,",
        "avg_n_neigh: float | dict = default_avg(),",
        ["the default default_avg() is not a literal"],
        id="non-literal-default",
    ),
    pytest.param(
        "sum_neighbors: bool = True,",
        "sum_neighbors: bool = 1,",
        ["sum_neighbors: source default 1, pinned True"],
        id="equal-but-another-type",
    ),
]


@pytest.mark.parametrize(("old", "new", "expected"), CONSTRUCTOR_CHANGES)
def test_a_changed_constructor_of_a_supported_class_is_reported(
    copy, old, new, expected
):
    edit(copy, "instructions/compute.py", old, new, SPBF_SCALAR)
    found = ia.problems(copy)
    for text in expected:
        assert any(text in p for p in found), (text, found)


def test_a_removed_class_is_reported_as_a_stale_registry_entry(copy):
    edit(
        copy,
        "instructions/compute.py",
        "class ZBLPotential(",
        "class ZBLPotential_renamed(",
    )
    found = ia.problems(copy)
    assert any(
        f"{C}ZBLPotential: the registry names a class the source does not define" in p
        for p in found
    )
    assert any("ZBLPotential_renamed" in p for p in found)


def test_losing_the_capture_decorator_of_a_supported_class_is_reported(copy):
    edit(
        copy,
        "instructions/compute.py",
        "@capture_init_args\nclass BondLength(",
        "class BondLength(",
    )
    assert any(
        "BondLength: capture_init_args was removed" in p for p in ia.problems(copy)
    )


def test_a_non_literal_default_of_a_supported_class_is_reported(copy):
    edit(
        copy,
        "instructions/compute.py",
        "avg_n_neigh: float | dict = 1.0,",
        "avg_n_neigh: float | dict = default_avg(),",
        "SingleParticleBasisFunctionScalarInd",
    )
    assert any(
        "the default default_avg() is not a literal" in p for p in ia.problems(copy)
    )


def test_the_check_command_exits_one_on_a_planted_class_and_zero_on_this_tree(
    copy, capsys
):
    assert ia.main(["check"]) == 0
    assert "ok: 39 instruction classes" in capsys.readouterr().out
    edit(
        copy,
        "instructions/compute.py",
        "lm_first: bool = False",
        "lm_first: bool = True",
        "ProductFunction",
    )
    assert ia.main(["check", "--root", str(copy)]) == 1
    assert "lm_first" in capsys.readouterr().out
