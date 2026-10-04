"""Tests of ``torch_backend/spec/loader.py``: the TF-free loader of ``model.yaml``.

Two layers:

- logic: the three yaml layouts, the order of the instructions, every kind of malformed input, dangling, cyclic
  and forward references, rejected classes (``model_grace.yaml`` among them), defaults, and the safety of the
  ``__cls__`` string;
- physical values: the loaded spec is compared with what TensorFlow's own ``load_instructions`` builds from the
  same file (instruction names and order, class, and every constructor argument that ``to_dict`` reports, which
  includes the defaults TF applies), for the yamls the twins accept and for the same yamls with the defaulted
  keys removed. TF is the oracle here (rule R1) and is never called by the loader.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tensorpotential.torch_backend.spec import options as opt
from tensorpotential.torch_backend.spec.errors import (
    CyclicReferenceError,
    DanglingReferenceError,
    ForwardReferenceError,
    InstructionReferenceError,
    MalformedModelError,
    SpecError,
    UnsupportedModelError,
)
from tensorpotential.torch_backend.spec.loader import (
    InstructionRef,
    InstructionSpec,
    ModelSpec,
    load_model_spec,
    _resolve,
    parse_model_spec,
)

ROOT = Path(__file__).resolve().parents[1]
YAMLS = ROOT / "tests/data/model_yamls"
OMAT = ROOT / "tests/model_grace_2L_omat.yaml"
LARGE_BASE = ROOT / "tests/model_grace_2L_omat_large_base.yaml"
LEGACY = ROOT / "tests/model_grace.yaml"

# Shipped yamls whose classes are all supported, and the layout and parameter dtype they have.
LOADABLE = {
    OMAT: ("flat", "float64"),
    LARGE_BASE: ("flat", "float64"),
    YAMLS / "GRACE-2L-OAM.yaml": ("flat", "float64"),
    YAMLS / "GRACE-1L-OMAT-large-base.yaml": ("wrapped", "float32"),
    YAMLS / "GRACE-1L-OMAT-medium-base.yaml": ("wrapped", "float32"),
    YAMLS / "GRACE-2L-OMAT-large-base.yaml": ("wrapped", "float32"),
    YAMLS / "GRACE-2L-OMAT-medium-base.yaml": ("wrapped", "float32"),
}
# Shipped yamls with at least one class outside the first twin set, and one class that each of them contains.
REJECTED = {
    LEGACY: "FunctionReduce",
    YAMLS / "GRACE-1L-SMAX-large.yaml": "BondSpecificRadialBasisFunction",
    YAMLS / "GRACE-2L-SMAX-large.yaml": "BondSpecificRadialBasisFunction",
    YAMLS / "GRACE-2L-SMAX-medium.yaml": "BondSpecificRadialBasisFunction",
    YAMLS / "GRACE-3L-OMAT-large.yaml": "SPBF",
    YAMLS / "GRACE-FS-OMAT.yaml": "LinearRadialFunction",
}

COMPUTE = opt.COMPUTE
OUTPUT = opt.OUTPUT

# Options that a yaml of the shipped models always writes although the class has a default, and that the TF
# constructor cannot do without or builds differently when absent, or whose default the strict loader rejects: not
# removed in the defaults test.
KEPT_WHEN_STRIPPING = {
    # defaults that the option rules do not accept (tests/test_options.py::DEFAULTS_NOT_ACCEPTED): without the key
    # the strict loader rejects the model
    ("ProductFunction", "normalize"),
    ("SingleParticleBasisFunctionEquivariantInd", "normalize"),
    ("FCRight2Left", "n_out"),
    ("FCRight2Left", "norm_out"),
    ("MLPRadialFunction", "basis"),
    ("MLPRadialFunction", "hidden_layers"),
    ("MLPRadialFunction", "activation"),
    ("MLPRadialFunction_v2", "basis"),
    ("MLPRadialFunction_v2", "hidden_layers"),
    ("MLPRadialFunction_v2", "activation"),
    ("SingleParticleBasisFunctionScalarInd", "indicator"),
    ("SingleParticleBasisFunctionEquivariantInd", "keep_parity"),
    ("ProductFunction", "keep_parity"),
    ("LinMLPOut2ScalarTarget", "hidden_layers"),
}

# Defaults the spec holds that TF's ``to_dict`` does not report, because they are arguments of the basis function that
# ``RadialBasis`` forwards ``**kwargs`` to, not of its own constructor (``tests/test_options.py`` pins them against
# that signature).
FORWARDED_TO_THE_BASIS = {
    ("RadialBasis", k) for k in ("p", "normalized", "kind", "reversed")
}
# The value TF reports for an option the yaml leaves as None: the constructor resolves ``dense_nbr=None`` to the
# default of the active InstructionManager (False outside one) while the pinned default stays None; the twin accepts
# both and computes the same sum (sheet SingleParticleBasisFunctionEquivariantInd, decision of 2026-10-04).
RESOLVED_BY_TF = {("SingleParticleBasisFunctionEquivariantInd", "dense_nbr"): False}


def ref(name: str) -> dict[str, Any]:
    return {"_instruction_": True, "name": name}


def entry(
    name: str, cls: str = COMPUTE + "BondLength", **options: Any
) -> dict[str, Any]:
    return {"__cls__": cls, "name": name, **options}


def reduce_entry(name: str, instructions: Any) -> dict[str, Any]:
    """A ``FunctionReduceN`` with the options the class needs, over ``instructions`` (a list or a mapping of references)."""
    return entry(
        name,
        COMPUTE + "FunctionReduceN",
        instructions=instructions,
        ls_max=0,
        n_out=1,
        allowed_l_p=[[0, 1]],
    )


def chain() -> list[dict[str, Any]]:
    """A valid three-instruction model in list layout."""
    return [
        entry("BondLength"),
        entry(
            "ScaledBondVector",
            COMPUTE + "ScaledBondVector",
            bond_length=ref("BondLength"),
        ),
        entry("Y", COMPUTE + "SphericalHarmonic", lmax=2, vhat=ref("ScaledBondVector")),
    ]


def flat(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {e["name"]: e for e in entries}


def wrapped(entries: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    return {"metadata": metadata, "instructions": flat(entries)}


# --- layouts and the shipped yamls -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"), LOADABLE.items(), ids=lambda p: getattr(p, "name", str(p))
)
def test_shipped_yamls_of_the_first_twin_set_load(path, expected):
    spec = load_model_spec(path)
    assert (spec.format, spec.param_dtype) == expected
    raw = yaml.safe_load(path.read_text())
    entries = list((raw["instructions"] if "instructions" in raw else raw).values())
    assert [s.name for s in spec.instructions] == [e["name"] for e in entries]
    assert [s.index for s in spec.instructions] == list(range(len(entries)))


def test_both_2l_yamls_load_with_their_instruction_counts():
    assert len(load_model_spec(OMAT).instructions) == 29
    assert len(load_model_spec(LARGE_BASE).instructions) == 32


@pytest.mark.parametrize(
    ("path", "cls"), REJECTED.items(), ids=lambda p: getattr(p, "name", str(p))
)
def test_shipped_yamls_outside_the_first_twin_set_are_rejected_naming_the_class(
    path, cls
):
    with pytest.raises(UnsupportedModelError) as caught:
        load_model_spec(path)
    assert cls in str(caught.value)
    assert "is not supported" in str(caught.value)


def test_model_grace_is_rejected_with_a_message_that_names_every_unsupported_instruction():
    """The legacy list-format model: one error, every instruction of an unsupported class, with the reason."""
    raw = yaml.safe_load(LEGACY.read_text())
    bad = [e for e in raw if e["__cls__"] in opt.REJECTED_CLASSES]
    assert len(bad) >= 3
    with pytest.raises(UnsupportedModelError) as caught:
        load_model_spec(LEGACY)
    message = str(caught.value)
    for e in bad:
        assert (
            f"instruction {e['name']!r}: class {e['__cls__']} is not supported"
            in message
        )
    assert "legacy list-format model_grace.yaml" in message


def test_list_layout_is_read_in_file_order():
    spec = parse_model_spec(chain())
    assert spec.format == "list"
    assert [s.name for s in spec.instructions] == [
        "BondLength",
        "ScaledBondVector",
        "Y",
    ]
    assert spec.param_dtype == "float64"


def test_wrapped_layout_reads_param_dtype_and_ignores_other_metadata():
    spec = parse_model_spec(
        wrapped(chain(), param_dtype="float32", saved_at="2026-01-01", extra=[1])
    )
    assert (spec.format, spec.param_dtype) == ("wrapped", "float32")


def test_wrapped_layout_without_metadata_is_an_old_model():
    for raw in (
        {"instructions": flat(chain())},
        {"metadata": None, "instructions": flat(chain())},
    ):
        spec = parse_model_spec(raw)
        assert (spec.format, spec.param_dtype) == (
            "wrapped",
            opt.MODEL_DEFAULTS["param_dtype"],
        )


def test_flat_layout_drops_the_legacy_metadata_sentinel_and_its_dtype():
    raw = {"__metadata__": {"param_dtype": "float32"}, **flat(chain())}
    spec = parse_model_spec(raw)
    assert [s.name for s in spec.instructions] == [
        "BondLength",
        "ScaledBondVector",
        "Y",
    ]
    assert (
        spec.param_dtype == "float64"
    )  # TF does not read it either (metadata_utils.read_model_metadata)


def test_unsupported_param_dtype_is_rejected_with_the_supported_values():
    with pytest.raises(UnsupportedModelError, match=r"float16.*float32, float64"):
        parse_model_spec(wrapped(chain(), param_dtype="float16"))


def test_loading_does_not_modify_its_input():
    raw = wrapped(chain(), param_dtype="float32")
    before = copy.deepcopy(raw)
    parse_model_spec(raw)
    assert raw == before


def test_lookup_by_name():
    spec = parse_model_spec(chain())
    assert spec["Y"].cls == COMPUTE + "SphericalHarmonic"
    with pytest.raises(KeyError):
        spec["nope"]


# --- malformed input ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, 3, "text"])
def test_a_model_is_a_mapping_or_a_list(raw):
    with pytest.raises(MalformedModelError, match="mapping or a list"):
        parse_model_spec(raw)


def test_wrapped_instructions_must_be_a_mapping():
    with pytest.raises(MalformedModelError, match="'instructions' must map"):
        parse_model_spec({"instructions": [entry("A")]})


def test_wrapped_metadata_must_be_a_mapping():
    with pytest.raises(MalformedModelError, match="'metadata' must be a mapping"):
        parse_model_spec({"metadata": [1], "instructions": flat(chain())})


@pytest.mark.parametrize("bad", [None, "BondLength", 3, ["__cls__"]])
def test_an_entry_must_be_a_mapping(bad):
    with pytest.raises(MalformedModelError, match="entry 1 of the model is a"):
        parse_model_spec([chain()[0], bad])


@pytest.mark.parametrize("name", [None, "", 3, ["A"]])
def test_an_entry_needs_a_string_name(name):
    with pytest.raises(MalformedModelError, match="entry 0 has no string 'name'"):
        parse_model_spec([{"__cls__": COMPUTE + "BondLength", "name": name}])


def test_an_entry_without_name_is_rejected_even_when_the_flat_key_gives_one():
    raw = {"BondLength": {"__cls__": COMPUTE + "BondLength"}}
    with pytest.raises(MalformedModelError, match="entry 0 has no string 'name'"):
        parse_model_spec(raw)


@pytest.mark.parametrize("cls", [None, "", 3])
def test_an_entry_needs_a_string_class(cls):
    with pytest.raises(
        MalformedModelError, match="instruction 'A' has no string '__cls__'"
    ):
        parse_model_spec([{"__cls__": cls, "name": "A"}])


def test_duplicate_names_are_rejected_in_a_list():
    with pytest.raises(
        MalformedModelError, match="'BondLength' is used twice .entries 0 and 2"
    ):
        parse_model_spec([*chain()[:2], entry("BondLength")])


def test_a_flat_key_that_differs_from_the_name_does_not_hide_a_duplicate():
    raw = {"one": entry("BondLength"), "two": entry("BondLength")}
    with pytest.raises(MalformedModelError, match="used twice"):
        parse_model_spec(raw)


@pytest.mark.parametrize("target", [None, "", 3])
def test_a_reference_needs_the_name_of_its_target(target):
    raw = [
        entry("BondLength"),
        entry(
            "S",
            COMPUTE + "ScaledBondVector",
            bond_length={"_instruction_": True, "name": target},
        ),
    ]
    with pytest.raises(MalformedModelError, match=r"S\.bond_length: a reference needs"):
        parse_model_spec(raw)


def test_a_nameless_reference_is_reported_with_its_position_inside_a_list():
    raw = [
        entry("BondLength"),
        entry(
            "F",
            COMPUTE + "FunctionReduceN",
            instructions=[ref("BondLength"), {"_instruction_": True}],
        ),
    ]
    with pytest.raises(MalformedModelError, match=r"F\.instructions\[1\]"):
        parse_model_spec(raw)


# --- references --------------------------------------------------------------------------------------------


def test_references_become_refs_and_depends_on_lists_each_target_once_in_order():
    raw = [
        entry("A"),
        entry("B"),
        reduce_entry("F", [ref("B"), ref("A"), ref("B")]),
    ]
    spec = parse_model_spec(raw)
    f = spec["F"]
    assert f.options["instructions"] == [
        InstructionRef("B"),
        InstructionRef("A"),
        InstructionRef("B"),
    ]
    assert f.depends_on == ("B", "A")
    assert spec["A"].depends_on == ()


def test_a_reference_nested_in_a_mapping_is_resolved():
    """No supported class takes a mapping of references, so the resolver is called directly."""
    refs: list[str] = []
    resolved = _resolve({"x": [ref("A")], "y": {"z": ref("B")}}, refs, "F.instructions")
    assert resolved == {"x": [InstructionRef("A")], "y": {"z": InstructionRef("B")}}
    assert refs == ["A", "B"]


def test_a_dangling_reference_names_both_instructions():
    raw = chain()
    raw[2]["vhat"] = ref("Missing")
    with pytest.raises(
        DanglingReferenceError,
        match="'Y' refers to 'Missing', which is not in the model",
    ):
        parse_model_spec(raw)


def test_a_dangling_reference_inside_a_list_is_found():
    raw = [entry("F", COMPUTE + "FunctionReduceN", instructions=[ref("Missing")])]
    with pytest.raises(DanglingReferenceError, match="'F' refers to 'Missing'"):
        parse_model_spec(raw)


def test_a_self_reference_is_a_cycle():
    raw = [entry("A", COMPUTE + "ScaledBondVector", bond_length=ref("A"))]
    with pytest.raises(CyclicReferenceError, match="'A' -> 'A'"):
        parse_model_spec(raw)


def test_a_two_cycle_is_reported_with_its_path():
    raw = [
        entry("A", COMPUTE + "ScaledBondVector", bond_length=ref("B")),
        entry("B", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
    ]
    with pytest.raises(CyclicReferenceError, match="'A' -> 'B' -> 'A'"):
        parse_model_spec(raw)


def test_a_cycle_behind_a_tail_is_reported_without_the_tail():
    raw = [
        entry("T", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
        entry("A", COMPUTE + "ScaledBondVector", bond_length=ref("B")),
        entry("B", COMPUTE + "ScaledBondVector", bond_length=ref("C")),
        entry("C", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
    ]
    with pytest.raises(CyclicReferenceError) as caught:
        parse_model_spec(raw)
    assert "'A' -> 'B' -> 'C' -> 'A'" in str(caught.value)
    assert "'T'" not in str(caught.value)


def test_a_diamond_is_not_a_cycle():
    raw = [
        entry("A"),
        entry("B", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
        entry("C", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
        reduce_entry("D", [ref("B"), ref("C")]),
    ]
    assert parse_model_spec(raw)["D"].depends_on == ("B", "C")


def test_a_ladder_of_diamonds_is_checked_in_linear_time():
    """2**levels paths reach the last rung: a check that forgot the rungs it had finished would not return."""
    levels = 40
    raw = [entry("r0")]
    for i in range(1, levels + 1):
        raw += [
            entry(f"a{i}", COMPUTE + "ScaledBondVector", bond_length=ref(f"r{i - 1}")),
            entry(f"b{i}", COMPUTE + "ScaledBondVector", bond_length=ref(f"r{i - 1}")),
            reduce_entry(f"r{i}", [ref(f"a{i}"), ref(f"b{i}")]),
        ]
    assert len(parse_model_spec(raw).instructions) == 3 * levels + 1


def test_a_reference_to_a_later_instruction_is_an_error_not_a_reordering():
    raw = chain()
    raw[0], raw[1] = raw[1], raw[0]
    with pytest.raises(
        ForwardReferenceError,
        match=r"'ScaledBondVector' \(entry 0\) refers to 'BondLength'.*\(entry 1\)",
    ):
        parse_model_spec(raw)


def test_all_reference_errors_share_a_base_type():
    for exc in (DanglingReferenceError, CyclicReferenceError, ForwardReferenceError):
        assert issubclass(exc, InstructionReferenceError)
        assert issubclass(exc, SpecError)
        assert issubclass(exc, ValueError)


def test_a_cycle_is_reported_as_a_cycle_even_though_it_also_has_a_forward_reference():
    raw = [
        entry("A", COMPUTE + "ScaledBondVector", bond_length=ref("B")),
        entry("B", COMPUTE + "ScaledBondVector", bond_length=ref("A")),
    ]
    with pytest.raises(CyclicReferenceError):
        parse_model_spec(raw)


@pytest.mark.parametrize("path", [OMAT, LARGE_BASE])
def test_output_instructions_keep_their_file_order_and_read_earlier_ones(path):
    """The energy chain overwrites its target in place, so its order is the model."""
    raw = yaml.safe_load(path.read_text())
    spec = load_model_spec(path)
    in_file = [name for name, e in raw.items() if e["__cls__"].startswith(OUTPUT)]
    in_spec = [s.name for s in spec.instructions if s.cls.startswith(OUTPUT)]
    assert in_spec == in_file
    assert len(in_spec) >= 3
    position = {s.name: s.index for s in spec.instructions}
    for s in spec.instructions:
        assert all(position[d] < s.index for d in s.depends_on)


def test_outputs_in_a_different_order_than_their_targets_are_rejected():
    """A shift that reads the energy before the instruction that creates it comes later in the file."""
    out = OUTPUT
    raw = [
        entry(
            "shift",
            out + "ConstantScaleShiftTarget",
            target=ref("atomic_energy"),
            scale=1.0,
            shift=0.0,
        ),
        entry("atomic_energy", out + "CreateOutputTarget", initial_value=0.0),
    ]
    with pytest.raises(
        ForwardReferenceError, match="'shift' .entry 0. refers to 'atomic_energy'"
    ):
        parse_model_spec(raw)


# --- classes and defaults ----------------------------------------------------------------------------------


def test_an_unknown_class_is_rejected_as_unknown_not_as_listed():
    with pytest.raises(
        UnsupportedModelError,
        match=r"instruction 'A': class pkg\.Mystery is unknown to the torch backend",
    ):
        parse_model_spec([entry("A", "pkg.Mystery")])


def test_every_unsupported_instruction_is_listed_in_one_error():
    raw = [
        entry("A", "pkg.Mystery"),
        entry("B", COMPUTE + "FunctionReduce"),
        *[entry("C")],
    ]
    with pytest.raises(UnsupportedModelError) as caught:
        parse_model_spec(raw)
    lines = str(caught.value).splitlines()
    assert sum("instruction" in line for line in lines) == 2
    assert "'A'" in lines[1] and "'B'" in lines[2]


def test_a_cls_string_is_looked_up_never_executed():
    """TF's str_to_class runs exec on it; here a statement is just an unknown class."""
    sentinel = "os.path import join as EXTRACTED_C; EXTRACTED_C = len #.join"
    with pytest.raises(UnsupportedModelError, match="is unknown to the torch backend"):
        parse_model_spec([entry("A", sentinel)])


def test_the_3l_model_lists_every_kind_of_finding_in_one_error():
    """Real yaml, every rejection kind at once: 14 instructions of rejected classes and 7 rejected option values."""
    with pytest.raises(UnsupportedModelError) as caught:
        load_model_spec(YAMLS / "GRACE-3L-OMAT-large.yaml")
    kinds = {}
    for problem in caught.value.problems:
        kinds.setdefault(problem.kind, set()).add(problem.cls.rsplit(".", 1)[-1])
    assert kinds == {
        "rejected_class": {"SPBF", "GeneralProductFunction", "EquivariantRMSNorm"},
        "rejected_value": {"FCRight2Left", "ConstantScaleShiftTarget"},
    }
    assert len(caught.value.problems) == 21
    assert str(caught.value).count("\n  instruction ") == 21


def test_an_unknown_option_of_a_supported_class_is_rejected_naming_the_nearest_key():
    raw = chain()
    raw[2]["lmaxx"] = 3
    with pytest.raises(
        UnsupportedModelError, match=r"unknown option 'lmaxx'; did you mean 'lmax'"
    ) as caught:
        parse_model_spec(raw)
    assert [(p.kind, p.instruction) for p in caught.value.problems] == [
        ("unknown_option", "Y")
    ]


def test_a_missing_required_option_is_rejected_naming_it():
    raw = chain()
    del raw[1]["bond_length"]
    with pytest.raises(
        UnsupportedModelError, match="required option 'bond_length' is missing"
    ):
        parse_model_spec(raw)


def test_an_unsupported_value_is_rejected_with_the_supported_ones():
    raw = chain()
    raw[2]["lmax"] = "two"
    with pytest.raises(
        UnsupportedModelError,
        match=r"option 'lmax' = 'two' \(str\) is not supported \(supported kinds: int\)",
    ):
        parse_model_spec(raw)


def test_a_class_is_matched_by_its_exact_name():
    """A subclass of a supported class (here a class of another module with the same short name) is not supported."""
    raw = chain()
    raw[0]["__cls__"] = "mypackage.instructions.BondLength"
    with pytest.raises(
        UnsupportedModelError, match="classes are matched by exact name"
    ) as caught:
        parse_model_spec(raw)
    assert caught.value.problems[0].kind == "unknown_class"


def test_findings_of_several_instructions_are_all_listed():
    raw = chain()
    raw[0]["__cls__"] = COMPUTE + "SPBF"
    raw[2]["lmax"] = "two"
    raw[2]["extra"] = 1
    with pytest.raises(UnsupportedModelError) as caught:
        parse_model_spec(raw)
    assert [(p.instruction, p.kind) for p in caught.value.problems] == [
        ("BondLength", "rejected_class"),
        ("Y", "unknown_option"),
        ("Y", "unsupported_value"),
    ]


def test_the_keys_the_defaults_filled_in_are_recorded():
    spec = parse_model_spec(chain())
    assert spec["BondLength"].defaulted == {"instruction_with_bonds"}
    assert spec["ScaledBondVector"].defaulted == {"bonds"}
    assert spec["Y"].defaulted == frozenset()
    raw = chain()
    raw[0]["instruction_with_bonds"] = None
    assert parse_model_spec(raw)["BondLength"].defaulted == frozenset()


def test_a_key_the_tf_constructor_ignores_still_loads():
    """The shipped yamls carry ``radia_basis`` and ``init_norm`` (accepted, never read by TF)."""
    spec = load_model_spec(OMAT)
    assert "init_norm" in spec["MLPOut2ScalarTarget"].options
    assert "radia_basis" in spec["YI"].options


def test_python_tags_in_the_file_are_refused_by_the_safe_loader(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("A: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(yaml.YAMLError):
        load_model_spec(path)


def test_defaults_fill_the_keys_the_yaml_omits_and_keep_the_ones_it_writes():
    cls = COMPUTE + "SingleParticleBasisFunctionScalarInd"
    base = chain()[:2]
    raw = [
        *base,
        entry(
            "Z",
            COMPUTE + "ScalarChemicalEmbedding",
            element_map={"Cu": 0},
            embedding_size=4,
        ),
        entry(
            "R",
            COMPUTE + "MLPRadialFunction",
            n_rad_max=2,
            lmax=1,
            basis=ref("ScaledBondVector"),
            hidden_layers=[8],
            activation="tanh",
        ),
        entry("Y", COMPUTE + "SphericalHarmonic", lmax=2, vhat=ref("ScaledBondVector")),
        entry(
            "A",
            cls,
            radial=ref("R"),
            angular=ref("Y"),
            indicator=ref("Z"),
            avg_n_neigh=3.5,
            lm_first=True,
        ),
    ]
    options = parse_model_spec(raw)["A"].options
    expected = opt.DEFAULTS[cls]
    assert (
        options["lm_first"] is True and options["avg_n_neigh"] == 3.5
    )  # written values win
    assert (
        options["sum_neighbors"] is expected["sum_neighbors"]
    )  # omitted: the pinned default
    assert options["lmax"] is None and options["lora_config"] is None
    assert set(options) == {
        "radial",
        "angular",
        "indicator",
        "avg_n_neigh",
        "lm_first",
        *expected,
    }


def test_a_value_written_as_null_is_not_replaced_by_the_default():
    raw = chain()
    raw[0]["instruction_with_bonds"] = None
    spec = parse_model_spec(raw)
    assert "instruction_with_bonds" in spec["BondLength"].options
    cls = COMPUTE + "FCRight2Left"
    raw = [
        *chain()[:1],
        entry(
            "L",
            cls,
            left=ref("BondLength"),
            right=ref("BondLength"),
            n_out=3,
            left_coefs=None,
        ),
    ]
    # left_coefs accepts only a bool, so the written null is reported (it would be silently True had the default
    # replaced it)
    with pytest.raises(UnsupportedModelError, match=r"left_coefs.* = None") as caught:
        parse_model_spec(raw)
    (problem,) = [p for p in caught.value.problems if p.option == "left_coefs"]
    assert problem.kind == "unsupported_value"
    assert "absent from the yaml" not in problem.message


def test_a_class_without_defaults_gets_no_extra_keys():
    spec = parse_model_spec(chain())
    assert set(spec["Y"].options) == {"lmax", "vhat"}


def test_options_are_read_only_and_exclude_the_class_and_the_name():
    spec = parse_model_spec(chain())
    options = spec["Y"].options
    assert "__cls__" not in options and "name" not in options
    with pytest.raises(TypeError):
        options["lmax"] = 3  # ty: ignore[invalid-assignment]


def test_specs_are_frozen_values():
    spec = parse_model_spec(chain())
    assert isinstance(spec, ModelSpec) and isinstance(
        spec.instructions[0], InstructionSpec
    )
    with pytest.raises(AttributeError):
        spec.param_dtype = "float32"  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        spec.instructions[0].name = "x"  # ty: ignore[invalid-assignment]
    assert InstructionRef("A") == InstructionRef("A") and hash(
        InstructionRef("A")
    ) == hash(InstructionRef("A"))


# --- physical values: TensorFlow's own loader is the oracle ----------------------------------------------


def _tf_instruction_dicts(path: Path) -> dict[str, dict[str, Any]]:
    """What TF builds from the file: name -> ``to_dict()`` with the instruction objects replaced by their names."""
    import tensorpotential  # noqa: F401  (first TensorFlow-side import runs the options)
    from tensorpotential.instructions.base import (
        load_instructions,
        recursive_walk_and_modify,
        replace_TPInstruction_to_name,
    )

    loaded = load_instructions(str(path))
    result = {}
    for name, instruction in loaded.items():
        dct = instruction.to_dict()
        recursive_walk_and_modify(dct, replace_TPInstruction_to_name)
        result[name] = dct
    return result


def _as_tf_dict(value: Any) -> Any:
    """The spec's options in the form of ``to_dict``: a reference is ``{_instruction_: True, name: ...}``."""
    if isinstance(value, InstructionRef):
        return ref(value.name)
    if isinstance(value, dict):
        return {k: _as_tf_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_tf_dict(v) for v in value]
    return value


def _comparable(s: InstructionSpec, oracle: dict[str, Any]) -> dict[str, Any]:
    """The options of ``s`` without the forwarded defaults that TF does not report."""
    short = s.cls.rsplit(".", 1)[-1]
    options = {
        k: RESOLVED_BY_TF[(short, k)]
        if v is None and (short, k) in RESOLVED_BY_TF
        else v
        for k, v in s.options.items()
        if k in oracle or (short, k) not in FORWARDED_TO_THE_BASIS
    }
    return _as_tf_dict(options)


def _same_as_tf(path: Path) -> None:
    spec = load_model_spec(path)
    oracle = _tf_instruction_dicts(path)
    assert [s.name for s in spec.instructions] == list(oracle)
    for s in spec.instructions:
        expected = dict(oracle[s.name])
        assert expected.pop("__cls__") == s.cls
        assert expected.pop("name") == s.name
        assert _comparable(s, expected) == expected, s.name


@pytest.mark.parametrize(
    "path",
    [OMAT, LARGE_BASE, YAMLS / "GRACE-2L-OMAT-medium-base.yaml"],
    ids=lambda p: p.name,
)
def test_spec_equals_what_tf_loads(path):
    _same_as_tf(path)


def _strip_defaulted_keys(raw: dict[str, Any]) -> tuple[dict[str, Any], int]:
    stripped = copy.deepcopy(raw)
    removed = 0
    for e in stripped.values():
        short = e["__cls__"].rsplit(".", 1)[-1]
        for key in opt.DEFAULTS.get(e["__cls__"], {}):
            if key in e and (short, key) not in KEPT_WHEN_STRIPPING:
                del e[key]
                removed += 1
    return stripped, removed


@pytest.mark.parametrize("path", [OMAT, LARGE_BASE], ids=lambda p: p.name)
def test_defaults_applied_to_a_yaml_without_them_equal_the_defaults_tf_applies(
    path, tmp_path
):
    """Remove every defaulted key from a model: TF fills them from its constructors, the loader from DEFAULTS."""
    stripped, removed = _strip_defaulted_keys(yaml.safe_load(path.read_text()))
    assert removed > 40
    target = tmp_path / "stripped.yaml"
    target.write_text(yaml.safe_dump(stripped, sort_keys=False))
    spec = load_model_spec(target)
    oracle = _tf_instruction_dicts(target)
    for s in spec.instructions:
        expected = dict(oracle[s.name])
        expected.pop("__cls__"), expected.pop("name")
        assert _comparable(s, expected) == expected, s.name
