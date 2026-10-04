"""Tests of ``torch_backend/spec/options.py``: the option matrix and the pinned constructor defaults.

Two layers:

- logic: every cell of the scanned yamls has a status (the exit criterion of SPEC1), the committed matrix is not
  stale, the module is TF-free, the rules are well formed, and planted changes to the decisions are caught;
- physical values: the rules and defaults are compared with things the module does not call: the constructor
  signatures of the TF classes (``tests/data/instruction_constructor_defaults.json`` and ``inspect``), the
  source of the TF classes (an AST check that the keys listed as ignored are never read), and
  ``metadata_utils.resolve_param_dtype``, which decides the dtype of a yaml without metadata.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import scan_options as so  # noqa: E402

from tensorpotential.torch_backend.spec import options as opt  # noqa: E402

YAML_TARGETS = [
    ROOT / "tests/data/model_yamls",
    *sorted((ROOT / "tests").glob("model_grace*.yaml")),
]
MATRIX_DOC = ROOT / "docs/torch_backend/spec/option_matrix.md"
OPTIONS_SOURCE = Path(opt.__file__)
SIGNATURES = json.loads(
    (ROOT / "tests/data/instruction_constructor_defaults.json").read_text()
)

# Options that a yaml may only omit if the TF default is also an accepted value: these 14 defaults are not
# accepted, so a yaml has to write the key (capture_init_args always does; the loader applies DEFAULTS first and
# rejects the result). Pinned so that a change of a rule or a default shows up here.
DEFAULTS_NOT_ACCEPTED = {
    ("MLPRadialFunction", "basis"),
    ("MLPRadialFunction", "hidden_layers"),
    ("MLPRadialFunction", "activation"),
    ("MLPRadialFunction_v2", "basis"),
    ("MLPRadialFunction_v2", "hidden_layers"),
    ("MLPRadialFunction_v2", "activation"),
    ("SingleParticleBasisFunctionScalarInd", "indicator"),
    ("SingleParticleBasisFunctionEquivariantInd", "keep_parity"),
    ("SingleParticleBasisFunctionEquivariantInd", "normalize"),
    ("ProductFunction", "keep_parity"),
    ("ProductFunction", "normalize"),
    ("FCRight2Left", "n_out"),
    ("FCRight2Left", "norm_out"),
    ("LinMLPOut2ScalarTarget", "hidden_layers"),
}


def short(cls: str) -> str:
    return cls.rsplit(".", 1)[-1]


@pytest.fixture(scope="module")
def files() -> list[Path]:
    return so.expand(YAML_TARGETS)


@pytest.fixture(scope="module")
def rows(files):
    return so.scan(files)


def params(cls: str) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in SIGNATURES[cls]["parameters"]}


# ---------------------------------------------------------------------------------------------- logic


def test_scanned_yaml_set_is_the_documented_one(files):
    names = [p.name for p in files]
    assert len(names) == 13
    assert {
        "model_grace.yaml",
        "model_grace_2L_omat.yaml",
        "model_grace_2L_omat_large_base.yaml",
    } <= set(names)
    assert sum(n.startswith("GRACE-") for n in names) == 10


def test_every_scanned_value_is_supported_or_rejected_with_a_reason(rows):
    """The exit criterion: no cell of the matrix is left without a decision."""
    options = so.load_options(OPTIONS_SOURCE)
    unclassified = [r for r in rows if so.classify(r, options)[0] == "unclassified"]
    assert unclassified == []
    statuses = {so.classify(r, options)[0] for r in rows}
    assert statuses == {"supported", "ignored", "rejected"}
    for r in rows:
        status, reason = so.classify(r, options)
        assert status != "rejected" or reason, (
            f"{r.cls} {r.option}: rejected without a reason"
        )


def test_the_committed_matrix_is_current(files):
    expected = so.to_markdown(files, so.load_options(OPTIONS_SOURCE))
    assert MATRIX_DOC.read_text() == expected, (
        "docs/torch_backend/spec/option_matrix.md is stale; regenerate it with the command in its header"
    )


def test_the_models_the_first_twin_set_covers(files):
    """Five of the ten foundation models and the three yamls of tests/ need only the 17 classes."""
    options = so.load_options(OPTIONS_SOURCE)
    covered = sorted(
        p.name
        for p in files
        if not so.blockers(p, options) and p.name != "model_grace.yaml"
    )
    assert covered == [
        "GRACE-1L-OMAT-large-base.yaml",
        "GRACE-1L-OMAT-medium-base.yaml",
        "GRACE-2L-OAM.yaml",
        "GRACE-2L-OMAT-large-base.yaml",
        "GRACE-2L-OMAT-medium-base.yaml",
        "model_grace_2L_omat.yaml",
        "model_grace_2L_omat_large_base.yaml",
    ]
    assert so.blockers(
        next(p for p in files if p.name == "GRACE-2L-SMAX-large.yaml"), options
    ) == ["compute.BondSpecificRadialBasisFunction"]
    assert so.blockers(next(p for p in files if p.name == "model_grace.yaml"), options)


def test_the_module_does_not_import_tensorflow_or_torch():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('tensorflow', 'tf_keras', 'torch'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('options', {str(OPTIONS_SOURCE)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['options'] = m; spec.loader.exec_module(m)\n"
        "assert m.SUPPORTED_OPTIONS and m.DEFAULTS\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_supported_and_rejected_classes_are_disjoint_and_named_by_full_path():
    assert not set(opt.SUPPORTED_OPTIONS) & set(opt.REJECTED_CLASSES)
    assert len(opt.SUPPORTED_OPTIONS) == 17
    assert all(c.startswith((opt.COMPUTE, opt.OUTPUT)) for c in opt.SUPPORTED_OPTIONS)
    for cls in (
        *opt.SUPPORTED_OPTIONS,
        *opt.REJECTED_CLASSES,
        *opt.DEFAULTS,
        *opt.REJECTED_OPTIONS,
    ):
        assert cls.startswith((opt.BASE, opt.COMPUTE, opt.OUTPUT, opt.UQ))
    assert set(opt.DEFAULTS) <= set(opt.SUPPORTED_OPTIONS)
    assert set(opt.REJECTED_OPTIONS) <= set(opt.SUPPORTED_OPTIONS)


def test_rules_are_well_formed():
    for options in (*opt.SUPPORTED_OPTIONS.values(), opt.MODEL_OPTIONS):
        for name, rule in options.items():
            assert rule.kinds, name
            assert rule.values is None or len(rule.values) > 0, name
    for per_class in opt.REJECTED_OPTIONS.values():
        for option, rejections in per_class.items():
            assert all(r.reason for r in rejections), option


def test_a_rejection_is_for_a_value_the_rule_does_not_accept():
    """A rejection that the rule also accepts would be dead text (the rule is checked first)."""
    for cls, per_class in opt.REJECTED_OPTIONS.items():
        for option, rejections in per_class.items():
            rule = opt.SUPPORTED_OPTIONS[cls][option]
            assert all(r.kind not in rule.kinds for r in rejections), (cls, option)


def test_every_class_with_lora_config_accepts_only_null(rows):
    classes = [c for c in opt.SUPPORTED_OPTIONS if "lora_config" in params(c)]
    assert len(classes) == 7
    for cls in classes:
        rule = opt.SUPPORTED_OPTIONS[cls]["lora_config"]
        assert (rule.kinds, rule.values) == (("null",), (None,)), cls
    assert "NotImplementedError" in opt.LORA_REJECTION
    lora_rows = [r for r in rows if r.option == "lora_config"]
    assert lora_rows
    assert {(r.kind, r.values) for r in lora_rows} == {("null", (None,))}


def test_both_layout_options_are_supported(rows):
    lm_first = [c for c in opt.SUPPORTED_OPTIONS if "lm_first" in params(c)]
    assert sorted(short(c) for c in lm_first) == [
        "FCRight2Left",
        "FunctionReduceN",
        "ProductFunction",
        "SingleParticleBasisFunctionEquivariantInd",
        "SingleParticleBasisFunctionScalarInd",
    ]
    for cls in lm_first:
        rule = opt.SUPPORTED_OPTIONS[cls]["lm_first"]
        assert rule.kinds == ("bool",)
        assert rule.values is None  # True as well as False
        assert opt.DEFAULTS[cls]["lm_first"] is False
    (dense,) = [c for c in opt.SUPPORTED_OPTIONS if "dense_nbr" in params(c)]
    assert short(dense) == "SingleParticleBasisFunctionEquivariantInd"
    assert set(opt.SUPPORTED_OPTIONS[dense]["dense_nbr"].kinds) == {"bool", "null"}
    assert opt.DEFAULTS[dense]["dense_nbr"] is None
    # the scanned yamls write lm_first only as False and never write dense_nbr: True is covered by TF fixture pairs
    seen = {v for r in rows if r.option == "lm_first" for v in r.values}
    assert seen == {False}
    assert not [r for r in rows if r.option == "dense_nbr"]


def test_the_metadata_option_pins_the_dtype():
    assert opt.MODEL_OPTIONS["param_dtype"].values == ("float32", "float64")
    assert opt.MODEL_DEFAULTS == {"param_dtype": "float64"}


# --------------------------------------------------------------------------------- planted mutants


def _mutated(tmp_path: Path, old: str, new: str) -> Any:
    source = OPTIONS_SOURCE.read_text()
    assert source.count(old) >= 1, f"the mutant text {old!r} is not in options.py"
    path = tmp_path / "options_mutant.py"
    path.write_text(source.replace(old, new, 1))
    return so.load_options(path)


MUTANTS = [
    pytest.param(
        '"lora_config": _NO_LORA,\n        "lmax": _NULL,',
        '"lmax": _NULL,',
        id="drop-a-rule",
    ),
    pytest.param(
        '"basis_type": _r("str", values=("Cheb", "SBessel"))',
        '"basis_type": _r("str", values=("Cheb",))',
        id="narrow-a-value-set",
    ),
    pytest.param(
        '"n_out": _r("int", values=(1,))',
        '"n_out": _r("int", values=(2,))',
        id="change-an-allowed-number",
    ),
    pytest.param(
        '"ls_max": _r("int", "list[int]")',
        '"ls_max": _r("list[int]")',
        id="drop-a-kind",
    ),
    pytest.param(
        '"activation": _r("list[str]", values=("silu",))',
        '"activation": _r("list[str]", values=("tanh",))',
        id="change-an-activation",
    ),
    pytest.param(
        'COMPUTE + "EquivariantRMSNorm": _OUTSIDE.format(models="the 3L model"),',
        "",
        id="forget-a-rejected-class",
    ),
    pytest.param(
        '"n_out": (Rejection("null", _ONLY_3L),)',
        '"n_out": (Rejection("int", _ONLY_3L),)',
        id="rejection-for-the-wrong-kind",
    ),
]


@pytest.mark.parametrize(("old", "new"), MUTANTS)
def test_a_planted_change_of_the_decisions_leaves_a_cell_unclassified(
    tmp_path, rows, old, new
):
    mutant = _mutated(tmp_path, old, new)
    unclassified = [r for r in rows if so.classify(r, mutant)[0] == "unclassified"]
    assert unclassified, "the exit-criterion test would not notice this change"


def test_a_planted_change_of_a_default_is_caught_by_the_signature_comparison(tmp_path):
    mutant = _mutated(
        tmp_path,
        '"lm_first": False,\n        "dense_nbr": None,',
        '"lm_first": True,\n        "dense_nbr": None,',
    )
    assert _default_differences(mutant.DEFAULTS)


# ------------------------------------------------------------------------ physical values (oracles)


def _default_differences(defaults: Any) -> list[str]:
    """Differences between ``defaults`` and the constructor defaults recorded from the TF classes."""
    from tensorpotential.functions.radial import ChebSqrRadialBasisFunction

    found: list[str] = []
    for cls, expected in defaults.items():
        if (
            short(cls) == "RadialBasis"
        ):  # nfunc, rcut and the rest go to the basis function through **kwargs
            cheb = inspect.signature(ChebSqrRadialBasisFunction.__init__).parameters
            reference = {
                n: cheb[n].default for n in ("p", "normalized", "kind", "reversed")
            }
        else:
            reference = {
                n: ast.literal_eval(p["default_repr"])
                for n, p in params(cls).items()
                if "default_repr" in p and n != "name"
            }
        for name in sorted(set(reference) | set(expected)):
            if reference.get(name, "<absent>") != expected.get(name, "<absent>"):
                found.append(
                    f"{short(cls)}.{name}: TF {reference.get(name, '<absent>')!r}, pinned {expected.get(name, '<absent>')!r}"
                )
    return found


def test_pinned_defaults_equal_the_tf_constructor_defaults():
    assert _default_differences(opt.DEFAULTS) == []


def test_every_supported_class_has_its_defaults_pinned():
    no_defaults = {short(c) for c in opt.SUPPORTED_OPTIONS if c not in opt.DEFAULTS}
    assert no_defaults == {"SphericalHarmonic"}  # all its parameters are required
    assert not [
        p
        for p in params(
            next(c for c in opt.SUPPORTED_OPTIONS if short(c) == "SphericalHarmonic")
        ).values()
        if "default_repr" in p and p["name"] != "name"
    ]


def test_every_option_of_a_rule_is_a_parameter_a_consumed_keyword_or_an_ignored_key():
    consumed = {"RadialBasis": {"nfunc", "rcut", "p", "normalized", "kind", "reversed"}}
    for cls, options in opt.SUPPORTED_OPTIONS.items():
        parameters = set(params(cls)) - {"name", "kwargs"}
        extra = {name for name, rule in options.items() if name not in parameters}
        assert {n for n in extra if not options[n].ignored} <= consumed.get(
            short(cls), set()
        ), cls
        assert {n for n in options if options[n].ignored} <= extra, (
            f"{cls}: an ignored key that is a parameter"
        )
        missing = parameters - set(options) - {"name"}
        assert not missing, (
            f"{short(cls)}: constructor parameters without a rule: {sorted(missing)}"
        )


def test_ignored_keys_are_swallowed_by_kwargs_and_never_read():
    """Oracle: the TF source. An ignored key has no parameter, the class takes ``**kwargs``, and nothing reads it."""
    trees = {
        opt.COMPUTE: ast.parse(
            (ROOT / "tensorpotential/instructions/compute.py").read_text()
        ),
        opt.OUTPUT: ast.parse(
            (ROOT / "tensorpotential/instructions/output.py").read_text()
        ),
    }
    checked = 0
    for cls, options in opt.SUPPORTED_OPTIONS.items():
        if not any(rule.ignored for rule in options.values()):
            continue
        prefix = opt.COMPUTE if cls.startswith(opt.COMPUTE) else opt.OUTPUT
        (node,) = [
            n
            for n in ast.walk(trees[prefix])
            if isinstance(n, ast.ClassDef) and n.name == short(cls)
        ]
        assert "kwargs" in params(cls), f"{short(cls)} cannot swallow keys"
        reads = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Name)
            and n.id == "kwargs"
            and isinstance(n.ctx, ast.Load)
        ]
        assert not reads, f"{short(cls)} reads **kwargs (line {reads[0].lineno})"
        for name, rule in options.items():
            if rule.ignored:
                strings = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and n.value == name
                ]
                assert not strings, f"{short(cls)} mentions the ignored key {name!r}"
                checked += 1
    assert checked == 10


def test_defaults_that_are_not_accepted_values_are_the_pinned_ones():
    found = set()
    for cls, defaults in opt.DEFAULTS.items():
        for name, value in defaults.items():
            rule = opt.SUPPORTED_OPTIONS[cls][name]
            if so.value_kind(value) not in rule.kinds or (
                rule.values is not None and value not in rule.values
            ):
                found.add((short(cls), name))
    assert found == DEFAULTS_NOT_ACCEPTED


def test_old_yamls_omit_only_keys_whose_default_is_accepted():
    """The yamls written before a key existed (the two tests/ 2L yamls) load through DEFAULTS: those defaults must be accepted."""
    for name in ("model_grace_2L_omat.yaml", "model_grace_2L_omat_large_base.yaml"):
        data = yaml.safe_load((ROOT / "tests" / name).read_text())
        for ins in data.values():
            cls = ins["__cls__"]
            for option, default in opt.DEFAULTS.get(cls, {}).items():
                if option not in ins:
                    assert (short(cls), option) not in DEFAULTS_NOT_ACCEPTED, (
                        f"{name}: {short(cls)} {option}"
                    )
                    rule = opt.SUPPORTED_OPTIONS[cls][option]
                    assert so.value_kind(default) in rule.kinds


def test_param_dtype_defaults_agree_with_the_tf_resolution(tmp_path):
    import tensorflow as tf

    from tensorpotential.metadata_utils import resolve_param_dtype
    from tensorpotential.utils import get_param_dtype_from_config

    old = tmp_path / "old.yaml"
    old.write_text(
        "A: {__cls__: tensorpotential.instructions.compute.BondLength, name: A}\n"
    )
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text("metadata: {param_dtype: float32}\ninstructions: {}\n")
    no_dtype = tmp_path / "no_dtype.yaml"
    no_dtype.write_text(
        "metadata: {tensorpotential_version: '0.5.10'}\ninstructions: {}\n"
    )

    default = opt.MODEL_DEFAULTS["param_dtype"]
    assert resolve_param_dtype(str(old)) == getattr(tf, default)
    assert resolve_param_dtype(str(no_dtype)) == getattr(tf, default)
    assert resolve_param_dtype(str(wrapped)) == tf.float32
    # the training default is float32 and does not apply when a yaml is loaded: the three code defaults disagree
    assert get_param_dtype_from_config({})[1] == "float32"
    assert (
        inspect.signature(_tensor_potential_init()).parameters["param_dtype"].default
        == tf.float32
    )
    for kind in opt.MODEL_OPTIONS["param_dtype"].values or ():
        assert getattr(tf, kind).name == kind


def _tensor_potential_init():
    from tensorpotential.tensorpot import TensorPotential

    return TensorPotential.__init__


def test_scanned_metadata_param_dtype_is_float32_where_present(rows):
    (row,) = [
        r for r in rows if r.cls == so.METADATA_CLASS and r.option == "param_dtype"
    ]
    assert row.values == ("float32",)
    assert (
        len(row.files) == 8
    )  # the other five yamls have no metadata block: they are read as float64
