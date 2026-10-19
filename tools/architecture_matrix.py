"""Architecture matrix: every model architecture the TensorFlow side can build, and what the torch backend covers of it.

A row is one architecture: a preset of ``tensorpotential/potentials/presets.py`` with its default arguments, the same
preset with one argument changed (a *variant*: an argument value that changes the class set or an option value),
a model file of the repository (``tests/data/model_yamls/*.yaml``, ``tests/model_*.yaml``) or a tensor model of
``tensorpotential/extra/gen_tensor/model.py``. Each row is run through the strict check of the torch loader
(:func:`tensorpotential.torch_backend.spec.loader.parse_model_spec`, in both parameter dtypes) and records its classes,
the option values they hold and the *gaps*: every class or option value the loader rejects, each with the board issue
that closes it (:data:`GAP_ISSUES`). A gap without an issue is written as ``unassigned``; the completeness test
(``tests_torch/test_architectures.py``) fails on it.

Usage (from the repository root)::

    python tools/architecture_matrix.py write   # TensorFlow job: docs/torch_backend/architectures.json and .md
    python tools/architecture_matrix.py table   # docs/torch_backend/architectures.md from the committed JSON

``write`` builds the presets with the real TensorFlow classes; the completeness test builds them again with the
TensorFlow stand-in of ``tests_torch/tools/tfstub.py`` (no TensorFlow) and requires the same document, so the JSON
cannot go stale unnoticed. The Markdown table adds what the tree holds now (golden fixture, twin golden tests,
converter and real-weight checks), so it is regenerated whenever one of those changes.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

ROOT: Final = Path(__file__).resolve().parents[1]
# appended, as in make_golden.py: the library imported is the one on PYTHONPATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensorpotential.torch_backend.errors import (  # noqa: E402
    SpecError,
    UnsupportedModelError,
)
from tensorpotential.torch_backend.spec.loader import parse_model_spec  # noqa: E402
from tensorpotential.torch_backend.spec.registry import (  # noqa: E402
    GOLDEN_TESTS,
    REGISTRY,
)

JSON_PATH: Final = ROOT / "docs" / "torch_backend" / "architectures.json"
MD_PATH: Final = ROOT / "docs" / "torch_backend" / "architectures.md"
FORMAT_VERSION: Final = 1

ELEMENT_MAP: Final = {"H": 0, "O": 1}
"""Two elements: enough for every element-pair table, and small enough to keep the JSON short."""
DTYPES: Final = ("float32", "float64")
PRESETS: Final = (
    "LINEAR",
    "FS",
    "GRACE_1LAYER_v1_24",
    "GRACE_2LAYER_v1_24",
    "GRACE_2LAYER_scaled",
    "GRACE_1LAYER_v2_25",
    "GRACE_2LAYER_v2_25",
)
"""The preset functions of ``potentials/presets.py``, by function name (``GRACE_2LAYER_scaled`` is not registered)."""
TENSOR_MODELS: Final = ("grace_1", "grace_2")
MODEL_FILE_GLOBS: Final = ("tests/data/model_yamls/*.yaml", "tests/model_*.yaml")

# A preset argument and the values that change the class set or an option value; each value that differs from the
# preset's default makes one variant row (one argument at a time). Numeric widths are left out: they change sizes,
# not the classes or the kind of an option.
ARGUMENT_VALUES: Final[Mapping[str, tuple[Any, ...]]] = {
    "basis_type": ("Cheb", "SBessel", "Gaussian", "RadSinBessel"),
    "cutoff_dict": ({"HH": 4.0, "HO": 4.5, "OO": 5.0},),
    "zbl_cutoff": (1.0,),
    "dense_nbr": (True,),
    "mlp_radial": (False,),
    "atomic_shift_map": (
        {0: -1.0, 1: -2.0},
    ),  # element index -> energy, as gracemaker passes it
    "func_init": ("zeros",),
    "chem_init": ("zeros", "delta"),
    "crad_init": ("delta",),
    "norm_out": ("layer", None),
    "full_origin_norm": (True,),
    "normalize_prod": (True, False),
    "parity": ("full",),
    "simplify_prod": (False,),
    "compute_energy": (True,),
    "tensor_components": ([1], [2]),
}

# The issue that closes a gap, keyed by (class name, option); the option ``None`` stands for the class itself.
GAP_ISSUES: Final[Mapping[tuple[str, str | None], str]] = {
    ("BondSpecificRadialBasisFunction", None): "TWIN11",
    ("EquivariantRMSNorm", None): "TWIN12",
    ("GeneralProductFunction", None): "TWIN13",
    ("SPBF", None): "TWIN14",
    ("FCRight2Left", "n_out"): "TWIN14",
    ("LinearRadialFunction", None): "TWIN15",
    ("CropProductFunction", None): "TWIN15",
    ("FSOut2ScalarTarget", None): "TWIN15",
    ("FunctionReduceN", "n_in"): "TWIN15",
    ("LinearOut2Target", None): "TWIN16",
    ("MLPOut2ScalarTarget", None): "TWIN16",
    ("ZBLPotential", None): "TWIN17",
    ("RadialBasis", "basis_type"): "RAD2",
    ("FunctionReduceN", "init_vars"): "SPEC6",
    ("ScalarChemicalEmbedding", "init"): "SPEC6",
    ("FunctionReduce", None): "TWIN16",
    ("MLPRadialFunction", "activation"): "TWIN16",
    ("MLPRadialFunction", "hidden_layers"): "TWIN16",
    ("SingleParticleBasisFunctionEquivariantInd", "normalize"): "TWIN16",
    ("ProductFunction", "normalize"): "TWIN16",
    ("CreateOutputTarget", "l"): "TWIN19",
    ("FunctionReduceParticular", None): "TWIN19",
    ("LinearOut2EquivarTarget", None): "TWIN19",
    # classes that no architecture uses (gap (g)): TWIN18 records the rejection of each
    ("BondAvgSphericalHarmonic", None): "TWIN18",
    ("CollectInvarBasis", None): "TWIN18",
    ("EquivariantGate", None): "TWIN18",
    ("LinMLPScalarReadOut", None): "TWIN18",
    ("TrainableShiftTarget_v2", None): "TWIN18",
}
UNASSIGNED: Final = "unassigned"
REJECTED: Final = "rejected"

# Gaps the owner decided not to close, with the reason (written into the row instead of an issue).
GAP_REJECTIONS: Final[Mapping[tuple[str, str | None], str]] = {
    ("FunctionReduceN", "scale"): (
        "used only by GRACE_2LAYER_scaled, a preset function that is not registered, so no training run selects it "
        "(owner decision, 2026-10-19)"
    ),
}

GOLDEN_DIR: Final = ROOT / "tests_torch" / "golden"
"""The committed tiny fixtures of ``tools/make_golden.py`` (its ``GOLDEN_DIR``)."""
DTYPE_OF_LABEL: Final = {"f64": "float64", "f32": "float32"}
"""The dtype labels of the fixture names (``make_golden.DTYPES``), as a manifest's ``not_made`` uses them."""

# Row sources without a golden fixture, with the reason (``fixture_cell``); every other row has one (FAM2).
FIXTURE_FREE: Final[Mapping[str, str]] = {
    "tensor": "TWIN19",
}

# Rows whose converter equivalence (EQUIV5) or real-weight check (EQUIV6) exists; those issues add their rows.
CONVERTER_CHECKED: Final[frozenset[str]] = frozenset()
REAL_WEIGHT_CHECKED: Final[frozenset[str]] = frozenset()

# Foundation models without a model file in the repository, mapped to the row of their architecture by their name
# (not verified: EQUIV6 checks the real files on the HPC).
FOUNDATION_BY_NAME: Final[Mapping[str, str]] = {
    "GRACE-FS-OAM": "file:GRACE-FS-OMAT",
    "GRACE-1L-OAM": "preset:GRACE_1LAYER_v1_24",
    "GRACE-1L-OMAT": "preset:GRACE_1LAYER_v1_24",
    "GRACE-2L-OMAT": "file:GRACE-2L-OAM",
    "GRACE-1L-SMAX-OMAT-large": "file:GRACE-1L-SMAX-large",
    "GRACE-2L-SMAX-OMAT-large": "file:GRACE-2L-SMAX-large",
    "GRACE-2L-SMAX-OMAT-medium": "file:GRACE-2L-SMAX-medium",
}
PRECISION_SUFFIX: Final = re.compile(r"-fp64$")
FINE_TUNE_SUFFIX: Final = re.compile(r"-ft-(E|AM)$")
SHORT_VALUE: Final = 60
"""An option value whose JSON is longer than this many characters is written as its kind."""


# ------------------------------------------------------------------ instruction dicts


def _plain(value: Any) -> Any:
    """``value`` with tuples as lists and numpy scalars as Python ones (what a yaml holds)."""
    if isinstance(value, Mapping):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def serialise(instructions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the safe-yaml form of the ``instructions`` block of the saved ``model.yaml``, in order."""
    raw = {name: _plain(ins.to_dict()) for name, ins in instructions.items()}
    return yaml.safe_load(yaml.safe_dump(raw, sort_keys=False))


def _factory(module: str, name: str) -> Callable[..., Any]:
    """Return the preset function or tensor-model function (TensorFlow or its stand-in must be importable)."""
    if module == "preset":
        from tensorpotential.potentials import presets

        return getattr(presets, name)
    from tensorpotential.extra.gen_tensor import model

    return getattr(model, name)


def build_instructions(
    module: str,
    name: str,
    arguments: Mapping[str, Any],
    element_map: Mapping[str, int] = ELEMENT_MAP,
) -> dict[str, dict[str, Any]]:
    """Build a preset or tensor model with ``arguments`` and return its serialised instructions."""
    built = _factory(module, name)(element_map=dict(element_map), **arguments)
    return serialise(built.get_instructions() if module == "preset" else built)


def variants(module: str, name: str) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, arguments)`` of the default build and of each one-argument variant of a preset or tensor model."""
    parameters = inspect.signature(_factory(module, name)).parameters
    return [("", {})] + [
        (f"{argument}={json.dumps(value, sort_keys=True)}", {argument: value})
        for argument, values in ARGUMENT_VALUES.items()
        if argument in parameters
        for value in values
        if value != parameters[argument].default
    ]


# ------------------------------------------------------------------ rows


def _short(value: Any) -> Any:
    from scan_options import value_kind

    if isinstance(value, dict) and "_instruction_" in value:
        return "<ref>"
    text = json.dumps(value, sort_keys=True)
    return value if len(text) <= SHORT_VALUE else f"<{value_kind(value)}>"


def class_options(
    instructions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    """``{class path: {option: distinct values}}``; long values are replaced by their kind."""
    table: dict[str, dict[str, list[Any]]] = {}
    for ins in instructions.values():
        options = table.setdefault(ins["__cls__"], {})
        for key, value in ins.items():
            if key in {"__cls__", "name"}:
                continue
            seen = options.setdefault(key, [])
            shown = _short(value)
            if shown not in seen:
                seen.append(shown)
    for options in table.values():
        for key, values in options.items():
            options[key] = sorted(values, key=lambda v: json.dumps(v, sort_keys=True))
    return dict(
        sorted((cls, dict(sorted(opts.items()))) for cls, opts in table.items())
    )


def _short_name(cls: str) -> str:
    return cls.rsplit(".", 1)[-1]


def _closed_by(cls: str, option: str | None) -> dict[str, str]:
    short = _short_name(cls)
    if (short, option) in GAP_REJECTIONS:
        return {"issue": REJECTED, "reason": GAP_REJECTIONS[short, option]}
    return {
        "issue": GAP_ISSUES.get((short, option))
        or GAP_ISSUES.get((short, None))
        or UNASSIGNED
    }


def gaps_of(raw: object) -> list[dict[str, Any]]:
    """Every class and option value of a model that the strict loader rejects, each with its issue."""
    try:
        parse_model_spec(raw)
    except UnsupportedModelError as error:
        found = {(p.kind, p.cls, p.option) for p in error.problems}
    except SpecError as error:
        return [
            {
                "kind": type(error).__name__,
                "class": None,
                "option": None,
                "issue": UNASSIGNED,
            }
        ]
    else:
        return []
    return [
        {"kind": kind, "class": cls, "option": option} | _closed_by(cls, option)
        for kind, cls, option in sorted(found, key=lambda t: (t[1], t[2] or "", t[0]))
    ]


def _gaps_all_dtypes(instructions: Mapping[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for gap in gaps_of({
            "metadata": {"param_dtype": dtype},
            "instructions": dict(instructions),
        }):
            if gap not in gaps:
                gaps.append(gap)
    return gaps


def _diff(
    base: Mapping[str, Mapping[str, list[Any]]],
    other: Mapping[str, Mapping[str, list[Any]]],
) -> dict[str, Any]:
    changed = {
        cls: {
            key: values
            for key, values in options.items()
            if base[cls].get(key) != values
        }
        for cls, options in other.items()
        if cls in base
    }
    return {
        "classes_added": sorted(set(other) - set(base)),
        "classes_removed": sorted(set(base) - set(other)),
        "options_changed": {cls: opts for cls, opts in changed.items() if opts},
    }


def row_id(module: str, name: str, label: str = "") -> str:
    """Return the id of a preset or tensor-model row: ``preset:FS``, ``preset:FS[basis_type="Gaussian"]`` for a variant."""
    return f"{module}:{name}" + (f"[{label}]" if label else "")


def file_row_id(path: Path) -> str:
    """Return the id of the row of a model file: ``file:<stem>``."""
    return f"file:{path.stem}"


def generated_rows(module: str, name: str) -> list[dict[str, Any]]:
    """Build the base row of a preset or tensor model and its variant rows."""
    base_id = row_id(module, name)
    rows: list[dict[str, Any]] = []
    base_classes: dict[str, dict[str, list[Any]]] = {}
    for label, arguments in variants(module, name):
        instructions = build_instructions(module, name, arguments)
        classes = class_options(instructions)
        row: dict[str, Any] = {
            "id": row_id(module, name, label),
            "source": module,
            "name": name,
        }
        if label:
            row |= {"base": base_id, "arguments": arguments} | _diff(
                base_classes, classes
            )
        else:
            base_classes = classes
            row |= {"param_dtypes": list(DTYPES), "classes": classes}
        rows.append(row | {"gaps": _gaps_all_dtypes(instructions)})
    return rows


def model_files() -> list[Path]:
    """Return the model files of the repository, sorted by path."""
    return sorted(path for pattern in MODEL_FILE_GLOBS for path in ROOT.glob(pattern))


def file_row(path: Path) -> dict[str, Any]:
    """Build the row of a model file (its own parameter dtype; old files without one are float64)."""
    from scan_options import instruction_dicts

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata, entries = instruction_dicts(raw)
    instructions = {
        str(entry.get("name", index)): entry for index, entry in enumerate(entries)
    }
    return {
        "id": file_row_id(path),
        "source": "file",
        "name": path.relative_to(ROOT).as_posix(),
        "param_dtypes": [metadata.get("param_dtype", "float64")],
        "classes": class_options(instructions),
        "gaps": gaps_of(raw),
    }


def _base_name(name: str, files: set[str]) -> str:
    """Return the model that ``name`` was fine-tuned from: ``X-ft-E`` comes from ``X-base`` or ``X``."""
    if not FINE_TUNE_SUFFIX.search(name):
        return name
    stem = FINE_TUNE_SUFFIX.sub("", name)
    return f"{stem}-base" if f"{stem}-base" in files else stem


def foundation_rows(
    names: Iterable[str], aliases: Mapping[str, str], row_ids: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Map every foundation-model name (and alias) to its architecture row, with how the mapping is known."""
    files = {
        row_id.removeprefix("file:") for row_id in row_ids if row_id.startswith("file:")
    }
    table: dict[str, dict[str, str]] = {}
    for name in sorted(set(names) | set(aliases)):
        base = _base_name(PRECISION_SUFFIX.sub("", aliases.get(name, name)), files)
        if base in files:
            how = (
                "model file"
                if base == name
                else f"architecture of {base} (fine-tuned weights, alias or fp64 build), which has a model file"
            )
            table[name] = {"row": f"file:{base}", "evidence": how}
        elif base in FOUNDATION_BY_NAME:
            table[name] = {
                "row": FOUNDATION_BY_NAME[base],
                "evidence": f"by name ({base}): no model file here",
            }
        else:
            table[name] = {"row": UNASSIGNED, "evidence": "no rule"}
    return table


def unused_classes(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Registry classes that no row uses, with the decision recorded for them and its issue (gap (g) of FAM1)."""
    used = {cls for row in rows for cls in row.get("classes", {})}
    used |= {cls for row in rows for cls in row.get("classes_added", [])}
    return {
        path: {
            "decision": entry.reason or "supported",
            "issue": GAP_ISSUES.get((entry.name, None), "-"),
        }
        for path, entry in REGISTRY.items()
        if path not in used
    }


def build_document(names: Iterable[str], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Build the whole matrix: rows, foundation-model map, unused classes (needs TensorFlow or its stand-in)."""
    rows: list[dict[str, Any]] = []
    for name in PRESETS:
        rows += generated_rows("preset", name)
    rows += [file_row(path) for path in model_files()]
    for name in TENSOR_MODELS:
        rows += generated_rows("tensor", name)
    return {
        "format_version": FORMAT_VERSION,
        "element_map": dict(ELEMENT_MAP),
        "rows": rows,
        "foundation_models": foundation_rows(
            names, aliases, (row["id"] for row in rows)
        ),
        "unused_classes": unused_classes(rows),
    }


def foundation_names() -> tuple[list[str], dict[str, str]]:
    """Return the built-in foundation-model names and aliases (not the user registry, not experimental models)."""
    from tensorpotential.calculator import foundation_models as fm

    return list(fm.CORE_MODELS_NAME_LIST), dict(fm.MODELS_ALIASES_DICT)


def dumps(document: Mapping[str, Any]) -> str:
    """Return the committed JSON text (sorted keys, two-space indent, final newline)."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


# ------------------------------------------------------------------ table


def _gap_cell(gaps: Sequence[Mapping[str, Any]]) -> str:
    if not gaps:
        return "none"
    cells = []
    for gap in gaps:
        what = _short_name(gap["class"] or gap["kind"])
        if gap["option"]:
            what += f".{gap['option']}"
        cells.append(f"{what} ({gap['issue']})")
    return ", ".join(sorted(set(cells)))


@functools.cache
def fixture_dtypes() -> dict[str, frozenset[str]]:
    """Return, per row, the parameter dtypes its committed tiny fixtures cover, from the manifests' ``row``.

    A dtype that TensorFlow cannot evaluate for the row is covered by the reason its manifests give
    (``not_made`` of ``tools/make_golden.py``). An option pair (``dense_nbr``, ``lm_first``) is a second layout of
    its row's float64 fixture and is not counted.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("tier") != "tiny" or not manifest.get("row"):
            continue
        dtypes = found.setdefault(manifest["row"], set())
        dtypes |= {
            DTYPE_OF_LABEL[k]
            for k in manifest.get("not_made", {})
            if k in DTYPE_OF_LABEL
        }
        if not manifest.get("option"):
            dtypes.add(manifest["dtype"])
    return {row: frozenset(dtypes) for row, dtypes in found.items()}


def fixture_present(row: Mapping[str, Any]) -> bool:
    """Whether the row has a committed tiny golden fixture in every parameter dtype TensorFlow can evaluate."""
    return set(DTYPES) <= fixture_dtypes().get(row["id"], frozenset())


def fixture_cell(row: Mapping[str, Any]) -> str:
    """``yes``, ``no``, or ``free (<issue>)`` for a row whose source is fixture-free."""
    if row["source"] in FIXTURE_FREE:
        return f"free ({FIXTURE_FREE[row['source']]})"
    return _yes(fixture_present(row))


def twin_tests(row: Mapping[str, Any]) -> str:
    """``yes`` when every class of the row has a twin golden test, else how many have one."""
    classes = list(row.get("classes", {})) + list(row.get("classes_added", []))
    covered = sum(cls in GOLDEN_TESTS for cls in classes)
    return "yes" if covered == len(classes) else f"{covered}/{len(classes)}"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _row_line(row: Mapping[str, Any]) -> str:
    classes = ", ".join(_short_name(c) for c in row["classes"])
    cells = (
        f"`{row['id']}`",
        row["name"] if row["source"] == "file" else row["source"],
        classes,
        _gap_cell(row["gaps"]),
        fixture_cell(row),
        twin_tests(row),
        _yes(row["id"] in CONVERTER_CHECKED),
        _yes(row["id"] in REAL_WEIGHT_CHECKED),
    )
    return "| " + " | ".join(cells) + " |"


def _variant_line(row: Mapping[str, Any]) -> str:
    added = ", ".join(_short_name(c) for c in row["classes_added"]) or "-"
    changed = (
        "; ".join(
            f"{_short_name(cls)}: {', '.join(opts)}"
            for cls, opts in row["options_changed"].items()
        )
        or "-"
    )
    argument = ", ".join(
        f"{k}={json.dumps(v, sort_keys=True)}" for k, v in row["arguments"].items()
    )
    return f"| `{row['base']}` | `{argument}` | {added} | {changed} | {_gap_cell(row['gaps'])} | {fixture_cell(row)} |"


TABLE_HEADER: Final = """# Architecture matrix

Generated by `tools/architecture_matrix.py table` from `architectures.json` (written by
`tools/architecture_matrix.py write`); do not edit by hand. `tests_torch/test_architectures.py` fails when either
file is stale, when a gap has no issue and when a foundation model has no row. A gap is a class or an option value
that the strict torch loader rejects today, in either parameter dtype, with the issue that closes it.
"""


def to_markdown(document: Mapping[str, Any]) -> str:
    """Return the Markdown table of a matrix document, with the tree's current fixtures and checks."""
    rows = document["rows"]
    lines = [
        TABLE_HEADER,
        "## Architectures",
        "",
        "| Row | Source | Classes | Gaps (issue) | Fixture (FAM2) | Twin golden tests | Converter (EQUIV5) | Real weights (EQUIV6) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [_row_line(row) for row in rows if "base" not in row]
    lines += [
        "",
        "## Preset variants (one argument changed)",
        "",
        "| Base | Argument | Classes added | Options changed | Gaps (issue) | Fixture (FAM2) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [_variant_line(row) for row in rows if "base" in row]
    rejected = {
        (_short_name(g["class"]), g["option"]): g["reason"]
        for row in rows
        for g in row["gaps"]
        if g["issue"] == REJECTED
    }
    lines += [
        "",
        "## Rejected gaps",
        "",
        "| Class | Option | Reason |",
        "|---|---|---|",
    ]
    lines += [
        f"| {cls} | {option or '-'} | {reason} |"
        for (cls, option), reason in sorted(rejected.items())
    ]
    lines += [
        "",
        "## Foundation models",
        "",
        "| Name | Row | Evidence |",
        "|---|---|---|",
    ]
    lines += [
        f"| {n} | `{m['row']}` | {m['evidence']} |"
        for n, m in document["foundation_models"].items()
    ]
    lines += [
        "",
        "## Classes in no architecture",
        "",
        "| Class | Issue | Decision today |",
        "|---|---|---|",
    ]
    lines += [
        f"| {_short_name(c)} | {entry['issue']} | {entry['decision']} |"
        for c, entry in document["unused_classes"].items()
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ command line


def _cmd_write(_: argparse.Namespace) -> int:
    names, aliases = foundation_names()
    document = build_document(names, aliases)
    JSON_PATH.write_text(dumps(document), encoding="utf-8")
    MD_PATH.write_text(to_markdown(document), encoding="utf-8")
    unassigned = [
        row["id"]
        for row in document["rows"]
        if any(g["issue"] == UNASSIGNED for g in row["gaps"])
    ]
    sys.stdout.write(
        f"{len(document['rows'])} rows written; rows with an unassigned gap: {len(unassigned)}\n"
    )
    return 1 if unassigned else 0


def _cmd_table(_: argparse.Namespace) -> int:
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    MD_PATH.write_text(to_markdown(document), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "write",
        help="build every architecture (TensorFlow) and write the JSON and the table",
    )
    commands.add_parser(
        "table", help="write the Markdown table from the committed JSON"
    )
    args = parser.parse_args(argv)
    return {"write": _cmd_write, "table": _cmd_table}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
