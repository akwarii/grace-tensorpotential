"""Golden-fixture generator of the TensorFlow GRACE models (the numerical ground truth of the torch twins).

For every architecture row of ``docs/torch_backend/architectures.json`` (a source: a model file of the repository,
or a preset of ``potentials/presets.py`` with its default arguments or one argument changed; the tensor models are
fixture-free, ``architecture_matrix.FIXTURE_FREE``) the generator builds the TensorFlow model with float64 or float32
parameters, overwrites every trainable floating variable with seeded non-zero values (several layers
are zero-initialised and would make the outputs trivial) and runs the structures of
``tests_torch/structures`` through it. A fixture holds

* ``<name>.npz``: per structure the model inputs (``<case>/in/<key>``), the output of **every instruction**
  (``<case>/ins/<name>``), the target key before and after each output instruction
  (``<case>/out_before/<name>``, ``<case>/out_after/<name>``: those instructions overwrite
  ``input_data[target.name]`` in place) and the results (``<case>/res/{energy, atomic_energy, forces,
  virial, stress, pair_f}``); and the index tables of the instructions (``tables/<instruction>/<attribute>``:
  ``left_ind``, ``right_ind``, ``m_sum_ind``, ``cg``, ``w_tile_*``, ``collect_*``, ``norm_map``, ...);
* ``<name>.weights.npz``: every variable, keyed by its place in the checkpoint
  (``<instruction>/<attribute path>``, never by the TF variable name, which is not unique);
* ``<name>.json``: the manifest (library, TensorFlow, numpy and pandas versions, git sha, seeds, the
  module-level switch ``_USE_GEMM_COUPLE``, the resolved constructor arguments of every instruction, the
  shape and dtype of every array, file sizes and sha256).

Two tiers: ``tiny`` for every source (3 to 4 elements, small widths; the same option paths as the parent yaml;
committed under ``tests_torch/golden/``, each fixture under 1 MB) and ``faithful`` for the two anchor models ``omat``
(``tests/model_grace_2L_omat.yaml``) and ``large`` (``tests/model_grace_2L_omat_large_base.yaml``) only (the real
widths, 6 elements; generated locally into ``tests_torch/fixtures/``, of which only the manifests are committed). A
manifest names the architecture row its fixture belongs to (``row``). An **option pair**
is a second fixture of the same weights with ``dense_nbr`` or ``lm_first`` switched on in every instruction that
takes it (made for the default row of each source family, not for its variants); its manifest records which outputs
equal the default layout and which weights differ in shape.
Random weights only, never foundation weights.

Usage (repository root, project ``.venv``, CPU, one thread)::

    python tools/make_golden.py yamls                              # write the derived yamls
    python tools/make_golden.py write --tier tiny                  # tests_torch/golden/ (--models LABEL ... for some)
    python tools/make_golden.py write --tier faithful --out tests_torch/fixtures
    python tools/make_golden.py verify tests_torch/golden          # reload every fixture in TF
    python tools/make_golden.py compare DIR_A DIR_B                # two runs (for example pre-cleanup and HEAD)
    python tools/make_golden.py cells                              # the tiny yamls exercise every option cell
    python tools/make_golden.py sources                            # label and architecture row of every source

The library imported is the one on ``PYTHONPATH`` (the manifest records its location and git sha), which is how
the generator runs once from a worktree of the ``pre-cleanup`` tag and once from the head.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grace_probe as gp  # noqa: E402
import oracle_snapshot as osn  # noqa: E402
import scan_options  # noqa: E402

logger = logging.getLogger("make_golden")

ROOT = Path(__file__).resolve().parents[1]
# appended, not inserted first: the library imported must be the one on PYTHONPATH (a worktree of an older tag),
# and the repository root is needed only for tests_torch
sys.path.append(str(ROOT))
TESTS = ROOT / "tests"
GOLDEN_DIR = ROOT / "tests_torch" / "golden"
FIXTURES_DIR = ROOT / "tests_torch" / "fixtures"
SEED = osn.SEED
FORMAT_VERSION = 2
"""2: the manifest names its architecture ``row`` and its ``source`` (file or preset and arguments)."""

_ZBL_FLOAT32 = (
    "ZBLPotential.frwrd compares the float64 bond lengths with a float32 cutoff in a float32 model "
    "(tf.Less: InvalidArgumentError), so TensorFlow gives no float32 reference (FAM2 finding)"
)
_LM_FIRST_OUTPUT = (
    "the output target reads the per-atom features in the default layout, so the lm_first model fails in "
    "TensorFlow ({}); no reference to compare with (FAM2 finding)"
)
#: label -> {dtype label or option-pair label: why TensorFlow cannot make that fixture}; measured, never guessed.
#: The manifests of the source record it (``not_made``), and the matrix counts the row as complete without it.
NOT_MADE: dict[str, dict[str, str]] = {
    **{
        f"{preset}__zbl_cutoff_1_0": {"f32": _ZBL_FLOAT32}
        for preset in (
            "grace_1layer_v1_24",
            "grace_2layer_v1_24",
            "grace_2layer_scaled",
            "grace_1layer_v2_25",
            "grace_2layer_v2_25",
        )
    },
    **{
        label: {
            "lm_first": _LM_FIRST_OUTPUT.format(
                "FSOut2ScalarTarget: StridedSlice out of bounds"
            )
        }
        for label in ("fs", "grace_fs_omat")
    },
    "model_grace": {
        "lm_first": _LM_FIRST_OUTPUT.format("MLPOut2ScalarTarget: MatMul size mismatch")
    },
}

#: anchor label -> parent yaml (in ``tests/``): the models of the faithful tier, and the labels of their rows
MODELS = {
    "omat": "model_grace_2L_omat.yaml",
    "large": "model_grace_2L_omat_large_base.yaml",
}
DTYPES = {"f64": "float64", "f32": "float32"}
DTYPE_LABELS = {name: label for label, name in DTYPES.items()}
#: option pair label -> the constructor option switched on
OPTIONS = {"dense": "dense_nbr", "lm_first": "lm_first"}
#: a tiny anchor stays under this many bytes, all its files (arrays, weights, manifest) together
TINY_BYTE_LIMIT = 1_000_000
#: a faithful fixture aims at this many bytes in each of its files: the float64 weights of the large model
#: alone take 19 MB at the real widths, so the sum of the files of one fixture cannot meet it
FAITHFUL_BYTE_TARGET = 25_000_000

# Tolerances of the reload check: the rows of the oracle snapshot (one named place for the numbers).
# The option-pair comparison has its own row: the dense and the default layout sum the bonds of an atom in a
# different order, so the results agree to a few ulp of the largest element, not exactly.
OPTION_PAIR_SCALE_RTOL = 1e-11


@dataclass(frozen=True)
class Tier:
    """How a tier differs from the parent yaml: its elements, whether widths shrink, and its structures."""

    name: str
    elements: tuple[str, ...]
    shrink_widths: bool
    structures: tuple[str, ...]
    #: species of the structure set that the model does not have, mapped to ones it has
    aliases: Mapping[str, str]

    @property
    def element_map(self) -> dict[str, int]:
        """Return the element map of the tier's models: the symbols in sorted order."""
        return {symbol: index for index, symbol in enumerate(sorted(self.elements))}


TIERS = {
    "tiny": Tier(
        "tiny",
        ("Cu", "H", "Mg", "O"),
        True,
        ("isolated_atom", "dimer", "self_image_cell", "periodic_triple"),
        {"Au": "H", "Ca": "Mg", "Si": "Cu"},
    ),
    "faithful": Tier(
        "faithful",
        ("Ca", "Cu", "H", "Mg", "O", "Si"),
        False,
        (
            "isolated_atom",
            "dimer",
            "self_image_cell",
            "fcc4",
            "icosahedron_satellite",
            "rattled_multi8",
        ),
        {"Au": "Ca"},
    ),
}

# Widths of the tiny tier: the real widths map to distinct small numbers where a parent keeps them distinct, so that
# a twin that mixes two axes up fails instead of passing (``distinct_widths_merged`` checks it for every source). The
# entries after 12 serve the sources added for FAM2 (2L v1: 48; LINEAR: 24; FS: 20, 15, 10 and 5) and do not occur
# in the two anchor yamls, whose derived files stay as they were.
TINY_WIDTHS = {
    128: 6,
    64: 8,
    42: 5,
    32: 4,
    17: 3,
    16: 3,
    13: 2,
    12: 2,
    48: 7,
    24: 5,
    20: 7,
    15: 6,
    10: 5,
    5: 3,
}
#: label -> widths that replace those of ``TINY_WIDTHS`` for that source. The 3L model takes 2.3 MB per fixture at the
#: common tiny widths (8 channels for 64) and still 1.0 MB with three distinct ones (64: 2, 32: 3, 128: 4), over the 1 MB of
#: the tier; its ranks 32 and 64 (``GeneralProductFunction``, ``FCRight2Left``) share the width 2 (``MERGED_WIDTHS``).
TINY_WIDTHS_OF: dict[str, dict[int, int]] = {
    "grace_3l_omat_large": {128: 3, 64: 2, 32: 2},
}
#: label -> distinct parent widths that its tiny yaml maps to one width, accepted by the ``cells`` check
MERGED_WIDTHS: dict[str, set[tuple[int, ...]]] = {"grace_3l_omat_large": {(32, 64)}}
TINY_HIDDEN = {
    "MLPRadialFunction": [6, 7],
    "MLPRadialFunction_v2": [6, 7],
    "LinMLPOut2ScalarTarget": [9],
    "SPBF": [6, 7],
}
#: constructor arguments that hold a width (an int, or a list of ints for ``FunctionReduceN.n_in``)
WIDTH_KEYS = (
    "embedding_size",
    "n_rad_max",
    "n_out",
    "n_crop",
    "rank",
    "n_in",
)


# ------------------------------------------------------------------ yamls


def derive_yaml(parent: Path, tier: Tier) -> dict[str, Any]:
    """Return the yaml dictionary of the model file ``parent`` for ``tier`` (``derive`` of its content)."""
    return derive(yaml.safe_load(parent.read_text()), tier)


def derive(raw: Any, tier: Tier, widths: Mapping[int, int] = TINY_WIDTHS) -> Any:
    """Return a copy of the model yaml ``raw`` for ``tier``: new element map and, for tiny, small widths.

    Only widths and the element set change; every class, option, ``lmax``, ``Lmax``, parity list and layout
    option stays as in the parent, which is what keeps every (class, option) pair exercised. Element-keyed
    values follow the new element map: ``atomic_shift_map`` keeps the shifts of the tier's elements. ``widths``
    maps a real width to its tiny one (``widths_of_source``).
    """
    raw = copy.deepcopy(raw)
    entries = scan_options.instruction_dicts(raw)[
        1
    ]  # the dictionaries of ``raw``, changed in place
    element_map = tier.element_map
    parent_map = next(
        (e["element_map"] for e in entries if isinstance(e.get("element_map"), dict)),
        {},
    )
    for entry in entries:
        cls = entry["__cls__"].rsplit(".", 1)[-1]
        if "element_map" in entry:
            entry["element_map"] = dict(element_map)
        if entry.get("number_of_atom_types"):
            entry["number_of_atom_types"] = len(element_map)
        if isinstance(entry.get("atomic_shift_map"), dict):
            entry["atomic_shift_map"] = remap_elements(
                entry["atomic_shift_map"], parent_map, element_map
            )
        if not tier.shrink_widths:
            continue
        for key in WIDTH_KEYS:
            if key in entry:
                entry[key] = tiny_width(entry[key], widths)
        if isinstance(entry.get("hidden_layers"), list):
            entry["hidden_layers"] = list(TINY_HIDDEN[cls])
    return raw


def tiny_width(value: Any, widths: Mapping[int, int] = TINY_WIDTHS) -> Any:
    """``value`` with every width of ``widths`` replaced by its tiny one (an int or a list of ints)."""
    if isinstance(value, list):
        return [tiny_width(v, widths) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return widths.get(value, value)
    return value


def widths_of_source(label: str) -> dict[int, int]:
    """Return the tiny widths of the source ``label``: ``TINY_WIDTHS`` with its entry of ``TINY_WIDTHS_OF``."""
    return {**TINY_WIDTHS, **TINY_WIDTHS_OF.get(label, {})}


def remap_elements(
    values: Mapping[Any, Any],
    parent_map: Mapping[str, int],
    element_map: Mapping[str, int],
) -> dict[Any, Any]:
    """Keep the entries of ``values`` (keyed by element index or symbol of ``parent_map``) of the elements of ``element_map``.

    A key stays a symbol or an index as it was; an element of ``element_map`` without an entry gets none.
    """
    symbol_of = {index: symbol for symbol, index in parent_map.items()}
    out: dict[Any, Any] = {}
    for key, value in values.items():
        symbol = key if key in parent_map else symbol_of.get(key)
        if symbol is not None and symbol in element_map:
            out[symbol if key == symbol else element_map[symbol]] = value
    return dict(sorted(out.items(), key=lambda item: str(item[0])))


def widths_of(raw: Any) -> set[int]:
    """Every width value of a model yaml (``WIDTH_KEYS``; ``hidden_layers`` are replaced as a whole, not mapped)."""
    found: set[int] = set()
    for entry in scan_options.instruction_dicts(raw)[1]:
        for key in WIDTH_KEYS:
            values = entry.get(key)
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, int) and not isinstance(value, bool):
                    found.add(value)
    return found


def distinct_widths_merged(
    parent: Any, widths: Mapping[int, int] = TINY_WIDTHS
) -> list[tuple[int, ...]]:
    """Return the groups of distinct widths of ``parent`` that ``widths`` maps to one tiny width (empty: none)."""
    groups: dict[int, list[int]] = {}
    for width in sorted(widths_of(parent) & set(widths)):
        groups.setdefault(widths[width], []).append(width)
    return [tuple(g) for g in groups.values() if len(g) > 1]


def yaml_text(raw: Mapping[str, Any] | list[Any]) -> str:
    """Return the text written for a derived yaml (stable across runs); the old list format stays a list."""
    return yaml.safe_dump(raw if isinstance(raw, list) else dict(raw), sort_keys=False)


def yaml_path(model: str, tier: str, base: Path | None = None) -> Path:
    """Where the derived yaml of ``model`` in ``tier`` is kept: next to the fixtures of the tier."""
    folder = base if base is not None else tier_dir(tier)
    return folder / "yamls" / f"{model}_{tier}.yaml"


def tier_dir(tier: str) -> Path:
    """``tests_torch/golden`` for the tiny tier, ``tests_torch/fixtures`` for the faithful one."""
    return GOLDEN_DIR if tier == "tiny" else FIXTURES_DIR


def write_yamls(tiers: Iterable[str] = TIERS, base: Path | None = None) -> list[Path]:
    """Write the derived yamls of the anchors (``MODELS``) for ``tiers``; returns the paths."""
    return [
        write_derived_yaml(anchor_source(model), tier_name, base)
        for tier_name in tiers
        for model in MODELS
    ]


def write_derived_yaml(source: Source, tier: str, base: Path | None = None) -> Path:
    """Write the derived yaml of ``source`` in ``tier`` next to the fixtures of ``base``; returns its path."""
    path = yaml_path(source.label, tier, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text(derived_raw(source, tier)))
    return path


# ------------------------------------------------------------------ sources: the architecture rows


@dataclass(frozen=True)
class Source:
    """Where the parent model of a fixture comes from: a model file, or a preset built with ``arguments``.

    ``row`` is the id of its row in ``docs/torch_backend/architectures.json``; a variant (``arguments`` not empty)
    differs from the default build of its preset in that one argument.
    """

    label: str
    row: str
    path: Path | None = None
    preset: str | None = None
    arguments: tuple[tuple[str, Any], ...] = ()

    @property
    def is_variant(self) -> bool:
        """Whether the source is a preset with one argument changed (no option pairs are made for it)."""
        return bool(self.arguments)

    def describe(self, tier: Tier) -> dict[str, Any]:
        """Return the source as the manifest records it: the file, or the preset and the arguments it was built with."""
        if self.path is not None:
            return {"file": self.path.relative_to(ROOT).as_posix()}
        return {
            "preset": self.preset,
            "arguments": tier_arguments(self.arguments, tier),
        }


def slug(text: str) -> str:
    """Lower case, every run of other characters than letters and digits made one ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def source_label(preset: str, arguments: Mapping[str, Any]) -> str:
    """``grace_2layer_v2_25``, ``grace_2layer_v2_25__basis_type_gaussian``; a dict value is not spelt out."""
    label = slug(preset)
    for argument, value in arguments.items():
        label += "__" + slug(argument)
        if not isinstance(value, dict):
            label += "_" + slug(json.dumps(value))
    return label


def anchor_source(label: str) -> Source:
    """Return the source of an anchor model (``MODELS``), without TensorFlow."""
    path = TESTS / MODELS[label]
    # the id of architecture_matrix.file_row_id, spelt out: this runs without the matrix (from older trees)
    return Source(label, f"file:{path.stem}", path=path)


def sources() -> dict[str, Source]:
    """Every source with a tiny fixture, by label: the rows of ``architecture_matrix`` that are not fixture-free.

    The presets and their variants come from ``architecture_matrix.variants`` (TensorFlow imports the presets),
    the files from ``architecture_matrix.model_files``; the two anchors keep their labels ``omat`` and ``large``.
    """
    import architecture_matrix as am

    anchors = {anchor_source(label).row: label for label in MODELS}
    out: dict[str, Source] = {}
    for name in am.PRESETS:
        for label, arguments in am.variants("preset", name):
            row = am.row_id("preset", name, label)
            one = Source(
                source_label(name, arguments),
                row,
                preset=name,
                arguments=tuple(arguments.items()),
            )
            out[one.label] = one
    for path in am.model_files():
        row = am.file_row_id(path)
        one = Source(anchors.get(row, slug(path.stem)), row, path=path)
        out[one.label] = one
    return out


def tier_arguments(arguments: Iterable[tuple[str, Any]], tier: Tier) -> dict[str, Any]:
    """Return the preset arguments for the elements of ``tier``: element-keyed values name every element of the tier.

    The matrix's ``cutoff_dict`` and ``atomic_shift_map`` name its own two elements (a shift map that misses an
    element of the model fails in TensorFlow); the tier's ones stay dicts of distinct values: cutoffs of 4.0 A plus
    0.25 A per pair in sorted pair order, shifts of -1 eV minus 1 eV per element index.
    """
    out = dict(arguments)
    symbols = sorted(tier.elements)
    if isinstance(out.get("cutoff_dict"), dict):
        pairs = [a + b for i, a in enumerate(symbols) for b in symbols[i:]]
        out["cutoff_dict"] = {pair: 4.0 + 0.25 * k for k, pair in enumerate(pairs)}
    if isinstance(out.get("atomic_shift_map"), dict):
        out["atomic_shift_map"] = {index: -1.0 - index for index in range(len(symbols))}
    return out


def parent_raw(source: Source, tier: Tier = TIERS["tiny"]) -> Any:
    """Return the parent model yaml of ``source``: the file, or the preset built for the elements of ``tier``.

    A preset is built with the element map of the tier (its element-dependent tables follow it) and its real
    widths; ``derive`` then shrinks the widths as it does for a file.
    """
    if source.path is not None:
        return yaml.safe_load(source.path.read_text())
    # the thread and determinism settings must come before the first TensorFlow operation of the process
    osn.configure_tensorflow()
    import architecture_matrix as am

    return am.build_instructions(
        "preset",
        str(source.preset),
        tier_arguments(source.arguments, tier),
        element_map=tier.element_map,
    )


def option_cells(paths: Sequence[Path]) -> set[tuple]:
    """The (class, option, kind[, value]) cells exercised by the yamls: a value counts where the kind is categorical."""
    cells: set[tuple] = set()
    for row in scan_options.scan(list(paths)):
        if row.values:
            cells.update((row.cls, row.option, row.kind, value) for value in row.values)
        else:
            cells.add((row.cls, row.option, row.kind))
    return cells


def derived_raw(source: Source, tier: str) -> Any:
    """Return the yaml of ``source`` in ``tier``: ``derive`` of its parent with the source's tiny widths."""
    return derive(
        parent_raw(source, TIERS[tier]), TIERS[tier], widths_of_source(source.label)
    )


def missing_cells(parent: Path, derived: Path) -> list[tuple]:
    """Cells of the parent yaml that the derived yaml does not exercise (empty: nothing is lost)."""
    return sorted(option_cells([parent]) - option_cells([derived]), key=repr)


def missing_source_cells(source: Source, raw: Any, derived: Path) -> list[tuple]:
    """``missing_cells`` for a source whose parent yaml is ``raw``: a preset's parent goes to a scratch file first."""
    if source.path is not None:
        return missing_cells(source.path, derived)
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp) / "parent.yaml"
        parent.write_text(yaml_text(raw))
        return missing_cells(parent, derived)


# ------------------------------------------------------------------ structures


def relabel(atoms, tier: Tier):
    """A copy of ``atoms`` with the species the tier's model lacks replaced by ``tier.aliases``.

    The geometry is untouched; only the species labels change (a model with random weights sees element
    indices only). A species with no alias that the model lacks is an error.
    """
    symbols = [tier.aliases.get(s, s) for s in atoms.get_chemical_symbols()]
    unknown = sorted(set(symbols) - set(tier.elements))
    if unknown:
        raise ValueError(
            f"species {unknown} are not in the {tier.name} element set {tier.elements}; "
            "add an alias to the tier"
        )
    out = atoms.copy()
    out.set_chemical_symbols(symbols)
    return out


def periodic_triple():
    """Three atoms of Cu, Mg and O at generic positions of a small orthorhombic cell (79 bonds at the cutoff of 6 A).

    The structures of ``tests_torch/structures`` that are small enough for a tiny anchor are symmetric (``fcc4``,
    ``self_image_cell``: an inversion centre on every atom or the mid-point, so the forces vanish to 1e-15), and
    ``dimer`` is aperiodic: without this case no tiny anchor has a non-zero force in a periodic cell. The cell is smaller
    than the cutoff along every axis, so atoms are their own neighbours through images.
    """
    from ase import Atoms

    return Atoms(
        "CuMgO",
        scaled_positions=[[0.0, 0.0, 0.0], [0.27, 0.41, 0.13], [0.63, 0.19, 0.71]],
        cell=[4.4, 4.7, 5.0],
        pbc=True,
    )


#: structures that belong to the generator and not to the shared set of ``tests_torch/structures``
EXTRA_STRUCTURES: dict[str, Callable[[], Any]] = {"periodic_triple": periodic_triple}


def tier_cases(tier: Tier) -> dict[str, Any]:
    """The structures of ``tier`` (relabelled), in the order of ``tier.structures``."""
    from tests_torch.structures.build_structures import load_structures

    shared = load_structures()
    every = {**shared, **{n: make() for n, make in EXTRA_STRUCTURES.items()}}
    return {name: relabel(every[name], tier) for name in tier.structures}


# ------------------------------------------------------------------ models and weights


def checkpoint_paths(model) -> dict[int, str]:
    """``id(variable)`` -> ``<instruction>/<attribute path>``, the checkpoint key without its frame."""
    prefix, suffix = "model/instructions/", gp.SAVED_SUFFIX
    out = {}
    for ident, key in gp.checkpoint_keys(model).items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            raise ValueError(f"unexpected checkpoint key {key!r}")
        out[ident] = key[len(prefix) : -len(suffix)]
    return out


def seeded_weight(key: str, shape: tuple[int, ...], seed: int = SEED) -> np.ndarray:
    """Non-trivial values for the variable at ``key``: ``N(0, 1/sqrt(shape[0]))`` for matrices, ``N(1, 0.1)`` else.

    Seeded by the key (unique), not by the TF name (``Variable:0`` occurs twice in the large model), so the
    values do not depend on creation order.
    """
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    if len(shape) >= 2:
        return rng.normal(0.0, 1.0 / np.sqrt(shape[0]), size=shape)
    return rng.normal(1.0, 0.1, size=shape)


def build_model(path: Path, dtype: str, option: str | None = None, seed: int = SEED):
    """The TF model of the yaml at ``path`` with ``dtype`` parameters and seeded non-zero weights.

    ``option`` (``dense_nbr`` or ``lm_first``) is switched on first in every instruction that takes it.
    """
    osn.configure_tensorflow(seed)
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import TPModel

    instructions = load_instructions(str(path))
    if option is not None:
        instructions = osn.with_layout_option(instructions, option)
    model = TPModel(instructions)
    import tensorflow as tf

    model.build(getattr(tf, dtype))
    keys = checkpoint_paths(model)
    for var in model.trainable_variables:
        if var.dtype.is_floating:
            values = seeded_weight(keys[id(var)], tuple(var.shape), seed)
            var.assign(values.astype(var.dtype.as_numpy_dtype))
    return model


def weights_of(model) -> dict[str, np.ndarray]:
    """Every variable of ``model`` keyed ``<instruction>/<attribute path>`` (numpy; strings as unicode arrays)."""
    keys = checkpoint_paths(model)
    return {keys[id(v)]: _variable_numpy(v) for v in model.variables}


def _variable_numpy(variable) -> np.ndarray:
    """The value of a variable as numpy; a string variable becomes a unicode array (no object arrays, no pickle)."""
    value = np.asarray(variable.numpy())
    if value.dtype == object:
        return np.char.decode(value.astype("S"), "utf-8")
    return value


def weight_table(model) -> list[dict[str, Any]]:
    """Key, shape, dtype and trainability of every variable, sorted by key."""
    keys = checkpoint_paths(model)
    rows = [
        {
            "key": keys[id(v)],
            "shape": list(v.shape),
            "dtype": v.dtype.name,
            "trainable": bool(v.trainable),
        }
        for v in model.variables
    ]
    return sorted(rows, key=lambda r: r["key"])


def assign_weights(model, weights: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Assign ``weights`` (by key) to the variables of ``model``; report what could not be matched.

    A key present on one side only, or with a different shape, is reported and left alone (the variable keeps
    its own value); nothing is dropped silently. A model of the old list-format yaml keys its checkpoint by the
    position of the instruction (``11/reducing_YI``); its option pair, rebuilt from a name map, by the name
    (``B/reducing_YI``): such keys are matched by position (``renamed_by_position`` counts them).
    """
    keys = checkpoint_paths(model)
    by_key = {keys[id(v)]: v for v in model.variables}
    names = [ins.name for ins in gp._instructions_of(model)]  # noqa: SLF001
    weights, renamed = by_position(weights, names, set(by_key))
    report: dict[str, Any] = {
        "renamed_by_position": renamed,
        "only_in_model": sorted(set(by_key) - set(weights)),
        "only_in_weights": sorted(set(weights) - set(by_key)),
        "shape_differs": {},
        "value_differs": [],
        "assigned": 0,
    }
    for key, var in by_key.items():
        if key not in weights:
            continue
        value = np.asarray(weights[key])
        if tuple(var.shape) != value.shape:
            report["shape_differs"][key] = {
                "model": list(var.shape),
                "weights": list(value.shape),
            }
            continue
        if var.dtype.is_floating or var.dtype.is_integer:
            var.assign(value.astype(var.dtype.as_numpy_dtype, copy=False))
        elif not np.array_equal(_variable_numpy(var), value):
            # a string variable (the element symbols) is fixed by the yaml: it must already agree
            report["value_differs"].append(key)
            continue
        report["assigned"] += 1
    return report


def by_position(
    weights: Mapping[str, np.ndarray], names: Sequence[str], model_keys: set[str]
) -> tuple[dict[str, np.ndarray], int]:
    """Return ``weights`` with ``<position>/<path>`` keys renamed ``<names[position]>/<path>`` where the model has that key.

    The second value is the number of renamed keys; a key that is already the model's, or whose renamed form the model
    lacks, is kept as it is.
    """
    out: dict[str, np.ndarray] = {}
    renamed = 0
    for key, value in weights.items():
        head, _, rest = key.partition("/")
        if key not in model_keys and head.isdigit() and int(head) < len(names):
            candidate = f"{names[int(head)]}/{rest}"
            if candidate in model_keys:
                out[candidate] = value
                renamed += 1
                continue
        out[key] = value
    return out, renamed


def _own_tables(instruction, others: set[int]) -> Iterable[tuple[str, Any]]:
    """``(attribute path, value)`` of the non-variable tensors and arrays an instruction owns.

    The walk enters modules nested in the instruction (layers, coupling helpers) but not the other
    instructions of the model that it refers to (``left``, ``radial``, ``indicator``, ...): those are listed
    under their own name, so a table is stored once, under its owner.
    """
    import tensorflow as tf

    seen = {id(instruction)}

    def walk(obj, path: str) -> Iterable[tuple[str, Any]]:
        for name, value in gp._children(obj):  # noqa: SLF001 - the probe's flattening of containers
            kind = gp._tensor_kind(value)  # noqa: SLF001
            if kind in ("tensor", "ndarray"):
                yield f"{path}/{name}", value
            elif (
                isinstance(value, tf.Module)
                and id(value) not in seen
                and id(value) not in others
            ):
                seen.add(id(value))
                yield from walk(value, f"{path}/{name}")

    return walk(instruction, instruction.name)


def tables_of(model) -> dict[str, np.ndarray]:
    """The non-variable tensors and arrays of every instruction (index tables, Clebsch-Gordan values, constants)."""
    instructions = gp._instructions_of(model)  # noqa: SLF001
    others = {id(i) for i in instructions}
    out: dict[str, np.ndarray] = {}
    for ins in instructions:
        for path, value in _own_tables(ins, others):
            out[f"tables/{path}"] = np.asarray(
                value.numpy() if hasattr(value, "numpy") else value
            )
    return out


def init_args_of(model) -> dict[str, Any]:
    """The resolved constructor arguments of every instruction, as the saved ``model.yaml`` holds them."""
    return {
        ins.name: {
            "class": type(ins).__name__,
            "init_args": json.loads(json.dumps(ins.to_dict(), default=_jsonable)),
        }
        for ins in gp._instructions_of(model)  # noqa: SLF001
    }


def _jsonable(value: Any) -> Any:
    """JSON form of the odd values a constructor argument can hold (numpy scalars and arrays, sets)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)
    return repr(value)


# ------------------------------------------------------------------ evaluation


class Recorder:
    """An instruction wrapper that keeps what the instruction writes into the data dictionary.

    The data dictionary is shared by all instructions and an output instruction overwrites
    ``data[target.name]`` in place, so the value of a target is snapshotted before and after each of them.
    """

    def __init__(self, instruction, sink: dict[str, np.ndarray]) -> None:
        from tensorpotential.instructions.output import TPOutputInstruction

        self.instruction = instruction
        self.sink = sink
        self.is_output = isinstance(instruction, TPOutputInstruction)

    def __call__(self, input_data: dict, training: bool = False, local: bool = False):
        """Run the instruction and record its output (the target before and after for an output instruction)."""
        name = self.instruction.name
        if self.is_output:
            target = self.instruction.target.name
            self.sink[f"out_before/{name}"] = _numpy(input_data[target])
        result = self.instruction(input_data, training=training, local=local)
        if self.is_output:
            self.sink[f"out_after/{name}"] = _numpy(input_data[target])
        else:
            self.sink[f"ins/{name}"] = _numpy(input_data[name])
        return result


def _numpy(value) -> np.ndarray:
    """A copy of a tensor (or array) as numpy."""
    return np.array(value.numpy() if hasattr(value, "numpy") else value)


def evaluate_case(
    model, atoms, batch: Mapping[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Inputs, every instruction output and the results for one structure.

    ``batch`` is the model input of the structure; when it is not given it is built from ``atoms`` by the
    neighbour-list builder of the library (the structure itself is left unchanged). Keys are
    ``in/<key>``, ``ins/<instruction>``, ``out_before/<instruction>``, ``out_after/<instruction>`` and
    ``res/<quantity>``.
    """
    import tensorflow as tf

    from tensorpotential import constants

    if batch is None:
        batch = osn._geometry_batch(model, atoms)  # noqa: SLF001 - one place builds the batch
    data = tf.data.Dataset.from_tensors(dict(batch)).get_single_element()
    out: dict[str, np.ndarray] = {
        f"in/{key}": _numpy(value) for key, value in data.items()
    }
    sink: dict[str, np.ndarray] = {}
    recorders = [Recorder(i, sink) for i in gp._instructions_of(model)]  # noqa: SLF001
    result = model.compute_function(recorders, data, training=False)
    out.update(sink)
    out["res/energy"] = _numpy(result[constants.PREDICT_TOTAL_ENERGY])
    out["res/atomic_energy"] = _numpy(result[constants.PREDICT_ATOMIC_ENERGY])
    out["res/forces"] = _numpy(result[constants.PREDICT_FORCES])
    out["res/virial"] = _numpy(result[constants.PREDICT_VIRIAL])
    out["res/pair_f"] = _numpy(result["z_" + constants.PREDICT_PAIR_FORCES])
    virial = out["res/virial"].reshape(6)[osn.VIRIAL_TO_VOIGT]
    out["res/stress"] = (
        -virial / atoms.get_volume() if atoms.cell.rank == 3 else virial * 0.0
    )
    return out


def evaluate_cases(
    model,
    cases: Mapping[str, Any],
    batches: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """``evaluate_case`` for every structure, keyed ``<case>/<key>``."""
    arrays: dict[str, np.ndarray] = {}
    for case, atoms in cases.items():
        given = None if batches is None else batches[case]
        for key, value in evaluate_case(model, atoms, given).items():
            arrays[f"{case}/{key}"] = value
    return arrays


# ------------------------------------------------------------------ fixtures


def fixture_name(model: str, tier: str, dtype: str, option: str | None = None) -> str:
    """``omat_tiny_f64``, or ``omat_tiny_f64_dense`` for the option pair."""
    return "_".join([model, tier, dtype, *([option] if option else [])])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(folder: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argument list
            ["git", "-C", str(folder), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _git_dirty(folder: Path) -> bool | None:
    """Whether the files under ``folder`` differ from the commit of their checkout (``None``: not a checkout)."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argument list
            ["git", "-C", str(folder), "status", "--porcelain", "--", "."],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def library_info() -> dict[str, Any]:
    """Where the imported library is, its commit, the versions and the module-level coupling switch."""
    import tensorpotential
    from tensorpotential.instructions import compute

    library = Path(tensorpotential.__file__).resolve().parent
    tf = osn.configure_tensorflow()
    return {
        "library_git_sha": _git_sha(library.parent),
        "library_dirty": _git_dirty(library),
        "generator_git_sha": _git_sha(ROOT),
        "generator_dirty": _git_dirty(ROOT / "tools"),
        "versions": osn._versions(tf),  # noqa: SLF001
        "use_gemm_couple": bool(compute._USE_GEMM_COUPLE),  # noqa: SLF001 - recorded, never changed
    }


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ``arrays`` compressed, without pickle, with the keys sorted so that equal content gives equal bytes."""
    np.savez_compressed(path, **{k: arrays[k] for k in sorted(arrays)})  # ty: ignore[invalid-argument-type]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read an npz written by ``save_npz`` (pickle is never enabled)."""
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


@dataclass(frozen=True)
class FixtureRequest:
    """One fixture to make."""

    model: str
    tier: str
    dtype: str
    option: str | None = None

    @property
    def name(self) -> str:
        """The file stem of the fixture."""
        return fixture_name(self.model, self.tier, self.dtype, self.option)


def requests_for(tier: str, models: Iterable[str] = MODELS) -> list[FixtureRequest]:
    """Base fixtures ({model} x {f64, f32}) and, in float64, one option pair per layout option.

    The rule of the anchors, which take both layout options; ``write_tier`` decides per source (``NOT_MADE``,
    ``layout_options``, variants).
    """
    out = []
    for model in models:
        out += [FixtureRequest(model, tier, dtype) for dtype in DTYPES]
        out += [FixtureRequest(model, tier, "f64", option) for option in OPTIONS]
    return out


def compare_arrays(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    scale_rtol: float,
) -> dict[str, Any]:
    """Key-by-key comparison with the scaled tolerance of ``oracle_snapshot.compare_snapshots``.

    Arrays of text (the element symbols) are compared for equality; they cannot be a tolerance question.
    """

    def is_text(array: np.ndarray) -> bool:
        return array.dtype.kind in "USO"

    texts = {k for k, v in first.items() if is_text(v)} | {
        k for k, v in second.items() if is_text(v)
    }
    report = osn.compare_snapshots(
        {k: v for k, v in first.items() if k not in texts},
        {k: v for k, v in second.items() if k not in texts},
        scale_rtol=scale_rtol,
    )
    for key in sorted(texts & first.keys() & second.keys()):
        report["n_compared"] += 1
        if not np.array_equal(first[key], second[key]):
            report["exceeding"].append(key)
    report["only_in_first"] += sorted(texts & first.keys() - second.keys())
    report["only_in_second"] += sorted(texts & second.keys() - first.keys())
    report["ok"] = report_ok(report)
    return report


def reload_scale_rtol(dtype: str) -> float:
    """The named tolerance row (``oracle_snapshot.TOLERANCE_ROWS``) for the dtype label ``f64`` or ``f32``."""
    return osn.TOLERANCE_ROWS["float32" if dtype == "f32" else "float64"].scale_rtol


def _relation(default: np.ndarray, paired: np.ndarray) -> str:
    """How an array of an option pair relates to the default-layout array of the same key.

    ``equal``: same shape and values (to ``OPTION_PAIR_SCALE_RTOL`` of the largest element);
    ``lm_axis_first``: the last axis of the default (``lm``, or the coupling axis of a table) is the first
    axis of the pair, the others keep their order (the layout of ``lm_first``);
    ``differs``: neither (a different table or shape).
    """
    atol = OPTION_PAIR_SCALE_RTOL * float(
        np.max(np.abs(default)) if default.size else 1.0
    )
    if default.shape == paired.shape and np.allclose(
        default, paired, rtol=0.0, atol=atol
    ):
        return "equal"
    moved = np.moveaxis(default, -1, 0)
    if (
        default.ndim >= 2
        and moved.shape == paired.shape
        and np.allclose(moved, paired, rtol=0.0, atol=atol)
    ):
        return "lm_axis_first"
    return "differs"


def option_pair_report(
    base: Mapping[str, np.ndarray],
    paired: Mapping[str, np.ndarray],
    weights_report: Mapping[str, Any],
) -> dict[str, Any]:
    """What differs between a default-layout fixture and its option pair.

    ``results_max_abs_diff`` is the largest absolute difference of every result quantity; ``outputs`` lists
    the instruction outputs and tables by their relation to the default layout (``_relation``; ``missing``:
    only the default has the key); ``weights`` is the report of ``assign_weights`` (equal shapes: nothing
    in ``shape_differs``).
    """
    results: dict[str, float] = {}
    outputs: dict[str, list[str]] = {
        "equal": [],
        "lm_axis_first": [],
        "differs": [],
        "missing": [],
    }
    for key in sorted(base):
        if key not in paired:
            outputs["missing"].append(key)
        elif "/res/" in key:
            same_shape = base[key].shape == paired[key].shape
            results[key] = (
                float(np.max(np.abs(base[key] - paired[key])))
                if same_shape and base[key].size
                else (0.0 if same_shape else float("nan"))
            )
        elif "/in/" not in key:
            outputs[_relation(base[key], paired[key])].append(key)
    return {
        "results_max_abs_diff": results,
        "outputs": outputs,
        "weights": dict(weights_report),
    }


@dataclass(frozen=True)
class MadeFixture:
    """What ``make_fixture`` returns: the manifest and the arrays kept in memory (an option pair needs them)."""

    manifest: dict[str, Any]
    weights: dict[str, np.ndarray]
    arrays: dict[str, np.ndarray]


def _file_entry(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


@dataclass(frozen=True)
class Written:
    """What ``make_fixture`` wrote for one request, as the manifest describes it."""

    ypath: Path
    files: Mapping[str, Path]
    option_pair: dict[str, Any] | None
    n_arrays: int


def _manifest(
    request: FixtureRequest,
    source: Source,
    model,
    cases: Mapping[str, Any],
    written: Written,
    seed: int,
) -> dict[str, Any]:
    """The manifest of a written fixture (no clock, no host name: the same run gives the same text)."""
    tier = TIERS[request.tier]
    ypath = written.ypath
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "name": request.name,
        "model": request.model,
        "row": source.row,
        "source": source.describe(tier),
        "tier": request.tier,
        "dtype": DTYPES[request.dtype],
        "option": OPTIONS.get(request.option) if request.option else None,
        "seed": seed,
        "yaml": ypath.name,
        "yaml_sha256": hashlib.sha256(ypath.read_bytes()).hexdigest(),
        "elements": sorted(tier.elements),
        "species_aliases": dict(tier.aliases),
        "structures": {n: osn.describe_structure(a) for n, a in cases.items()},
        "library": library_info(),
        "instructions": init_args_of(model),
        "variables": weight_table(model),
        "layout_options": layout_options(model),
        "not_made": NOT_MADE.get(request.model, {}),
        "n_arrays": written.n_arrays,
        "weights_from": None
        if not request.option
        else fixture_name(request.model, request.tier, request.dtype),
        "option_pair": written.option_pair,
        "files": {kind: _file_entry(path) for kind, path in written.files.items()},
    }
    manifest["total_bytes"] = sum(
        entry["bytes"] for entry in manifest["files"].values()
    )
    return manifest


def layout_options(model) -> dict[str, bool]:
    """Return, per layout option (``dense_nbr``, ``lm_first``), whether one instruction of ``model`` takes it."""
    instructions = gp._instructions_of(model)  # noqa: SLF001
    return {
        option: any(osn._accepts_option(i, option) for i in instructions)  # noqa: SLF001
        for option in OPTIONS.values()
    }


def make_fixture(
    request: FixtureRequest,
    out: Path,
    base: MadeFixture | None = None,
    seed: int = SEED,
    source: Source | None = None,
) -> MadeFixture:
    """Build, evaluate and write one fixture.

    For an option pair ``base`` is what this function returned for the default fixture of the same model,
    tier and dtype: its weights are assigned by key to the option model, and its arrays are what the pair is
    compared with. ``source`` defaults to the anchor of ``request.model``.
    """
    source = source if source is not None else anchor_source(request.model)
    ypath = yaml_path(request.model, request.tier, out)
    if not ypath.exists():
        write_derived_yaml(source, request.tier, out)
    option = OPTIONS.get(request.option) if request.option else None
    model = build_model(ypath, DTYPES[request.dtype], option, seed)
    weights_report: dict[str, Any] | None = None
    if request.option:
        if base is None:
            raise ValueError(
                "an option pair needs the default fixture of the same weights"
            )
        weights_report = assign_weights(model, base.weights)
    cases = tier_cases(TIERS[request.tier])
    arrays = evaluate_cases(model, cases)
    arrays.update(tables_of(model))
    weights = weights_of(model)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / request.name
    files = {"arrays": stem.with_suffix(".npz")}
    save_npz(files["arrays"], arrays)
    if not request.option:
        files["weights"] = stem.with_name(stem.name + ".weights.npz")
        save_npz(files["weights"], weights)
    pair = None
    if base is not None and weights_report is not None:
        pair = option_pair_report(base.arrays, arrays, weights_report)
    written = Written(ypath, files, pair, len(arrays))
    manifest = _manifest(request, source, model, cases, written, seed)
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    return MadeFixture(manifest, weights, arrays)


def write_tier(
    tier: str,
    out: Path | None = None,
    models: Iterable[str] = MODELS,
    seed: int = SEED,
    every: Mapping[str, Source] | None = None,
) -> list[dict[str, Any]]:
    """Every fixture of ``models`` in ``tier`` into ``out`` (default: the directory of the tier); returns the manifests.

    ``every`` maps a label to its source (default: the anchors only). A model gets its two dtypes and, unless it is a
    preset variant, one float64 option pair per layout option that one of its instructions takes
    (``layout_options`` of the manifest).
    """
    folder = out if out is not None else tier_dir(tier)
    known = every if every is not None else {m: anchor_source(m) for m in models}
    manifests = []
    for label in models:
        source = known[label]
        write_derived_yaml(source, tier, folder)
        skip = NOT_MADE.get(label, {})
        made = [
            make_fixture(FixtureRequest(label, tier, dtype), folder, None, seed, source)
            for dtype in DTYPES
            if dtype not in skip
        ]
        taken = made[0].manifest["layout_options"]
        pairs = (
            []
            if source.is_variant
            else [o for o in OPTIONS if taken[OPTIONS[o]] and o not in skip]
        )
        made += [
            make_fixture(
                FixtureRequest(label, tier, "f64", o), folder, made[0], seed, source
            )
            for o in pairs
        ]
        manifests += [one.manifest for one in made]
        for one in made:
            logger.info(
                "%s: %d bytes", one.manifest["name"], one.manifest["total_bytes"]
            )
    return manifests


# ------------------------------------------------------------------ verification


def load_fixture(folder: Path, name: str) -> dict[str, Any]:
    """The manifest, arrays and weights of a fixture (an option pair reads the weights of its default fixture)."""
    manifest = json.loads((folder / f"{name}.json").read_text())
    owner = manifest["weights_from"] or name
    return {
        "manifest": manifest,
        "arrays": load_npz(folder / f"{name}.npz"),
        "weights": load_npz(folder / f"{owner}.weights.npz"),
    }


def verify_fixture(
    folder: Path, name: str, scale_rtol: float | None = None
) -> dict[str, Any]:
    """Reload fixture ``name`` in TF (yaml, weights, stored inputs) and compare with the stored outputs.

    The model is built from the yaml next to the fixture, given the stored weights, and run on the stored
    inputs, not on a neighbour list built again; the comparison is ``compare_arrays`` on every output. It is
    first seeded with a seed other than the one of the fixture, so that its result can only be the stored one
    if the stored weights were really assigned (the same seed would reproduce them without the file).
    """
    fixture = load_fixture(folder, name)
    manifest = fixture["manifest"]
    dtype = DTYPE_LABELS[manifest["dtype"]]
    ypath = folder / "yamls" / manifest["yaml"]
    other_seed = manifest["seed"] + 1
    model = build_model(ypath, manifest["dtype"], manifest["option"], other_seed)
    report = assign_weights(model, fixture["weights"])
    if any(
        report[k]
        for k in ("only_in_model", "only_in_weights", "shape_differs", "value_differs")
    ):
        raise ValueError(f"{name}: the weights do not fit the model: {report}")
    cases = tier_cases(TIERS[manifest["tier"]])
    stored = fixture["arrays"]
    batches = {
        case: {
            key.split("/in/", 1)[1]: value
            for key, value in stored.items()
            if key.startswith(f"{case}/in/")
        }
        for case in cases
    }
    again = evaluate_cases(model, cases, batches)
    again.update(tables_of(model))
    tolerance = reload_scale_rtol(dtype) if scale_rtol is None else scale_rtol
    return compare_arrays(stored, again, tolerance)


def compare_dirs(first: Path, second: Path) -> dict[str, Any]:
    """Compare every fixture present in two directories (two runs of the generator), array by array."""
    names = sorted(p.stem for p in first.glob("*.json"))
    other = sorted(p.stem for p in second.glob("*.json"))
    report: dict[str, Any] = {
        "only_in_first": sorted(set(names) - set(other)),
        "only_in_second": sorted(set(other) - set(names)),
        "fixtures": {},
    }
    for name in sorted(set(names) & set(other)):
        manifest = json.loads((first / f"{name}.json").read_text())
        dtype = DTYPE_LABELS[manifest["dtype"]]
        a = load_npz(first / f"{name}.npz")
        b = load_npz(second / f"{name}.npz")
        report["fixtures"][name] = compare_arrays(a, b, reload_scale_rtol(dtype))
        owner = manifest["weights_from"] or name
        wa = load_npz(first / f"{owner}.weights.npz")
        wb = load_npz(second / f"{owner}.weights.npz")
        report["fixtures"][name]["weights"] = compare_arrays(wa, wb, 0.0)
    return report


def report_ok(report: Mapping[str, Any]) -> bool:
    """Whether a ``compare_dirs`` or ``compare_arrays`` report found no difference above the tolerance."""
    if "fixtures" in report:
        return not (report["only_in_first"] or report["only_in_second"]) and all(
            report_ok(r) for r in report["fixtures"].values()
        )
    flat = (
        report.get("only_in_first"),
        report.get("only_in_second"),
        report.get("shape_mismatch"),
        report.get("exceeding"),
    )
    return not any(flat) and ("weights" not in report or report_ok(report["weights"]))


# ------------------------------------------------------------------ command line


def _cmd_yamls(args: argparse.Namespace) -> int:
    for tier in args.tier or TIERS:
        for source in tier_sources(tier).values():
            logger.info("wrote %s", write_derived_yaml(source, tier))
    return 0


def size_violations(manifest: Mapping[str, Any], manifest_bytes: int = 0) -> list[str]:
    """The files of a fixture that break the size rule of its tier (empty: all within).

    Tiny: the files of one fixture, the manifest included, stay under ``TINY_BYTE_LIMIT`` together.
    Faithful: each file stays under ``FAITHFUL_BYTE_TARGET``.
    """
    files = manifest["files"]
    if manifest["tier"] == "tiny":
        total = sum(f["bytes"] for f in files.values()) + manifest_bytes
        return [manifest["name"]] if total >= TINY_BYTE_LIMIT else []
    return [f["name"] for f in files.values() if f["bytes"] >= FAITHFUL_BYTE_TARGET]


def tier_sources(tier: str) -> dict[str, Source]:
    """Return the sources of a tier: every source for tiny, the anchors for faithful."""
    if tier == "tiny":
        return sources()
    return {label: anchor_source(label) for label in MODELS}


def _cmd_write(args: argparse.Namespace) -> int:
    folder = args.out if args.out is not None else tier_dir(args.tier)
    every = tier_sources(args.tier)
    labels = list(every) if args.models is None else args.models
    unknown = sorted(set(labels) - set(every))
    if unknown:
        logger.error("no %s source is labelled %s (see: sources)", args.tier, unknown)
        return 2
    over: list[str] = []
    for manifest in write_tier(args.tier, folder, labels, every=every):
        size = (folder / f"{manifest['name']}.json").stat().st_size
        over += size_violations(manifest, size)
    for name in over:
        logger.error("over the size limit of its tier: %s", name)
    return 1 if over else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    failed = 0
    for path in sorted(args.folder.glob("*.json")):
        report = verify_fixture(args.folder, path.stem)
        ok = report_ok(report)
        failed += not ok
        logger.info(
            "%s: %s (max scaled diff %.2e)",
            path.stem,
            "ok" if ok else "DIFFERS",
            report["max_scaled_diff"],
        )
    return 1 if failed else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    report = compare_dirs(args.first, args.second)
    for name, one in report["fixtures"].items():
        logger.info(
            "%s: max abs %.2e, max scaled %.2e, %d arrays",
            name,
            one["max_abs_diff"],
            one["max_scaled_diff"],
            one["n_compared"],
        )
    return 0 if report_ok(report) else 1


def _cmd_cells(_: argparse.Namespace) -> int:
    lost = 0
    for label, source in sources().items():
        raw = parent_raw(source)
        for missing in missing_source_cells(source, raw, yaml_path(label, "tiny")):
            logger.error("%s: the tiny yaml lacks %s", label, missing)
            lost += 1
        merges = distinct_widths_merged(raw, widths_of_source(label))
        for merged in set(merges) - MERGED_WIDTHS.get(label, set()):
            logger.error("%s: the distinct widths %s get one tiny width", label, merged)
            lost += 1
    return 1 if lost else 0


def _cmd_sources(_: argparse.Namespace) -> int:
    for label, source in sources().items():
        logger.info("%s\t%s", label, source.row)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("yamls", help="write the derived yamls")
    one.add_argument("--tier", choices=list(TIERS), nargs="*")
    one.set_defaults(run=_cmd_yamls)
    one = sub.add_parser("write", help="write the fixtures of a tier")
    one.add_argument("--tier", choices=list(TIERS), required=True)
    one.add_argument("--out", type=Path, default=None)
    one.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="labels of the sources to write (default: every source of the tier)",
    )
    one.set_defaults(run=_cmd_write)
    one = sub.add_parser("verify", help="reload every fixture of a folder in TF")
    one.add_argument("folder", type=Path)
    one.set_defaults(run=_cmd_verify)
    one = sub.add_parser("compare", help="compare the fixtures of two folders")
    one.add_argument("first", type=Path)
    one.add_argument("second", type=Path)
    one.set_defaults(run=_cmd_compare)
    one = sub.add_parser(
        "cells",
        help="check that the tiny yamls exercise every option cell of their parents",
    )
    one.set_defaults(run=_cmd_cells)
    one = sub.add_parser("sources", help="list the label and row of every source")
    one.set_defaults(run=_cmd_sources)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import tensorpotential

    logger.info("library: %s", tensorpotential.__file__)
    args = build_parser().parse_args(argv)
    run: Callable[[argparse.Namespace], int] = args.run
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
