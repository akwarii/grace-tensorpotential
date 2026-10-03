"""Numeric oracle snapshot of the TensorFlow GRACE models.

Builds the random-weight TensorFlow models of the test yamls, overwrites every
floating-point variable with seeded values (several layers are zero-initialised, which
would make the outputs trivial) and records, for a few small structures, the energy,
atomic energies, forces, stress and every tensor that the instructions write into the
input dictionary. Two snapshots (for example before and after a cleanup) are then
compared key by key, with the maximum difference reported.

Keys are ``<model>/<case>/<quantity>``. The model label is the yaml stem (float64, the
original 369 keys) or the stem with a suffix: ``.f32`` (float32 parameters, float64
inputs, the precision of the foundation models), ``.lm_first`` and ``.dense`` (the layout
options, switched on in memory), or ``preset.<NAME>`` (a preset of
``tensorpotential/potentials/presets.py`` with a small element set). The cases are
``s0``, ``s1``, ``s2`` (structures of the test data) and the hand-built ``isolated``,
``dimer``, ``slab`` and ``selfimage`` structures.

Usage (from the repository root, one thread, CPU)::

    python tools/oracle_snapshot.py write baselines/oracle_snapshot.npz
    python tools/oracle_snapshot.py compare baselines/oracle_snapshot.npz new.npz
    python tools/oracle_snapshot.py spread run1.npz run2.npz run3.npz run4.npz run5.npz

``compare`` exits with status 1 when the key sets differ or when any difference exceeds
the tolerance. Two snapshots of the same code can differ by about one ulp in a few
intermediate tensors (energy, forces and stress did not change in any measured run, see
``baselines/README.md``), hence the default scaled tolerance, a named row per precision
(``TOLERANCE_ROWS``); ``--scale-rtol 0`` gives an exact comparison. ``spread`` reports
the repeat-to-repeat difference of snapshots of the same tree, which is where a
tolerance row comes from.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger("oracle_snapshot")

TESTS = Path(__file__).resolve().parents[1] / "tests"
DEFAULT_YAMLS = (
    "model_grace.yaml",
    "model_grace_2L_omat.yaml",
    "model_grace_2L_omat_large_base.yaml",
)
DEFAULT_STRUCTURES = "data/MoNbTaW_test50.pkl.gz"
N_STRUCTURES = 3
SEED = 20260930
META_KEY = "__meta__"
# virial order (xx, yy, zz, xy, xz, yz) -> ASE Voigt order (xx, yy, zz, yz, xz, xy)
VIRIAL_TO_VOIGT = [0, 1, 2, 5, 4, 3]


class ScaleTolerance(NamedTuple):
    """A comparison tolerance relative to the largest element of each array, and why."""

    scale_rtol: float
    rationale: str


# The named tolerance rows of ``compare``; a row changes only with a stated reason.
TOLERANCE_ROWS = {
    "float64": ScaleTolerance(
        1e-12,
        "intermediate tensors of the large_base model differ by about one ulp between "
        "processes (largest observed 8e-18 of the array maximum); see baselines/README.md",
    ),
    "float32": ScaleTolerance(
        0.0,
        "seven repeats of the untouched tree (4130 arrays each; six run concurrently, one alone) "
        "gave a spread of exactly 0 for every float32 quantity; one float32 ulp of a large element "
        "is 6e-8 of the array maximum, so any nonzero difference is a change of the code",
    ),
}
DEFAULT_SCALE_RTOL = TOLERANCE_ROWS["float64"].scale_rtol
F32_SUFFIX = ".f32"

GROUPS = ("base", "float32", "lm_first", "dense", "presets")
GROUP_SUFFIX = {
    "base": "",
    "float32": F32_SUFFIX,
    "lm_first": ".lm_first",
    "dense": ".dense",
}
GROUP_OPTION = {"lm_first": "lm_first", "dense": "dense_nbr"}
PRESET_PREFIX = "preset:"
PRESET_ELEMENTS = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
# Small versions of the presets: the structures of the test data have four elements.
PRESET_SETTINGS: dict[str, dict[str, Any]] = {
    "LINEAR": {"lmax": 2, "n_rad_base": 4, "n_rad_max": 6, "embedding_size": 4},
    "FS": {
        "lmax": (3, 3, 2, 2),
        "Lmax": (None, 2, 0, 0),
        "max_sum_l": (None, None, 4, 3),
        "lmax_hist": (None, None, None, 2),
        "n_rad_base": 6,
        "n_rad_max": (6, 5, 4, 3),
        "embedding_size": 8,
    },
    "GRACE_1LAYER_v2_25": {
        "lmax": 3,
        "n_rad_base": 6,
        "n_rad_max": 8,
        "prod_func_n_max": 8,
        "embedding_size": 8,
        "n_mlp_dens": 4,
        "max_order": 3,
    },
    "GRACE_2LAYER_v2_25": {
        "lmax": (3, 2),
        "n_rad_base": 6,
        "n_rad_max": (8, 6),
        "prod_func_n_max": (8, 8),
        "embedding_size": 8,
        "n_mlp_dens": 4,
        "max_order": 3,
        "indicator_lmax": 2,
    },
}
# (yaml, layout option) pairs the library cannot run; recorded in the metadata, not skipped silently
UNSUPPORTED: dict[tuple[str, str], str] = {
    ("model_grace.yaml", "lm_first"): (
        "MLPOut2ScalarTarget.frwrd (instructions/output.py) reads [:, :, 0] without the "
        "lm_first transpose, so its MLP gets the wrong axis (MatMul shape error)"
    )
}


@dataclass(frozen=True)
class ModelSpec:
    """One model of the snapshot: where its instructions come from and how it is built."""

    label: str
    source: str
    dtype: str = "float64"
    option: str | None = None


@functools.cache
def configure_tensorflow(seed: int = SEED):
    """Import TensorFlow on CPU with a single thread and deterministic ops."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # tensorpotential must come first: it selects the legacy Keras backend
    import tensorpotential  # noqa: F401, I001
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def seeded_values(name: str, shape: tuple[int, ...], seed: int = SEED) -> np.ndarray:
    """Non-trivial values for one variable, independent of the creation order.

    Matrices get ``N(0, 1/sqrt(shape[0]))``; scalars and vectors (biases, scales,
    norms) get ``N(1, 0.1)`` so that no layer is zeroed out.
    """
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    if len(shape) >= 2:
        return rng.normal(0.0, 1.0 / np.sqrt(shape[0]), size=shape)
    return rng.normal(1.0, 0.1, size=shape)


def _as_list(instructions) -> list:
    """The instructions of a list or of a ``{name: instruction}`` dictionary."""
    return list(
        instructions.values() if hasattr(instructions, "values") else instructions
    )


def _accepts_option(instruction, option: str) -> bool:
    """Whether the constructor of ``instruction`` takes the layout ``option``."""
    if option not in instruction._init_args:  # noqa: SLF001 - the captured constructor arguments
        return False
    return option != "dense_nbr" or getattr(instruction, "dense_capable", False)


def with_layout_option(instructions, option: str):
    """Instructions rebuilt with ``option=True`` on every instruction that takes it.

    The option is set in the serialised form, as a yaml edited by hand would be, so
    the constructors resolve it exactly as they do when a saved model is loaded.
    """
    import yaml
    from tensorpotential.instructions import load_instructions
    from tensorpotential.instructions.base import save_instructions_dict

    names = [i.name for i in _as_list(instructions) if _accepts_option(i, option)]
    if not names:
        raise ValueError(f"no instruction of the model takes the option {option!r}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.yaml"
        save_instructions_dict(str(path), instructions)
        raw = yaml.safe_load(path.read_text())
        for entry in _entries(raw):
            if entry["name"] in names:
                entry[option] = True
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return load_instructions(str(path))


def _entries(raw) -> list[dict]:
    """The instruction dictionaries of a loaded model yaml (list or ``instructions`` map)."""
    collection = raw["instructions"] if "instructions" in raw else raw
    return (
        list(collection.values()) if isinstance(collection, dict) else list(collection)
    )


def preset_instructions(name: str):
    """Instructions of the preset ``name`` with the small settings of ``PRESET_SETTINGS``."""
    configure_tensorflow()
    from tensorpotential.potentials import presets

    manager = getattr(presets, name)(
        element_map=PRESET_ELEMENTS, **PRESET_SETTINGS[name]
    )
    return manager.get_instructions()


def build_from_instructions(instructions, seed: int = SEED, dtype: str = "float64"):
    """Build a TPModel with parameters of ``dtype`` and seeded non-zero weights."""
    tf = configure_tensorflow(seed)
    from tensorpotential.tpmodel import TPModel

    model = TPModel(instructions)
    model.build(getattr(tf, dtype))
    # only trainable variables: constants such as the cutoff must keep their values
    for var in model.trainable_variables:
        if var.dtype.is_floating:
            values = seeded_values(var.name, tuple(var.shape), seed)
            var.assign(values.astype(var.dtype.as_numpy_dtype))
    return model


def build_model(
    yaml_path: Path, seed: int = SEED, dtype: str = "float64", option: str | None = None
):
    """Build the model of ``yaml_path``; ``option`` switches a layout option on first."""
    configure_tensorflow(seed)
    from tensorpotential.instructions import load_instructions

    instructions = load_instructions(str(yaml_path))
    if option is not None:
        instructions = with_layout_option(instructions, option)
    return build_from_instructions(instructions, seed, dtype)


def build_spec_model(spec: ModelSpec, seed: int = SEED):
    """Build the model that ``spec`` describes."""
    if spec.source.startswith(PRESET_PREFIX):
        instructions = preset_instructions(spec.source[len(PRESET_PREFIX) :])
        if spec.option is not None:
            instructions = with_layout_option(instructions, spec.option)
        return build_from_instructions(instructions, seed, spec.dtype)
    return build_model(TESTS / spec.source, seed, spec.dtype, spec.option)


def load_structures(pickle_path: Path, n: int) -> list:
    """One structure for each of the ``n`` smallest distinct sizes (first in file)."""
    import pandas as pd

    chosen: dict[int, object] = {}
    for atoms in pd.read_pickle(pickle_path)["ase_atoms"]:  # noqa: S301 - repo test data
        chosen.setdefault(len(atoms), atoms)
    return [chosen[size] for size in sorted(chosen)[:n]]


def edge_structures() -> dict[str, Any]:
    """Hand-built structures that the test data does not contain.

    ``isolated``: one atom, whose only bond is the dummy one that the data builder adds.
    ``dimer``: two atoms, no periodicity, no cell (the stress is recorded as zero).
    ``slab``: four atoms with vacuum, periodic in x and y only. ``selfimage``: two atoms
    in a cell smaller than the cutoff, so that bonds join an atom to its own image
    (``i == j`` with a non-zero shift); the positions are generic so that the forces
    do not vanish by symmetry.
    """
    from ase import Atoms

    return {
        "isolated": Atoms("Mo", positions=[[0.0, 0.0, 0.0]], pbc=False),
        "dimer": Atoms("MoNb", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]], pbc=False),
        "slab": Atoms(
            "Nb4",
            positions=[[0, 0, 0], [1.65, 1.65, 0], [0, 0, 1.65], [1.65, 1.65, 1.65]],
            cell=[3.3, 3.3, 20.0],
            pbc=[True, True, False],
        ),
        "selfimage": Atoms(
            "MoW",
            scaled_positions=[[0.0, 0.0, 0.0], [0.37, 0.41, 0.29]],
            cell=[2.7, 2.9, 3.1],
            pbc=True,
        ),
    }


def dense_layout_active(model) -> bool:
    """Whether some instruction of ``model`` aggregates bonds in the dense layout."""
    return any(
        getattr(i, "dense_capable", False) and getattr(i, "dense_nbr", False)
        for i in _as_list(model.instructions)
    )


def _geometry_batch(model, atoms) -> dict:
    """The input batch of one structure, in the bond layout the model expects.

    The structure itself is left unchanged.
    """
    from tensorpotential.data.databuilder import (
        GeometricalDataBuilder,
        construct_batches,
    )
    from tensorpotential.tpmodel import extract_cutoff_and_elements

    cutoff, symbols, index = extract_cutoff_and_elements(model.instructions)
    dense = dense_layout_active(model)
    builder = GeometricalDataBuilder(
        elements_map=dict(zip(symbols, index, strict=True)),
        cutoff=float(cutoff),
        dense_nbr=dense,
    )
    if dense:
        batches = construct_batches(
            [atoms.copy()],
            data_builders=[builder],
            batch_size=1,
            max_n_buckets=1,
            verbose=False,
        )
        return batches[0]
    # extract_from_ase_atoms edits its argument (enforce_pbc gives a non-periodic structure a
    # cell and moves it), which would change the structure for the models evaluated next
    return builder.join_to_batch([builder.extract_from_ase_atoms(atoms.copy())])


def evaluate(model, atoms) -> dict[str, np.ndarray]:
    """Energy, forces, stress and every instruction output for one structure."""
    tf = configure_tensorflow()
    from tensorpotential import constants

    batch = _geometry_batch(model, atoms)
    data = tf.data.Dataset.from_tensors(batch).get_single_element()
    # compute_function runs the instructions on ``data`` itself (not a copy)
    result = model.compute_function(model.instructions, data, training=False)

    out = {
        "energy": result[constants.PREDICT_TOTAL_ENERGY].numpy(),
        "atomic_energy": result[constants.PREDICT_ATOMIC_ENERGY].numpy(),
        "forces": result[constants.PREDICT_FORCES].numpy(),
        "virial": result[constants.PREDICT_VIRIAL].numpy(),
    }
    virial = out["virial"].reshape(6)[VIRIAL_TO_VOIGT]
    out["stress"] = -virial / atoms.get_volume() if atoms.cell.rank == 3 else virial * 0
    for key, value in data.items():
        if isinstance(value, tf.Tensor):
            out[f"data/{key}"] = value.numpy()
    return out


def variable_table(model) -> list[list]:
    """(name, shape, dtype) of every variable, in model order."""
    return [[v.name, list(v.shape), v.dtype.name] for v in model.variables]


def snapshot_specs(
    groups: Sequence[str] = GROUPS, yamls: Sequence[str] = DEFAULT_YAMLS
) -> tuple[list[ModelSpec], dict[str, str]]:
    """The models of the groups, and the combinations skipped with their reasons."""
    specs: list[ModelSpec] = []
    skipped: dict[str, str] = {}
    for group in groups:
        if group == "presets":
            specs += [
                ModelSpec(f"preset.{name}", PRESET_PREFIX + name)
                for name in PRESET_SETTINGS
            ]
        elif group in GROUP_SUFFIX:
            for name in yamls:
                label = Path(name).stem + GROUP_SUFFIX[group]
                reason = UNSUPPORTED.get((name, group))
                if reason is not None:
                    skipped[label] = reason
                    continue
                dtype = "float32" if group == "float32" else "float64"
                specs.append(ModelSpec(label, name, dtype, GROUP_OPTION.get(group)))
        else:
            raise ValueError(f"unknown group {group!r}; choose from {GROUPS}")
    return specs, skipped


def describe_structure(atoms) -> dict:
    """Formula, size, periodicity and cell of a structure, for the metadata."""
    return {
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "pbc": [bool(p) for p in atoms.pbc],
        "cell": np.asarray(atoms.cell.array).tolist(),
    }


def _versions(tf) -> dict[str, str]:
    import ase
    import pandas

    return {
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "ase": ase.__version__,
        "python": sys.version.split()[0],
    }


def take_snapshot(
    yamls: tuple[str, ...] = DEFAULT_YAMLS,
    structures: str = DEFAULT_STRUCTURES,
    n_structures: int = N_STRUCTURES,
    seed: int = SEED,
    groups: tuple[str, ...] = GROUPS,
    edge: bool = True,
) -> dict[str, np.ndarray]:
    """Arrays keyed ``<model>/<case>/<quantity>`` plus a JSON ``__meta__`` entry."""
    tf = configure_tensorflow(seed)
    cases = dict(enumerate_cases(load_structures(TESTS / structures, n_structures)))
    if edge:
        cases.update(edge_structures())
    specs, skipped = snapshot_specs(groups, yamls)
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {
        "seed": seed,
        "versions": _versions(tf),
        "groups": list(groups),
        "specs": [asdict(spec) for spec in specs],
        "skipped": skipped,
        "structures": {name: describe_structure(a) for name, a in cases.items()},
        "yaml_sha256": {},
        "variables": {},
    }
    for spec in specs:
        if not spec.source.startswith(PRESET_PREFIX):
            data = (TESTS / spec.source).read_bytes()
            meta["yaml_sha256"][Path(spec.source).stem] = hashlib.sha256(
                data
            ).hexdigest()
        model = build_spec_model(spec, seed)
        meta["variables"][spec.label] = variable_table(model)
        for case, atoms in cases.items():
            for key, value in evaluate(model, atoms).items():
                arrays[f"{spec.label}/{case}/{key}"] = value
        logger.info("%s: %d arrays so far", spec.label, len(arrays))
    arrays[META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    return arrays


def enumerate_cases(atoms_list: list) -> list[tuple[str, Any]]:
    """``("s0", atoms_0), ("s1", atoms_1), ...``."""
    return [(f"s{i}", atoms) for i, atoms in enumerate(atoms_list)]


def precision_of(key: str) -> str:
    """The precision row (``float32`` or ``float64``) of an array key."""
    return "float32" if key.split("/", 1)[0].endswith(F32_SUFFIX) else "float64"


def scale_rtol_for_key(
    key: str, rows: Mapping[str, ScaleTolerance] = TOLERANCE_ROWS
) -> float:
    """The scaled tolerance of the named row of ``rows`` that applies to ``key``."""
    return rows[precision_of(key)].scale_rtol


def compare_snapshots(
    a: Mapping[str, np.ndarray],
    b: Mapping[str, np.ndarray],
    atol: float = 0.0,
    rtol: float = 0.0,
    scale_rtol: float | Callable[[str], float] = 0.0,
) -> dict:
    """Key-by-key comparison; reports key mismatches and the maximum differences.

    An element passes when ``|a - b| <= atol + rtol * |b| + scale_rtol * max|b|``, the
    last term being relative to the largest element of the same array, which absorbs
    round-off noise in elements that are nearly zero. ``scale_rtol`` is a number or a
    function of the key (see ``scale_rtol_for_key``).
    """
    keys_a = {k for k in a if k != META_KEY}
    keys_b = {k for k in b if k != META_KEY}
    report = {
        "only_in_first": sorted(keys_a - keys_b),
        "only_in_second": sorted(keys_b - keys_a),
        "shape_mismatch": [],
        "exceeding": [],
        "max_abs_diff": 0.0,
        "max_abs_key": None,
        "max_scaled_diff": 0.0,
        "n_compared": 0,
    }
    for key in sorted(keys_a & keys_b):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            report["shape_mismatch"].append(key)
            continue
        report["n_compared"] += 1
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        worst = float(diff.max()) if diff.size else 0.0
        if worst > report["max_abs_diff"]:
            report["max_abs_diff"], report["max_abs_key"] = worst, key
        reference = np.abs(y.astype(np.float64))
        peak = float(reference.max()) if reference.size else 0.0
        if peak > 0:
            report["max_scaled_diff"] = max(report["max_scaled_diff"], worst / peak)
        rel = (
            float(scale_rtol)
            if isinstance(scale_rtol, int | float)
            else scale_rtol(key)
        )
        bound = atol + rtol * reference + rel * peak
        if not np.all(diff <= bound):
            report["exceeding"].append(key)
    report["ok"] = not (
        report["only_in_first"]
        or report["only_in_second"]
        or report["shape_mismatch"]
        or report["exceeding"]
    )
    return report


def repeat_spread(snapshots: Sequence[Mapping[str, np.ndarray]]) -> dict[str, dict]:
    """Largest repeat-to-repeat difference per precision and quantity.

    For every key present in all snapshots the spread is the largest elementwise range
    over the repeats divided by the largest element of the first snapshot (the scale of
    ``compare``). Keys are grouped as ``<precision>/<quantity>``, for example
    ``float32/energy`` or ``float64/data/B``; each group reports its maximum and the key
    that gives it.
    """
    if len(snapshots) < 2:
        raise ValueError("a repeat spread needs at least two snapshots")
    keys = set.intersection(*({k for k in s if k != META_KEY} for s in snapshots))
    groups: dict[str, dict] = {}
    for key in sorted(keys):
        stack = np.stack([np.asarray(s[key], dtype=np.float64) for s in snapshots])
        scale = float(np.abs(stack[0]).max()) if stack[0].size else 0.0
        spread = (
            float(np.max(np.max(stack, 0) - np.min(stack, 0))) if stack.size else 0.0
        )
        entry = groups.setdefault(
            f"{precision_of(key)}/{key.split('/', 2)[2]}",
            {"max_scaled_spread": 0.0, "worst_key": None},
        )
        scaled = spread / scale if scale > 0 else float(spread > 0)
        if scaled >= entry["max_scaled_spread"]:
            entry["max_scaled_spread"], entry["worst_key"] = scaled, key
    return groups


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="take a snapshot and save it as .npz")
    write.add_argument("out", type=Path)
    write.add_argument(
        "--yamls",
        nargs="+",
        default=list(DEFAULT_YAMLS),
        help="model yamls in tests/ (default: the three test models)",
    )
    write.add_argument("--n-structures", type=int, default=N_STRUCTURES)
    write.add_argument(
        "--groups",
        nargs="+",
        default=list(GROUPS),
        choices=GROUPS,
        help="model groups to record (default: all)",
    )
    write.add_argument(
        "--no-edge", action="store_true", help="leave out the hand-built structures"
    )
    cmp_ = sub.add_parser("compare", help="compare two snapshots")
    cmp_.add_argument("first", type=Path)
    cmp_.add_argument("second", type=Path)
    cmp_.add_argument("--atol", type=float, default=0.0)
    cmp_.add_argument("--rtol", type=float, default=0.0)
    cmp_.add_argument(
        "--scale-rtol",
        type=float,
        default=None,
        help="tolerance relative to the largest element of each array "
        "(default: the row of TOLERANCE_ROWS for the precision of the key)",
    )
    spread = sub.add_parser("spread", help="repeat spread of snapshots of one tree")
    spread.add_argument("snapshots", nargs="+", type=Path)
    return parser.parse_args(argv)


def _write(args: argparse.Namespace) -> int:
    arrays = take_snapshot(
        tuple(args.yamls),
        n_structures=args.n_structures,
        groups=tuple(args.groups),
        edge=not args.no_edge,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)  # ty: ignore[invalid-argument-type]
    meta = json.loads(str(arrays[META_KEY]))
    meta["npz_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta["n_arrays"] = len(arrays) - 1
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n"
    )
    size = args.out.stat().st_size / 1e6
    sys.stdout.write(
        f"{len(arrays) - 1} arrays written to {args.out} ({size:.1f} MB)\n"
    )
    return 0


def _spread(paths: list[Path]) -> int:
    opened = [np.load(path) for path in paths]
    try:
        report = repeat_spread(opened)
    finally:
        for snapshot in opened:
            snapshot.close()
    sys.stdout.write(json.dumps(report, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``write``, ``compare`` or ``spread`` command; return the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.command == "write":
        return _write(args)
    if args.command == "spread":
        return _spread(args.snapshots)
    scale = scale_rtol_for_key if args.scale_rtol is None else args.scale_rtol
    with np.load(args.first) as fa, np.load(args.second) as fb:
        report = compare_snapshots(fa, fb, args.atol, args.rtol, scale)
    sys.stdout.write(json.dumps(report, indent=1) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
