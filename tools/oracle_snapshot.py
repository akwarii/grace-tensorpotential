"""Numeric oracle snapshot of the TensorFlow GRACE models.

Builds the random-weight TensorFlow models of the test yamls, overwrites every
floating-point variable with seeded values (several layers are zero-initialised, which
would make the outputs trivial) and records, for a few small structures, the energy,
atomic energies, forces, stress and every tensor that the instructions write into the
input dictionary. Two snapshots (for example before and after a cleanup) are then
compared key by key, with the maximum difference reported.

Usage (from the repository root, one thread, CPU)::

    python tools/oracle_snapshot.py write baselines/oracle_snapshot.npz
    python tools/oracle_snapshot.py compare baselines/oracle_snapshot.npz new.npz

``compare`` exits with status 1 when the key sets differ or when any difference exceeds
the tolerance. Two snapshots of the same code can differ by about one ulp in a few
intermediate tensors (energy, forces and stress did not change in any measured run, see
``baselines/README.md``), hence the default ``--scale-rtol 1e-12``; pass ``0`` for an
exact comparison.
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
import zlib
from pathlib import Path

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
DEFAULT_SCALE_RTOL = 1e-12
# virial order (xx, yy, zz, xy, xz, yz) -> ASE Voigt order (xx, yy, zz, yz, xz, xy)
VIRIAL_TO_VOIGT = [0, 1, 2, 5, 4, 3]


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


def build_model(yaml_path: Path, seed: int = SEED):
    """Build a float64 TPModel from ``yaml_path`` with seeded non-zero weights."""
    tf = configure_tensorflow(seed)
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import TPModel

    model = TPModel(load_instructions(str(yaml_path)))
    model.build(tf.float64)
    # only trainable variables: constants such as the cutoff must keep their values
    for var in model.trainable_variables:
        if var.dtype.is_floating:
            var.assign(seeded_values(var.name, tuple(var.shape), seed))
    return model


def load_structures(pickle_path: Path, n: int) -> list:
    """One structure for each of the ``n`` smallest distinct sizes (first in file)."""
    import pandas as pd

    chosen: dict[int, object] = {}
    for atoms in pd.read_pickle(pickle_path)["ase_atoms"]:  # noqa: S301 - repo test data
        chosen.setdefault(len(atoms), atoms)
    return [chosen[size] for size in sorted(chosen)[:n]]


def evaluate(model, atoms) -> dict[str, np.ndarray]:
    """Energy, forces, stress and every instruction output for one structure."""
    tf = configure_tensorflow()
    from tensorpotential import constants
    from tensorpotential.data.databuilder import GeometricalDataBuilder
    from tensorpotential.tpmodel import extract_cutoff_and_elements

    cutoff, symbols, index = extract_cutoff_and_elements(model.instructions)
    builder = GeometricalDataBuilder(
        elements_map=dict(zip(symbols, index, strict=True)), cutoff=float(cutoff)
    )
    batch = builder.join_to_batch([builder.extract_from_ase_atoms(atoms)])
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


def take_snapshot(
    yamls: tuple[str, ...] = DEFAULT_YAMLS,
    structures: str = DEFAULT_STRUCTURES,
    n_structures: int = N_STRUCTURES,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Arrays keyed ``<yaml stem>/s<i>/<quantity>`` plus a JSON ``__meta__`` entry."""
    tf = configure_tensorflow(seed)
    import ase
    import pandas

    atoms_list = load_structures(TESTS / structures, n_structures)
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {
        "seed": seed,
        "versions": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "ase": ase.__version__,
            "python": sys.version.split()[0],
        },
        "structures": [
            {"n_atoms": len(a), "formula": a.get_chemical_formula()} for a in atoms_list
        ],
        "yaml_sha256": {},
        "variables": {},
    }
    for name in yamls:
        stem = Path(name).stem
        path = TESTS / name
        meta["yaml_sha256"][stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = build_model(path, seed)
        meta["variables"][stem] = variable_table(model)
        for i, atoms in enumerate(atoms_list):
            for key, value in evaluate(model, atoms).items():
                arrays[f"{stem}/s{i}/{key}"] = value
        logger.info("%s: %d arrays so far", stem, len(arrays))
    arrays[META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    return arrays


def compare_snapshots(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    atol: float = 0.0,
    rtol: float = 0.0,
    scale_rtol: float = 0.0,
) -> dict:
    """Key-by-key comparison; reports key mismatches and the maximum differences.

    An element passes when ``|a - b| <= atol + rtol * |b| + scale_rtol * max|b|``, the
    last term being relative to the largest element of the same array, which absorbs
    round-off noise in elements that are nearly zero.
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
        scale = float(reference.max()) if reference.size else 0.0
        if scale > 0:
            report["max_scaled_diff"] = max(report["max_scaled_diff"], worst / scale)
        bound = atol + rtol * reference + scale_rtol * scale
        if not np.all(diff <= bound):
            report["exceeding"].append(key)
    report["ok"] = not (
        report["only_in_first"]
        or report["only_in_second"]
        or report["shape_mismatch"]
        or report["exceeding"]
    )
    return report


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
    cmp_ = sub.add_parser("compare", help="compare two snapshots")
    cmp_.add_argument("first", type=Path)
    cmp_.add_argument("second", type=Path)
    cmp_.add_argument("--atol", type=float, default=0.0)
    cmp_.add_argument("--rtol", type=float, default=0.0)
    cmp_.add_argument(
        "--scale-rtol",
        type=float,
        default=DEFAULT_SCALE_RTOL,
        help="tolerance relative to the largest element of each array",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ``write`` or ``compare`` command; return the process exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.command == "write":
        arrays = take_snapshot(tuple(args.yamls), n_structures=args.n_structures)
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
    with np.load(args.first) as fa, np.load(args.second) as fb:
        report = compare_snapshots(
            dict(fa), dict(fb), args.atol, args.rtol, args.scale_rtol
        )
    sys.stdout.write(json.dumps(report, indent=1) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
