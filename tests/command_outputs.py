"""Run ``grace_utils export``, ``grace_utils export_kokkos`` and ``grace_preprocess`` on seeded inputs and digest what they write.

The digests of ``tests/data/command_outputs.json`` were taken before the library code moved out of ``scripts/`` (CLI2);
a move must leave them unchanged. A file is digested by its content, not its bytes, where the container is not stable
across library versions (an ``npz`` and a pickled frame); a TensorFlow dataset by its samples (its file names carry random ids), anything else by bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from ase.build import bulk

from tensorpotential.data.process_df import ENERGY_CORRECTED_COL, FORCES_COL, STRESS_COL
from tensorpotential.scripts import grace_preprocess, grace_utils
from tests.saved_models import FS_CONFIG, save_seeded_model

_UNSTABLE = {"fingerprint.pb", "saved_model.pb"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_npz(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            array = data[name]
            digest.update(f"{name}|{array.dtype}|{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    """Digest of one output file: the content of an ``npz``, a pickled frame or a json file, else its bytes."""
    if path.suffix == ".npz":
        return _digest_npz(path)
    if path.name.endswith(".pkl.gz"):
        return _sha(str(pd.read_pickle(path).to_json()).encode())  # noqa: S301 (a frame this test wrote)
    if path.suffix == ".json":
        return _sha(json.dumps(json.loads(path.read_text()), sort_keys=True).encode())
    return _sha(path.read_bytes())


def _digest_dataset(path: Path) -> str:
    """Digest of the samples of a TensorFlow dataset directory (its file names carry random ids)."""
    digest = hashlib.sha256()
    for sample in tf.data.Dataset.load(str(path), compression="GZIP"):
        for key in sorted(sample):
            value = sample[key].numpy()
            digest.update(f"{key}|{value.dtype}|{value.shape}".encode())
            digest.update(
                value.tobytes()
                if value.dtype != object
                else repr(value.tolist()).encode()
            )
    return digest.hexdigest()


def digest_tree(root: Path) -> dict[str, str]:
    """Digest of every file below ``root`` (a ``shard_*`` dataset counts as one entry), keyed by relative path.

    ``fingerprint.pb`` of a SavedModel holds a random id and ``saved_model.pb`` names its traced functions by a
    process-wide counter, so the two are left out; the weights and the metadata of a SavedModel stay.
    """
    shards = [p for p in sorted(root.rglob("shard_*")) if p.is_dir()]
    digests = {str(p.relative_to(root)): _digest_dataset(p) for p in shards}
    for path in sorted(root.rglob("*")):
        inside_shard = any(shard in path.parents for shard in shards)
        if path.is_file() and not inside_shard and path.name not in _UNSTABLE:
            digests[str(path.relative_to(root))] = digest_file(path)
    return digests


def seeded_frame() -> pd.DataFrame:
    """Three rattled four-atom Cu cells of different size with energy, forces and stress."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        atoms = bulk("Cu", "fcc", a=3.5 + 0.1 * i, cubic=True)
        atoms.rattle(stdev=0.05, rng=rng)
        rows.append({
            "ase_atoms": atoms,
            ENERGY_CORRECTED_COL: -3.5 * len(atoms),
            FORCES_COL: rng.normal(size=(len(atoms), 3)),
            STRESS_COL: np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]),
        })
    return pd.DataFrame(rows)


def kokkos_output(directory: Path) -> Path:
    """``grace_utils export_kokkos`` of a seeded GRACE-1L model; returns the ``npz`` written."""
    model, checkpoint = save_seeded_model(directory, {"H": 0})
    out = directory / "out" / "weights.npz"
    out.parent.mkdir()
    grace_utils.main([
        "-p",
        str(model),
        "-c",
        str(checkpoint),
        "export_kokkos",
        "-o",
        str(out),
    ])
    return out.parent


def fs_yaml_output(directory: Path) -> Path:
    """``grace_utils export -sf`` of a seeded FS model; returns the directory holding the yaml."""
    model, checkpoint = save_seeded_model(
        directory, {"Al": 0, "Li": 1}, preset="FS", config=FS_CONFIG
    )
    out = directory / "out"
    out.mkdir()
    grace_utils.main([
        "-p",
        str(model),
        "-c",
        str(checkpoint),
        "export",
        "-sf",
        "-n",
        str(out / "fs_model"),
    ])
    return out


def saved_model_output(directory: Path) -> Path:
    """``grace_utils export`` of a seeded GRACE-1L model to a SavedModel; returns the directory holding it."""
    model, checkpoint = save_seeded_model(directory, {"H": 0})
    out = directory / "out"
    out.mkdir()
    grace_utils.main([
        "-p",
        str(model),
        "-c",
        str(checkpoint),
        "export",
        "-n",
        str(out / "saved_model"),
    ])
    return out


def preprocess_output(directory: Path) -> Path:
    """All four stages of ``grace_preprocess`` on ``seeded_frame`` (batches of two structures, two buckets)."""
    source = directory / "data.pkl.gz"
    seeded_frame().to_pickle(source)
    out = directory / "out"
    for stage in (1, 2, 3, 4):
        grace_preprocess.main([
            str(source),
            "-o",
            str(out),
            "-e",
            "Cu",
            "-c",
            "4.0",
            "--is-fit-stress",
            f"--stage-{stage}",
            "-b",
            "2",
            "-bu",
            "2",
        ])
    return out


OUTPUTS = {
    "export_kokkos": kokkos_output,
    "export_sf": fs_yaml_output,
    "export_saved_model": saved_model_output,
    "grace_preprocess": preprocess_output,
}
