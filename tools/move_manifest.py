"""Prove that the library code moved out of ``scripts/`` and ``cli/`` (CLI2) kept every unit and every old name.

``record`` is run once on the unmodified tree: it stores, per moved module, the AST hash of every unit that moves
(``hash_source`` of ``ast_manifest.py``) and every module-level name importable from the old path. ``check`` is run
after the moves: the hash of each unit at its new path must equal the stored one, and every stored name must still be
a top-level name of the old path (a re-export module imports it explicitly).

Usage::

    python tools/move_manifest.py record baselines/cli2_moves.json --rev <rev of the unmodified tree>
    python tools/move_manifest.py check baselines/cli2_moves.json
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ast_manifest import hash_source

ROOT = Path(__file__).resolve().parent.parent
PKG = "tensorpotential"


@dataclass(frozen=True)
class Move:
    """One module whose code moves: where it is, where it goes and which units go (``None``: all of them)."""

    old: str
    new: str
    units: tuple[str, ...] | None


MOVES: tuple[Move, ...] = (
    Move(f"{PKG}/cli/data.py", f"{PKG}/data/datasets.py", None),
    Move(f"{PKG}/scripts/_kokkos_export.py", f"{PKG}/export_kokkos.py", None),
    Move(
        f"{PKG}/scripts/grace_preprocess.py",
        f"{PKG}/data/preprocess.py",
        (
            "EQUI_STRUCTURE_STRATEGY",
            "EQUI_ATOMS_STRATEGY",
            "EQUI_NEIGH_STRATEGY",
            "orders",
            "remap_type_to_tf_dtype",
            "get_elements",
            "get_batch_dtypes",
            "databuilder_generator",
            "infer_tensorspec",
            "get_window_stat",
            "compute_batches_df",
            "get_batch_list",
            "get_equi_structures_batch_list",
            "get_equi_batch_list",
            "get_batch_stat_tf",
            "get_bound_for_batch",
            "dataframe_iterrows_generator",
            "join_and_pad_generator",
            "infer_batch_signature",
            "save_batches_df",
            "load_and_reduce_stats",
            "load_params",
            "save_params",
            "get_databuilders",
        ),
    ),
    # the operations are rewritten (keyword arguments instead of a Namespace), so only the helper that moves as it is
    Move(f"{PKG}/scripts/grace_utils.py", f"{PKG}/model_ops.py", ("load_checkpoint",)),
)


def top_level_units(source: str) -> dict[str, str]:
    """Return ``{name: hash}`` for every function, class and plain assignment at module level.

    Parameters
    ----------
    source : str
        Python source of a module.

    Returns
    -------
    dict[str, str]
        AST hash of the source segment of each definition, keyed by its name (the first target of an assignment).
    """
    units: dict[str, str] = {}
    for node in ast.parse(source).body:
        name = _bound_name(node)
        if name is not None:
            segment = ast.get_source_segment(source, node) or ""
            units[name] = hash_source(segment)
    return units


def _bound_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _read(path: str, rev: str | None, root: Path = ROOT) -> str:
    if rev is None:
        return (root / path).read_text()
    out = subprocess.run(  # noqa: S603
        ["git", "show", f"{rev}:{path}"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def record(rev: str | None, root: Path = ROOT) -> dict[str, dict]:
    """Return the manifest of the unmodified tree (``rev``) or of the working tree."""
    manifest: dict[str, dict] = {}
    for move in MOVES:
        units = top_level_units(_read(move.old, rev, root))
        wanted = units if move.units is None else {n: units[n] for n in move.units}
        manifest[move.old] = {"new": move.new, "names": sorted(units), "units": wanted}
    return manifest


def check(manifest: dict[str, dict], root: Path = ROOT) -> list[str]:
    """Return one message per moved unit that changed or went missing, and per old name that is gone."""
    problems: list[str] = []
    for old, entry in manifest.items():
        new_units = top_level_units(_read(entry["new"], None, root))
        for name, digest in entry["units"].items():
            if name not in new_units:
                problems.append(f"{entry['new']}: {name} is missing")
            elif new_units[name] != digest:
                problems.append(
                    f"{entry['new']}: {name} changed (AST hash differs from {old})"
                )
        old_names = _old_names(old, root)
        problems += [
            f"{old}: {n} is no longer importable"
            for n in entry["names"]
            if n not in old_names
        ]
    return problems


def _old_names(old: str, root: Path) -> set[str]:
    """Top-level names of the old module, imports included: a re-export module binds them by import."""
    source = (root / old).read_text()
    names: set[str] = set(top_level_units(source))
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom | ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def main(argv: list[str] | None = None) -> int:
    """Run ``record`` or ``check``; ``check`` exits with status 1 on any problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("record", "check"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--rev",
        default=None,
        help="git revision to read for record (default: working tree)",
    )
    args = parser.parse_args(argv)
    if args.command == "record":
        args.manifest.write_text(
            json.dumps(record(args.rev, ROOT), indent=1, sort_keys=True) + "\n"
        )
        return 0
    problems = check(json.loads(args.manifest.read_text()), ROOT)
    sys.stdout.write(
        "".join(f"{line}\n" for line in problems) + f"{len(problems)} problem(s)\n"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
