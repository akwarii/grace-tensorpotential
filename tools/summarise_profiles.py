"""Summarise the JSON files of ``tools/profile_instructions.py`` into one markdown table (QUAL9).

One row per profile (architecture, dtype, bond layout, gradients, size): whole-model time, time per atom,
peak GPU and host memory (per bond), and the three classes that take most of the summed
forward+backward time. Used on the output directory of ``tools/slurm/gpu_architectures.sh`` and on a CPU run.

Usage::

    python tools/summarise_profiles.py DIRECTORY [DIRECTORY ...] [--out summary.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

TOP_CLASSES = 3


def load_profiles(paths: list[Path]) -> list[dict[str, Any]]:
    """Return every profile of the ``*.json`` files under ``paths``, each with its ``environment``.

    Files that are not profile output (no ``profiles`` key) are skipped.
    """
    payloads = [
        json.loads(file.read_text())
        for base in paths
        for file in sorted(base.rglob("*.json"))
    ]
    return [
        {**profile, "environment": payload.get("environment", {})}
        for payload in payloads
        if isinstance(payload, dict) and "profiles" in payload
        for profile in payload["profiles"]
    ]


def class_shares(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Return ``(class, percent)`` of the summed forward+backward time per class, largest first."""
    by_class: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        if row.get("fwd_bwd_ms") is not None:
            by_class[row["class"]] += row["fwd_bwd_ms"]
    total = sum(by_class.values())
    if total <= 0:
        return []
    return sorted(
        ((c, 100 * v / total) for c, v in by_class.items()), key=lambda kv: -kv[1]
    )


def _kb_per_bond(megabytes: float | None, bonds: int) -> str:
    return "" if megabytes is None or bonds == 0 else f"{megabytes * 1024 / bonds:.0f}"


def _device(profile: dict[str, Any]) -> str:
    env = profile["environment"]
    return env.get("gpu_name") or env.get("device", "?")


def row_of(profile: dict[str, Any]) -> list[str]:
    """Return the table cells of one profile."""
    atoms, bonds = profile["n_atoms"], profile["n_bonds"]
    gpu, host = profile.get("gpu_peak_memory_mb"), profile.get("peak_rss_mb")
    top = ", ".join(
        f"{c} {p:.0f}%" for c, p in class_shares(profile["rows"])[:TOP_CLASSES]
    )
    return [
        Path(profile["yaml"]).stem,
        _device(profile),
        profile["dtype"],
        profile["option"],
        profile.get("grads", "all") if profile["rows"] else "",
        str(atoms),
        str(bonds),
        f"{profile['whole_ms']:.0f}",
        f"{profile['whole_ms'] / atoms:.1f}",
        "" if gpu is None else f"{gpu:.0f}",
        _kb_per_bond(gpu, bonds),
        "" if host is None else f"{host:.0f}",
        top,
    ]


HEADER = [
    "model", "device", "dtype", "layout", "grads", "atoms", "bonds", "whole ms", "ms/atom",
    "peak GPU MB", "GPU KB/bond", "peak host MB", "top classes (share of fwd+bwd)",
]  # fmt: skip


def render(profiles: list[dict[str, Any]]) -> str:
    """Return the markdown table, sorted by model, dtype, layout, grads and size."""
    rows = [row_of(p) for p in profiles]
    rows.sort(key=lambda r: (r[0], r[2], r[3], r[4], int(r[5])))
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print (and optionally write) the table of the profiles under the given directories."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="also write the table here")
    args = parser.parse_args(argv)
    profiles = load_profiles(args.directories)
    if not profiles:
        sys.stderr.write("no profile JSON found\n")
        return 1
    table = render(profiles)
    sys.stdout.write(table + "\n")
    if args.out:
        args.out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
