"""The committed golden anchors of the torch twins, their tolerance table and its edit-lock; a reader of the anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

GOLDEN: Final = Path(__file__).resolve().parent
TINY_F64: Final = ("large_tiny_f64", "omat_tiny_f64")
"""The float64 anchors of the default layout; ``*_f32`` has the same weights in float32."""


def load(name: str) -> dict[str, np.ndarray]:
    """Return every array of the anchor ``name`` (``omat_tiny_f64``, ...), read without pickle."""
    with np.load(GOLDEN / f"{name}.npz", allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def cases(arrays: dict[str, np.ndarray]) -> list[str]:
    """Return the structures of an anchor (the first part of its ``<case>/res/...`` keys), sorted."""
    return sorted({key.split("/")[0] for key in arrays if "/res/" in key})
