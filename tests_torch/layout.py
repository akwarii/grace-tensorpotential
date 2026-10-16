"""The test boundary between TF's feature layouts and the backend's ``[atoms, lm, n]`` (``ops.precision.FEATURE_LAYOUT``).

TF stores an equivariant feature as ``[atoms, n, lm]`` by default and as ``[lm, atoms, n]`` with ``lm_first``; the
twins use ``[atoms, lm, n]`` and never transpose in a forward pass. A golden is permuted here, once, when a test reads it.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

TfLayout = Literal["default", "lm_first"]

LM_AXIS: Final[dict[TfLayout, int]] = {"default": -1, "lm_first": 0}
"""Where TF puts the angular axis of a three-axis feature."""


def to_internal(feature: np.ndarray, layout: TfLayout) -> np.ndarray:
    """Return a TF feature of three axes in the backend's layout ``[atoms, lm, n]`` (a view, no copy).

    Raises
    ------
    ValueError
        When ``feature`` does not have three axes.
    """
    if feature.ndim != 3:
        msg = f"a feature has three axes, got shape {feature.shape}"
        raise ValueError(msg)
    return np.moveaxis(feature, LM_AXIS[layout], 1)
