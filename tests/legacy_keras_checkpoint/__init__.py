"""A checkpoint written by the tree before DEPS3, with legacy Keras 2 (``tf_keras``), and its model.

``make_fixture.py`` wrote ``checkpoint.*`` and ``weights.npz`` once; the tests only read them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential

import tensorflow as tf  # after tensorpotential

LEGACY_DIR = Path(__file__).resolve().parent
LEGACY_CHECKPOINT = str(LEGACY_DIR / "checkpoint")
MANIFEST = json.loads((LEGACY_DIR / "manifest.json").read_text())
LEGACY_OPTIONS = MANIFEST["optimizer"]
LEGACY_STEPS = MANIFEST["steps"]
LEGACY_EPOCH = MANIFEST["epoch"]
_MODEL_ARGUMENTS = {
    k: v for k, v in MANIFEST["model"].items() if k not in ("preset", "elements")
}


def small_tp(optimizer_options: dict | None = None) -> TensorPotential:
    """The model of the fixture (4,810 float32 parameters) with Adam; ``None`` options give the fixture's.

    Pass ``{}`` for an optimizer with default options. The weights are the random initial ones.
    """
    options = LEGACY_OPTIONS if optimizer_options is None else optimizer_options
    instructions = get_preset(MANIFEST["model"]["preset"])(
        element_map=MANIFEST["model"]["elements"], **_MODEL_ARGUMENTS
    ).get_instructions()
    return TensorPotential(
        instructions,
        param_dtype=tf.float32,
        fit_config={"optimizer": "Adam", "opt_params": options},
    )


def legacy_weights() -> list[np.ndarray]:
    """The model weights stored in the checkpoint, in ``variables_to_train`` order."""
    stored = np.load(LEGACY_DIR / "weights.npz")
    return [stored[f"arr_{i}"] for i in range(len(stored.files))]
