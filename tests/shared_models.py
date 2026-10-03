"""Models that several tests would otherwise build identically.

A model built here is shared by tests that only read it (a calculator wrapped
around it, a forward pass, ``train_function``); a test that saves, trains or
edits a model builds its own. Every model is fingerprinted when it is built and
compared when the session of the sharing tests ends, so a test that changes a
shared model makes the run fail instead of silently changing its neighbours.
"""

from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf

from tensorpotential import TPModel
from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25


def weights_fingerprint(model: TPModel) -> str:
    """Hash of every variable of ``model``: name, dtype, shape and values.

    Parameters
    ----------
    model : TPModel
        A built model.

    Returns
    -------
    str
        Hex digest; equal for equal weights, different if any value changes.
    """
    digest = hashlib.sha256()
    for var in sorted(model.variables, key=lambda v: v.name):
        value = var.numpy()
        digest.update(f"{var.name}|{value.dtype}|{value.shape}".encode())
        # a string variable is an object array: its bytes would be pointers, not content
        digest.update(
            repr(value.tolist()).encode() if value.dtype == object else value.tobytes()
        )
    return digest.hexdigest()


class CuTwoLayerModels:
    """``GRACE_2LAYER_v2_25`` on Cu (``rcut=6``, float64, seed 7), one per ``dense_nbr``.

    Built on first use and decorated for the calculator
    (``decorate_compute_function``), exactly as the tests did locally.
    """

    SEED = 7

    def __init__(self) -> None:
        self._models: dict[bool, TPModel] = {}
        self._fingerprints: dict[bool, str] = {}

    def get(self, dense: bool) -> TPModel:
        """Return the shared model with ``dense_nbr=dense``, building it once."""
        if dense not in self._models:
            tf.random.set_seed(self.SEED)
            np.random.seed(self.SEED)
            instructions = GRACE_2LAYER_v2_25(
                element_map={"Cu": 0}, rcut=6.0, dense_nbr=dense
            ).get_instructions()
            model = TPModel(instructions)
            model.build(tf.float64)
            model.decorate_compute_function(input_signature_float_dtype=tf.float64)
            self._models[dense] = model
            self._fingerprints[dense] = weights_fingerprint(model)
        return self._models[dense]

    def changed(self) -> list[bool]:
        """The ``dense_nbr`` values whose shared model no longer has its built weights."""
        return [
            dense
            for dense, model in self._models.items()
            if weights_fingerprint(model) != self._fingerprints[dense]
        ]
