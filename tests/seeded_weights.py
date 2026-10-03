"""Seeded non-zero weights for tests of instructions and models.

Several layers start at zero (``InvariantLayerRMSNorm`` with ``init="zeros"``,
``LinMLPOut2ScalarTarget`` scales at ``1e-16``, ``TrainableShiftTarget`` shifts), so the output of a
freshly built instruction is trivial and says nothing about its forward pass. ``seed_trainable_variables``
overwrites every trainable floating variable with reproducible values that depend on the variable name,
not on the order in which the variables were created; constants (cutoffs, element maps, index tables) are
not trainable variables and are left alone.
"""

from __future__ import annotations

import zlib
from collections import Counter

import numpy as np
import tensorflow as tf

DEFAULT_SEED = 12345


def seeded_values(
    name: str, shape: tuple[int, ...], seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Reproducible non-zero values for one variable.

    Arrays with two or more axes get ``N(0, 1 / sqrt(shape[0]))`` (the scale of a normalised weight
    matrix); scalars and vectors (biases, scales, norms) get ``N(1, 0.1)``.

    Parameters
    ----------
    name : str
        Key of the variable; the stream of random numbers depends on ``(seed, name)`` only.
    shape : tuple of int
        Shape of the variable.
    seed : int
        Base seed.

    Returns
    -------
    numpy.ndarray
        Float64 values of the requested shape.
    """
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    if len(shape) >= 2:
        return rng.normal(0.0, 1.0 / np.sqrt(max(shape[0], 1)), size=shape)
    return rng.normal(1.0, 0.1, size=shape)


def seed_trainable_variables(
    module: tf.Module, seed: int = DEFAULT_SEED
) -> dict[str, np.ndarray]:
    """Assign seeded values to every trainable floating variable of ``module``.

    The module must be built. Variables that are not trainable (cutoffs, element maps, any
    ``trainable=False`` table) and variables that are not floating point are not touched. Two variables
    with the same name (``scale:0`` of two instructions) get different values, because the position among
    the variables of that name enters the key.

    Parameters
    ----------
    module : tf.Module
        An instruction, a model, or any module whose ``trainable_variables`` are the weights.
    seed : int
        Base seed; the same module and seed always give the same values.

    Returns
    -------
    dict of str to numpy.ndarray
        The values assigned, keyed by ``"<variable name>#<position among that name>"``.
    """
    assigned: dict[str, np.ndarray] = {}
    seen: Counter[str] = Counter()
    for var in module.trainable_variables:
        if not var.dtype.is_floating:
            continue
        key = f"{var.name}#{seen[var.name]}"
        seen[var.name] += 1
        value = seeded_values(key, tuple(var.shape), seed)
        var.assign(tf.cast(value, var.dtype))
        assigned[key] = value
    return assigned
