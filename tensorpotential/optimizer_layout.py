"""Whether the optimizer state of a checkpoint fits the optimizer that would restore it.

A checkpoint written by a different Keras line (legacy Keras 2) stores the optimizer variables in
another order and number: the same ``_variables/<n>`` keys, other shapes behind them. ``tf.train.Checkpoint``
does not refuse such a restore: it raises on a shape mismatch, or, when the shapes happen to agree, assigns
every slot to the wrong variable. :func:`optimizer_state_fits` compares the saved layout with the one of a
fresh optimizer of the same configuration, so that the caller can restore the model weights only and start
the optimizer from scratch with a warning.
"""

from __future__ import annotations

import re
from typing import Any

from tensorpotential import _tf_options  # noqa: F401
import tensorflow as tf

OPTIMIZER_PREFIX = "optimizer/"
SLOT_KEY = re.compile(r"^optimizer/_variables/(\d+)/\.ATTRIBUTES/VARIABLE_VALUE$")

Layout = dict[int, tuple[int, ...]]


class LegacyOptimizerStateWarning(UserWarning):
    """The optimizer state of a checkpoint was not restored because its layout does not fit."""


def saved_optimizer_layout(checkpoint_name: str) -> Layout | None:
    """Shape of every optimizer slot in the checkpoint, by position, or ``None`` without optimizer state.

    Parameters
    ----------
    checkpoint_name : str
        Checkpoint prefix, as given to ``tf.train.Checkpoint.read``.

    Returns
    -------
    dict of int to tuple of int, or None
        ``{n: shape}`` for the keys ``optimizer/_variables/<n>``; ``None`` when no key starts with
        ``optimizer/``. The iteration counter and the learning rate are stored under their own keys.
    """
    entries = tf.train.list_variables(checkpoint_name)
    if not any(key.startswith(OPTIMIZER_PREFIX) for key, _ in entries):
        return None
    layout: Layout = {}
    for key, shape in entries:
        match = SLOT_KEY.match(key)
        if match:
            layout[int(match.group(1))] = tuple(int(d) for d in shape)
    return layout


def expected_optimizer_layout(optimizer: Any, variables: list[tf.Variable]) -> Layout:
    """Slot layout that a fresh optimizer of the same configuration saves once it is built.

    A probe optimizer is built for ``variables`` and discarded; ``optimizer`` itself is not touched.
    """
    probe = type(optimizer)(**optimizer.get_config())
    probe.build(variables)
    own_keys = {
        id(probe.iterations),
        id(probe.learning_rate),
    }  # stored under their own checkpoint keys
    return {
        i: tuple(int(d) for d in v.shape)
        for i, v in enumerate(probe.variables)
        if id(v) not in own_keys
    }


def optimizer_state_fits(
    checkpoint_name: str, optimizer: Any, variables: list[tf.Variable]
) -> bool:
    """Whether the checkpoint holds exactly the slots, in the order, that ``optimizer`` would restore.

    ``True`` when the checkpoint has no optimizer state at all (nothing can be restored wrongly).
    """
    saved = saved_optimizer_layout(checkpoint_name)
    if saved is None:
        return True
    return saved == expected_optimizer_layout(optimizer, variables)
