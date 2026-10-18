"""Index tables and weight shapes of ``FCRight2Left``, in pure numpy.

TensorFlow's ``FCRight2Left.__init__`` (``instructions/compute.py``) groups the rows of the coupling metadata of its left
and right inputs by ``(l, parity, hist)``, groups sorted by that key as pandas' ``groupby`` sorts them, each holding the
positions of its rows in row order. One left weight is shared by the rows of a left group; every right group with the
same ``(l, parity)`` as a left group (``hist`` may differ) is mixed into that left group with a weight of its own:

.. code-block:: text

    out = left mixed by w_left[..., w_tile_left]          (or left itself when left_coefs is false)
    out[collect_to] += right[collect_from] mixed by w_right[..., w_tile_right]
    out *= norm_map                                       (when norm_out is set; norm_out_factor in TensorFlow)

``norm_map`` of a left row is ``1 / sqrt(1 + k)``, ``k`` the number of right groups mixed into its group. The plan also
gives the weight shapes ``w_left`` ``[n_out, n_left, w_shape_left]`` and ``w_right`` ``[n_out, n_right, w_shape_right]``,
each with a leading ``number_of_atom_types`` axis when that side depends on the central atom type, and the scalar
norms ``norm_left`` and ``norm_right``. The instruction's own coupling metadata is a copy of the left input's.

:meth:`FCRight2LeftWidths.from_options` reads them from the constructor arguments of a yaml.

Layer: plans; imports numpy, the errors and the pinned defaults of the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

_GroupKey = tuple[int, int, str]


@dataclass(frozen=True)
class FCRight2LeftWidths:
    """The widths and options of one ``FCRight2Left`` that fix its weight shapes and norms.

    Attributes
    ----------
    n_out
        Output channels (the ``n_out`` option, or the left input's when the option is ``None``).
    n_left, n_right
        Channels of the left and right input (their ``n_out``).
    left_coefs
        Whether the left input is mixed by a weight of its own.
    per_element
        ``is_central_atom_type_dependent`` of the left and right weight.
    number_of_atom_types
        Size of the leading element axis; required when either side of ``per_element`` is set.
    normalize
        Whether ``norm_left`` and ``norm_right`` are ``1 / sqrt(n_left)`` and ``1 / sqrt(n_right)`` (else 1).
    """

    n_out: int
    n_left: int
    n_right: int
    left_coefs: bool = True
    per_element: tuple[bool, bool] = (False, False)
    number_of_atom_types: int | None = None
    normalize: bool = True

    @classmethod
    def from_options(
        cls, options: Mapping[str, Any], n_left: int, n_right: int
    ) -> FCRight2LeftWidths:
        """Read the widths from the constructor arguments of a yaml and the operands' channel counts.

        Missing keys take the pinned defaults of the spec; ``n_out`` null means ``n_left``.

        Raises
        ------
        OptionValueError
            For an ``is_central_atom_type_dependent`` that is not null, a bool or two bools.
        """
        merged = {**DEFAULTS[COMPUTE + "FCRight2Left"], **options}
        n_out, n_types = merged["n_out"], merged["number_of_atom_types"]
        return cls(
            n_out=n_left if n_out is None else int(n_out),
            n_left=n_left,
            n_right=n_right,
            left_coefs=bool(merged["left_coefs"]),
            per_element=per_element_sides(merged["is_central_atom_type_dependent"]),
            number_of_atom_types=None if n_types is None else int(n_types),
            normalize=bool(merged["normalize"]),
        )


def per_element_sides(value: object) -> tuple[bool, bool]:
    """Return ``is_central_atom_type_dependent`` as ``(left, right)``: null is neither, a bool both, two bools each.

    Raises
    ------
    OptionValueError
        For any other value (TF: a ``ValueError``).
    """
    if value is None:
        return (False, False)
    if isinstance(value, bool):
        return (value, value)
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(v, bool) for v in value)
    ):
        return (value[0], value[1])
    msg = f"is_central_atom_type_dependent must be null, a bool or a list of two bools, got {value!r}"
    raise OptionValueError(msg)


@dataclass(frozen=True)
class FCRight2LeftPlan:
    """The gather, scatter and weight-tile tables of one ``FCRight2Left``, with its weight shapes.

    Attributes
    ----------
    w_tile_left
        ``[n_left_rows]`` int32: the left weight (group) of each left row.
    w_shape_left
        Number of left groups.
    collect_to, collect_from
        ``[n_pairs]`` int32: the left row each right row ``collect_from`` is added to.
    w_tile_right
        ``[n_pairs]`` int32: the right weight of each pair.
    w_shape_right
        Number of matching (left group, right group) pairs.
    norm_map
        ``[n_left_rows]`` float64: ``1 / sqrt(1 + k)`` of each left row.
    w_left_shape
        Shape of ``w_left``; ``None`` when ``left_coefs`` is false (no left weight).
    w_right_shape
        Shape of ``w_right``.
    norm_left
        Scalar factor of the mixed left input; ``None`` without a left weight.
    norm_right
        Scalar factor of the mixed right input.
    """

    w_tile_left: np.ndarray
    w_shape_left: int
    collect_to: np.ndarray
    collect_from: np.ndarray
    w_tile_right: np.ndarray
    w_shape_right: int
    norm_map: np.ndarray
    w_left_shape: tuple[int, ...] | None
    w_right_shape: tuple[int, ...]
    norm_left: float | None
    norm_right: float


def _groups(coupling_meta_data: pd.DataFrame) -> dict[_GroupKey, np.ndarray]:
    """Return the row positions of each ``(l, parity, hist)`` group, groups sorted by key, rows in order."""
    keys = list(
        zip(
            (int(v) for v in coupling_meta_data["l"]),
            (int(v) for v in coupling_meta_data["parity"]),
            (str(v) for v in coupling_meta_data["hist"]),
            strict=True,
        )
    )
    positions: dict[_GroupKey, list[int]] = {}
    for position, key in enumerate(keys):
        positions.setdefault(key, []).append(position)
    return {
        key: np.asarray(positions[key], dtype=np.int32) for key in sorted(positions)
    }


def _norm(n_in: int, normalize: bool) -> float:
    return 1 / n_in**0.5 if normalize else 1.0


def _weight_shapes(
    widths: FCRight2LeftWidths, w_shape_left: int, w_shape_right: int
) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    n_elements = widths.number_of_atom_types
    if any(widths.per_element) and n_elements is None:
        msg = "number_of_atom_types is required when is_central_atom_type_dependent is set"
        raise OptionValueError(msg)
    if not widths.left_coefs and widths.n_left != widths.n_out:
        msg = f"left_coefs false needs n_out equal to the left input's n_out, got {widths.n_out} and {widths.n_left}"
        raise OptionValueError(msg)
    left: tuple[int, ...] = (widths.n_out, widths.n_left, w_shape_left)
    right: tuple[int, ...] = (widths.n_out, widths.n_right, w_shape_right)
    if n_elements is not None and widths.per_element[0]:
        left = (n_elements, *left)
    if n_elements is not None and widths.per_element[1]:
        right = (n_elements, *right)
    return (left if widths.left_coefs else None), right


def fc_right2left_plan(
    left_meta: pd.DataFrame, right_meta: pd.DataFrame, widths: FCRight2LeftWidths
) -> FCRight2LeftPlan:
    """Return the tables and weight shapes of an ``FCRight2Left``.

    Parameters
    ----------
    left_meta, right_meta
        The ``coupling_meta_data`` frames of the left and right input, with the columns ``l``, ``parity`` and
        ``hist``.
    widths
        Channel counts and the options that fix the weight shapes.

    Raises
    ------
    ShapeError
        When no ``(l, parity)`` of the left input occurs in the right input (TensorFlow fails with a bare numpy
        error), or a matched left and right group differ in size.
    OptionValueError
        When ``widths`` asks for an element axis without ``number_of_atom_types``, or for no left weight with
        ``n_out`` different from the left input's.
    """
    left_groups = _groups(left_meta)
    right_groups = _groups(right_meta)
    w_tile_left = np.zeros(len(left_meta), dtype=np.int32)
    counts = np.ones(len(left_groups))
    collect_to, collect_from, w_tile_right = [], [], []
    for group, ((l_left, p_left, _), rows_left) in enumerate(left_groups.items()):
        w_tile_left[rows_left] = group
        for (l_right, p_right, _), rows_right in right_groups.items():
            if (l_right, p_right) != (l_left, p_left):
                continue
            if len(rows_right) != len(rows_left):
                msg = (
                    f"left group (l={l_left}, parity={p_left}) has {len(rows_left)} rows, the matching right group "
                    f"{len(rows_right)}; they must be equal"
                )
                raise ShapeError(msg)
            collect_to.append(rows_left)
            collect_from.append(rows_right)
            w_tile_right.append(
                np.full(len(rows_right), len(w_tile_right), dtype=np.int32)
            )
            counts[group] += 1
    if not collect_to:
        msg = "no (l, parity) of the left input occurs in the right input: nothing to collect"
        raise ShapeError(msg)
    w_left_shape, w_right_shape = _weight_shapes(
        widths, len(left_groups), len(w_tile_right)
    )
    return FCRight2LeftPlan(
        w_tile_left=w_tile_left,
        w_shape_left=len(left_groups),
        collect_to=np.concatenate(collect_to),
        collect_from=np.concatenate(collect_from),
        w_tile_right=np.concatenate(w_tile_right),
        w_shape_right=len(w_tile_right),
        norm_map=1 / counts[w_tile_left] ** 0.5,
        w_left_shape=w_left_shape,
        w_right_shape=w_right_shape,
        norm_left=_norm(widths.n_left, widths.normalize) if widths.left_coefs else None,
        norm_right=_norm(widths.n_right, widths.normalize),
    )
