"""Output layout, collection tables and weight shapes of ``FunctionReduceN``, in pure numpy.

TensorFlow's ``FunctionReduceN`` (``instructions/compute.py``, ``collect_functions`` in ``instructions/base.py``; spec
sheet ``FunctionReduceN.md``, sections 2 to 4) reduces several sources to one tensor:

.. code-block:: text

    out[a, k, row] = norm_map[row] * sum over sources s, and collected entries w of s with total_sum_ind[s][w] = row, of
                     norm_s * sum_n reducing_s[(mu_a,) k, n, w_l_tile[s][w]] * source_s[a, n, func_collect_ind[s][w]]

* The **output layout** has one row per ``(l, m)`` of every allowed ``(l, parity)`` with ``l <= max(ls_max)``, sorted by
  ``l``, then parity (``-1`` first), then ``m``; ``hist`` is empty and ``sum_of_ls = l``.
* Per source, the entries of its table are grouped by ``(parity, l, hist)`` in ascending key order (pandas
  ``groupby(...).indices``); the groups with ``l <= ls_max[s]`` and an allowed ``(l, parity)`` are kept.
  ``func_collect_ind`` lists their rows group after group, ``w_l_tile`` gives each collected entry the position of its
  group (``w_shape`` groups: one weight per group), and ``total_sum_ind`` the output row with its ``(l, m, parity)``.
* ``norm_map = 1 / sqrt(count)`` of the collected entries that land on a row (1 for none); the forward pass uses it
  only with ``out_norm`` (else the factor is 1). ``norm_s = scale / sqrt(n_in)`` with ``normalize``, else 1.
* The weight ``reducing_<source>`` is ``[n_out, n_in, w_shape]``, with a leading ``number_of_atom_types`` axis when
  ``is_central_atom_type_dependent``; ``n_in`` is the source's channel count.

``only_invar`` (every ``ls_max`` is 0) makes TensorFlow sum all collected entries into a single row, whatever the
layout says. Plans keep floats in float64; the cast to the parameter dtype belongs to the op.

Layer: plans; imports numpy, pandas and the errors only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError

GroupKey = tuple[int, int, str]
"""``(parity, l, hist)`` of a row of a coupling table."""


@dataclass(frozen=True)
class ReduceSource:
    """One instruction a ``FunctionReduceN`` collects from.

    Attributes
    ----------
    name
        Instruction name (the weight is ``reducing_<name>``).
    coupling_meta_data
        Its table, with the columns ``l, m, hist, parity``.
    n_out
        Its channel count, the ``n_in`` of the weight.
    """

    name: str
    coupling_meta_data: pd.DataFrame
    n_out: int


@dataclass(frozen=True)
class FunctionReduceNOptions:
    """The constructor arguments that fix the tables and weight shapes (names as in the yaml).

    Attributes
    ----------
    ls_max
        Highest ``l`` collected from each source, one per source (an int in the yaml is repeated: :func:`per_source`).
    allowed_l_p
        The ``(l, parity)`` pairs collected, and the layout of the output.
    n_out
        Output channels.
    is_central_atom_type_dependent, number_of_atom_types
        Per-element weights and their element count.
    normalize, scale
        ``norm_s = scale / sqrt(n_in)`` when ``normalize``, else 1.
    out_norm
        Whether the forward pass multiplies by ``norm_map``.
    """

    ls_max: tuple[int, ...]
    allowed_l_p: tuple[tuple[int, int], ...]
    n_out: int
    is_central_atom_type_dependent: bool = False
    number_of_atom_types: int | None = None
    normalize: bool = True
    scale: float = 1.0
    out_norm: bool = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> FunctionReduceNOptions:
        """Read the fields from the constructor arguments of a yaml (TF defaults for missing keys; others ignored).

        ``ls_max`` given as an int is repeated for every entry of ``instructions``.
        """
        n_atom_types = options.get("number_of_atom_types")
        return cls(
            ls_max=per_source(options["ls_max"], len(options["instructions"])),
            allowed_l_p=tuple((int(l), int(p)) for l, p in options["allowed_l_p"]),
            n_out=int(options["n_out"]),
            is_central_atom_type_dependent=bool(
                options.get("is_central_atom_type_dependent", False)
            ),
            number_of_atom_types=None if n_atom_types is None else int(n_atom_types),
            normalize=bool(options.get("normalize", True)),
            scale=float(options.get("scale", 1.0)),
            out_norm=bool(options.get("out_norm", False)),
        )


@dataclass(frozen=True)
class CollectorPlan:
    """What a ``FunctionReduceN`` takes from one source.

    Attributes
    ----------
    name
        The source.
    func_collect_ind
        ``[n_collected]`` int64: the columns of the source that are collected (TensorFlow's numpy index array).
    w_l_tile
        ``[n_collected]`` int32: the weight group of each collected entry.
    w_shape
        Number of weight groups.
    total_sum_ind
        ``[n_collected, 1]`` int32: the output row of each collected entry.
    weight_shape
        Shape of ``reducing_<name>``.
    norm
        ``norm_<name>``, the factor of the contribution.
    """

    name: str
    func_collect_ind: np.ndarray
    w_l_tile: np.ndarray
    w_shape: int
    total_sum_ind: np.ndarray
    weight_shape: tuple[int, ...]
    norm: float


@dataclass(frozen=True)
class FunctionReduceNPlan:
    """The tables of one ``FunctionReduceN``.

    Attributes
    ----------
    coupling_meta_data
        The output layout (``l, m, hist, parity, sum_of_ls``), the table the next instructions read.
    lmax
        ``max(ls_max)``.
    only_invar
        Every ``ls_max`` is 0: all collected entries are summed into one row.
    collectors
        One per source, in the order of the sources.
    norm_map
        ``[n_rows]`` float64: ``1 / sqrt`` of the number of collected entries per output row (1 for none).
    out_norm
        Whether the forward pass applies ``norm_map``; TensorFlow's runtime constant is then ``norm_map`` as
        ``[n_rows, 1, 1]``, else the scalar 1 (:attr:`runtime_norm_map`).
    """

    coupling_meta_data: pd.DataFrame
    lmax: int
    only_invar: bool
    collectors: tuple[CollectorPlan, ...]
    norm_map: np.ndarray
    out_norm: bool

    @property
    def runtime_norm_map(self) -> np.ndarray:
        """The factor of the output: ``norm_map`` as ``[n_rows, 1, 1]`` with ``out_norm``, else the 0-d array 1."""
        return self.norm_map.reshape(-1, 1, 1) if self.out_norm else np.array(1.0)


def per_source(ls_max: int | Sequence[int], n_sources: int) -> tuple[int, ...]:
    """``ls_max`` as one value per source: an int is repeated, a list is kept."""
    if isinstance(ls_max, int):
        return (ls_max,) * n_sources
    return tuple(int(v) for v in ls_max)


def output_layout(lmax: int, allowed_l_p: Sequence[Sequence[int]]) -> pd.DataFrame:
    """Return the output table: a row ``(l, m, "", parity, l)`` per ``m`` of each allowed ``(l, parity)`` with ``l <= lmax``.

    Sorted by ``l``, parity and ``m``, with a fresh index, as TensorFlow's ``sort_values(["l", "parity", "hist", "m"])``.
    """
    allowed = {(int(l), int(p)) for l, p in allowed_l_p}
    rows = [
        (l, m, "", p, l)
        for l in range(lmax + 1)
        for p in (-1, 1)
        if (l, p) in allowed
        for m in range(-l, l + 1)
    ]
    return pd.DataFrame(
        rows, columns=pd.Index(["l", "m", "hist", "parity", "sum_of_ls"])
    )


def _groups(meta: pd.DataFrame) -> dict[GroupKey, list[int]]:
    keys = zip(meta["parity"], meta["l"], meta["hist"], strict=True)
    groups: dict[GroupKey, list[int]] = {}
    for row, (p, l, hist) in enumerate(keys):
        groups.setdefault((int(p), int(l), str(hist)), []).append(row)
    return dict(sorted(groups.items()))


def _check(sources: Sequence[ReduceSource], options: FunctionReduceNOptions) -> None:
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        msg = f"the sources of a FunctionReduceN must have distinct names, got {names}"
        raise OptionValueError(msg)
    if len(options.ls_max) != len(sources):
        msg = f"ls_max has {len(options.ls_max)} values for {len(sources)} sources {names}; give one per source"
        raise OptionValueError(msg)
    if options.is_central_atom_type_dependent and options.number_of_atom_types is None:
        msg = "is_central_atom_type_dependent needs number_of_atom_types"
        raise OptionValueError(msg)


def _collector(
    source: ReduceSource,
    max_l: int,
    options: FunctionReduceNOptions,
    out_rows: dict[tuple[int, int, int], int],
) -> CollectorPlan:
    allowed = {(int(l), int(p)) for l, p in options.allowed_l_p}
    kept = [
        rows
        for (p, l, _), rows in _groups(source.coupling_meta_data).items()
        if l <= max_l and (l, p) in allowed
    ]
    if not kept:
        msg = (
            f"source {source.name!r} has no entry with l <= {max_l} and (l, parity) in {sorted(allowed)}; "
            "TensorFlow fails on such a FunctionReduceN"
        )
        raise ShapeError(msg)
    func_collect_ind = np.concatenate([
        np.asarray(rows, dtype=np.int64) for rows in kept
    ])
    meta = source.coupling_meta_data
    keys = zip(
        meta["l"].iloc[func_collect_ind],
        meta["m"].iloc[func_collect_ind],
        meta["parity"].iloc[func_collect_ind],
        strict=True,
    )
    total = [out_rows[(int(l), int(m), int(p))] for l, m, p in keys]
    n_in = source.n_out
    shape: tuple[int, ...] = (options.n_out, n_in, len(kept))
    if options.is_central_atom_type_dependent:
        shape = (int(options.number_of_atom_types or 0), *shape)
    return CollectorPlan(
        name=source.name,
        func_collect_ind=func_collect_ind,
        w_l_tile=np.repeat(
            np.arange(len(kept), dtype=np.int32), [len(rows) for rows in kept]
        ),
        w_shape=len(kept),
        total_sum_ind=np.asarray(total, dtype=np.int32).reshape(-1, 1),
        weight_shape=shape,
        norm=options.scale / n_in**0.5 if options.normalize else 1.0,
    )


def function_reduce_n_plan(
    sources: Sequence[ReduceSource], options: FunctionReduceNOptions
) -> FunctionReduceNPlan:
    """Return the output layout, the collection tables and the weight shapes of a ``FunctionReduceN``.

    Parameters
    ----------
    sources
        The collected instructions, in the order of the yaml's ``instructions`` list.
    options
        The constructor arguments that fix the tables.

    Raises
    ------
    OptionValueError
        When two sources share a name, ``ls_max`` does not have one value per source, or per-element weights lack
        ``number_of_atom_types`` (TensorFlow: an ``AssertionError``).
    ShapeError
        When a source has no entry to collect (TensorFlow: a bare numpy error).
    """
    _check(sources, options)
    lmax = max(options.ls_max)
    layout = output_layout(lmax, options.allowed_l_p)
    out_rows = {
        (int(l), int(m), int(p)): row
        for row, (l, m, p) in enumerate(
            zip(layout["l"], layout["m"], layout["parity"], strict=True)
        )
    }
    collectors = tuple(
        _collector(source, max_l, options, out_rows)
        for source, max_l in zip(sources, options.ls_max, strict=True)
    )
    counts = np.zeros(len(layout))
    for collector in collectors:
        np.add.at(counts, collector.total_sum_ind.ravel(), 1)
    counts[counts == 0] = 1
    return FunctionReduceNPlan(
        coupling_meta_data=layout,
        lmax=lmax,
        only_invar=all(v == 0 for v in options.ls_max),
        collectors=collectors,
        norm_map=1 / counts**0.5,
        out_norm=options.out_norm,
    )
