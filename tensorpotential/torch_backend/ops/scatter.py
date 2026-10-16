"""Scatter sums of the twins: the torch counterpart of ``tf.math.unsorted_segment_sum``.

TF sums bond quantities onto atoms and atomic quantities onto structures with ``unsorted_segment_sum`` (for example
the forces and the energy in ``tpmodel.py:332-345``); :func:`segment_sum` does the same with ``index_add_``. An empty
segment (an atom without bonds, a structure without atoms) gets zeros, and no bond or no atom at all gives a
tensor of zeros or an empty one, as in TF.

The function reads no tensor value on the host (no ``.item()``, no ``max()`` of the indices), so it never
synchronises a device: an index outside ``[0, num_segments)`` raises the error of torch on the CPU and a device
assertion on a GPU. Indices are checked once, where they are made (the neighbour-list contract of NBR1).

Layer: ops; imports torch and the errors of the backend only.
"""

from __future__ import annotations

import torch

from tensorpotential.torch_backend.errors import DtypeError, ShapeError

INDEX_DTYPES = (torch.int32, torch.int64)
"""The integer dtypes ``segment_ids`` may have (TF's index tensors are int32, torch's own are int64)."""


def segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum the rows of ``data`` that share a segment index.

    ``out[s] = sum(data[k] for k where segment_ids[k] == s)``, as ``numpy.add.at(out, segment_ids, data)`` on zeros.

    Parameters
    ----------
    data
        ``[n, ...]``: one row per item (bond or atom); any floating dtype, kept in the result.
    segment_ids
        ``[n]``: the segment of each row, ``int32`` or ``int64``, each in ``[0, num_segments)``; any order.
    num_segments
        Number of segments (atoms or structures), at least 0; segments that no row names are zero.

    Returns
    -------
    torch.Tensor
        ``[num_segments, ...]``, with the dtype and device of ``data``; differentiable with respect to ``data``.

    Raises
    ------
    ShapeError
        When the shapes disagree or ``num_segments`` is negative.
    DtypeError
        When ``segment_ids`` is not an ``int32`` or ``int64`` tensor.
    """
    if segment_ids.dtype not in INDEX_DTYPES:
        msg = f"segment_ids must be int32 or int64, got {segment_ids.dtype}"
        raise DtypeError(msg)
    if (
        segment_ids.dim() != 1
        or data.dim() == 0
        or segment_ids.shape[0] != data.shape[0]
    ):
        msg = (
            f"segment_ids must be one index per row of data: shapes {tuple(segment_ids.shape)} "
            f"and {tuple(data.shape)}"
        )
        raise ShapeError(msg)
    if num_segments < 0:
        msg = f"num_segments must be at least 0, got {num_segments}"
        raise ShapeError(msg)
    out = torch.zeros(
        (num_segments, *data.shape[1:]), dtype=data.dtype, device=data.device
    )
    return out.index_add_(0, segment_ids, data)
