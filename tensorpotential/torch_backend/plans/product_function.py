"""Index tables of the Clebsch-Gordan product of ``ProductFunction``, in pure numpy.

TensorFlow's ``ProductFunction.init_coupling`` (``instructions/compute.py``) reads three per-row lists of its
``coupling_meta_data`` (one row per coupled output function ``(l, m, parity, hist)``): ``left_inds`` and ``right_inds``,
the angular columns of the left and right inputs that the row multiplies, and ``cg_list``, the Clebsch-Gordan
coefficient of each pair. The tables flatten them in row order:

.. code-block:: text

    out[..., f] = sum over pairs p with m_sum_ind[p] = f of
                  cg[p] * left[..., left_ind[p]] * right[..., right_ind[p]]

``nfunc``, the number of output functions, is ``max(m_sum_ind) + 1`` as in TensorFlow. The plan keeps ``cg`` flat in
float64; the layout (``[1, 1, n]``, or ``[n, 1, 1]`` with ``lm_first``) and the cast to the parameter dtype belong to the
op. The coupling metadata itself comes from ``functions.couplings.real_coupling_metainformation`` and is an input here.

Layer: plans; imports numpy and the errors only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tensorpotential.torch_backend.errors import ShapeError

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ProductFunctionPlan:
    """The gather and sum tables of one ``ProductFunction``.

    Attributes
    ----------
    left_ind, right_ind
        ``[n_pairs]`` int32: the angular column of the left and right input of each pair.
    m_sum_ind
        ``[n_pairs]`` int32: the output function each pair is summed into (the row of the coupling metadata).
    cg
        ``[n_pairs]`` float64: the Clebsch-Gordan coefficient of each pair.
    nfunc
        Number of output functions, ``max(m_sum_ind) + 1``.
    """

    left_ind: np.ndarray
    right_ind: np.ndarray
    m_sum_ind: np.ndarray
    cg: np.ndarray
    nfunc: int


def _flat(rows: list, dtype: type) -> np.ndarray:
    return np.concatenate([np.asarray(row, dtype=dtype).ravel() for row in rows])


def product_function_plan(coupling_meta_data: pd.DataFrame) -> ProductFunctionPlan:
    """Return the tables of the ``ProductFunction`` whose own coupling metadata is ``coupling_meta_data``.

    Parameters
    ----------
    coupling_meta_data
        The ``coupling_meta_data`` frame of the instruction (not of its inputs), with the columns ``left_inds``,
        ``right_inds`` and ``cg_list``.

    Raises
    ------
    ShapeError
        When the frame has no row (TensorFlow: "No coupling channels found"), no pair at all, or a row whose three
        lists differ in length.
    """
    lefts = list(coupling_meta_data["left_inds"])
    rights = list(coupling_meta_data["right_inds"])
    cgs = list(coupling_meta_data["cg_list"])
    if not cgs:
        msg = "the coupling metadata has no row: no coupling channel between the left and right input"
        raise ShapeError(msg)
    sizes = [np.size(cg) for cg in cgs]
    for row, (left, right, size) in enumerate(zip(lefts, rights, sizes, strict=True)):
        if not np.size(left) == np.size(right) == size:
            msg = (
                f"row {row} of the coupling metadata has {np.size(left)} left, {np.size(right)} right indices "
                f"and {size} coefficients; they must be equal"
            )
            raise ShapeError(msg)
    if not sum(sizes):
        msg = "the coupling metadata has no Clebsch-Gordan pair"
        raise ShapeError(msg)
    m_sum_ind = np.repeat(np.arange(len(cgs), dtype=np.int32), sizes)
    return ProductFunctionPlan(
        left_ind=_flat(lefts, np.int32),
        right_ind=_flat(rights, np.int32),
        m_sum_ind=m_sum_ind,
        cg=_flat(cgs, np.float64),
        nfunc=int(m_sum_ind.max()) + 1,
    )
