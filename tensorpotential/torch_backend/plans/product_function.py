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
op. The coupling metadata itself is :func:`product_coupling_meta_data`: ``functions.couplings.real_coupling_metainformation``
of the two operands' tables with the arguments of the TF constructor (legacy history format), as ``init_coupling`` calls it.

Layer: plans; imports numpy, the shared coupling generator, the errors and the pinned defaults of the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from tensorpotential.functions.couplings import real_coupling_metainformation
from tensorpotential.torch_backend.errors import ShapeError
from tensorpotential.torch_backend.plans.spbf_equivariant_ind import natural_parity
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


_LIMITS: Final = (
    "lmax_left",
    "lmax_right",
    "lmax_hist",
    "lmax_hist_left",
    "lmax_hist_right",
    "max_sum_l",
)


def _pairs(value: Any) -> tuple[tuple[Any, ...], ...] | None:
    return None if value is None else tuple(tuple(p) for p in value)


def _lists(pairs: tuple[tuple[Any, ...], ...] | None) -> list[list[Any]] | None:
    return None if pairs is None else [list(p) for p in pairs]


@dataclass(frozen=True)
class ProductFunctionOptions:
    """The constructor arguments of a ``ProductFunction`` that fix its coupling (names as in the yaml).

    Attributes
    ----------
    lmax
        Highest ``l`` of an operand entry that is coupled.
    Lmax
        Highest ``L`` of the output.
    is_left_right_equal
        Use the symmetric table (pairs with ``l2 > l1`` and swapped histories skipped, coefficients of swapped pairs
        merged); a null in the yaml means that both operands are the same instruction.
    lmax_left, lmax_right, lmax_hist, lmax_hist_left, lmax_hist_right, max_sum_l
        Extra limits of the coupling generator (``None`` in every shipped yaml).
    history_drop_list
        Extra ``(hist, L)`` rows to drop, in the legacy history format.
    keep_parity
        The ``(L, parity)`` pairs kept; ``None`` is natural parity, ``(L, (-1)**L)`` for ``L = 0 .. Lmax``.
    normalize
        Scale the real Clebsch-Gordan blocks by the generator's normalisation table.
    """

    lmax: int
    Lmax: int
    is_left_right_equal: bool
    lmax_left: int | None = None
    lmax_right: int | None = None
    lmax_hist: int | None = None
    lmax_hist_left: int | None = None
    lmax_hist_right: int | None = None
    history_drop_list: tuple[tuple[str, int], ...] | None = None
    max_sum_l: int | None = None
    keep_parity: tuple[tuple[int, int], ...] | None = None
    normalize: bool = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> ProductFunctionOptions:
        """Read the fields from the constructor arguments of a yaml; missing keys take the pinned defaults of the spec.

        A null ``is_left_right_equal`` compares the ``left`` and ``right`` references, as TF compares their names.
        """
        merged = {**DEFAULTS[COMPUTE + "ProductFunction"], **options}
        equal = merged["is_left_right_equal"]
        return cls(
            lmax=int(merged["lmax"]),
            Lmax=int(merged["Lmax"]),
            is_left_right_equal=bool(
                merged["left"] == merged["right"] if equal is None else equal
            ),
            **{key: merged[key] for key in _LIMITS},
            history_drop_list=_pairs(merged["history_drop_list"]),
            keep_parity=_pairs(merged["keep_parity"]),
            normalize=bool(merged["normalize"]),
        )


def product_coupling_meta_data(
    left_meta: pd.DataFrame, right_meta: pd.DataFrame, options: ProductFunctionOptions
) -> pd.DataFrame:
    """Return the ``coupling_meta_data`` of a ``ProductFunction``: one row per output function, in output order.

    Parameters
    ----------
    left_meta, right_meta
        The ``coupling_meta_data`` frames of the two operands (columns ``l, m, hist, parity, sum_of_ls``).
    options
        The constructor arguments that fix the coupling.
    """
    keep = (
        natural_parity(options.Lmax)
        if options.keep_parity is None
        else _lists(options.keep_parity)
    )
    return real_coupling_metainformation(
        A=left_meta,
        B=right_meta,
        lmax=options.lmax,
        lmax_A=options.lmax_left,
        lmax_B=options.lmax_right,
        lmax_hist=options.lmax_hist,
        lmax_hist_A=options.lmax_hist_left,
        lmax_hist_B=options.lmax_hist_right,
        Lmax=options.Lmax,
        is_A_B_equal=options.is_left_right_equal,
        history_drop_list=_lists(options.history_drop_list),
        max_sum_l=options.max_sum_l,
        keep_parity=keep,
        normalize=options.normalize,
        legacy_format=True,
    )


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
