"""Coupling and index tables of ``SingleParticleBasisFunctionEquivariantInd``, in pure numpy.

TensorFlow's ``SingleParticleBasisFunctionEquivariantInd.__init__`` (``instructions/compute.py``; spec sheet
``SingleParticleBasisFunctionEquivariantInd.md``, section 2) couples the radial-angular bond product (columns of the
angular input) with the equivariant indicator of the neighbour (columns of the indicator) to functions of definite
``(L, M, parity)``:

.. code-block:: text

    out[a, n, f] = (1 / avg_n_neigh) * sum over pairs k with m_sum_ind[k] = f of
                   cg[k] * sum_{bonds j of atom a} R[j, n, lr_inds[k, 0]] Y[j, lr_inds[k, 0]] I[j_atom, n, lr_inds[k, 1]]

The coupling table is ``functions.couplings.real_coupling_metainformation`` of the two input tables, called with
the arguments of the TF constructor (the shared TF-free generator, decision D1); the index tables flatten its
per-row lists in row order. ``cg`` stays flat in float64: the layout ``[n_cg, 1, 1]`` and the cast to the parameter
dtype belong to the op. ``keep_parity = None`` means natural parity up to ``Lmax``, as in TensorFlow.

Layer: plans; imports numpy, pandas, the shared coupling generator and the errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tensorpotential.functions.couplings import real_coupling_metainformation
from tensorpotential.torch_backend.errors import OptionValueError, ShapeError

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class EquivariantIndOptions:
    """The constructor arguments that fix the tables (names as in the yaml).

    Attributes
    ----------
    lmax
        Highest degree of the angular part and of the indicator.
    Lmax
        Highest degree of the output.
    keep_parity
        The ``(L, parity)`` pairs kept; ``None`` is natural parity, ``(L, (-1)**L)`` for ``L = 0 .. Lmax``.
    history_drop_list, l_max_ind, max_sum_l
        Extra limits of the coupling generator (``None`` in every shipped yaml).
    normalize
        Scale the real Clebsch-Gordan blocks by the generator's normalisation table.
    avg_n_neigh
        Average number of neighbours; the neighbour sum is multiplied by its inverse.
    """

    lmax: int
    Lmax: int
    keep_parity: tuple[tuple[int, int], ...] | None = None
    history_drop_list: tuple[tuple[str, int], ...] | None = None
    l_max_ind: int | None = None
    max_sum_l: int | None = None
    normalize: bool = False
    avg_n_neigh: float = 1.0

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> EquivariantIndOptions:
        """Read the fields from the constructor arguments of a yaml (TF defaults for missing keys; others ignored)."""

        def pairs(key: str) -> tuple[tuple[Any, ...], ...] | None:
            value = options.get(key)
            return None if value is None else tuple(tuple(p) for p in value)

        return cls(
            lmax=int(options["lmax"]),
            Lmax=int(options["Lmax"]),
            keep_parity=pairs("keep_parity"),
            history_drop_list=pairs("history_drop_list"),
            l_max_ind=options.get("l_max_ind"),
            max_sum_l=options.get("max_sum_l"),
            normalize=bool(options.get("normalize", False)),
            avg_n_neigh=float(options.get("avg_n_neigh", 1.0)),
        )


@dataclass(frozen=True)
class EquivariantIndPlan:
    """The tables of one ``SingleParticleBasisFunctionEquivariantInd``.

    Attributes
    ----------
    coupling_meta_data
        One row per output function (``l, m, hist, left_inds, right_inds, l1, l2, parity, sum_of_ls, cg_list``),
        in the order of the output channels; the table the next instruction (``FunctionReduceN``) collects from.
    lr_inds
        ``[n_cg, 2]`` int32: the angular column (0) and the indicator column (1) of each Clebsch-Gordan term.
    m_sum_ind
        ``[n_cg]`` int32: the output function each term is summed into.
    cg
        ``[n_cg]`` float64: the coefficient of each term.
    nfunc
        Number of output functions, ``max(m_sum_ind) + 1``.
    inv_avg_n_neigh
        ``1 / avg_n_neigh``.
    slice_angular
        ``(lmax + 1)**2`` when the angular input has a higher degree than ``lmax`` (its columns are cut), else ``None``.
    """

    coupling_meta_data: pd.DataFrame
    lr_inds: np.ndarray
    m_sum_ind: np.ndarray
    cg: np.ndarray
    nfunc: int
    inv_avg_n_neigh: float
    slice_angular: int | None


def natural_parity(Lmax: int) -> list[list[int]]:
    """``[[L, (-1)**L] for L in 0 .. Lmax]``, the default ``keep_parity``."""
    return [[L, 1 if L % 2 == 0 else -1] for L in range(Lmax + 1)]


def _slice_angular(angular_lmax: int, radial_lmax: int, lmax: int) -> int | None:
    if angular_lmax > lmax:
        if radial_lmax != lmax:
            msg = f"the radial function is prepared for lmax={radial_lmax}, while the instruction has lmax={lmax}"
            raise OptionValueError(msg)
        return (lmax + 1) ** 2
    if radial_lmax != angular_lmax:
        msg = f"the radial part (lmax={radial_lmax}) and the angular part (lmax={angular_lmax}) do not match"
        raise OptionValueError(msg)
    return None


def spbf_equivariant_ind_plan(
    angular_meta: pd.DataFrame,
    indicator_meta: pd.DataFrame,
    options: EquivariantIndOptions,
    *,
    radial_lmax: int,
) -> EquivariantIndPlan:
    """Return the tables of a ``SingleParticleBasisFunctionEquivariantInd``.

    Parameters
    ----------
    angular_meta, indicator_meta
        The ``coupling_meta_data`` frames of the angular input (a ``SphericalHarmonic``) and of the indicator (a
        ``FunctionReduceN``), with the columns ``l, m, hist, parity, sum_of_ls``.
    options
        The constructor arguments that fix the tables.
    radial_lmax
        ``lmax`` of the radial input, which has to match (TensorFlow asserts it).

    Raises
    ------
    OptionValueError
        When the degrees of the radial and angular inputs do not fit ``lmax`` (TensorFlow: an ``AssertionError``).
    ShapeError
        When no coupling survives the limits (TensorFlow fails with a bare numpy error).
    """
    angular_lmax = int(angular_meta["l"].max())
    slice_angular = _slice_angular(angular_lmax, radial_lmax, options.lmax)
    keep = (
        natural_parity(options.Lmax)
        if options.keep_parity is None
        else [list(p) for p in options.keep_parity]
    )
    drop = (
        None
        if options.history_drop_list is None
        else [list(d) for d in options.history_drop_list]
    )
    meta = real_coupling_metainformation(
        A=angular_meta,
        B=indicator_meta,
        lmax=options.lmax,
        lmax_B=options.l_max_ind,
        Lmax=options.Lmax,
        history_drop_list=drop,
        max_sum_l=options.max_sum_l,
        keep_parity=keep,
        normalize=options.normalize,
        optimize_ms_comb=False,
    )
    cgs = [np.asarray(cg, dtype=np.float64).ravel() for cg in meta["cg_list"]]
    if not cgs:
        msg = (
            f"no coupling of the angular input (lmax={angular_lmax}) and the indicator survives lmax={options.lmax}, "
            f"Lmax={options.Lmax} and keep_parity={keep}"
        )
        raise ShapeError(msg)
    left = np.concatenate([np.asarray(i).ravel() for i in meta["left_inds"]])
    right = np.concatenate([np.asarray(i).ravel() for i in meta["right_inds"]])
    m_sum_ind = np.repeat(np.arange(len(cgs), dtype=np.int32), [len(cg) for cg in cgs])
    return EquivariantIndPlan(
        coupling_meta_data=meta,
        lr_inds=np.stack([left, right], axis=1).astype(np.int32),
        m_sum_ind=m_sum_ind,
        cg=np.concatenate(cgs),
        nfunc=int(m_sum_ind.max()) + 1,
        inv_avg_n_neigh=1.0 / options.avg_n_neigh,
        slice_angular=slice_angular,
    )
