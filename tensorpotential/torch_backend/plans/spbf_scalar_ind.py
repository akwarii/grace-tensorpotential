"""Tables of ``SingleParticleBasisFunctionScalarInd``, in pure numpy.

TensorFlow's ``SingleParticleBasisFunctionScalarInd.__init__`` (``instructions/compute.py``; spec sheet
``SingleParticleBasisFunctionScalarInd.md``, section 2) derives from its options and inputs:

* ``lmax``: the option, or the degree of the angular input when it is null; the radial input must have it too.
* ``slice_angular``: ``(lmax + 1)**2`` leading columns of the angular input when its degree is higher, else ``None``.
* ``coupling_meta_data``: the table of the angular input, restricted to ``l <= lmax`` when it is cut. The product
  instructions build their own tables from it.
* ``l_tile``: the degree ``l`` of each of the ``(lmax + 1)**2`` columns (the radial function's ``l_tile``), which spreads an
  ``l``-dependent species weight over ``m``.
* ``inv_avg_n_neigh``: ``1 / avg_n_neigh`` for a float; for a dict ``element index -> mean neighbour count`` an array
  ``[len(dict), 1]`` of ``1 / v``, with 1 where ``v <= 0`` (a per-central-element factor).

The values stay in float64; the cast to the parameter dtype belongs to the op.

Layer: plans; imports numpy and the errors only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tensorpotential.torch_backend.errors import OptionValueError

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ScalarIndPlan:
    """The tables of one ``SingleParticleBasisFunctionScalarInd``.

    Attributes
    ----------
    lmax
        Highest degree used.
    slice_angular
        Number of leading angular columns used when the angular input has a higher degree, else ``None``.
    coupling_meta_data
        One row per output column (``l, m, hist, parity, sum_of_ls``), in output order.
    l_tile
        ``[(lmax + 1)**2]`` int64: the degree of each output column.
    inv_avg_n_neigh
        ``1 / avg_n_neigh`` as a 0-d float64 array, or ``[n_elements, 1]`` per central element.
    """

    lmax: int
    slice_angular: int | None
    coupling_meta_data: pd.DataFrame
    l_tile: np.ndarray
    inv_avg_n_neigh: np.ndarray

    @property
    def per_element_average(self) -> bool:
        """Whether the neighbour sum is scaled by a factor of the central atom's element."""
        return self.inv_avg_n_neigh.ndim == 2


def inverse_avg_n_neigh(avg_n_neigh: float | Mapping[int, float]) -> np.ndarray:
    """Return ``1 / avg_n_neigh``, 0-d for a float, ``[len(dict), 1]`` for a dict (1 where a count is ``<= 0``).

    Raises
    ------
    OptionValueError
        For any other type (TensorFlow: ``TypeError``; an ``int`` too, as in TF), or dict keys that are not
        ``0 .. len(dict) - 1``.
    """
    if isinstance(avg_n_neigh, float):
        return np.asarray(1.0 / avg_n_neigh, dtype=np.float64)
    if isinstance(avg_n_neigh, Mapping):
        if sorted(avg_n_neigh) != list(range(len(avg_n_neigh))):
            msg = f"the keys of a per-element avg_n_neigh must be the element indices 0 .. {len(avg_n_neigh) - 1}, got {sorted(avg_n_neigh)}"
            raise OptionValueError(msg)
        inv = np.zeros((len(avg_n_neigh), 1))
        for key, value in avg_n_neigh.items():
            inv[key] = 1.0 / (value if value > 0 else 1.0)
        return inv
    msg = f"avg_n_neigh must be a float or a dict element index -> float, got {avg_n_neigh!r} ({type(avg_n_neigh).__name__})"
    raise OptionValueError(msg)


def spbf_scalar_ind_plan(
    angular_meta: pd.DataFrame,
    *,
    lmax: int | None,
    radial_lmax: int,
    avg_n_neigh: float | Mapping[int, float],
) -> ScalarIndPlan:
    """Return the tables of a ``SingleParticleBasisFunctionScalarInd``.

    Parameters
    ----------
    angular_meta
        The ``coupling_meta_data`` of the angular input (a ``SphericalHarmonic``), one row per column.
    lmax
        The ``lmax`` option; ``None`` means the degree of the angular input.
    radial_lmax
        ``lmax`` of the radial input, which has to equal ``lmax`` (TF asserts it).
    avg_n_neigh
        The option of the same name.

    Raises
    ------
    OptionValueError
        When ``lmax`` exceeds the degree of the angular input, the radial input has another degree, or
        ``avg_n_neigh`` is not a float or a dict of element indices (TF: ``AssertionError`` or ``TypeError``).
    """
    angular_lmax = int(angular_meta["l"].max())
    if lmax is None:
        lmax = angular_lmax
    if lmax > angular_lmax:
        msg = f"lmax={lmax} is larger than the degree of the angular input ({angular_lmax})"
        raise OptionValueError(msg)
    if radial_lmax != lmax:
        msg = f"the radial part (lmax={radial_lmax}) and the instruction (lmax={lmax}) do not match"
        raise OptionValueError(msg)
    cut = angular_lmax > lmax
    degrees = np.arange(lmax + 1)
    return ScalarIndPlan(
        lmax=lmax,
        slice_angular=(lmax + 1) ** 2 if cut else None,
        coupling_meta_data=angular_meta.query(f"l <= {lmax}") if cut else angular_meta,
        l_tile=np.repeat(degrees, 2 * degrees + 1).astype(np.int64),
        inv_avg_n_neigh=inverse_avg_n_neigh(avg_n_neigh),
    )
