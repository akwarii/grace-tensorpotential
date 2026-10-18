"""Coupling metadata of real spherical harmonics and of their product, for the tests of the plans.

Built with ``functions.couplings`` (TF-free) as ``SphericalHarmonic`` and ``ProductFunction`` build theirs, so the
plan tests get a real coupling without the TF stand-in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tensorpotential.functions.couplings import real_coupling_metainformation


def uncoupled(lmax: int) -> pd.DataFrame:
    """The metadata of the real harmonics up to ``lmax``, as ``SphericalHarmonic`` builds it (``l*l + l + m`` order)."""
    l = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
    m = np.concatenate([np.arange(-d, d + 1) for d in range(lmax + 1)])
    columns = {
        "l": l,
        "m": m,
        "hist": [""] * len(l),
        "parity": (-1) ** l,
        "sum_of_ls": l,
    }
    return pd.DataFrame(columns)


def coupled(lmax: int, Lmax: int) -> pd.DataFrame:
    """The metadata of the product of two sets of harmonics, as ``ProductFunction`` builds it."""
    harmonics = uncoupled(lmax)
    keep = [[l, 1 if l % 2 == 0 else -1] for l in range(Lmax + 1)]
    return real_coupling_metainformation(
        A=harmonics,
        B=harmonics,
        lmax=lmax,
        Lmax=Lmax,
        is_A_B_equal=False,
        keep_parity=keep,
        normalize=True,
        legacy_format=True,
    )
