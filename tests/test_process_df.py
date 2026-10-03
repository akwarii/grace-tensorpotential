import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ase import Atoms

from tensorpotential.data.process_df import (
    ASE_ATOMS,
    COMP_DICT,
    COMP_TUPLE,
    E_CHULL_DIST_PER_ATOM,
    NUMBER_OF_ATOMS,
    compute_compositions,
    compute_convexhull_dist,
)
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

log = logging.getLogger()

prefix = Path(__file__).parent.resolve()


def test_compute_convex_hull():
    fname = str(prefix / "data" / "MoNbTaW_test50.pkl.gz")
    df = pd.read_pickle(fname)
    assert E_CHULL_DIST_PER_ATOM not in df.columns
    compute_convexhull_dist(df, verbose=True)
    assert E_CHULL_DIST_PER_ATOM in df.columns
    echmin = df[E_CHULL_DIST_PER_ATOM].min()
    echmax = df[E_CHULL_DIST_PER_ATOM].max()
    print(f"ECHMIN: {echmin:.15f}, ECHMAX: {echmax:.15f}")

    assert np.allclose(echmin, 0.0)
    assert np.allclose(echmax, 0.435098487307693)


# ------------------------------------------------------------- compute_compositions


def _toy_frames() -> pd.DataFrame:
    """H2O, H2 and O: hand-countable compositions with two elements."""
    atoms = [Atoms("H2O"), Atoms("H2"), Atoms("O")]
    return pd.DataFrame({ASE_ATOMS: atoms})


def test_compute_compositions_counts_and_concentrations_by_hand():
    df = _toy_frames()

    elements = compute_compositions(df)

    assert elements == ["H", "O"]
    assert df[COMP_DICT].map(dict).tolist() == [{"H": 2, "O": 1}, {"H": 2}, {"O": 1}]
    assert df[NUMBER_OF_ATOMS].tolist() == [3, 2, 1]
    assert df["n_H"].tolist() == [2, 2, 0]
    assert df["n_O"].tolist() == [1, 0, 1]
    np.testing.assert_allclose(
        df["c_H"], [2 / 3, 1.0, 0.0], rtol=ARITH.rtol, atol=ARITH.atol
    )
    np.testing.assert_allclose(
        df["c_O"], [1 / 3, 0.0, 1.0], rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_compute_compositions_conserves_the_atoms():
    df = _toy_frames()
    elements = compute_compositions(df)

    n_total = sum(df["n_" + el] for el in elements)
    c_total = sum(df["c_" + el] for el in elements)
    np.testing.assert_array_equal(n_total, df[NUMBER_OF_ATOMS])
    np.testing.assert_allclose(c_total, 1.0, rtol=ARITH.rtol, atol=ARITH.atol)


def test_compute_compositions_adds_the_tuple_column_only_on_request():
    with_tuples = _toy_frames()
    without_tuples = _toy_frames()

    compute_compositions(with_tuples)
    compute_compositions(without_tuples, compute_composition_tuples=False)

    assert with_tuples[COMP_TUPLE].tolist()[1] == (("H", 1.0),)
    assert COMP_TUPLE not in without_tuples.columns


def test_compute_compositions_columns_belong_to_their_own_element():
    # every element column holds that element's counts, none repeats another element's
    df = pd.DataFrame({ASE_ATOMS: [Atoms("HLi2"), Atoms("Li"), Atoms("Be3H")]})

    elements = compute_compositions(df)

    assert elements == ["Be", "H", "Li"]
    assert df["n_Be"].tolist() == [0, 0, 3]
    assert df["n_H"].tolist() == [1, 0, 1]
    assert df["n_Li"].tolist() == [2, 1, 0]
