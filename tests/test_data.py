"""Characterization tests for ``load_and_prepare_datasets`` (``tensorpotential/cli/data.py``).

Logic layer: every branch of the function (reference energy options, stress extraction, shift,
scale, convex hull, duplicate indices, element maps, saving, extra builders, the streaming and the
in-memory pipelines), driven through real files and real data builders.
Value layer: expected numbers come from hand-built energies and from independent oracles (numpy
least squares on known compositions, the lower convex hull of a hand-made set of compositions,
``ase.neighborlist`` pair counts, a plain-Python neighbour estimate), never from the unit.
The function writes under ``./seed/<seed>`` of the working directory, so every test runs in
``tmp_path``.
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from tensorpotential import constants as tc
from tensorpotential.cli import data as cli_data
from tensorpotential.cli.data import load_and_prepare_datasets
from tensorpotential.data.streaming import StreamingDatasetWrapper
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH
from tests.tolerances import LEAST_SQUARES_F64 as LSTSQ

SEED = 7
RCUT = 3.0
ELEMENTS = ["Al", "Cu"]
# single-atom energies (eV) of the synthetic data sets; total energy = n_Al * E_AL + n_Cu * E_CU
E_AL, E_CU = -3.0, -5.0
LOSS_EF = {"energy": {"weight": 1}, "forces": {"weight": 1}}


# ------------------------------------------------------------------------------ builders
def cell(n_al: int, seed: int = 0) -> Atoms:
    """Four-atom cubic cell with ``n_al`` Al atoms, rattled so no pair sits on a cutoff tie."""
    at = bulk("Cu", cubic=True)
    at.set_chemical_symbols(["Al"] * n_al + ["Cu"] * (4 - n_al))
    at.rattle(0.03, seed=seed)
    return at


def frame(n_al_list, energies=None, seed=0, force_scale=0.5) -> pd.DataFrame:
    """Data set of ``cell`` structures; default energy is exactly ``n_Al*E_AL + n_Cu*E_CU``."""
    atoms = [cell(n, seed=seed + i) for i, n in enumerate(n_al_list)]
    if energies is None:
        energies = [n * E_AL + (4 - n) * E_CU for n in n_al_list]
    rng = np.random.RandomState(seed)
    forces = [rng.normal(scale=force_scale, size=(len(a), 3)) for a in atoms]
    return pd.DataFrame({"ase_atoms": atoms, "energy": energies, "forces": forces})


def make_args(
    tmp_path,
    train,
    test=None,
    *,
    data=None,
    potential=None,
    fit=None,
    reference_energy=0,
):
    """Input dict for the function; ``reference_energy=None`` leaves the key out of ``data``."""
    train_file = tmp_path / "train.pkl.gz"
    train.to_pickle(train_file)
    data_cfg = {"filename": str(train_file)}
    if reference_energy is not None:
        data_cfg[tc.INPUT_REFERENCE_ENERGY] = reference_energy
    if test is not None:
        test_file = tmp_path / "test.pkl.gz"
        test.to_pickle(test_file)
        data_cfg["test_filename"] = str(test_file)
    data_cfg.update(data or {})
    return {
        tc.INPUT_DATA_SECTION: data_cfg,
        tc.INPUT_POTENTIAL_SECTION: {"elements": list(ELEMENTS), **(potential or {})},
        tc.INPUT_FIT_SECTION: {tc.INPUT_FIT_LOSS: LOSS_EF, **(fit or {})},
        tc.INPUT_CUTOFF: RCUT,
    }


def saved(tmp_path, name="training_set.pkl.gz") -> pd.DataFrame:
    return pd.read_pickle(tmp_path / "seed" / str(SEED) / name)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(args, batch_size=2, **kwargs):
    return load_and_prepare_datasets(args, batch_size, seed=SEED, **kwargs)


def lower_hull_distance(points, query):
    """Distance above the lower hull of 2D points (c, e), by brute force over all point pairs."""
    best = math.inf
    for i, (c1, e1) in enumerate(points):
        for c2, e2 in points[i + 1 :]:
            if c1 == c2 or not min(c1, c2) <= query[0] <= max(c1, c2):
                continue
            line = e1 + (e2 - e1) * (query[0] - c1) / (c2 - c1)
            if all(
                e >= e1 + (e2 - e1) * (c - c1) / (c2 - c1) - 1e-12
                for c, e in points
                if min(c1, c2) <= c <= max(c1, c2)
            ):
                best = min(best, query[1] - line)
    return 0.0 if best == math.inf else best


# --------------------------------------------------------------------------- dispatch
def test_distributed_data_is_delegated(workdir, monkeypatch):
    calls = {}

    def spy(args_yaml, seed, strategy):
        calls.update(args=args_yaml, seed=seed, strategy=strategy)
        return "delegated"

    # replaces the distributed loader, which needs a pre-computed shard directory
    # (it is exercised by tests/test_distrib.py in a subprocess)
    monkeypatch.setattr(cli_data, "load_and_prepare_distributed_datasets", spy)
    args = make_args(workdir, frame([4, 0]), data={"distributed": True})
    assert run(args, strategy="S") == "delegated"
    assert calls["seed"] == SEED
    assert calls["strategy"] == "S"
    assert calls["args"] is args


# -------------------------------------------------------------------- reference energy
def test_reference_energy_zero_copies_energy(workdir):
    train, test = frame([4, 0, 2, 1]), frame([3, 2], seed=5)
    args = make_args(workdir, train, test, data={"reference_energy": 0})
    run(args)
    for name, source in (("training_set.pkl.gz", train), ("test_set.pkl.gz", test)):
        out = saved(workdir, name)
        np.testing.assert_array_equal(out["energy_corrected"], source["energy"])
        np.testing.assert_allclose(
            out["energy_corrected_per_atom"],
            source["energy"] / 4,
            rtol=ARITH.rtol,
            atol=ARITH.atol,
        )


def test_reference_energy_dict_subtracts_single_atom_energies(workdir):
    interaction = [0.1, -0.2, 0.3, -0.4]
    n_al = [4, 0, 2, 1]
    energies = [n * E_AL + (4 - n) * E_CU + e for n, e in zip(n_al, interaction)]
    test = frame([3], energies=[3 * E_AL + E_CU + 0.7], seed=5)
    args = make_args(
        workdir,
        frame(n_al, energies=energies),
        test,
        data={"reference_energy": {"Al": E_AL, "Cu": E_CU}},
    )
    run(args)
    np.testing.assert_allclose(
        saved(workdir)["energy_corrected"],
        interaction,
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )
    np.testing.assert_allclose(
        saved(workdir, "test_set.pkl.gz")["energy_corrected"],
        [0.7],
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )


def test_reference_energy_auto_recovers_single_atom_energies(workdir, caplog):
    # energies are exactly linear in the composition: the fit residual is zero for train and test
    test = frame([3, 1], seed=5)
    args = make_args(
        workdir, frame([4, 0, 2, 1, 3]), test, data={"reference_energy": "auto"}
    )
    with caplog.at_level(logging.INFO):
        run(args)
    for name in ("training_set.pkl.gz", "test_set.pkl.gz"):
        np.testing.assert_allclose(
            saved(workdir, name)["energy_corrected"],
            0.0,
            rtol=LSTSQ.rtol,
            atol=LSTSQ.atol,
        )
    assert any("reference_energy=auto" in r.message for r in caplog.records)


def test_reference_energy_auto_without_test_set(workdir):
    args = make_args(workdir, frame([4, 0, 2]), data={"reference_energy": "auto"})
    run(args)
    np.testing.assert_allclose(
        saved(workdir)["energy_corrected"], 0.0, rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )


def test_reference_energy_unsupported_value_raises(workdir):
    args = make_args(workdir, frame([4, 0]), data={"reference_energy": "vasp"})
    with pytest.raises(RuntimeError, match="reference_energy"):
        run(args)


def test_missing_energy_corrected_column_raises(workdir):
    with pytest.raises(RuntimeError, match="energy_corrected"):
        run(make_args(workdir, frame([4, 0]), reference_energy=None))


def test_energy_corrected_missing_from_test_only_raises(workdir):
    train = frame([4, 0])
    train["energy_corrected"] = train["energy"]
    with pytest.raises(RuntimeError, match="energy_corrected"):
        run(make_args(workdir, train, frame([2]), reference_energy=None))


def test_external_energy_corrected_column_is_used(workdir):
    train, test = frame([4, 0]), frame([2], seed=3)
    for df in (train, test):
        df["energy_corrected"] = df["energy"] + 1.0
    run(make_args(workdir, train, test, reference_energy=None))
    np.testing.assert_allclose(
        saved(workdir)["energy_corrected_per_atom"],
        (train["energy"] + 1.0) / 4,
        rtol=ARITH.rtol,
    )
    np.testing.assert_allclose(
        saved(workdir, "test_set.pkl.gz")["energy_corrected_per_atom"],
        (test["energy"] + 1.0) / 4,
        rtol=ARITH.rtol,
    )


def test_forces_only_fit_skips_the_energy_reference_section(workdir):
    train = frame([4, 0])
    train["energy_corrected"] = train[
        "energy"
    ]  # the batch builder reads it; the section is skipped
    args = make_args(
        workdir,
        train,
        fit={tc.INPUT_FIT_LOSS: {"forces": {"weight": 1}}},
        reference_energy=None,
    )
    train_batches, *_ = run(args)
    assert "energy_corrected_per_atom" not in saved(workdir).columns
    assert len(train_batches) == 1


# ------------------------------------------------------------------------------ stress
def stress_fit():
    return {tc.INPUT_FIT_LOSS: {**LOSS_EF, "stress": {"weight": 1}}}


def test_stress_column_is_kept(workdir):
    train = frame([4, 0, 2])
    train["stress"] = [np.arange(6.0) * 0.01 * (i + 1) for i in range(3)]
    args = make_args(workdir, train, data={"reference_energy": 0}, fit=stress_fit())
    run(args)
    out = saved(workdir)
    # the saved frame keeps the stress column as given, in the original row order
    np.testing.assert_array_equal(np.vstack(out["stress"]), np.vstack(train["stress"]))


@pytest.mark.parametrize("key", ["stress", "virial"])
def test_stress_is_extracted_from_results_for_train_and_test(workdir, key):
    def with_results(df):
        df["results"] = [{"stress": np.full(6, 0.1 * (i + 1))} for i in range(len(df))]
        return df

    train, test = with_results(frame([4, 0, 2])), with_results(frame([1], seed=4))
    fit = {tc.INPUT_FIT_LOSS: {**LOSS_EF, key: {"weight": 1}}}
    run(make_args(workdir, train, test, data={"reference_energy": 0}, fit=fit))
    np.testing.assert_allclose(
        np.vstack(saved(workdir)["stress"]),
        np.vstack([np.full(6, 0.1 * (i + 1)) for i in range(3)]),
    )
    np.testing.assert_allclose(
        np.vstack(saved(workdir, "test_set.pkl.gz")["stress"]), np.full((1, 6), 0.1)
    )


def test_stress_without_stress_or_results_raises(workdir):
    args = make_args(
        workdir, frame([4, 0]), data={"reference_energy": 0}, fit=stress_fit()
    )
    with pytest.raises(ValueError, match="stress"):
        run(args)


# ------------------------------------------------------------------------------- shift
def test_shift_recovers_single_atom_energies_by_element_index(workdir):
    args = make_args(
        workdir,
        frame([4, 0, 2, 1, 3]),
        data={"reference_energy": 0},
        potential={"shift": True, "elements": {"Cu": 0, "Al": 1}},
    )
    _, _, element_map, stats, *_ = run(args)
    assert element_map == {"Cu": 0, "Al": 1}
    shifts = stats["atomic_shift_map"]
    assert sorted(shifts) == [0, 1]
    np.testing.assert_allclose(
        shifts[0], E_CU, rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )  # Cu is index 0 here
    np.testing.assert_allclose(shifts[1], E_AL, rtol=LSTSQ.rtol, atol=LSTSQ.atol)
    assert stats["fm_shift_dict"] is None
    assert stats["constant_out_shift"] == 0


def test_shift_auto_without_finetuning_uses_the_standard_fit(workdir):
    args = make_args(
        workdir,
        frame([4, 0, 2]),
        data={"reference_energy": 0},
        potential={"shift": "auto"},
    )
    stats = run(args)[3]
    np.testing.assert_allclose(
        stats["atomic_shift_map"][0], E_AL, rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )
    np.testing.assert_allclose(
        stats["atomic_shift_map"][1], E_CU, rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )


def test_no_shift_gives_no_atomic_shift_map(workdir):
    stats = run(make_args(workdir, frame([4, 0]), data={"reference_energy": 0}))[3]
    assert stats["atomic_shift_map"] is None
    assert stats["fm_shift_dict"] is None


def fm_potential(checkpoint="models/fm/checkpoint"):
    return {
        "shift": "auto",
        tc.INPUT_POTENTIAL_FINETUNE_FOUNDATION_MODEL: "GRACE-FM",
        tc.INPUT_POTENTIAL_CHECKPOINT_NAME: checkpoint,
    }


def test_fm_auto_shift_requires_a_checkpoint(workdir):
    potential = fm_potential(checkpoint="")
    args = make_args(
        workdir, frame([4, 0]), data={"reference_energy": 0}, potential=potential
    )
    with pytest.raises(ValueError, match="checkpoint"):
        run(args)


@pytest.mark.parametrize("pipeline", ["in_memory", "streaming"])
def test_fm_auto_shift_is_routed_to_fm_shift_dict(workdir, monkeypatch, pipeline):
    calls = {}
    result = {"Al": 0.25, "Cu": -0.5}

    def spy(train_df, checkpoint_folder, seed):
        calls.update(n_train=len(train_df), folder=checkpoint_folder, seed=seed)
        return result

    # replaces the evaluation of a foundation model (a TF checkpoint, HPC-only weights);
    # the function itself is covered in tests/test_fm_shift_auto.py
    monkeypatch.setattr(cli_data, "compute_fm_energy_shift_auto", spy)
    args = make_args(
        workdir,
        frame([4, 0, 2]),
        data={"reference_energy": 0, "pipeline": pipeline},
        potential={**fm_potential(), "avg_n_neigh": 11.0},
    )
    stats = run(args)[3]
    assert calls == {"n_train": 3, "folder": "models/fm", "seed": SEED}
    assert stats["fm_shift_dict"] == result
    assert stats["atomic_shift_map"] is None


# ------------------------------------------------------------------------------- scale
def test_scale_float_is_used_as_is(workdir):
    args = make_args(
        workdir, frame([4, 0]), data={"reference_energy": 0}, potential={"scale": 0.75}
    )
    assert run(args)[3]["constant_out_scale"] == 0.75


def test_default_scale_is_one(workdir):
    assert (
        run(make_args(workdir, frame([4, 0]), data={"reference_energy": 0}))[3][
            "constant_out_scale"
        ]
        == 1.0
    )


def test_auto_scale_is_rms_of_train_forces(workdir, caplog):
    train = frame([4, 0, 2], force_scale=2.0)
    expected = math.sqrt(np.mean(np.vstack(train["forces"]) ** 2))
    assert expected > 0.8  # no warning in this regime
    args = make_args(
        workdir, train, data={"reference_energy": 0}, potential={"scale": True}
    )
    with caplog.at_level(logging.WARNING):
        scale = run(args)[3]["constant_out_scale"]
    np.testing.assert_allclose(scale, expected, rtol=ARITH.rtol)
    assert not any("possibly too small" in r.message for r in caplog.records)


def test_auto_scale_warns_when_small(workdir, caplog):
    train = frame([4, 0, 2], force_scale=0.5)
    expected = math.sqrt(np.mean(np.vstack(train["forces"]) ** 2))
    assert 0.1 < expected < 0.8
    args = make_args(
        workdir, train, data={"reference_energy": 0}, potential={"scale": True}
    )
    with caplog.at_level(logging.WARNING):
        scale = run(args)[3]["constant_out_scale"]
    np.testing.assert_allclose(scale, expected, rtol=ARITH.rtol)
    assert any("possibly too small" in r.message for r in caplog.records)


def test_auto_scale_has_a_floor(workdir):
    train = frame([4, 0, 2], force_scale=1e-6)
    args = make_args(
        workdir, train, data={"reference_energy": 0}, potential={"scale": True}
    )
    assert run(args)[3]["constant_out_scale"] == 0.1


# ------------------------------------------------------------------------- convex hull
# Hand-built compositions (Al fraction c, energy per atom, eV): pure Al -1.0, pure Cu -2.0, a stable
# half-half compound -2.0 (formation energy -0.5 against the pure-element line -1.5), a half-half
# structure at -1.4 (formation +0.1, so 0.6 above the compound), and one at +0.5 (2.5 above).
HULL_CASES = {
    "al": (4, -1.0),
    "cu": (0, -2.0),
    "stable": (2, -2.0),
    "above": (2, -1.4),
    "far": (2, 0.5),
}


def hull_frame(names, seed=0):
    n_al = [HULL_CASES[n][0] for n in names]
    return frame(n_al, energies=[4 * HULL_CASES[n][1] for n in names], seed=seed)


def hull_args(workdir, train, test=None, **kwargs):
    return make_args(
        workdir,
        train,
        test,
        data={"reference_energy": 0},
        fit={tc.INPUT_FIT_LOSS: LOSS_EF, "compute_convex_hull": True},
        **kwargs,
    )


def test_hull_distance_of_train_only_set(workdir):
    names = ["al", "cu", "stable", "above", "far"]
    out = run(hull_args(workdir, hull_frame(names)))
    dist = saved(workdir)["e_chull_dist_per_atom"].to_numpy()
    np.testing.assert_allclose(
        dist, [0.0, 0.0, 0.0, 0.6, 2.5], rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )
    train_groups = out[4]
    assert train_groups["group__low"].tolist() == [True, True, True, True, False]
    assert train_groups["NUMBER_OF_ATOMS"].tolist() == [4] * 5
    assert out[5] is None


def test_hull_distance_matches_brute_force_lower_hull(workdir):
    names = ["al", "cu", "stable", "above", "far"]
    run(hull_args(workdir, hull_frame(names)))
    # coordinates (c_Al, formation energy per atom): the pure-element line is -1.0 -> -2.0
    pts = [
        (HULL_CASES[n][0] / 4, HULL_CASES[n][1] - (-2.0 + HULL_CASES[n][0] / 4))
        for n in names
    ]
    expected = [lower_hull_distance(pts, p) for p in pts]
    np.testing.assert_allclose(
        saved(workdir)["e_chull_dist_per_atom"],
        expected,
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )


def test_joint_hull_uses_train_and_test_points(workdir):
    # the stable compound is only in the test set: the train structure at -1.4 is 0.6 above the
    # joint hull, but only 0.1 above the hull of the train set alone (the pure-element line)
    train, test = (
        hull_frame(["al", "cu", "above"]),
        hull_frame(["stable", "far"], seed=9),
    )
    out = run(hull_args(workdir, train, test))
    train_out, test_out = saved(workdir), saved(workdir, "test_set.pkl.gz")
    np.testing.assert_allclose(
        train_out["e_chull_dist_per_atom"],
        [0.0, 0.0, 0.6],
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )
    np.testing.assert_allclose(
        test_out["e_chull_dist_per_atom"], [0.0, 2.5], rtol=LSTSQ.rtol, atol=LSTSQ.atol
    )
    train_only = hull_frame(["al", "cu", "above"])
    run(hull_args(workdir, train_only))
    np.testing.assert_allclose(
        saved(workdir)["e_chull_dist_per_atom"],
        [0.0, 0.0, 0.1],
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )
    assert out[4]["group__low"].tolist() == [True, True, True]
    assert out[5]["group__low"].tolist() == [True, False]


def test_joint_hull_keeps_rows_order_and_drops_the_helper_column(workdir):
    train, test = (
        hull_frame(["al", "cu", "above"]),
        hull_frame(["stable", "far"], seed=9),
    )
    run(hull_args(workdir, train, test))
    for name, source in (("training_set.pkl.gz", train), ("test_set.pkl.gz", test)):
        out = saved(workdir, name)
        assert "is_train" not in out.columns
        np.testing.assert_array_equal(out["energy"], source["energy"])
        assert [a.get_chemical_formula() for a in out["ase_atoms"]] == [
            a.get_chemical_formula() for a in source["ase_atoms"]
        ]
        assert out.index.tolist() == list(range(len(source)))


def test_joint_hull_does_not_alter_the_files_it_read(workdir):
    train, test = (
        hull_frame(["al", "cu", "above"]),
        hull_frame(["stable", "far"], seed=9),
    )
    run(hull_args(workdir, train, test))
    for name, source in (("train.pkl.gz", train), ("test.pkl.gz", test)):
        on_disk = pd.read_pickle(workdir / name)
        assert list(on_disk.columns) == list(source.columns)


def test_hull_joint_call_gets_one_frame_with_train_rows_first(workdir, monkeypatch):
    seen = {}
    real = cli_data.compute_convexhull_dist

    def spy(df, **kwargs):
        seen["is_train"] = df["is_train"].tolist()
        seen["index"] = df.index.tolist()
        return real(df, **kwargs)

    # wraps (does not replace) the hull routine to see the frame it is given
    monkeypatch.setattr(cli_data, "compute_convexhull_dist", spy)
    train, test = (
        hull_frame(["al", "cu", "above"]),
        hull_frame(["stable", "far"], seed=9),
    )
    run(hull_args(workdir, train, test))
    assert seen["is_train"] == [1, 1, 1, 0, 0]
    assert seen["index"] == [0, 1, 2, 0, 1]


def test_hull_off_leaves_no_hull_column(workdir, caplog):
    with caplog.at_level(logging.INFO):
        out = run(
            make_args(
                workdir, hull_frame(["al", "cu", "above"]), data={"reference_energy": 0}
            )
        )
    assert "e_chull_dist_per_atom" not in saved(workdir).columns
    assert out[4] is None
    assert any("compute_convex_hull is False" in r.message for r in caplog.records)


# ---------------------------------------------------------------------- duplicate index
def test_duplicate_indices_are_reset(workdir, monkeypatch):
    train, test = frame([4, 0, 2]), frame([1, 3], seed=4)
    train.index = [5, 5, 6]
    test.index = [2, 2]

    # replaces file loading: the real loaders always return a unique index, so only an injected
    # frame can reach the defensive resets
    monkeypatch.setattr(
        cli_data,
        "load_train_test_datasets",
        lambda data_config, seed=None: (train, test),
    )
    args = make_args(workdir, frame([4]), data={"reference_energy": 0})
    run(args)
    assert saved(workdir).index.tolist() == [0, 1, 2]
    assert saved(workdir, "test_set.pkl.gz").index.tolist() == [0, 1]


# ---------------------------------------------------------------------------- elements
def test_elements_default_to_train_and_test_union(workdir):
    args = make_args(
        workdir, frame([0, 0]), frame([4], seed=2), potential={"elements": None}
    )
    element_map = run(args)[2]
    assert element_map == {"Al": 0, "Cu": 1}


def test_elements_default_without_test_set(workdir):
    args = make_args(workdir, frame([0, 0]), potential={"elements": None})
    assert run(args)[2] == {"Cu": 0}


def test_elements_list_defines_the_index_order(workdir):
    args = make_args(workdir, frame([0, 4]), potential={"elements": ["Cu", "Al"]})
    assert run(args)[2] == {"Cu": 0, "Al": 1}


def test_elements_dict_is_used_verbatim(workdir):
    mapping = {"Cu": 3, "Al": 5}
    args = make_args(workdir, frame([0, 4]), potential={"elements": mapping})
    assert run(args)[2] == mapping


# ----------------------------------------------------------------------------- saving
def test_save_dataset_false_writes_nothing(workdir):
    args = make_args(
        workdir, frame([4, 0]), frame([2], seed=2), data={"save_dataset": False}
    )
    run(args)
    assert not (workdir / "seed" / str(SEED) / "training_set.pkl.gz").exists()
    assert not (workdir / "seed" / str(SEED) / "test_set.pkl.gz").exists()


def test_save_dataset_without_test_set_writes_only_train(workdir):
    run(make_args(workdir, frame([4, 0])))
    assert (workdir / "seed" / str(SEED) / "training_set.pkl.gz").exists()
    assert not (workdir / "seed" / str(SEED) / "test_set.pkl.gz").exists()


# ------------------------------------------------------------------------- extra builders
def test_unknown_extra_builder_raises(workdir):
    args = make_args(
        workdir, frame([4, 0]), data={"extra_components": {"NoSuchBuilder": {}}}
    )
    with pytest.raises(NameError, match="NoSuchBuilder"):
        run(args)


def test_extra_builders_missing_modules_are_tolerated_until_used(workdir, monkeypatch):
    # a None entry in sys.modules makes the import raise ImportError, as if the module were absent
    for module in (
        "tensorpotential.experimental.extra_data_builders",
        "tensorpotential.extra.extra_data_builders",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    args = make_args(
        workdir,
        frame([4, 0]),
        data={"extra_components": {"ReferenceTensorDataBuilder": {"tensor_rank": 2}}},
    )
    with pytest.raises(NameError, match="ReferenceTensorDataBuilder"):
        run(args)


# ----------------------------------------------------------------- in-memory pipeline
def counted_neighbours(frames) -> float:
    """Average pair count per atom from ``ase.neighborlist`` (independent of the data builder)."""
    pairs = atoms = 0
    for at in (a for f in frames for a in f["ase_atoms"]):
        pairs += len(neighbor_list("i", at, RCUT))
        atoms += len(at)
    return pairs / atoms


def test_in_memory_batches_and_computed_neighbour_average(workdir):
    train, test = frame([4, 0, 2, 1, 3, 2]), frame([1, 3, 0], seed=4)
    train_batches, test_batches, element_map, stats, train_groups, test_groups = run(
        make_args(workdir, train, test, data={"reference_energy": 0}), batch_size=2
    )
    assert len(train_batches) == 3
    assert len(test_batches) == 2
    assert element_map == {"Al": 0, "Cu": 1}
    assert stats["constant_out_shift"] == 0
    np.testing.assert_allclose(
        stats["avg_n_neigh"], counted_neighbours([train]), rtol=ARITH.rtol
    )
    assert train_groups is None
    assert test_groups is None


def test_provided_neighbour_average_is_not_recomputed(workdir):
    args = make_args(workdir, frame([4, 0]), potential={"avg_n_neigh": 7.5})
    assert run(args)[3]["avg_n_neigh"] == 7.5


def test_test_batch_size_overrides_batch_size(workdir):
    args = make_args(
        workdir,
        frame([4, 0, 2, 1]),
        frame([1, 3, 0, 2], seed=4),
        fit={tc.INPUT_FIT_LOSS: LOSS_EF, "test_max_n_buckets": 1},
    )
    train_batches, test_batches, *_ = run(args, batch_size=4, test_batch_size=2)
    assert len(train_batches) == 1
    assert len(test_batches) == 2


def test_direct_split_without_buckets_has_no_padding_stats(workdir, caplog):
    fit = {
        tc.INPUT_FIT_LOSS: LOSS_EF,
        "train_max_n_buckets": None,
        "test_max_n_buckets": None,
    }
    # direct_split makes n_structures // batch_size groups: 4 // 2 = 2 and 5 // 2 = 2 (sizes 3 and 2)
    args = make_args(
        workdir, frame([4, 0, 2, 1]), frame([1, 3, 0, 2, 4], seed=4), fit=fit
    )
    with caplog.at_level(logging.INFO):
        train_batches, test_batches, *_ = run(args, batch_size=2)
    assert len(train_batches) == 2
    assert len(test_batches) == 2
    messages = [r.message for r in caplog.records]
    assert any(
        m.startswith("[TRAIN] dataset stats:  num. batches: 2") for m in messages
    )
    assert any(m.startswith("[TEST] dataset stats:  num. batches: 2") for m in messages)


def test_dense_neighbour_layout_runs_and_reports_padding(workdir, caplog):
    args = make_args(
        workdir,
        frame([4, 0, 2, 1]),
        frame([1, 3], seed=4),
        potential={tc.INPUT_POTENTIAL_DENSE_NBR: True},
    )
    with caplog.at_level(logging.INFO):
        train_batches, test_batches, *_ = run(args, batch_size=2)
    assert len(train_batches) == 2
    assert len(test_batches) == 1
    messages = [r.message for r in caplog.records]
    assert any(m.startswith("[TRAIN] dense padding") for m in messages)
    assert any(m.startswith("[TEST] dense padding") for m in messages)


def test_histogram_failure_is_not_fatal(workdir, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise OSError("no display")

    # replaces matplotlib plotting, an external boundary that can fail on headless machines
    monkeypatch.setattr(cli_data.DatasetHistPlotter, "plot", broken)
    with caplog.at_level(logging.WARNING):
        train_batches, *_ = run(make_args(workdir, frame([4, 0])))
    assert len(train_batches) == 1
    assert any("Failed to make histograms plot" in r.message for r in caplog.records)


# -------------------------------------------------------------------- streaming pipeline
STREAMING = {
    "pipeline": "streaming",
    "streaming": {"target_metric": 400, "num_bins": 2, "sorting_buffer_size": 4},
}


def plain_estimate(frames) -> float:
    """Neighbour estimate of the streaming branch worked out in plain Python."""
    total_neigh = total_atoms = 0
    for at in (a for f in frames for a in f["ase_atoms"]):
        n = len(at)
        density = n / at.get_volume() if at.pbc.any() and at.get_volume() > 0 else 1.0
        total_neigh += int(n * density * 4.0 / 3.0 * math.pi * RCUT**3)
        total_atoms += n
    return total_neigh / total_atoms


def test_streaming_returns_wrappers_and_estimates_neighbours(workdir):
    train = frame([4, 0, 2])
    molecule = Atoms(
        "Cu2", positions=[[0, 0, 0], [0, 0, 2.3]]
    )  # not periodic: density falls back to 1
    train = pd.concat(
        [
            train,
            pd.DataFrame({
                "ase_atoms": [molecule],
                "energy": [-9.0],
                "forces": [np.zeros((2, 3))],
            }),
        ],
        ignore_index=True,
    )
    args = make_args(
        workdir, train, frame([1], seed=4), data={"reference_energy": 0, **STREAMING}
    )
    train_batches, test_batches, element_map, stats, train_groups, test_groups = run(
        args
    )
    assert isinstance(train_batches, StreamingDatasetWrapper)
    assert isinstance(test_batches, StreamingDatasetWrapper)
    assert element_map == {"Al": 0, "Cu": 1}
    np.testing.assert_allclose(
        stats["avg_n_neigh"], plain_estimate([train]), rtol=ARITH.rtol
    )
    assert stats["atomic_shift_map"] is None
    assert train_groups is None
    assert test_groups is None


def test_streaming_without_test_set_and_with_provided_neighbours(workdir):
    args = make_args(
        workdir, frame([4, 0]), data=STREAMING, potential={"avg_n_neigh": 9.0}
    )
    train_batches, test_batches, _, stats, *_ = run(args)
    assert isinstance(train_batches, StreamingDatasetWrapper)
    assert test_batches is None
    assert stats["avg_n_neigh"] == 9.0


def test_streaming_shift_is_keyed_by_element_index(workdir):
    args = make_args(
        workdir,
        frame([4, 0, 2, 1]),
        data={"reference_energy": 0, **STREAMING},
        potential={"shift": True, "avg_n_neigh": 9.0, "elements": {"Cu": 0, "Al": 1}},
    )
    shifts = run(args)[3]["atomic_shift_map"]
    np.testing.assert_allclose(shifts[0], E_CU, rtol=LSTSQ.rtol, atol=LSTSQ.atol)
    np.testing.assert_allclose(shifts[1], E_AL, rtol=LSTSQ.rtol, atol=LSTSQ.atol)


def test_streaming_hull_groups(workdir):
    train, test = (
        hull_frame(["al", "cu", "above"]),
        hull_frame(["stable", "far"], seed=9),
    )
    args = hull_args(workdir, train, test)
    args[tc.INPUT_DATA_SECTION].update(STREAMING)
    args[tc.INPUT_POTENTIAL_SECTION]["avg_n_neigh"] = 9.0
    _, _, _, _, train_groups, test_groups = run(args)
    assert train_groups["group__low"].tolist() == [True, True, True]
    assert test_groups["group__low"].tolist() == [True, False]
    assert test_groups["NUMBER_OF_ATOMS"].tolist() == [4, 4]
    np.testing.assert_allclose(
        saved(workdir)["e_chull_dist_per_atom"],
        [0.0, 0.0, 0.6],
        rtol=LSTSQ.rtol,
        atol=LSTSQ.atol,
    )
