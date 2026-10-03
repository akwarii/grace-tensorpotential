"""Tests for tools/oracle_snapshot.py.

Logic tests run without TensorFlow. The physics tests use the real 2-layer test model
with seeded random weights and check the snapshot against oracles that do not use the
code under test: central finite differences of the energy for forces and stress, and
translation invariance.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oracle_snapshot as osn

# ---------------------------------------------------------------------- logic, no TF


def test_seeded_values_are_reproducible_and_name_dependent():
    a = osn.seeded_values("layer/w:0", (4, 3))
    assert np.array_equal(a, osn.seeded_values("layer/w:0", (4, 3)))
    assert not np.array_equal(a, osn.seeded_values("other/w:0", (4, 3)))
    assert not np.array_equal(a, osn.seeded_values("layer/w:0", (4, 3), seed=1))


def test_seeded_values_matrices_are_scaled_and_vectors_centred_on_one():
    # Bounds are N_SIGMA standard errors of the sampling distribution of each statistic
    # (mean: sd/sqrt(n); standard deviation of normal data: relative 1/sqrt(2n)).
    n_sigma = 5
    w = osn.seeded_values("big", (400, 400))
    sd_w = 1 / np.sqrt(400)
    assert w.shape == (400, 400)
    assert abs(w.mean()) < n_sigma * sd_w / np.sqrt(w.size)
    assert w.std() == pytest.approx(sd_w, rel=n_sigma / np.sqrt(2 * w.size), abs=0)
    vec = osn.seeded_values("vec", (50_000,))
    assert vec.mean() == pytest.approx(
        1.0, abs=n_sigma * 0.1 / np.sqrt(vec.size), rel=0
    )
    assert vec.std() == pytest.approx(0.1, rel=n_sigma / np.sqrt(2 * vec.size), abs=0)
    scalar = osn.seeded_values("scalar", ())
    assert scalar.shape == () and 0.5 < scalar < 1.5
    assert np.all(osn.seeded_values("never zero", (64,)) != 0)


def _snap(**arrays):
    return {k: np.asarray(v, dtype=float) for k, v in arrays.items()}


def test_compare_identical_snapshots_is_ok_with_zero_difference():
    a = _snap(x=[1.0, 2.0], y=[[3.0]])
    report = osn.compare_snapshots(a, dict(a))
    assert report["ok"]
    assert report["max_abs_diff"] == 0.0
    assert report["n_compared"] == 2


def test_compare_reports_largest_difference_and_its_key():
    a = _snap(x=[1.0, 2.0], y=[5.0])
    b = _snap(x=[1.0, 2.5], y=[5.25])
    report = osn.compare_snapshots(a, b)
    assert not report["ok"]
    assert report["max_abs_diff"] == 0.5  # 2.5 - 2.0, exact in binary
    assert report["max_abs_key"] == "x"
    assert report["exceeding"] == ["x", "y"]


def test_compare_tolerances_are_absolute_plus_relative():
    a = _snap(x=[100.0])
    b = _snap(x=[100.5])
    assert not osn.compare_snapshots(a, b)["ok"]
    assert not osn.compare_snapshots(a, b, atol=0.1)["ok"]
    assert osn.compare_snapshots(a, b, atol=0.5)["ok"]
    assert osn.compare_snapshots(a, b, rtol=0.01)["ok"]
    # the relative part scales with the reference (second) snapshot
    assert not osn.compare_snapshots(a, b, rtol=0.004)["ok"]


def test_compare_scale_tolerance_is_relative_to_the_largest_element():
    a = _snap(x=[2.0, 1e-14])
    b = _snap(x=[2.0, 3e-14])  # noise in a near-zero element, far below the scale
    assert not osn.compare_snapshots(a, b, rtol=1e-6)["ok"]
    assert osn.compare_snapshots(a, b, scale_rtol=1e-12)["ok"]
    assert not osn.compare_snapshots(a, b, scale_rtol=5e-15)["ok"]
    report = osn.compare_snapshots(a, b)
    assert report["max_scaled_diff"] == pytest.approx(1e-14, rel=1e-12, abs=0)
    # an all-zero reference array has no scale: only exact equality passes
    zero = _snap(z=[0.0, 0.0])
    assert (
        osn.compare_snapshots(zero, _snap(z=[0.0, 1e-30]), scale_rtol=0.5)["ok"]
        is False
    )


def test_compare_flags_missing_extra_and_reshaped_keys():
    a = _snap(x=[1.0], gone=[1.0], shape=[1.0, 2.0])
    b = _snap(x=[1.0], new=[1.0], shape=[1.0])
    report = osn.compare_snapshots(a, b)
    assert report["only_in_first"] == ["gone"]
    assert report["only_in_second"] == ["new"]
    assert report["shape_mismatch"] == ["shape"]
    assert not report["ok"]


def test_compare_ignores_meta_and_handles_integer_and_empty_arrays():
    meta_a = np.array(json.dumps({"t": 1}))
    meta_b = np.array(json.dumps({"t": 2}))
    a = {osn.META_KEY: meta_a, "i": np.array([1, 2]), "e": np.zeros((0, 3))}
    b = {osn.META_KEY: meta_b, "i": np.array([1, 2]), "e": np.zeros((0, 3))}
    assert osn.compare_snapshots(a, b)["ok"]
    b["i"] = np.array([1, 3])
    assert not osn.compare_snapshots(a, b)["ok"]


def test_cli_compare_exit_status(tmp_path, capsys):
    same, other = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(same, x=np.array([1.0]))
    np.savez(other, x=np.array([2.0]))
    assert osn.main(["compare", str(same), str(same)]) == 0
    assert osn.main(["compare", str(same), str(other)]) == 1
    assert osn.main(["compare", str(same), str(other), "--atol", "1.5"]) == 0
    assert osn.main(["compare", str(same), str(other), "--scale-rtol", "0.6"]) == 0

    noisy = tmp_path / "c.npz"
    np.savez(noisy, x=np.array([1.0 + 1e-14]))
    assert osn.main(["compare", str(same), str(noisy)]) == 0  # default 1e-12
    assert osn.main(["compare", str(same), str(noisy), "--scale-rtol", "0"]) == 1
    assert '"ok": false' in capsys.readouterr().out


def test_cli_write_saves_a_real_snapshot_with_metadata(tmp_path, capsys):
    _require_tensorflow()
    out = tmp_path / "sub" / "snap.npz"
    args = [
        "write", str(out), "--yamls", "model_grace.yaml", "--n-structures", "1",
        "--groups", "base", "--no-edge",
    ]
    assert osn.main(args) == 0
    with np.load(out) as saved:
        arrays = dict(saved)
    assert "model_grace/s0/energy" in arrays
    assert np.isfinite(arrays["model_grace/s0/energy"]).all()
    assert "model_grace/s1/energy" not in arrays
    assert f"{len(arrays) - 1} arrays written" in capsys.readouterr().out
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_arrays"] == len(arrays) - 1
    assert meta["npz_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert osn.main(["compare", str(out), str(out)]) == 0


def test_load_structures_takes_first_of_each_smallest_distinct_size(tmp_path):
    import pandas as pd
    from ase import Atoms

    sizes = [5, 2, 2, 9, 5, 3]
    structures = []
    for index, n in enumerate(sizes):
        atoms = Atoms("H" * n, positions=np.zeros((n, 3)))
        atoms.info["index_in_file"] = index
        structures.append(atoms)
    path = tmp_path / "s.pkl.gz"
    pd.DataFrame({"ase_atoms": structures}).to_pickle(path)
    chosen = osn.load_structures(path, 3)
    assert [len(a) for a in chosen] == [2, 3, 5]
    assert [a.info["index_in_file"] for a in chosen] == [1, 5, 0]


# ------------------------------------------------------------------ physics, with TF

MODEL = "model_grace_2L_omat.yaml"


def _require_tensorflow():
    """Skip without TensorFlow; tensorpotential is imported first, as it requires."""
    pytest.importorskip("tensorpotential")
    pytest.importorskip("tensorflow")


STEP = 1e-4
# numpy.isclose semantics: |actual - desired| <= atol + rtol * |desired|
TOLERANCES = {
    "fd_forces": (1e-4, 1e-8),
    "fd_stress": (1e-4, 1e-9),
    "translation_energy": (1e-10, 0.0),
    "translation_forces": (1e-8, 1e-10),
}


def _assert_close(actual, desired, name):
    rtol, atol = TOLERANCES[name]
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


@pytest.fixture(scope="module")
def tf_model():
    _require_tensorflow()
    return osn.build_model(osn.TESTS / MODEL)


@pytest.fixture(scope="module")
def structure():
    _require_tensorflow()
    return osn.load_structures(osn.TESTS / osn.DEFAULT_STRUCTURES, 2)[1]


def _energy(model, atoms):
    return float(osn.evaluate(model, atoms)["energy"].ravel()[0])


def test_build_model_keeps_constants_and_sets_nonzero_trainables(tf_model):
    constants = [v for v in tf_model.variables if not v.trainable]
    assert {v.name.split("/")[-1] for v in constants} >= {"cutoff:0"}
    cutoff = next(v for v in constants if v.name.endswith("cutoff:0"))
    assert float(cutoff.numpy()) == 6.0  # rcut of the yaml, stored as float64
    for var in tf_model.trainable_variables:
        assert np.all(np.isfinite(var.numpy()))
        assert np.any(var.numpy() != 0), var.name


def test_forces_equal_minus_energy_gradient(tf_model, structure):
    forces = osn.evaluate(tf_model, structure)["forces"]
    assert np.abs(forces).max() > 1e-3, "structure must carry non-trivial forces"
    for atom in (0, 7, 15):
        for axis in range(3):
            plus, minus = structure.copy(), structure.copy()
            plus.positions[atom, axis] += STEP
            minus.positions[atom, axis] -= STEP
            fd = -(_energy(tf_model, plus) - _energy(tf_model, minus)) / (2 * STEP)
            _assert_close(forces[atom, axis], fd, "fd_forces")


def test_stress_equals_energy_strain_derivative_over_volume(tf_model, structure):
    stress = osn.evaluate(tf_model, structure)["stress"]  # ASE Voigt order
    assert np.abs(stress).max() > 1e-4
    voigt = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}
    for index, (a, b) in voigt.items():
        energies = []
        for sign in (+1, -1):
            eps = np.zeros((3, 3))
            eps[a, b] += sign * STEP * (1.0 if a == b else 0.5)
            if a != b:
                eps[b, a] += sign * STEP * 0.5
            strained = structure.copy()
            strained.set_cell(
                structure.cell.array @ (np.eye(3) + eps), scale_atoms=True
            )
            energies.append(_energy(tf_model, strained))
        fd = (energies[0] - energies[1]) / (2 * STEP) / structure.get_volume()
        _assert_close(stress[index], fd, "fd_stress")


def test_energy_and_forces_are_translation_invariant(tf_model, structure):
    base = osn.evaluate(tf_model, structure)
    moved = structure.copy()
    moved.translate([0.37, -1.21, 2.05])
    moved.wrap()
    shifted = osn.evaluate(tf_model, moved)
    _assert_close(shifted["energy"], base["energy"], "translation_energy")
    _assert_close(shifted["forces"], base["forces"], "translation_forces")


def test_snapshot_is_reproducible_and_detects_a_changed_weight():
    _require_tensorflow()
    yamls = ("model_grace.yaml",)
    first = osn.take_snapshot(yamls, n_structures=2, groups=("base",), edge=False)
    second = osn.take_snapshot(yamls, n_structures=2, groups=("base",), edge=False)
    assert json.loads(str(first[osn.META_KEY]))["structures"]["s1"]["n_atoms"] == 16
    assert any(k.endswith("/energy") for k in first)
    assert osn.compare_snapshots(first, second)["ok"]
    assert osn.compare_snapshots(first, second)["max_abs_diff"] == 0.0

    perturbed = {k: v.copy() for k, v in second.items()}
    energy_key = next(k for k in perturbed if k.endswith("s1/energy"))
    perturbed[energy_key] = perturbed[energy_key] * (1 + 1e-12)
    report = osn.compare_snapshots(first, perturbed)
    assert not report["ok"] and report["exceeding"] == [energy_key]


# ----------------------------------------------- widened snapshot: logic, no TF


def test_snapshot_specs_label_every_group_and_name_the_options():
    specs, skipped = osn.snapshot_specs(osn.GROUPS, ("model_grace_2L_omat.yaml",))
    by_label = {spec.label: spec for spec in specs}
    assert set(by_label) == {
        "model_grace_2L_omat",
        "model_grace_2L_omat.f32",
        "model_grace_2L_omat.lm_first",
        "model_grace_2L_omat.dense",
        "preset.LINEAR",
        "preset.FS",
        "preset.GRACE_1LAYER_v2_25",
        "preset.GRACE_2LAYER_v2_25",
    }
    assert skipped == {}
    assert by_label["model_grace_2L_omat"].dtype == "float64"
    assert by_label["model_grace_2L_omat"].option is None
    assert by_label["model_grace_2L_omat.f32"].dtype == "float32"
    assert by_label["model_grace_2L_omat.lm_first"].option == "lm_first"
    assert by_label["model_grace_2L_omat.dense"].option == "dense_nbr"
    assert by_label["preset.FS"].source == "preset:FS"
    assert by_label["preset.FS"].dtype == "float64"


def test_snapshot_specs_records_unsupported_combinations_instead_of_dropping_them():
    specs, skipped = osn.snapshot_specs(("lm_first",), osn.DEFAULT_YAMLS)
    assert [s.label for s in specs] == [
        "model_grace_2L_omat.lm_first",
        "model_grace_2L_omat_large_base.lm_first",
    ]
    assert list(skipped) == ["model_grace.lm_first"]
    assert "MLPOut2ScalarTarget" in skipped["model_grace.lm_first"]


def test_snapshot_specs_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="unknown group 'bogus'"):
        osn.snapshot_specs(("base", "bogus"))


def test_model_spec_is_immutable():
    spec = osn.ModelSpec("a", "b.yaml")
    with pytest.raises(AttributeError):
        spec.label = "c"  # ty: ignore[invalid-assignment]


def test_precision_rows_are_named_and_select_by_label():
    assert set(osn.TOLERANCE_ROWS) == {"float32", "float64"}
    assert osn.precision_of("model_grace.f32/s0/energy") == "float32"
    assert osn.precision_of("model_grace/s0/energy") == "float64"
    assert osn.precision_of("preset.FS/dimer/data/B") == "float64"
    assert osn.precision_of("model_grace.dense/s0/energy") == "float64"
    for row in osn.TOLERANCE_ROWS.values():
        assert row.scale_rtol >= 0 and row.rationale
    assert osn.scale_rtol_for_key("m/s0/energy") == osn.DEFAULT_SCALE_RTOL == 1e-12
    assert (
        osn.scale_rtol_for_key("m.f32/s0/energy")
        == osn.TOLERANCE_ROWS["float32"].scale_rtol
    )


def test_tolerance_row_is_chosen_per_key_from_the_table_given():
    rows = {
        "float32": osn.ScaleTolerance(1e-3, "test row"),
        "float64": osn.ScaleTolerance(1e-9, "test row"),
    }
    assert osn.scale_rtol_for_key("m.f32/s0/x", rows) == 1e-3
    assert osn.scale_rtol_for_key("m/s0/x", rows) == 1e-9
    noise = 1e-5  # between the two rows
    a = _snap(**{"m.f32/s0/x": [1.0, 2.0], "m/s0/x": [1.0, 2.0]})
    b = _snap(**{"m.f32/s0/x": [1.0, 2.0 + 2 * noise], "m/s0/x": [1.0, 2.0 + 2 * noise]})

    def by_key(key):
        return osn.scale_rtol_for_key(key, rows)

    report = osn.compare_snapshots(a, b, scale_rtol=by_key)
    assert report["exceeding"] == ["m/s0/x"]  # the float64 key only
    assert osn.compare_snapshots(a, b, scale_rtol=1e-3)["ok"]


def test_float32_row_is_exact_because_no_repeat_spread_was_measured():
    row = osn.TOLERANCE_ROWS["float32"]
    assert row.scale_rtol == 0.0
    assert "exactly 0" in row.rationale
    a = _snap(**{"m.f32/s0/x": [1.0, 2.0]})
    b = _snap(**{"m.f32/s0/x": [1.0, 2.0 * (1 + 2**-23)]})  # one float32 ulp
    assert not osn.compare_snapshots(a, b, scale_rtol=osn.scale_rtol_for_key)["ok"]


def test_repeat_spread_of_identical_repeats_is_zero_and_groups_by_precision():
    a = _snap(**{"m/s0/energy": [1.0], "m.f32/s0/energy": [2.0], "m/s0/data/B": [3.0]})
    report = osn.repeat_spread([a, dict(a), dict(a)])
    assert set(report) == {"float64/energy", "float32/energy", "float64/data/B"}
    assert all(entry["max_scaled_spread"] == 0.0 for entry in report.values())


def test_repeat_spread_is_range_over_first_array_maximum():
    runs = [
        _snap(**{"m.f32/s0/forces": [4.0, 0.0], "m/s0/forces": [4.0, 0.0]}),
        _snap(**{"m.f32/s0/forces": [4.0, 0.5], "m/s0/forces": [4.0, 0.0]}),
        _snap(**{"m.f32/s0/forces": [4.0 - 1.0, 0.25], "m/s0/forces": [4.0, 0.0]}),
    ]
    report = osn.repeat_spread(runs)
    # range 1.0 in the first element, scale max|first| = 4.0
    assert report["float32/forces"]["max_scaled_spread"] == 0.25
    assert report["float32/forces"]["worst_key"] == "m.f32/s0/forces"
    assert report["float64/forces"]["max_scaled_spread"] == 0.0


def test_repeat_spread_takes_the_maximum_over_cases_and_keeps_the_worst_key():
    runs = [
        _snap(**{"m/s0/energy": [10.0], "m/s1/energy": [10.0]}),
        _snap(**{"m/s0/energy": [10.5], "m/s1/energy": [12.0]}),
    ]
    entry = osn.repeat_spread(runs)["float64/energy"]
    assert entry["max_scaled_spread"] == 0.2
    assert entry["worst_key"] == "m/s1/energy"


def test_repeat_spread_ignores_meta_and_keys_missing_from_a_repeat():
    meta = np.array(json.dumps({}))
    runs = [
        {osn.META_KEY: meta, "m/s0/energy": np.array([1.0]), "m/s0/only_first": np.array([1.0])},
        {osn.META_KEY: meta, "m/s0/energy": np.array([1.0])},
    ]
    assert set(osn.repeat_spread(runs)) == {"float64/energy"}


def test_repeat_spread_of_a_zero_scale_array_is_one_when_it_moves():
    runs = [_snap(**{"m/s0/z": [0.0, 0.0]}), _snap(**{"m/s0/z": [0.0, 1e-30]})]
    assert osn.repeat_spread(runs)["float64/z"]["max_scaled_spread"] == 1.0


def test_repeat_spread_needs_two_snapshots():
    with pytest.raises(ValueError, match="at least two"):
        osn.repeat_spread([_snap(x=[1.0])])


def test_cli_spread_prints_the_report(tmp_path, capsys):
    paths = []
    for index, value in enumerate((1.0, 1.0, 1.5)):
        path = tmp_path / f"run{index}.npz"
        np.savez(path, **{"m.f32/s0/energy": np.array([value])})
        paths.append(str(path))
    assert osn.main(["spread", *paths]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["float32/energy"]["max_scaled_spread"] == pytest.approx(0.5)


def test_cli_compare_uses_the_float32_row_for_float32_keys(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(first, **{"m.f32/s0/x": np.array([1.0, 1.0]), "m/s0/x": np.array([1.0, 1.0])})
    noise = 1e-14  # inside the float64 row, outside the exact float32 row
    np.savez(second, **{"m.f32/s0/x": np.array([1.0, 1.0]), "m/s0/x": np.array([1.0, 1.0 + noise])})
    assert osn.main(["compare", str(first), str(second)]) == 0
    np.savez(second, **{"m.f32/s0/x": np.array([1.0, 1.0 + noise]), "m/s0/x": np.array([1.0, 1.0])})
    assert osn.main(["compare", str(first), str(second)]) == 1
    assert osn.main(["compare", str(first), str(second), "--scale-rtol", "1e-12"]) == 0


def test_cli_write_rejects_an_unknown_group(capsys):
    with pytest.raises(SystemExit):
        osn.main(["write", "x.npz", "--groups", "bogus"])
    assert "invalid choice" in capsys.readouterr().err


def test_as_list_accepts_lists_and_dictionaries():
    assert osn._as_list([1, 2]) == [1, 2]
    assert osn._as_list({"a": 1, "b": 2}) == [1, 2]


def test_entries_accepts_the_three_yaml_layouts():
    entry = {"name": "a"}
    assert osn._entries([entry]) == [entry]
    assert osn._entries({"a": entry}) == [entry]
    assert osn._entries({"metadata": {}, "instructions": {"a": entry}}) == [entry]


def test_edge_structures_have_the_properties_their_names_promise():
    edge = osn.edge_structures()
    assert list(edge) == ["isolated", "dimer", "slab", "selfimage"]
    isolated, dimer, slab, selfimage = (edge[k] for k in edge)
    assert len(isolated) == 1 and not isolated.pbc.any()
    assert len(dimer) == 2 and not dimer.pbc.any()
    assert dimer.get_distance(0, 1) < 6.0  # inside the cutoff of the test models
    assert slab.pbc.tolist() == [True, True, False]
    assert slab.cell.lengths()[2] > 2 * 6.0  # vacuum wider than twice the cutoff
    assert selfimage.pbc.all() and len(selfimage) == 2
    assert selfimage.cell.lengths().max() < 6.0  # an atom sees its own images
    # generic positions: no symmetry may make the forces vanish
    fractional = selfimage.get_scaled_positions()
    assert not np.allclose(fractional[1], 0.5)
    for atoms in edge.values():
        assert {*atoms.get_chemical_symbols()} <= {"Mo", "Nb", "Ta", "W"}


def test_describe_structure_records_what_the_edge_cases_are():
    slab = osn.edge_structures()["slab"]
    described = osn.describe_structure(slab)
    assert described["n_atoms"] == 4
    assert described["formula"] == "Nb4"
    assert described["pbc"] == [True, True, False]
    assert described["cell"][2][2] == 20.0
    json.dumps(described)  # the metadata is JSON


# --------------------------------------- widened snapshot: physics and options, TF

TOLERANCES.update(
    {
        # layout options reorder sums only: float64 round-off (a few ulp)
        "layout_parity": (1e-10, 1e-12),
        # float32 parameters against float64: about 1e-7 relative per operation,
        # a few tens of operations between the parameters and the energy
        "float32_vs_float64": (1e-5, 1e-6),
        "fd_selfimage_forces": (1e-4, 1e-8),
        "fd_selfimage_stress": (1e-4, 1e-9),
        "newton_third_law": (0.0, 1e-12),
    }
)
OPTION_MODEL = "model_grace_2L_omat.yaml"


@pytest.fixture(scope="module")
def cases():
    _require_tensorflow()
    structures = osn.load_structures(osn.TESTS / osn.DEFAULT_STRUCTURES, 2)
    return {"s0": structures[0], "s1": structures[1], **osn.edge_structures()}


@pytest.fixture(scope="module")
def reference(tf_model, cases):
    return {name: osn.evaluate(tf_model, atoms) for name, atoms in cases.items()}


@pytest.mark.parametrize("option", ["lm_first", "dense_nbr"])
def test_layout_options_leave_the_physical_outputs_unchanged(option, reference, cases):
    model = osn.build_model(osn.TESTS / OPTION_MODEL, option=option)
    for name, atoms in cases.items():
        got = osn.evaluate(model, atoms)
        for quantity in ("energy", "forces", "stress"):
            _assert_close(got[quantity], reference[name][quantity], "layout_parity")


def test_with_layout_option_switches_the_option_on_where_it_is_accepted():
    _require_tensorflow()
    from tensorpotential.instructions import load_instructions

    plain = osn._as_list(load_instructions(str(osn.TESTS / OPTION_MODEL)))
    accepting = {i.name for i in plain if osn._accepts_option(i, "lm_first")}
    assert accepting and not all(getattr(i, "lm_first", False) for i in plain)
    switched = osn.with_layout_option(
        load_instructions(str(osn.TESTS / OPTION_MODEL)), "lm_first"
    )
    flags = {i.name: getattr(i, "lm_first", False) for i in osn._as_list(switched)}
    assert {name for name, flag in flags.items() if flag} == accepting
    assert [i.name for i in osn._as_list(switched)] == [i.name for i in plain]


def test_with_layout_option_changes_the_layout_of_the_stored_tensors(reference, cases):
    model = osn.build_model(osn.TESTS / OPTION_MODEL, option="lm_first")
    got = osn.evaluate(model, cases["s0"])
    changed = [
        key
        for key, value in got.items()
        if key.startswith("data/")
        and key in reference["s0"]
        and value.shape != reference["s0"][key].shape
    ]
    assert changed, "lm_first must move the lm axis of some stored tensor"


def test_with_layout_option_rejects_an_option_nothing_takes():
    _require_tensorflow()
    linear = osn.preset_instructions("LINEAR")
    assert not any(osn._accepts_option(i, "dense_nbr") for i in osn._as_list(linear))
    with pytest.raises(ValueError, match="dense_nbr"):
        osn.with_layout_option(linear, "dense_nbr")


def test_dense_layout_is_active_only_with_the_dense_option(tf_model):
    dense = osn.build_model(osn.TESTS / OPTION_MODEL, option="dense_nbr")
    assert osn.dense_layout_active(dense)
    assert not osn.dense_layout_active(tf_model)


def test_float32_model_has_float32_parameters_with_the_seeded_values():
    _require_tensorflow()
    model = osn.build_model(osn.TESTS / MODEL, dtype="float32")
    trainable = model.trainable_variables
    assert {v.dtype.name for v in trainable if v.dtype.is_floating} == {"float32"}
    var = trainable[0]
    expected = osn.seeded_values(var.name, tuple(var.shape)).astype(np.float32)
    assert np.array_equal(var.numpy(), expected)


def test_float32_outputs_agree_with_float64_to_single_precision(reference, cases):
    model = osn.build_model(osn.TESTS / OPTION_MODEL, dtype="float32")
    for name, atoms in cases.items():
        got = osn.evaluate(model, atoms)
        for quantity in ("energy", "forces", "stress"):
            _assert_close(got[quantity], reference[name][quantity], "float32_vs_float64")
    # float32 is not float64 in disguise: the energy must actually differ
    got = osn.evaluate(model, cases["s1"])
    assert np.any(got["energy"] != reference["s1"]["energy"])


def test_isolated_atom_has_only_the_dummy_bond_and_no_energy_forces_or_stress(reference):
    out = reference["isolated"]
    assert out["data/n_neigh_real"].item() == 1  # the builder's placeholder bond
    assert np.linalg.norm(out["data/bond_vector"][0]) > 6.0  # beyond the cutoff
    assert out["energy"].item() == 0.0
    assert not out["forces"].any() and not out["stress"].any()


def test_dimer_forces_obey_newton_and_point_along_the_bond(reference):
    out = reference["dimer"]
    forces = out["forces"]
    assert np.abs(forces).max() > 1e-3
    _assert_close(forces[0] + forces[1], np.zeros(3), "newton_third_law")
    _assert_close(forces[:, :2], np.zeros((2, 2)), "newton_third_law")
    assert not out["stress"].any()  # no cell: no stress
    assert out["virial"].reshape(6)[2] != 0.0  # the bond lies along z
    assert not np.delete(out["virial"].reshape(6), 2).any()


def test_evaluate_leaves_the_structure_unchanged(tf_model, cases):
    for name in ("dimer", "isolated"):
        before = cases[name].copy()
        osn.evaluate(tf_model, cases[name])
        assert cases[name] == before  # no cell, no centring, same periodicity
        assert cases[name].cell.rank == 0 and not cases[name].pbc.any()


def test_selfimage_cell_has_bonds_from_an_atom_to_its_own_image(reference):
    out = reference["selfimage"]
    assert out["data/ind_i"].shape[0] > 0
    assert np.any(out["data/ind_i"] == out["data/ind_j"])


def test_selfimage_forces_and_stress_match_finite_differences(tf_model, cases):
    atoms = cases["selfimage"]
    out = osn.evaluate(tf_model, atoms)
    assert np.abs(out["forces"]).max() > 1e-3
    for axis in range(3):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[1, axis] += STEP
        minus.positions[1, axis] -= STEP
        fd = -(_energy(tf_model, plus) - _energy(tf_model, minus)) / (2 * STEP)
        _assert_close(out["forces"][1, axis], fd, "fd_selfimage_forces")
    for axis in range(3):  # diagonal strain
        energies = []
        for sign in (+1, -1):
            eps = np.zeros((3, 3))
            eps[axis, axis] = sign * STEP
            strained = atoms.copy()
            strained.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
            energies.append(_energy(tf_model, strained))
        fd = (energies[0] - energies[1]) / (2 * STEP) / atoms.get_volume()
        _assert_close(out["stress"][axis], fd, "fd_selfimage_stress")


def test_slab_energy_is_invariant_to_a_shift_along_the_periodic_axes(tf_model, cases):
    base = osn.evaluate(tf_model, cases["slab"])
    moved = cases["slab"].copy()
    moved.translate([0.61, 1.17, 0.0])
    moved.wrap()
    shifted = osn.evaluate(tf_model, moved)
    _assert_close(shifted["energy"], base["energy"], "translation_energy")
    _assert_close(shifted["forces"], base["forces"], "translation_forces")


@pytest.mark.parametrize("name", list(osn.PRESET_SETTINGS))
def test_presets_build_and_are_translation_invariant(name, cases):
    _require_tensorflow()
    model = osn.build_spec_model(osn.ModelSpec(f"preset.{name}", osn.PRESET_PREFIX + name))
    atoms = cases["s1"]
    base = osn.evaluate(model, atoms)
    assert np.isfinite(base["energy"]).all() and np.abs(base["forces"]).max() > 1e-6
    moved = atoms.copy()
    moved.translate([0.37, -1.21, 2.05])
    moved.wrap()
    shifted = osn.evaluate(model, moved)
    _assert_close(shifted["energy"], base["energy"], "translation_energy")
    _assert_close(shifted["forces"], base["forces"], "translation_forces")


def test_take_snapshot_records_variants_edge_cases_and_the_skipped_combinations():
    _require_tensorflow()
    yamls = ("model_grace.yaml",)
    arrays = osn.take_snapshot(
        yamls, n_structures=1, groups=("base", "float32", "lm_first", "dense")
    )
    meta = json.loads(str(arrays[osn.META_KEY]))
    labels = {k.split("/")[0] for k in arrays if k != osn.META_KEY}
    assert labels == {"model_grace", "model_grace.f32", "model_grace.dense"}
    assert list(meta["skipped"]) == ["model_grace.lm_first"]
    assert set(meta["structures"]) == {"s0", "isolated", "dimer", "slab", "selfimage"}
    assert {s["label"] for s in meta["specs"]} == labels
    for label in labels:
        assert f"{label}/selfimage/energy" in arrays
        assert f"{label}/s0/data/bond_vector" in arrays
    assert set(meta["variables"]) == labels
    f32 = {dtype for _, _, dtype in meta["variables"]["model_grace.f32"]}
    assert "float32" in f32
    assert meta["yaml_sha256"]["model_grace"] == hashlib.sha256(
        (osn.TESTS / "model_grace.yaml").read_bytes()
    ).hexdigest()


def test_take_snapshot_of_presets_has_no_yaml_hash():
    _require_tensorflow()
    arrays = osn.take_snapshot((), n_structures=1, groups=("presets",), edge=False)
    meta = json.loads(str(arrays[osn.META_KEY]))
    assert meta["yaml_sha256"] == {}
    assert {k.split("/")[0] for k in arrays if k != osn.META_KEY} == {
        f"preset.{name}" for name in osn.PRESET_SETTINGS
    }
