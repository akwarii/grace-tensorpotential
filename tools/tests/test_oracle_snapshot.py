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
    w = osn.seeded_values("big", (400, 400))
    assert w.shape == (400, 400)
    assert abs(w.mean()) < 0.01
    assert w.std() == pytest.approx(1 / np.sqrt(400), rel=0.05)
    vec = osn.seeded_values("vec", (50_000,))
    assert vec.mean() == pytest.approx(1.0, abs=0.01)
    assert vec.std() == pytest.approx(0.1, rel=0.05)
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
    assert report["max_abs_diff"] == pytest.approx(0.5)
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
    assert report["max_scaled_diff"] == pytest.approx(1e-14, rel=1e-6, abs=1e-20)
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
    args = ["write", str(out), "--yamls", "model_grace.yaml", "--n-structures", "1"]
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
    assert float(cutoff.numpy()) == pytest.approx(6.0)
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
    first = osn.take_snapshot(yamls, n_structures=2)
    second = osn.take_snapshot(yamls, n_structures=2)
    assert json.loads(str(first[osn.META_KEY]))["structures"][1]["n_atoms"] == 16
    assert any(k.endswith("/energy") for k in first)
    assert osn.compare_snapshots(first, second)["ok"]
    assert osn.compare_snapshots(first, second)["max_abs_diff"] == 0.0

    perturbed = {k: v.copy() for k, v in second.items()}
    energy_key = next(k for k in perturbed if k.endswith("s1/energy"))
    perturbed[energy_key] = perturbed[energy_key] * (1 + 1e-12)
    report = osn.compare_snapshots(first, perturbed)
    assert not report["ok"] and report["exceeding"] == [energy_key]
