"""Tests for tools/make_golden.py.

Logic tests: the tier tables, the derived yamls (only widths and elements change, no option cell is lost),
the species relabelling, the seeded weights, the comparison and size rules, the option-pair relations, and
that ``verify`` and ``compare`` catch a planted corruption. Physical-value tests run the tiny models of the
generator and check them against oracles that do not use the generator's own code: a central finite
difference of the energy for the forces and the stress (ASE sign convention), rotation, translation and
permutation of the structure, Newton's third law, the virial rebuilt from the pair forces with numpy, the
yaml's own scale for ``ConstantScaleShiftTarget``, the stored shifts of ``TrainableShiftTarget``, and the dtype
chain documented in the class sheets (the bond functions are float64 in a float32 model, the rest float32).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import tensorpotential  # noqa: E402, F401, I001 - before TensorFlow: it selects the legacy Keras backend
import make_golden as mg  # noqa: E402
from tests.shared_models import weights_fingerprint  # noqa: E402
from tests.tolerances import (  # noqa: E402
    COVARIANCE_F64,
    FINITE_DIFFERENCE_F64,
    FLOAT64_ARITHMETIC,
    SOFTENED_UNIT_VECTOR_F64,
    sample_std_rtol,
)

pytest.importorskip("tensorpotential")
tf = pytest.importorskip("tensorflow")
from ase import Atoms  # noqa: E402

# One TensorFlow configuration for every test module of tools/ (one thread, deterministic ops).
mg.osn.configure_tensorflow()

SCALE_OMAT = 1.892985414710868
"""``ConstantScaleShiftTarget.scale`` of ``tests/model_grace_2L_omat.yaml``, read off the yaml by hand."""
FD_STEP = 1e-4
CORE_CASES = ("isolated_atom", "dimer", "self_image_cell")


def _isclose(actual, expected, tolerance) -> bool:
    return bool(np.allclose(actual, expected, rtol=tolerance.rtol, atol=tolerance.atol))


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    """The four omat tiny fixtures (float64, float32, and the two option pairs) in a scratch directory."""
    folder = tmp_path_factory.mktemp("golden")
    mg.write_tier("tiny", folder, models=["omat"])
    return folder


@pytest.fixture(scope="module")
def omat_model(tiny_dir):
    """The tiny omat model (float64), built from the yaml next to the fixtures; read-only for the tests."""
    model = mg.build_model(tiny_dir / "yamls" / "omat_tiny.yaml", "float64")
    fingerprint = weights_fingerprint(model)
    yield model
    assert weights_fingerprint(model) == fingerprint, "a test changed the shared model"


@pytest.fixture(scope="module")
def large_model(tiny_dir):
    """The tiny large model (float64): the one with ``TrainableShiftTarget`` and the RMS norms."""
    mg.write_yamls(["tiny"], tiny_dir)
    model = mg.build_model(tiny_dir / "yamls" / "large_tiny.yaml", "float64")
    fingerprint = weights_fingerprint(model)
    yield model
    assert weights_fingerprint(model) == fingerprint, "a test changed the shared model"


@pytest.fixture(scope="module")
def cases() -> dict:
    return mg.tier_cases(mg.TIERS["tiny"])


def _evaluate(model, atoms) -> dict[str, np.ndarray]:
    return mg.evaluate_case(model, atoms)


# ------------------------------------------------------------------ logic: tiers and yamls


def test_tier_structures_and_aliases_are_consistent():
    from tests_torch.structures.build_structures import load_structures

    every = load_structures()
    for tier in mg.TIERS.values():
        assert set(tier.structures) <= set(every)
        assert set(tier.aliases.values()) <= set(tier.elements)
        assert len(set(tier.structures)) == len(tier.structures)
    assert 3 <= len(mg.TIERS["tiny"].elements) <= 4
    assert 4 <= len(mg.TIERS["faithful"].elements) <= 6


def test_tiny_yaml_changes_widths_and_elements_only():
    parent = mg.TESTS / mg.MODELS["omat"]
    raw, tiny = (
        yaml.safe_load(parent.read_text()),
        mg.derive_yaml(parent, mg.TIERS["tiny"]),
    )
    assert list(raw) == list(tiny)
    for name, entry in raw.items():
        derived = tiny[name]
        assert derived["__cls__"] == entry["__cls__"]
        unchanged = {
            k
            for k in entry
            if k
            not in (
                *mg.WIDTH_KEYS,
                "element_map",
                "number_of_atom_types",
                "hidden_layers",
            )
        }
        assert {k: derived[k] for k in unchanged} == {k: entry[k] for k in unchanged}
    assert tiny["Z"]["element_map"] == {"Cu": 0, "H": 1, "Mg": 2, "O": 3}
    assert tiny["Z"]["embedding_size"] == 6
    assert tiny["I0"]["number_of_atom_types"] == 4  # 89 in the parent
    assert tiny["R"]["hidden_layers"] == [6, 7]  # distinct widths: an axis mix-up fails
    assert raw["R"]["hidden_layers"] == [64, 64]  # the parent dictionary is untouched


def test_faithful_yaml_changes_only_the_elements():
    for parent_name in mg.MODELS.values():
        parent = mg.TESTS / parent_name
        raw, faithful = (
            yaml.safe_load(parent.read_text()),
            mg.derive_yaml(parent, mg.TIERS["faithful"]),
        )
        for name, entry in raw.items():
            rest = {
                k: v
                for k, v in entry.items()
                if k not in ("element_map", "number_of_atom_types")
            }
            assert {
                k: v
                for k, v in faithful[name].items()
                if k not in ("element_map", "number_of_atom_types")
            } == rest
        assert faithful["Z"]["element_map"] == {
            s: i for i, s in enumerate(sorted(mg.TIERS["faithful"].elements))
        }


def test_derived_yaml_text_is_stable():
    parent = mg.TESTS / mg.MODELS["large"]
    text = mg.yaml_text(mg.derive_yaml(parent, mg.TIERS["tiny"]))
    assert text == mg.yaml_text(mg.derive_yaml(parent, mg.TIERS["tiny"]))
    assert yaml.safe_load(text) == mg.derive_yaml(parent, mg.TIERS["tiny"])


@pytest.mark.parametrize("model", list(mg.MODELS))
def test_tiny_yaml_exercises_every_option_cell_of_its_parent(model, tmp_path):
    (path,) = [
        p for p in mg.write_yamls(["tiny"], tmp_path) if p.name.startswith(model)
    ]
    assert mg.missing_cells(mg.TESTS / mg.MODELS[model], path) == []


def test_missing_cells_names_a_lost_option_value(tmp_path):
    parent = mg.TESTS / mg.MODELS["omat"]
    raw = mg.derive_yaml(parent, mg.TIERS["tiny"])
    raw["A1"]["left_coefs"] = True  # the parent has True only: no loss
    for name in (
        "R",
        "R1",
    ):  # the parent's only "tanh" radial functions: the value is gone
        raw[name]["activation"] = "silu"
    path = tmp_path / "lost.yaml"
    path.write_text(mg.yaml_text(raw))
    lost = mg.missing_cells(parent, path)
    assert any(
        cell[:3]
        == (
            "tensorpotential.instructions.compute.MLPRadialFunction",
            "activation",
            "str",
        )
        and cell[3] == "tanh"
        for cell in lost
    )


def test_relabel_keeps_geometry_and_maps_species(cases):
    from tests_torch.structures.build_structures import load_structures

    tier = mg.TIERS["tiny"]
    original = load_structures()["self_image_cell"]
    relabelled = mg.relabel(original, tier)
    assert set(original.get_chemical_symbols()) == {"Au", "Cu"}
    expected = [
        "H" if symbol == "Au" else symbol for symbol in original.get_chemical_symbols()
    ]
    assert relabelled.get_chemical_symbols() == expected  # Au -> H, Cu stays
    assert np.array_equal(relabelled.positions, original.positions)
    assert np.array_equal(relabelled.cell.array, original.cell.array)
    assert set(cases) == set(tier.structures)


def test_relabel_rejects_an_unknown_species():
    atoms = Atoms("Xe", positions=[[0, 0, 0]])
    with pytest.raises(ValueError, match="Xe"):
        mg.relabel(atoms, mg.TIERS["tiny"])


def test_tier_cases_stay_inside_the_element_set(cases):
    for atoms in cases.values():
        assert set(atoms.get_chemical_symbols()) <= set(mg.TIERS["tiny"].elements)


def test_requests_cover_both_dtypes_and_both_option_pairs():
    names = [r.name for r in mg.requests_for("tiny")]
    assert len(names) == len(set(names)) == 8
    assert "omat_tiny_f32" in names
    assert "large_tiny_f64_dense" in names
    assert "large_tiny_f64_lm_first" in names
    assert all(r.dtype == "f64" for r in mg.requests_for("tiny") if r.option)


# ------------------------------------------------------------------ logic: seeds, files, comparisons


def test_seeded_weight_is_deterministic_and_keyed():
    first = mg.seeded_weight("A/w", (5, 3))
    assert np.array_equal(first, mg.seeded_weight("A/w", (5, 3)))
    assert not np.array_equal(first, mg.seeded_weight("B/w", (5, 3)))
    assert not np.array_equal(first, mg.seeded_weight("A/w", (5, 3), seed=1))


def test_seeded_weight_follows_the_documented_distributions():
    matrix = mg.seeded_weight("big/w", (400, 300))
    assert abs(matrix.mean()) < 5 * matrix.std() / np.sqrt(matrix.size)
    assert matrix.std() == pytest.approx(
        1 / np.sqrt(400), rel=sample_std_rtol(matrix.size)
    )
    vector = mg.seeded_weight("big/scale", (50_000,))
    assert vector.mean() == pytest.approx(1.0, abs=5 * 0.1 / np.sqrt(vector.size))
    assert vector.std() == pytest.approx(0.1, rel=sample_std_rtol(vector.size))
    assert mg.seeded_weight("s", ()).shape == ()


def test_npz_round_trip_refuses_pickled_objects(tmp_path):
    arrays = {"b": np.arange(3), "a": np.ones((2, 2)), "s": np.array(["Cu", "H"])}
    path = tmp_path / "x.npz"
    mg.save_npz(path, arrays)
    back = mg.load_npz(path)
    assert sorted(back) == ["a", "b", "s"]
    assert np.array_equal(back["s"], arrays["s"])
    objects = tmp_path / "o.npz"
    np.savez(objects, o=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        mg.load_npz(objects)


def test_compare_arrays_uses_the_scaled_tolerance_and_text_equality():
    a = {"x": np.array([1.0, 2.0, 1e-3]), "s": np.array(["Cu", "H"])}
    near = {"x": np.array([1.0, 2.0, 1e-3 + 1e-14]), "s": np.array(["Cu", "H"])}
    far = {"x": np.array([1.0, 2.0, 1e-3 + 1e-9]), "s": np.array(["Cu", "H"])}
    other = {"x": a["x"], "s": np.array(["Cu", "O"])}
    assert mg.compare_arrays(a, near, 1e-12)["ok"]
    assert mg.compare_arrays(a, far, 1e-12)["exceeding"] == ["x"]
    assert mg.compare_arrays(a, other, 1e-12)["exceeding"] == ["s"]
    assert not mg.compare_arrays(a, {"x": a["x"]}, 1e-12)["ok"]


def test_reload_tolerance_rows_are_the_named_ones():
    assert mg.reload_scale_rtol("f64") == mg.osn.TOLERANCE_ROWS["float64"].scale_rtol
    assert mg.reload_scale_rtol("f32") == mg.osn.TOLERANCE_ROWS["float32"].scale_rtol


def test_relation_distinguishes_equal_lm_first_and_different():
    rng = np.random.default_rng(0)
    default = rng.normal(size=(3, 4, 5))
    assert mg._relation(default, default.copy()) == "equal"  # noqa: SLF001
    assert mg._relation(default, np.moveaxis(default, -1, 0)) == "lm_axis_first"  # noqa: SLF001
    assert mg._relation(default, default[..., ::-1]) == "differs"  # noqa: SLF001
    assert mg._relation(default, default[:2]) == "differs"  # noqa: SLF001
    assert mg._relation(np.zeros(()), np.zeros(())) == "equal"  # noqa: SLF001


def test_size_rules_of_the_two_tiers():
    tiny = {
        "tier": "tiny",
        "name": "t",
        "files": {"a": {"name": "t.npz", "bytes": 600_000}},
    }
    assert mg.size_violations(tiny, 100_000) == []
    assert mg.size_violations(tiny, 500_000) == ["t"]
    big = {
        "tier": "faithful",
        "name": "f",
        "files": {
            "a": {"name": "f.npz", "bytes": 24_999_999},
            "w": {"name": "f.weights.npz", "bytes": 25_000_000},
        },
    }
    assert mg.size_violations(big) == ["f.weights.npz"]


def test_report_ok_reads_every_part_of_a_report():
    good = {
        "only_in_first": [],
        "only_in_second": [],
        "shape_mismatch": [],
        "exceeding": [],
    }
    assert mg.report_ok(good)
    assert not mg.report_ok({**good, "exceeding": ["k"]})
    assert not mg.report_ok({**good, "weights": {**good, "shape_mismatch": ["w"]}})
    assert mg.report_ok({
        "only_in_first": [],
        "only_in_second": [],
        "fixtures": {"a": good},
    })
    assert not mg.report_ok({
        "only_in_first": ["b"],
        "only_in_second": [],
        "fixtures": {},
    })


# ------------------------------------------------------------------ weights and instructions


def test_every_trainable_float_variable_is_nonzero(omat_model, large_model):
    # several layers start at zero (InvariantLayerRMSNorm.init, the output norms); a fixture of zeros proves nothing
    for model in (omat_model, large_model):
        for variable in model.trainable_variables:
            if variable.dtype.is_floating:
                assert np.any(variable.numpy() != 0), variable.name
    scales = {
        k: v for k, v in mg.weights_of(large_model).items() if k.endswith("/scale")
    }
    assert {"I_0_LN/scale", "I_nl_LN/scale"} <= set(scales)
    assert all(np.all(v != 0) for v in scales.values())


def test_weight_keys_are_unique_where_variable_names_are_not(large_model):
    names = [v.name for v in large_model.variables]
    keys = [r["key"] for r in mg.weight_table(large_model)]
    assert len(set(names)) < len(
        names
    )  # the two InvariantLayerRMSNorm scales are both Variable:0
    assert len(set(keys)) == len(keys) == len(names)
    assert all("/" in k and not k.startswith("model") for k in keys)


def test_the_cutoff_is_not_overwritten(omat_model):
    weights = mg.weights_of(omat_model)
    assert weights["RadialBasis/rc"] == pytest.approx(6.0)
    assert weights["Z/element_map_symbols"].tolist() == ["Cu", "H", "Mg", "O"]
    assert weights["Z/element_map_index"].tolist() == [0, 1, 2, 3]
    assert weights["Z/element_map_symbols"].dtype.kind == "U"


def test_assign_weights_round_trip_and_reports_mismatches(omat_model, tiny_dir):
    other = mg.build_model(tiny_dir / "yamls" / "omat_tiny.yaml", "float64", seed=7)
    weights = mg.weights_of(omat_model)
    assert not np.array_equal(weights["A1/w_left"], mg.weights_of(other)["A1/w_left"])
    report = mg.assign_weights(other, weights)
    assert report["assigned"] == len(weights)
    assert not any(
        report[k]
        for k in ("only_in_model", "only_in_weights", "shape_differs", "value_differs")
    )
    assert np.array_equal(mg.weights_of(other)["A1/w_left"], weights["A1/w_left"])

    damaged = dict(weights)
    damaged["A1/w_left"] = weights["A1/w_left"][:-1]
    damaged["Z/element_map_symbols"] = np.array(["Cu", "H", "Mg", "Xe"])
    damaged["extra/w"] = np.zeros(2)
    del damaged["Z/w"]
    report = mg.assign_weights(other, damaged)
    assert report["only_in_weights"] == ["extra/w"]
    assert report["only_in_model"] == ["Z/w"]
    assert list(report["shape_differs"]) == ["A1/w_left"]
    assert report["value_differs"] == ["Z/element_map_symbols"]
    assert np.array_equal(
        mg.weights_of(other)["A1/w_left"], weights["A1/w_left"]
    )  # left alone


def test_init_args_hold_the_resolved_constructor_arguments(omat_model):
    args = mg.init_args_of(omat_model)
    assert list(args) == [i.name for i in mg.gp._instructions_of(omat_model)]  # noqa: SLF001
    assert args["R"]["class"] == "MLPRadialFunction"
    assert args["R"]["init_args"]["hidden_layers"] == [6, 7]
    assert args["ConstantScaleShiftTarget"]["init_args"]["scale"] == pytest.approx(
        SCALE_OMAT
    )
    json.dumps(args)


def test_tables_are_stored_once_under_their_owner(omat_model):
    tables = mg.tables_of(omat_model)
    assert {
        "tables/AA/left_ind",
        "tables/AA/right_ind",
        "tables/AA/m_sum_ind",
        "tables/AA/cg",
    } <= set(tables)
    assert {
        "tables/A1/w_tile_left",
        "tables/A1/collect_from",
        "tables/A1/norm_map",
    } <= set(tables)
    # a nested reference (``left``, ``indicator``, ``radial``) is not entered: no path repeats another instruction's tables
    assert not any("/left/" in key or "/indicator/" in key for key in tables)
    assert len(tables) < 200


def test_tables_are_the_arrays_of_the_instruction(omat_model):
    tables = mg.tables_of(omat_model)
    instruction = next(i for i in mg.gp._instructions_of(omat_model) if i.name == "AA")  # noqa: SLF001
    for name in ("left_ind", "right_ind", "m_sum_ind", "cg"):
        assert np.array_equal(
            tables[f"tables/AA/{name}"], getattr(instruction, name).numpy()
        ), name


# ------------------------------------------------------------------ evaluation


def test_every_instruction_is_recorded_once(omat_model, cases):
    out = _evaluate(omat_model, cases["dimer"])
    names = [i.name for i in mg.gp._instructions_of(omat_model)]  # noqa: SLF001
    recorded = [k.split("/", 1)[1] for k in out if k.startswith(("ins/", "out_after/"))]
    assert sorted(recorded) == sorted(names)
    for output in ("MLPOut2ScalarTarget", "ConstantScaleShiftTarget"):
        assert f"out_before/{output}" in out
        assert f"ins/{output}" not in out


def test_the_energy_target_is_snapshotted_before_and_after_each_output_instruction(
    omat_model, cases
):
    out = _evaluate(omat_model, cases["self_image_cell"])
    initial = out["ins/atomic_energy"]
    assert float(initial) == 0.0  # CreateOutputTarget.initial_value
    assert np.array_equal(out["out_before/MLPOut2ScalarTarget"], initial)
    # the second output instruction reads what the first one wrote, and its result is the atomic energy of the model
    assert np.array_equal(
        out["out_before/ConstantScaleShiftTarget"],
        out["out_after/MLPOut2ScalarTarget"],
    )
    assert np.array_equal(
        out["out_after/ConstantScaleShiftTarget"], out["res/atomic_energy"]
    )
    assert not np.array_equal(
        out["out_before/ConstantScaleShiftTarget"],
        out["out_after/ConstantScaleShiftTarget"],
    )


def test_constant_scale_shift_applies_the_scale_of_the_yaml(omat_model, cases):
    out = _evaluate(omat_model, cases["self_image_cell"])
    expected = (
        SCALE_OMAT * out["out_before/ConstantScaleShiftTarget"] + 0
    )  # shift: 0 in the yaml
    assert _isclose(
        out["out_after/ConstantScaleShiftTarget"], expected, FLOAT64_ARITHMETIC
    )


def test_trainable_shift_adds_the_stored_shift_of_each_element(large_model, cases):
    out = _evaluate(large_model, cases["self_image_cell"])
    shifts = mg.weights_of(large_model)["TrainableShiftTarget/at_shifts"][:, 0]
    mu = out["in/atomic_mu_i"]
    expected = out["out_before/TrainableShiftTarget"][:, 0] + shifts[mu]
    assert _isclose(
        out["out_after/TrainableShiftTarget"][:, 0], expected, FLOAT64_ARITHMETIC
    )
    assert (
        len(set(mu.tolist())) == 2
    )  # the two atoms have different elements, so the gather is tested


def test_bond_functions_follow_their_formulas(omat_model, cases):
    out = _evaluate(omat_model, cases["self_image_cell"])
    r = out["in/bond_vector"]
    length = out["ins/BondLength"][:, 0]
    assert _isclose(
        length, np.sqrt((r**2).sum(1) + 1e-10), FLOAT64_ARITHMETIC
    )  # the 1e-10 softening
    assert _isclose(
        out["ins/ScaledBondVector"], r / length[:, None], FLOAT64_ARITHMETIC
    )
    # addition theorem of the spherical harmonics, whatever the real-basis convention: the sum over m of Y_lm^2
    # is (2l+1)/(4 pi) for orthonormal harmonics and 2l+1 for the 4 pi-scaled ones that GRACE uses (Y_00 = 1)
    y = out["ins/Y"]
    for degree in range(5):
        block = y[:, degree * degree : (degree + 1) ** 2]
        assert _isclose((block**2).sum(1), 2 * degree + 1, SOFTENED_UNIT_VECTOR_F64), (
            degree
        )


def test_embedding_output_is_the_stored_weight(omat_model, cases):
    out = _evaluate(omat_model, cases["dimer"])
    assert np.array_equal(out["ins/Z"], mg.weights_of(omat_model)["Z/w"])


def test_results_are_consistent_with_each_other(omat_model, cases):
    out = _evaluate(omat_model, cases["self_image_cell"])
    assert _isclose(
        out["res/energy"].sum(), out["res/atomic_energy"].sum(), FLOAT64_ARITHMETIC
    )
    # Newton's third law: the forces on a closed system sum to zero
    assert _isclose(out["res/forces"].sum(0), 0.0, FINITE_DIFFERENCE_F64)
    # the total force of an atom is the sum of the pair forces towards it minus those away from it
    i, j, pair = out["in/ind_i"], out["in/ind_j"], out["res/pair_f"]
    forces = np.zeros_like(out["res/forces"])
    np.add.at(forces, j, pair)
    np.subtract.at(forces, i, pair)
    assert _isclose(out["res/forces"], forces, FLOAT64_ARITHMETIC)
    # virial = sum over bonds of r (x) f, stored as xx, yy, zz, xy, xz, yz
    r = out["in/bond_vector"]
    matrix = np.einsum("ba,bc->ac", r, pair)
    expected = [
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 2],
    ]
    assert _isclose(out["res/virial"], expected, FLOAT64_ARITHMETIC)


def test_forces_equal_minus_the_finite_difference_of_the_energy(omat_model):
    atoms = _periodic_three()
    forces = _evaluate(omat_model, atoms)["res/forces"]
    numeric = np.zeros_like(forces)
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (+1, -1):
                shifted = atoms.copy()
                shifted.positions[atom, axis] += sign * FD_STEP
                energies.append(
                    float(_evaluate(omat_model, shifted)["res/energy"].sum())
                )
            numeric[atom, axis] = -(energies[0] - energies[1]) / (2 * FD_STEP)
    assert np.abs(forces).max() > 1e-5  # a non-trivial force
    assert _isclose(forces, numeric, FINITE_DIFFERENCE_F64)


def test_stress_equals_the_finite_difference_of_the_energy_under_strain(
    omat_model, cases
):
    atoms = cases["self_image_cell"]
    stress = _evaluate(omat_model, atoms)["res/stress"]  # Voigt: xx, yy, zz, yz, xz, xy
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    numeric = np.zeros(6)
    for k, (a, b) in enumerate(voigt):
        energies = []
        for sign in (+1, -1):
            strain = np.eye(3)
            strain[a, b] += sign * FD_STEP / (2 if a != b else 1)
            strain[b, a] = strain[a, b]
            strained = atoms.copy()
            strained.set_cell(atoms.cell.array @ strain, scale_atoms=True)
            energies.append(float(_evaluate(omat_model, strained)["res/energy"].sum()))
        numeric[k] = (energies[0] - energies[1]) / (2 * FD_STEP) / atoms.get_volume()
    assert np.abs(stress).max() > 1e-6
    assert _isclose(stress, numeric, FINITE_DIFFERENCE_F64)


def _periodic_three() -> Atoms:
    """Three atoms of three elements at generic positions of a small orthorhombic cell: forces and stress are not zero by symmetry."""
    return Atoms(
        "CuHO",
        scaled_positions=[[0.0, 0.0, 0.0], [0.41, 0.33, 0.12], [0.67, 0.78, 0.54]],
        cell=[3.1, 3.4, 3.7],
        pbc=True,
    )


def _three_atoms() -> Atoms:
    return Atoms(
        "CuHO",
        positions=[[0.0, 0.0, 0.0], [1.3, 0.4, -0.2], [-0.7, 1.9, 0.8]],
        pbc=False,
    )


def test_energy_is_invariant_and_forces_covariant_under_rotation_and_translation(
    omat_model,
):
    atoms = _three_atoms()
    base = _evaluate(omat_model, atoms)
    angle = 0.7
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T + np.array([5.0, -3.0, 2.0])
    out = _evaluate(omat_model, moved)
    assert _isclose(out["res/energy"], base["res/energy"], COVARIANCE_F64)
    assert _isclose(out["res/forces"], base["res/forces"] @ rotation.T, COVARIANCE_F64)
    assert np.abs(base["res/forces"]).max() > 1e-5


def test_energy_is_invariant_and_forces_follow_a_permutation_of_the_atoms(omat_model):
    atoms = _three_atoms()
    order = [2, 0, 1]
    base, out = _evaluate(omat_model, atoms), _evaluate(omat_model, atoms[order])
    assert _isclose(out["res/energy"], base["res/energy"], COVARIANCE_F64)
    assert _isclose(out["res/forces"], base["res/forces"][order], COVARIANCE_F64)
    assert _isclose(
        out["res/atomic_energy"], base["res/atomic_energy"][order], COVARIANCE_F64
    )


def test_an_isolated_atom_has_only_the_dummy_bond(omat_model, cases):
    out = _evaluate(omat_model, cases["isolated_atom"])
    assert out["in/bond_vector"].shape == (1, 3)
    assert np.linalg.norm(out["in/bond_vector"][0]) > 6.0  # far outside the cutoff
    assert _isclose(out["res/forces"], 0.0, FLOAT64_ARITHMETIC)


def test_a_float32_model_keeps_the_documented_dtype_chain(tiny_dir, cases):
    model = mg.build_model(tiny_dir / "yamls" / "omat_tiny.yaml", "float32")
    out = _evaluate(model, cases["self_image_cell"])
    for name in ("BondLength", "ScaledBondVector", "RadialBasis", "Y"):
        assert out[f"ins/{name}"].dtype == np.float64, name
    for name in ("R", "A", "A1", "AA", "I", "I_out", "YI", "B"):
        assert out[f"ins/{name}"].dtype == np.float32, name
    assert out["out_after/ConstantScaleShiftTarget"].dtype == np.float32
    assert out["res/energy"].dtype == np.float64
    assert out["in/bond_vector"].dtype == np.float64


# ------------------------------------------------------------------ fixtures


def test_fixture_manifests_are_complete_and_hashes_match(tiny_dir):
    for request in mg.requests_for("tiny", ["omat"]):
        manifest = json.loads((tiny_dir / f"{request.name}.json").read_text())
        assert manifest["name"] == request.name
        assert manifest["format_version"] == mg.FORMAT_VERSION
        assert manifest["seed"] == mg.SEED
        assert manifest["elements"] == ["Cu", "H", "Mg", "O"]
        # the order of the instructions is the order they run in: the manifest keeps the order of the yaml
        parent = yaml.safe_load((tiny_dir / "yamls" / manifest["yaml"]).read_text())
        assert list(manifest["instructions"]) == list(parent)
        assert list(manifest["structures"]) == list(mg.TIERS["tiny"].structures)
        assert isinstance(manifest["library"]["use_gemm_couple"], bool)
        assert {"tensorflow", "numpy", "pandas"} <= set(manifest["library"]["versions"])
        assert (
            manifest["library"]["library_git_sha"] is None
            or len(manifest["library"]["library_git_sha"]) == 40
        )
        for entry in manifest["files"].values():
            path = tiny_dir / entry["name"]
            assert path.stat().st_size == entry["bytes"]
            assert mg._sha256(path) == entry["sha256"]  # noqa: SLF001
        assert (
            mg.size_violations(
                manifest, (tiny_dir / f"{request.name}.json").stat().st_size
            )
            == []
        )


def test_an_option_pair_has_the_weights_of_its_default_fixture(tiny_dir):
    for option in mg.OPTIONS:
        manifest = json.loads((tiny_dir / f"omat_tiny_f64_{option}.json").read_text())
        assert manifest["weights_from"] == "omat_tiny_f64"
        assert "weights" not in manifest["files"]
        assert not (tiny_dir / f"omat_tiny_f64_{option}.weights.npz").exists()


def test_option_pairs_give_equal_results_and_equal_weight_shapes(tiny_dir):
    for option in mg.OPTIONS:
        pair = json.loads((tiny_dir / f"omat_tiny_f64_{option}.json").read_text())[
            "option_pair"
        ]
        weights = pair["weights"]
        assert (
            weights["shape_differs"] == {}
            and weights["only_in_model"] == []
            and weights["only_in_weights"] == []
        )
        assert weights["assigned"] > 30
        assert max(pair["results_max_abs_diff"].values()) < 1e-12
        assert pair["outputs"]["differs"] == [] and pair["outputs"]["missing"] == []
    dense = json.loads((tiny_dir / "omat_tiny_f64_dense.json").read_text())[
        "option_pair"
    ]
    assert dense["outputs"]["lm_axis_first"] == []
    lm_first = json.loads((tiny_dir / "omat_tiny_f64_lm_first.json").read_text())[
        "option_pair"
    ]
    assert any(k.endswith("/ins/A") for k in lm_first["outputs"]["lm_axis_first"])
    assert any(k.endswith("/ins/R") for k in lm_first["outputs"]["equal"])


def test_the_option_pair_stores_its_own_inputs(tiny_dir):
    default = mg.load_npz(tiny_dir / "omat_tiny_f64.npz")
    dense = mg.load_npz(tiny_dir / "omat_tiny_f64_dense.npz")
    assert "dimer/in/bond_vector" in dense
    assert default["dimer/in/bond_vector"].shape == dense["dimer/in/bond_vector"].shape


@pytest.mark.parametrize("name", [r.name for r in mg.requests_for("tiny", ["omat"])])
def test_reloading_a_fixture_reproduces_it(tiny_dir, name):
    report = mg.verify_fixture(tiny_dir, name)
    assert mg.report_ok(report), report
    assert report["n_compared"] > 200


def test_verify_catches_a_planted_corruption(tiny_dir, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(tiny_dir, folder)
    arrays = mg.load_npz(folder / "omat_tiny_f64.npz")
    arrays["dimer/res/energy"] = arrays["dimer/res/energy"] + 1e-6
    mg.save_npz(folder / "omat_tiny_f64.npz", arrays)
    report = mg.verify_fixture(folder, "omat_tiny_f64")
    assert report["exceeding"] == ["dimer/res/energy"]
    assert not mg.report_ok(report)


def test_verify_refuses_weights_that_do_not_fit_the_model(tiny_dir, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(tiny_dir, folder)
    weights = mg.load_npz(folder / "omat_tiny_f64.weights.npz")
    del weights["Z/w"]
    mg.save_npz(folder / "omat_tiny_f64.weights.npz", weights)
    with pytest.raises(ValueError, match="do not fit"):
        mg.verify_fixture(folder, "omat_tiny_f64")


def test_compare_dirs_is_zero_for_equal_runs_and_names_a_difference(tiny_dir, tmp_path):
    assert mg.report_ok(mg.compare_dirs(tiny_dir, tiny_dir))
    folder = tmp_path / "copy"
    shutil.copytree(tiny_dir, folder)
    weights = mg.load_npz(folder / "omat_tiny_f64.weights.npz")
    weights["A1/w_left"] = weights["A1/w_left"] * (1 + 1e-9)
    mg.save_npz(folder / "omat_tiny_f64.weights.npz", weights)
    report = mg.compare_dirs(tiny_dir, folder)
    assert not mg.report_ok(report)
    assert report["fixtures"]["omat_tiny_f64"]["weights"]["exceeding"] == ["A1/w_left"]
    shutil.rmtree(folder / "yamls")
    (folder / "omat_tiny_f32.json").unlink()
    assert mg.compare_dirs(tiny_dir, folder)["only_in_first"] == ["omat_tiny_f32"]


def test_generation_is_repeatable(tiny_dir, tmp_path):
    again = tmp_path / "again"
    mg.write_tier("tiny", again, models=["omat"])
    assert mg.report_ok(mg.compare_dirs(tiny_dir, again))
    one = mg.load_npz(tiny_dir / "omat_tiny_f64.npz")
    two = mg.load_npz(again / "omat_tiny_f64.npz")
    assert sorted(one) == sorted(two)
    assert all(np.array_equal(one[k], two[k]) for k in one)


def test_an_option_pair_needs_its_default_fixture(tmp_path):
    with pytest.raises(ValueError, match="default fixture"):
        mg.make_fixture(mg.FixtureRequest("omat", "tiny", "f64", "dense"), tmp_path)


def test_a_fixture_holds_a_dump_for_every_structure_and_instruction(tiny_dir):
    arrays = mg.load_npz(tiny_dir / "omat_tiny_f64.npz")
    manifest = json.loads((tiny_dir / "omat_tiny_f64.json").read_text())
    for case in mg.TIERS["tiny"].structures:
        assert manifest["structures"][case]["n_atoms"] >= 1
        for name in manifest["instructions"]:
            assert any(
                k in arrays for k in (f"{case}/ins/{name}", f"{case}/out_after/{name}")
            ), (case, name)
        for key in ("energy", "atomic_energy", "forces", "virial", "stress", "pair_f"):
            assert f"{case}/res/{key}" in arrays
    assert manifest["n_arrays"] == len(arrays)


def test_command_line_cells_and_yamls(tmp_path, capsys):
    assert mg.main(["cells"]) == 0
    assert mg.main(["yamls", "--tier", "tiny"]) == 0
    capsys.readouterr()


def test_the_committed_tiny_anchors_reproduce_in_tensorflow():
    for path in sorted(mg.GOLDEN_DIR.glob("*.json")):
        report = mg.verify_fixture(mg.GOLDEN_DIR, path.stem)
        assert mg.report_ok(report), (path.stem, report["exceeding"][:5])
