"""Tests of the committed tiny golden anchors (``tests_torch/golden``), without TensorFlow.

The anchors are what the twins are compared with, so a corrupted, stale or inconsistent anchor must fail here
and not as a mysterious mismatch in a twin test. Logic: the files are the ones the manifests describe (size,
sha256), each fixture is under 1 MB, the yamls are what the generator derives from their parents and lose no
option cell, every structure and every instruction has a dump, the provenance is recorded and the tree was clean.
Physical values: numpy-only oracles that do not call the generator or TensorFlow: the neighbour list against a
brute-force one, the bond functions and the spherical harmonics from their definitions, Newton's third law, the
forces, the virial and the stress rebuilt from the pair forces, the output chain against the scale of the yaml and
the stored shifts, the float32 fixtures against the float64 ones, and the two layout options against the default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.fresh_python import run_fresh_python
from tests.neighbour_oracle import brute_force_pairs
from tests.tolerances import (
    FLOAT32_NETWORK,
    FLOAT64_ARITHMETIC,
    NEIGHBOUR_VECTOR_F64,
    SOFTENED_UNIT_VECTOR_F64,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_golden as mg  # noqa: E402  # ty: ignore[unresolved-import]  (tools/ is on sys.path at run time)

GOLDEN = ROOT / "tests_torch" / "golden"
NAMES = tuple(sorted(r.name for r in mg.requests_for("tiny")))
BASE_NAMES = tuple(n for n in NAMES if n.endswith(("_f64", "_f32")))
CUTOFF = 6.0
"""``RadialBasis.rcut`` of both parent yamls."""
SOFTENING = 1e-10
SCALE_OMAT = 1.892985414710868
"""``ConstantScaleShiftTarget.scale`` of ``tests/model_grace_2L_omat.yaml``, read off the yaml by hand."""
MAX_TINY_BYTES = 1_000_000
HEX40 = set("0123456789abcdef")
OUTPUT_NAME = {"omat": "MLPOut2ScalarTarget", "large": "LinMLPOut2ScalarTarget"}
SHIFT_NAME = {"omat": "ConstantScaleShiftTarget", "large": "TrainableShiftTarget"}


def close(actual, expected, tolerance, scale=None) -> bool:
    """``numpy.isclose`` semantics of a named row; ``scale`` adds ``rtol * scale`` for values near zero."""
    bound = tolerance.atol + (tolerance.rtol * scale if scale is not None else 0.0)
    return bool(np.allclose(actual, expected, rtol=tolerance.rtol, atol=bound))


def manifest_of(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text())


def arrays_of(name: str) -> dict[str, np.ndarray]:
    return mg.load_npz(GOLDEN / f"{name}.npz")


def weights_of(name: str) -> dict[str, np.ndarray]:
    owner = manifest_of(name)["weights_from"] or name
    return mg.load_npz(GOLDEN / f"{owner}.weights.npz")


def structures_of(name: str) -> dict:
    return mg.tier_cases(mg.TIERS[manifest_of(name)["tier"]])


def element_index(name: str) -> dict[str, int]:
    return {s: i for i, s in enumerate(manifest_of(name)["elements"])}


# ------------------------------------------------------------------ logic: the committed files


def test_the_committed_set_is_the_eight_tiny_fixtures():
    on_disk = tuple(sorted(p.stem for p in GOLDEN.glob("*.json")))
    assert on_disk == NAMES
    assert len(NAMES) == 8
    assert {n.split("_")[0] for n in NAMES} == {"omat", "large"}


@pytest.mark.parametrize("name", NAMES)
def test_files_are_the_ones_the_manifest_describes(name):
    manifest = manifest_of(name)
    assert manifest["name"] == name
    total = (GOLDEN / f"{name}.json").stat().st_size
    for entry in manifest["files"].values():
        path = GOLDEN / entry["name"]
        assert path.stat().st_size == entry["bytes"]
        assert mg._sha256(path) == entry["sha256"]
        total += entry["bytes"]
    assert total < MAX_TINY_BYTES, f"{total} bytes"
    assert mg.size_violations(manifest, (GOLDEN / f"{name}.json").stat().st_size) == []
    assert bool(manifest["weights_from"]) == ("weights" not in manifest["files"])
    assert manifest["n_arrays"] == len(arrays_of(name))


@pytest.mark.parametrize("name", NAMES)
def test_provenance_is_recorded_and_the_tree_was_clean(name):
    manifest = manifest_of(name)
    library = manifest["library"]
    assert manifest["format_version"] == mg.FORMAT_VERSION
    assert manifest["seed"] == mg.SEED
    assert {"tensorflow", "numpy", "pandas", "ase", "python"} <= set(
        library["versions"]
    )
    assert isinstance(library["use_gemm_couple"], bool)
    for key in ("library_git_sha", "generator_git_sha"):
        assert len(library[key]) == 40 and set(library[key]) <= HEX40
    assert library["library_dirty"] is False
    assert library["generator_dirty"] is False


@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_tiny_yaml_is_what_the_generator_derives_and_loses_no_option_cell(model):
    parent = mg.TESTS / mg.MODELS[model]
    committed = mg.yaml_path(model, "tiny")
    assert committed.read_text() == mg.yaml_text(
        mg.derive_yaml(parent, mg.TIERS["tiny"])
    )
    assert mg.missing_cells(parent, committed) == []
    for name in NAMES:
        if name.startswith(model):
            sha = manifest_of(name)["yaml_sha256"]
            assert sha == mg._sha256(committed)


@pytest.mark.parametrize("name", NAMES)
def test_every_structure_and_instruction_has_a_dump(name):
    arrays, manifest = arrays_of(name), manifest_of(name)
    assert list(manifest["structures"]) == list(mg.TIERS["tiny"].structures)
    # the order of the instructions is the order they run in: it is the order of the yaml
    parent = yaml.safe_load(mg.yaml_path(manifest["model"], "tiny").read_text())
    assert list(manifest["instructions"]) == list(parent)
    for case in manifest["structures"]:
        for instruction in manifest["instructions"]:
            assert any(
                key in arrays
                for key in (
                    f"{case}/ins/{instruction}",
                    f"{case}/out_after/{instruction}",
                )
            ), (case, instruction)
        for quantity in (
            "energy",
            "atomic_energy",
            "forces",
            "virial",
            "stress",
            "pair_f",
        ):
            assert f"{case}/res/{quantity}" in arrays
    tables = [k for k in arrays if k.startswith("tables/")]
    assert {
        "tables/AA/left_ind",
        "tables/AA/right_ind",
        "tables/AA/m_sum_ind",
        "tables/AA/cg",
    } <= set(tables)


@pytest.mark.parametrize("name", BASE_NAMES)
def test_weights_match_the_variable_table(name):
    weights, manifest = weights_of(name), manifest_of(name)
    table = {row["key"]: row for row in manifest["variables"]}
    assert set(weights) == set(table)
    for key, row in table.items():
        assert list(weights[key].shape) == row["shape"], key
        if row["dtype"] in ("float32", "float64"):
            assert weights[key].dtype.name == row["dtype"] == manifest["dtype"], key
        if row["trainable"] and row["dtype"] in ("float32", "float64"):
            assert np.any(weights[key] != 0), (
                key
            )  # seeded: no layer is left at its zero initialisation
    assert weights["Z/element_map_symbols"].tolist() == manifest["elements"]


def test_a_fixture_loads_without_tensorflow(tmp_path):
    code = (
        "import sys\n"
        "sys.modules['tensorflow'] = None\n"
        "import numpy as np\n"
        "with np.load(sys.argv[1], allow_pickle=False) as d:\n"
        "    print(len(d.files), float(d['dimer/res/energy'].sum()) != 0.0)\n"
    )
    done = run_fresh_python(code, tmp_path, args=[str(GOLDEN / "omat_tiny_f64.npz")])
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[1] == "True"


# ------------------------------------------------------------------ physics: inputs


def _bonds(arrays: dict, case: str) -> dict[str, np.ndarray]:
    prefix = f"{case}/in/"
    return {
        k.removeprefix(prefix): v for k, v in arrays.items() if k.startswith(prefix)
    }


@pytest.mark.parametrize("name", ["omat_tiny_f64", "large_tiny_f64"])
def test_the_neighbour_lists_equal_a_brute_force_list(name):
    arrays = arrays_of(name)
    for case, atoms in structures_of(name).items():
        inputs = _bonds(arrays, case)
        cell = (
            atoms.cell.array if atoms.cell.rank == 3 else np.eye(3)
        )  # no cell: nothing is periodic
        pairs = brute_force_pairs(
            atoms.positions,
            cell,
            atoms.get_chemical_symbols(),
            CUTOFF,
            tuple(bool(p) for p in atoms.pbc),
        )
        vectors = inputs["bond_vector"]
        if not pairs:
            # an atom with no neighbour gets one dummy bond far outside the cutoff
            assert vectors.shape == (1, 3) and np.linalg.norm(vectors[0]) > CUTOFF, case
            continue
        expected = np.array([p.vector for p in pairs])
        order = np.lexsort(expected.T[::-1]), np.lexsort(vectors.T[::-1])
        assert vectors.shape == expected.shape, case
        assert close(vectors[order[1]], expected[order[0]], NEIGHBOUR_VECTOR_F64), case


@pytest.mark.parametrize("name", ["omat_tiny_f64", "large_tiny_f64"])
def test_bond_indices_and_elements_are_consistent(name):
    arrays, index = arrays_of(name), element_index(name)
    for case, atoms in structures_of(name).items():
        inputs = _bonds(arrays, case)
        atomic_mu = np.array([index[s] for s in atoms.get_chemical_symbols()])
        assert np.array_equal(inputs["atomic_mu_i"], atomic_mu)
        assert np.array_equal(inputs["mu_i"], atomic_mu[inputs["ind_i"]])
        assert np.array_equal(inputs["mu_j"], atomic_mu[inputs["ind_j"]])
        assert inputs["ind_i"].max() < len(atoms) and inputs["ind_j"].max() < len(atoms)
        assert len(inputs["map_atoms_to_structure"]) == len(atoms)
        assert set(inputs["map_bonds_to_structure"].tolist()) == {0}


@pytest.mark.parametrize("name", NAMES)
def test_bond_functions_follow_their_definitions(name):
    arrays = arrays_of(name)
    for case in mg.TIERS["tiny"].structures:
        r = arrays[f"{case}/in/bond_vector"]
        length = arrays[f"{case}/ins/BondLength"][:, 0]
        assert close(length, np.sqrt((r**2).sum(1) + SOFTENING), FLOAT64_ARITHMETIC), (
            case
        )
        assert close(
            arrays[f"{case}/ins/ScaledBondVector"],
            r / length[:, None],
            FLOAT64_ARITHMETIC,
        ), case
        y = arrays[f"{case}/ins/Y"]
        for degree in range(5):
            # the 4 pi-scaled real harmonics of GRACE: the sum over m of Y_lm^2 is 2l+1 (Y_00 = 1)
            block = y[:, degree * degree : (degree + 1) ** 2]
            assert close((block**2).sum(1), 2 * degree + 1, SOFTENED_UNIT_VECTOR_F64), (
                case,
                degree,
            )


# ------------------------------------------------------------------ physics: results


@pytest.mark.parametrize("name", BASE_NAMES)
def test_results_are_consistent_with_each_other(name):
    arrays = arrays_of(name)
    for case, atoms in structures_of(name).items():
        res = {
            k: arrays[f"{case}/res/{k}"]
            for k in ("energy", "atomic_energy", "forces", "virial", "stress", "pair_f")
        }
        inputs = _bonds(arrays, case)
        pair, r = res["pair_f"], inputs["bond_vector"]
        scale = float(np.abs(pair).max())
        assert close(
            res["energy"].sum(),
            res["atomic_energy"].sum(),
            FLOAT64_ARITHMETIC,
            abs(res["energy"]).max(),
        )
        # Newton's third law: the pair forces enter an atom once with each sign, so the total force vanishes
        assert close(res["forces"].sum(0), 0.0, FLOAT64_ARITHMETIC, scale), case
        # the force on an atom: pair forces towards it minus those away from it
        forces = np.zeros_like(res["forces"])
        np.add.at(forces, inputs["ind_j"], pair)
        np.subtract.at(forces, inputs["ind_i"], pair)
        assert close(res["forces"], forces, FLOAT64_ARITHMETIC, scale), case
        # virial: sum over bonds of r (x) f, stored xx, yy, zz, xy, xz, yz
        matrix = np.einsum("ba,bc->ac", r, pair)
        virial = [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 2],
        ]
        assert close(res["virial"], virial, FLOAT64_ARITHMETIC, np.abs(virial).max()), (
            case
        )
        # stress, ASE sign and Voigt order (xx, yy, zz, yz, xz, xy); zero without a cell
        expected = np.zeros(6)
        if atoms.cell.rank == 3:
            expected = -np.array(virial)[mg.osn.VIRIAL_TO_VOIGT] / atoms.get_volume()
        assert close(
            res["stress"], expected, FLOAT64_ARITHMETIC, np.abs(expected).max()
        ), case


@pytest.mark.parametrize("name", BASE_NAMES)
def test_an_isolated_atom_has_zero_force_and_the_shift_only_energy(name):
    arrays = arrays_of(name)
    assert np.all(arrays["isolated_atom/res/forces"] == 0)
    assert arrays["isolated_atom/res/forces"].shape == (1, 3)
    assert arrays["isolated_atom/res/atomic_energy"].shape == (1, 1)


@pytest.mark.parametrize("name", BASE_NAMES)
def test_the_energy_chain_follows_the_yaml_and_the_stored_weights(name):
    arrays, weights, manifest = arrays_of(name), weights_of(name), manifest_of(name)
    model = manifest["model"]
    out, shift = OUTPUT_NAME[model], SHIFT_NAME[model]
    for case in mg.TIERS["tiny"].structures:
        initial = arrays[f"{case}/ins/atomic_energy"]
        assert float(initial) == 0.0
        assert np.array_equal(arrays[f"{case}/out_before/{out}"], initial)
        chain = [
            f"{case}/out_after/{out}",
            f"{case}/out_before/{shift}",
            f"{case}/out_after/{shift}",
        ]
        assert np.array_equal(arrays[chain[0]], arrays[chain[1]])
        assert np.array_equal(
            arrays[chain[2]].astype(np.float64), arrays[f"{case}/res/atomic_energy"]
        )
        before, after = arrays[chain[1]], arrays[chain[2]]
        if model == "omat":
            scale = manifest["instructions"][shift]["init_args"]["scale"]
            assert scale == SCALE_OMAT
            expected = scale * before
        else:
            mu = arrays[f"{case}/in/atomic_mu_i"]
            expected = before + weights["TrainableShiftTarget/at_shifts"][mu]
        tolerance = (
            FLOAT32_NETWORK if manifest["dtype"] == "float32" else FLOAT64_ARITHMETIC
        )
        assert close(after, expected, tolerance), case


@pytest.mark.parametrize("name", BASE_NAMES)
def test_the_embedding_output_is_the_stored_weight(name):
    arrays, weights = arrays_of(name), weights_of(name)
    for case in mg.TIERS["tiny"].structures:
        assert np.array_equal(arrays[f"{case}/ins/Z"], weights["Z/w"])


@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_float32_anchors_are_the_float64_ones_in_float32(model):
    low, high = f"{model}_tiny_f32", f"{model}_tiny_f64"
    w32, w64 = weights_of(low), weights_of(high)
    assert set(w32) == set(w64)
    for key, value in w64.items():
        if value.dtype == np.float64:
            assert np.array_equal(w32[key], value.astype(np.float32)), key
    a32, a64 = arrays_of(low), arrays_of(high)
    for case in mg.TIERS["tiny"].structures:
        energy32, energy64 = a32[f"{case}/res/energy"], a64[f"{case}/res/energy"]
        assert close(energy32, energy64, FLOAT32_NETWORK), case
        # the dtype chain of a float32 model: the bond functions are float64, the networks float32
        for instruction in ("BondLength", "ScaledBondVector", "RadialBasis", "Y"):
            assert a32[f"{case}/ins/{instruction}"].dtype == np.float64, instruction
        for instruction in ("R", "A", "A1", "AA", "I", "YI", "B"):
            assert a32[f"{case}/ins/{instruction}"].dtype == np.float32, instruction
        assert a32[f"{case}/res/energy"].dtype == np.float64
        assert a32[f"{case}/in/bond_vector"].dtype == np.float64
    assert float(np.abs(a64["self_image_cell/res/energy"]).max()) > 1e-3


@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_option_pairs_agree_with_the_default_layout(model):
    default = arrays_of(f"{model}_tiny_f64")
    dense, lm_first = (
        arrays_of(f"{model}_tiny_f64_dense"),
        arrays_of(f"{model}_tiny_f64_lm_first"),
    )
    for case in mg.TIERS["tiny"].structures:
        for quantity in (
            "energy",
            "atomic_energy",
            "forces",
            "virial",
            "stress",
            "pair_f",
        ):
            key = f"{case}/res/{quantity}"
            scale = float(np.abs(default[key]).max())
            assert close(dense[key], default[key], FLOAT64_ARITHMETIC, scale), key
            assert close(lm_first[key], default[key], FLOAT64_ARITHMETIC, scale), key
    manifest_dense = manifest_of(f"{model}_tiny_f64_dense")
    manifest_lm = manifest_of(f"{model}_tiny_f64_lm_first")
    for manifest, option in ((manifest_dense, "dense_nbr"), (manifest_lm, "lm_first")):
        assert manifest["option"] == option
        assert manifest["weights_from"] == f"{model}_tiny_f64"
        pair = manifest["option_pair"]
        assert pair["weights"]["shape_differs"] == {}
        assert pair["outputs"]["differs"] == [] and pair["outputs"]["missing"] == []
    # lm_first: the angular axis comes first in every tensor that has one; everything else is unchanged
    moved = [
        k for k in default if k.endswith(("/ins/A", "/ins/AA", "/ins/YI", "/ins/I"))
    ]
    assert moved
    for key in moved:
        scale = float(np.abs(default[key]).max())
        assert close(
            lm_first[key], np.moveaxis(default[key], -1, 0), FLOAT64_ARITHMETIC, scale
        ), key
    for key in (k for k in default if k.endswith(("/ins/R", "/ins/Y", "/ins/Z"))):
        scale = float(np.abs(default[key]).max())
        assert close(lm_first[key], default[key], FLOAT64_ARITHMETIC, scale), key


# ------------------------------------------------------------------ the faithful tier (local files, committed manifests)

FIXTURES = ROOT / "tests_torch" / "fixtures"
FAITHFUL_NAMES = tuple(sorted(r.name for r in mg.requests_for("faithful")))


def test_the_faithful_manifests_are_committed_and_state_their_sizes():
    on_disk = tuple(sorted(p.stem for p in FIXTURES.glob("*.json")))
    assert on_disk == FAITHFUL_NAMES
    for name in FAITHFUL_NAMES:
        manifest = json.loads((FIXTURES / f"{name}.json").read_text())
        assert manifest["tier"] == "faithful" and manifest["name"] == name
        assert manifest["library"]["library_dirty"] is False
        assert manifest["library"]["generator_dirty"] is False
        assert mg.size_violations(manifest) == [], name
        assert all(entry["bytes"] > 0 for entry in manifest["files"].values())
        assert len(manifest["elements"]) == 6
        assert list(manifest["structures"]) == list(mg.TIERS["faithful"].structures)


@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_faithful_yaml_is_what_the_generator_derives(model):
    parent = mg.TESTS / mg.MODELS[model]
    committed = mg.yaml_path(model, "faithful")
    assert committed.read_text() == mg.yaml_text(
        mg.derive_yaml(parent, mg.TIERS["faithful"])
    )
    sha = mg._sha256(committed)
    assert all(
        json.loads((FIXTURES / f"{n}.json").read_text())["yaml_sha256"] == sha
        for n in FAITHFUL_NAMES
        if n.startswith(model)
    )


@pytest.mark.parametrize("name", FAITHFUL_NAMES)
def test_a_local_faithful_file_is_the_one_the_manifest_describes(name):
    manifest = json.loads((FIXTURES / f"{name}.json").read_text())
    for entry in manifest["files"].values():
        path = FIXTURES / entry["name"]
        if not path.exists():
            pytest.skip(
                f"{entry['name']} is generated locally (python tools/make_golden.py write --tier faithful)"
            )
        assert mg._sha256(path) == entry["sha256"]


@pytest.mark.parametrize("name", BASE_NAMES)
def test_a_periodic_anchor_has_non_zero_forces_and_stress(name):
    # the symmetric cells have zero forces; without this structure a wrong force path through periodic images would pass
    arrays = arrays_of(name)
    assert np.abs(arrays["periodic_triple/res/forces"]).max() > 1e-5
    assert np.abs(arrays["periodic_triple/res/stress"]).max() > 1e-6
    assert np.abs(arrays["dimer/res/forces"]).max() > 1e-5
    assert np.abs(arrays["self_image_cell/res/stress"]).max() > 1e-6
