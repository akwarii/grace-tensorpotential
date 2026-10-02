"""Characterization tests for ``TPCalculator.__init__`` (``calculator/asecalculator.py``).

The constructor loads one model or an ensemble, resolves the aggregation engine against what
the models can run, extracts the cutoff, the element map and the optional per-pair cutoff
matrix, and creates the data builders and padding managers.

Logic layer: argument errors, the defaults of every attribute, the engine fallbacks, the
ensemble consistency check, the cutoff message, the optional data builders. Physics layer:
what the extracted cutoff and element map mean for the energy. A pair beyond the cutoff
contributes nothing, so a far dimer has the sum of the isolated-atom energies; a per-pair
cutoff matrix switches interactions pair by pair; the energy is invariant under rotation and
translation, and the forces are minus the finite-difference gradient of the energy.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf
import yaml
from ase import Atoms
from scipy.spatial.transform import Rotation

from tensorpotential import TPModel, constants
from tensorpotential.calculator import TPCalculator
from tensorpotential.calculator.asecalculator import DensePaddingManager, PaddingManager
from tensorpotential.data.databuilder import GeometricalDataBuilder
from tensorpotential.extra.gen_tensor.databuilder import (
    CellDataBuilder,
    PositionsDataBuilder,
)
from tensorpotential.extra.gen_tensor.model import ComputeBatchEFTensor, grace_2
from tensorpotential.instructions.base import TPInstruction
from tensorpotential.potentials.presets import (
    GRACE_1LAYER_v2_25,
    GRACE_2LAYER_v2_25,
)
from tests.tolerances import COVARIANCE_F64, FINITE_DIFFERENCE_F64

ELEMENTS = {"Al": 0, "Li": 1}
RCUT = 4.0


def build_model(instructions, compute_function=None) -> TPModel:
    model = (
        TPModel(instructions)
        if compute_function is None
        else TPModel(instructions, compute_function, compute_function)
    )
    model.build(tf.float64)
    model.decorate_compute_function(input_signature_float_dtype=tf.float64)
    return model


def one_layer(
    cutoff_dict: dict | None = None, rcut: float = RCUT, seed: int = 3
) -> TPModel:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return build_model(
        GRACE_1LAYER_v2_25(
            element_map=ELEMENTS,
            rcut=rcut,
            lmax=2,
            n_rad_base=4,
            n_rad_max=6,
            prod_func_n_max=8,
            embedding_size=6,
            n_mlp_dens=4,
            max_order=2,
            cutoff_dict=cutoff_dict,
        ).get_instructions()
    )


def two_layer(dense: bool, seed: int = 3) -> TPModel:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return build_model(
        GRACE_2LAYER_v2_25(
            element_map=ELEMENTS,
            rcut=RCUT,
            lmax=[2, 2],
            n_rad_base=4,
            n_rad_max=[6, 8],
            prod_func_n_max=[6, 8],
            embedding_size=6,
            max_order=2,
            dense_nbr=dense,
        ).get_instructions()
    )


def tensor_model() -> TPModel:
    tf.random.set_seed(3)
    compute = ComputeBatchEFTensor({
        "tensor_components": [2],
        "compute_energy": False,
        "compute_forces": False,
    })
    instructions = grace_2(
        element_map=ELEMENTS,
        rcut=RCUT,
        tensor_components=[2],
        lmax=2,
        n_rad_base=4,
        n_rad_max=[6, 8],
        embedding_size=6,
        n_mlp_dens=4,
        max_order=1,
    )
    return build_model(instructions, compute)


class NeedsData(TPInstruction):
    """Declares extra input keys and computes nothing; a model with it asks for them."""

    def __init__(self, keys: dict[str, list], name: str = "needs_data"):
        super().__init__(name=name)
        self.input_tensor_spec = {
            k: {"shape": shape, "dtype": "float"} for k, shape in keys.items()
        }

    def build(self, float_dtype):
        self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        return tf.zeros([])


def model_needing(keys: dict[str, list]) -> TPModel:
    instructions = GRACE_1LAYER_v2_25(
        element_map=ELEMENTS,
        rcut=RCUT,
        lmax=2,
        n_rad_base=4,
        n_rad_max=6,
        prod_func_n_max=8,
        embedding_size=6,
        n_mlp_dens=4,
        max_order=2,
    ).get_instructions()
    instructions["needs_data"] = NeedsData(keys)
    return build_model(instructions)


# ----------------------------------------------------------------------------- logic


def test_missing_model_is_refused():
    with pytest.raises(ValueError, match='"model" parameter is not provided'):
        TPCalculator(model=None)


def test_unrecognised_model_object_is_refused():
    with pytest.raises(ValueError, match="model type is not recognized"):
        TPCalculator(model=42)


def test_mode_is_case_insensitive_and_validated():
    model = one_layer()
    assert TPCalculator(model=model, mode="DIVERSE").mode == "diverse"
    with pytest.raises(ValueError, match="mode must be 'uniform'"):
        TPCalculator(model=model, mode="fast")


def test_attribute_defaults_after_construction():
    calc = TPCalculator(model=one_layer())
    assert calc.compute_properties == ["energy", "forces", "free_energy", "stress"]
    assert calc.eval_time == 0
    assert calc.extra_properties is None
    assert calc.truncate_extras_by_natoms is False
    assert calc.min_dist is None
    assert calc.gmm_uq_model is None
    assert calc.dense_padding_manager is None
    assert calc.dense_reshape is False
    assert not calc._uq_enabled
    assert calc.available_uq_modes == []
    assert len(calc.models) == 1
    assert set(calc.data_keys) == set(calc.models[0].compute_specs)


def test_constructor_options_are_stored():
    calc = TPCalculator(
        model=one_layer(),
        min_dist=0.5,
        extra_properties=("a", "b"),
        truncate_extras_by_natoms=["a"],
    )
    assert calc.min_dist == 0.5
    assert calc.extra_properties == ["a", "b"]  # copied into a list
    assert calc.truncate_extras_by_natoms == ["a"]


def test_padding_options_reach_the_padding_manager():
    calc = TPCalculator(
        model=one_layer(),
        pad_neighbors_fraction=0.2,
        pad_atoms_number=3,
        max_number_reduction_recompilation=5,
        adaptive_padding=False,
        debug_padding_verbose=1,
    )
    pm = calc.padding_manager
    assert isinstance(pm, PaddingManager)
    assert pm.pad_neighbors_fraction == 0.2
    assert pm.pad_atoms_number == 3
    assert pm.max_number_reduction_recompilation == 5
    assert pm.adaptive_padding is False
    assert pm.debug_padding_verbose == 1
    assert pm.data_builders is calc.data_builders


def test_the_geometry_builder_gets_the_extracted_cutoff_and_elements():
    calc = TPCalculator(model=one_layer())
    builder = calc.geom_data_builder
    assert isinstance(builder, GeometricalDataBuilder)
    assert calc.data_builders == [builder]
    assert builder.cutoff == RCUT == calc.cutoff
    assert dict(builder.elements_map) == ELEMENTS


def test_element_map_follows_the_model():
    calc = TPCalculator(model=one_layer())
    assert {str(k): int(v) for k, v in calc.element_map.items()} == ELEMENTS


def test_cutoff_of_the_model_wins_over_the_keyword_and_says_so(capsys):
    calc = TPCalculator(model=one_layer(), cutoff=7.5)
    assert calc.cutoff == RCUT
    assert "different from calculator's 7.5" in capsys.readouterr().out


def test_matching_cutoff_keyword_is_silent(capsys):
    calc = TPCalculator(model=one_layer(), cutoff=RCUT)
    assert calc.cutoff == RCUT
    assert capsys.readouterr().out == ""


def test_dense_model_uses_the_dense_engine_and_a_dense_padding_manager():
    calc = TPCalculator(model=two_layer(dense=True), mode="uniform")
    assert calc.dense_reshape is True
    assert isinstance(calc.dense_padding_manager, DensePaddingManager)
    assert calc.geom_data_builder.dense_nbr is True


def test_ensemble_of_matching_models_keeps_every_member():
    calc = TPCalculator(model=[one_layer(seed=1), one_layer(seed=2)])
    assert len(calc.models) == 2
    assert (
        calc._saved_model_uq_sigs == {}
    )  # only a single SavedModel carries UQ signatures


def test_ensemble_cutoff_is_the_largest_of_its_members():
    calc = TPCalculator(model=[one_layer(rcut=3.0), one_layer(rcut=4.5)])
    assert calc.cutoff == 4.5  # every member must see all of its neighbours


def test_ensemble_pair_cutoffs_are_the_elementwise_maximum():
    first = {("Al", "Al"): 2.0, ("Al", "Li"): 3.0, ("Li", "Li"): 2.5}
    second = {("Al", "Al"): 2.5, ("Al", "Li"): 2.0, ("Li", "Li"): 2.75}
    calc = TPCalculator(
        model=[one_layer(cutoff_dict=first), one_layer(cutoff_dict=second)]
    )
    got = {tuple(map(str, k)): float(v) for k, v in calc.cutoff_dict.items()}
    assert got == {("Al", "Al"): 2.5, ("Al", "Li"): 3.0, ("Li", "Li"): 2.75}
    assert calc.cutoff == 3.0


def test_ensemble_of_a_dense_and_a_segment_sum_model_has_no_common_engine():
    with pytest.raises(ValueError, match="neither a segment_sum nor a dense"):
        TPCalculator(model=[two_layer(dense=True), two_layer(dense=False)])


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_ensemble_with_different_data_keys_is_refused(order):
    models = [one_layer(), tensor_model()]
    with pytest.raises(ValueError, match="inconsistent data keys"):
        TPCalculator(model=[models[i] for i in order])


def test_extra_input_keys_add_the_matching_data_builders():
    keys = {
        constants.ATOMIC_POS: [None, 3],
        constants.CELL_VECTORS: [None, 3, 3],
    }
    calc = TPCalculator(model=model_needing(keys))
    kinds = [type(b) for b in calc.data_builders]
    assert kinds == [GeometricalDataBuilder, PositionsDataBuilder, CellDataBuilder]
    assert all(b.cutoff == RCUT for b in calc.data_builders[1:])


@pytest.mark.parametrize(
    ("key", "shape", "builder"),
    [
        (constants.ATOMIC_POS, [None, 3], PositionsDataBuilder),
        (constants.CELL_VECTORS, [None, 3, 3], CellDataBuilder),
    ],
)
def test_each_extra_key_adds_only_its_builder(key, shape, builder):
    calc = TPCalculator(model=model_needing({key: shape}))
    assert [type(b) for b in calc.data_builders] == [GeometricalDataBuilder, builder]


@pytest.mark.skipif(
    importlib.util.find_spec("tensorpotential.experimental") is not None,
    reason="tensorpotential.experimental exists here, so the mag builder imports",
)
def test_magnetic_moments_need_the_missing_experimental_package():
    model = model_needing({constants.ATOMIC_MAGMOM: [None, 3]})
    with pytest.raises(ImportError, match="mag.databuilder not found"):
        TPCalculator(model=model)


def test_magnetic_moment_builder_is_added_when_the_package_is_present(monkeypatch):
    # Stands in for tensorpotential.experimental, which is not part of this tree.
    class MagMomDataBuilder:
        pass

    for name in ("experimental", "experimental.mag", "experimental.mag.databuilder"):
        module = types.ModuleType(f"tensorpotential.{name}")
        monkeypatch.setitem(sys.modules, f"tensorpotential.{name}", module)
    module.MagMomDataBuilder = MagMomDataBuilder
    calc = TPCalculator(model=model_needing({constants.ATOMIC_MAGMOM: [None, 3]}))
    assert [type(b) for b in calc.data_builders] == [
        GeometricalDataBuilder,
        MagMomDataBuilder,
    ]


@pytest.mark.parametrize(
    ("key", "shape"),
    [(constants.ATOMIC_POS, [None, 3]), (constants.CELL_VECTORS, [None, 3, 3])],
)
def test_a_missing_gen_tensor_package_is_reported(monkeypatch, key, shape):
    model = model_needing({key: shape})
    # A None entry in sys.modules makes the import raise ModuleNotFoundError, as if the
    # optional gen_tensor package had not been installed.
    monkeypatch.setitem(
        sys.modules, "tensorpotential.extra.gen_tensor.databuilder", None
    )
    with pytest.raises(ImportError, match="gen_tensor.databuilder not found"):
        TPCalculator(model=model)


def test_dense_only_model_serves_a_segment_sum_request_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        calc = TPCalculator(model=two_layer(dense=True), mode="diverse")
    assert calc.dense_reshape is True
    assert "only provides the dense engine" in caplog.text


def test_segment_sum_only_model_serves_a_dense_request_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        calc = TPCalculator(model=one_layer(), mode="uniform")
    assert calc.dense_reshape is False
    assert calc.dense_padding_manager is None
    assert "has no `compute_dense` signature" in caplog.text


# --- saved models (committed fixtures with random weights, and one saved here)

HERE = Path(__file__).parent.resolve()
SAVED_MODEL = str(HERE / "test_calculator_model")
SAVED_MODEL_CUSTOM_CUTOFF = str(HERE / "test_model_custom_cutoff")
MO_NB_TA_W = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}


def test_saved_model_takes_cutoff_and_elements_from_its_metadata():
    calc = TPCalculator(model=SAVED_MODEL)
    metadata = yaml.safe_load(
        (HERE / "test_calculator_model" / "metadata.yaml").read_text()
    )
    assert calc.cutoff == metadata["cutoff"] == 6.0
    assert {str(k): int(v) for k, v in calc.element_map.items()} == MO_NB_TA_W
    assert calc.cutoff_dict is None
    assert not isinstance(calc.models[0], TPModel)  # the loaded SavedModel object


def test_saved_model_with_a_cutoff_matrix_builds_a_pair_cutoff_dict():
    calc = TPCalculator(model=SAVED_MODEL_CUSTOM_CUTOFF)
    metadata = yaml.safe_load(
        (HERE / "test_model_custom_cutoff" / "metadata.yaml").read_text()
    )
    matrix = np.array(metadata["cutoff_matrix"])
    assert calc.cutoff == matrix.max() == 7.0
    symbols = list(MO_NB_TA_W)
    expected = {
        (symbols[i], symbols[j]): matrix[i, j] for i in range(4) for j in range(i, 4)
    }
    got = {tuple(map(str, k)): float(v) for k, v in calc.cutoff_dict.items()}
    assert got == expected  # one entry per unordered pair, i <= j
    assert len(got) == 10


def test_a_zero_cutoff_in_the_metadata_falls_back_to_the_keyword(tmp_path, capsys):
    copy = tmp_path / "model"
    shutil.copytree(SAVED_MODEL, copy)
    (copy / "metadata.yaml").write_text(
        yaml.safe_dump({"chemical_symbols": list(MO_NB_TA_W), "cutoff": 0.0})
    )
    calc = TPCalculator(model=str(copy), cutoff=5.5)
    assert calc.cutoff == 5.5
    assert "Couldn't extract cutoff value from the model" in capsys.readouterr().out


def test_dual_engine_saved_model_switches_engine_with_the_mode(tmp_path):
    path = str(tmp_path / "dual")
    two_layer(dense=True).save_model(path, input_signature_float_dtype=tf.float64)
    metadata = yaml.safe_load((Path(path) / "metadata.yaml").read_text())
    assert metadata["cutoff"] == RCUT  # what the model was built with
    assert metadata["chemical_symbols"] == list(ELEMENTS)

    dense = TPCalculator(model=path, mode="uniform")
    assert dense.dense_reshape is True
    assert dense.models[0].compute is dense.models[0].compute_dense  # rebound
    assert isinstance(dense.dense_padding_manager, DensePaddingManager)
    assert dense.cutoff == RCUT

    seg = TPCalculator(model=path, mode="diverse")
    assert seg.dense_reshape is False
    assert seg.models[0].compute is not seg.models[0].compute_dense
    assert set(dense.data_keys) == set(
        seg.data_keys
    )  # one input signature for both engines


# ---------------------------------------------------------------------------- physics


def dimer(distance: float, symbols: str = "AlLi") -> Atoms:
    return Atoms(symbols, positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])


def cluster() -> Atoms:
    # four atoms, no symmetry, all pairs inside the cutoff
    return Atoms(
        "Al2Li2",
        positions=[[0, 0, 0], [2.3, 0.2, 0.1], [0.4, 2.1, 0.5], [1.1, 0.9, 2.2]],
    )


def energy(calc: TPCalculator, atoms: Atoms) -> float:
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy()


def isolated(calc: TPCalculator, symbol: str) -> float:
    return energy(calc, Atoms(symbol, positions=[[0.0, 0.0, 0.0]]))


def test_atoms_beyond_the_extracted_cutoff_do_not_interact():
    calc = TPCalculator(model=one_layer())
    far = energy(calc, dimer(RCUT + 0.5))
    assert far == pytest.approx(
        isolated(calc, "Al") + isolated(calc, "Li"),
        rel=COVARIANCE_F64.rtol,
        abs=COVARIANCE_F64.atol,
    )
    assert (
        abs(energy(calc, dimer(RCUT - 0.5)) - far) > 1e-6
    )  # inside the cutoff they do


def test_pair_cutoffs_switch_interactions_pair_by_pair():
    cutoffs = {("Al", "Al"): 2.0, ("Al", "Li"): 3.0, ("Li", "Li"): 2.5}
    calc = TPCalculator(model=one_layer(cutoff_dict=cutoffs))
    assert calc.cutoff == 3.0  # the largest entry
    assert {
        tuple(map(str, k)): float(v) for k, v in calc.cutoff_dict.items()
    } == cutoffs
    iso = {s: isolated(calc, s) for s in ("Al", "Li")}
    # r = 2.4: beyond the Al-Al cutoff (2.0), inside Al-Li (3.0) and Li-Li (2.5)
    assert energy(calc, dimer(2.4, "AlAl")) == pytest.approx(
        2 * iso["Al"], rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
    )
    assert abs(energy(calc, dimer(2.4, "AlLi")) - iso["Al"] - iso["Li"]) > 1e-6
    assert abs(energy(calc, dimer(2.4, "LiLi")) - 2 * iso["Li"]) > 1e-6
    # r = 2.8: now beyond Li-Li too
    assert energy(calc, dimer(2.8, "LiLi")) == pytest.approx(
        2 * iso["Li"], rel=COVARIANCE_F64.rtol, abs=COVARIANCE_F64.atol
    )


def test_energy_is_invariant_and_forces_rotate_under_a_rigid_motion():
    calc = TPCalculator(model=one_layer())
    atoms = cluster()
    rot = Rotation.from_rotvec(0.7 * np.array([1.0, 2.0, 3.0]) / 14**0.5).as_matrix()
    moved = atoms.copy()
    moved.positions = atoms.positions @ rot.T + np.array([0.3, -0.2, 5.0])
    atoms.calc = calc
    f0, e0 = atoms.get_forces(), atoms.get_potential_energy()
    moved.calc = calc
    f1, e1 = moved.get_forces(), moved.get_potential_energy()
    assert np.abs(f0).max() > 1e-3
    np.testing.assert_allclose(
        e1, e0, rtol=COVARIANCE_F64.rtol, atol=COVARIANCE_F64.atol
    )
    np.testing.assert_allclose(
        f1, f0 @ rot.T, rtol=COVARIANCE_F64.rtol, atol=COVARIANCE_F64.atol
    )
    np.testing.assert_allclose(
        f0.sum(axis=0), 0.0, rtol=COVARIANCE_F64.rtol, atol=COVARIANCE_F64.atol
    )


def test_forces_are_minus_the_finite_difference_of_the_energy():
    calc = TPCalculator(model=one_layer())
    atoms = cluster()
    atoms.calc = calc
    forces = atoms.get_forces()
    step = 1e-4
    numeric = np.zeros_like(forces)
    for i in range(len(atoms)):
        for k in range(3):
            e = []
            for sign in (+1.0, -1.0):
                shifted = atoms.copy()
                shifted.positions[i, k] += sign * step
                e.append(energy(calc, shifted))
            numeric[i, k] = -(e[0] - e[1]) / (2 * step)
    np.testing.assert_allclose(
        forces,
        numeric,
        rtol=FINITE_DIFFERENCE_F64.rtol,
        atol=FINITE_DIFFERENCE_F64.atol,
    )


def test_swapping_the_labels_of_two_atoms_does_not_change_the_energy():
    calc = TPCalculator(model=one_layer())
    atoms = cluster()
    permuted = atoms.copy()
    order = [1, 0, 3, 2]  # swap within each species
    permuted.positions = atoms.positions[order]
    np.testing.assert_allclose(
        energy(calc, permuted),
        energy(calc, atoms),
        rtol=COVARIANCE_F64.rtol,
        atol=COVARIANCE_F64.atol,
    )
