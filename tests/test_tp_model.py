import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk

from tensorpotential.instructions.base import LORAInstructionMixin

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

import tensorflow as tf

from tensorpotential.utils import convert_model_reduce_elements
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import (
    ExtractBasisFunctions,
    TPModel,
    _get_bond_cutoff_map,
    _get_cutoff,
    _get_element_map,
    extract_cutoff_and_elements,
    extract_cutoff_dict,
    extract_cutoff_matrix,
    load_model_metadata,
)

from tensorpotential.potentials import get_preset, presets
from tests.tolerances import FLOAT64_ARITHMETIC
from tensorpotential.instructions import (
    load_instructions,
    save_instructions_dict,
)
from tensorpotential.utils import get_dtype_by_name
from tensorpotential.metadata_utils import read_model_metadata
from tensorpotential.calculator import TPCalculator

from ase import Atoms

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


# TODO: maybe parameterise with @pytest.mark.parametrize
def test_tp_model_repr_verbose_0():
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    list_of_instructions = GRACE_2LAYER(
        element_map={"Be": 0, "Li": 1}, lmax=2
    ).get_instructions()

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=0)
    print("Your summary (verbose = 0) is:\n", res, sep="\n")
    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are NOT present
    assert "name=" not in res
    assert "l=" not in res

    # assert Coupling functions table is NOT present
    assert "Coupling functions" not in res
    assert " parity " not in res

    # assert Coupling data table is NOT present
    assert "left_inds" not in res
    assert "right_inds" not in res
    assert "cg_list" not in res


def test_tp_model_repr_verbose_1():
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    list_of_instructions = GRACE_2LAYER(
        element_map={"Be": 0, "Li": 1}, lmax=2
    ).get_instructions()

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=1)
    print("Your summary (verbose=1) is:\n", res, sep="\n")

    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are present
    assert "name=" in res
    assert "l=" in res

    # assert Coupling functions table
    assert "Coupling Functions" in res
    assert " parity " in res

    # assert Coupling data columns are NOT present
    assert "left_inds" not in res
    assert "right_inds" not in res
    assert "cg_list" not in res


def test_tp_model_repr_verbose_2():
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    list_of_instructions = GRACE_2LAYER(
        element_map={"Be": 0, "Li": 1}, lmax=2
    ).get_instructions()

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=2)
    print("Your summary (verbose=2) is:\n", res, sep="\n")

    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are present
    assert "name=" in res
    assert "l=" in res

    # assert Coupling functions table is not included
    assert "Coupling Functions" not in res

    # assert Coupling data table
    assert "Coupling Table" in res
    assert " parity " in res
    assert "left_inds" in res
    assert "right_inds" in res
    assert "cg_list" in res


def test_set_trainable_variables():
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    list_of_instructions = GRACE_2LAYER(
        element_map={"Be": 0, "Li": 1}, lmax=2
    ).get_instructions()
    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    Z = tp.instructions["Z"]
    assert len(Z.trainable_variables) > 0
    trainable_names = []
    tp.set_trainable_variables(trainable_names)

    assert len(Z.trainable_variables) == 0


def test_convert_model_reduce_elements(tmp_path):
    TEST_PATH = str(tmp_path / "test_checkpoints")
    os.makedirs(TEST_PATH, exist_ok=True)

    float_dtype = tf.float64
    # stage 1: convert_model_reduce_elements
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3},
        lmax=0,
        basis_type="Cheb",
        cutoff_dict={"Mo": 4, "Nb": 5, "Ta": 6, "W": 7},
    ).get_instructions()
    tp = TensorPotential(instructions, param_dtype=float_dtype)

    potential_file_name = os.path.join(TEST_PATH, "model.MoNbTaW.yaml")
    checkpoint_name = os.path.join(TEST_PATH, "checkpoint.MoNbTaW")

    save_instructions_dict(potential_file_name, instructions, param_dtype=float_dtype)
    tp.save_checkpoint(checkpoint_name=checkpoint_name)

    assert os.path.isfile(potential_file_name)
    assert os.path.isfile(checkpoint_name + ".index")

    new_potential_file_name = os.path.join(TEST_PATH, "model.MoTa.yaml")
    new_checkpoint_name = os.path.join(TEST_PATH, "checkpoint.MoTa")

    assert not os.path.isfile(new_potential_file_name)
    assert not os.path.isfile(new_checkpoint_name + ".index")

    convert_model_reduce_elements(
        element_map={"Mo": 0, "Ta": 1},
        potential_file_name=potential_file_name,
        checkpoint_name=checkpoint_name,
        new_potential_file_name=new_potential_file_name,
        new_checkpoint_name=new_checkpoint_name,
        param_dtype=float_dtype,
    )

    assert os.path.isfile(new_potential_file_name)
    assert os.path.isfile(new_checkpoint_name + ".index")

    # stage 2: compare predictions — read param_dtype from model.yaml metadata
    instructions1 = load_instructions(potential_file_name)
    meta1 = read_model_metadata(potential_file_name)
    dtype1 = (
        get_dtype_by_name(meta1["param_dtype"])
        if "param_dtype" in meta1
        else tf.float64
    )
    model1 = TensorPotential(instructions1, param_dtype=dtype1)
    model1.load_checkpoint(checkpoint_name=checkpoint_name)

    instructions2 = load_instructions(new_potential_file_name)
    meta2 = read_model_metadata(new_potential_file_name)
    dtype2 = (
        get_dtype_by_name(meta2["param_dtype"])
        if "param_dtype" in meta2
        else tf.float64
    )
    model2 = TensorPotential(instructions2, param_dtype=dtype2)
    model2.load_checkpoint(checkpoint_name=new_checkpoint_name)

    calc1 = TPCalculator(model1.model)
    calc2 = TPCalculator(model2.model)

    at = Atoms("MoTa", positions=[[0, 0, 0], [0, 0, 2]], pbc=False)

    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)

    assert np.allclose(e1, e2)

    # structure that does not work for potential 2
    at2 = Atoms("MoNb", positions=[[0, 0, 0], [0, 0, 2]], pbc=False)

    at2.calc = calc1
    e1 = at2.get_potential_energy()
    print("e1=", e1)

    at2.calc = calc2
    with pytest.raises(AssertionError):
        e2 = at2.get_potential_energy()


@pytest.mark.parametrize(
    "param_dtype,expected_str",
    [
        (tf.float32, "float32"),
        (tf.float64, "float64"),
    ],
)
def test_save_load_model_metadata_param_dtype(tmp_path, param_dtype, expected_str):
    """Verify param_dtype round-trips through model.yaml metadata."""
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    instructions = GRACE_2LAYER(
        element_map={"Cu": 0, "Zn": 1}, lmax=0, basis_type="Cheb"
    ).get_instructions()

    model_path = str(tmp_path / "model.yaml")

    # save with param_dtype metadata
    save_instructions_dict(model_path, instructions, param_dtype=param_dtype)

    # read metadata and verify
    metadata = read_model_metadata(model_path)
    assert "param_dtype" in metadata, "param_dtype missing from model.yaml metadata"
    assert metadata["param_dtype"] == expected_str

    # load instructions — should work without error (metadata is skipped)
    loaded = load_instructions(model_path)
    assert isinstance(loaded, dict)
    assert len(loaded) == len(instructions)


def test_load_model_metadata_missing_for_old_models(tmp_path):
    """Old model.yaml without metadata should return empty dict from read_model_metadata."""
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    instructions = GRACE_2LAYER(
        element_map={"Cu": 0, "Zn": 1}, lmax=0, basis_type="Cheb"
    ).get_instructions()

    model_path = str(tmp_path / "model.yaml")

    # save WITHOUT param_dtype (simulating old behavior)
    save_instructions_dict(model_path, instructions)

    metadata = read_model_metadata(model_path)
    assert "saved_at" in metadata
    assert "tensorpotential_version" in metadata

    # load should still work
    loaded = load_instructions(model_path)
    assert isinstance(loaded, dict)
    assert len(loaded) == len(instructions)


@pytest.mark.xfail
def test_activate_reduce_lora():
    LORA_CONFIG = {"rank": 4, "alpha": 1}
    float_dtype = tf.float64
    GRACE_2LAYER = get_preset("GRACE_2LAYER")
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    )
    tp = TensorPotential(instructions, param_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_no_lora")

    at = bulk("Mo")

    calc1 = TPCalculator(tp.model)
    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    instructions = tp.model.instructions
    Z = instructions["Z"]
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars=", Z_trainable_vars)
    assert Z_trainable_vars == ["Z/ChemicalEmbedding:0"]

    lora_config = {
        ins_name: LORA_CONFIG
        for ins_name, ins in instructions.items()
        if isinstance(ins, LORAInstructionMixin)
    }
    print("lora_config=", lora_config)
    for k, v in lora_config.items():
        print(" -", k)

    tp.enable_lora_adaptation(lora_config)

    assert tp.is_lora_enabled()
    assert Z.lora
    assert hasattr(Z, "lora_tensors")

    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_with_lora")

    calc2 = TPCalculator(tp.model)
    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)
    assert np.allclose(e1, e2)

    Z_trainable_vars_after = [var.name for var in Z.trainable_variables]

    print("Z_trainable_vars_after=", Z_trainable_vars_after)
    assert Z_trainable_vars_after == ["Z/w/LORA/A:0", "Z/w/LORA/B:0"]

    for t in Z.lora_tensors:
        t.assign(t + tf.ones_like(t))

    calc3 = TPCalculator(tp.model)
    at.calc = calc3
    e2b = at.get_potential_energy()
    print("e2b=", e2b)

    assert not np.allclose(e1, e2b)

    ### stage 2 : reduce LORA
    tp.finalize_lora_update()

    assert not tp.is_lora_enabled()
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars_2 = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars_2=", Z_trainable_vars_2)
    assert Z_trainable_vars_2 == ["Z/ChemicalEmbedding:0"]

    calc4 = TPCalculator(tp.model)
    at.calc = calc4
    e4 = at.get_potential_energy()
    print("e4=", e4)

    assert np.allclose(e4, e2b)


@pytest.mark.xfail
def test_activate_reduce_additive():
    LORA_CONFIG = {"mode": "full_additive"}
    float_dtype = tf.float64
    GRACE_2LAYER = get_preset("GRACE_2LAYER")
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    )
    tp = TensorPotential(instructions, param_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_no_lora")

    at = bulk("Mo")

    calc1 = TPCalculator(tp.model)
    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    instructions = tp.model.instructions
    Z = instructions["Z"]
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars=", Z_trainable_vars)
    assert Z_trainable_vars == ["Z/ChemicalEmbedding:0"]

    lora_config = {
        ins_name: LORA_CONFIG
        for ins_name, ins in instructions.items()
        if isinstance(ins, LORAInstructionMixin)
    }
    print("lora_config=", lora_config)
    for k, v in lora_config.items():
        print(" -", k)

    tp.enable_lora_adaptation(lora_config)

    assert tp.is_lora_enabled()
    assert Z.lora
    assert hasattr(Z, "lora_tensors")

    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_with_lora")

    calc2 = TPCalculator(tp.model)
    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)
    assert np.allclose(e1, e2)

    Z_trainable_vars_after = [var.name for var in Z.trainable_variables]

    print("Z_trainable_vars_after=", Z_trainable_vars_after)
    assert Z_trainable_vars_after == ["Z/w/ADDITIVE/delta_W:0"]

    for t in Z.lora_tensors:
        t.assign(t + tf.ones_like(t))

    calc3 = TPCalculator(tp.model)
    at.calc = calc3
    e2b = at.get_potential_energy()
    print("e2b=", e2b)

    assert not np.allclose(e1, e2b)

    ### stage 2 : reduce LORA
    tp.finalize_lora_update()

    assert not tp.is_lora_enabled()
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars_2 = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars_2=", Z_trainable_vars_2)
    assert Z_trainable_vars_2 == ["Z/ChemicalEmbedding:0"]

    calc4 = TPCalculator(tp.model)
    at.calc = calc4
    e4 = at.get_potential_energy()
    print("e4=", e4)

    assert np.allclose(e4, e2b)


def test_extract_basis_functions_GRACE_1LAYER_latest():
    instr_1L = get_preset("GRACE_1LAYER_latest")(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0
    ).get_instructions()
    tp = TensorPotential(
        instr_1L,
        model_compute_function=ExtractBasisFunctions(
        ),
    )
    calc = TPCalculator(
        model=tp.model, extra_properties=["1L_basis"], truncate_extras_by_natoms=True
    )
    at = bulk("Mo") * (1, 1, 3)
    at.rattle(0.01)
    at.calc = calc
    at.get_potential_energy()
    projs = at.calc.results["1L_basis"]
    print(f"Shape projs: {projs.shape}")
    assert len(projs.shape) == 2
    assert projs.shape[0] == len(at)
    assert projs.shape[1] == 224


def test_extract_basis_functions_GRACE_2LAYER_latest():

    instr_2L = get_preset("GRACE_2LAYER_latest")(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0
    ).get_instructions()
    tp = TensorPotential(
        instr_2L,
        model_compute_function=ExtractBasisFunctions(
            extract_2L_basis=True
        ),
    )
    calc = TPCalculator(
        model=tp.model,
        extra_properties=["1L_basis", "2L_basis"],
        truncate_extras_by_natoms=True,
    )
    at = bulk("Mo") * (1, 1, 3)
    at.rattle(0.01)
    at.calc = calc
    at.get_potential_energy()
    projs1 = at.calc.results["1L_basis"]
    projs2 = at.calc.results["2L_basis"]
    print(f"Shape projs-1L: {projs1.shape}")
    print(f"Shape projs-2L: {projs2.shape}")
    assert len(projs1.shape) == 2
    assert len(projs2.shape) == 2
    assert projs1.shape[0] == len(at)
    assert projs2.shape[0] == len(at)

    assert projs1.shape[1] == 168
    assert projs2.shape[1] == 256


def test_repr_numbers_the_instructions_in_file_order(cu_two_layer):
    # TensorFlow formats repr(model) in its "retraced too often" warning, which used to be
    # the only thing that ever ran __repr__; this pins it without depending on that warning.
    model = cu_two_layer(False)
    lines = repr(model).split("\n")
    names = list(model.instructions)
    assert len(lines) == len(names) > 10
    assert lines == [f"{i + 1}. {name!r}" for i, name in enumerate(names)]


# ------------------------------------------------- default functions are not shared


def _small_instructions():
    preset = get_preset("GRACE_2LAYER_v1_24")
    return preset(element_map={"Be": 0, "Li": 1}, lmax=0).get_instructions()


def test_each_tpmodel_gets_its_own_default_functions():
    from tensorpotential.tpmodel import (
        ComputeBatchEnergyAndForces,
        ComputeStructureEnergyAndForcesAndVirial,
    )

    first = TPModel(_small_instructions())
    second = TPModel(_small_instructions())

    assert isinstance(first.compute_function, ComputeStructureEnergyAndForcesAndVirial)
    assert isinstance(first.train_function, ComputeBatchEnergyAndForces)
    assert first.compute_function is not second.compute_function
    assert first.train_function is not second.train_function


def test_tpmodel_keeps_explicit_functions_by_identity():
    from tensorpotential.tpmodel import (
        ComputeBatchEnergyForcesVirials,
        ComputeStructureEnergyAndForcesAndVirial,
    )

    compute = ComputeStructureEnergyAndForcesAndVirial()
    train = ComputeBatchEnergyForcesVirials()
    model = TPModel(
        _small_instructions(), compute_function=compute, train_function=train
    )

    assert model.compute_function is compute
    assert model.train_function is train


def test_tpmodel_function_arguments_default_to_none():
    import inspect

    parameters = inspect.signature(TPModel.__init__).parameters

    assert [parameters[n].default for n in ("compute_function", "train_function")] == [
        None,
        None,
    ]



# --------------------------------------------------------- cutoffs, element maps and saved-model metadata
# Hand-computed oracle: GRACE_1LAYER_v1_24 with ``cutoff_dict={"CuCu": 3, "ZnZn": 4, "CuZn": 5}`` and
# ``symmetric_bond`` cutoffs has the pair-cutoff matrix [[3, 5], [5, 4]] in element-index order (Cu = 0, Zn = 1),
# and a pair that the dict does not name gets the default cutoff ``rcut``.

CU_ZN_CUTOFFS = {"CuCu": 3.0, "ZnZn": 4.0, "CuZn": 5.0}
CU_ZN_DICT = {("Cu", "Cu"): 3.0, ("Cu", "Zn"): 5.0, ("Zn", "Cu"): 5.0, ("Zn", "Zn"): 4.0}
CU_ZN_MATRIX = [[3.0, 5.0], [5.0, 4.0]]


def _bond_cutoff_model(element_map=None, cutoff_dict=None, rcut=6.0, **kwargs) -> TPModel:
    """A built float64 1-layer model whose radial basis has one cutoff per element pair."""
    instructions = presets.GRACE_1LAYER_v1_24(
        element_map=element_map or {"Cu": 0, "Zn": 1},
        cutoff_dict=CU_ZN_CUTOFFS if cutoff_dict is None else cutoff_dict,
        rcut=rcut,
        lmax=1,
        n_rad_max=4,
        embedding_size=4,
        n_mlp_dens=2,
        **kwargs,
    ).get_instructions()
    model = TPModel(instructions)
    model.build(tf.float64)
    return model


def _linear_model(rcut=4.5) -> TPModel:
    """A built float64 LINEAR model with one cutoff for all pairs."""
    instructions = presets.LINEAR(
        element_map={"Cu": 0, "Zn": 1}, rcut=rcut, lmax=1, n_rad_max=4, embedding_size=4
    ).get_instructions()
    model = TPModel(instructions)
    model.build(tf.float64)
    return model


def _saved_metadata(path: str) -> dict:
    """``load_model_metadata`` of a directory that must have a metadata file."""
    metadata = load_model_metadata(path)
    assert metadata is not None
    return metadata


@pytest.fixture(scope="module")
def saved_bond_model(tmp_path_factory):
    """(path of the SavedModel, the model it was saved from): saving takes seconds, so once per module."""
    model = _bond_cutoff_model()
    path = str(tmp_path_factory.mktemp("saved") / "bond_model")
    model.save_model(path)
    return path, model


@pytest.fixture(scope="module")
def saved_linear_model(tmp_path_factory):
    model = _linear_model()
    path = str(tmp_path_factory.mktemp("saved") / "linear_model")
    model.save_model(path)
    return path, model


# -------------------------------------------------------------------------------------------- _get_cutoff


def test_get_cutoff_prefers_the_method_of_the_instruction():
    # ``SimpleNamespace`` stands for a restored SavedModel object or an instruction with every attribute set
    both = SimpleNamespace(get_cutoff=lambda: 3.25, rc=tf.constant(9.0, tf.float64))

    assert _get_cutoff(both) == 3.25


def test_get_cutoff_method_returning_none_is_not_replaced_by_the_attribute():
    instruction = SimpleNamespace(get_cutoff=lambda: None, rc=tf.constant(9.0, tf.float64))

    assert _get_cutoff(instruction) is None


def test_get_cutoff_non_callable_attribute_falls_back_to_rc():
    instruction = SimpleNamespace(get_cutoff=2.0, rc=tf.constant(4.5, tf.float64))

    assert _get_cutoff(instruction) == 4.5


def test_get_cutoff_reads_rc_as_a_python_float():
    cutoff = _get_cutoff(SimpleNamespace(rc=tf.Variable(4.5, dtype=tf.float64)))

    assert cutoff == 4.5
    assert type(cutoff) is float


def test_get_cutoff_legacy_radial_basis_stores_the_cutoff_in_basis_function():
    legacy = SimpleNamespace(basis_function=SimpleNamespace(rc=tf.constant(3.5, tf.float64)))

    assert _get_cutoff(legacy) == 3.5


def test_get_cutoff_rc_wins_over_the_legacy_location():
    instruction = SimpleNamespace(
        rc=tf.constant(4.0, tf.float64), basis_function=SimpleNamespace(rc=tf.constant(3.5, tf.float64))
    )

    assert _get_cutoff(instruction) == 4.0


@pytest.mark.parametrize(
    "instruction", [SimpleNamespace(), SimpleNamespace(basis_function=SimpleNamespace()), object()]
)
def test_get_cutoff_without_any_cutoff_is_none(instruction):
    assert _get_cutoff(instruction) is None


def test_get_cutoff_of_real_instructions_of_a_built_model():
    model = _linear_model(rcut=4.5)

    cutoffs = {name: _get_cutoff(ins) for name, ins in model.instructions.items()}

    assert cutoffs["RadialBasis"] == 4.5
    assert cutoffs["R"] is None  # the radial function has no cutoff of its own
    assert max(c for c in cutoffs.values() if c is not None) == 4.5


def test_get_cutoff_of_a_restored_saved_model_reads_rc(saved_linear_model):
    path, _ = saved_linear_model
    restored = tf.saved_model.load(path)

    assert _get_cutoff(restored.instructions["RadialBasis"]) == 4.5
    assert _get_cutoff(restored.instructions["R"]) is None


def test_get_cutoff_of_a_restored_bond_cutoff_instruction_is_none(saved_bond_model):
    # PINNED: a restored BondSpecificRadialBasisFunction has bond_cutoff_map and no rc, so the scalar cutoff of a
    # restored bond-cutoff model is 0 (the maximum over no value); the calculator uses the matrix instead
    path, _ = saved_bond_model
    restored = tf.saved_model.load(path)

    assert _get_cutoff(restored.instructions["BondSpecificRadialBasisFunction"]) is None
    assert extract_cutoff_and_elements(restored.instructions)[0] == 0.0


# ----------------------------------------------------------------------- element map and bond cutoff map


def test_get_element_map_of_a_real_instruction_and_of_a_restored_one(saved_bond_model):
    path, model = saved_bond_model
    restored = tf.saved_model.load(path)

    for ins in (model.instructions["Z"], restored.instructions["Z"]):
        element_map = _get_element_map(ins)
        assert element_map is not None
        symbols, index = element_map
        assert list(symbols) == ["Cu", "Zn"]
        assert list(index) == [0, 1]
    assert _get_element_map(model.instructions["R"]) is None
    assert _get_element_map(SimpleNamespace()) is None


def test_get_bond_cutoff_map_is_the_flat_column_of_the_matrix(saved_bond_model):
    path, model = saved_bond_model
    restored = tf.saved_model.load(path)
    expected = np.array(CU_ZN_MATRIX).reshape(-1, 1)

    np.testing.assert_array_equal(_get_bond_cutoff_map(model.instructions["BondSpecificRadialBasisFunction"]), expected)
    np.testing.assert_array_equal(
        _get_bond_cutoff_map(restored.instructions["BondSpecificRadialBasisFunction"]), expected
    )
    assert _get_bond_cutoff_map(model.instructions["R"]) is None
    assert _get_bond_cutoff_map(SimpleNamespace()) is None


# ------------------------------------------------------------------- extract_cutoff_and_elements / matrix


def test_extract_cutoff_and_elements_takes_the_largest_cutoff_and_the_last_element_map():
    first = SimpleNamespace(get_cutoff=lambda: 3.0, get_element_map=lambda: (np.array(["A"]), np.array([0])))
    second = SimpleNamespace(get_cutoff=lambda: 5.5, get_element_map=lambda: (np.array(["B"]), np.array([7])))
    third = SimpleNamespace(get_cutoff=lambda: 4.0)

    cutoff, symbols, index = extract_cutoff_and_elements({"a": first, "b": second, "c": third})

    assert cutoff == 5.5
    assert list(symbols) == ["B"]
    assert list(index) == [7]


def test_extract_cutoff_and_elements_of_nothing_has_cutoff_zero_and_no_elements():
    cutoff, symbols, index = extract_cutoff_and_elements([SimpleNamespace()])

    assert (cutoff, symbols, index) == (0.0, None, None)


def test_extract_cutoff_matrix_is_the_elementwise_maximum_reshaped_to_a_square():
    one = SimpleNamespace(get_bond_cutoff_map=lambda: np.array([[1.0], [5.0], [2.0], [4.0]]))
    two = SimpleNamespace(get_bond_cutoff_map=lambda: np.array([[3.0], [2.0], [2.5], [1.0]]))

    matrix = extract_cutoff_matrix([one, SimpleNamespace(), two])

    np.testing.assert_array_equal(matrix, [[3.0, 5.0], [2.5, 4.0]])


def test_extract_cutoff_matrix_without_bond_cutoffs_is_none():
    assert extract_cutoff_matrix(_linear_model().instructions) is None


def test_extract_cutoff_matrix_that_is_not_square_fails_an_assertion():
    three = SimpleNamespace(get_bond_cutoff_map=lambda: np.ones((3, 1)))

    with pytest.raises(AssertionError):
        extract_cutoff_matrix([three])


# ---------------------------------------------------------------------------------- extract_cutoff_dict


def test_extract_cutoff_dict_of_a_model_with_pair_cutoffs():
    model = _bond_cutoff_model()

    assert extract_cutoff_dict(model.instructions) == CU_ZN_DICT
    assert extract_cutoff_dict(list(model.instructions.values())) == CU_ZN_DICT
    np.testing.assert_array_equal(extract_cutoff_matrix(model.instructions), CU_ZN_MATRIX)


def test_extract_cutoff_dict_fills_unnamed_pairs_with_the_default_cutoff():
    model = _bond_cutoff_model(
        element_map={"Al": 0, "Cu": 1, "Zn": 2}, cutoff_dict={"AlAl": 2.0, "CuZn": 3.0}, rcut=4.0
    )
    matrix = [[2.0, 4.0, 4.0], [4.0, 4.0, 3.0], [4.0, 3.0, 4.0]]
    symbols = ["Al", "Cu", "Zn"]

    result = extract_cutoff_dict(model.instructions)

    assert result == {(a, b): matrix[i][j] for i, a in enumerate(symbols) for j, b in enumerate(symbols)}
    assert len(result) == 9


def test_extract_cutoff_dict_values_are_python_floats_and_keys_are_str_pairs():
    result = extract_cutoff_dict(_bond_cutoff_model().instructions)

    assert all(type(k[0]) is str and type(k[1]) is str for k in result)
    assert all(type(v) is float for v in result.values())


def test_extract_cutoff_dict_without_bond_cutoffs_is_none():
    assert extract_cutoff_dict(_linear_model().instructions) is None


def test_extract_cutoff_dict_without_an_element_map_is_none():
    model = _bond_cutoff_model()
    only_the_radial_basis = [model.instructions["BondSpecificRadialBasisFunction"]]

    assert extract_cutoff_matrix(only_the_radial_basis) is not None
    assert extract_cutoff_dict(only_the_radial_basis) is None


def test_extract_cutoff_dict_of_a_restored_saved_model_equals_the_original(saved_bond_model):
    path, _ = saved_bond_model

    assert extract_cutoff_dict(tf.saved_model.load(path).instructions) == CU_ZN_DICT


def test_extract_cutoff_dict_labels_rows_by_dict_order_not_by_element_index():
    # PINNED, not endorsed (TEST8 finding): with the same indices (Cu = 0, Zn = 1) but the element map written
    # Zn first, the symbols are listed in insertion order, so Cu-Cu and Zn-Zn swap (the matrix is still in index order)
    model = _bond_cutoff_model(element_map={"Zn": 1, "Cu": 0})

    np.testing.assert_array_equal(extract_cutoff_matrix(model.instructions), CU_ZN_MATRIX)
    assert extract_cutoff_dict(model.instructions) == {
        ("Zn", "Zn"): 3.0,
        ("Zn", "Cu"): 5.0,
        ("Cu", "Zn"): 5.0,
        ("Cu", "Cu"): 4.0,
    }


# ------------------------------------------------------------------------ load_model_metadata, save_model


def test_load_model_metadata_of_a_saved_model_directory():
    path = os.path.join(os.path.dirname(__file__), "test_calculator_model")

    assert load_model_metadata(path) == {"chemical_symbols": ["Mo", "Nb", "Ta", "W"], "cutoff": 6.0}


def test_load_model_metadata_without_the_file_is_none(tmp_path):
    assert load_model_metadata(str(tmp_path)) is None
    assert load_model_metadata(str(tmp_path / "missing")) is None


def test_load_model_metadata_of_something_that_is_not_a_str_is_none(tmp_path):
    # PINNED: a pathlib.Path is not a str, so a directory given as a Path has no metadata either
    (tmp_path / "metadata.yaml").write_text("cutoff: 5.0\n")

    assert load_model_metadata(str(tmp_path)) == {"cutoff": 5.0}
    assert load_model_metadata(tmp_path) is None
    assert load_model_metadata(None) is None
    assert load_model_metadata(_linear_model()) is None


def test_save_model_writes_symbols_cutoff_and_the_pair_cutoff_matrix(saved_bond_model):
    path, _ = saved_bond_model

    metadata = load_model_metadata(path)

    assert metadata == {"chemical_symbols": ["Cu", "Zn"], "cutoff": 5.0, "cutoff_matrix": CU_ZN_MATRIX}


def test_save_model_json_metadata_equals_the_yaml_metadata(saved_bond_model):
    path, _ = saved_bond_model

    with open(os.path.join(path, "metadata.json")) as f:
        as_json = json.load(f)

    assert as_json == load_model_metadata(path)


def test_save_model_without_pair_cutoffs_has_no_matrix_and_the_scalar_cutoff(saved_linear_model):
    path, _ = saved_linear_model

    assert load_model_metadata(path) == {"chemical_symbols": ["Cu", "Zn"], "cutoff": 4.5}


def test_save_model_lists_the_symbols_in_element_index_order(tmp_path):
    # the element map is written Zn first; the saved symbols follow the indices (Cu = 0, Zn = 1) and the matrix too
    model = _bond_cutoff_model(element_map={"Zn": 1, "Cu": 0})
    path = str(tmp_path / "model")

    model.save_model(path)

    metadata = _saved_metadata(path)
    assert metadata["chemical_symbols"] == ["Cu", "Zn"]
    assert metadata["cutoff_matrix"] == CU_ZN_MATRIX
    assert metadata["cutoff"] == 5.0


def test_save_model_without_a_cutoff_matrix_uses_the_largest_instruction_cutoff(tmp_path):
    model = _linear_model(rcut=3.75)
    path = str(tmp_path / "model")

    model.save_model(path)

    assert _saved_metadata(path)["cutoff"] == 3.75


def test_save_model_has_no_uq_key_without_a_uq_model(saved_bond_model):
    path, _ = saved_bond_model

    assert "has_uq" not in load_model_metadata(path)
    assert "parallel_communication" not in load_model_metadata(path)


def test_save_model_refuses_a_uq_model_without_thresholds(tmp_path):
    # ``SimpleNamespace`` stands for a GMM artifact: the check reads one attribute and runs before anything is saved
    model = _linear_model()

    with pytest.raises(ValueError, match="interp_thresholds must be set before exporting to SavedModel"):
        model.save_model(str(tmp_path / "model"), gmm_uq_model=SimpleNamespace(interp_thresholds=None))

    assert not (tmp_path / "model").exists()


def test_save_model_refuses_a_uq_model_that_is_not_a_basis_rp_artifact(tmp_path):
    model = _linear_model()
    artifact = SimpleNamespace(interp_thresholds=[1.0], extra_data={})

    with pytest.raises(ValueError, match="SavedModel export requires a basis-RP UQ artifact"):
        model.save_model(str(tmp_path / "model"), gmm_uq_model=artifact)

    assert not (tmp_path / "model").exists()


def test_saved_model_reloads_with_the_same_energy(saved_bond_model):
    # physical value: the saved and reloaded model is the same function as the model it was saved from
    path, model = saved_bond_model
    atoms = Atoms("CuZnCu", positions=[[0, 0, 0], [0, 0, 2.3], [1.9, 0.3, 4.0]], pbc=False)
    atoms.calc = TPCalculator(model=model)
    expected = atoms.get_potential_energy()
    reloaded = Atoms(atoms)
    reloaded.calc = TPCalculator(model=path)

    assert reloaded.get_potential_energy() == pytest.approx(
        expected, rel=FLOAT64_ARITHMETIC.rtol, abs=FLOAT64_ARITHMETIC.atol
    )
