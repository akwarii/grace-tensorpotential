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
from tensorpotential import constants
from tensorpotential.tpmodel import (
    ComputeBatchEnergyAndForces,
    ComputeBatchEnergyForcesVirials,
    ComputeBlock,
    ComputeBlockInputGradient,
    ComputeBlockOutputGradient,
    ComputeEnergy,
    ComputeEquivariantForces,
    ComputeFunction,
    ComputePlaceholder,
    ComputeStructureEnergyAndForcesAndVirial,
    ExtractBasisFunctions,
    TPModel,
    TrainFunction,
    execute_instructions,
    to_instructions_list,
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


def test_activate_reduce_lora(tmp_path):
    LORA_CONFIG = {"rank": 4, "alpha": 1}
    float_dtype = tf.float64
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    ).get_instructions()
    tp = TensorPotential(instructions, param_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name=str(tmp_path / "checkpoint_no_lora"))

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

    tp.save_checkpoint(checkpoint_name=str(tmp_path / "checkpoint_with_lora"))

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


def test_activate_reduce_additive(tmp_path):
    LORA_CONFIG = {"mode": "full_additive"}
    float_dtype = tf.float64
    GRACE_2LAYER = get_preset("GRACE_2LAYER_v1_24")
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    ).get_instructions()
    tp = TensorPotential(instructions, param_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name=str(tmp_path / "checkpoint_no_lora"))

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

    tp.save_checkpoint(checkpoint_name=str(tmp_path / "checkpoint_with_lora"))

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


# ---------------------------------------- LoRA activation at the level of the model (characterization)
#
# ``TPModel.enable_lora_adaptation`` / ``finalize_lora_update`` / ``is_lora_enabled`` decide which
# instructions get low-rank update tensors; the numerics of one instruction are tested in
# test_compute.py, test_output.py and test_functions_nn.py.

LORA_RANK_4 = {"rank": 4, "alpha": 1}
LORA_RANK_2 = {"rank": 2, "alpha": 1}


def _lora_model() -> TPModel:
    instructions = get_preset("GRACE_2LAYER_v1_24")(
        element_map={"Mo": 0, "Nb": 1}, lmax=0
    ).get_instructions()
    model = TPModel(instructions)
    model.build(tf.float64)
    return model


def _lora_capable(model: TPModel) -> set[str]:
    return {
        name
        for name, ins in model.instructions.items()
        if isinstance(ins, LORAInstructionMixin)
    }


def _lora_active(model: TPModel) -> set[str]:
    return {
        name
        for name, ins in model.instructions.items()
        if isinstance(ins, LORAInstructionMixin) and ins.lora
    }


def _weights(model: TPModel) -> dict[str, np.ndarray]:
    return {v.name: v.numpy().copy() for v in model.variables if v.dtype == tf.float64}


def test_a_fresh_model_has_lora_capable_instructions_and_none_active():
    model = _lora_model()

    assert {"R", "Z", "A", "LinMLPOut2ScalarTarget"} <= _lora_capable(model)
    assert not model.is_lora_enabled()
    assert _lora_active(model) == set()


def test_enabling_with_no_config_changes_nothing():
    model = _lora_model()
    before = _weights(model)

    model.enable_lora_adaptation(None)

    assert not model.is_lora_enabled()
    assert before.keys() == _weights(model).keys()


def test_enabling_all_activates_every_lora_capable_instruction_and_only_those():
    model = _lora_model()
    capable = _lora_capable(model)
    assert capable < set(model.instructions)  # the preset also holds instructions without LoRA

    model.enable_lora_adaptation({"all": dict(LORA_RANK_4)})

    assert model.is_lora_enabled()
    assert _lora_active(model) == capable
    assert all(model.instructions[n].lora_config == LORA_RANK_4 for n in capable)


def test_an_explicit_entry_overrides_the_all_entry():
    model = _lora_model()

    model.enable_lora_adaptation({"all": dict(LORA_RANK_4), "Z": dict(LORA_RANK_2)})

    Z = model.instructions["Z"]
    assert Z.lora_config == LORA_RANK_2
    assert [t.shape[-1] for t in Z.lora_tensors] == [2, 2]  # one rank-2 factor per matrix side
    assert model.instructions["R"].lora_config == LORA_RANK_4


def test_only_the_named_instructions_are_activated():
    model = _lora_model()

    model.enable_lora_adaptation({"Z": dict(LORA_RANK_2), "R": dict(LORA_RANK_4)})

    assert _lora_active(model) == {"Z", "R"}


def test_empty_unknown_and_non_lora_entries_are_skipped():
    model = _lora_model()
    not_capable = sorted(set(model.instructions) - _lora_capable(model))[0]

    model.enable_lora_adaptation(
        {"Z": {}, "no_such_instruction": dict(LORA_RANK_4), not_capable: dict(LORA_RANK_4)}
    )

    assert not model.is_lora_enabled()
    assert not getattr(model.instructions[not_capable], "lora", False)


def test_enabling_freezes_the_original_weight_and_trains_the_update_tensors():
    model = _lora_model()
    Z = model.instructions["Z"]
    assert [v.name for v in Z.trainable_variables] == ["Z/ChemicalEmbedding:0"]

    model.enable_lora_adaptation({"Z": dict(LORA_RANK_2)})

    assert [v.name for v in Z.trainable_variables] == ["Z/w/LORA/A:0", "Z/w/LORA/B:0"]


def test_finalizing_reduces_every_active_instruction_and_restores_trainability():
    model = _lora_model()
    model.enable_lora_adaptation({"all": dict(LORA_RANK_4)})

    model.finalize_lora_update()

    assert not model.is_lora_enabled()
    assert _lora_active(model) == set()
    Z = model.instructions["Z"]
    assert not hasattr(Z, "lora_tensors")
    assert [v.name for v in Z.trainable_variables] == ["Z/ChemicalEmbedding:0"]
    assert "lora_config" not in Z._init_args


def test_finalizing_leaves_instructions_without_lora_untouched():
    model = _lora_model()
    model.enable_lora_adaptation({"Z": dict(LORA_RANK_2)})
    R_before = {v.name: v.numpy().copy() for v in model.instructions["R"].variables}

    model.finalize_lora_update()

    for v in model.instructions["R"].variables:
        np.testing.assert_array_equal(v.numpy(), R_before[v.name])


def test_finalizing_a_model_without_lora_changes_nothing():
    model = _lora_model()
    before = _weights(model)

    model.finalize_lora_update()

    after = _weights(model)
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_activation_is_written_into_the_saved_yaml_and_restored_from_it(tmp_path):
    model = _lora_model()
    model.enable_lora_adaptation({"all": dict(LORA_RANK_4)})
    path = str(tmp_path / "model.yaml")
    save_instructions_dict(path, model.instructions, param_dtype=tf.float64)

    restored = TPModel(load_instructions(path))
    restored.build(tf.float64)

    assert _lora_active(restored) == _lora_active(model)
    assert restored.instructions["Z"].lora_config == LORA_RANK_4
    assert sorted(v.name for v in restored.trainable_variables) == sorted(
        v.name for v in model.trainable_variables
    )


def test_enabling_does_not_edit_the_callers_config():
    model = _lora_model()
    config = {"all": dict(LORA_RANK_4), "Z": dict(LORA_RANK_2)}

    model.enable_lora_adaptation(config)

    assert config == {"all": LORA_RANK_4, "Z": LORA_RANK_2}


def test_enabling_twice_keeps_the_tensors_of_an_instruction_that_is_already_active():
    # a model restored from a yaml with LoRA, then given the same input again: nothing is re-initialised
    model = _lora_model()
    model.enable_lora_adaptation({"Z": dict(LORA_RANK_2)})
    Z = model.instructions["Z"]
    tensors = list(Z.lora_tensors)
    for t in tensors:
        t.assign(tf.ones_like(t))

    model.enable_lora_adaptation({"all": dict(LORA_RANK_4)})

    assert Z.lora_config == LORA_RANK_2
    assert all(a is b for a, b in zip(Z.lora_tensors, tensors))
    assert all(np.all(t.numpy() == 1.0) for t in Z.lora_tensors)
    assert model.instructions["R"].lora_config == LORA_RANK_4  # the others are activated as asked


# ------------------------------------------------------------------ the compute and train functions
#
# The functions turn the atomic energies an instruction list writes into energies, forces and virials. They are
# tested with a spring pair potential written as an instruction, E_i = k/2 sum_j |x_j - x_i|^2, whose results are
# known in closed form from the positions and never from the bonds: the energy of a structure is
# k sum_(i<j) |x_i - x_j|^2, the force on atom a is 2 k sum_(j != a) (x_j - x_a), and the virial is the
# tensor sum_a x_a (x) F_a of an open cluster (Voigt order xx, yy, zz, xy, xz, yz).

SPRING_K = 0.8
ISOTROPIC = np.eye(3)
# an anisotropic spring has pair forces that are not along the bond: its virial is not symmetric
ANISOTROPIC = np.array([[1.0, 0.3, 0.1], [0.3, 2.0, 0.4], [0.1, 0.4, 3.0]])
CLUSTERS = [  # the atoms of two structures of one batch
    np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.3], [-0.4, 1.2, 0.7]]),
    np.array([[0.2, 0.1, 0.0], [1.1, -0.7, 0.4]]),
]


class _Spring:
    """The spring pair potential as an instruction; it records the flags it was called with."""

    def __init__(self, k=SPRING_K, metric=ISOTROPIC):
        self.k = k
        self.metric = tf.constant(metric, tf.float64)
        self.calls = []

    def __call__(self, input_data, training=False, local=False):
        self.calls.append({"training": training, "local": local})
        r = input_data[constants.BOND_VECTOR]
        pair_energy = 0.5 * self.k * tf.einsum("bi,ij,bj->b", r, self.metric, r)
        n_atoms = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        input_data[constants.PREDICT_ATOMIC_ENERGY] = tf.math.unsorted_segment_sum(
            pair_energy, input_data[constants.BOND_IND_I], num_segments=n_atoms
        )


def _cluster_batch(structures: list[int]) -> dict[str, tf.Tensor]:
    """The tensors of a batch of the chosen clusters: every ordered pair of atoms of a structure is a bond."""
    ind_i, ind_j, vectors, bond_structure, atom_structure = [], [], [], [], []
    offset = 0
    for s, which in enumerate(structures):
        x = CLUSTERS[which]
        atom_structure += [s] * len(x)
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    ind_i.append(offset + i)
                    ind_j.append(offset + j)
                    vectors.append(x[j] - x[i])
                    bond_structure.append(s)
        offset += len(x)
    return {
        constants.BOND_IND_I: tf.constant(ind_i, tf.int32),
        constants.BOND_IND_J: tf.constant(ind_j, tf.int32),
        constants.BOND_VECTOR: tf.constant(np.array(vectors), tf.float64),
        constants.ATOMIC_MU_I: tf.zeros([offset], tf.int32),
        constants.ATOMS_TO_STRUCTURE_MAP: tf.constant(atom_structure, tf.int32),
        constants.BONDS_TO_STRUCTURE_MAP: tf.constant(bond_structure, tf.int32),
        constants.N_STRUCTURES_BATCH_TOTAL: tf.constant(len(structures), tf.int32),
        constants.N_ATOMS_BATCH_TOTAL: tf.constant(offset, tf.int32),
    }


def _spring_energy(x: np.ndarray, m: np.ndarray = ISOTROPIC) -> float:
    return SPRING_K * sum((x[i] - x[j]) @ m @ (x[i] - x[j]) for i in range(len(x)) for j in range(i))


def _spring_forces(x: np.ndarray, m: np.ndarray = ISOTROPIC) -> np.ndarray:
    return np.array([2 * SPRING_K * m @ sum(x[j] - x[a] for j in range(len(x)) if j != a) for a in range(len(x))])


def _cluster_virial(x: np.ndarray, m: np.ndarray = ISOTROPIC) -> np.ndarray:
    # sum_a x_a (x) F_a: entry [b, a] is x_b F_a, and the Voigt entry xy is the force along y times the position along x
    tensor = sum(np.outer(x[a], f) for a, f in enumerate(_spring_forces(x, m)))
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])


def _close(found, expected):
    np.testing.assert_allclose(found, expected, rtol=FLOAT64_ARITHMETIC.rtol, atol=1e-12)


def test_the_spring_oracle_is_a_derivative_of_its_energy():
    # the closed forms agree with each other before they are used: forces are minus the gradient of the energy
    x = CLUSTERS[0]
    step = 1e-6
    for a in range(3):
        for c in range(3):
            moved = x.copy()
            moved[a, c] += step
            plus = _spring_energy(moved)
            moved[a, c] -= 2 * step
            minus = _spring_energy(moved)
            np.testing.assert_allclose(-(plus - minus) / (2 * step), _spring_forces(x)[a, c], rtol=1e-8, atol=1e-8)


def test_a_structure_compute_function_returns_energy_forces_virial_and_pair_forces():
    batch = _cluster_batch([0])
    result = ComputeStructureEnergyAndForcesAndVirial()([_Spring()], batch)

    x = CLUSTERS[0]
    _close(result[constants.PREDICT_TOTAL_ENERGY], [[_spring_energy(x)]])
    _close(result[constants.PREDICT_FORCES], _spring_forces(x))
    _close(result[constants.PREDICT_VIRIAL], _cluster_virial(x))
    # the energy of the first atom: half the sum of the squared bond lengths times k
    _close(result[constants.PREDICT_ATOMIC_ENERGY][0], 0.5 * SPRING_K * (np.sum(x[1] ** 2) + np.sum((x[2]) ** 2)))
    _close(result["z_" + constants.PREDICT_PAIR_FORCES], -SPRING_K * batch[constants.BOND_VECTOR].numpy())
    assert result[constants.PREDICT_TOTAL_ENERGY].shape == (1, 1)
    assert list(result)[-1] == "z_" + constants.PREDICT_PAIR_FORCES  # the key sorts last on purpose


@pytest.mark.parametrize("metric", [ISOTROPIC, ANISOTROPIC], ids=["isotropic", "anisotropic"])
def test_the_virial_of_a_structure_is_the_sum_of_position_times_force_also_for_forces_off_the_bond(metric):
    x = CLUSTERS[0]
    result = ComputeStructureEnergyAndForcesAndVirial()([_Spring(metric=metric)], _cluster_batch([0]))
    _close(result[constants.PREDICT_TOTAL_ENERGY], [[_spring_energy(x, metric)]])
    _close(result[constants.PREDICT_FORCES], _spring_forces(x, metric))
    _close(result[constants.PREDICT_VIRIAL], _cluster_virial(x, metric))


def test_the_anisotropic_virial_is_not_symmetric_so_the_oracle_tells_the_off_diagonal_entries_apart():
    tensor = _cluster_virial(CLUSTERS[0], ANISOTROPIC)
    assert abs(tensor[3] - _cluster_virial(CLUSTERS[0], ISOTROPIC)[3]) > 1e-3


def test_the_batch_virial_of_forces_off_the_bond_has_one_row_per_structure():
    result = ComputeBatchEnergyForcesVirials()([_Spring(metric=ANISOTROPIC)], _cluster_batch([0, 1]))
    _close(result[constants.PREDICT_VIRIAL], np.vstack([_cluster_virial(c, ANISOTROPIC) for c in CLUSTERS]))


def test_the_net_force_and_the_net_torque_of_an_isolated_cluster_vanish():
    x = CLUSTERS[0]
    forces = ComputeStructureEnergyAndForcesAndVirial()([_Spring()], _cluster_batch([0]))[constants.PREDICT_FORCES].numpy()
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(x, forces).sum(axis=0), 0.0, atol=1e-12)


def test_extra_keys_that_the_instructions_wrote_are_returned_and_missing_ones_are_skipped():
    class _Marker(_Spring):
        def __call__(self, input_data, training=False, local=False):
            super().__call__(input_data, training, local)
            input_data["marker"] = tf.constant([7.0])

    batch = _cluster_batch([0])
    function = ComputeStructureEnergyAndForcesAndVirial(extra_return_keys=["marker", "absent"])
    result = function([_Marker()], batch)
    np.testing.assert_array_equal(result["marker"], [7.0])
    assert "absent" not in result
    assert "marker" not in ComputeStructureEnergyAndForcesAndVirial()([_Marker()], _cluster_batch([0]))


def test_the_flags_reach_the_instructions_and_the_local_flag_adds_a_spec(monkeypatch):
    # the class-level specs dictionary is edited by the local flag (pinned): a copy keeps the class as it was
    monkeypatch.setattr(ComputeStructureEnergyAndForcesAndVirial, "specs", dict(ComputeStructureEnergyAndForcesAndVirial.specs))
    spring = _Spring()
    function = ComputeStructureEnergyAndForcesAndVirial(local=True)
    function([spring], _cluster_batch([0]), training=True)
    assert spring.calls == [{"training": True, "local": True}]
    assert constants.ATOMIC_MU_I_LOCAL in ComputeStructureEnergyAndForcesAndVirial.specs
    plain = _Spring()
    ComputeStructureEnergyAndForcesAndVirial(local=False)([plain], _cluster_batch([0]))
    assert plain.calls == [{"training": False, "local": False}]


def test_the_batch_function_sums_the_energy_of_each_structure_and_gives_every_atom_its_force():
    batch = _cluster_batch([0, 1])
    result = ComputeBatchEnergyAndForces()([_Spring()], batch)

    _close(result[constants.PREDICT_TOTAL_ENERGY], [[_spring_energy(CLUSTERS[0])], [_spring_energy(CLUSTERS[1])]])
    _close(result[constants.PREDICT_FORCES], np.vstack([_spring_forces(c) for c in CLUSTERS]))
    assert constants.PREDICT_VIRIAL not in result
    assert result[constants.PREDICT_ATOMIC_ENERGY].shape == (5, 1)


def test_the_batch_function_returns_the_requested_extra_keys():
    batch = _cluster_batch([0, 1])
    batch["extra"] = tf.constant([3.0])
    result = ComputeBatchEnergyAndForces(extra_return_keys=["extra", "absent"])([_Spring()], batch)
    np.testing.assert_array_equal(result["extra"], [3.0])
    assert "absent" not in result


def test_the_batch_virial_function_gives_one_virial_per_structure():
    batch = _cluster_batch([0, 1])
    result = ComputeBatchEnergyForcesVirials()([_Spring()], batch)

    _close(result[constants.PREDICT_VIRIAL], np.vstack([_cluster_virial(c) for c in CLUSTERS]))
    _close(result[constants.PREDICT_TOTAL_ENERGY], [[_spring_energy(c)] for c in CLUSTERS])
    _close(result[constants.PREDICT_FORCES], np.vstack([_spring_forces(c) for c in CLUSTERS]))
    assert result[constants.PREDICT_VIRIAL].shape == (2, 6)


def test_the_batch_virial_function_returns_the_requested_extra_keys():
    batch = _cluster_batch([0, 1])
    batch["extra"] = tf.constant([5.0])
    result = ComputeBatchEnergyForcesVirials(extra_return_keys=["extra"])([_Spring()], batch)
    np.testing.assert_array_equal(result["extra"], [5.0])


class _CastSpring:
    """The spring potential evaluated in float64 whatever the precision of the bond vectors."""

    def __call__(self, input_data, training=False, local=False):
        r = tf.cast(input_data[constants.BOND_VECTOR], tf.float64)
        pair_energy = 0.5 * SPRING_K * tf.reduce_sum(r**2, axis=1)
        input_data[constants.PREDICT_ATOMIC_ENERGY] = tf.math.unsorted_segment_sum(
            pair_energy, input_data[constants.BOND_IND_I], num_segments=3
        )


def test_the_energy_function_returns_the_atomic_energies_in_the_precision_of_the_bonds():
    batch = _cluster_batch([0])
    batch[constants.BOND_VECTOR] = tf.cast(batch[constants.BOND_VECTOR], tf.float32)

    result = ComputeEnergy()([_CastSpring()], batch)

    assert list(result) == [constants.PREDICT_ATOMIC_ENERGY]
    assert result[constants.PREDICT_ATOMIC_ENERGY].dtype == tf.float32
    assert result[constants.PREDICT_ATOMIC_ENERGY].shape == (3, 1)
    np.testing.assert_allclose(result[constants.PREDICT_ATOMIC_ENERGY].numpy().sum(), _spring_energy(CLUSTERS[0]), rtol=1e-5)


def test_the_energy_function_keeps_extra_keys_and_the_local_flag(monkeypatch):
    monkeypatch.setattr(ComputeEnergy, "specs", dict(ComputeEnergy.specs))
    spring = _Spring()
    function = ComputeEnergy(local=True, extra_return_keys=["extra", "absent"])
    batch = _cluster_batch([0])
    batch["extra"] = tf.constant([1.0])
    result = function([spring], batch, training=True)
    assert spring.calls == [{"training": True, "local": True}]
    assert set(result) == {constants.PREDICT_ATOMIC_ENERGY, "extra"}
    assert constants.ATOMIC_MU_I_LOCAL in ComputeEnergy.specs


def test_the_equivariant_force_function_returns_the_forces_the_instructions_wrote_and_zero_energy():
    class _Writer:
        def __call__(self, input_data, training=False, local=False):
            input_data[constants.PREDICT_FORCES] = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tf.float32)

    batch = {constants.BOND_VECTOR: tf.zeros([1, 3], tf.float64)}
    result = ComputeEquivariantForces()([_Writer()], batch)
    np.testing.assert_array_equal(result[constants.PREDICT_FORCES], [[1, 2, 3], [4, 5, 6]])
    assert result[constants.PREDICT_FORCES].dtype == tf.float64
    np.testing.assert_array_equal(result[constants.PREDICT_TOTAL_ENERGY], [[0.0]])


def test_the_placeholder_function_returns_the_placeholder_the_instructions_leave():
    class _Writer:
        def __call__(self, input_data, training=False, local=False):
            input_data[constants.PLACEHOLDER] = tf.constant([9.0])

    np.testing.assert_array_equal(ComputePlaceholder()([_Writer()], {})[constants.PLACEHOLDER], [9.0])


def test_the_abstract_function_bases_cannot_be_called_or_made():
    with pytest.raises(TypeError, match="abstract"):
        TrainFunction()
    with pytest.raises(TypeError, match="abstract"):
        ComputeFunction()

    class _Train(TrainFunction):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    class _Compute(ComputeFunction):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    for function in (_Train(), _Compute()):
        with pytest.raises(NotImplementedError):
            function()


def test_instructions_run_in_the_order_of_a_list_or_of_the_values_of_a_dict_and_a_third_type_is_refused():
    order = []

    class _Named:
        def __init__(self, name):
            self.name = name

        def __call__(self, input_data, training=False, local=False):
            order.append((self.name, training, local))
            input_data[self.name] = len(order)

    data = {}
    execute_instructions(data, [_Named("a"), _Named("b")], training=True)
    assert order == [("a", True, False), ("b", True, False)] and data == {"a": 1, "b": 2}
    order.clear()
    execute_instructions({}, {"first": _Named("x"), "second": _Named("y")}, local=True)
    assert order == [("x", False, True), ("y", False, True)]
    with pytest.raises(ValueError, match="Invalid instructions container type <class 'tuple'>"):
        execute_instructions({}, (_Named("t"),))


def test_a_container_of_instructions_becomes_a_list_and_anything_else_is_refused():
    a, b = _Spring(), _Spring()
    assert to_instructions_list({"x": a, "y": b}) == [a, b]
    as_list = [a, b]
    assert to_instructions_list(as_list) is as_list
    with pytest.raises(ValueError, match="`instructions` must be list or dict, but <class 'set'> given"):
        to_instructions_list({a})


# --- the block functions of the split models ---


class _Square:
    """``y = x ** 2`` element-wise, and ``s = sum(gather(x, rows) ** 2)``: a dense and a sparse gradient."""

    def __init__(self):
        self.flags = []

    def __call__(self, input_data, training=False, local=False):
        self.flags.append((training, local))
        input_data["y"] = input_data["x"] ** 2
        input_data["s"] = tf.reduce_sum(tf.gather(input_data["table"], input_data["rows"]) ** 2)


def _block_data():
    return {
        "x": tf.constant([1.0, -2.0, 3.0], tf.float64),
        "table": tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], tf.float64),
        "rows": tf.constant([0, 2], tf.int32),
        constants.BOND_VECTOR: tf.zeros([1, 3], tf.float32),
    }


def test_a_block_runs_its_instructions_as_local_and_returns_the_requested_keys():
    square = _Square()
    block = ComputeBlock([square], output_keys=["y"])
    result = block(None, _block_data(), training=True)
    np.testing.assert_array_equal(result["y"], [1.0, 4.0, 9.0])
    assert square.flags == [(True, True)]  # the local flag is always set
    assert ComputeBlock([_Square()])(None, _block_data()) == {}
    assert ComputeBlock([_Square()], specs={"k": {"shape": [], "dtype": "int"}}).specs == {"k": {"shape": [], "dtype": "int"}}
    assert ComputeBlock([_Square()]).specs == ComputeBlock.specs


def test_the_input_gradient_block_returns_the_derivative_the_target_and_the_other_outputs():
    block = ComputeBlockInputGradient([_Square()], wrt_keys=["x"], target_key="s", output_keys=["y", "s"])
    # the target does not depend on x: its gradient is None, which becomes zeros of the shape of x
    result = block(None, _block_data())
    np.testing.assert_array_equal(result["grad_x"], [0.0, 0.0, 0.0])
    # the target is cast to the precision of the bond vectors and the other output keys are passed on
    assert result["s"].dtype == tf.float32
    np.testing.assert_allclose(result["s"], 1.0 + 4.0 + 25.0 + 36.0)
    np.testing.assert_array_equal(result["y"], [1.0, 4.0, 9.0])


def test_the_input_gradient_block_of_a_dense_dependence_is_twice_the_input():
    class _Sum(_Square):
        def __call__(self, input_data, training=False, local=False):
            super().__call__(input_data, training, local)
            input_data["t"] = tf.reduce_sum(input_data["x"] ** 2)

    result = ComputeBlockInputGradient([_Sum()], wrt_keys=["x"], target_key="t")(None, _block_data())
    np.testing.assert_array_equal(result["grad_x"], [2.0, -4.0, 6.0])


def test_a_sparse_gradient_is_converted_to_a_dense_tensor():
    data = _block_data()
    result = ComputeBlockInputGradient([_Square()], wrt_keys=["table"], target_key="s")(None, data)
    # d/dtable of the sum of squares of rows 0 and 2: 2 * table there, zero in row 1
    np.testing.assert_array_equal(result["grad_table"], [[2.0, 4.0], [0.0, 0.0], [10.0, 12.0]])
    assert isinstance(result["grad_table"], tf.Tensor)


def test_the_input_gradient_block_updates_its_specs_when_given_some(monkeypatch):
    monkeypatch.setattr(ComputeBlockInputGradient, "specs", dict(ComputeBlockInputGradient.specs))
    ComputeBlockInputGradient([], wrt_keys=[], target_key="t", specs={"extra": {"shape": [], "dtype": "int"}})
    assert "extra" in ComputeBlockInputGradient.specs


def test_the_output_gradient_block_pulls_the_given_gradients_back_through_the_instructions():
    data = _block_data()
    data["grad_y"] = tf.constant([1.0, 10.0, 100.0], tf.float64)
    result = ComputeBlockOutputGradient([_Square()], wrt_keys=["x", "rows_float"], output_keys=["y"])(None, {**data, "rows_float": tf.constant([1.0])})
    # y = x**2 pulled back with v: 2 x v; the second input does not enter: zeros of its shape
    np.testing.assert_array_equal(result["grad_x"], [2.0, -40.0, 600.0])
    np.testing.assert_array_equal(result["grad_rows_float"], [0.0])
    assert set(result) == {"grad_x", "grad_rows_float"}


def test_the_output_gradient_block_updates_its_specs_when_given_some(monkeypatch):
    monkeypatch.setattr(ComputeBlockOutputGradient, "specs", dict(ComputeBlockOutputGradient.specs))
    ComputeBlockOutputGradient([], wrt_keys=[], output_keys=[], specs={"extra": {"shape": [], "dtype": "int"}})
    assert "extra" in ComputeBlockOutputGradient.specs


def test_the_output_gradient_block_converts_a_sparse_gradient():
    data = _block_data()
    data["grad_s"] = tf.constant(2.0, tf.float64)
    result = ComputeBlockOutputGradient([_Square()], wrt_keys=["table"], output_keys=["s"])(None, data)
    np.testing.assert_array_equal(result["grad_table"], [[4.0, 8.0], [0.0, 0.0], [20.0, 24.0]])


# --- the regularisation helpers of the model ---


class _Penalised:
    def __init__(self, penalty, active=True):
        self.penalty = penalty
        self.compute_l2_regularization = active

    def compute_l2_regularization_loss(self):
        return tf.constant(self.penalty, tf.float64)


def test_the_regularisation_loss_sums_the_penalties_of_the_instructions_that_ask_for_one():
    model = TPModel([_Penalised(1.5), _Penalised(100.0, active=False), _Penalised(2.5)])
    result = model.compute_regularization_loss()
    assert list(result) == [constants.L2_LOSS_COMPONENT]
    assert float(result[constants.L2_LOSS_COMPONENT]) == 4.0


def test_the_regularisation_loss_of_a_model_without_penalties_is_an_empty_mapping():
    assert dict(TPModel([_Penalised(1.0, active=False)]).compute_regularization_loss()) == {}


def test_the_regularisation_loss_of_real_instructions_fails_for_the_missing_flag():
    # pinned: no shipped instruction has the flag, and nothing in the library calls this method
    instructions = get_preset("LINEAR")(
        element_map={"Mo": 0}, lmax=1, n_rad_max=4, n_rad_base=4, embedding_size=4, rcut=5.0
    ).get_instructions()
    with pytest.raises(AttributeError, match="compute_l2_regularization"):
        TPModel(instructions).compute_regularization_loss()


def test_the_l2_loss_of_the_model_adds_the_loss_of_every_instruction_that_has_one():
    model = TPModel([_Penalised(1.5), object(), _Penalised(2.5, active=False)])
    assert float(model.compute_l2_regularization_loss()) == 4.0
    assert TPModel([object()]).compute_l2_regularization_loss() == 0.0
    with pytest.raises(ValueError, match="must be list or dict"):
        TPModel(("not", "a list")).compute_l2_regularization_loss()


def test_the_l2_loss_of_a_real_model_is_the_sum_of_squares_of_the_weights_it_owns():
    instructions = get_preset("LINEAR")(
        element_map={"Mo": 0}, lmax=1, n_rad_max=4, n_rad_base=4, embedding_size=4, rcut=5.0
    ).get_instructions()
    model = TPModel(instructions)
    model.build(tf.float64)
    for variable in model.variables_to_train:
        variable.assign(tf.fill(variable.shape, tf.constant(0.5, variable.dtype)))
    elements = sum(
        int(np.prod(v.shape))
        for ins in instructions.values()
        if hasattr(ins, "compute_l2_regularization_loss")
        for v in ins.trainable_variables
    )
    assert float(model.compute_l2_regularization_loss()) == pytest.approx(0.25 * elements, rel=1e-12)
    assert elements > 0


# --- flat weights and the auxiliary compute functions ---


def _linear_instructions():
    return get_preset("LINEAR")(
        element_map={"Mo": 0, "Ta": 1}, lmax=1, n_rad_max=4, n_rad_base=4, embedding_size=4, rcut=5.0
    ).get_instructions()


def test_the_flat_weight_vector_lists_every_trainable_variable_in_order_and_is_written_back_in_place():
    model = TPModel(_linear_instructions())
    model.build(tf.float64)
    variables = model.variables_to_train
    expected = np.concatenate([v.numpy().ravel() for v in variables])

    flat = model.get_flat_trainable_variables()

    np.testing.assert_array_equal(flat.numpy(), expected)
    assert int(model.count_coefs) == expected.size == sum(int(np.prod(v.shape)) for v in variables)
    sizes = [int(np.prod(v.shape)) for v in variables]
    assert [int(x) for x in model.slices] == [0, *np.cumsum(sizes)]
    model.set_flat_trainable_variables(flat + 1.0)
    for variable, before in zip(variables, np.split(expected, np.cumsum(sizes)[:-1])):
        np.testing.assert_array_equal(variable.numpy().ravel(), before + 1.0)


def test_a_flat_vector_of_another_precision_is_cast_to_the_parameter_precision():
    model = TPModel(_linear_instructions())
    model.build(tf.float32)
    flat = model.get_flat_trainable_variables()
    model.set_flat_trainable_variables(tf.cast(flat, tf.float64) * 2.0)
    assert all(v.dtype == tf.float32 for v in model.variables_to_train)
    np.testing.assert_allclose(model.get_flat_trainable_variables().numpy(), 2.0 * flat.numpy(), rtol=1e-6)


def test_an_auxiliary_compute_function_is_traced_with_the_model_inputs_and_filters_extra_keys():
    from tensorpotential.data.databuilder import GeometricalDataBuilder, construct_batches

    atoms = Atoms("MoTaMo", positions=[[0, 0, 0], [0.3, 0.1, 2.2], [2.1, -0.4, 0.9]], cell=[8, 8, 8], pbc=True)
    batch = construct_batches([atoms], [GeometricalDataBuilder({"Mo": 0, "Ta": 1}, cutoff=5.0)], batch_size=1, verbose=False)[0]
    tensors = {k: tf.constant(v) for k, v in batch.items()}
    model = TPModel(_linear_instructions(), aux_compute={"energy_only": ComputeEnergy()})
    model.build(tf.float64, jit_compile=False)

    aux = model.energy_only({**tensors, "unrelated": tf.constant([1.0])})
    full = model.compute(tensors)

    assert set(aux) == {constants.PREDICT_ATOMIC_ENERGY}
    np.testing.assert_allclose(aux[constants.PREDICT_ATOMIC_ENERGY], full[constants.PREDICT_ATOMIC_ENERGY], rtol=1e-12)
    assert set(model._aux_compute_sigs["energy_only"]) >= set(ComputeEnergy.specs)
    assert "energy_only" in model._aux_tf_funcs


def test_a_train_call_runs_the_train_function_on_a_copy_of_the_input():
    spring = _Spring()
    model = TPModel([spring])
    batch = _cluster_batch([0, 1])
    keys = set(batch)
    result = model(batch, training=True)
    assert set(batch) == keys  # the caller's dictionary does not gain the atomic energies
    assert spring.calls == [{"training": True, "local": False}]
    _close(result[constants.PREDICT_TOTAL_ENERGY], [[_spring_energy(c)] for c in CLUSTERS])


def test_a_compute_call_runs_the_compute_function_on_a_copy_of_the_input():
    spring = _Spring()
    model = TPModel([spring])
    batch = _cluster_batch([0])
    keys = set(batch)
    result = model.compute(batch)
    assert set(batch) == keys
    assert spring.calls == [{"training": False, "local": False}]
    _close(result[constants.PREDICT_FORCES], _spring_forces(CLUSTERS[0]))


def test_trainable_variables_are_selected_by_name_fragments_and_the_others_frozen():
    model = TPModel(_linear_instructions())
    model.build(tf.float64)
    names = [v.name for v in model.variables_to_train]
    keep = names[0].split("/")[0]
    model.set_trainable_variables([keep])
    trainable = [v.name for v in model.variables_to_train]
    assert trainable and all(keep in n for n in trainable)
    assert len(trainable) < len(names)
    # pinned: the selection flips the flag of every variable, so the fixed ones (cutoff, element maps) become
    # trainable too when their name matches
    model.set_trainable_variables([""], verbose=True)  # the empty fragment is in every name
    flipped = {v.name for v in model.variables_to_train} - set(names)
    assert {"element_map_index:0", "element_map_symbols:0"} <= flipped


def test_a_model_exports_its_instructions_to_yaml(tmp_path):
    from tensorpotential.potentials.presets import FS

    instructions = FS(
        element_map={"Al": 0, "Li": 1}, lmax=2, n_rad_base=4, n_rad_max=6, embedding_size=4, max_order=3
    ).get_instructions()
    model = TPModel(instructions)
    model.build(tf.float64)
    target = tmp_path / "model.yaml"
    model.export_to_yaml(str(target))
    import yaml

    assert yaml.safe_load(target.read_text())["elements"] == ["Al", "Li"]


def test_the_moved_symbols_of_the_projection_module_are_still_reachable_from_tpmodel():
    import tensorpotential.tpmodel as tpmodel
    from tensorpotential.uq import instructions as uq_instructions

    assert tpmodel.RandomProjectedBasisFeatures is uq_instructions.RandomProjectedBasisFeatures
    assert tpmodel.generate_rp_matrix is uq_instructions.generate_rp_matrix
    with pytest.raises(AttributeError, match="module 'tensorpotential.tpmodel' has no attribute 'nothing_here'"):
        tpmodel.nothing_here
