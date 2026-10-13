"""The model operations behind ``grace_utils``: convert, resave, reduce, cast, export and summarize a model.

Each operation takes explicit keyword arguments, so it can be called from code; the ``grace_utils`` command turns
its parsed arguments into these calls. ``checkpoint_path`` is given without the ``.index`` suffix.

``update_model`` and ``reduce_elements`` end the process with ``sys.exit(0)`` when the model is already in (or not yet in)
the dict format, as the command always did; a caller that must survive catches ``SystemExit``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

# the TensorFlow options are set before the first import of tensorflow
from tensorpotential import _tf_options  # noqa: F401

# isort: split
import tensorflow as tf

from tensorpotential.instructions import (
    SingleParticleBasisFunctionEquivariantInd,
)
from tensorpotential.metadata_utils import read_model_metadata
from tensorpotential.utils import get_dtype_by_name

logger = logging.getLogger(__name__)


def load_checkpoint(tp, checkpoint_path):
    tp.load_checkpoint(
        checkpoint_name=checkpoint_path,
        expect_partial=True,
        verbose=True,
        raise_errors=True,
        assert_consumed=False,
        assert_existing_objects_matched=False,
    )


def resolve_param_dtype(param_dtype_name: str | None, model_path: str):
    """Determine param_dtype: use explicit --param_dtype if given, else read from
    model.yaml metadata, else fall back to float64 for backward compatibility."""
    if param_dtype_name is not None:
        return get_dtype_by_name(param_dtype_name)
    metadata = read_model_metadata(model_path)
    if "param_dtype" in metadata:
        dtype_str = metadata["param_dtype"]
        logger.info(f"param_dtype={dtype_str} inferred from model.yaml metadata")
        return get_dtype_by_name(dtype_str)
    logger.info(
        "No param_dtype in model.yaml (old model) — using float64 for backward compatibility"
    )
    return tf.float64


def update_model(
    *,
    potential: str,
    checkpoint_path: str,
    output_suffix: str,
    param_dtype_name: str | None = None,
) -> None:
    """Convert model.yaml and corresponding checkpoint to model as dict-of-instructions and corresponding checkpoint"""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import (
        load_instructions,
        save_instructions_dict,
    )

    model_path = potential
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Converting model {model_path} with checkpoint at {checkpoint_path} to dict-like model"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)
    if isinstance(instructions, dict):
        logger.info(f"Model in {model_path} is already in new format (dict-like)")
        sys.exit(0)

    # convert to dict-of-instructions
    instructions_dict = {ins.name: ins for ins in instructions}
    assert len(instructions_dict) == len(instructions)

    tp = TensorPotential(potential=instructions, param_dtype=param_dtype)
    load_checkpoint(tp, checkpoint_path)

    tp.model.instructions = instructions_dict

    new_checkpoint_path = checkpoint_path + output_suffix
    logger.info(f"Saving converted checkpoint to {new_checkpoint_path}")
    tp.save_checkpoint(checkpoint_name=new_checkpoint_path, verbose=True)

    new_model_path = model_path
    for ext in [".yaml", ".yml"]:
        new_model_path = new_model_path.replace(ext, output_suffix + ext)

    logger.info(f"Saving converted model to {new_model_path}")
    save_instructions_dict(new_model_path, instructions_dict, param_dtype=param_dtype)

    # try to load again
    logger.info(f"Trying to load converted model from {new_model_path}")
    new_instructions = load_instructions(new_model_path)
    tp_new = TensorPotential(potential=new_instructions, param_dtype=param_dtype)
    load_checkpoint(tp_new, new_checkpoint_path)

    logger.info("Finished")


def resave_checkpoint(
    *,
    potential: str,
    checkpoint_path: str,
    output_suffix: str,
    param_dtype_name: str | None = None,
) -> None:
    """Resave the checkpoint of model.yaml without the optimizer state, as ``checkpoint_path + output_suffix``."""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import (
        load_instructions,
    )

    model_path = potential
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    new_checkpoint_path = checkpoint_path + output_suffix

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Resaving model {model_path} with checkpoint at {checkpoint_path} to new checkpoint {new_checkpoint_path}"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)

    tp = TensorPotential(potential=instructions, param_dtype=param_dtype)

    load_checkpoint(tp, checkpoint_path)

    logger.info(f"Saving converted checkpoint to {new_checkpoint_path}")
    tp.setup_checkpoint(with_optimizer=False)  # will reset all except model's weights
    tp.save_checkpoint(checkpoint_name=new_checkpoint_path, verbose=True)

    # try to load again
    logger.info(f"Trying to load again model from {model_path}")
    new_instructions = load_instructions(model_path)
    tp_new = TensorPotential(potential=new_instructions, param_dtype=param_dtype)
    load_checkpoint(tp_new, new_checkpoint_path)

    logger.info("Finished")


def reduce_elements(
    *,
    potential: str,
    checkpoint_path: str,
    output_suffix: str,
    elements: list[str],
    param_dtype_name: str | None = None,
) -> None:
    """Keep only ``elements`` in the model and its checkpoint; the energies of their dimers must not change."""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.calculator.asecalculator import TPCalculator
    from tensorpotential.utils import convert_model_reduce_elements

    model_path = potential
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Selecting {elements} elements from model {model_path}, with checkpoint {checkpoint_path}"
    )

    logger.info(f"Loading model from {model_path}")
    instructions_dict = load_instructions(model_path)
    if not isinstance(instructions_dict, dict):
        logger.info(
            f"Model in {model_path} is NOT in new format (dict-like). Convert it using `update_model` option"
        )
        sys.exit(0)

    tp = TensorPotential(potential=instructions_dict, param_dtype=param_dtype)
    load_checkpoint(tp, checkpoint_path)
    tp.model.decorate_compute_function()

    new_checkpoint_path = checkpoint_path + output_suffix
    new_model_path = model_path
    for ext in [".yaml", ".yml"]:
        new_model_path = new_model_path.replace(ext, output_suffix + ext)

    # generate test structures
    test_atoms = {}
    calc = TPCalculator(model=tp.model)
    for el in elements:
        at = Atoms(positions=[[0, 0, 0], [2, 0, 0]], symbols=[el, el], pbc=False)
        at.calc = calc
        en = at.get_potential_energy()
        logger.info(f"{el} dimer: energy = {en}")
        test_atoms[el] = {"atoms": at, "e": en}

    convert_model_reduce_elements(
        elements,
        potential_file_name=model_path,
        checkpoint_name=checkpoint_path,
        new_potential_file_name=new_model_path,
        new_checkpoint_name=new_checkpoint_path,
        param_dtype=param_dtype,
    )

    # try to load again
    logger.info("==========================")
    logger.info(f"Trying to load converted model from {new_model_path}")
    new_instructions = load_instructions(new_model_path)
    tp_new = TensorPotential(potential=new_instructions, param_dtype=param_dtype)
    load_checkpoint(tp_new, new_checkpoint_path)
    tp_new.model.decorate_compute_function()

    calc_new = TPCalculator(model=tp_new.model)
    for el, info in test_atoms.items():
        at = info["atoms"]
        en_ref = info["e"]
        at.calc = calc_new
        en = at.get_potential_energy()
        if np.allclose(en, en_ref):
            diff = en - en_ref
            logger.info(f"{el} dimer: OK, diff = {diff:.5g} eV")
        assert np.allclose(
            en, en_ref
        ), f"Energy not equal for element {el}: {en} != {en_ref}"

    logger.info("Finished")


def cast_model_param(
    *,
    potential: str,
    checkpoint_path: str,
    output_suffix: str,
    curr: str,
    to: str,
) -> None:
    """Cast the floating point parameters of a model from ``curr`` to ``to`` (``fp32`` or ``fp64``)."""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import (
        load_instructions,
        save_instructions_dict,
    )
    from tensorflow import float32, float64, Variable, Module

    def recursive_cast_instruction_vars(
        instruction, from_dtype, to_dtype, max_depth=None, current_depth=0
    ):
        if max_depth is not None and current_depth >= max_depth:
            return

        next_depth = current_depth + 1

        if isinstance(instruction, Module):
            for attr_name in instruction.__dict__:
                attr_value = getattr(instruction, attr_name)
                if isinstance(attr_value, Variable) and attr_value.dtype == from_dtype:
                    setattr(
                        instruction,
                        attr_name,
                        Variable(
                            attr_value.numpy(),
                            dtype=to_dtype,
                            name=attr_value.name.replace(":0", ""),
                        ),
                    )
                elif isinstance(attr_value, list) or isinstance(attr_value, tuple):
                    for item in attr_value:
                        recursive_cast_instruction_vars(
                            item, from_dtype, to_dtype, max_depth, next_depth
                        )
                elif isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        recursive_cast_instruction_vars(
                            value, from_dtype, to_dtype, max_depth, next_depth
                        )

    dtypes = {"fp32": float32, "fp64": float64}

    model_path, from_dtype, to_dtype = potential, curr, to
    assert from_dtype in dtypes, f"Invalid dtype, {from_dtype} not in {dtypes.keys()}"
    assert to_dtype in dtypes, f"Invalid dtype, {to_dtype} not in {dtypes.keys()}"
    logger.info(
        f"Trying to cast model at model_path={model_path},"
        f" with checkpoint_path={checkpoint_path},"
        f" from dtype={from_dtype} to dtype={to_dtype}"
    )

    d_from = dtypes[from_dtype]
    d_to = dtypes[to_dtype]

    pot = load_instructions(model_path)
    tp = TensorPotential(potential=pot, fit_config={}, param_dtype=d_from)
    load_checkpoint(tp, checkpoint_path)
    assert isinstance(
        tp.model.instructions, dict
    ), "Model is not dict format, cannot cast it. Use `update_model` option to convert it to dict format."
    for iname, ins in tp.model.instructions.items():
        recursive_cast_instruction_vars(ins, d_from, d_to, max_depth=5)
    logger.info(f"Successfully casted model at {checkpoint_path} to {to_dtype} dtype")

    # Emit the casted result as a self-contained, drop-in model directory with STANDARD
    # file names — <ckpt_dir>/casted<suffix>/{model.yaml, checkpoint.index, checkpoint.data-*}.
    # A dedicated dir keeps the standard `model.yaml` name (not model-<suffix>.yaml) without
    # clobbering the source fp64 model.yaml/checkpoint. The model.yaml is written in the new
    # wrapped format stamped with the target param_dtype, so it is self-describing:
    # resolve_param_dtype() returns the target dtype and every downstream tool (export,
    # export_kokkos, grace_uq, TPCalculator-from-checkpoint) rebuilds at the matching precision
    # instead of defaulting to float64 and dtype-mismatching the casted weights.
    casted_dir = Path(checkpoint_path).resolve().parent / f"casted{output_suffix}"
    casted_dir.mkdir(parents=True, exist_ok=True)
    casted_ckpt = str(casted_dir / "checkpoint")
    tp.save_checkpoint(checkpoint_name=casted_ckpt, verbose=True)
    save_instructions_dict(str(casted_dir / "model.yaml"), tp.model.instructions, param_dtype=d_to)
    logger.info(
        f"Saved casted model ({to_dtype}) to {casted_dir}/ "
        f"(standard names: model.yaml + checkpoint.*)"
    )

    current_directory = Path.cwd()
    full_path = current_directory / "casted_model"
    instructions = load_instructions(model_path)
    tp = TensorPotential(potential=instructions, param_dtype=d_to)
    load_checkpoint(tp, casted_ckpt)
    tp.save_model(exact_path=full_path, jit_compile=True)
    logger.info(f"Saved casted model to {full_path}")


def export_model(
    *,
    potential: str,
    checkpoint_path: str,
    saved_model_name: str,
    sf: bool = False,
    communicated_keys: list[str] | None = None,
    param_dtype_name: str | None = None,
) -> None:
    """Export model.yaml and its checkpoint to a SavedModel, or (``sf``) to the GRACE-FS yaml format."""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions

    model_path = potential
    save_fs = sf
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Converting model {model_path} with checkpoint at {checkpoint_path} to dict-like model"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)
    tp = TensorPotential(potential=instructions, param_dtype=param_dtype)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(tp, checkpoint_path)

    if not save_fs:
        logger.info(f"Exporting model to {saved_model_name}")
        tp.save_model_with_aux_computes(
            exact_path=saved_model_name,
            jit_compile=True,
            communicated_keys=communicated_keys,
        )
    else:
        if not (
            saved_model_name.endswith(".yaml") or saved_model_name.endswith(".yml")
        ):
            saved_model_name += ".yaml"
        logger.info(f"Exporting FS model to {saved_model_name}")
        tp.export_to_yaml(exact_filename=saved_model_name)

    logger.info("Finished")


def export_kokkos(
    *,
    potential: str,
    checkpoint_path: str | None,
    output: str,
    arch: str = "auto",
    uq_artifacts_path: str | None = None,
    param_dtype_name: str | None = None,
) -> None:
    """Export GRACE-1L/2L weights to .npz for the LAMMPS Kokkos pair styles."""
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.export_kokkos import (
        export_1l_npz,
        export_2l_npz,
        export_3l_npz,
    )

    model_path = potential
    if checkpoint_path and checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path[: -len(".index")]

    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    dtype_str = "float64" if param_dtype == tf.float64 else "float32"

    instructions = load_instructions(model_path)
    instr_dict = (
        instructions if isinstance(instructions, dict)
        else {ins.name: ins for ins in instructions}
    )

    if arch == "auto":
        # 3L first: it uses `SPBF` instructions in equivariant mode (>=2 of them
        # for the two equivariant layers), a class disjoint from the 2L
        # `SingleParticleBasisFunctionEquivariantInd`, so the checks don't collide.
        n_equivariant_spbf = sum(
            1
            for ins in instr_dict.values()
            if type(ins).__name__ == "SPBF" and getattr(ins, "equivariant_mode", False)
        )
        is_2l = any(
            type(ins).__name__ == "SingleParticleBasisFunctionEquivariantInd"
            for ins in instr_dict.values()
        )
        if n_equivariant_spbf >= 2:
            arch = "3l"
        elif is_2l:
            arch = "2l"
        else:
            arch = "1l"
        logger.info(f"Auto-detected architecture: GRACE-{arch.upper()}")
    else:
        logger.info(f"Forced architecture: GRACE-{arch.upper()}")

    exporter = {"1l": export_1l_npz, "2l": export_2l_npz, "3l": export_3l_npz}[arch]
    exporter(
        model_path,
        checkpoint_path,
        output,
        dtype_str,
        uq_artifacts_path=uq_artifacts_path,
    )
    logger.info("Finished")


def model_summary(*, potential: str, verbose: int = 0, param_dtype_name: str | None = None) -> None:
    """Log the summary of the model of ``potential`` at verbosity 0, 1 or 2."""
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel

    model_path = potential

    logger.info(f"Loading model from {model_path}")
    instructions_dict = load_instructions(model_path)
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    tp = TPModel(instructions_dict)
    tp.build(param_dtype)
    res = tp.summary(verbose=verbose)
    logger.info(f"Summary (verbose={verbose}) is:\n{res}")


def aux_model(
    *,
    potential: str,
    checkpoint_path: str,
    output_path: str,
    communicated_keys: list[str],
    param_dtype_name: str | None = None,
) -> None:
    """Save the model as a SavedModel with ``compute_energy``; a GRACE-2L model is also split at ``communicated_keys``."""
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import (
        ComputeEnergy,
        TPModel,
    )
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.instructions.instruction_graph_utils import (
        build_split_tpmodel,
        build_dependency_graph,
        find_non_local_keys,
    )

    model_path = potential
    param_dtype = resolve_param_dtype(param_dtype_name, model_path)
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(f"Upgrading model {model_path} with checkpoint at {checkpoint_path}")

    instr = load_instructions(model_path)

    extra_aux_computes = {"compute_energy": ComputeEnergy()}

    has_spbfei = False
    for ins_name, ins in instr.items():
        if isinstance(ins, SingleParticleBasisFunctionEquivariantInd):
            has_spbfei = True
            logger.info(
                "GRACE-2L model is identified (SingleParticleBasisFunctionEquivariantInd instruction found)"
            )
            break
    if has_spbfei:
        dep_graph = build_dependency_graph(instr)
        non_local_comm_keys = find_non_local_keys(instr, communicated_keys)
        # pretty print dep_graph
        for k, v in dep_graph.items():
            if k in communicated_keys:
                if k in non_local_comm_keys:
                    logger.info(f"  [NON-LOCAL COMMUNICATED] {k} : {v}")
                else:
                    logger.info(f"  [COMMUNICATED] {k} : {v}")
            else:
                logger.info(f"  {k}: {v}")

        logger.info(f"Communicated keys: {communicated_keys}")

        # assert that communicated_keys in dep_graph
        for ck in communicated_keys:
            assert ck in dep_graph, f"Key {ck} not found in dependency graph"

        m = build_split_tpmodel(
            instr,
            communicated_keys=communicated_keys,
            param_dtype=param_dtype,
            jit_compile=True,
            extra_aux_computes=extra_aux_computes,
        )
    else:
        m = TPModel(instructions=instr, aux_compute=extra_aux_computes)
        m.build(param_dtype=param_dtype, jit_compile=True)
        m.decorate_compute_function(jit_compile=True)

    tp = TensorPotential(m.instructions, param_dtype=param_dtype)
    tp.model = m
    tp.load_checkpoint(
        checkpoint_name=checkpoint_path,
        assert_consumed=False,
        assert_existing_objects_matched=False,
        expect_partial=True,
    )

    logger.info(f"Saving upgraded model to {output_path}")
    tp.save_model(exact_path=output_path)

    logger.info("Finished")
