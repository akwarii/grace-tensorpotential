from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml
from ase.data import chemical_symbols

import tensorpotential
from tensorpotential import constants as tc
from tensorpotential.calculator.foundation_models import get_or_download_checkpoint
from tensorpotential.cli.data import (
    load_and_prepare_datasets,
    inject_or_update_atomic_shift_in_model,
)
from tensorpotential.cli.prepare import (
    construct_model,
    convert_to_tensors_for_model,
    construct_model_functions,
)
from tensorpotential.cli.wizard import generate_template_input
from tensorpotential.cli.train import try_load_checkpoint, train_adam, train_bfgs
from tensorpotential.instructions.base import (
    load_instructions,
    save_instructions_dict,
)
from tensorpotential.tpmodel import TPModel
from tensorpotential.tensorpot import TensorPotential, get_output_dir
from tensorpotential.utils import (
    convert_model_reduce_elements,
    get_param_dtype_from_config,
    is_chief,
)
from tensorpotential.metadata_utils import read_model_metadata

from tensorpotential import _tf_options  # noqa: F401
import tensorflow as tf

MODEL_CONFIG_YAML = "model.yaml"

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

DEFAULT_TARGET_UPDATES = {
    "Adam": {"scratch": 50_000, "finetune": 10_000},
    "BFGS": {"scratch": 500, "finetune": 100},
    "L-BFGS-B": {"scratch": 500, "finetune": 100},
}


def build_parser(prog: str = "gracemaker"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Fitting utility for (graph) atomic cluster expansion "
        "potentials.\n",
    )

    parser.add_argument(
        "input",
        help="input YAML file, default: input.yaml",
        nargs="?",
        type=str,
        default="input.yaml",
    )

    parser.add_argument(
        "-l",
        "--log",
        help="log filename, default: log.txt",
        type=str,
        default="log.txt",
    )

    parser.add_argument(
        "-m",
        "--multigpu",
        action="store_true",
        default=False,
        help="Single host/multi GPU distributed fit",
    )

    parser.add_argument(
        "-rl",
        "--restart-latest",
        action="store_true",
        default=False,
        help="Restart from latest checkpoint (use separately from -r/-rs)",
    )

    parser.add_argument(
        "-r",
        "--restart-best-test",
        action="store_true",
        default=False,
        help="Restart from latest best test checkpoint (use separately from -rs/-rl)",
    )

    parser.add_argument(
        "-rs",
        "--restart-suffix",
        default=None,
        type=str,
        dest="restart_suffix",
        help="Suffix of checkpoint to restart from, i.e. .epoch_10  (use separately from -r/-rl)",
    )

    parser.add_argument(
        "--no-intra-epoch-redo",
        dest="intra_epoch_redo",
        action="store_false",
        default=True,
        help="When resuming from a mid-epoch checkpoint (intra_epoch_save=True), "
        "skip the interrupted epoch's remaining batches entirely and bump the "
        "step counter to the nominal end of that epoch. This keeps the LR "
        "scheduler aligned with the original maxiter budget at the cost of "
        "under-training by (nominal_end - saved_step) gradient updates. "
        "Default is to fast-forward the iterator and train only the unseen "
        "tail, which is usually cheaper and preserves full training amount.",
    )

    parser.add_argument(
        "-p",
        "--potential",
        type=str,
        help="Potential configuration to load, model.yaml file",
        default=None,
    )

    parser.add_argument(
        "-s",
        "--save-model",
        action="store_true",
        default=False,
        help="Export model as TF saved model",
    )

    parser.add_argument(
        "-sf",
        "--save--fs",
        action="store_true",
        default=False,
        dest="save_fs",
        help="Export FS model as yaml to be loaded in CPP",
    )

    parser.add_argument(
        "-e",
        "--eager",
        action="store_true",
        default=False,
        help="Eager graph execution",
    )

    parser.add_argument(
        "-nj", "--no-jit", action="store_true", default=False, help="No JIT"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (will overwrite value from input.yaml)",
    )

    parser.add_argument(
        "-cm",
        "--check-model",
        action="store_true",
        default=False,
        help="Check model consistency, without performing fit",
    )

    parser.add_argument(
        "-t",
        "--template",
        help="Generate a template 'input.yaml' file by dialog",
        dest="template",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-cn",
        "--checkpoint-name",
        dest="checkpoint_name",
        type=str,
        default=None,
        help="Explicit name of the checkpoint (omit .index suffix)",
    )

    parser.add_argument(
        "--reset-epoch-and-step",
        dest="reset_epoch_and_step",
        action="store_true",
        default=False,
        help="Reset epoch and step counters from prev. runs",
    )

    return parser


def add_loaded_model_parameter(potential_file_name, args_yaml):
    log.info(f"Loading model config from `{potential_file_name}`")
    with open(potential_file_name, "rt") as f:
        list_of_dict_instructions = yaml.safe_load(f)
    if isinstance(list_of_dict_instructions, dict):
        # New format: top-level dict with sections like "metadata" and "instructions"
        if "instructions" in list_of_dict_instructions:
            list_of_dict_instructions = list(
                list_of_dict_instructions["instructions"].values()
            )
        else:
            list_of_dict_instructions = list(list_of_dict_instructions.values())

    for ins_dict in list_of_dict_instructions:
        if "ScalarChemicalEmbedding" in ins_dict["__cls__"]:
            # if not "reduce_elements" in args_yaml::potential,
            # then set original elements map
            if not args_yaml[tc.INPUT_POTENTIAL_SECTION].get(
                tc.INPUT_POTENTIAL_REDUCE_ELEMENTS, False
            ):
                element_map = ins_dict["element_map"]
                assert isinstance(element_map, dict), "Element map is not a dict"

                log.info(
                    f"Setting {tc.INPUT_POTENTIAL_SECTION}::elements to {element_map}"
                )

                if "elements" not in args_yaml[tc.INPUT_POTENTIAL_SECTION]:
                    args_yaml[tc.INPUT_POTENTIAL_SECTION]["elements"] = element_map
                else:
                    old_elements = args_yaml[tc.INPUT_POTENTIAL_SECTION]["elements"]
                    old_element_map = {e: i for i, e in enumerate(old_elements)}
                    if element_map != old_element_map:
                        raise RuntimeError(
                            f"Provided input.yaml::{tc.INPUT_POTENTIAL_SECTION}::elements  ({old_elements}) "
                            + f"differs from that read from model ({element_map})"
                        )
            # otherwise skip until load_and_prepare_datasets
        model_cutoff = ins_dict.get("rcut", ins_dict.get("cutoff"))
        if model_cutoff is not None:
            user_cutoff = args_yaml.get(tc.INPUT_CUTOFF)
            if isinstance(user_cutoff, (int, float)) and user_cutoff != model_cutoff:
                log.warning(
                    f"User-provided cutoff ({user_cutoff}) differs from "
                    f"foundation model cutoff ({model_cutoff}). "
                    f"Using model cutoff={model_cutoff}."
                )
            args_yaml[tc.INPUT_CUTOFF] = model_cutoff
            log.info(f"Setting data::{tc.INPUT_CUTOFF} to {args_yaml[tc.INPUT_CUTOFF]}")
        if tc.INPUT_CUTOFF_DICT in ins_dict:
            args_yaml[tc.INPUT_CUTOFF_DICT] = ins_dict[tc.INPUT_CUTOFF_DICT]
            log.info(
                f"Setting data::{tc.INPUT_CUTOFF_DICT} to {args_yaml[tc.INPUT_CUTOFF_DICT]}"
            )
        if "ConstantScaleShiftTarget" in ins_dict["__cls__"]:
            scale = ins_dict["scale"]
            args_yaml[tc.INPUT_POTENTIAL_SECTION]["scale"] = scale
            log.info(f"Setting {tc.INPUT_POTENTIAL_SECTION}::scale to {scale}")
        if "avg_n_neigh" in ins_dict:
            args_yaml[tc.INPUT_POTENTIAL_SECTION]["avg_n_neigh"] = ins_dict[
                "avg_n_neigh"
            ]
            log.info(
                f"Setting {tc.INPUT_POTENTIAL_SECTION}::avg_n_neigh to {ins_dict['avg_n_neigh']}"
            )

    return args_yaml


def compute_maxiter(target_total_updates: int, num_batches: int, optimizer: str) -> int:
    """Convert total gradient updates to epochs.

    - For Adam: updates are per-mini-batch.
    - For BFGS/L-BFGS-B: updates are per-epoch (full dataset).

    Rounds up so the resulting epoch count delivers *at least*
    ``target_total_updates`` gradient updates, never fewer.
    """
    if optimizer in ("L-BFGS-B", "BFGS"):
        raw = target_total_updates
    else:
        raw = target_total_updates / max(num_batches, 1)

    rounded = int(math.ceil(raw))  # smallest integer that meets the target
    return max(10, min(5000, rounded))  # clamped [10, 5000]


def finalize_lora_if_enabled(tp):
    """Merge the LoRA updates of ``tp`` into its weights when any are active (otherwise nothing)."""
    if tp.is_lora_enabled():
        log.info("LORA enabled, finalizing it.")
        tp.finalize_lora_update()


def apply_lora_options(tp, potential_config, model_file, param_dtype):
    """Activate and/or reduce LoRA as ``potential::lora`` and ``potential::reduce_lora`` ask.

    Instructions that already hold LoRA (a restart from a saved ``model.yaml``) keep their trained
    tensors. ``model_file`` is re-saved with ``param_dtype`` after each step, so that a restart
    rebuilds the model as it is now.
    """
    lora_config = potential_config.get(tc.INPUT_POTENTIAL_LORA)
    if lora_config:
        tp.enable_lora_adaptation(lora_config)
        log.info(f"Saving model config to {model_file} after LoRA activation")
        save_instructions_dict(
            model_file, tp.model.instructions, param_dtype=param_dtype
        )

    if potential_config.get(tc.INPUT_POTENTIAL_REDUCE_LORA):
        if not tp.is_lora_enabled():
            log.warning(
                "`reduce_lora` is set, but no LoRA is active: nothing to reduce"
            )
            return
        tp.finalize_lora_update()
        log.info(f"Saving model config to {model_file} after LoRA reduction")
        save_instructions_dict(
            model_file, tp.model.instructions, param_dtype=param_dtype
        )


def lora_restart_model_file(args_parse, potential_config, output_dir):
    """The ``model.yaml`` of the run being restarted, when ``potential::lora`` asks for LoRA.

    A restart (``-r``, ``-rl``, ``-rs``) otherwise rebuilds the model from the preset of the input,
    which has no LoRA tensors, so the trained ones in the checkpoint would be replaced by new ones.
    Returns ``None`` when the run is not a restart, asks for no LoRA, or has no saved model file.
    """
    restarting = (
        args_parse.restart_best_test
        or args_parse.restart_latest
        or args_parse.restart_suffix is not None
    )
    saved = os.path.join(output_dir, MODEL_CONFIG_YAML)
    if (
        restarting
        and potential_config.get(tc.INPUT_POTENTIAL_LORA)
        and os.path.exists(saved)
    ):
        log.info(f"LoRA restart: the model is loaded from `{saved}`")
        return saved
    return None


class _RunFinished(Exception):
    """The run is complete or was asked to stop early (check-model, save-model, Ctrl-C): ``main`` exits with 0."""


@dataclass
class _Run:
    """The state of one gracemaker run, filled in step by step (see ``main`` for the order)."""

    args: argparse.Namespace
    args_yaml: dict
    potential_config: dict
    fit_config: dict
    seed: int
    output_dir: str
    rcut: float
    strategy: Any = None
    batch_size: int = 8
    test_batch_size: Any = 1
    global_batch_size: int = 8
    global_test_batch_size: Any = None
    param_dtype: Any = None
    param_dtype_flag: str = ""
    user_set_param_dtype: bool = False
    potential_file_name: str | None = None
    checkpoint_name: str | None = None
    element_map: dict | None = None
    data_stats: dict | None = None
    train_data: Any = None
    test_data: Any = None
    train_grouping_df: Any = None
    test_grouping_df: Any = None
    has_test_set: bool = False
    communicated_keys: Any = None
    instructions: Any = None
    target_potential_file_name: str = ""
    tp: Any = None


def _restarting(args_parse) -> bool:
    return bool(args_parse.restart_best_test or args_parse.restart_latest)


def _start_run(args_parse, strategy) -> _Run:
    """Read input.yaml, set the seed, open the log file and print the banner."""
    input_yaml_filename = args_parse.input
    with open(input_yaml_filename) as f:
        args_yaml = yaml.safe_load(f)
    seed = args_parse.seed or args_yaml.get("seed", 1)
    output_dir = get_output_dir(seed=seed)
    if "log" in args_parse and is_chief(strategy):
        log_file_name = os.path.join(output_dir, args_parse.log)
        os.makedirs(output_dir, exist_ok=True)
        log.info("Redirecting log into file {}".format(log_file_name))
        fileh = logging.FileHandler(log_file_name, "a")
        formatter = logging.Formatter(LOG_FMT)
        fileh.setFormatter(formatter)
        log.addHandler(fileh)

    log.info("=" * 40)
    log.info(" " * 12 + "Start GRACEmaker")
    log.info("=" * 40)
    log.info(f"GRACEmaker version: {tensorpotential.__version__}")
    log.info(f"Tensorflow version: {tf.__version__}")
    log.info("Loaded {}... ".format(input_yaml_filename))
    assert isinstance(args_yaml, dict)

    log.info(f"Set seed to {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return _Run(
        args=args_parse,
        args_yaml=args_yaml,
        potential_config=args_yaml["potential"],
        fit_config=args_yaml["fit"],
        seed=seed,
        output_dir=output_dir,
        rcut=args_yaml["cutoff"],
    )


def _select_strategy(run: _Run, strategy, strategy_desc: str) -> None:
    """Take the distribution strategy handed over, else MirroredStrategy on request, else the default one."""
    if strategy is None:
        if run.fit_config.get("strategy") == "mirrored" or run.args.multigpu:
            # The auto-MWMS upgrade for single-host multi-GPU is done by
            # scripts.gracemaker.main; it must run before any tf.config call
            # so MWMS can configure collective ops at program startup. We
            # only reach here in two cases:
            #   1. NUM_VIRTUAL_DEVICES is set, which forces this fallback
            #      (virtual device setup and MWMS are mutually exclusive at
            #      the tf.distribute API level).
            #   2. Someone imported cli.gracemaker.main directly, bypassing
            #      the entry-point script.
            strategy = tf.distribute.MirroredStrategy()
            strategy_desc = "Single host/multi GPU"
        else:
            # default - single GPU mode
            strategy = tf.distribute.get_strategy()
            strategy_desc = "Single GPU"
    else:
        log.info("Multi host/multi GPU distributed strategy is already initialized")

    current_ctx = tf.distribute.get_replica_context()
    replica_id_in_sync_group = current_ctx.replica_id_in_sync_group
    log.info(f"Data distribution strategy: {strategy_desc}")
    log.info(f"Number of replicas: {strategy.num_replicas_in_sync}")
    log.info(f"Replica ID in sync group: {replica_id_in_sync_group}")
    run.strategy = strategy


def _set_batch_sizes(run: _Run) -> None:
    """The per-replica and global batch sizes of the training and the test set."""
    replicas = run.strategy.num_replicas_in_sync
    run.batch_size = run.fit_config.get(
        "batch_size", 8
    )  # TODO: get batch_size from stats.json for distributed dataset
    run.global_batch_size = run.batch_size * replicas
    if not run.args.save_model:
        log.info(
            f"Global TRAIN batch size (= minibatch size * num GPUS): {run.global_batch_size}, "
            f"num. GPUS: {replicas}, minibatch size: {run.batch_size}"
        )
    run.test_batch_size = run.fit_config.get("test_batch_size", 1)
    if run.test_batch_size:
        run.global_test_batch_size = run.test_batch_size * replicas
        if not run.args.save_model:
            log.info(
                f"Global TEST batch size (= minibatch size * num GPUS): {run.global_test_batch_size}, "
                f"num. GPUS: {replicas}, minibatch size: {run.test_batch_size}"
            )


def _check_model_and_stop(run: _Run) -> None:
    check_model(run.args_yaml, run.param_dtype)
    log.info("Exiting...")
    raise _RunFinished


def _use_foundation_checkpoint(run: _Run) -> None:
    """Fine-tuning of a foundation model: point the potential at its downloaded model.yaml and checkpoint."""
    finetune_model = run.potential_config.get(tc.INPUT_POTENTIAL_FINETUNE_FOUNDATION_MODEL)
    if finetune_model and not _restarting(run.args):
        log.info(f"FINETUNING {finetune_model} foundation model")
        checkpoint_path = get_or_download_checkpoint(finetune_model)
        run.potential_config["filename"] = os.path.join(checkpoint_path, "model.yaml")
        run.potential_config[tc.INPUT_POTENTIAL_CHECKPOINT_NAME] = os.path.join(
            checkpoint_path, "checkpoint"
        )
        log.info(
            f'FINETUNING: set model path to {run.potential_config["filename"]} and '
            + f"checkpoint to {run.potential_config[tc.INPUT_POTENTIAL_CHECKPOINT_NAME]}"
        )


def _find_potential_file(run: _Run) -> None:
    run.potential_file_name = (
        run.args.potential
        or run.potential_config.get("filename")
        or lora_restart_model_file(run.args, run.potential_config, run.output_dir)
    )
    if run.potential_file_name:
        # model will be loaded from file,
        # need inject some parameters (cutoff, elements, scale, shift) into args_yaml
        log.info("Model YAML file is provided, try to adjust input")
        run.args_yaml = add_loaded_model_parameter(run.potential_file_name, run.args_yaml)


def _resolve_maxiter(fit_config: dict, potential_config: dict, num_train_batches: int) -> None:
    """Set fit_config['maxiter'] from target_total_updates or 'auto'; an integer stays as it is."""
    target_total_updates = fit_config.get("target_total_updates")
    maxiter_raw = fit_config.get("maxiter", "auto")
    optimizer = fit_config.get("optimizer", "Adam")
    if target_total_updates is not None:
        resolved_maxiter = compute_maxiter(target_total_updates, num_train_batches, optimizer)
        log.info(
            f"maxiter resolved from target_total_updates={target_total_updates}: {resolved_maxiter} epochs"
        )
        fit_config["maxiter"] = resolved_maxiter
    elif maxiter_raw == "auto":
        is_finetune = bool(potential_config.get(tc.INPUT_POTENTIAL_FINETUNE_FOUNDATION_MODEL))
        mode = "finetune" if is_finetune else "scratch"
        default_target = DEFAULT_TARGET_UPDATES.get(optimizer, {}).get(mode)
        if default_target is None:
            # fallback if optimizer not in dict
            default_target = 100 if is_finetune else 500
        resolved_maxiter = compute_maxiter(default_target, num_train_batches, optimizer)
        log.info(
            f"maxiter auto ({optimizer}, {mode}): target={default_target} → {resolved_maxiter} epochs"
        )
        fit_config["maxiter"] = resolved_maxiter


def _resolve_switch_after_iter(fit_config: dict) -> None:
    """Turn loss.switch.after_iter from a fraction or 'auto' into an integer epoch count."""
    final_maxiter = fit_config.get("maxiter", 500)
    switch_params = fit_config.get("loss", {}).get("switch", {})
    if switch_params and "after_iter" in switch_params:
        after_iter_raw = switch_params["after_iter"]
        if after_iter_raw == "auto":
            after_iter_raw = 0.75  # default: switch after 75% of training
        if isinstance(after_iter_raw, float) and 0.0 < after_iter_raw < 1.0:
            resolved_after_iter = max(1, round(after_iter_raw * final_maxiter))
            log.info(
                f"switch.after_iter resolved from fraction {after_iter_raw}: "
                f"{resolved_after_iter} epochs (of {final_maxiter} total)"
            )
            switch_params["after_iter"] = resolved_after_iter


def _load_datasets(run: _Run) -> None:
    """Load and batch the data (not needed to only save a model) and resolve the epochs that depend on its size."""
    (
        run.train_data,
        run.test_data,
        run.element_map,
        run.data_stats,
        run.train_grouping_df,
        run.test_grouping_df,
    ) = load_and_prepare_datasets(
        run.args_yaml,
        run.batch_size,
        test_batch_size=run.test_batch_size,
        seed=run.seed,
        strategy=run.strategy,
    )
    run.has_test_set = run.test_data is not None
    _resolve_maxiter(run.fit_config, run.potential_config, len(run.train_data))
    _resolve_switch_after_iter(run.fit_config)


def _infer_param_dtype(run: _Run) -> None:
    """Infer param_dtype from model.yaml now that all potential_file_name sources are resolved.

    This must happen after the save_model fallback, otherwise -r -s would miss the local model.yaml and default to
    float32, corrupting the saved model.yaml.
    """
    name = run.potential_file_name
    if name and not run.user_set_param_dtype:
        if os.path.exists(name):
            metadata = read_model_metadata(name)
            if "param_dtype" in metadata:
                from tensorpotential.utils import get_dtype_by_name

                run.param_dtype = get_dtype_by_name(metadata["param_dtype"])
                run.param_dtype_flag = metadata["param_dtype"]
                log.info(
                    f"param_dtype={run.param_dtype_flag} inferred from model.yaml metadata"
                )
            else:
                run.param_dtype = tf.float64
                run.param_dtype_flag = "float64"
                log.info(
                    "No param_dtype in model.yaml (old model) — using float64 for backward compatibility"
                )
        else:
            log.info(f"Using default model parameter dtype: {run.param_dtype_flag}")
    elif run.user_set_param_dtype:
        log.info(f"Model parameter dtype: {run.param_dtype_flag}")
    else:
        # Fresh fit from scratch — no model.yaml exists yet
        log.info(f"Using default model parameter dtype: {run.param_dtype_flag}")


def _resolve_checkpoint_name(run: _Run) -> None:
    checkpoint_name = run.args.checkpoint_name or run.potential_config.get(
        tc.INPUT_POTENTIAL_CHECKPOINT_NAME
    )
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_name and checkpoint_name.endswith(".index"):
        checkpoint_name = checkpoint_name.replace(".index", "")
    run.checkpoint_name = checkpoint_name


def _reduce_elements_if_requested(run: _Run) -> None:
    """Model reduction / element selection: write the reduced model and checkpoint into the output folder."""
    is_reduce_elements = run.args_yaml[tc.INPUT_POTENTIAL_SECTION].get(
        tc.INPUT_POTENTIAL_REDUCE_ELEMENTS, False
    )
    if not is_reduce_elements or _restarting(run.args):
        return
    new_potential_file_name = os.path.join(run.output_dir, MODEL_CONFIG_YAML)
    new_checkpoint_name = os.path.join(run.output_dir, "checkpoints", "checkpoint")
    log.info(
        f"Selecting elements from model {run.potential_file_name} with checkpoint at {run.checkpoint_name}"
        + f" and saving to {new_potential_file_name} with checkpoint at {new_checkpoint_name}"
    )
    convert_model_reduce_elements(
        element_map=run.element_map,
        potential_file_name=run.potential_file_name,
        checkpoint_name=run.checkpoint_name,
        new_potential_file_name=new_potential_file_name,
        new_checkpoint_name=new_checkpoint_name,
        param_dtype=run.param_dtype,
    )
    run.potential_file_name = new_potential_file_name
    run.checkpoint_name = new_checkpoint_name


def _build_instructions(run: _Run) -> None:
    """Load the model config from file (restarts) or construct it from the potential section of the input."""
    # Initialize communicated_keys; may be set by preset or overridden by input.yaml
    run.communicated_keys = run.potential_config.get("communicated_keys", None)
    if run.potential_file_name:
        # load from potential_file_name (restart path: -r / -rl)
        log.info(f"Loading model config from `{run.potential_file_name}`")
        run.instructions = load_instructions(run.potential_file_name)
        return
    if run.args.save_model:  # save model, but initialized from scratch
        # TODO: should be possible
        raise ValueError("Cannot save just-initialized model")
    cut_dict = run.args_yaml.get(tc.INPUT_CUTOFF_DICT)
    if cut_dict:
        log.info(f"User-defined cutoff dict: {cut_dict}")
    log.info("Constructing model from config")
    run.instructions, preset_communicated_keys = construct_model(
        run.potential_config,
        element_map=run.element_map,
        rcut=run.rcut,
        cutoff_dict=cut_dict,
        **run.data_stats,
    )
    # Preset keys take priority; fall back to input.yaml key if preset didn't set any
    if preset_communicated_keys is not None:
        run.communicated_keys = preset_communicated_keys


def _inject_foundation_shift(run: _Run) -> None:
    """Model surgery: inject FM-based atomic energy shifts before saving."""
    fm_shift_dict = run.data_stats.get("fm_shift_dict") if run.data_stats else None
    if fm_shift_dict is not None:
        log.info("Injecting foundation model energy shifts into model")
        inject_or_update_atomic_shift_in_model(
            instructions_dict=run.instructions,
            fm_shift_dict=fm_shift_dict,
            element_map=run.element_map,
        )


def _create_tensorpotential(run: _Run):
    """Save the model config and build the TensorPotential with the loss and metric functions of the fit."""
    fit_config = run.fit_config
    # loss function spec from fit_config
    model_fns = construct_model_functions(fit_config)

    log.info(f"Saving model config to {run.target_potential_file_name}")
    save_instructions_dict(
        run.target_potential_file_name, run.instructions, param_dtype=run.param_dtype
    )
    if run.args.eager:
        log.warning("Eager execution")
    jit_compile = fit_config.get("jit_compile", True)
    if run.args.no_jit:
        log.info("--no-jit option is provided")
        jit_compile = False
    if not run.args.save_model:
        log.info(f"Loss function: {model_fns.loss_fn}")
        log.info(f"JIT compilation: {jit_compile}")
        log.info(f"Optimization options: {fit_config.get('opt_params')}")
    return TensorPotential(
        potential=run.instructions,
        fit_config=fit_config,
        global_batch_size=run.global_batch_size,
        global_test_batch_size=run.global_test_batch_size,
        loss_function=model_fns.loss_fn,
        regularization_loss=model_fns.regularization_loss_fn,
        compute_metrics=model_fns.metrics_fn,
        model_train_function=model_fns.train_fn,
        model_compute_function=model_fns.compute_fn,
        param_dtype=run.param_dtype,
        eager_mode=run.args.eager,
        jit_compile=jit_compile,
        seed=run.seed,
        strategy=run.strategy,
        loss_norm_by_batch_size=fit_config.get("loss_norm_by_batch_size", False),
    )


def _export_and_stop(run: _Run) -> None:
    """--save-model: export the saved model (and the FS model on request) and stop."""
    tp = run.tp
    finalize_lora_if_enabled(tp)
    if run.args.save_fs:
        log.info("Exporting FS-model to `saved_model.yaml`, please wait...")
        tp.export_to_yaml("saved_model.yaml")
        log.info("Exporting to `saved_model.yaml` done")
    log.info("Saving model to `saved_model` and exit.")
    # a save with jit_compile=True converts the compute function to JIT for the rest of the
    # process, so this is the only export made here
    tp.save_model_with_aux_computes(
        "saved_model", jit_compile=True, communicated_keys=run.communicated_keys
    )
    raise _RunFinished


def _load_best_test_checkpoint(run: _Run, expect_partial=False, verbose=True, assert_consumed=False):
    if run.has_test_set:
        log.info("Loading best test loss model")
        run.tp.load_checkpoint(
            suffix=".best_test_loss",
            expect_partial=expect_partial,
            verbose=verbose,
            assert_consumed=assert_consumed,
        )


def _save_model(run: _Run, name: str) -> None:
    log.info(f"Saving model to `{name}`")
    run.tp.save_model_with_aux_computes(
        name, jit_compile=True, communicated_keys=run.communicated_keys
    )


def _prepare_training(run: _Run):
    """LoRA options, trainable variables, the training tensors and the epoch counters; return the batches."""
    tp, fit_config = run.tp, run.fit_config
    apply_lora_options(tp, run.potential_config, run.target_potential_file_name, run.param_dtype)
    if fit_config.get("trainable_variable_names"):
        trainable_names = fit_config["trainable_variable_names"]
        assert isinstance(trainable_names, list), "trainable_names must be a list"
        log.info(f"Set trainable variable names: {trainable_names}")
        tp.set_trainable_variables(trainable_names, verbose=True)

    train_batches, test_batches = convert_to_tensors_for_model(
        tp, run.train_data, run.test_data, strategy=run.strategy
    )
    run.train_data = None
    run.test_data = None
    gc.collect()

    if fit_config.get("reset_epoch_and_step") or run.args.reset_epoch_and_step:
        log.info(f"Reset epochs ({tp.epoch} -> 0) and steps ({tp.step} -> 0) counters")
        tp.reset_epoch_and_step()
    return train_batches, test_batches


def _optimize(run: _Run, train_batches, test_batches) -> None:
    opt = run.fit_config.get("optimizer", "Adam")
    kwargs = dict(
        fit_config=run.fit_config,
        train_ds=train_batches,
        test_ds=test_batches,
        strategy=run.strategy,
        seed=run.seed,
        train_grouping_df=run.train_grouping_df,
        test_grouping_df=run.test_grouping_df,
    )
    if opt in ["L-BFGS-B", "BFGS"]:
        log.info(f"Start {opt} optimization")
        train_bfgs(run.tp, **kwargs)
    elif opt == "Adam":
        log.info("Start Adam optimization")
        train_adam(run.tp, intra_epoch_redo=run.args.intra_epoch_redo, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer: {opt}. Can be 'Adam' or 'L-BFGS-B', 'BFGS'"
        )


def _train_and_save(run: _Run, train_batches, test_batches) -> None:
    """Run the optimisation, then load the best checkpoint and write the final model (and the FS model)."""
    try:
        _optimize(run, train_batches, test_batches)
        _load_best_test_checkpoint(run, assert_consumed=True, verbose=True)
        finalize_lora_if_enabled(run.tp)
        _save_model(run, name="final_model")
        if run.args.save_fs:
            log.info("Exporting FS-model, please wait...")
            run.tp.export_to_yaml("FS_model.yaml")
            log.info("Exporting FS-model done")
    except KeyboardInterrupt:
        log.info("Keyboard interruption is captured")
        raise _RunFinished from None


def _setup_run(args_parse, strategy, strategy_desc) -> _Run:
    """Everything before the model: input, strategy, batch sizes, dtype, foundation checkpoint, potential file, data."""
    run = _start_run(args_parse, strategy)
    _select_strategy(run, strategy, strategy_desc)
    _set_batch_sizes(run)
    run.user_set_param_dtype = (
        "param_dtype" in run.potential_config or "float_dtype" in run.potential_config
    )
    run.param_dtype, run.param_dtype_flag = get_param_dtype_from_config(
        run.potential_config
    )  # log deferred
    if args_parse.check_model:
        _check_model_and_stop(run)
    _use_foundation_checkpoint(run)
    _find_potential_file(run)
    # no need to perform expensive data processing if only saving a model
    if not args_parse.save_model:
        _load_datasets(run)
    else:
        run.potential_file_name = run.potential_file_name or os.path.join(
            run.output_dir, MODEL_CONFIG_YAML
        )
    return run


def _build_model(run: _Run) -> None:
    """Everything between the data and the TensorPotential: dtype, checkpoint, element reduction, instructions."""
    _infer_param_dtype(run)
    _resolve_checkpoint_name(run)
    _reduce_elements_if_requested(run)
    run.target_potential_file_name = os.path.join(run.output_dir, MODEL_CONFIG_YAML)
    _build_instructions(run)
    _inject_foundation_shift(run)
    run.tp = _create_tensorpotential(run)
    try_load_checkpoint(
        run.tp,
        restart_best_test=run.args.restart_best_test,
        restart_latest=run.args.restart_latest,
        restart_suffix=run.args.restart_suffix,
        checkpoint_name=run.checkpoint_name,
        expect_partial=run.args.save_model,
        verbose=True,
        assert_consumed=False,
    )


def main(argv=None, strategy=None, strategy_desc="", prog="gracemaker"):
    if argv is None:
        argv = []
    args_parse = build_parser(prog).parse_args(argv)
    if args_parse.template:
        generate_template_input()
    try:
        run = _setup_run(args_parse, strategy, strategy_desc)
        _build_model(run)
        if args_parse.save_model:
            _export_and_stop(run)
        train_batches, test_batches = _prepare_training(run)
        _train_and_save(run, train_batches, test_batches)
    except _RunFinished:
        sys.exit(0)


def check_model(args_yaml, param_dtype):
    rcut = args_yaml["cutoff"]
    potential_config = args_yaml["potential"]

    element_map = {s: i for i, s in enumerate(chemical_symbols[1:90])}
    log.info(f"Checking model with {len(element_map)} elements")
    cut_dict = args_yaml.get(tc.INPUT_CUTOFF_DICT)
    if cut_dict:
        log.info(f"User-defined cutoff dict: {cut_dict}")
    pot, _ = construct_model(
        potential_config, element_map=element_map, rcut=rcut, cutoff_dict=cut_dict
    )
    model = TPModel(pot)
    model.build(param_dtype=param_dtype)
    init_flat_vars = model.get_flat_trainable_variables()
    log.info("Model is constructed")
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
