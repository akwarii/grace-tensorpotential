#!/usr/bin/env python
import argparse
from collections.abc import Sequence
import glob
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np
from tensorpotential import _tf_options  # noqa: F401
import tensorflow as tf
import tqdm

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    DEFAULT_STRESS_UNITS,
)
from tensorpotential.data.process_df import (
    ENERGY_CORRECTED_COL,
    FORCES_COL,
    STRESS_COL,
)
from tensorpotential.utils import NumpyEncoder
from tensorpotential.data.preprocess import (
    EQUI_STRUCTURE_STRATEGY,
    EQUI_ATOMS_STRATEGY,
    EQUI_NEIGH_STRATEGY,
    get_elements,
    get_batch_dtypes,
    databuilder_generator,
    infer_tensorspec,
    compute_batches_df,
    dataframe_iterrows_generator,
    join_and_pad_generator,
    infer_batch_signature,
    save_batches_df,
    load_and_reduce_stats,
    load_params,
    save_params,
    get_databuilders,
)
from tensorpotential.data.preprocess import (  # noqa: F401  (re-exported: the old import path)
    orders,
    remap_type_to_tf_dtype,
    get_window_stat,
    get_batch_list,
    get_equi_structures_batch_list,
    get_equi_batch_list,
    get_batch_stat_tf,
    get_bound_for_batch,
)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# The units of the stress column that ReferenceEnergyForcesStressesDataBuilder converts to eV/A3
# (same spelling as `data: stress_units` of the gracemaker input file).
STRESS_UNITS_CHOICES = (DEFAULT_STRESS_UNITS, "GPa", "kbar", "-kbar")

ALEXANDRIA_ELEMENTS = [
    "Ac",
    "Ag",
    "Al",
    "Ar",
    "As",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bi",
    "Br",
    "C",
    "Ca",
    "Cd",
    "Ce",
    "Cl",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Dy",
    "Er",
    "Eu",
    "F",
    "Fe",
    "Ga",
    "Gd",
    "Ge",
    "H",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "I",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Lu",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Ni",
    "Np",
    "O",
    "Os",
    "P",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Pr",
    "Pt",
    "Pu",
    "Rb",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Sc",
    "Se",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "U",
    "V",
    "W",
    "Xe",
    "Y",
    "Yb",
    "Zn",
    "Zr",
]

FULL_ELEMENTS = [
    "Ac",
    "Ag",
    "Al",
    "Ar",
    "As",
    "At",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bi",
    "Br",
    "C",
    "Ca",
    "Cd",
    "Ce",
    "Cl",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Dy",
    "Er",
    "Eu",
    "F",
    "Fe",
    "Fr",
    "Ga",
    "Gd",
    "Ge",
    "H",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "I",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Lu",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Ni",
    "Np",
    "O",
    "Os",
    "P",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Po",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rh",
    "Rn",
    "Ru",
    "S",
    "Sb",
    "Sc",
    "Se",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "U",
    "V",
    "W",
    "Xe",
    "Y",
    "Yb",
    "Zn",
    "Zr",
]  #  OMAT + UEA elements.

OMOL_N_ELEMENTS = [
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Br",
    "I",
]


def build_parser(prog: str = 'grace_preprocess'):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Precompute dataset and save into TF.Dataset format",
    )

    parser.add_argument("input", help="input pkl.gz file", type=str, nargs="+")

    parser.add_argument(
        "-o", "--output", help="output file name", type=str, default="tf_dataset"
    )

    parser.add_argument(
        "--sharded-input",
        action="store_true",
        default=False,
        help="Flag to show that input files are sharded",
    )
    parser.add_argument(
        "-e",
        "--elements",
        type=str,
        default=None,
        help="List of elements. Possible presets: `ALL` (except last 23 elements), `Alexandria` or `MP`",
    )

    parser.add_argument("-b", "--batch_size", type=int, default=8)

    parser.add_argument("-bu", "--max-n-buckets", type=int, default=5)

    parser.add_argument("-c", "--cutoff", type=float, default=6)

    parser.add_argument("-cd", "--cutoff_dict", type=str, default=None)

    parser.add_argument("--compression", type=str, default="GZIP")

    parser.add_argument("--energy-col", type=str, default=ENERGY_CORRECTED_COL)
    parser.add_argument("--forces-col", type=str, default=FORCES_COL)
    parser.add_argument("--stress-col", type=str, default=STRESS_COL)
    parser.add_argument("--is-fit-stress", action="store_true", default=False)
    parser.add_argument(
        "--stress-units",
        type=str,
        choices=STRESS_UNITS_CHOICES,
        default=DEFAULT_STRESS_UNITS,
        help="Units of the stress column, converted to eV/A3. Write `--stress-units=-kbar` "
        "for the last value (a separate `-kbar` argument is read as an option).",
    )
    parser.add_argument(
        "--precision", type=str, default="float64"
    )  # can be float32 or float64

    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default=EQUI_STRUCTURE_STRATEGY,
        help=f"Strategy to batch splitting. Possible values: {EQUI_STRUCTURE_STRATEGY} (default), {EQUI_ATOMS_STRATEGY}, {EQUI_NEIGH_STRATEGY}",
    )

    parser.add_argument("--task-id", type=int, default=0, help="ZERO based ID of task")

    parser.add_argument(
        "--total-task-num", type=int, default=1, help="Total number of tasks"
    )

    parser.add_argument(
        "--rerun", action="store_true", default=False, help="Enforce to rerun process"
    )

    parser.add_argument(
        "--stage-1",
        action="store_true",
        default=False,
        help="Run stage 1, precompute samples (non-batched)",
    )

    parser.add_argument(
        "--stage-2",
        action="store_true",
        default=False,
        help="Run stage 2, compute padding bounds",
    )

    parser.add_argument(
        "--stage-3",
        action="store_true",
        default=False,
        help="Run stage 3, padding batches",
    )

    parser.add_argument(
        "--stage-4",
        action="store_true",
        default=False,
        help="Run stage 4, compute statistics",
    )

    parser.add_argument(
        "--remove_stage1",
        action="store_true",
        default=False,
        dest="remove_stage1",
        help="If True - during stage 3, remove corresponding shard from stage1 folder",
    )

    parser.add_argument(
        "--dense-nbr",
        action="store_true",
        default=False,
        dest="dense_nbr",
        help="Mark the dataset for the dense (reshape) aggregation. Reserved for the "
        "batched dense data prep (per-atom-uniform layout bucketed by max_neigh, TODO); "
        "should match the model (input.yaml::potential::dense_nbr).",
    )

    return parser


class StageSelectionError(ValueError):
    """The command line selects no stage, or more than one."""


@dataclass(frozen=True)
class _StagePaths:
    """The files and folders of one task, below the output folder."""

    stage1: str
    batches_df: str
    stage3: str

    @property
    def stage1_temp(self) -> str:
        """The folder stage 1 writes to before it renames it to ``stage1``."""
        return self.stage1 + ".tmp"

    @property
    def stage3_temp(self) -> str:
        """The folder stage 3 writes to before it renames it to ``stage3``."""
        return self.stage3 + ".tmp"

    @classmethod
    def of_task(cls, output_path: str, task_index: int, total_num_tasks: int) -> "_StagePaths":
        shard = f"{task_index + 1}-of-{total_num_tasks}"
        stage1 = os.path.join(output_path, "stage1", f"shard_{shard}")
        stage3 = os.path.join(output_path, "stage3", f"shard_{shard}")
        batches_df = os.path.join(output_path, "stage2", f"batches_df-{shard}.pkl.gz")
        return cls(stage1, batches_df, stage3)


@dataclass
class _Settings:
    """The parameters of the run: those stored by stage 1 when there are any, else those of the command line."""

    elements_map: dict | None
    cutoff: float
    cutoff_dict: str | None
    precision: str
    databuilders: list | None = None
    batch_dtypes: dict | None = None


def _selected_stage(args_parse) -> int:
    """The number (1 to 4) of the one stage flag that is set; raise StageSelectionError for none or several."""
    flags = [args_parse.stage_1, args_parse.stage_2, args_parse.stage_3, args_parse.stage_4]
    if sum(flags) == 0:
        raise StageSelectionError(
            "Error: EITHER --stage-1 or --stage-2 or --stage-3 options must be provided"
        )
    if sum(flags) > 1:
        raise StageSelectionError(
            "Error: ONLY --stage-1 or --stage-2 or --stage-3 options must be provided"
        )
    return flags.index(True) + 1


def _log_options(args_parse) -> None:
    logging.info(f"Input filename: {args_parse.input}")
    logging.info(f"Output path: {args_parse.output}")
    logging.info(f"Cutoff: {args_parse.cutoff}")
    logging.info(f"Cutoff dict: {args_parse.cutoff_dict}")
    logging.info(f"Batch size: {args_parse.batch_size}")
    logging.info(f"Buckets: {args_parse.max_n_buckets}")
    logging.info(f"Compression: {args_parse.compression}")
    logging.info(f"Elements: {args_parse.elements}")


def _load_settings(args_parse) -> _Settings:
    """Read the parameters stored by stage 1; the data builders and dtypes come with them."""
    settings = _Settings(
        None, args_parse.cutoff, args_parse.cutoff_dict, args_parse.precision
    )
    params = load_params(args_parse.output)
    if params is None:
        return settings
    logging.info(f"Params loaded: {params}")
    settings.elements_map, settings.cutoff, settings.cutoff_dict, settings.precision = params
    # TODO: provide parameters
    settings.databuilders = get_databuilders(
        settings.elements_map,
        settings.cutoff,
        args_parse,
        precision=settings.precision,
        cutoff_dict=settings.cutoff_dict,
    )
    settings.batch_dtypes = get_batch_dtypes(settings.databuilders)
    settings.batch_dtypes["n_neigh"] = tf.int32
    return settings


def _elements_of_run(elements, elements_map, df_rows) -> list:
    """The elements of stage 1: given (a preset or a list), stored by an earlier run, or read from the data."""
    if elements is None:
        if elements_map is not None:
            elements = sorted(elements_map.keys())
            logging.info(f"Loaded elements: {elements}")
            return elements
        logging.warning(
            "Elements are not provided, so they will be extracted from ALL dataframes. "
            + "It may lead to unnecessary slowdown. Better use --elements option"
        )
        logging.info("Extracting elements")
        elements = get_elements(df_iterrows=df_rows(True))
        logging.info(f"Extracted elements: {elements}")
        return elements
    if elements == "ALL":
        elements = FULL_ELEMENTS
    elif elements == "OMOL_N":
        elements = OMOL_N_ELEMENTS
    elif elements in ["MP", "Alexandria"]:
        elements = ALEXANDRIA_ELEMENTS
    else:
        elements = sorted(elements.split())
    logging.info(f"Provided elements: {elements}")
    return elements


def _input_files_of_task(args_parse) -> list:
    input_fnames = args_parse.input
    if args_parse.sharded_input:
        input_fnames = input_fnames[args_parse.task_id :: args_parse.total_task_num]
        logging.info(f"Sharded {len(input_fnames)} input file(s): {input_fnames}")
    else:
        logging.info(f"Total {len(input_fnames)} input file(s): {input_fnames}")
    return input_fnames


def _move_into_place(temp_path: str, path: str) -> None:
    logging.info(f"Renaming {temp_path} to {path}")
    if os.path.isdir(path):
        shutil.rmtree(path)
    shutil.move(temp_path, path)


def _run_stage_1(args_parse, paths: _StagePaths, settings: _Settings) -> None:
    """Stage 1: the neighbour-list dataset of every structure, not batched."""
    if os.path.isdir(paths.stage1) and not args_parse.rerun:
        logging.info(f"Stage 1 is already finished, folder {paths.stage1} exists")
        return
    input_fnames = _input_files_of_task(args_parse)

    def df_rows(verbose):
        return dataframe_iterrows_generator(
            input_fnames,
            shard_by_row=not args_parse.sharded_input,
            task_index=args_parse.task_id,
            total_num_tasks=args_parse.total_task_num,
            verbose=verbose,
        )

    elements = _elements_of_run(args_parse.elements, settings.elements_map, df_rows)
    elements_map = {e: i for i, e in enumerate(elements)}
    logging.info(f"Elements map: {elements_map}")
    save_params(
        args_parse.output,
        elements_map,
        settings.cutoff,
        settings.precision,
        settings.cutoff_dict,
    )
    databuilders = get_databuilders(
        elements_map,
        settings.cutoff,
        args_parse,
        precision=settings.precision,
        cutoff_dict=settings.cutoff_dict,
    )

    logging.info("Stage 1: Computing neigh-list dataset")
    logging.info(f"Cleaning temp folder {paths.stage1_temp}")
    shutil.rmtree(paths.stage1_temp, ignore_errors=True)
    output_signature = infer_tensorspec(df_rows(False), databuilders)
    dataset = tf.data.Dataset.from_generator(
        lambda: databuilder_generator(df_rows(True), databuilders),
        output_signature=output_signature,
    )
    logging.info(f"Processing and saving to {paths.stage1_temp}")
    dataset.save(paths.stage1_temp, compression=args_parse.compression)
    _move_into_place(paths.stage1_temp, paths.stage1)
    logging.info("Stage 1 complete")


def _run_stage_2(args_parse, paths: _StagePaths) -> None:
    """Stage 2: the statistics of the batches (atoms, neighbours, structures per batch)."""
    if os.path.isfile(paths.batches_df) and not args_parse.rerun:
        logging.info(f"Stage 2: {paths.batches_df} already exists, stopping")
        return
    logging.info("Stage 2: Computing batches statistics")
    logging.info(f"Loading {paths.stage1}, batch size={args_parse.batch_size}")
    dataset = tf.data.Dataset.load(paths.stage1, compression=args_parse.compression)
    batches_df = compute_batches_df(
        dataset, batch_size=args_parse.batch_size, strategy=args_parse.strategy
    )
    save_batches_df(batches_df, paths.batches_df)
    logging.info(f"Batches statistics saved to {paths.batches_df}")


def _write_padding_bounds(args_parse) -> np.ndarray:
    """Reduce the statistics of all tasks to padding bounds, store them and return them as an array."""
    logging.info(f"Reducing padding bounds in {args_parse.output}")
    padding_bounds = load_and_reduce_stats(
        args_parse.output, args_parse.total_task_num, args_parse.max_n_buckets
    )
    os.makedirs(os.path.join(args_parse.output, "stage3"), exist_ok=True)
    bounds_file = os.path.join(args_parse.output, "stage3", "reduced_padding_bounds.json")
    with open(bounds_file, "w") as f:
        json.dump(padding_bounds, f)
    logging.info(f"Loaded padding_bounds={padding_bounds}")
    return np.array(padding_bounds)


def _save_padded_dataset(args_parse, paths: _StagePaths, settings: _Settings, padding_bounds) -> None:
    logging.info(f"Stage 3: Padding dataset and saving to {paths.stage3}")
    dataset = tf.data.Dataset.load(paths.stage1, compression=args_parse.compression)
    logging.info(f"Cleaning temp folder {paths.stage3_temp}")
    shutil.rmtree(paths.stage3_temp, ignore_errors=True)
    logging.info(f"Processing and saving to temp folder {paths.stage3_temp}")

    def padded_batches(**kwargs):
        return join_and_pad_generator(
            dataset,
            settings.databuilders,
            padding_bounds,
            args_parse.batch_size,
            strategy=args_parse.strategy,
            **kwargs,
        )

    batch = next(iter(padded_batches()))
    batch_signature = infer_batch_signature(batch, settings.batch_dtypes)
    padded_dataset = tf.data.Dataset.from_generator(
        lambda: padded_batches(verbose=True), output_signature=batch_signature
    )
    padded_dataset.save(paths.stage3_temp, compression=args_parse.compression)
    _move_into_place(paths.stage3_temp, paths.stage3)
    logging.info(f"Saved to {paths.stage3}")


def _run_stage_3(args_parse, paths: _StagePaths, settings: _Settings) -> None:
    """Stage 3: pad every batch to the bounds of its bucket and save the padded dataset."""
    if os.path.isdir(paths.stage3) and not args_parse.rerun:
        logging.info(f"Folder {paths.stage3} already exists, skipping")
    else:
        padding_bounds = _write_padding_bounds(args_parse)
        _save_padded_dataset(args_parse, paths, settings, padding_bounds)
        if args_parse.remove_stage1:
            logging.info(
                f"Shard {paths.stage3} complete, removing corresponding {paths.stage1}"
            )
            shutil.rmtree(paths.stage1)
    logging.info("Done")


def _sums_over_batches(shard_paths: list, compression: str) -> dict:
    """Walk all padded batches of the shards and sum the real atoms, neighbours, structures and squared forces."""
    datasets = [
        tf.data.Dataset.load(filepath, compression=compression) for filepath in shard_paths
    ]
    dataset = tf.data.Dataset.from_tensor_slices(datasets).interleave(lambda x: x)
    sums = {"batches": 0, "sum_sqr_forces": 0, "atoms": 0, "neighbours": 0, "structures": 0}
    for b in tqdm.tqdm(dataset):
        sums["batches"] += 1
        nat_real = int(b[tc.N_ATOMS_BATCH_REAL].numpy())
        sums["neighbours"] += int(b[tc.N_NEIGHBORS_REAL].numpy())
        sums["structures"] += int(b[tc.N_STRUCTURES_BATCH_REAL].numpy())
        cur_forces = b[tc.DATA_REFERENCE_FORCES].numpy()[:nat_real]
        sums["sum_sqr_forces"] += np.sum(cur_forces**2)
        sums["atoms"] += nat_real
    return sums


def _run_stage_4(args_parse) -> None:
    """Stage 4 (task 0 only): the statistics of the whole dataset, stored as stats.json."""
    if args_parse.task_id != 0:
        return
    output_path = args_parse.output
    stats_fname = os.path.join(output_path, "stage3", "stats.json")
    elements_map, cutoff, cutoff_dict, precision = load_params(output_path)
    shard_paths = glob.glob(os.path.join(output_path, "stage3", "shard_*-of-*"))
    logging.info(f"Number of found shards: {len(shard_paths)}")
    sums = _sums_over_batches(shard_paths, args_parse.compression)
    # currently avg per-component. Maybe must be per-vector ?, i.e. sqrt(3) times arger ?
    scale = np.sqrt(sums["sum_sqr_forces"] / sums["atoms"] / 3)
    stats = {
        "element_map": elements_map,
        "scale": scale,
        "avg_n_neigh": sums["neighbours"] / sums["atoms"],
        "total_num_of_neighs": sums["neighbours"],
        "total_num_of_atoms": sums["atoms"],
        "sum_sqr_forces": sums["sum_sqr_forces"],
        "total_num_structures": sums["structures"],
        "total_num_of_batches": sums["batches"],
        "cutoff": cutoff,
        "cutoff_dict": cutoff_dict,
        "batch_size": args_parse.batch_size,
        "precision": precision,
    }
    logging.info(f"Aggregated stats: {stats}")
    with open(stats_fname, "w") as f:
        json.dump(stats, f, cls=NumpyEncoder)


def main(argv: Sequence[str] | None = None, *, prog: str = "grace_preprocess") -> int:
    # TODO: apply reference energy (esa_dict)
    # TODO: if  shift -> compute esa_dict from lstsq (optional), scale (necessary!)
    # TODO: Extract elements and element_map
    # TODO: apply weighting
    # TODO: user_cutoff_dict = args_yaml.get(tc.INPUT_CUTOFF_DICT)

    args_parse = build_parser(prog).parse_args(argv)
    log_format = (
        f"%(asctime)s %(levelname).1s [Worker {args_parse.task_id + 1}/{args_parse.total_task_num}]"
        "- %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y/%m/%d %H:%M:%S")
    try:
        stage = _selected_stage(args_parse)
    except StageSelectionError as error:
        logging.error(str(error))
        sys.exit(1)

    _log_options(args_parse)
    paths = _StagePaths.of_task(args_parse.output, args_parse.task_id, args_parse.total_task_num)
    settings = _load_settings(args_parse)
    if stage == 1:
        _run_stage_1(args_parse, paths, settings)
    elif stage == 2:
        _run_stage_2(args_parse, paths)
    elif stage == 3:
        _run_stage_3(args_parse, paths, settings)
    else:
        _run_stage_4(args_parse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
