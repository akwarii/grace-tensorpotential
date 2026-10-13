"""Batching, padding and statistics of the dataset preprocessing (the library side of ``grace_preprocess``)."""

import json
import logging
import os

import numpy as np
import pandas as pd
import tqdm

# the TensorFlow options are set before the first import of tensorflow
from tensorpotential import _tf_options  # noqa: F401

# isort: split
import tensorflow as tf

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
    split_batches_into_buckets,
)
from tensorpotential.data.process_df import (
    ASE_ATOMS,
)

EQUI_STRUCTURE_STRATEGY = "structures"

EQUI_ATOMS_STRATEGY = "atoms"

EQUI_NEIGH_STRATEGY = "neighbours"

orders = ["n_neighbours", "n_atoms", "n_structures"]

remap_type_to_tf_dtype = {
    np.int32: tf.int32,
    np.float64: tf.float64,
    np.float32: tf.float32,
    np.dtype("int32"): tf.int32,
    np.dtype("float64"): tf.float64,
    np.dtype("float32"): tf.float32,
}


def get_elements(df_iterrows):
    elements_set = set()
    for _, row in df_iterrows:
        atoms = row[ASE_ATOMS]
        elements_set.update(atoms.get_chemical_symbols())
    return sorted(elements_set)


def get_batch_dtypes(databuilders_list):
    batch_dtypes = {}
    for db in databuilders_list:
        batch_dtypes.update(
            {k: remap_type_to_tf_dtype[v] for k, v in db.get_batch_dtypes().items()}
        )
    return batch_dtypes


# Function to yield dict[str->numpy array], that defines each structure
def databuilder_generator(dataframes_rows, databuilders_list):
    for row_id, row in dataframes_rows:
        res_dict = {}
        for db in databuilders_list:
            res_dict.update(
                db.extract_from_row(row, structure_id=np.array(row_id, dtype=np.int32))
            )
        yield res_dict


def infer_tensorspec(dataframes_rows, databuilders_list):
    batch = next(databuilder_generator(dataframes_rows, databuilders_list))
    tf_spec = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            tf_shape = list(v.shape)
            if len(tf_shape) > 0:
                tf_shape[0] = None
            tf_spec[k] = tf.TensorSpec(
                shape=tf_shape, dtype=remap_type_to_tf_dtype[v.dtype], name=k
            )

    return tf_spec


def get_window_stat(batch_list):
    stat_dict = {"n_structures": 0, "n_atoms": 0, "n_neighbours": 0}

    for b in batch_list:
        stat_dict["n_structures"] += int(b[tc.N_STRUCTURES_BATCH_REAL].numpy())
        stat_dict["n_atoms"] += int(b[tc.N_ATOMS_BATCH_REAL].numpy())
        stat_dict["n_neighbours"] += int(b[tc.N_NEIGHBORS_REAL].numpy())

    return stat_dict


def compute_batches_df(dataset, batch_size, strategy=EQUI_STRUCTURE_STRATEGY):
    pbar = tqdm.tqdm(dataset, mininterval=2)
    iterator = iter(pbar)

    stat_list = []
    while True:
        try:
            batch_list = get_batch_list(iterator, batch_size, strategy=strategy)
        except StopIteration:
            break

        stat_list.append(get_window_stat(batch_list))
    structures_df = pd.DataFrame(stat_list)
    structures_df["bid"] = np.arange(len(structures_df))
    batches_df = structures_df.groupby("bid").agg(
        {"n_atoms": "sum", "n_neighbours": "sum", "n_structures": "sum"}
    )

    return batches_df.sort_values(
        ["n_neighbours", "n_atoms", "n_structures"], ascending=False
    ).reset_index(drop=True)


def get_batch_list(iterator, batch_size, strategy=EQUI_STRUCTURE_STRATEGY, **kwargs):
    if strategy == EQUI_STRUCTURE_STRATEGY:
        return get_equi_structures_batch_list(iterator, batch_size, **kwargs)
    if strategy == EQUI_ATOMS_STRATEGY:
        return get_equi_batch_list(
            iterator, batch_size, key="batch_tot_nat_real", **kwargs
        )
    if strategy == EQUI_NEIGH_STRATEGY:
        return get_equi_batch_list(iterator, batch_size, key="n_neigh_real", **kwargs)
    # TODO: balance batch size by number of atoms/neighbours
    raise NotImplementedError(f"Strategy {strategy} not implemented")


def get_equi_structures_batch_list(iterator, batch_size, **kwargs):
    """return batch_list from the iterator stream,
    always returning batch_size samples (or less for remainder)

    """
    batch_list = []
    try:
        for _ in range(batch_size):
            batch_list.append(next(iterator))
    except StopIteration as e:
        if not batch_list:
            raise e
    return batch_list


def get_equi_batch_list(iterator, batch_size, key="batch_tot_nat_real", **kwargs):
    """return batch_list from the iterator stream,
    trying to have AT LEAST "batch_size" number of 'key' in a batch
    """
    batch_list = []
    try:
        n_entities = 0
        while n_entities < batch_size:
            b = next(iterator)
            batch_list.append(b)
            n_entities += b[key]
    except StopIteration as e:
        if not batch_list:
            raise e
    return batch_list


# @tf.function(jit_compile=False)
def get_batch_stat_tf(batch):
    return {
        "n_structures": batch["batch_total_num_structures"],
        "n_atoms": batch["batch_tot_nat_real"],
        "n_neighbours": batch["n_neigh"],
    }


# @tf.function  # (jit_compile=False)
def get_bound_for_batch(batch, padding_bounds):
    stats = get_batch_stat_tf(batch)
    tf_stats = [stats[c] for c in orders]
    cond = np.all(padding_bounds >= tf_stats, axis=1)
    ind = np.where(cond)
    ind = ind[0][0]
    return padding_bounds[ind]


def dataframe_iterrows_generator(
    file_paths,
    shard_by_file=False,
    shard_by_row=False,
    task_index=0,
    total_num_tasks=1,
    verbose=False,
):
    if shard_by_file:
        file_paths = file_paths[task_index::total_num_tasks]
    file_iterator = tqdm.tqdm(file_paths, mininterval=2) if verbose else file_paths
    for file_path in file_iterator:
        try:
            df = pd.read_pickle(file_path, compression="gzip")
            if verbose:
                logging.info(f"Loaded {file_path}, total shape: {df.shape}")
            if shard_by_row:
                df = df.iloc[task_index::total_num_tasks]
            row_iterator = df.iterrows()
            if verbose:
                row_iterator = tqdm.tqdm(row_iterator, mininterval=2, total=len(df))
            for ind, row in row_iterator:
                yield ind, row
        except Exception as e:  # noqa: BLE001  # any failure of one input file (corrupt pickle, missing module) skips that file only
            logging.error(f"Error reading file: {file_path}. Error: {e}")


def join_and_pad_generator(
    dataset,
    databuilders_list,
    padding_bounds,
    batch_size,
    strategy=EQUI_STRUCTURE_STRATEGY,
    verbose=False,
    **kwargs,
):
    pbar = tqdm.tqdm(dataset, mininterval=2) if verbose else dataset
    iterator = iter(pbar)
    while True:
        try:
            batch_list = get_batch_list(
                iterator, batch_size, strategy=strategy, **kwargs
            )
        except StopIteration:
            break

        # Apply join_to_batch function
        #  Note! number of keys in result dict will be larger
        batch = {}
        for db in databuilders_list:
            batch.update(db.join_to_batch(batch_list))
        batch["n_neigh"] = len(batch["bond_vector"])

        bound = get_bound_for_batch(batch, padding_bounds)

        batch_max_pad_dict = {
            tc.PAD_MAX_N_NEIGHBORS: bound[0],
            tc.PAD_MAX_N_ATOMS: bound[1],
            tc.PAD_MAX_N_STRUCTURES: bound[2],
        }

        for data_builder in databuilders_list:
            data_builder.pad_batch(batch, batch_max_pad_dict)

        yield batch


def infer_batch_signature(batch, batch_dtypes):
    tf_spec = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            tf_shape = list(v.shape)
            if len(tf_shape) > 0:
                tf_shape[0] = None
            tf_spec[k] = tf.TensorSpec(shape=tf_shape, dtype=batch_dtypes[k], name=k)
        else:
            tf_spec[k] = tf.TensorSpec(shape=None, dtype=batch_dtypes[k], name=k)
    return tf_spec


def save_batches_df(batches_df, batches_df_fname):
    os.makedirs(os.path.dirname(batches_df_fname), exist_ok=True)
    batches_df.to_pickle(batches_df_fname)


def load_and_reduce_stats(output_path, total_num_tasks, max_n_buckets):
    all_batches_df = []
    for task_index in range(total_num_tasks):
        batches_df_fname = os.path.join(
            output_path,
            "stage2",
            f"batches_df-{task_index + 1}-of-{total_num_tasks}.pkl.gz",
        )
        if not os.path.isfile(batches_df_fname):
            raise RuntimeError(
                f"Stage 2 is not completed, no such file {batches_df_fname}"
            )

        batches_df = pd.read_pickle(batches_df_fname)
        all_batches_df.append(batches_df)
    batches_df = pd.concat(all_batches_df, axis=0)

    buckets_list = split_batches_into_buckets(batches_df, max_n_buckets)

    def compute_split_loss(buckets_list):
        return sum((subdf.max(axis=0) - subdf).sum() for subdf in buckets_list)

    total_samples = batches_df[["n_atoms", "n_neighbours", "n_structures"]].sum()
    relative_split_loss = compute_split_loss(buckets_list) / total_samples
    logging.info(f"Relative split loss: {relative_split_loss.to_dict()}")

    padding_bounds = []
    for bucket in buckets_list:
        max_nstruct = bucket["n_structures"].max()
        max_nat = bucket["n_atoms"].max()
        max_nneigh = bucket["n_neighbours"].max()

        # pad at least one atom and structure if it has to pad nneigh
        is_pad_nneigh = np.any(bucket["n_neighbours"] != max_nneigh)
        is_pad_atoms = np.any(bucket["n_atoms"] != max_nat) or is_pad_nneigh
        is_pad_struct = is_pad_atoms

        if is_pad_atoms:
            max_nat += 1
        if is_pad_struct:
            max_nstruct += 1
        padding_bounds.append((max_nneigh, max_nat, max_nstruct))

    padding_bounds = sorted(padding_bounds)  # neigh, atoms, structures
    padding_bounds = np.array(padding_bounds).astype(np.int32).tolist()
    return padding_bounds


def load_params(output_path):
    # load elements_map
    fname = os.path.join(output_path, "params.json")
    if os.path.isfile(fname):
        try:
            with open(fname, "r") as f:
                params = json.load(f)
                elements_map = params["elements_map"]
                cutoff = params["cutoff"]
                cutoff_dict = params.get("cutoff_dict")
                precision = params["precision"]
            return elements_map, cutoff, cutoff_dict, precision
        except (OSError, ValueError, KeyError, TypeError) as e:
            # unreadable file, invalid JSON, a missing key, or JSON that is not a mapping
            logging.error(f"Can't parse {fname}: {e}")


def save_params(
    output_path, elements_map, cutoff, precision="float64", cutoff_dict=None
):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(
            {
                "elements_map": elements_map,
                "cutoff": cutoff,
                "cutoff_dict": cutoff_dict,
                "precision": precision,
            },
            f,
        )


def get_databuilders(
    elements_map,
    cutoff,
    args_parse,
    precision="float64",
    cutoff_dict=None,
):
    geom_db = GeometricalDataBuilder(
        elements_map=elements_map,
        cutoff=cutoff,
        cutoff_dict=cutoff_dict,  # TODO: provide parameter
        is_fit_stress=args_parse.is_fit_stress,
        float_dtype=precision,
        dense_nbr=getattr(args_parse, "dense_nbr", False),
    )
    ref_db = ReferenceEnergyForcesStressesDataBuilder(
        is_fit_stress=args_parse.is_fit_stress,
        energy_col=args_parse.energy_col,
        forces_col=args_parse.forces_col,
        stress_col=args_parse.stress_col,
        float_dtype=precision,
        stress_units=args_parse.stress_units,
    )
    return [geom_db, ref_db]
