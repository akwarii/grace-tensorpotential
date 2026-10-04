from __future__ import annotations

import json
import os

import os.path as p
import sys
import time

import numpy as np
import pandas as pd
import logging
from contextlib import contextmanager

from tensorpotential.instructions.base import ElementsReduceInstructionMixin
from yaml import safe_load

from tensorflow.dtypes import float32, float64, DType
from tensorpotential.core.cutoffs import CUTOFF_PRESETS, process_cutoff_dict  # noqa: F401  (re-exported; the code moved to core)
from tensorpotential.metadata_utils import get_dtype_by_name


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def is_chief(strategy) -> bool:
    """Whether the current process should perform chief-only work (file writes, etc.).

    Returns True for single-process strategies (default / MirroredStrategy) and for
    the chief worker under MultiWorkerMirroredStrategy. Returns False only for
    non-chief MWMS workers.
    """
    if strategy is None:
        return True
    resolver = getattr(strategy, "cluster_resolver", None)
    if resolver is None:
        return True
    task_type = getattr(resolver, "task_type", None)
    task_id = getattr(resolver, "task_id", None)
    if task_type is None:
        return True
    if task_type == "chief":
        return True
    return task_type == "worker" and task_id == 0


def load_metrics(fname):
    """Load metrics from YAML file, written by gracemaker"""
    with open(fname, "r") as f:
        df = pd.json_normalize(safe_load(f))
    return df


def load_fit(folder):
    """
    Load and compile the metrics and status from a specified folder.

    This function loads metrics from YAML files located in the given folder,
    specifically "test_metrics.yaml" and "train_metrics.yaml". It also checks
    for the presence of a "finished.txt" file to determine if the process is
    finalized.

    Args:
        folder (str): The path to the folder containing the metrics files and
                      the optional "finished.txt" file.

    Returns:
        dict: A dictionary containing the following keys:
            - "test": The loaded test metrics from "test_metrics.yaml".
            - "train": The loaded train metrics from "train_metrics.yaml".
            - "name": The name of the folder provided.
            - "final": A boolean indicating if the "finished.txt" file exists
                       in the folder, signifying that the process is complete.
    """
    res = {
        "test": load_metrics(os.path.join(folder, "test_metrics.yaml")),
        "train": load_metrics(os.path.join(folder, "train_metrics.yaml")),
        "name": folder,
        "final": False,
    }
    if os.path.isfile(os.path.join(folder, "finished.txt")):
        res["final"] = True
    return res


def plot_fit(
    fit,
    k="rmse/f_comp",
    name=None,
    plot_test=True,
    plot_train=True,
    ax=None,
    x_name=None,
):
    """
    Plot training and test metrics over epochs.

    This function visualizes the specified metric from both the training and test
    data over the course of training epochs, using Matplotlib for plotting.

    Args:
    fit (dict): A dictionary containing training and test metrics. Expected to
        have keys "train" and "test", each of which should be a DataFrame with metrics.

    k (str, optional): The key representing the metric to be plotted. Defaults to "rmse/f_comp".

    name (str, optional): The name to be used in the plot legend. If not provided, fit["shortname"] is used.

    plot_test (bool, optional): Whether to plot the test metric. Defaults to True.

    plot_train (bool, optional): Whether to plot the train metric. Defaults to True.

    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, the current axes will be used.

    x_name: (default - None): name of x-axis, if None - index+1 will be used
    Returns:
    None

    Notes:
    - The function plots the test metric with a solid line and includes the
    minimum value of the test metric in the legend.
    - The train metric is plotted with a dashed line and uses the same color
    as the test metric if both are plotted.
    - The x-axis is labeled as "Epoch" and the y-axis is labeled with the metric key k.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    if name is None:
        name = fit["shortname"]
    c = None
    if plot_test:
        y = fit["test"][k]
        y_min = y.min()
        short_name = name
        x = fit["test"].index + 1 if x_name is None else fit["test"][x_name]
        p = ax.plot(x, y, label=f"{short_name}({y_min * 1e3:.1f})")
        c = p[0].get_color()
    if plot_train:
        x = fit["train"].index + 1 if x_name is None else fit["train"][x_name]
        ax.plot(x, fit["train"][k], c=c, ls="--")

    ax.set_ylabel(k)
    ax.set_xlabel("Epoch")


def plot_many_fits(fits, k="rmse/f_comp", plot_test=True, plot_train=False):
    """
    Plot metrics for multiple training fits.

    This function visualizes the specified metric from multiple training fits,
    ordering them by the minimum test metric value in descending order. It can plot
    both training and test metrics.

    Args:
    fits (Union[dict, list]): A collection of training fits. If a dictionary is
    provided, its values are used. If a list is provided, it is used directly.

    k (str, optional): The key representing the metric to be plotted. Defaults to "rmse/f_comp".

    plot_test (bool, optional): Whether to plot the test metric for each fit. Defaults to True.

    plot_train (bool, optional): Whether to plot the train metric for each fit. Defaults to False.

    Returns:
    None

    Notes:
    - The function sorts the fits in descending order based on the minimum value
    of the test metric.
    - For each fit, the function prints the name of the fit and then calls
    plot_fit to create the plot.
    """
    if isinstance(fits, dict):
        list_of_fits = list(fits.values())
    elif isinstance(fits, list):
        list_of_fits = fits.copy()

    list_of_fits = sorted(
        list_of_fits, key=lambda fd: fd["test"][k].min(), reverse=True
    )
    for f in list_of_fits:
        print(f["name"])
        plot_fit(f, k=k, plot_test=plot_test, plot_train=plot_train)


def update_fit_metrics(fit_dict, folders, wait_time_seconds=3600, rerun=False):
    """
    Update INPLACE fit metrics for multiple folders.

    This function updates a dictionary of fit metrics by loading metrics from a
    list of specified folders. It skips folders that already have finished fits
    unless the metrics need updating.

    Args:
    fit_dict (dict): A dictionary where keys are folder names and values are fit metrics dictionaries.

    folders (list): A list of folder names from which to load and update fit metrics.

    Returns:
    None

    Notes:
    - The function skips updating fits that are already marked as finished in
    the fit_dict.
    - For each folder, it attempts to load the fit metrics using load_fit.
    If successful, the metrics are updated in the fit_dict.
    - If an error occurs during loading, it catches the exception and prints an error message.
    """
    for f in folders:
        if f in fit_dict:
            res = fit_dict[f]
            if res.get("finished", False) and not rerun:
                continue

            test_fname = os.path.join(f, "test_metrics.yaml")
            last_timestamp = os.path.getmtime(test_fname)
            current_time = time.time()
            pass_time = current_time - last_timestamp
            if pass_time > wait_time_seconds:
                print(
                    f"{f} - no update for {int(pass_time//3600)}h:{int((pass_time % 3600)//60)}m:{int((pass_time % 3600) % 60)}s, mark as finished"
                )
                res["finished"] = True
                continue
        try:
            fd = load_fit(f)
            print(f"{f}: {len(fd['test'])} epochs")
            fit_dict[f] = fd
        except Exception as e:
            print("Error:", e)


def get_common_prefix(names, align_folder=True):
    """
    Return common string prefix for given list of names (str)
    """
    common_prefix = p.commonprefix(list(names))

    if align_folder and not common_prefix.endswith("/"):
        common_prefix = "/".join(common_prefix.split("/")[:-1])

    return common_prefix


def process_fit_dict(fit_dict, fkey="rmse/f_comp", align_folder=True):
    """
    Process fit dict from gracemaker, return common prefix, dataframe of best fits and
    grouped by common prefix (different seeds) dataframe
    """
    for k, fd in fit_dict.items():
        short_name = k.split("/seed/")[0]
        fd["interim_name"] = short_name

    common_prefix = get_common_prefix(
        [fd["interim_name"] for k, fd in fit_dict.items()], align_folder=align_folder
    )

    for k, fd in fit_dict.items():
        fd["shortname"] = fd["interim_name"][len(common_prefix) :]
        fd["shortname_and_seed"] = k[len(common_prefix) :]

    row_list = []
    for name, fd in fit_dict.items():
        df = fd["test"]
        row = df.sort_values(fkey, ascending=True).iloc[0]
        row["name"] = name
        row["shortname"] = fd["shortname"]
        row_list.append(row)

    df = pd.DataFrame(row_list).sort_values(fkey)
    gdf = (
        df.drop(columns=["name"]).groupby("shortname").agg(["mean", "std", "min", list])
    )
    return (common_prefix, df, gdf)


def discovery_fit_folders(root):
    """
    Automatically discover fit folders

    Parameters:
        root (str): Root directory of fit folders

    Return:
        List of fit folders
    """
    folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "test_metrics.yaml" in filenames and "train_metrics.yaml" in filenames:
            folders.append(dirpath)
    return folders


def plot_dashboard(
    fit_dict,
    metric="rmse",
    plot_train=False,
    ax_e=None,
    ax_f=None,
    include_list=None,
    exclude_list=None,
    label="shortname",
    x_name=None,
):
    """Plot dashboard metrics

    Usage:

    fit_dict={}
    folders = discovery_fit_folders('/path/to/root/')
    update_fit_metrics(fit_dict, folders)
    fkey='mae/f_comp'
    common_prefix, df, gdf=process_fit_dict(fit_dict, fkey=fkey, align_folder=True)

    fig, ax_e, ax_f = plot_dashboard(fit_dict)
    ax_e.legend(ncol=2, bbox_to_anchor=(0.5,-0.15), loc="upper center")
    ax_f.legend(ncol=2, bbox_to_anchor=(0.5,-0.15), loc="upper center")

    ax_e.set_xscale('log')
    fig.suptitle(common_prefix)
    fig.tight_layout()


    Parameters:
        fit_dict (dict): Dictionary of fit metrics
        metric (str): Metric to plot. Default - rmse
        plot_train (bool): Plot training data. Default - False
        ax_e (Axes): Axes on which to plot
        ax_f (Axes): Axes on which to plot
        include_list (list): List of names/fits to include
        exclude_list (list): List of names/fits to exclude
        label: "shortname" or "shortname_and_seed" or "name"
    """
    from matplotlib import pyplot as plt

    if ax_f is None and ax_e is None:
        fig, (ax_f, ax_e) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            squeeze=True,
            figsize=(7, 12),
            facecolor="white",
        )
    else:
        fig = plt.gcf()

    for name, fd in fit_dict.items():
        if exclude_list is not None and name in exclude_list:
            continue

        if include_list is not None and name not in include_list:
            continue

        plot_fit(
            fd,
            k=metric + "/depa",
            ax=ax_e,
            plot_train=plot_train,
            name=fd[label],
            x_name=x_name,
        )
        plot_fit(
            fd,
            k=metric + "/f_comp",
            ax=ax_f,
            plot_train=plot_train,
            name=fd[label],
            x_name=x_name,
        )

    return fig, ax_e, ax_f


default_input_dict = {
    "cutoff": 6,
    "seed": 1,
    "data": {
        "filename": "path",
        "test_filename": "path",
        "reference_energy": 0.0,
        "save_dataset": False,
    },
    "potential": {"preset": "GRACE_1L", "scale": False, "shift": False},
    "fit": {
        "loss": {
            "energy": {"type": "square", "weight": 1.0},
            "forces": {"type": "square", "weight": 5.0},
        },
        "maxiter": "auto",
        "optimizer": "Adam",
        "opt_params": {
            "learning_rate": 0.01,
            "amsgrad": True,
            "use_ema": True,
            "ema_momentum": 0.99,
            "weight_decay": None,
            "clipvalue": 1.0,
        },
        "learning_rate_reduction": {
            "patience": 5,
            "factor": 0.98,
            "min": 0.0005,
            "stop_at_min": False,
        },
        "loss_norm_by_batch_size": True,
        "batch_size": 20,
        "test_batch_size": 200,
        "jit_compile": True,
        "train_max_n_buckets": "auto",
        "test_max_n_buckets": "auto",
        "auto_bucket_max_padding": 0.3,
        "progressbar": True,
        "train_shuffle": True,
    },
}


def select_elements_in_model(
    elements_to_select, instructions_dict, checkpoint_path, param_dtype
):
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.compute import ScalarChemicalEmbedding

    tp = TensorPotential(potential=instructions_dict, param_dtype=param_dtype)
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    tp.load_checkpoint(
        checkpoint_name=checkpoint_path,
        expect_partial=False,
        verbose=True,
        raise_errors=True,
    )

    new_element_map = {e: i for i, e in enumerate(elements_to_select)}
    # selection of ChemicalEmbedding
    Z = None
    for name, ins in instructions_dict.items():
        if isinstance(ins, ScalarChemicalEmbedding):
            if Z is None:
                Z = ins
                break
    if Z is None:
        raise ValueError("No ScalarChemicalEmbedding found in instructions dict")

    index_to_select = Z.get_index_to_select(elements_to_select)

    if len(index_to_select) != len(elements_to_select):
        available_elements_set = set(Z.element_map_symbols.numpy())
        missing_elements = set(elements_to_select) - available_elements_set
        raise ValueError(
            f"Not all elements are presented in the model. Missing elements: {missing_elements}.\n"
            f"Available elements: {available_elements_set}"
        )

    logging.info(
        f"Selecting {len(index_to_select)} elements, index_to_select={index_to_select.numpy()}"
    )

    objects_to_patch_dict = {}
    logging.info("Analyzing instructions  to patch:")
    for name, ins in instructions_dict.items():
        if isinstance(ins, ElementsReduceInstructionMixin):
            patch_dict = ins.prepare_variables_for_selected_elements(index_to_select)
            if patch_dict:
                objects_to_patch_dict[name] = patch_dict
                logging.info(f" - {name}: {len(patch_dict)} tensors to patch")

    # patching
    logging.info("Patching trainable variables:")
    for ins_name, patch_dict in objects_to_patch_dict.items():
        ins = instructions_dict[ins_name]
        for var_name, var in patch_dict.items():
            logging.info(f" - {ins.name}.{var_name}: shape={var.shape}")
            setattr(ins, var_name, var)

    logging.info("Patching model variables:")
    for name, ins in instructions_dict.items():
        if isinstance(ins, ElementsReduceInstructionMixin):
            logging.info(f" - {ins.name}")
            ins.upd_init_args_new_elements(new_element_map)

    return tp


def convert_model_reduce_elements(
    element_map: dict,
    potential_file_name: str,
    checkpoint_name: str,
    new_potential_file_name: str,
    new_checkpoint_name: str,
    param_dtype: DType = None,
):

    from tensorpotential.instructions.base import (
        load_instructions,
        save_instructions_dict,
    )
    from tensorpotential.tensorpot import TensorPotential

    param_dtype = param_dtype or float32
    assert param_dtype in [
        float32,
        float64,
    ], f"Unknown dtype {param_dtype}. Supported dtypes: [float32, float64]"

    instructions_dict = load_instructions(potential_file_name)
    if not isinstance(instructions_dict, dict):
        logging.info(
            f"Model in {potential_file_name} is NOT in new format (dict-like)."
            + " Convert it using `grace_utils update_model`"
        )
        sys.exit(0)
    tp = TensorPotential(potential=instructions_dict, param_dtype=param_dtype)
    logging.info(f"Loading checkpoint from {checkpoint_name}")
    tp.load_checkpoint(
        checkpoint_name=checkpoint_name,
        expect_partial=False,
        verbose=True,
        raise_errors=True,
    )

    elements_to_select = element_map

    if isinstance(element_map, dict):
        # ensure it is list with correct order
        inv_dict = {i: e for e, i in element_map.items()}
        elements_to_select = [inv_dict[i] for i in range(len(element_map))]

    tp = select_elements_in_model(
        elements_to_select=elements_to_select,
        instructions_dict=instructions_dict,
        checkpoint_path=checkpoint_name,
        param_dtype=param_dtype,
    )

    logging.info(f"Saving converted checkpoint to {new_checkpoint_name}")
    tp.save_checkpoint(checkpoint_name=new_checkpoint_name, verbose=True)

    logging.info(f"Saving converted model to {new_potential_file_name}")
    save_instructions_dict(
        new_potential_file_name, instructions_dict, param_dtype=param_dtype
    )


def enforce_pbc(atoms, cutoff):
    """Enforce periodic boundary conditions for a given cutoff."""
    pos = atoms.get_positions()
    if (atoms.get_pbc() == 0).all():
        max_d = np.max(np.linalg.norm(pos - pos[0], axis=1))
        cell = np.eye(3) * ((max_d + cutoff) * 2)
        atoms.set_cell(cell)
        atoms.center()
    atoms.set_pbc(True)

    return atoms


def get_param_dtype_from_config(potential_config: dict, log=None) -> tuple[DType, str]:
    param_dtype_flag = potential_config.get("param_dtype")

    if param_dtype_flag is None and "float_dtype" in potential_config:
        if log:
            log.warning(
                "The parameter 'float_dtype' is deprecated and will be removed in a future version. "
                "Please use 'param_dtype' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        param_dtype_flag = potential_config.get("float_dtype")

    if param_dtype_flag is None:
        param_dtype_flag = "float32"
        if log:
            log.info(f"Using default model parameter dtype: {param_dtype_flag}")
    else:
        if log:
            log.info(f"Model parameter dtype: {param_dtype_flag}")

    return get_dtype_by_name(param_dtype_flag), param_dtype_flag




@contextmanager
def suppress_tf_logging(level=logging.WARNING):
    """Temporarily raise the TensorFlow logger level to suppress noisy INFO messages."""
    tf_logger = logging.getLogger("tensorflow")
    prev_level = tf_logger.level
    tf_logger.setLevel(level)
    try:
        yield
    finally:
        tf_logger.setLevel(prev_level)


class Parity:
    FULL_PARITY = [
        [0, 1],
        [1, -1],
        [1, 1],
        [2, -1],
        [2, 1],
        [3, -1],
        [3, 1],
        [4, -1],
        [4, 1],
        [5, -1],
        [5, 1],
        [6, -1],
        [6, 1],
    ]

    REAL_PARITY = [
        [0, 1],
        [1, -1],
        [2, 1],
        [3, -1],
        [4, 1],
        [5, -1],
        [6, 1],
    ]

    PSEUDO_PARITY = [
        [0, -1],
        [1, 1],
        [2, -1],
        [3, 1],
        [4, -1],
        [5, 1],
        [6, -1],
    ]

    SCALAR = [[0, 1]]

    VECTOR = [[1, -1]]

    TENSOR = [[2, 1]]
