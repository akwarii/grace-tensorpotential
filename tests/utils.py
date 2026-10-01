import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from tensorpotential.cli.gracemaker import main
from tensorpotential.utils import load_metrics


def build_tf_batch(
    atoms, element_map, specs, cutoff=6.0, pad_atoms=0, pad_bonds=0, float_dtype=None
):
    """Single-structure batch as model-ready TF tensors, optionally padded.

    ``specs`` is a model's ``compute_specs``; only the keys it declares are
    converted and returned. ``pad_atoms`` / ``pad_bonds`` append that many fake
    atoms / dummy bonds, mirroring what ``PaddingManager`` does at inference time
    (given here as exact counts so callers can slice the padded tail).

    Returns ``(data, n_atoms_real, n_bonds_real)``.
    """
    import tensorflow as tf
    from tensorpotential import constants as tc
    from tensorpotential.data.databuilder import (
        GeometricalDataBuilder,
        get_number_of_real_atoms,
        get_number_of_real_neigh,
    )

    float_dtype = float_dtype or tf.float64
    db = GeometricalDataBuilder(element_map, cutoff=cutoff)
    batch = db.join_to_batch([db.extract_from_ase_atoms(atoms)])
    n_atoms_real = get_number_of_real_atoms(batch)
    n_bonds_real = get_number_of_real_neigh(batch)
    if pad_atoms or pad_bonds:
        db.pad_batch(
            batch,
            {
                tc.PAD_MAX_N_ATOMS: n_atoms_real + pad_atoms,
                tc.PAD_MAX_N_NEIGHBORS: n_bonds_real + pad_bonds,
                tc.PAD_MAX_N_STRUCTURES: 2,
            },
        )
    data = {
        k: tf.convert_to_tensor(
            v, dtype=tf.int32 if specs[k]["dtype"] == "int" else float_dtype
        )
        for k, v in batch.items()
        if k in specs
    }
    return data, n_atoms_real, n_bonds_real


def print_full(x):

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:20,.2f}".format)
    pd.set_option("display.max_colwidth", None)
    print(x)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


test_location_folder = Path(__file__).parent.resolve()


def general_integration_test(
    folder,
    train_ref_metrics,
    test_ref_metrics,
    ref_n_epochs=2,
    ref_n_init_epoch=0,
    input="input.yaml",
    many_runs=None,
    seed=42,
    rel=None,
    abs=None,
    top_folder=None,
):
    many_runs = many_runs or [[input]]
    ref_n_epochs = ref_n_epochs + len(many_runs) * ref_n_init_epoch
    prefix = test_location_folder if top_folder is None else top_folder
    path = str(prefix / folder)

    with isolated_run_dir(prefix / "data") as tmp_path, change_directory(tmp_path):
        for arg in many_runs:
            src, dst = os.path.join(path, arg[0]), os.path.join(tmp_path, arg[0])
            print(f"Copying {src} to {dst}")
            shutil.copy(src, dst)

        for inp in many_runs:
            main(inp)
        train_metrics = load_metrics(f"seed/{seed}/train_metrics.yaml")
        test_metrics = load_metrics(f"seed/{seed}/test_metrics.yaml")
        assert len(train_metrics) == len(
            test_metrics
        ), "len(train_metrics) != len(test_metrics)"
        assert len(test_metrics) == ref_n_epochs, "len(test_metrics) != ref_n_epochs"

        last_train_row = train_metrics.iloc[-1]
        print("TRAIN metrics:", last_train_row.to_dict())

        last_test_row = test_metrics.iloc[-1]
        print("TEST metrics:", last_test_row.to_dict())

        # 4. Assert on final metric values using the helper
        _compare_metrics(
            train_metrics,
            train_ref_metrics,
            label="TRAIN",
            rel=rel,
            abs=abs,
        )
        _compare_metrics(
            test_metrics,
            test_ref_metrics,
            label="TEST",
            rel=rel,
            abs=abs,
        )


@contextmanager
def isolated_run_dir(data_dir):
    """
    Context manager yielding an empty scratch directory for one training run.

    The input files of the integration tests refer to their data as
    ``../data/<file>``. The scratch directory is therefore created next to a
    ``data`` symlink to ``data_dir``, so those relative paths resolve while
    nothing is written into the source tree. The whole tree is removed on exit,
    also when the test fails.

    Parameters:
        data_dir (Path): The folder holding the datasets (``tests/data``).
    """
    with tempfile.TemporaryDirectory(prefix="grace_test_") as root:
        Path(root, "data").symlink_to(
            Path(data_dir).resolve(), target_is_directory=True
        )
        run_dir = Path(root, "run")
        run_dir.mkdir()
        yield run_dir


@contextmanager
def change_directory(new_path):
    """
    Context manager to change the current working directory to a specified path,
    and then change it back to the original directory upon exiting the context.
    """
    # Store the current working directory
    original_path = os.getcwd()

    try:
        # Change to the specified directory
        os.chdir(new_path)
        yield  # Allow code inside the 'with' block to run

    finally:
        # Change back to the original directory
        os.chdir(original_path)


def _compare_metrics(
    actual_metrics: pd.DataFrame,
    reference_metrics: dict,
    label: str,
    rel=None,
    abs=None,
):
    """
    Compares the last row of a metrics DataFrame with reference values.

    Args:
        actual_metrics: DataFrame containing metrics from the run.
        reference_metrics: Dictionary of expected metric values.
        label: A string label (e.g., "TRAIN") for error messages.
    """
    assert not actual_metrics.empty, f"{label} metrics DataFrame is empty."
    last_row = actual_metrics.iloc[-1]
    print(f"Comparing final {label} metrics:", last_row.to_dict())

    for key, expected_value in reference_metrics.items():
        # Skip non-deterministic values like time
        if "time" in key.lower():
            continue

        actual_value = last_row.get(key)
        assert actual_value is not None, f"Metric '{key}' not found in {label} results."

        # Using pytest.approx is idiomatic for floating-point comparisons
        assert actual_value == pytest.approx(
            expected_value, rel=rel, abs=abs
        ), f"{label} metric '{key}' mismatch: Got {actual_value}, expected {expected_value}"
