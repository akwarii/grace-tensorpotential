import json
import os
import shutil
import socket

import subprocess
from pathlib import Path

from .utils import isolated_run_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


prefix = Path(__file__).parent.resolve()


def _free_port():
    """A port the OS has just handed out, so parallel runs never share one."""
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


TF_DATASET = "tf_dataset"
DATA_DISTRIB = "data_distrib"


def test_compute_distributed_data_and_distrib_fit():
    # The scripts read ../data/<file> and write tf_dataset/ and seed/ next to
    # them, so they run on a copy of the folder in a scratch directory.
    with isolated_run_dir(prefix / "data") as work:
        shutil.copytree(
            prefix / DATA_DISTRIB,
            work,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(TF_DATASET, "seed", "__pycache__"),
        )
        _compute_distributed_data_and_distrib_fit(work)


def _compute_distributed_data_and_distrib_fit(work):
    tf_dataset_path = work / TF_DATASET
    if os.path.isdir(tf_dataset_path):
        shutil.rmtree(tf_dataset_path)

    tf_dataset_stats_json_path = (
        work / TF_DATASET / "stage3" / "stats.json"
    )

    assert not os.path.isfile(tf_dataset_stats_json_path)
    script_name = "compute_distributed_data.sh"
    subprocess.run(["bash", script_name], cwd=str(work), check=True)
    assert os.path.isfile(tf_dataset_stats_json_path)

    with open(tf_dataset_stats_json_path, "r") as f:
        stats = json.load(f)

    assert "scale" in stats
    assert "avg_n_neigh" in stats
    assert "total_num_of_neighs" in stats
    assert "total_num_of_atoms" in stats
    assert "total_num_structures" in stats
    assert "total_num_of_batches" in stats
    assert "cutoff" in stats
    assert "cutoff_dict" in stats
    assert "batch_size" in stats
    assert stats["total_num_of_atoms"] == 844
    assert stats["total_num_of_neighs"] == 48904
    assert stats["total_num_structures"] == 50
    assert stats["total_num_of_batches"] == 14

    seed_path = work / "seed"
    if os.path.isdir(seed_path):
        shutil.rmtree(seed_path)
    assert not os.path.isdir(seed_path)

    current_env = os.environ.copy()
    current_env["NUM_VIRTUAL_DEVICES"] = "2"
    current_env["CUDA_VISIBLE_DEVICES"] = "-1"
    current_env["TF_USE_LEGACY_KERAS"] = "1"

    subprocess.run(
        "gracemaker -m",
        cwd=str(work),
        check=True,
        shell=True,
        env=current_env,
    )

    test_metrics_path = work / "seed" / "1" / "test_metrics.yaml"
    assert os.path.isfile(test_metrics_path)

    # Regression (scripts/gracemaker.py): an externally-supplied TF_CONFIG -- as
    # set by a SLURM launcher or left over in the shell -- must NOT route the
    # NUM_VIRTUAL_DEVICES debug path through MultiWorkerMirroredStrategy. MWMS
    # initialises the TF context, after which the virtual-device setup crashes
    # with "Virtual devices cannot be modified after being initialized". With
    # NUM_VIRTUAL_DEVICES set, gracemaker must fall back to MirroredStrategy and
    # ignore TF_CONFIG. Re-run the fit with TF_CONFIG present and confirm it
    # still completes (reusing the tf_dataset already built above).
    shutil.rmtree(seed_path)
    assert not os.path.isdir(seed_path)

    tf_config_env = current_env.copy()
    tf_config_env["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": [f"localhost:{_free_port()}"]},
            "task": {"type": "worker", "index": 0},
        }
    )
    subprocess.run(
        "gracemaker -m",
        cwd=str(work),
        check=True,
        shell=True,
        env=tf_config_env,
    )
    assert os.path.isfile(test_metrics_path)
