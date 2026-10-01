from pathlib import Path

import pytest
import pandas as pd

from tensorpotential.data.dataset_plotter import (
    DatasetHistPlotter,
    DEFAULT_UNIT_TRANSFORM,
    UnitTransformTuple,
)

DATA_DIR = Path(__file__).parent.resolve() / "data"


@pytest.fixture
def load_datasets():
    train_path = DATA_DIR / "MoNbTaW_train50.pkl.gz"
    test_path = DATA_DIR / "MoNbTaW_test50.pkl.gz"

    ds_dict = {"train": pd.read_pickle(train_path), "test": pd.read_pickle(test_path)}
    return ds_dict


@pytest.fixture
def output_dir(tmp_path):
    out_dir = tmp_path / "hists"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_plot_creation(load_datasets, output_dir):
    # 'energy_corrected_per_atom is not present in the list so 'energy' is plotted instead
    targets = ("energy_per_atom", "forces", "stress")
    units_transform = {
        "energy": UnitTransformTuple(units="eV/A"),
        **DEFAULT_UNIT_TRANSFORM,
    }

    DatasetHistPlotter.plot(
        datasets=load_datasets,
        output_dir=output_dir,
        plot_targets=targets,
        units_transform=units_transform,
    )
    print("output_dir=", output_dir)
    for target in targets:
        plot_file = output_dir / f"{target}.png"
        assert plot_file.exists(), f"Plot file {plot_file} was not created"
