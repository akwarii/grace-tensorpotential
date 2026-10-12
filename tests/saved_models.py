"""Seeded models saved as ``model.yaml`` plus checkpoint, for the tests of the model commands.

A model saved from a path (``TensorPotential(potential="model.yaml")``) tracks no weights, so its checkpoint holds
only the counters; the helpers here build the model from the loaded instructions and give it seeded weights, so
that the energy of a dimer is a reproducible, non-trivial number.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import yaml
from ase import Atoms

from tensorpotential import TensorPotential
from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential.instructions.base import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tests.seeded_weights import seed_trainable_variables

DEFAULT_CONFIG = {"rcut": 4.5, "n_rad_max": 4, "lmax": 2}
FS_CONFIG = {
    "lmax": 2,
    "n_rad_base": 4,
    "n_rad_max": 12,
    "embedding_size": 8,
    "max_order": 3,
}


def save_seeded_model(
    directory: Path,
    element_map: dict,
    *,
    as_list: bool = False,
    preset: str = "GRACE_1LAYER_latest",
    config: dict | None = None,
) -> tuple[Path, Path]:
    """Save a float32 model with seeded weights and its checkpoint into ``directory``.

    Parameters
    ----------
    directory : Path
        Existing directory; ``model.yaml`` and ``checkpoint.*`` are written there.
    element_map : dict
        Element symbol to index.
    as_list : bool
        Write the oldest format (a bare list of instructions) and the checkpoint of a list model.
    preset : str
        Name of the preset.
    config : dict, optional
        Arguments of the preset (``DEFAULT_CONFIG`` when empty).

    Returns
    -------
    tuple[Path, Path]
        The model file and the checkpoint prefix.
    """
    config = config or DEFAULT_CONFIG
    instructions = get_preset(preset)(
        element_map=element_map, **config
    ).get_instructions()
    model = directory / "model.yaml"
    save_instructions_dict(str(model), instructions, param_dtype=tf.float32)
    if as_list:
        raw = yaml.safe_load(model.read_text())
        model.write_text(yaml.dump(list(raw["instructions"].values()), sort_keys=False))
    tp = TensorPotential(
        potential=load_instructions(str(model)), param_dtype=tf.float32
    )
    seed_trainable_variables(tp.model)
    checkpoint = directory / "checkpoint"
    tp.save_checkpoint(checkpoint_name=str(checkpoint))
    return model, checkpoint


def dimer_energy(
    model: Path, checkpoint: Path, element: str, dtype=tf.float32
) -> float:
    """Energy in eV of a two-atom cluster of ``element`` 2 Angstrom apart, from ``model`` and ``checkpoint``."""
    tp = TensorPotential(potential=load_instructions(str(model)), param_dtype=dtype)
    tp.load_checkpoint(
        checkpoint_name=str(checkpoint),
        expect_partial=True,
        assert_consumed=False,
        assert_existing_objects_matched=False,
    )
    tp.model.decorate_compute_function()
    atoms = Atoms(positions=[[0, 0, 0], [2, 0, 0]], symbols=[element] * 2, pbc=False)
    atoms.calc = TPCalculator(model=tp.model)
    return float(atoms.get_potential_energy())
