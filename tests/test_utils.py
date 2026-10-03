"""Tests of ``utils.py``: ``get_param_dtype_from_config`` and ``enforce_pbc``.

``get_param_dtype_from_config`` is the ``param_dtype`` default of a training config. This is one of the code defaults of ``param_dtype`` that disagree (see ``test_metadata_utils.py``): a config
without ``param_dtype`` gives float32, a yaml without metadata gives float64. Pinned, not endorsed.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

import numpy as np
import pytest
import tensorflow as tf
from ase import Atoms
from ase.build import fcc111

from tensorpotential.utils import enforce_pbc, get_param_dtype_from_config
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH


class RecordingLog:
    """The two methods of a logger that ``get_param_dtype_from_config`` calls, remembering the arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def info(self, *args, **kwargs):
        self.calls.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args, kwargs))


@pytest.mark.parametrize(
    ("name", "dtype"), [("float32", tf.float32), ("float64", tf.float64)]
)
def test_param_dtype_key_is_read(name, dtype):
    assert get_param_dtype_from_config({"param_dtype": name}) == (dtype, name)


def test_default_without_a_key_is_float32():
    # PINNED, not endorsed: float32 here; metadata_utils.resolve_param_dtype gives float64 for a yaml without the key
    assert get_param_dtype_from_config({}) == (tf.float32, "float32")


def test_unknown_name_is_a_key_error():
    with pytest.raises(KeyError, match="Unknown dtype name float16"):
        get_param_dtype_from_config({"param_dtype": "float16"})


def test_deprecated_float_dtype_is_used_when_param_dtype_is_absent():
    assert get_param_dtype_from_config({"float_dtype": "float64"}) == (
        tf.float64,
        "float64",
    )


def test_param_dtype_wins_over_the_deprecated_float_dtype():
    config = {"param_dtype": "float32", "float_dtype": "float64"}

    assert get_param_dtype_from_config(config) == (tf.float32, "float32")


def test_log_reports_the_chosen_dtype():
    log = RecordingLog()

    get_param_dtype_from_config({"param_dtype": "float64"}, log=log)

    assert log.calls == [("info", ("Model parameter dtype: float64",), {})]


def test_log_reports_the_default():
    log = RecordingLog()

    get_param_dtype_from_config({}, log=log)

    assert log.calls == [
        ("info", ("Using default model parameter dtype: float32",), {})
    ]


def test_deprecated_key_warns_with_the_stacklevel_and_class_as_extra_arguments():
    log = RecordingLog()

    get_param_dtype_from_config({"float_dtype": "float64"}, log=log)

    kind, args, kwargs = log.calls[0]
    assert kind == "warning"
    assert "'float_dtype' is deprecated" in args[0]
    assert args[1] is DeprecationWarning
    assert kwargs == {"stacklevel": 2}
    assert log.calls[1] == ("info", ("Model parameter dtype: float64",), {})


def test_deprecated_key_warning_cannot_be_formatted_by_a_real_logger():
    # PINNED, not endorsed (TEST8 finding): logging.Logger.warning(msg, DeprecationWarning, stacklevel=2) passes the
    # class as a %-argument of a message without placeholders, so formatting the record raises TypeError (logging
    # prints a "Logging error" to stderr instead of failing) and the text of the warning is lost.
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("test_utils.param_dtype")
    logger.setLevel(logging.DEBUG)
    logger.propagate = (
        False  # pytest's own handler formats records and re-raises the error
    )
    logger.addHandler(Collect())
    try:
        result = get_param_dtype_from_config({"float_dtype": "float64"}, log=logger)
    finally:
        logger.handlers.clear()

    assert result == (tf.float64, "float64")
    warning = next(r for r in records if r.levelno == logging.WARNING)
    with pytest.raises(TypeError, match="not all arguments converted"):
        warning.getMessage()


# ---------------------------------------------------------------------------------------------
# enforce_pbc. ``GeometricalDataBuilder.extract_from_ase_atoms`` calls it with the largest cutoff
# before it builds pairs. Every test below **pins current behaviour** and is not an endorsement:
# where a behaviour looks like a hazard (the caller's ``Atoms`` is edited in place, partial periodic
# boundaries are overridden, a zero cell vector is made periodic) the docstring says so, and a later
# fix is meant to change the test on purpose.
# ---------------------------------------------------------------------------------------------

CUTOFF = 4.0


def dimer(**kwargs) -> Atoms:
    return Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]], **kwargs)


def test_aperiodic_input_gets_a_cube_of_twice_the_extent_plus_cutoff() -> None:
    """Pins current behaviour: the edge is ``2 * (max |pos - pos[0]| + cutoff)``, here 2 * (2.5 + 4)."""
    atoms = enforce_pbc(dimer(), CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 13.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.get_volume() == pytest.approx(13.0**3, rel=ARITH.rtol)


def test_aperiodic_input_is_centred_in_the_new_cell() -> None:
    atoms = enforce_pbc(dimer(), CUTOFF)
    np.testing.assert_allclose(
        atoms.positions,
        [[5.25, 6.5, 6.5], [7.75, 6.5, 6.5]],
        rtol=ARITH.rtol,
        atol=ARITH.atol,
    )


def test_a_single_atom_gets_a_cell_of_twice_the_cutoff() -> None:
    atoms = enforce_pbc(Atoms("Cu", positions=[[1.0, 2.0, 3.0]]), CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 8.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )
    np.testing.assert_allclose(
        atoms.positions, [[4.0, 4.0, 4.0]], rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_the_extent_is_measured_from_the_first_atom_only() -> None:
    """Pins current behaviour: the same three collinear atoms give different cells depending on
    which one is listed first (extent 4 from an end atom, 2 from the middle atom)."""
    xs = [[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]]
    from_end = enforce_pbc(Atoms("Cu3", positions=xs), CUTOFF)
    from_middle = enforce_pbc(Atoms("Cu3", positions=[xs[1], xs[0], xs[2]]), CUTOFF)
    assert from_end.cell[0, 0] == pytest.approx(2 * (4.0 + CUTOFF), rel=ARITH.rtol)
    assert from_middle.cell[0, 0] == pytest.approx(2 * (2.0 + CUTOFF), rel=ARITH.rtol)


def test_new_cell_keeps_periodic_images_of_a_cluster_beyond_the_cutoff() -> None:
    """The cube is large enough that the closest image of any atom is farther than the cutoff:
    the box edge minus the cluster extent is at least ``2 * cutoff``."""
    rng = np.random.default_rng(4)
    positions = rng.normal(scale=2.0, size=(12, 3))
    atoms = enforce_pbc(Atoms("Cu12", positions=positions), CUTOFF)
    extent = np.ptp(positions, axis=0).max()
    assert atoms.cell[0, 0] - extent >= 2 * CUTOFF


def test_the_callers_atoms_is_edited_in_place_and_returned() -> None:
    """Pins current behaviour, a hazard for stress (the volume of an invented cell): the object
    passed in is changed (cell, pbc and positions), and the function returns that same object."""
    original = dimer()
    returned = enforce_pbc(original, CUTOFF)
    assert returned is original
    assert original.pbc.tolist() == [True, True, True]
    assert original.cell.array.any()
    assert original.positions[0, 1] == pytest.approx(6.5, rel=ARITH.rtol)


def test_a_copy_protects_the_caller() -> None:
    original = dimer()
    enforce_pbc(original.copy(), CUTOFF)
    assert original.pbc.tolist() == [False, False, False]
    assert not original.cell.array.any()
    assert original.positions[1, 0] == 2.5


def test_an_existing_cell_of_an_aperiodic_input_is_overwritten() -> None:
    """Pins current behaviour: a cell the caller set is replaced when ``pbc`` is all False."""
    atoms = dimer(cell=50.0 * np.eye(3), pbc=False)
    enforce_pbc(atoms, CUTOFF)
    np.testing.assert_allclose(
        atoms.cell.array, 13.0 * np.eye(3), rtol=ARITH.rtol, atol=ARITH.atol
    )


def test_a_fully_periodic_input_is_left_unchanged() -> None:
    cell = np.array([[5.0, 0, 0], [1.0, 6.0, 0], [0.0, 0.5, 7.0]])
    atoms = dimer(cell=cell, pbc=True)
    before = atoms.positions.copy()
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)
    np.testing.assert_array_equal(atoms.positions, before)


def test_partial_periodic_boundaries_are_overridden_without_touching_cell_or_positions() -> (
    None
):
    """Pins current behaviour, not an endorsement: a slab flagged ``pbc = (T, T, F)`` becomes
    periodic along z as well, with the cell and positions as they were."""
    slab = fcc111("Cu", size=(2, 2, 3), a=3.61, vacuum=1.0)
    assert slab.pbc.tolist() == [True, True, False]
    cell, positions = slab.cell.array.copy(), slab.positions.copy()
    enforce_pbc(slab, CUTOFF)
    assert slab.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(slab.cell.array, cell)
    np.testing.assert_array_equal(slab.positions, positions)


@pytest.mark.parametrize(
    "pbc", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_any_single_periodic_direction_prevents_a_new_cell(pbc) -> None:
    """Pins current behaviour: only an all-False ``pbc`` triggers the new cell."""
    cell = 20.0 * np.eye(3)
    atoms = dimer(cell=cell, pbc=pbc)
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)


def test_a_zero_third_cell_vector_is_made_periodic_and_stays_zero() -> None:
    """Pins current behaviour, not an endorsement: the ASE convention for a 2D system (a zero
    third vector with ``pbc = (T, T, F)``) ends with ``pbc`` True along that zero vector.
    The pair search then fails (see ``test_databuilder.py``)."""
    cell = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 0.0]])
    atoms = dimer(cell=cell, pbc=[True, True, False])
    enforce_pbc(atoms, CUTOFF)
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(atoms.cell.array, cell)


def test_an_aperiodic_input_with_a_zero_cell_gets_a_valid_cell() -> None:
    """Pins current behaviour: with ``pbc`` all False a zero cell is not a problem, it is replaced."""
    atoms = dimer(cell=np.zeros((3, 3)), pbc=False)
    enforce_pbc(atoms, CUTOFF)
    assert np.linalg.det(atoms.cell.array) > 0


def test_the_cutoff_enters_the_cell_linearly() -> None:
    small = enforce_pbc(dimer(), 3.0).cell[0, 0]
    large = enforce_pbc(dimer(), 5.0).cell[0, 0]
    assert large - small == pytest.approx(2 * (5.0 - 3.0), rel=ARITH.rtol)
