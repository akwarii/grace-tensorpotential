"""Characterization tests for ``extract_const_shift_scale`` (``tensorpotential/export.py``).

Logic layer: the three return shapes (no shift instruction; plain Python numbers before
``build``; TensorFlow constants after it), the flattening of the per-element shift map,
and the missing-key case. Value layer: what goes in comes out, in element-index order,
and survives the real ``export_to_yaml`` round trip as ``scale``, ``shift`` and ``E0``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import yaml

from tensorpotential.export import export_to_yaml, extract_const_shift_scale
from tensorpotential.instructions.output import (
    ConstantScaleShiftTarget,
    CreateOutputTarget,
)
from tensorpotential.potentials.presets import FS
from tests.tolerances import FLOAT64_ARITHMETIC as EXACT

NAME = "ConstantScaleShiftTarget"
ELEMENTS = {"Al": 0, "Li": 1, "H": 2}


def shift_target(**kwargs) -> ConstantScaleShiftTarget:
    return ConstantScaleShiftTarget(
        target=CreateOutputTarget(name="energy"), name=NAME, **kwargs
    )


def built(**kwargs) -> ConstantScaleShiftTarget:
    ins = shift_target(**kwargs)
    ins.build(tf.float64)
    return ins


def close(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=EXACT.rtol, atol=EXACT.atol)


# ----------------------------------------------------------------------------- logic


def test_without_the_instruction_the_result_is_none():
    assert extract_const_shift_scale({}) is None
    assert extract_const_shift_scale({"other": shift_target(scale=2.0)}) is None


def test_before_build_python_numbers_pass_through():
    ins = shift_target(scale=2.5, shift=-1.25)
    scale, shift, shift_map = extract_const_shift_scale({NAME: ins})
    assert (scale, shift, shift_map) == (2.5, -1.25, None)


def test_after_build_scale_and_shift_are_numpy_values():
    scale, shift, shift_map = extract_const_shift_scale({
        NAME: built(scale=2.5, shift=-1.25)
    })
    assert not isinstance(
        scale, tf.Tensor
    )  # ``.numpy()`` of a scalar is a numpy scalar
    assert not isinstance(shift, tf.Tensor)
    assert scale.dtype == shift.dtype == np.float64
    assert scale.shape == shift.shape == ()
    assert shift_map is None


def test_zero_shift_stays_a_python_number_after_build():
    # build() only makes a tensor of a non-zero constant shift; zero stays a float
    _, shift, _ = extract_const_shift_scale({NAME: built(scale=1.0, shift=0.0)})
    assert shift == 0.0
    assert isinstance(shift, float)


def test_shift_map_is_flattened_to_one_dimension_after_build():
    ins = built(atomic_shift_map={0: 1.0, 1: 2.0, 2: 3.0})
    assert tuple(ins.atomic_shift_map.shape) == (3, 1)  # build() gives a column
    _, _, shift_map = extract_const_shift_scale({NAME: ins})
    assert shift_map.shape == (3,)


def test_shift_map_before_build_is_returned_as_the_stored_array():
    ins = shift_target(atomic_shift_map={0: 1.0, 1: 2.0})
    _, _, shift_map = extract_const_shift_scale({NAME: ins})
    assert shift_map is ins.atomic_shift_map
    assert isinstance(shift_map, np.ndarray)


def test_extra_instructions_in_the_dict_are_ignored():
    ins = built(scale=3.0)
    out = extract_const_shift_scale({"energy": CreateOutputTarget("e"), NAME: ins})
    close(out[0], 3.0)


# ----------------------------------------------------------------------------- values


def test_shift_map_follows_element_index_order_not_insertion_order():
    # ConstantScaleShiftTarget sorts the dict by key: element 0 first, whatever the order given
    ins = built(atomic_shift_map={2: 7.5, 0: -2.0, 1: 3.25})
    _, _, shift_map = extract_const_shift_scale({NAME: ins})
    close(shift_map, [-2.0, 3.25, 7.5])


@pytest.mark.parametrize(("scale", "shift"), [(1.0, 0.0), (2.5, -1.25), (0.125, 8.0)])
def test_values_survive_the_build(scale, shift):
    out_scale, out_shift, _ = extract_const_shift_scale({
        NAME: built(scale=scale, shift=shift)
    })
    close(out_scale, scale)
    close(out_shift, shift)


def exported_yaml(tmp_path, **kwargs) -> dict:
    fs_ins = FS(
        element_map=ELEMENTS,
        lmax=2,
        n_rad_base=4,
        n_rad_max=12,
        embedding_size=8,
        max_order=3,
        **kwargs,
    ).get_instructions()
    for ins in fs_ins.values():
        ins.build(tf.float64)
    out = tmp_path / "pot.yaml"
    export_to_yaml(fs_ins, str(out))
    return yaml.safe_load(out.read_text())


def test_constant_scale_and_shift_reach_the_exported_yaml(tmp_path):
    pot = exported_yaml(tmp_path, constant_out_scale=2.5, constant_out_shift=-1.25)
    close(pot["scale"], 2.5)
    close(pot["shift"], -1.25)
    assert "E0" not in pot


def test_atomic_shift_map_reaches_the_exported_yaml_as_E0(tmp_path):  # noqa: N802
    pot = exported_yaml(
        tmp_path,
        constant_out_scale=0.5,
        atomic_shift_map={0: -2.0, 1: 3.25, 2: 7.5},  # element index -> E0
    )
    close(pot["E0"], [-2.0, 3.25, 7.5])
    close(pot["scale"], 0.5)
