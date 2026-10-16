"""``segment_sum`` against ``numpy.add.at`` and against the forces of the TF goldens.

Logic: the sum, shapes and dtypes, empty segments, no bonds and no atoms, both index dtypes, the gradient, the errors.
Physical values: the forces of every tiny anchor rebuilt from TF's pair forces, as the TF model does, against TF's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import DtypeError, ShapeError
from tensorpotential.torch_backend.ops.scatter import segment_sum
from tests_torch.golden import TINY_F64, cases, load
from tests_torch.golden.tolerances import (
    FEATURE_F32,
    FEATURE_F64,
    FORCES_F64,
    assert_close,
)

RNG_SEED = 20261016


def _numpy_segment_sum(data: np.ndarray, ids: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, *data.shape[1:]), dtype=data.dtype)
    np.add.at(out, ids, data)
    return out


@pytest.mark.parametrize(
    ("dtype", "row"), [(torch.float64, FEATURE_F64), (torch.float32, FEATURE_F32)]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_segment_sum_equals_numpy_add_at(dtype, row, index_dtype):
    rng = np.random.default_rng(RNG_SEED)
    data = rng.normal(size=(200, 3, 5))
    ids = rng.integers(0, 40, size=200)
    ids[ids == 7] = 8  # segment 7 is named by no row
    out = segment_sum(
        torch.as_tensor(data, dtype=dtype), torch.as_tensor(ids, dtype=index_dtype), 41
    )
    assert out.dtype == dtype
    assert out.shape == (41, 3, 5)
    expected = _numpy_segment_sum(
        data.astype(np.float32 if dtype == torch.float32 else np.float64), ids, 41
    )
    assert_close(out, expected, row)
    assert torch.all(out[7] == 0) and torch.all(out[40] == 0)


def test_segment_sum_of_one_axis_data():
    out = segment_sum(
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64), torch.tensor([2, 0, 2]), 3
    )
    assert out.tolist() == [2.0, 0.0, 5.0]


def test_no_bonds_give_zeros_for_every_atom():
    out = segment_sum(
        torch.zeros((0, 3), dtype=torch.float64), torch.zeros(0, dtype=torch.int64), 4
    )
    assert out.shape == (4, 3) and torch.all(out == 0)


def test_no_atoms_give_an_empty_result():
    out = segment_sum(
        torch.zeros((0, 3), dtype=torch.float64), torch.zeros(0, dtype=torch.int64), 0
    )
    assert out.shape == (0, 3) and out.dtype == torch.float64


def test_the_gradient_is_a_gather_and_passes_gradcheck():
    data = (
        torch
        .linspace(-1.0, 1.0, 12, dtype=torch.float64)
        .reshape(6, 2)
        .requires_grad_()
    )
    ids = torch.tensor([0, 2, 2, 1, 0, 2])
    weights = torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]], dtype=torch.float64)
    (segment_sum(data, ids, 3) * weights).sum().backward()
    assert data.grad is not None
    assert torch.equal(data.grad, weights[ids])
    assert torch.autograd.gradcheck(lambda d: segment_sum(d, ids, 3), (data,))


@pytest.mark.parametrize("index_dtype", [torch.float64, torch.int16, torch.bool])
def test_an_index_of_another_dtype_is_refused(index_dtype):
    with pytest.raises(DtypeError, match="int32 or int64"):
        segment_sum(
            torch.ones((2, 3), dtype=torch.float64),
            torch.zeros(2, dtype=index_dtype),
            1,
        )


@pytest.mark.parametrize(
    ("data_shape", "ids_shape"), [((3, 2), (2,)), ((3, 2), (3, 1)), ((), (1,))]
)
def test_shapes_that_do_not_fit_are_refused(data_shape, ids_shape):
    with pytest.raises(ShapeError, match="one index per row"):
        segment_sum(
            torch.ones(data_shape, dtype=torch.float64),
            torch.zeros(ids_shape, dtype=torch.int64),
            2,
        )


def test_a_negative_number_of_segments_is_refused():
    with pytest.raises(ShapeError, match="at least 0"):
        segment_sum(
            torch.ones((1, 3), dtype=torch.float64),
            torch.zeros(1, dtype=torch.int64),
            -1,
        )


def test_an_index_out_of_range_raises_the_error_of_torch_on_the_cpu():
    with pytest.raises((IndexError, RuntimeError)):
        segment_sum(torch.ones((2, 3), dtype=torch.float64), torch.tensor([0, 2]), 2)


@pytest.mark.golden
@pytest.mark.parametrize("name", TINY_F64)
def test_forces_from_the_tf_pair_forces_match_tf(name):
    arrays = load(name)
    for case in cases(arrays):
        pair_f = torch.as_tensor(arrays[f"{case}/res/pair_f"], dtype=torch.float64)
        ind_i = torch.as_tensor(arrays[f"{case}/in/ind_i"])
        ind_j = torch.as_tensor(arrays[f"{case}/in/ind_j"])
        n_atoms = arrays[f"{case}/res/forces"].shape[0]
        forces = segment_sum(pair_f, ind_j, n_atoms) - segment_sum(
            pair_f, ind_i, n_atoms
        )
        assert_close(forces, arrays[f"{case}/res/forces"], FORCES_F64, f"{name} {case}")
