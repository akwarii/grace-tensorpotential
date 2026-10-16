"""The test boundary of the feature layout: TF's two layouts permuted to the backend's ``[atoms, lm, n]``.

Logic: the axes on a hand array, the refusal of a tensor that is not a feature. Physical values: the default and the
``lm_first`` goldens of the same weights give the same ``[atoms, lm, n]`` features, with ``(lmax + 1)**2`` entries
on the angular axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorpotential.torch_backend.ops.precision import FEATURE_LAYOUT
from tests_torch.golden import cases, load
from tests_torch.golden.tolerances import FEATURE_F64, assert_close
from tests_torch.layout import to_internal

ATOMS, N, LM = 2, 3, 4


def test_the_backend_layout_is_atoms_lm_n():
    assert FEATURE_LAYOUT == ("atoms", "lm", "n")


@pytest.mark.parametrize(
    ("layout", "shape"), [("default", (ATOMS, N, LM)), ("lm_first", (LM, ATOMS, N))]
)
def test_both_tf_layouts_become_atoms_lm_n(layout, shape):
    atoms, n, lm = np.meshgrid(
        np.arange(ATOMS), np.arange(N), np.arange(LM), indexing="ij"
    )
    # every element names its own atom, angular index and channel
    code = 100 * atoms + 10 * lm + n
    order = {"default": (0, 1, 2), "lm_first": (2, 0, 1)}[layout]
    tf_feature = np.transpose(code, order)
    assert tf_feature.shape == shape
    internal = to_internal(tf_feature, layout)
    assert internal.shape == (ATOMS, LM, N)
    a, m, c = np.meshgrid(np.arange(ATOMS), np.arange(LM), np.arange(N), indexing="ij")
    assert np.array_equal(internal, 100 * a + 10 * m + c)


def test_a_tensor_that_is_not_a_feature_is_refused():
    with pytest.raises(ValueError, match="three axes"):
        to_internal(np.zeros((3, 4)), "default")


@pytest.mark.golden
@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_lm_first_golden_is_the_default_golden_in_the_backend_layout(model):
    default, lm_first = load(f"{model}_tiny_f64"), load(f"{model}_tiny_f64_lm_first")
    for case in cases(default):
        for name in ("A", "AA", "YI", "I"):
            key = f"{case}/ins/{name}"
            ours = to_internal(default[key], "default")
            assert_close(to_internal(lm_first[key], "lm_first"), ours, FEATURE_F64, key)
            assert ours.shape[0] == default[f"{case}/res/forces"].shape[0]
        assert to_internal(default[f"{case}/ins/A"], "default").shape[1] == 25  # lmax 4
