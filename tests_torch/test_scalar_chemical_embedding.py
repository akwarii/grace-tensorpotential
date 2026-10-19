"""The ``ScalarChemicalEmbedding`` twin: the weight table, its element map and its errors (logic only).

The instruction returns a weight, so there is no physical value to check beyond the identity with the stored table
(``test_spbf_scalar_ind_golden.py``).
"""

from __future__ import annotations

import pytest
import torch

from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.instructions.scalar_chemical_embedding import (
    ScalarChemicalEmbedding,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.spec.model import InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.init_only import assert_option_is_ignored, init_only_values

CLS = COMPUTE + "ScalarChemicalEmbedding"
ELEMENTS = {"Cu": 0, "H": 1, "Mg": 2, "O": 3}


def _twin(element_map=ELEMENTS, embedding_size=6, dtype=torch.float64, **given):
    options = {
        **DEFAULTS[CLS],
        "element_map": element_map,
        "embedding_size": embedding_size,
        **given,
    }
    spec = InstructionSpec(name="Z", cls=CLS, options=options, depends_on=(), index=0)
    return ScalarChemicalEmbedding(spec, DtypePolicy(dtype))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_output_is_the_weight_table_in_the_parameter_dtype(dtype):
    twin = _twin(dtype=dtype)
    assert list(twin.state_dict()) == ["w"]
    table = torch.arange(24, dtype=dtype).reshape(4, 6)
    twin.load_state_dict({"w": table}, strict=True)
    out = twin({})
    assert out.dtype == dtype and out.shape == (4, 6)
    torch.testing.assert_close(out, table, rtol=0, atol=0)
    out.sum().backward()
    assert twin.w.grad is not None and bool((twin.w.grad == 1).all())


def test_the_table_starts_at_zero_and_keeps_the_element_map_order():
    twin = _twin({"O": 1, "H": 0})
    assert bool((twin.w == 0).all()) and twin.w.shape == (2, 6)
    assert list(twin.element_map.items()) == [("O", 1), ("H", 0)]
    assert twin.embedding_size == 6


@pytest.mark.parametrize(
    "element_map", [{"H": 1, "O": 2}, {"H": 0, "O": 0}, {"H": 0, "O": 2}]
)
def test_an_element_map_whose_values_are_not_the_rows_is_refused(element_map):
    with pytest.raises(
        OptionValueError, match=r"element_map values must be the rows 0 \.\. 1"
    ):
        _twin(element_map)


def test_an_embedding_size_below_one_is_refused():
    with pytest.raises(
        OptionValueError, match="embedding_size must be at least 1, got 0"
    ):
        _twin(embedding_size=0)


def test_init_changes_neither_the_state_nor_the_output():
    assert_option_is_ignored(_twin, CLS, "init", {})


@pytest.mark.parametrize("option", ["embedding_size", "is_trainable"])
def test_the_check_refuses_an_option_that_is_not_initialisation_only(option):
    with pytest.raises(ValueError, match="is not an initialisation-only option"):
        init_only_values(CLS, option)
