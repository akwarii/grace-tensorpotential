"""The exception hierarchy of the torch backend: one base, typed families, the builtin types callers expect."""

from __future__ import annotations

import pytest

from tensorpotential.errors import TensorPotentialError
from tensorpotential.torch_backend import errors

FAMILIES = (
    errors.SpecError,
    errors.MalformedModelError,
    errors.InstructionReferenceError,
    errors.DanglingReferenceError,
    errors.CyclicReferenceError,
    errors.ForwardReferenceError,
    errors.UnsupportedModelError,
    errors.WeightMismatchError,
    errors.NeighbourListError,
    errors.ShapeError,
)


def test_the_base_is_a_library_error_and_nothing_more():
    assert issubclass(errors.TorchBackendError, TensorPotentialError)
    assert not issubclass(errors.TorchBackendError, ValueError)


@pytest.mark.parametrize("cls", FAMILIES, ids=lambda c: c.__name__)
def test_every_error_is_a_value_error(cls):
    assert issubclass(cls, ValueError)


def test_the_list_names_every_error_of_the_module():
    defined = {
        obj
        for obj in vars(errors).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseException)
        and obj.__module__ == errors.__name__
    }
    assert defined == {errors.TorchBackendError, errors.DtypeError, *FAMILIES}


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (errors.DanglingReferenceError, errors.InstructionReferenceError),
        (errors.CyclicReferenceError, errors.InstructionReferenceError),
        (errors.ForwardReferenceError, errors.InstructionReferenceError),
        (errors.MalformedModelError, errors.SpecError),
        (errors.InstructionReferenceError, errors.SpecError),
        (errors.UnsupportedModelError, errors.SpecError),
    ],
)
def test_the_spec_errors_keep_their_families(child, parent):
    assert issubclass(child, parent)


@pytest.mark.parametrize(
    "cls", [errors.WeightMismatchError, errors.NeighbourListError, errors.ShapeError]
)
def test_the_weight_neighbour_and_shape_errors_are_not_spec_errors(cls):
    assert not issubclass(cls, errors.SpecError)


def test_a_dtype_error_is_a_type_error_and_no_value_error():
    assert issubclass(errors.DtypeError, errors.TorchBackendError)
    assert issubclass(errors.DtypeError, TypeError)
    assert not issubclass(errors.DtypeError, ValueError)
