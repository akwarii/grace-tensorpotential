"""Helpers that other packages import are public, and the old private names resolve to the same objects."""

from __future__ import annotations

import importlib

import pytest

RENAMED = [
    ("tensorpotential.uq.factories", "basis_rp_spec_from_artifact"),
    ("tensorpotential.uq.artifact_builder", "default_eff_counts"),
    ("tensorpotential.uq.gmmuq", "MISSING_ELEM_INV_COV_DIAG"),
    ("tensorpotential.calculator.foundation_models", "tag_from_url"),
    ("tensorpotential.data.datasets", "log_dense_padding_report"),
]


@pytest.mark.parametrize(("module", "name"), RENAMED)
def test_the_old_private_name_is_the_public_object(module, name):
    owner = importlib.import_module(module)
    assert getattr(owner, name) is getattr(owner, f"_{name}")
