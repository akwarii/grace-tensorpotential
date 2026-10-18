"""The ``FunctionReduceN`` twin: logic against a numpy loop of the sheet's formula, and rotation equivariance.

Logic: the forward pass of the spec sheet (section 5) written as a loop over atoms in numpy, from the plan's tables
and the weights, for every option the twin ports (per-element weights, ``out_norm``, ``normalize``, ``scale``,
``init_target_value``, ``only_invar``), in both parameter dtypes; the state, the dtype of the output and the typed
errors. Physical values: with sources made of equivariant blocks, rotating every ``l`` block of the inputs by its
Wigner matrix (``tests_torch/cg_oracle.py``, fitted from scipy's harmonics, independent of the twin) rotates every
``l`` block of the output by the same matrix. The comparison with TensorFlow is ``test_function_reduce_n_golden.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from tensorpotential.torch_backend.errors import (
    DtypeError,
    OptionValueError,
    ShapeError,
    SpecError,
)
from tensorpotential.torch_backend.instructions.function_reduce_n import (
    FunctionReduceN,
)
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.plans.function_reduce_n import ReduceSource
from tensorpotential.torch_backend.spec.loader import parse_model_spec
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.cg_oracle import wigner_d
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close

RNG_SEED = 20261018
CLS = COMPUTE + "FunctionReduceN"
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_TYPES = 3
NATURAL = ((0, 1), (1, -1), (2, 1))
ONLY_INVAR = {"ls_max": 0, "allowed_l_p": [[0, 1], [0, -1]]}
"""Every ``ls_max`` 0, with a table of two ``l = 0`` rows."""


def _table(blocks: list[tuple[int, str, int]]) -> pd.DataFrame:
    """Return a coupling table with one row per ``m`` of each ``(l, hist, parity)`` block, in the given order."""
    rows = [(l, m, h, p, l) for l, h, p in blocks for m in range(-l, l + 1)]
    return pd.DataFrame(
        rows, columns=pd.Index(["l", "m", "hist", "parity", "sum_of_ls"])
    )


SOURCES = (
    ReduceSource("A", _table([(0, "", 1), (1, "", -1), (2, "", 1)]), 3),
    ReduceSource(
        "AA",
        _table([
            (1, "a", -1),
            (0, "a", 1),
            (0, "b", 1),
            (1, "b", 1),
            (2, "a", 1),
            (1, "c", -1),
        ]),
        2,
    ),
)


def _spec(sources=SOURCES, **options: object) -> InstructionSpec:
    given = {
        "instructions": tuple(InstructionRef(s.name) for s in sources),
        "ls_max": [2, 1],
        "allowed_l_p": [list(lp) for lp in NATURAL],
        "n_out": 4,
        **options,
    }
    return InstructionSpec(
        name="R",
        cls=CLS,
        options={**DEFAULTS[CLS], **given},
        depends_on=tuple(s.name for s in sources),
        index=len(sources),
    )


def _twin(
    policy=F64, sources=SOURCES, seed=RNG_SEED, **options: object
) -> FunctionReduceN:
    twin = FunctionReduceN(_spec(sources, **options), sources, policy)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in twin.parameters():
            p.copy_(torch.as_tensor(rng.normal(size=p.shape)))
    return twin


def _inputs(rng, n_atoms=5, sources=SOURCES, dtype=np.float64, extra_rows=0):
    inputs = {
        s.name: torch.as_tensor(
            rng.normal(
                size=(n_atoms + extra_rows, len(s.coupling_meta_data), s.n_out)
            ).astype(dtype)
        )
        for s in sources
    }
    inputs["atomic_mu_i"] = torch.as_tensor(rng.integers(0, N_TYPES, size=n_atoms))
    return inputs


def _reference(twin: FunctionReduceN, inputs) -> np.ndarray:
    """Sheet section 5, atom by atom, in float64: ``[atoms, rows, n_out]``."""
    plan = twin.plan
    mu = inputs["atomic_mu_i"].numpy()
    rows = 1 if plan.only_invar else len(plan.coupling_meta_data)
    out = np.full((len(mu), rows, twin.n_out), twin.init_value)
    for c in plan.collectors:
        w_all = twin.get_parameter(f"reducing_{c.name}").detach().double().numpy()
        x = inputs[c.name].double().numpy()
        for a in range(len(mu)):
            w = w_all[mu[a]] if twin.per_element else w_all
            for j, (col, g) in enumerate(
                zip(c.func_collect_ind, c.w_l_tile, strict=True)
            ):
                row = 0 if plan.only_invar else c.total_sum_ind[j, 0]
                out[a, row] += c.norm * w[:, :, g] @ x[a, col]
    return out * plan.runtime_norm_map.reshape(1, -1, 1)


# ------------------------------------------------------------------ logic

OPTION_CASES = {
    "plain": {},
    "per_element": {
        "is_central_atom_type_dependent": True,
        "number_of_atom_types": N_TYPES,
    },
    "out_norm": {"out_norm": True},
    "no_normalize": {"normalize": False},
    "scale": {"scale": 2.5},
    "ones": {"init_target_value": "ones"},
    "only_invar": ONLY_INVAR,
    "lm_first_is_a_tf_layout": {"lm_first": True},
}


@pytest.mark.parametrize(("policy", "row"), [(F64, FEATURE_F64), (F32, FEATURE_F32)])
@pytest.mark.parametrize("options", OPTION_CASES.values(), ids=OPTION_CASES.keys())
def test_the_forward_pass_is_the_formula_of_the_sheet(policy, row, options):
    twin = _twin(policy, **options)
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(inputs)
    assert out.dtype == policy.param
    assert_close(out, _reference(twin, inputs), row)


def test_only_invar_sums_every_collected_entry_into_one_row():
    twin = _twin(**ONLY_INVAR)
    assert (
        len(twin.plan.coupling_meta_data) == 2
    )  # the table has two rows, the output one, as in TF
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    assert twin(inputs).shape == (5, 1, 4)


def test_only_invar_with_out_norm_broadcasts_to_the_rows_of_the_table_as_tf():
    twin = _twin(**ONLY_INVAR, out_norm=True)
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    out = twin(inputs)
    assert out.shape == (5, 2, 4)
    assert_close(out, _reference(twin, inputs), FEATURE_F64)


def test_init_target_value_ones_adds_one_to_every_entry():
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    difference = _twin(init_target_value="ones")(inputs) - _twin()(inputs)
    assert_close(difference, np.ones(difference.shape), FEATURE_F64)


def test_the_output_has_the_parameter_dtype_whatever_the_input_dtype():
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=np.float64)
    assert _twin(F32)(inputs).dtype == torch.float32
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=np.float32)
    assert _twin(F64)(inputs).dtype == torch.float64


def test_per_element_weights_cut_padded_sources_to_the_atoms():
    rng = np.random.default_rng(RNG_SEED)
    twin = _twin(is_central_atom_type_dependent=True, number_of_atom_types=N_TYPES)
    padded = _inputs(rng, extra_rows=2)
    exact = {k: v[:5] for k, v in padded.items()}
    torch.testing.assert_close(twin(padded), twin(exact), rtol=0, atol=0)


def test_each_element_uses_its_own_weights():
    twin = _twin(is_central_atom_type_dependent=True, number_of_atom_types=N_TYPES)
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs["atomic_mu_i"] = torch.tensor([0, 1, 2, 0, 1])
    with torch.no_grad():
        for p in twin.parameters():
            p[1] = 0.0
    out = twin(inputs)
    assert torch.all(out[[1, 4]] == 0.0)
    assert torch.all(out[[0, 2, 3]].abs().sum(dim=(1, 2)) > 0.0)


def test_the_state_is_the_weights_of_the_checkpoint():
    twin = FunctionReduceN(
        _spec(is_central_atom_type_dependent=True, number_of_atom_types=N_TYPES),
        SOURCES,
        F32,
    )
    state = twin.state_dict()
    assert list(state) == ["reducing_A", "reducing_AA"]
    assert state["reducing_A"].shape == (N_TYPES, 4, 3, 3)
    assert state["reducing_AA"].shape == (N_TYPES, 4, 2, 4)  # (1, +1) is not allowed
    assert all(v.dtype == torch.float32 and not v.any() for v in state.values())


def test_inputs_are_not_modified_and_the_twin_passes_gradcheck():
    twin = _twin(
        is_central_atom_type_dependent=True, number_of_atom_types=N_TYPES, out_norm=True
    )
    inputs = _inputs(np.random.default_rng(RNG_SEED), n_atoms=2)
    saved = {k: v.clone() for k, v in inputs.items()}
    names = [s.name for s in SOURCES]

    def run(*features: torch.Tensor) -> torch.Tensor:
        return twin({**inputs, **dict(zip(names, features, strict=True))})

    features = tuple(inputs[n].clone().requires_grad_() for n in names)
    assert torch.autograd.gradcheck(run, features)
    assert all(torch.equal(inputs[k], saved[k]) for k in inputs)


def test_a_yaml_with_init_target_value_ones_loads():
    raw = [
        {"__cls__": COMPUTE + "BondLength", "name": "r"},
        {
            "__cls__": CLS,
            "name": "R",
            "instructions": [{"_instruction_": True, "name": "r"}],
            "ls_max": 0,
            "allowed_l_p": [[0, 1]],
            "n_out": 2,
            "init_target_value": "ones",
        },
    ]
    assert parse_model_spec(raw)["R"].options["init_target_value"] == "ones"


def test_sources_other_than_those_of_the_spec_are_refused():
    with pytest.raises(
        SpecError, match=r"collects \['A', 'AA'\].*given are \['AA', 'A'\]"
    ):
        FunctionReduceN(_spec(), SOURCES[::-1], F64)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"simplify": True}, "simplify = true is not supported"),
        (
            {"init_target_value": "twos"},
            "init_target_value 'twos' is not one of zeros, ones",
        ),
        ({"is_central_atom_type_dependent": True}, "needs number_of_atom_types"),
    ],
)
def test_unsupported_options_are_typed_errors(options, match):
    with pytest.raises(OptionValueError, match=match):
        FunctionReduceN(_spec(**options), SOURCES, F64)


def test_a_source_of_another_angular_size_is_refused():
    twin = _twin()
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs["AA"] = inputs["AA"][:, :-1]
    with pytest.raises(
        ShapeError, match=r"source 'AA' must be \[atoms, 16, n\], got \(5, 15, 2\)"
    ):
        twin(inputs)


def test_an_integer_source_is_refused():
    twin = _twin()
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs["A"] = inputs["A"].long()
    with pytest.raises(DtypeError):
        twin(inputs)


# ------------------------------------------------------------------ physical values


def _rotate(
    features: np.ndarray, table: pd.DataFrame, rotation: np.ndarray
) -> np.ndarray:
    """Rotate every ``(l, hist, parity)`` block (``m = -l .. l``, contiguous) of ``[atoms, lm, n]``."""
    directions = np.random.default_rng(RNG_SEED).normal(size=(40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    out = features.copy()
    start = 0
    while start < len(table):
        l = int(table["l"].iloc[start])
        block = slice(start, start + 2 * l + 1)
        out[:, block] = np.einsum(
            "mk,akn->amn", wigner_d(rotation, l, directions), features[:, block]
        )
        start += 2 * l + 1
    return out


@pytest.mark.parametrize("per_element", [False, True])
def test_rotating_the_sources_rotates_the_output(per_element):
    options = (
        {"is_central_atom_type_dependent": True, "number_of_atom_types": N_TYPES}
        if per_element
        else {}
    )
    twin = _twin(out_norm=True, **options)
    rng = np.random.default_rng(RNG_SEED)
    inputs = _inputs(rng)
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    rotated = dict(inputs)
    for s in SOURCES:
        rotated[s.name] = torch.as_tensor(
            _rotate(inputs[s.name].numpy(), s.coupling_meta_data, rotation)
        )
    with torch.no_grad():
        before, after = twin(inputs).numpy(), twin(rotated).numpy()
    expected = _rotate(before, twin.plan.coupling_meta_data, rotation)
    assert_close(after, expected, FEATURE_F64)
    assert np.abs(before[:, 1:]).max() > 1e-2  # the l > 0 rows are not trivially zero
