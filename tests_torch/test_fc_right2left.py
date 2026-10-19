"""The ``FCRight2Left`` twin: logic against a numpy loop of the sheet's formula, and rotation equivariance.

Logic: the forward pass of the spec sheet (section 5) written as a loop over atoms and table rows in numpy, from the
plan's tables and the weights, for every option the twin ports (``left_coefs``, per-element weights on either side,
``normalize``, ``norm_out``, ``n_out`` null), in both parameter dtypes; the state, the refusal of mixed dtypes (TF fails
there), and the typed errors. Physical values: with operands made of equivariant blocks, rotating every ``l`` block of
both operands by its Wigner matrix (``tests_torch/cg_oracle.py``, fitted from scipy's harmonics) rotates every ``l``
block of the output by the same matrix. The comparison with TensorFlow is ``test_fc_right2left_golden.py``.
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
from tensorpotential.torch_backend.instructions.fc_right2left import FCRight2Left
from tensorpotential.torch_backend.ops.precision import DtypePolicy
from tensorpotential.torch_backend.plans.fc_right2left import per_element_sides
from tensorpotential.torch_backend.plans.function_reduce_n import ReduceSource
from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS
from tests_torch.cg_oracle import wigner_d
from tests_torch.golden.tolerances import FEATURE_F32, FEATURE_F64, assert_close
from tests_torch.init_only import assert_option_is_ignored

RNG_SEED = 20261018
CLS = COMPUTE + "FCRight2Left"
F64, F32 = DtypePolicy(torch.float64), DtypePolicy(torch.float32)
N_TYPES = 3


def _table(blocks: list[tuple[int, str, int]]) -> pd.DataFrame:
    """Return a coupling table with one row per ``m`` of each ``(l, hist, parity)`` block, in the given order."""
    rows = [(l, m, h, p, l) for l, h, p in blocks for m in range(-l, l + 1)]
    return pd.DataFrame(
        rows, columns=pd.Index(["l", "m", "hist", "parity", "sum_of_ls"])
    )


LEFT = ReduceSource("L", _table([(0, "", 1), (1, "", -1), (2, "", 1), (1, "x", 1)]), 3)
RIGHT = ReduceSource(
    "R", _table([(1, "a", -1), (0, "a", 1), (1, "b", -1), (2, "a", 1), (3, "a", -1)]), 2
)


def _spec(left=LEFT, right=RIGHT, **options: object) -> InstructionSpec:
    given = {
        "left": InstructionRef(left.name),
        "right": InstructionRef(right.name),
        "n_out": 4,
        "norm_out": True,
        **options,
    }
    return InstructionSpec(
        name="F",
        cls=CLS,
        options={**DEFAULTS[CLS], **given},
        depends_on=(left.name, right.name),
        index=2,
    )


def _twin(policy=F64, left=LEFT, right=RIGHT, **options: object) -> FCRight2Left:
    twin = FCRight2Left(_spec(left, right, **options), left, right, policy)
    rng = np.random.default_rng(RNG_SEED)
    with torch.no_grad():
        for p in twin.parameters():
            p.copy_(torch.as_tensor(rng.normal(size=p.shape)))
    return twin


def _inputs(rng, n_atoms=5, dtype=np.float64, left=LEFT, right=RIGHT):
    inputs = {
        s.name: torch.as_tensor(
            rng.normal(size=(n_atoms, len(s.coupling_meta_data), s.n_out)).astype(dtype)
        )
        for s in (left, right)
    }
    inputs["atomic_mu_i"] = torch.as_tensor(rng.integers(0, N_TYPES, size=n_atoms))
    return inputs


def _weight(twin, name, mu, per_element):
    w = twin.get_parameter(name).detach().double().numpy()
    return w[mu] if per_element else w


def _reference(twin: FCRight2Left, inputs) -> np.ndarray:
    """Sheet section 5, atom by atom and row by row, in float64: ``[atoms, lm_left, n_out]``."""
    plan, (pl, pr), options = twin.plan, twin.per_element, twin.spec.options
    # the norms and norm_out from the options, not from the twin (sheet sections 1 and 4)
    norm_left, norm_right = (
        1 / n**0.5 if options["normalize"] else 1.0 for n in (LEFT.n_out, RIGHT.n_out)
    )
    x_left, x_right = (inputs[n].double().numpy() for n in ("L", "R"))
    mu = inputs["atomic_mu_i"].numpy()
    out = np.zeros((len(mu), x_left.shape[1], twin.n_out))
    for a in range(len(mu)):
        for row in range(x_left.shape[1]):
            if plan.w_left_shape is None:
                out[a, row] = x_left[a, row]
            else:
                w = _weight(twin, "w_left", mu[a], pl)[:, :, plan.w_tile_left[row]]
                out[a, row] = norm_left * w @ x_left[a, row]
        for to, frm, g in zip(
            plan.collect_to, plan.collect_from, plan.w_tile_right, strict=True
        ):
            w = _weight(twin, "w_right", mu[a], pr)[:, :, g]
            out[a, to] += norm_right * w @ x_right[a, frm]
    if options["norm_out"]:
        out *= plan.norm_map.reshape(1, -1, 1)
    return out


# ------------------------------------------------------------------ logic

OPTION_CASES = {
    "plain": {},
    "per_element_both": {
        "is_central_atom_type_dependent": True,
        "number_of_atom_types": N_TYPES,
    },
    "per_element_left": {
        "is_central_atom_type_dependent": [True, False],
        "number_of_atom_types": N_TYPES,
    },
    "per_element_right": {
        "is_central_atom_type_dependent": [False, True],
        "number_of_atom_types": N_TYPES,
    },
    "no_left_coefs": {"left_coefs": False, "n_out": 3},
    "n_out_of_the_left": {"n_out": None},
    "no_normalize": {"normalize": False},
    "no_norm_out": {"norm_out": False},
    "lm_first_is_a_tf_layout": {"lm_first": True},
}


@pytest.mark.parametrize(("policy", "row"), [(F64, FEATURE_F64), (F32, FEATURE_F32)])
@pytest.mark.parametrize("options", OPTION_CASES.values(), ids=OPTION_CASES.keys())
def test_the_forward_pass_is_the_formula_of_the_sheet(policy, row, options):
    twin = _twin(policy, **options)
    dtype = np.float64 if policy is F64 else np.float32
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=dtype)
    out = twin(inputs)
    assert out.dtype == policy.param
    assert_close(out, _reference(twin, inputs), row)


def test_the_output_has_the_layout_of_the_left_operand():
    twin = _twin()
    pd.testing.assert_frame_equal(twin.coupling_meta_data, LEFT.coupling_meta_data)
    assert twin.coupling_meta_data is not LEFT.coupling_meta_data
    assert twin(_inputs(np.random.default_rng(RNG_SEED))).shape == (5, 12, 4)


def test_n_out_null_takes_the_channels_of_the_left_operand():
    assert _twin(n_out=None).n_out == LEFT.n_out


def test_the_state_is_the_weights_of_the_checkpoint():
    state = FCRight2Left(
        _spec(
            is_central_atom_type_dependent=[True, False], number_of_atom_types=N_TYPES
        ),
        LEFT,
        RIGHT,
        F32,
    ).state_dict()
    assert list(state) == ["w_left", "w_right"]
    assert state["w_left"].shape == (
        N_TYPES,
        4,
        3,
        4,
    )  # groups (0,+), (1,-), (1,+,x), (2,+)
    assert state["w_right"].shape == (
        4,
        2,
        4,
    )  # pairs (0,+,a), (1,-,a), (1,-,b), (2,+,a)
    assert all(v.dtype == torch.float32 and not v.any() for v in state.values())
    no_left = FCRight2Left(_spec(left_coefs=False, n_out=3), LEFT, RIGHT, F64)
    assert list(no_left.state_dict()) == ["w_right"]


def test_left_rows_without_a_right_partner_keep_only_their_own_mixing():
    twin = _twin(norm_out=False)
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    without_right = dict(inputs, R=torch.zeros_like(inputs["R"]))
    changed = (twin(inputs) - twin(without_right)).abs().sum(dim=(0, 2))
    # rows 9..11 are (1, +, x): no right group has l = 1 with parity +1
    assert torch.all(changed[9:] == 0.0)
    assert torch.all(changed[:9] > 0.0)


def test_mixed_dtypes_are_refused_as_tf_refuses_them():
    inputs = _inputs(np.random.default_rng(RNG_SEED), dtype=np.float64)
    with pytest.raises(DtypeError, match="one dtype"):
        _twin(F32)(inputs)
    with pytest.raises(DtypeError, match="one dtype"):
        _twin(F32, left_coefs=False, n_out=3)(dict(inputs, R=inputs["R"].float()))


def test_inputs_are_not_modified_and_the_twin_passes_gradcheck():
    twin = _twin(is_central_atom_type_dependent=True, number_of_atom_types=N_TYPES)
    inputs = _inputs(np.random.default_rng(RNG_SEED), n_atoms=2)
    saved = {k: v.clone() for k, v in inputs.items()}

    def run(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return twin({**inputs, "L": left, "R": right})

    operands = tuple(inputs[n].clone().requires_grad_() for n in ("L", "R"))
    assert torch.autograd.gradcheck(run, operands)
    assert all(torch.equal(inputs[k], saved[k]) for k in inputs)


@pytest.mark.parametrize(
    ("value", "sides"),
    [
        (None, (False, False)),
        (True, (True, True)),
        (False, (False, False)),
        ([False, True], (False, True)),
    ],
)
def test_is_central_atom_type_dependent_is_read_per_side(value, sides):
    assert per_element_sides(value) == sides


@pytest.mark.parametrize("value", [1, "yes", [True], [True, 1], (True, False, True)])
def test_other_is_central_atom_type_dependent_values_are_refused(value):
    with pytest.raises(OptionValueError, match="null, a bool or a list of two bools"):
        per_element_sides(value)


def test_operands_other_than_those_of_the_spec_are_refused():
    with pytest.raises(SpecError, match=r"its left operand is .* but 'R' was given"):
        FCRight2Left(_spec(), RIGHT, RIGHT, F64)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"is_central_atom_type_dependent": True}, "number_of_atom_types is required"),
        ({"left_coefs": False, "n_out": 4}, "left_coefs false needs n_out"),
    ],
)
def test_option_errors_of_the_plan_reach_the_caller(options, match):
    with pytest.raises(OptionValueError, match=match):
        FCRight2Left(_spec(**options), LEFT, RIGHT, F64)


def test_operands_without_a_shared_l_and_parity_are_refused():
    odd = ReduceSource("R", _table([(1, "a", 1)]), 2)
    only_vectors = ReduceSource("L", _table([(1, "", -1)]), 3)
    with pytest.raises(ShapeError, match="nothing to collect"):
        FCRight2Left(_spec(only_vectors, odd), only_vectors, odd, F64)


@pytest.mark.parametrize("operand", ["L", "R"])
def test_an_operand_of_another_angular_size_is_refused(operand):
    twin = _twin()
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    inputs[operand] = inputs[operand][:, :-1]
    with pytest.raises(
        ShapeError, match=rf"operand '{operand}' must be \[atoms, \d+, n\]"
    ):
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
def test_rotating_the_operands_rotates_the_output(per_element):
    options = (
        {"is_central_atom_type_dependent": True, "number_of_atom_types": N_TYPES}
        if per_element
        else {}
    )
    twin = _twin(**options)
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    rotation = Rotation.random(random_state=RNG_SEED).as_matrix()
    rotated = dict(inputs)
    for s in (LEFT, RIGHT):
        rotated[s.name] = torch.as_tensor(
            _rotate(inputs[s.name].numpy(), s.coupling_meta_data, rotation)
        )
    with torch.no_grad():
        before, after = twin(inputs).numpy(), twin(rotated).numpy()
    assert_close(after, _rotate(before, LEFT.coupling_meta_data, rotation), FEATURE_F64)
    assert np.abs(before[:, 1:]).max() > 1e-2  # the l > 0 rows are not trivially zero


@pytest.mark.parametrize("case", sorted(OPTION_CASES))
def test_init_vars_changes_neither_the_state_nor_the_output(case):
    inputs = _inputs(np.random.default_rng(RNG_SEED))
    assert_option_is_ignored(
        lambda **o: _twin(**OPTION_CASES[case], **o), CLS, "init_vars", inputs
    )
