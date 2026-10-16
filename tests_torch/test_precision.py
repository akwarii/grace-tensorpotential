"""The dtype, device and precision policy (``torch_backend/ops/precision.py``) and the dtype discipline of the backend.

Logic: the parameter dtype and the casts, the constants under both torch default dtypes, TF's promotion rule, the
float64 energy sum, the TF32 scope (set, restored, also after an error and around the legacy API), no global state
changed by importing the backend, the dispatch-mode audit (no op mixes float dtypes, no host read) and a scan of the
sources (explicit dtypes, no default dtype read, casts only in the policy). Physical values: the parameter dtype of the
stored golden weights, and the energies of the TF goldens summed from their atomic energies.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorpotential.torch_backend.errors import DtypeError, TorchBackendError
from tensorpotential.torch_backend.ops import precision as p
from tensorpotential.torch_backend.ops.scatter import segment_sum, structure_energy
from tests.fresh_python import run_fresh_python
from tests_torch.dtype_audit import DtypeAudit
from tests_torch.golden import TINY_F32, TINY_F64, cases, load
from tests_torch.golden.tolerances import ENERGY_F64, assert_close

BACKEND = Path(p.__file__).resolve().parents[1]
CPU = torch.device("cpu")


@pytest.fixture(
    params=[torch.float32, torch.float64], ids=["default_f32", "default_f64"]
)
def default_dtype(request):
    old = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield request.param
    torch.set_default_dtype(old)


@pytest.fixture
def restore_precision():
    saved = p.fp32_precision_state()
    yield
    p.set_fp32_precision(saved)


# ------------------------------------------------------------------ parameter dtype and casts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_policy_has_its_parameter_dtype_and_float64_geometry(dtype):
    policy = p.DtypePolicy(dtype)
    assert policy.param is dtype
    assert p.GEOMETRY_DTYPE is p.ENERGY_DTYPE is torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_another_parameter_dtype_is_refused(dtype):
    with pytest.raises(DtypeError, match="float32 or float64"):
        p.DtypePolicy(dtype)


@pytest.mark.golden
@pytest.mark.parametrize(
    ("suffix", "dtype"), [("f32", torch.float32), ("f64", torch.float64)]
)
@pytest.mark.parametrize("model", ["omat", "large"])
def test_the_parameter_dtype_is_read_from_the_stored_weights(model, suffix, dtype):
    weights = load(f"{model}_tiny_{suffix}.weights")
    kinds = {w.dtype.kind for w in weights.values()}
    policy = p.DtypePolicy.from_weights(w.dtype for w in weights.values())
    assert kinds >= {"f", "U"}  # the element symbols are text and are ignored
    assert policy.param is dtype


@pytest.mark.parametrize(
    "dtypes",
    [[np.float32, np.float64], [np.int32, "<U2"], [], [np.float16]],
    ids=["mixed", "no_float", "empty", "float16"],
)
def test_weights_without_one_parameter_dtype_are_refused(dtypes):
    with pytest.raises(DtypeError, match="float32 or all float64"):
        p.DtypePolicy.from_weights(dtypes)


def test_the_casts_give_the_policy_dtypes_and_copy_only_when_needed():
    policy = p.DtypePolicy(torch.float32)
    x64 = torch.ones(3, dtype=torch.float64)
    assert policy.to_param(x64).dtype == torch.float32
    assert policy.to_geometry(x64) is x64
    x32 = policy.to_param(x64)
    assert policy.to_param(x32) is x32


def test_an_integer_tensor_is_not_cast():
    with pytest.raises(DtypeError, match="only floating tensors"):
        p.DtypePolicy(torch.float64).to_param(torch.ones(2, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64], ids=str)
def test_check_floating_refuses_other_tensors_and_passes_floats(dtype):
    for floating in (torch.float32, torch.float64):
        p.check_floating(torch.ones(2, dtype=floating))
    with pytest.raises(DtypeError, match="a floating tensor is needed"):
        p.check_floating(torch.ones(2, dtype=dtype))


@pytest.mark.usefixtures("default_dtype")
@pytest.mark.parametrize("param", [torch.float32, torch.float64])
def test_constants_have_an_explicit_dtype_under_either_default(param):
    policy = p.DtypePolicy(param)
    cg = policy.param_constant([0.5, -1.0 / 3.0], CPU)
    eps = policy.geometry_constant(1e-10, CPU)
    assert (cg.dtype, eps.dtype) == (param, torch.float64)
    assert cg.device == eps.device == CPU
    # converted straight from the Python float: no detour through the default dtype
    assert eps.item() == 1e-10
    assert cg[1].item() == float(torch.tensor(-1.0 / 3.0, dtype=param))


# ------------------------------------------------------------------ TF promotion and the energy sum


F32, F64 = torch.float32, torch.float64


# measured on TF 2.20 with numpy behaviour: float32 with a 0-d float64 tensor gives float64; torch keeps float32
@pytest.mark.parametrize(
    ("right", "torch_gives", "tf_gives"),
    [
        (torch.tensor(2.0, dtype=F64), F32, F64),
        (torch.ones(3, dtype=F64), F64, F64),
        (torch.ones(3, dtype=F32), F32, F32),
    ],
    ids=["0d_float64", "float64_vector", "float32_only"],
)
def test_promotion_follows_tf_where_torch_would_not(right, torch_gives, tf_gives):
    left = torch.ones(3, dtype=F32)
    assert (left * right).dtype == torch_gives
    promoted = p.promote_like_tf(left, right)
    assert [t.dtype for t in promoted] == [tf_gives, tf_gives]
    assert (promoted[0] * promoted[1]).dtype == tf_gives


def test_promotion_leaves_index_tensors_alone_and_needs_a_float():
    index = torch.arange(3)
    _, kept = p.promote_like_tf(torch.ones(1, dtype=torch.float32), index)
    assert kept is index
    with pytest.raises(DtypeError, match="at least one floating"):
        p.promote_like_tf(index)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_cast_to_the_dtype_of_a_reference_narrows_and_widens(dtype):
    reference = torch.zeros(2, dtype=dtype)
    for source in (torch.float32, torch.float64):
        assert p.to_dtype_of(torch.ones(3, dtype=source), reference).dtype == dtype
    with pytest.raises(DtypeError, match="only floating"):
        p.to_dtype_of(torch.ones(3, dtype=torch.int32), reference)


def test_operands_of_one_dtype_pass_and_mixed_operands_are_named():
    f32, f64 = torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.float64)
    assert p.same_dtype(f32, f32) == torch.float32
    with pytest.raises(DtypeError, match=r"torch\.float32, torch\.float64"):
        p.same_dtype(f64, f32)
    with pytest.raises(DtypeError, match="none"):
        p.same_dtype()


@pytest.mark.usefixtures("default_dtype")
def test_float32_atomic_energies_are_summed_in_float64():
    rng = np.random.default_rng(7)
    atomic = rng.normal(size=(9, 1)).astype(np.float32)
    structure = np.array([0, 0, 2, 2, 2, 0, 3, 3, 3])
    energy = structure_energy(torch.as_tensor(atomic), torch.as_tensor(structure), 4)
    expected = np.zeros((4, 1))
    np.add.at(expected, structure, atomic.astype(np.float64))
    assert energy.dtype == torch.float64
    assert energy.shape == (4, 1)
    # float64 sums of float32 values: exact here
    assert np.array_equal(energy.numpy(), expected)
    assert energy[1].item() == 0.0


@pytest.mark.golden
@pytest.mark.parametrize("name", [*TINY_F64, *TINY_F32])
def test_the_energies_of_tf_are_the_float64_sums_of_its_atomic_energies(name):
    arrays = load(name)
    for case in cases(arrays):
        atomic = torch.as_tensor(arrays[f"{case}/res/atomic_energy"])
        to_structure = torch.as_tensor(arrays[f"{case}/in/map_atoms_to_structure"])
        energy = structure_energy(
            atomic, to_structure, int(arrays[f"{case}/in/n_struct_total"])
        )
        assert_close(energy, arrays[f"{case}/res/energy"], ENERGY_F64, f"{name} {case}")


# ------------------------------------------------------------------ the TF32 scope


@pytest.mark.usefixtures("restore_precision")
def test_the_scope_sets_full_precision_and_restores_the_caller():
    p.set_fp32_precision({"cuda.matmul": "tf32", "mkldnn.matmul": "bf16"})
    before = p.fp32_precision_state()
    assert before["cuda.matmul"] == "tf32" and before["mkldnn.matmul"] == "bf16"
    with p.precision_scope():
        assert set(p.fp32_precision_state().values()) == {"ieee"}
        with p.precision_scope():
            assert set(p.fp32_precision_state().values()) == {"ieee"}
        assert set(p.fp32_precision_state().values()) == {"ieee"}
    assert p.fp32_precision_state() == before


@pytest.mark.usefixtures("restore_precision")
def test_the_scope_restores_the_caller_after_an_error():
    before = p.fp32_precision_state()
    with pytest.raises(ZeroDivisionError), p.precision_scope():
        _ = 1 / 0
    assert p.fp32_precision_state() == before


@pytest.mark.usefixtures("restore_precision")
def test_only_the_known_attributes_are_set_and_nothing_is_set_otherwise():
    before = p.fp32_precision_state()
    with pytest.raises(TorchBackendError, match=r"policy: cudnn\.attention"):
        p.set_fp32_precision({"cuda.matmul": "tf32", "cudnn.attention": "ieee"})
    assert p.fp32_precision_state() == before


_LEGACY_CALLER = """
import json, sys, torch
from tensorpotential.torch_backend.ops.precision import precision_scope

def legacy():
    out = {}
    for name, read in (
        ("matmul", lambda: torch.backends.cuda.matmul.allow_tf32),
        ("cudnn", lambda: torch.backends.cudnn.allow_tf32),
        ("precision", torch.get_float32_matmul_precision),
    ):
        try:
            out[name] = read()
        except RuntimeError:
            out[name] = "raises"
    return out

if sys.argv[1] == "legacy":
    torch.backends.cuda.matmul.allow_tf32 = True
elif sys.argv[1] == "new":
    torch.backends.cuda.matmul.fp32_precision = "tf32"
before = legacy()
with precision_scope():
    pass
print(json.dumps([before, legacy()]))
"""


@pytest.mark.parametrize(
    ("caller", "matmul_flag"),
    [("untouched", False), ("legacy", True), ("new", "raises")],
)
def test_the_scope_leaves_a_caller_of_either_api_as_it_was(
    tmp_path, caller, matmul_flag
):
    # "raises": after the new API is used, torch refuses to read a legacy flag; the scope must not change that either
    done = run_fresh_python(_LEGACY_CALLER, tmp_path, args=[caller])
    assert done.returncode == 0, done.stderr
    before, after = json.loads(done.stdout.strip().splitlines()[-1])
    assert after == before
    assert before["matmul"] == matmul_flag


_IMPORT_STATE = """
import importlib, json, pkgutil, random
import numpy as np, torch

def state():
    backends = torch.backends
    knobs = [backends, backends.cuda.matmul, backends.cudnn.conv, backends.cudnn.rnn,
             backends.mkldnn.matmul, backends.mkldnn.conv, backends.mkldnn.rnn]
    return {
        "fp32_precision": [k.fp32_precision for k in knobs],
        "legacy": [backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()],
        "default_dtype": str(torch.get_default_dtype()),
        "torch_rng": torch.random.get_rng_state().sum().item(),
        "numpy_rng": int(np.random.get_state()[1].sum()),
        "python_rng": hash(random.getstate()),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "grad": torch.is_grad_enabled(),
    }

before = state()
import tensorpotential.torch_backend as backend
names = [m.name for m in pkgutil.walk_packages(backend.__path__, backend.__name__ + ".")]
for name in names:
    importlib.import_module(name)
print(json.dumps([before, state(), names]))
"""


def test_importing_the_backend_changes_no_global_state_of_torch(tmp_path):
    done = run_fresh_python(_IMPORT_STATE, tmp_path)
    assert done.returncode == 0, done.stderr
    before, after, names = json.loads(done.stdout.strip().splitlines()[-1])
    assert "tensorpotential.torch_backend.ops.precision" in names
    assert after == before


# ------------------------------------------------------------------ the dispatch-mode audit


@pytest.mark.usefixtures("default_dtype")
def test_the_policy_and_the_scatter_sum_mix_no_dtypes_and_read_nothing_on_the_host():
    policy = p.DtypePolicy(torch.float32)
    bonds = (
        torch.linspace(0.5, 3.0, 18, dtype=torch.float64).reshape(6, 3).requires_grad_()
    )
    ind_j = torch.tensor([0, 1, 1, 2, 0, 2])
    with p.precision_scope(), DtypeAudit() as audit:
        features = policy.to_param(bonds) * policy.param_constant(2.0, CPU)
        atomic = segment_sum(features, ind_j, 3).sum(dim=1, keepdim=True)
        energy = structure_energy(atomic, torch.zeros(3, dtype=torch.int64), 1)
        (grad,) = torch.autograd.grad(energy.sum(), bonds)
    audit.assert_clean()
    assert grad.dtype == torch.float64
    assert torch.equal(grad, torch.full_like(bonds, 2.0))


def test_the_audit_sees_a_silent_promotion_and_host_reads():
    with DtypeAudit() as audit:
        _ = torch.ones(2, dtype=torch.float32) * torch.tensor(2.0, dtype=torch.float64)
        _ = torch.ones(2, dtype=torch.float64).sum().item()
        _ = torch.nonzero(torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert [name for name, _ in audit.mixed] == ["aten::mul"]
    assert audit.host_reads == ["aten::_local_scalar_dense", "aten::nonzero"]
    with pytest.raises(AssertionError, match="mixing float dtypes"):
        audit.assert_clean()


# ------------------------------------------------------------------ the dtype discipline of the sources

FACTORIES = frozenset({
    "arange",
    "as_tensor",
    "empty",
    "eye",
    "full",
    "linspace",
    "ones",
    "rand",
    "randn",
    "tensor",
    "zeros",
})
"""torch functions that create a tensor in the default dtype unless they are given one."""
FORBIDDEN = frozenset({
    "get_default_dtype",
    "set_default_dtype",
    "set_float32_matmul_precision",
    "allow_tf32",
    "manual_seed",
    "seed",
    "set_rng_state",
    "use_deterministic_algorithms",
    "set_num_threads",
})
"""Attributes that read the default dtype or change global state of torch."""
CASTS = frozenset({"to", "type", "double", "float", "half", "bfloat16", "cpu", "cuda"})
"""Tensor methods that change a dtype or a device: allowed only in the policy module."""


def _violations(source: str, *, may_cast: bool) -> list[str]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            found.append((node.lineno, node.attr))
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        func = node.func
        on_torch = isinstance(func.value, ast.Name) and func.value.id == "torch"
        if (
            on_torch
            and func.attr in FACTORIES
            and "dtype" not in {k.arg for k in node.keywords}
        ):
            found.append((node.lineno, f"torch.{func.attr} without dtype"))
        if not on_torch and func.attr in CASTS and not may_cast:
            found.append((node.lineno, f".{func.attr}() outside the policy"))
    return [f"line {line}: {what}" for line, what in sorted(found)]


@pytest.mark.parametrize(
    "path",
    sorted(BACKEND.rglob("*.py")),
    ids=lambda q: q.relative_to(BACKEND).as_posix(),
)
def test_the_backend_creates_every_tensor_with_a_dtype_and_casts_only_in_the_policy(
    path,
):
    assert _violations(path.read_text(), may_cast=path == Path(p.__file__)) == []


def test_the_source_scan_sees_each_kind_of_violation():
    planted = (
        "x = torch.zeros(3)\n"
        "y = torch.zeros(3, dtype=torch.float64)\n"
        "d = torch.get_default_dtype()\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "z = y.double()\n"
    )
    assert _violations(planted, may_cast=False) == [
        "line 1: torch.zeros without dtype",
        "line 3: get_default_dtype",
        "line 4: allow_tf32",
        "line 5: .double() outside the policy",
    ]
    assert _violations("z = y.to(torch.float32)\n", may_cast=True) == []
