"""The tolerance table (``tests_torch/golden/tolerances.py``) and the edit-lock of the table and the goldens.

Logic: the rows (one per dtype and quantity, each with a unit and a rationale), the ``numpy.isclose`` semantics of
``deviations`` and ``assert_close`` on hand values, the lock (it matches the tree, and sees a changed, an added and a
removed file). Physical values: the floors the rows are derived from, measured again on the committed anchors: the
float64 rows leave at least 50 times the summation-order floor, and the float32 rows hold TF's float32 spread.
"""

from __future__ import annotations

import shutil
from typing import get_args

import numpy as np
import pytest
import torch

from tests_torch.golden import TINY_F64, cases, load
from tests_torch.golden import lock as lk
from tests_torch.golden import tolerances as tol

HAND = tol.Row("HAND", "float64", "feature", 0.5, 0.25, "1", "hand row of the tests")


# ------------------------------------------------------------------ the rows


def test_every_quantity_has_a_row_in_each_dtype():
    pairs = {(r.dtype, r.quantity) for r in tol.ROWS if r.quantity != "any"}
    quantities = [q for q in get_args(tol.Quantity) if q != "any"]
    assert pairs == {(d, q) for d in get_args(tol.Dtype) for q in quantities}
    assert all(tol.row_for(d, q) in tol.ROWS for d, q in pairs)


def test_the_rows_are_named_after_their_constant_and_explain_themselves():
    assert len({r.name for r in tol.ROWS}) == len(tol.ROWS)
    for row in tol.ROWS:
        assert getattr(tol, row.name) is row
        assert row.rtol > 0 and row.atol > 0 and row.unit and len(row.rationale) > 30


def test_the_rows_are_in_units_of_the_float_epsilon():
    assert {r.rtol for r in tol.ROWS if r.dtype == "float64"} == {tol.F64_RTOL}
    assert pytest.approx(100 * 2 * tol.EPS64, rel=0.15) == tol.F64_RTOL
    assert {r.rtol for r in tol.ROWS if r.dtype == "float32"} == {
        tol.F32_RTOL,
        tol.CROSS_CPU_RTOL,
    }
    assert (tol.F32_RTOL / tol.EPS32, tol.CROSS_CPU_RTOL / tol.EPS32) == (256, 64)


# ------------------------------------------------------------------ isclose semantics


def test_equal_arrays_have_no_deviation_and_infinite_headroom():
    found = tol.deviations(np.ones(4), np.ones(4), HAND)
    assert (found.max_abs, found.max_rel, found.ratio, found.headroom) == (
        0.0,
        0.0,
        0.0,
        float("inf"),
    )


def test_the_bound_is_atol_plus_rtol_times_the_reference():
    expected = np.array([1.0, 0.0, -2.0])
    # 0.25 + 0.5 * |expected|, all exact in binary
    at_bound = np.array([1.75, 0.25, -3.25])
    found = tol.assert_close(at_bound, expected, HAND)
    assert found.ratio == 1.0 and found.headroom == 1.0
    # the largest relative error is the 0.25 off a zero
    assert found.max_abs == 1.25 and found.max_rel == float("inf")
    with pytest.raises(AssertionError, match=r"HAND .* at \(2,\) -3\.5 against -2, 1\.2 times"):
        tol.assert_close(np.array([1.0, 0.0, -3.5]), expected, HAND, "hand")


def test_a_zero_absolute_part_makes_an_error_on_zero_infinitely_large():
    exact = tol.Row("EXACT", "float64", "feature", 1e-3, 0.0, "1", "relative only")
    assert tol.deviations(np.array([1e-300]), np.array([0.0]), exact).ratio == float(
        "inf"
    )
    assert tol.deviations(np.array([0.0]), np.array([0.0]), exact).ratio == 0.0


def test_shapes_must_be_equal_and_empty_arrays_agree():
    with pytest.raises(ValueError, match="shapes differ"):
        tol.deviations(np.ones((3, 1)), np.ones(3), HAND)
    assert tol.deviations(np.ones((0, 3)), np.ones((0, 3)), HAND).ratio == 0.0


def test_a_torch_tensor_that_needs_a_gradient_is_measured_as_it_is():
    actual = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    found = tol.deviations(actual, np.array([1.0, 2.5]), HAND)
    assert found.max_abs == 0.5 and found.index == (1,)


# ------------------------------------------------------------------ the floors behind the rows


def _summation_order_rebuild(arrays, case):
    pair_f = arrays[f"{case}/res/pair_f"]
    ind_i, ind_j = arrays[f"{case}/in/ind_i"][::-1], arrays[f"{case}/in/ind_j"][::-1]
    forces = np.zeros_like(arrays[f"{case}/res/forces"])
    np.add.at(forces, ind_j, pair_f[::-1])
    np.add.at(forces, ind_i, -pair_f[::-1])
    energy = np.sum(arrays[f"{case}/res/atomic_energy"][::-1]).reshape(1, 1)
    full = np.einsum("bi,bj->ij", arrays[f"{case}/in/bond_vector"][::-1], pair_f[::-1])
    virial = full[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]  # xx, yy, zz, xy, xz, yz
    return {"forces": forces, "energy": energy, "virial": virial}


@pytest.mark.golden
@pytest.mark.parametrize("name", TINY_F64)
def test_the_float64_rows_leave_fifty_times_the_summation_order_floor(name):
    arrays = load(name)
    for case in cases(arrays):
        for quantity, value in _summation_order_rebuild(arrays, case).items():
            found = tol.deviations(
                value,
                arrays[f"{case}/res/{quantity}"],
                tol.row_for("float64", quantity),
            )
            assert found.headroom >= 50, (case, quantity, found)


@pytest.mark.golden
@pytest.mark.parametrize("name", TINY_F64)
def test_the_float32_rows_hold_the_float32_spread_of_tf(name):
    f64, f32 = load(name), load(name.replace("f64", "f32"))
    quantities = ("energy", "forces", "virial", "stress")
    for case in cases(f64):
        for quantity in quantities:
            key = f"{case}/res/{quantity}"
            found = tol.deviations(f32[key], f64[key], tol.row_for("float32", quantity))
            assert found.headroom >= 5, (key, found)
        for key in (k for k in f64 if k.startswith(f"{case}/ins/") and f64[k].size):
            found = tol.deviations(f32[key], f64[key], tol.FEATURE_F32)
            assert found.headroom >= 5, (key, found)


# ------------------------------------------------------------------ the edit-lock


def test_the_lock_matches_the_table_and_the_goldens():
    recorded = lk.read_lock()
    assert "tests_torch/golden/tolerances.py" in recorded
    assert {p.split("/")[1] for p in recorded} == {"golden", "fixtures"}
    assert lk.differences(recorded, lk.digests()) == []


def test_the_lock_sees_a_changed_an_added_and_a_removed_file(tmp_path, capsys):
    for path in lk.locked_files():
        target = tmp_path / path.relative_to(lk.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    lock = tmp_path / "LOCK.sha256"
    assert lk.main(["record"], tmp_path, lock) == 0
    assert lock.read_text() == lk.LOCK.read_text()
    assert lk.main(["check"], tmp_path, lock) == 0
    table = tmp_path / "tests_torch/golden/tolerances.py"
    table.write_text(table.read_text().replace("5e-14", "5e-13"))
    (tmp_path / "tests_torch/golden/extra.json").write_text("{}")
    next((tmp_path / "tests_torch/golden/yamls").glob("*.yaml")).unlink()
    assert lk.main(["check"], tmp_path, lock) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == [
        "changed",
        "not in the lock",
        "missing",
    ]
    assert lines[0].endswith("tolerances.py")


def test_the_lock_command_needs_a_verb(tmp_path):
    assert lk.main([], tmp_path, tmp_path / "LOCK.sha256") == 2
