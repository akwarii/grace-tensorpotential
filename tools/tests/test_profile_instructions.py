"""Tests for tools/profile_instructions.py.

Logic tests use hand-made rows. Value tests run the tool on the 1-layer test model and compare with
eager TensorFlow, which does not go through the jitted functions of the tool: the forward output of
every instruction, the gradient of one instruction against a central difference, the bond count
against the neighbour list of the structure, and the dtype of the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import profile_instructions as pi  # noqa: E402

pytest.importorskip("tensorpotential")
tf = pytest.importorskip("tensorflow")

YAML = "model_grace.yaml"


@pytest.fixture(scope="module")
def setup():
    model = pi.build_model(tf, YAML, "float64", "none")
    atoms = pi.structure(1)
    data = pi.input_batch(tf, model, atoms)
    return model, atoms, data


def _row(name, fwd, both, error=None):
    return {
        "name": name,
        "class": "C",
        "fwd_ms": fwd,
        "fwd_bwd_ms": both,
        "error": error,
    }


# ------------------------------------------------------------------ logic


def test_shares_are_percentages_of_the_summed_forward_backward_times():
    rows = [_row("a", 1.0, 30.0), _row("b", 2.0, 10.0), _row("c", 1.0, None, "boom")]

    pi.add_shares(rows)

    assert [r["share"] for r in rows] == [75.0, 25.0, None]


def test_shares_of_rows_without_any_time_are_none():
    rows = [_row("a", None, None, "x")]

    pi.add_shares(rows)

    assert rows[0]["share"] is None


def test_render_orders_by_forward_backward_time_and_totals_the_isolated_times():
    rows = [
        _row("small", 1.0, 2.0),
        _row("big", 5.0, 40.0),
        _row("bad", 3.0, None, "e"),
    ]
    pi.add_shares(rows)

    lines = pi.render(rows, whole_ms=77.0).splitlines()

    assert lines[2].startswith("| big |")
    assert lines[3].startswith("| small |")
    assert lines[4] == "| bad | C | 3.00 | n/a | n/a |"
    assert lines[5] == "| sum of isolated | | 9.00 | 42.00 | |"
    assert lines[6] == "| whole model, fused (energy, forces, virial) | | | 77.00 | |"


def test_render_top_keeps_the_largest_rows_but_totals_all_of_them():
    rows = [_row("a", 1.0, 1.0), _row("b", 1.0, 5.0), _row("c", 1.0, 3.0)]
    pi.add_shares(rows)

    lines = pi.render(rows, 1.0, top=1).splitlines()

    assert [ln.split("|")[1].strip() for ln in lines[2:-2]] == ["b"]
    assert lines[-2] == "| sum of isolated | | 3.00 | 9.00 | |"


def test_a_gpu_run_without_a_gpu_is_refused(monkeypatch):
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda _kind=None: [])

    with pytest.raises(RuntimeError, match="sees no GPU"):
        pi.configure_tensorflow("gpu")


def test_the_cpu_device_hides_every_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    pi.configure_tensorflow("cpu")

    assert pi.os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_environment_names_the_device_and_versions():
    info = pi.environment(tf, "cpu")

    assert info["device"] == "cpu"
    assert info["tensorflow"] == tf.__version__
    assert "gpu_name" not in info


def test_environment_of_a_gpu_run_reads_the_device_details(monkeypatch):
    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda _kind=None: ["GPU:0"]
    )
    monkeypatch.setattr(
        tf.config.experimental,
        "get_device_details",
        lambda _gpu: {"device_name": "A100", "compute_capability": (8, 0)},
    )

    info = pi.environment(tf, "gpu")

    assert (
        info["gpu_name"],
        info["gpu_compute_capability"],
        info["n_gpus_visible"],
    ) == ("A100", [8, 0], 1)


def test_an_instruction_xla_cannot_compile_becomes_an_error_row():
    def not_compilable():
        return tf.py_function(lambda: tf.constant(1.0), [], tf.float32)

    ms, error = pi.timed_or_error(tf, not_compilable, repeats=1)

    assert ms is None
    assert error


def test_median_ms_calls_the_function_once_to_compile_then_repeats_times():
    calls = []

    def fn():
        calls.append(1)
        return tf.constant(1.0)

    pi.median_ms(tf, fn, repeats=4)

    assert len(calls) == 5


def test_the_option_is_validated_by_the_command_line():
    with pytest.raises(SystemExit):
        pi.main([YAML, "--option", "nonsense"])


# ------------------------------------------------------------------ values


def test_the_input_batch_has_the_bonds_of_the_structure(setup):
    _, atoms, data = setup
    from tests.neighbour_oracle import brute_force_pairs

    pairs = brute_force_pairs(
        atoms.get_positions(), atoms.get_cell()[:], atoms.get_chemical_symbols(), 6.0
    )

    assert int(data["ind_i"].shape[0]) == len(pairs)


def test_every_instruction_gets_one_row_in_model_order(setup):
    model, _, data = setup

    rows = pi.profile_instructions(tf, model, data, repeats=1)

    import oracle_snapshot as osn

    names = [r["name"] for r in rows]
    assert names == [i.name for i in osn._as_list(model.instructions)]
    assert names[0] == "BondLength"
    assert all(r["fwd_ms"] > 0 for r in rows if r["fwd_ms"] is not None)


def test_the_jitted_forward_equals_the_eager_forward_of_every_instruction(setup):
    model, _, data = setup

    for ins, snapshot in pi.instruction_inputs(model, data):
        forward, _ = pi.instruction_functions(tf, ins, snapshot)
        jitted = tf.nest.flatten(tf.function(forward, jit_compile=True)())
        eager_data = dict(snapshot)
        ins(eager_data, training=False, local=False)
        eager = [eager_data[k] for k in sorted(set(eager_data) - set(snapshot))]
        assert len(jitted) == len(eager), ins.name
        for got, want in zip(jitted, eager, strict=True):
            np.testing.assert_allclose(
                got.numpy(), want.numpy(), rtol=1e-10, atol=1e-12, err_msg=ins.name
            )


def test_the_forward_backward_gradient_matches_a_central_difference(setup):
    model, _, data = setup
    ins, snapshot = next(
        p for p in pi.instruction_inputs(model, data) if p[0].name == "RadialBasis"
    )
    _, forward_backward = pi.instruction_functions(tf, ins, snapshot)
    base = snapshot["BondLength"]  # the float input of the basis

    def loss_at(lengths):
        eager = {**snapshot, "BondLength": lengths}
        ins(eager, training=False, local=False)
        return float(tf.reduce_sum(eager["RadialBasis"]))

    grads = [g.numpy() for g in forward_backward()]
    step = 1e-6
    index = 5
    bump = np.zeros(base.shape)
    bump[index] = step
    numeric = (loss_at(base + bump) - loss_at(base - bump)) / (2 * step)

    assert any(
        np.isclose(g.reshape(-1)[index], numeric, rtol=1e-5)
        for g in grads
        if g.size == base.size
    )


def test_profile_model_reports_the_layout_and_a_gpu_peak_only_on_a_gpu():
    args = pi.argparse.Namespace(
        dtype="float64",
        option="dense_nbr",
        supercell=1,
        repeats=1,
        device="cpu",
        grads="all",
    )

    result = pi.profile_model(args, tf, YAML)

    assert result["option"] == "dense_nbr"
    assert result["n_atoms"] == 54
    assert result["whole_ms"] > 0
    assert "gpu_peak_memory_mb" not in result
    assert sum(
        r["share"] for r in result["rows"] if r["share"] is not None
    ) == pytest.approx(100.0)


def test_main_prints_the_table_and_writes_the_json(tmp_path, capsys):
    out = tmp_path / "p.json"

    assert (
        pi.main([
            YAML,
            "--supercell",
            "1",
            "--repeats",
            "1",
            "--top",
            "3",
            "--json",
            str(out),
        ])
        == 0
    )

    text = capsys.readouterr().out
    assert "| instruction | class |" in text
    assert "model_grace.yaml: 54 atoms" in text
    payload = json.loads(out.read_text())
    assert payload["environment"]["device"] == "cpu"
    assert payload["profiles"][0]["rows"][0]["name"] == "BondLength"


def test_gradients_of_inputs_only_leave_out_the_variables(setup):
    model, _, data = setup
    ins, snapshot = next(
        p for p in pi.instruction_inputs(model, data) if p[0].name == "R"
    )
    n_vars = len(ins.variables)

    _, both = pi.instruction_functions(tf, ins, snapshot, with_variables=True)
    _, inputs_only = pi.instruction_functions(tf, ins, snapshot, with_variables=False)

    assert n_vars > 0
    assert len(both()) > len(inputs_only())


def test_main_accepts_the_grads_option_and_records_it(tmp_path):
    out = tmp_path / "g.json"

    pi.main([
        YAML,
        "--supercell",
        "1",
        "--repeats",
        "1",
        "--grads",
        "inputs",
        "--json",
        str(out),
    ])

    assert json.loads(out.read_text())["profiles"][0]["grads"] == "inputs"
