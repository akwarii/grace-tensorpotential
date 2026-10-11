"""``tensorpotential.scripts.grace_dashboard``: discovery of fit folders, the Flask app and the command line (``main``).

The server is never started: ``flask.Flask.run`` is replaced by a recorder (it binds a network port, an external
boundary), and the routes are called through Flask's test client. Fit folders are created in ``tmp_path`` from the layout the
dashboard documents (``input.yaml`` and ``seed/<N>/`` with ``test_metrics.yaml``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

from tensorpotential.scripts import grace_dashboard
from tests import cli_help


def make_fit(
    folder: Path,
    seeds_with_metrics: tuple[int, ...] = (),
    seeds_without: tuple[int, ...] = (),
) -> Path:
    """A fit folder: ``input.yaml`` and one ``seed/<N>/`` directory per seed, with ``test_metrics.yaml`` for some."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.yaml").write_text("seed: 1\n")
    for seed in (*seeds_with_metrics, *seeds_without):
        (folder / "seed" / str(seed)).mkdir(parents=True, exist_ok=True)
    for seed in seeds_with_metrics:
        (folder / "seed" / str(seed) / "test_metrics.yaml").write_text("")
    return folder


@pytest.fixture
def served(monkeypatch) -> dict:
    """Record what ``app.run`` is asked to serve instead of binding a port (external boundary)."""
    calls: dict = {}

    def run(self, host=None, port=None, debug=None, **options):
        calls.update(app=self, host=host, port=port, debug=debug)

    monkeypatch.setattr(Flask, "run", run)
    return calls


def run_dashboard(*argv: str) -> tuple[int, str, str]:
    return cli_help.run_main(grace_dashboard.main, "grace_dashboard", list(argv))


# --- main ------------------------------------------------------------------------------------------------------------


def test_main_scans_the_base_directory_and_serves_the_app(tmp_path, served):
    make_fit(tmp_path / "fit_a", seeds_with_metrics=(1, 2))
    make_fit(tmp_path / "fit_b" / "inner", seeds_with_metrics=(1,), seeds_without=(2,))
    code, out, _ = run_dashboard("--base", str(tmp_path))
    assert code == 0
    assert f"[grace_dashboard] scanning {tmp_path}" in out
    assert "[grace_dashboard] discovered 2 fit folder(s), 3 seed(s) total" in out
    assert isinstance(served["app"], Flask)


def test_main_defaults_are_all_interfaces_port_5000_and_no_debug(
    tmp_path, served, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run_dashboard()
    assert code == 0
    assert (served["host"], served["port"], served["debug"]) == ("0.0.0.0", 5000, False)
    assert f"scanning {tmp_path}" in out
    assert "discovered 0 fit folder(s), 0 seed(s) total" in out


def test_main_passes_host_port_and_debug_to_the_server(tmp_path, served):
    run_dashboard(
        "--base", str(tmp_path), "--host", "127.0.0.1", "--port", "8123", "--debug"
    )
    assert (served["host"], served["port"], served["debug"]) == (
        "127.0.0.1",
        8123,
        True,
    )


def test_main_relative_base_is_made_absolute(tmp_path, served, monkeypatch):
    make_fit(tmp_path / "runs" / "fit", seeds_with_metrics=(3,))
    monkeypatch.chdir(tmp_path)
    _, out, _ = run_dashboard("--base", "runs")
    assert f"scanning {tmp_path / 'runs'}" in out
    assert "discovered 1 fit folder(s), 1 seed(s) total" in out


def test_main_stops_when_the_base_directory_does_not_exist(tmp_path, served):
    code, _, err = run_dashboard("--base", str(tmp_path / "absent"))
    assert code == 1
    assert f"base directory does not exist: {tmp_path / 'absent'}" in err
    assert "app" not in served


def test_main_rejects_a_non_integer_port(tmp_path, served):
    code, _, err = run_dashboard("--base", str(tmp_path), "--port", "http")
    assert code == 2
    assert "invalid int value: 'http'" in err


def test_main_target_epochs_reach_the_api(tmp_path, served):
    make_fit(tmp_path / "fit", seeds_with_metrics=(1,))
    run_dashboard("--base", str(tmp_path), "--target-epochs", "77")
    reply = served["app"].test_client().get("/api/data")
    assert reply.get_json()["target_epochs"] == 77


# --- the app ---------------------------------------------------------------------------------------------------------


def test_index_names_the_base_directory(tmp_path):
    app = grace_dashboard.create_app(str(tmp_path / "results"))
    reply = app.test_client().get("/")
    assert reply.status_code == 200
    assert b"results" in reply.data


def test_api_data_describes_the_base_and_the_metric_keys(tmp_path):
    base = tmp_path / "results"
    base.mkdir()
    payload = (
        grace_dashboard
        .create_app(str(base), target_epochs=50)
        .test_client()
        .get("/api/data")
        .get_json()
    )
    assert payload["base"] == str(base)
    assert payload["base_name"] == "results"
    assert payload["target_epochs"] == 50
    assert payload["experiments"] == []
    assert payload["common_metric_keys"] == grace_dashboard.COMMON_METRIC_KEYS


def test_api_data_target_query_overrides_the_default(tmp_path):
    client = grace_dashboard.create_app(str(tmp_path), target_epochs=50).test_client()
    assert client.get("/api/data?target=120").get_json()["target_epochs"] == 120
    assert client.get("/api/data?target=0").get_json()["target_epochs"] == 50
    assert client.get("/api/data").get_json()["target_epochs"] == 50


def test_a_fit_without_readable_metrics_is_skipped_not_fatal(tmp_path, capsys):
    make_fit(tmp_path / "fit", seeds_with_metrics=(1,))
    payload = grace_dashboard.build_payload(str(tmp_path))
    assert payload["experiments"] == []


# --- discovery -------------------------------------------------------------------------------------------------------


def test_discover_fits_lists_seeds_that_have_test_metrics(tmp_path):
    make_fit(tmp_path / "a", seeds_with_metrics=(2, 10), seeds_without=(5,))
    assert grace_dashboard.discover_fits(str(tmp_path)) == [
        {"fit_folder": str(tmp_path / "a"), "seeds": [2, 10]}
    ]


def test_discover_fits_needs_input_yaml_and_a_numeric_seed_directory(tmp_path):
    no_input = tmp_path / "no_input"
    (no_input / "seed" / "1").mkdir(parents=True)
    no_seed_dir = tmp_path / "no_seed_dir"
    no_seed_dir.mkdir()
    (no_seed_dir / "input.yaml").write_text("")
    non_numeric = make_fit(tmp_path / "non_numeric")
    (non_numeric / "seed" / "best").mkdir(parents=True)
    assert grace_dashboard.discover_fits(str(tmp_path)) == []


def test_discover_fits_does_not_descend_into_a_fit_folder(tmp_path):
    outer = make_fit(tmp_path / "outer", seeds_with_metrics=(1,))
    make_fit(outer / "nested", seeds_with_metrics=(1,))
    assert [f["fit_folder"] for f in grace_dashboard.discover_fits(str(tmp_path))] == [
        str(outer)
    ]


def test_a_just_started_fit_is_discovered_with_an_empty_seed_list(tmp_path):
    make_fit(tmp_path / "new", seeds_without=(1,))
    assert grace_dashboard.discover_fits(str(tmp_path)) == [
        {"fit_folder": str(tmp_path / "new"), "seeds": []}
    ]


def test_compute_fit_names_strips_the_common_prefix(tmp_path):
    folders = [str(tmp_path / "exp" / "lr1" / "a"), str(tmp_path / "exp" / "lr2" / "a")]
    names = grace_dashboard.compute_fit_names(folders, str(tmp_path))
    assert names == {
        folders[0]: os.path.join("lr1", "a"),
        folders[1]: os.path.join("lr2", "a"),
    }


def test_compute_fit_names_for_a_single_fit_is_its_path_under_the_base(tmp_path):
    folder = str(tmp_path / "exp" / "fit")
    assert grace_dashboard.compute_fit_names([folder], str(tmp_path)) == {
        folder: os.path.join("exp", "fit")
    }


def test_compute_fit_names_without_folders_is_empty(tmp_path):
    assert grace_dashboard.compute_fit_names([], str(tmp_path)) == {}


def test_compute_fit_names_for_the_base_itself_uses_its_own_name(tmp_path):
    assert grace_dashboard.compute_fit_names([str(tmp_path)], str(tmp_path)) == {
        str(tmp_path): "."
    }


# --- log and metric file parsers ------------------------------------------------------------------------------------------


def write_metrics(path: Path, rows: list[dict]) -> None:
    """gracemaker's per-epoch metric files: one JSON mapping per line, written as YAML list items."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"- {json.dumps(row)}\n" for row in rows))


def power_law(epoch: float, a: float = 0.1, b: float = 1.0, m: float = -0.7) -> float:
    return a + b * epoch**m


def test_load_yaml_lines_reads_json_mappings_and_skips_everything_else(tmp_path):
    path = tmp_path / "metrics.yaml"
    path.write_text(
        '# comment\n- {"epoch": 1, "x": 2.5}\nnot json {broken\n\n- {"epoch": 2}\n'
    )
    assert grace_dashboard.load_yaml_lines(str(path)) == [
        {"epoch": 1, "x": 2.5},
        {"epoch": 2},
    ]


def test_extract_n_params_reads_the_first_parameter_count_with_thousands_separators(
    tmp_path,
):
    log = tmp_path / "log.txt"
    log.write_text(
        "start\nNumber of trainable parameters: 1,234,567\nNumber of trainable parameters: 9\n"
    )
    assert grace_dashboard.extract_n_params(str(log)) == 1234567


def test_extract_n_params_is_none_without_a_file_a_count_or_readable_text(tmp_path):
    assert grace_dashboard.extract_n_params(str(tmp_path / "absent.txt")) is None
    plain = tmp_path / "plain.txt"
    plain.write_text("nothing to see\n")
    assert grace_dashboard.extract_n_params(str(plain)) is None
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x80 Number of trainable parameters: 5\n")
    assert grace_dashboard.extract_n_params(str(binary)) is None


def test_extract_times_takes_the_last_line_after_the_first_iteration(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "Iteration #1/50 Time(mcs/at): 900.0 (800.0)\n"
        "Iteration #2/50 Time(mcs/at): 12.5 (6.25)\n"
        "Iteration #3/50 Time(mcs/at): 11 (5)\n"
        "no timing here\n"
    )
    assert grace_dashboard.extract_times_mcs_atom(str(log)) == (11.0, 5.0)


def test_extract_times_is_none_without_steady_state_lines_or_a_readable_file(tmp_path):
    assert grace_dashboard.extract_times_mcs_atom(str(tmp_path / "absent.txt")) == (
        None,
        None,
    )
    only_first = tmp_path / "first.txt"
    only_first.write_text("Iteration #1/50 Time(mcs/at): 900.0 (800.0)\n")
    assert grace_dashboard.extract_times_mcs_atom(str(only_first)) == (None, None)
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x80\n")
    assert grace_dashboard.extract_times_mcs_atom(str(binary)) == (None, None)


def test_epoch_timing_is_seconds_between_the_first_and_last_iteration_per_epoch(
    tmp_path,
):
    log = tmp_path / "log.txt"
    log.write_text(
        "2024-05-01 10:00:00 I Iteration #2/50 something\n"
        "2024-05-01 10:00:30 I Iteration #3/50 something\n"
        "2024-05-01 10:02:00 I Iteration #6/50 something\n"
    )
    assert grace_dashboard.extract_epoch_timing(str(log)) == pytest.approx(120.0 / 4)


def test_epoch_timing_is_none_for_a_single_epoch_missing_timestamps_or_a_missing_file(
    tmp_path,
):
    assert grace_dashboard.extract_epoch_timing(str(tmp_path / "absent.txt")) is None
    single = tmp_path / "single.txt"
    single.write_text(
        "2024-05-01 10:00:00 I Iteration #2/50 x\n2024-05-01 10:00:05 I Iteration #2/50 x\n"
    )
    assert grace_dashboard.extract_epoch_timing(str(single)) is None
    no_stamp = tmp_path / "nostamp.txt"
    no_stamp.write_text("Iteration #2/50\nIteration #9/50\n")
    assert grace_dashboard.extract_epoch_timing(str(no_stamp)) is None
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x80\n")
    assert grace_dashboard.extract_epoch_timing(str(binary)) is None


def test_extrapolation_follows_the_power_law_it_was_fitted_to():
    rows = [{"epoch": e, "mae": power_law(e)} for e in range(1, 41)]
    epochs, values, std = grace_dashboard.extrapolate_series(rows, "mae", 80)
    assert epochs == [40, *range(41, 81)]
    np.testing.assert_allclose(values[0], power_law(40), rtol=1e-12)
    np.testing.assert_allclose(
        values[1:], [power_law(e) for e in range(41, 81)], rtol=1e-4
    )
    assert std is None or std > 0


def test_extrapolation_needs_enough_positive_points_and_a_target_beyond_the_last_epoch():
    rows = [{"epoch": e, "mae": power_law(e)} for e in range(1, 9)]
    assert (
        grace_dashboard.extrapolate_series(rows, "mae", 80) is None
    )  # fewer than 10 points
    rows = [{"epoch": e, "mae": power_law(e)} for e in range(1, 41)]
    assert (
        grace_dashboard.extrapolate_series(rows, "mae", 40) is None
    )  # already at the target
    assert grace_dashboard.extrapolate_series(rows, "missing_key", 80) is None
    with_gaps = [
        {"epoch": e, "mae": None if e % 2 else power_law(e)} for e in range(1, 41)
    ]
    assert (
        grace_dashboard.extrapolate_series(with_gaps, "mae", 80)[0][0] == 40
    )  # the 20 usable points are enough
    ten = [{"epoch": e, "mae": None if e % 4 else power_law(e)} for e in range(1, 41)]
    assert (
        grace_dashboard.extrapolate_series(ten, "mae", 80) is not None
    )  # exactly 10 usable points: enough
    eight = [{"epoch": e, "mae": None if e % 5 else power_law(e)} for e in range(1, 41)]
    assert (
        grace_dashboard.extrapolate_series(eight, "mae", 80) is None
    )  # 8 usable points: too few


def test_extrapolation_ignores_zero_epochs_and_non_positive_values():
    rows = [
        {"epoch": 0, "mae": 5.0},
        *({"epoch": e, "mae": power_law(e)} for e in range(1, 31)),
        {"epoch": 31, "mae": -1.0},
    ]
    result = grace_dashboard.extrapolate_series(rows, "mae", 40)
    assert result is not None
    assert result[0][0] == 30


def test_extrapolation_gives_up_when_the_fit_fails():
    rows = [{"epoch": e, "mae": power_law(e)} for e in range(1, 31)]
    rows[5]["mae"] = float("inf")
    assert grace_dashboard.extrapolate_series(rows, "mae", 60) is None


def test_extrapolated_values_are_clipped_to_a_band_around_the_observed_ones():
    rows = [{"epoch": e, "mae": power_law(e)} for e in range(1, 41)]
    _, values, _ = grace_dashboard.extrapolate_series(rows, "mae", 5000)
    smallest = min(r["mae"] for r in rows)
    assert min(values[1:]) >= 0.3 * smallest - 1e-12
    assert max(values[1:]) <= 1.5 * rows[-1]["mae"] + 1e-12


def test_collect_seed_and_build_payload_describe_a_running_fit(tmp_path):
    fit = make_fit(tmp_path / "fit", seeds_with_metrics=(1,))
    seed = fit / "seed" / "1"
    test_rows = [
        {
            "epoch": e,
            "step": 10 * e,
            "total_loss/test": power_law(e),
            "mae/depa": power_law(e),
            "lr_epoch_begin": 0.01,
        }
        for e in range(1, 21)
    ]
    train_rows = [
        {
            "epoch": e,
            "step": 10 * e,
            "total_loss/train": power_law(e),
            "mae/depa": 0.5 * power_law(e),
        }
        for e in range(1, 21)
    ]
    write_metrics(seed / "test_metrics.yaml", test_rows)
    write_metrics(seed / "train_metrics.yaml", train_rows)
    (seed / "log.txt").write_text(
        "Number of trainable parameters: 4,321\nIteration #2/160 Time(mcs/at): 7.5 (3.5)\n"
    )
    payload = grace_dashboard.build_payload(str(tmp_path), target_epochs=40)
    (experiment,) = payload["experiments"]
    assert experiment["n_params"] == 4321
    assert (experiment["train_us_per_atom"], experiment["test_us_per_atom"]) == (
        7.5,
        3.5,
    )
    assert experiment["n_epochs"] == 20
    assert experiment["complete"] is False
    assert experiment["best_epoch"] == 20
    np.testing.assert_allclose(experiment["best_loss"], power_law(20), rtol=1e-12)
    np.testing.assert_allclose(
        experiment["best"]["train/mae/depa"], 0.5 * power_law(20), rtol=1e-12
    )
    assert experiment["state"] == "running"
    assert experiment["test"]["epoch"] == list(range(1, 21))
    assert experiment["extrap_test"]["epoch_mae/depa"][-1] == 40
    np.testing.assert_allclose(
        experiment["extrap_best"]["mae/depa"], power_law(40), rtol=1e-3
    )


def test_a_complete_fit_has_no_extrapolation_and_several_seeds_get_a_suffix(tmp_path):
    fit = make_fit(tmp_path / "fit", seeds_with_metrics=(1, 2))
    for seed in (1, 2):
        rows = [{"epoch": e, "total_loss/test": 1.0 / e} for e in range(1, 13)]
        write_metrics(fit / "seed" / str(seed) / "test_metrics.yaml", rows)
    payload = grace_dashboard.build_payload(str(tmp_path), target_epochs=12)
    assert [e["name"] for e in payload["experiments"]] == [
        "fit (seed=1)",
        "fit (seed=2)",
    ]
    assert all(
        e["complete"] and e["extrap_test"] is None for e in payload["experiments"]
    )


def test_a_seed_with_an_unusable_metrics_file_is_skipped_with_a_warning(
    tmp_path, capsys
):
    make_fit(tmp_path / "fit", seeds_with_metrics=(1,))
    (tmp_path / "fit" / "seed" / "1" / "test_metrics.yaml").write_bytes(
        b"\xff\xfe\x00\x80\n"
    )
    payload = grace_dashboard.build_payload(str(tmp_path))
    assert payload["experiments"] == []
    assert "[warn] skipping" in capsys.readouterr().out


def test_main_takes_argv_returns_zero_and_does_not_read_sys_argv(tmp_path, served, monkeypatch):
    cli_help.poison_sys_argv(monkeypatch)
    assert grace_dashboard.main(["--base", str(tmp_path), "--port", "8124"]) == 0
    assert served["port"] == 8124


def test_main_without_argv_reads_sys_argv(tmp_path, served, monkeypatch):
    monkeypatch.setattr("sys.argv", ["grace_dashboard", "--base", str(tmp_path), "--port", "8125"])
    assert grace_dashboard.main() == 0
    assert served["port"] == 8125
