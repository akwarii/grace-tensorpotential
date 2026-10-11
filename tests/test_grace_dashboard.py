"""``tensorpotential.scripts.grace_dashboard``: discovery of fit folders, the Flask app and the command line (``main``).

The server is never started: ``flask.Flask.run`` is replaced by a recorder (it binds a network port, an external
boundary), and the routes are called through Flask's test client. Fit folders are created in ``tmp_path`` from the layout the
dashboard documents (``input.yaml`` and ``seed/<N>/`` with ``test_metrics.yaml``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from flask import Flask

from tensorpotential.scripts import grace_dashboard
from tests import cli_help


def make_fit(folder: Path, seeds_with_metrics: tuple[int, ...] = (), seeds_without: tuple[int, ...] = ()) -> Path:
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


def test_main_defaults_are_all_interfaces_port_5000_and_no_debug(tmp_path, served, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run_dashboard()
    assert code == 0
    assert (served["host"], served["port"], served["debug"]) == ("0.0.0.0", 5000, False)  # noqa: S104
    assert f"scanning {tmp_path}" in out
    assert "discovered 0 fit folder(s), 0 seed(s) total" in out


def test_main_passes_host_port_and_debug_to_the_server(tmp_path, served):
    run_dashboard("--base", str(tmp_path), "--host", "127.0.0.1", "--port", "8123", "--debug")
    assert (served["host"], served["port"], served["debug"]) == ("127.0.0.1", 8123, True)


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
    payload = grace_dashboard.create_app(str(base), target_epochs=50).test_client().get("/api/data").get_json()
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
    assert grace_dashboard.discover_fits(str(tmp_path)) == [{"fit_folder": str(tmp_path / "a"), "seeds": [2, 10]}]


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
    assert [f["fit_folder"] for f in grace_dashboard.discover_fits(str(tmp_path))] == [str(outer)]


def test_a_just_started_fit_is_discovered_with_an_empty_seed_list(tmp_path):
    make_fit(tmp_path / "new", seeds_without=(1,))
    assert grace_dashboard.discover_fits(str(tmp_path)) == [{"fit_folder": str(tmp_path / "new"), "seeds": []}]


def test_compute_fit_names_strips_the_common_prefix(tmp_path):
    folders = [str(tmp_path / "exp" / "lr1" / "a"), str(tmp_path / "exp" / "lr2" / "a")]
    names = grace_dashboard.compute_fit_names(folders, str(tmp_path))
    assert names == {folders[0]: os.path.join("lr1", "a"), folders[1]: os.path.join("lr2", "a")}


def test_compute_fit_names_for_a_single_fit_is_its_path_under_the_base(tmp_path):
    folder = str(tmp_path / "exp" / "fit")
    assert grace_dashboard.compute_fit_names([folder], str(tmp_path)) == {folder: os.path.join("exp", "fit")}


def test_compute_fit_names_without_folders_is_empty(tmp_path):
    assert grace_dashboard.compute_fit_names([], str(tmp_path)) == {}


def test_compute_fit_names_for_the_base_itself_uses_its_own_name(tmp_path):
    assert grace_dashboard.compute_fit_names([str(tmp_path)], str(tmp_path)) == {str(tmp_path): "."}
