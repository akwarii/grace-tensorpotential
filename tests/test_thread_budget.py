from __future__ import annotations

import os

import pytest

from .thread_budget import (
    THREAD_VARS,
    WORKER_COUNT_VAR,
    apply_thread_budget,
    threads_per_worker,
)


@pytest.mark.parametrize(
    ("cores", "workers", "expected"),
    [(14, 1, 14), (14, 2, 7), (14, 4, 3), (14, 6, 2), (14, 8, 1), (4, 8, 1), (1, 1, 1)],
)
def test_threads_per_worker_is_floor_division_with_a_minimum_of_one(
    cores: int, workers: int, expected: int
) -> None:
    assert threads_per_worker(cores, workers) == expected


@pytest.mark.parametrize(("cores", "workers"), [(0, 1), (4, 0), (-1, 2), (4, -3)])
def test_threads_per_worker_rejects_non_positive_arguments(
    cores: int, workers: int
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        threads_per_worker(cores, workers)


def test_serial_run_is_left_untouched() -> None:
    env: dict[str, str] = {}
    assert apply_thread_budget(env, cores=14) is None
    assert env == {}


def test_worker_gets_cores_over_workers_in_every_variable() -> None:
    env = {WORKER_COUNT_VAR: "4"}
    assert apply_thread_budget(env, cores=14) == 3
    assert {var: env[var] for var in THREAD_VARS} == dict.fromkeys(THREAD_VARS, "3")
    # the variables TensorFlow and the OpenMP runtime actually read
    for name in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        assert env[name] == "3"


def test_values_set_by_the_contributor_are_kept() -> None:
    env = {WORKER_COUNT_VAR: "2", "TF_NUM_INTRAOP_THREADS": "5"}
    apply_thread_budget(env, cores=14)
    assert env["TF_NUM_INTRAOP_THREADS"] == "5"
    assert env["OMP_NUM_THREADS"] == "7"


def test_default_core_count_comes_from_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.cpu_count is the hardware boundary: pin it so the expectation is exact.
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    env = {WORKER_COUNT_VAR: "2"}
    assert apply_thread_budget(env) == 4


def test_unknown_core_count_falls_back_to_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.cpu_count returns None on platforms that cannot tell.
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert apply_thread_budget({WORKER_COUNT_VAR: "1"}) == 1


def test_process_environment_is_used_when_no_mapping_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKER_COUNT_VAR, "2")
    for var in THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    assert apply_thread_budget(cores=10) == 5
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_malformed_worker_count_is_an_actionable_error() -> None:
    with pytest.raises(ValueError, match=r"PYTEST_XDIST_WORKER_COUNT.*'many'"):
        apply_thread_budget({WORKER_COUNT_VAR: "many"})


def test_xdist_worker_received_its_budget_before_tensorflow_started() -> None:
    # Only meaningful inside a pytest-xdist worker; the root conftest.py must have
    # applied the budget to this very process.
    raw = os.environ.get(WORKER_COUNT_VAR)
    if raw is None:
        pytest.skip("not running under pytest-xdist")
    expected = threads_per_worker(os.cpu_count() or 1, int(raw))
    for var in THREAD_VARS:
        if os.environ.get(var) != str(expected):
            pytest.skip(f"{var} was set by the caller: {os.environ.get(var)!r}")
    assert all(os.environ[var] == str(expected) for var in THREAD_VARS)
