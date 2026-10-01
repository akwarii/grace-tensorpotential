"""Thread budget of the parallel test run.

Imported by ``conftest.py`` before TensorFlow, so it must not import TensorFlow
(or anything that does). With ``pytest -n N`` every worker is a TensorFlow
process; left alone each one starts one intra-op thread per core and the workers
oversubscribe the machine.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

WORKER_COUNT_VAR = "PYTEST_XDIST_WORKER_COUNT"
"""Set by pytest-xdist in every worker process, absent in a serial run."""

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
"""Variables read by the BLAS/OpenMP runtimes and by TensorFlow at start-up."""


def threads_per_worker(cores: int, workers: int) -> int:
    """Return the number of threads each of ``workers`` processes may use.

    Parameters
    ----------
    cores : int
        Number of CPU cores of the machine.
    workers : int
        Number of concurrent test processes.

    Returns
    -------
    int
        ``cores // workers``, at least 1.

    Raises
    ------
    ValueError
        If ``cores`` or ``workers`` is smaller than 1.
    """
    if cores < 1 or workers < 1:
        msg = f"cores and workers must be at least 1, got cores={cores}, workers={workers}"
        raise ValueError(msg)
    return max(1, cores // workers)


def apply_thread_budget(
    env: MutableMapping[str, str] | None = None, cores: int | None = None
) -> int | None:
    """Cap the threads of this process when it is an xdist worker.

    Variables that are already set are kept, so a contributor can override the
    budget from the shell. Child processes (``gracemaker`` runs) inherit it.

    Parameters
    ----------
    env : MutableMapping[str, str], optional
        Environment to read and update; ``os.environ`` by default.
    cores : int, optional
        Number of cores; ``os.cpu_count()`` by default.

    Returns
    -------
    int or None
        The thread budget, or ``None`` when not running under xdist (nothing is
        changed then).

    Raises
    ------
    ValueError
        If ``PYTEST_XDIST_WORKER_COUNT`` is not a positive integer.
    """
    env = os.environ if env is None else env
    raw = env.get(WORKER_COUNT_VAR)
    if raw is None:
        return None
    try:
        workers = int(raw)
    except ValueError:
        msg = f"{WORKER_COUNT_VAR} must be an integer, got {raw!r}"
        raise ValueError(msg) from None
    budget = threads_per_worker(cores or os.cpu_count() or 1, workers)
    for var in THREAD_VARS:
        env.setdefault(var, str(budget))
    return budget
