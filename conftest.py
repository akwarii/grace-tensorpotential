"""Root conftest: runs before ``tests/conftest.py`` imports TensorFlow.

TensorFlow reads its thread variables once, at the first import, so the thread
budget of the pytest-xdist workers has to be applied here, not after that import.
"""

from tests.slow_marks import apply_slow_marks, deal_slow_first, load_slow_ids
from tests.thread_budget import apply_thread_budget

apply_thread_budget()


def pytest_addoption(parser):
    parser.addoption(
        "--slow-first",
        choices=("deal", "off"),
        default="deal",
        help="with pytest-xdist: deal the tests of tests/slow_tests.txt over the first "
        "block of each worker, longest first ('off' keeps the collection order)",
    )


def pytest_collection_modifyitems(config, items):
    """Mark the tests listed in ``tests/slow_tests.txt`` as ``slow`` and start them first."""
    slow_ids = load_slow_ids()
    apply_slow_marks(items, slow_ids)
    # Only xdist workers collect; every one of them must compute the same order.
    workers = getattr(config, "workerinput", {}).get("workercount", 1)
    if config.getoption("--slow-first") == "deal":
        items[:] = deal_slow_first(items, slow_ids, workers)
