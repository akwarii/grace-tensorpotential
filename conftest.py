"""Root conftest: runs before ``tests/conftest.py`` imports TensorFlow.

TensorFlow reads its thread variables once, at the first import, so the thread
budget of the pytest-xdist workers has to be applied here, not after that import.
"""

from tests.slow_first import deal_slow_first
from tests.thread_budget import apply_thread_budget

apply_thread_budget()


def pytest_addoption(parser):
    parser.addoption(
        "--slow-first",
        choices=("deal", "off"),
        default="deal",
        help="with pytest-xdist: deal the tests marked slow over the first block of each "
        "worker ('off' keeps the collection order)",
    )


def pytest_collection_modifyitems(config, items):
    """Start the tests marked ``slow`` first (see ``tests/slow_first.py``)."""
    if config.getoption("--slow-first") == "off":
        return
    # Only xdist workers collect; every one of them must compute the same order.
    workers = getattr(config, "workerinput", {}).get("workercount", 1)
    items[:] = deal_slow_first(
        items, lambda item: item.get_closest_marker("slow") is not None, workers
    )
