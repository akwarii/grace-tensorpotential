"""Root conftest: runs before ``tests/conftest.py`` imports TensorFlow.

TensorFlow reads its thread variables once, at the first import, so the thread
budget of the pytest-xdist workers has to be applied here, not after that import.
"""

from tests.slow_marks import apply_slow_marks, load_slow_ids
from tests.thread_budget import apply_thread_budget

apply_thread_budget()


def pytest_collection_modifyitems(items):
    """Mark the tests listed in ``tests/slow_tests.txt`` as ``slow``."""
    apply_slow_marks(items, load_slow_ids())
