"""Root conftest: runs before ``tests/conftest.py`` imports TensorFlow.

TensorFlow reads its thread variables once, at the first import, so the thread
budget of the pytest-xdist workers has to be applied here, not after that import.
"""

from tests.thread_budget import apply_thread_budget

apply_thread_budget()
