"""Configuration of the torch-backend tests; imports neither TensorFlow nor anything that does.

The tests need the ``torch`` extra and no TensorFlow (CI's ``test-torch`` job has exactly that). The markers and
the capability contract are in :mod:`tests_torch.capabilities`: a capability marker skips a test where the
capability is missing, ``GRACE_REQUIRE_CAPS`` turns that into a failure of the whole run, and a test parametrised
over an empty list (for example "every available backend" with none available) fails the collection instead of
being skipped.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest

from tests_torch import capabilities as caps

HERE = Path(__file__).resolve().parent


@functools.cache
def _available(name: str) -> bool:
    return caps.CAPABILITIES[name].probe()


def pytest_configure(config: pytest.Config) -> None:
    for name, capability in caps.CAPABILITIES.items():
        config.addinivalue_line("markers", f"{name}: needs {capability.description}")
    for name, description in caps.OTHER_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    try:
        guaranteed = caps.required(os.environ)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from None
    lacking = caps.missing(guaranteed, _available)
    if lacking:
        raise pytest.UsageError(caps.require_message(lacking))


def _empty_parameter_set(item: pytest.Item) -> bool:
    # pytest marks such a test with skip(reason="got empty parameter set ..."), always as a keyword
    return any(
        caps.is_empty_parameter_set(m.kwargs.get("reason", ""))
        for m in item.iter_markers("skip")
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    ours = [item for item in items if item.path.is_relative_to(HERE)]
    empty = [item.nodeid for item in ours if _empty_parameter_set(item)]
    if empty:
        msg = "parametrised over an empty list (nothing would run): " + ", ".join(empty)
        raise pytest.UsageError(msg)
    for item in ours:
        lacking = caps.missing((m.name for m in item.iter_markers()), _available)
        if lacking:
            item.add_marker(pytest.mark.skip(reason=caps.skip_reason(lacking)))
