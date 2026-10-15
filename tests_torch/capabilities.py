"""Capability contract of ``tests_torch``: what a test needs beyond the torch extra, and what a CI job guarantees.

A test (or one parameter of it, ``pytest.param(..., marks=pytest.mark.vesin)``) marked with a capability runs where
the capability is available and is skipped elsewhere, with the reason. A job that exports ``GRACE_REQUIRE_CAPS``
(comma-separated names, e.g. ``torchsim,vesin``) guarantees those capabilities: the run stops at its start when one
is missing or the variable names an unknown capability, so a test that needs it can never become a skip there.

The probes import nothing heavy: a package is looked up with ``importlib.util.find_spec`` (never imported, so
TensorFlow is not loaded next to torch), the GPU is asked of torch, and the HPC is the job's own declaration.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Final

REQUIRE_VARIABLE: Final = "GRACE_REQUIRE_CAPS"
HPC_VARIABLE: Final = "GRACE_HPC"


def _has_package(name: str) -> Callable[[], bool]:
    return lambda: importlib.util.find_spec(name) is not None


def _has_gpu() -> bool:
    import torch  # only a test that needs a GPU pays for the import

    return torch.cuda.is_available()


def _on_hpc() -> bool:
    return os.environ.get(HPC_VARIABLE) == "1"


CAPABILITIES: Final[Mapping[str, tuple[str, Callable[[], bool]]]] = {
    "tf": (
        "TensorFlow is installed (a comparison with the TF code itself)",
        _has_package("tensorflow"),
    ),
    "torchsim": (
        "torch-sim is installed (the torch_sim package)",
        _has_package("torch_sim"),
    ),
    "vesin": ("the vesin neighbour list is installed", _has_package("vesin")),
    "gpu": ("torch sees a CUDA device", _has_gpu),
    "cueq": (
        "cuEquivariance for torch is installed",
        _has_package("cuequivariance_torch"),
    ),
    "hpc": (
        f"the job runs on the HPC with the foundation-model weights ({HPC_VARIABLE}=1)",
        _on_hpc,
    ),
}

OTHER_MARKERS: Final[Mapping[str, str]] = {
    "golden": "compares with the golden fixtures of tests_torch/golden or tests_torch/fixtures",
    "slow": "a test of 30 s or more (leave out with -m 'not slow')",
}


def required(environ: Mapping[str, str]) -> frozenset[str]:
    """Return the capabilities a job guarantees through ``GRACE_REQUIRE_CAPS``.

    Raises
    ------
    ValueError
        When the variable names a capability that does not exist.
    """
    names = frozenset(
        n.strip() for n in environ.get(REQUIRE_VARIABLE, "").split(",") if n.strip()
    )
    unknown = sorted(names - CAPABILITIES.keys())
    if unknown:
        msg = (
            f"{REQUIRE_VARIABLE} names unknown capabilities {', '.join(unknown)}; "
            f"the known ones are {', '.join(CAPABILITIES)}"
        )
        raise ValueError(msg)
    return names


def missing(names: Iterable[str], available: Callable[[str], bool]) -> list[str]:
    """Return the names of ``names`` that are capabilities and not available, sorted."""
    return sorted(n for n in set(names) if n in CAPABILITIES and not available(n))


def skip_reason(names: Iterable[str]) -> str:
    """Return the reason a test that needs the capabilities ``names`` is skipped."""
    return "needs " + "; ".join(f"{n}: {CAPABILITIES[n][0]}" for n in sorted(names))


def require_message(names: Iterable[str]) -> str:
    """Return the message that stops a run whose job guarantees the capabilities ``names`` and lacks them."""
    return (
        f"{REQUIRE_VARIABLE} guarantees {', '.join(sorted(names))}, which this environment lacks; "
        "install them or remove them from the variable of the job"
    )


def is_empty_parameter_set(reason: str) -> bool:
    """Tell whether a skip ``reason`` is pytest's for a parametrisation over an empty list."""
    return reason.startswith("got empty parameter set")
