"""Tests for ``tensorpotential.core.backends`` (``require_backend`` and the install hint).

Stub packages written into ``tmp_path`` stand in for the installed backends: a directory named
``tensorflow`` whose ``__init__`` raises proves that the check finds a package without importing it.
Each case runs in a fresh interpreter whose ``sys.path`` and finders are controlled, so the result
does not depend on what the test environment has installed.

Logic layer: the package list of each backend, one missing package among several, an unknown
backend, the typed error with its attributes. Behaviour layer: the message names the missing
packages, the extra and what asked for the backend, a package that the finders refuse counts
as "not installed".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorpotential.core.backends import (
    BACKEND_PACKAGES,
    BackendNotInstalledError,
    backend_not_installed,
    missing_packages,
    require_backend,
)
from tests.fresh_python import run_fresh_python

STUB = 'raise RuntimeError("the stub must not be imported")\n'

PROBE = """
import json, sys

class Refuse:
    # the environment of the test may have the real backends: whatever is not "installed" is refused
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in sys.argv[3].split(","):
            raise ImportError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Refuse())
from tensorpotential.core.backends import BackendNotInstalledError, missing_packages, require_backend

backend, needed_by = sys.argv[1], sys.argv[2] or None
out = {"missing": list(missing_packages(backend))}
try:
    require_backend(backend, needed_by=needed_by)
except BackendNotInstalledError as exc:
    out.update(message=str(exc), name=exc.name, backend=exc.backend, missing_attr=list(exc.missing))
    out["is_import_error"] = isinstance(exc, ImportError)
print(json.dumps(out))
"""


def stub_site(tmp_path: Path, packages: tuple[str, ...]) -> Path:
    """A directory holding one stub package per name; put it on ``PYTHONPATH``."""
    site = tmp_path / "site"
    for package in packages:
        (site / package).mkdir(parents=True, exist_ok=True)
        (site / package / "__init__.py").write_text(STUB)
    return site


def probe(
    tmp_path: Path, backend: str, installed: tuple[str, ...], needed_by: str = ""
) -> dict:
    site = stub_site(tmp_path, installed)
    refused = sorted(
        {p for ps in BACKEND_PACKAGES.values() for p in ps} - set(installed)
    )
    result = run_fresh_python(
        PROBE, tmp_path, args=[backend, needed_by, ",".join(refused)], pythonpath=[site]
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_each_backend_names_the_packages_its_extra_installs() -> None:
    assert BACKEND_PACKAGES["tf"] == ("tensorflow", "tf_keras")
    assert BACKEND_PACKAGES["torch"] == ("torch",)
    assert BACKEND_PACKAGES["torch-sim"] == ("torch", "torch_sim", "vesin")


def test_an_unknown_backend_is_a_value_error_naming_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown backend 'jax'.*tf, torch, torch-sim"):
        require_backend("jax")
    with pytest.raises(ValueError, match="unknown backend"):
        missing_packages("")


def test_installed_packages_are_found_without_being_imported(tmp_path: Path) -> None:
    out = probe(tmp_path, "tf", ("tensorflow", "tf_keras"))
    assert out == {"missing": []}


def test_one_missing_package_is_enough_to_fail(tmp_path: Path) -> None:
    out = probe(tmp_path, "tf", ("tensorflow",))
    assert out["missing"] == ["tf_keras"]
    assert out["missing_attr"] == ["tf_keras"]
    assert out["name"] == "tf_keras"


def test_nothing_installed_lists_every_package_in_table_order(tmp_path: Path) -> None:
    out = probe(tmp_path, "torch-sim", ())
    assert out["missing"] == ["torch", "torch_sim", "vesin"]
    assert out["backend"] == "torch-sim"
    assert out["is_import_error"] is True


def test_torch_sim_needs_torch_too(tmp_path: Path) -> None:
    out = probe(tmp_path, "torch-sim", ("torch_sim", "vesin"))
    assert out["missing"] == ["torch"]


def test_the_message_names_the_missing_packages_the_extra_and_the_caller(
    tmp_path: Path,
) -> None:
    message = probe(tmp_path, "tf", (), needed_by="grace_utils")["message"]
    assert message.startswith("grace_utils needs TensorFlow, which is not installed")
    assert "missing: tensorflow, tf_keras" in message
    assert "pip install 'tensorpotential[tf]'" in message
    assert "'tensorpotential.core', still work" in message


def test_the_message_without_a_caller_still_names_the_backend(tmp_path: Path) -> None:
    message = probe(tmp_path, "torch", ())["message"]
    assert message.startswith("This needs PyTorch, which is not installed")
    assert "pip install 'tensorpotential[torch]'" in message


def test_the_error_type_is_an_import_error_with_the_backend_attributes() -> None:
    error = BackendNotInstalledError("tf", ("tensorflow",), "text")
    assert isinstance(error, ImportError)
    assert (error.backend, error.missing, error.name, str(error)) == (
        "tf",
        ("tensorflow",),
        "tensorflow",
        "text",
    )


def test_backend_not_installed_builds_the_error_without_checking_anything() -> None:
    error = backend_not_installed("torch-sim", ("vesin",), "TorchSimModel")
    assert isinstance(error, BackendNotInstalledError)
    assert (error.backend, error.missing, error.name) == (
        "torch-sim",
        ("vesin",),
        "vesin",
    )
    assert str(error).startswith(
        "TorchSimModel needs torch-sim, which is not installed"
    )
    assert "pip install 'tensorpotential[torch-sim]'" in str(error)
