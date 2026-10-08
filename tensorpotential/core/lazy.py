"""Lazy public names for a package whose submodules need TensorFlow (PEP 562).

``tensorpotential`` and ``tensorpotential.calculator`` used to import their TensorFlow classes when
the package was imported, so that ``import tensorpotential`` failed without TensorFlow and a
PyTorch-only environment could not reach the TensorFlow-free modules below them. The package now
offers the same names through :func:`lazy_exports`: the submodule is imported on the first access.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any

from tensorpotential.core.backends import (
    BackendNotInstalledError,
    backend_not_installed,
)

TF_PACKAGES = frozenset({"tensorflow", "tf_keras"})


def is_tf_import_error(exc: ImportError) -> bool:
    """Whether ``exc`` was raised because ``tensorflow`` or ``tf_keras`` could not be imported."""
    return (exc.name or "").split(".")[0] in TF_PACKAGES


def missing_tf_error(name: str, exc: ImportError) -> ImportError:
    """Build the error raised when ``name`` is requested and TensorFlow is unavailable.

    Parameters
    ----------
    name
        The public name that was requested, e.g. ``"TPModel"``.
    exc
        The original ``ImportError``; its text is kept in the message, except that a
        :class:`~tensorpotential.core.backends.BackendNotInstalledError` already carries the
        install hint and is only renamed after ``name``.
    """
    if isinstance(exc, BackendNotInstalledError):
        # already says what is missing and how to install it: only the requested name is added
        return backend_not_installed(exc.backend, exc.missing, needed_by=f"'{name}'")
    return ImportError(
        f"'{name}' needs TensorFlow, which could not be imported ({exc}). Install it with the "
        "'tf' extra: pip install 'tensorpotential[tf]'. The TensorFlow-free parts of the package, "
        "for example 'tensorpotential.core', do not need it.",
        name=exc.name,
    )


def lazy_exports(
    package: str, exports: Mapping[str, str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build the module-level ``__getattr__`` and ``__dir__`` of ``package``.

    Parameters
    ----------
    package
        The ``__name__`` of the package that will define the two functions.
    exports
        Maps each public name to the module that defines it.

    Returns
    -------
    tuple
        ``(__getattr__, __dir__)``. The first imports the defining module on the first access,
        stores the object in the package so that the next access is a plain attribute lookup,
        and raises :func:`missing_tf_error` when TensorFlow is the missing piece. A name that is
        not in ``exports`` raises ``AttributeError`` as usual.
    """

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        try:
            value = getattr(importlib.import_module(module_name), name)
        except ImportError as exc:
            if is_tf_import_error(exc):
                raise missing_tf_error(name, exc) from exc
            raise
        vars(sys.modules[package])[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*vars(sys.modules[package]), *exports})

    return __getattr__, __dir__
