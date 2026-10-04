"""Which optional backend an entry point needs, and the error that says how to install it.

The base installation of ``tensorpotential`` has neither TensorFlow nor PyTorch; each comes with an
extra of the same name (``tensorpotential[tf]``, ``tensorpotential[torch]``,
``tensorpotential[torch-sim]``). An entry point that cannot work without one of them calls
:func:`require_backend` before its first import of the backend, so that a missing backend gives one
actionable message instead of a bare ``ModuleNotFoundError``.

The check looks for the packages with ``importlib.util.find_spec`` and never imports them (importing
TensorFlow takes seconds); a package that is installed but broken fails later, with its own error.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

#: Importable top-level packages that each backend (and extra) needs.
BACKEND_PACKAGES: Mapping[str, tuple[str, ...]] = {
    "tf": ("tensorflow", "tf_keras"),
    "torch": ("torch",),
    "torch-sim": ("torch", "torch_sim", "vesin"),
}

_DISPLAY_NAMES = {"tf": "TensorFlow", "torch": "PyTorch", "torch-sim": "torch-sim"}


class BackendNotInstalledError(ImportError):
    """A backend that an entry point needs is not installed.

    Attributes
    ----------
    backend
        The backend (and extra) name, e.g. ``"tf"``.
    missing
        The top-level packages that could not be found.
    """

    def __init__(self, backend: str, missing: tuple[str, ...], message: str) -> None:
        super().__init__(message, name=missing[0])
        self.backend = backend
        self.missing = missing


def _is_installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ImportError:
        # a finder that refuses the package (as the no-TensorFlow tests install) counts as absent
        return False


def missing_packages(backend: str) -> tuple[str, ...]:
    """Return the packages of ``backend`` that cannot be found, in the order of :data:`BACKEND_PACKAGES`.

    Parameters
    ----------
    backend
        ``"tf"``, ``"torch"`` or ``"torch-sim"``.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the names above.
    """
    if backend not in BACKEND_PACKAGES:
        known = ", ".join(sorted(BACKEND_PACKAGES))
        msg = f"unknown backend {backend!r}; the backends are: {known}"
        raise ValueError(msg)
    return tuple(p for p in BACKEND_PACKAGES[backend] if not _is_installed(p))


def require_backend(backend: str, *, needed_by: str | None = None) -> None:
    """Return when ``backend`` is installed; raise :class:`BackendNotInstalledError` otherwise.

    Parameters
    ----------
    backend
        ``"tf"``, ``"torch"`` or ``"torch-sim"``; also the name of the extra to install.
    needed_by
        What asked for the backend (a command or a class name); named in the message when given.

    Raises
    ------
    BackendNotInstalledError
        With the missing packages and the ``pip install 'tensorpotential[<backend>]'`` command.
    ValueError
        If ``backend`` is unknown.
    """
    missing = missing_packages(backend)
    if not missing:
        return
    name = _DISPLAY_NAMES[backend]
    who = f"{needed_by} needs" if needed_by else "This needs"
    message = (
        f"{who} {name}, which is not installed (missing: {', '.join(missing)}). "
        f"Install it with: pip install 'tensorpotential[{backend}]'. "
        f"The parts of the package that do not need {name}, for example 'tensorpotential.core', still work."
    )
    raise BackendNotInstalledError(backend, missing, message)
