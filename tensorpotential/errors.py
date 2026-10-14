"""The exceptions the library raises on purpose.

Every type derives from :class:`TensorPotentialError` and from the builtin type the same error was raised as before, so
an ``except ValueError`` (or ``RuntimeError``, ``ImportError``) written against an earlier version keeps catching it and a
caller can also catch every error of the package with one ``except TensorPotentialError``.

The module imports no TensorFlow and is on the import contract of the TensorFlow-free modules.
"""

from __future__ import annotations


class TensorPotentialError(Exception):
    """Base of the errors the library raises on purpose."""


class ModelFormatError(TensorPotentialError, ValueError):
    """The model file is in a format the operation does not accept (the old list of instructions)."""


class ClassPathError(TensorPotentialError, ValueError):
    """A dotted class path of a ``model.yaml`` or ``input.yaml`` is malformed, names nothing or names the wrong kind of object."""


class ArchiveError(TensorPotentialError, ValueError):
    """A downloaded archive has an unexpected layout or a member that would be written outside its folder."""


class KerasVariableUnavailableError(TensorPotentialError, RuntimeError):
    """``keras.Variable`` or its ``__eq__`` is missing, so the comparison fix cannot be installed."""
