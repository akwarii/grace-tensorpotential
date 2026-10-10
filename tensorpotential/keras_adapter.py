"""The one place that works around Keras 3 under TensorFlow's numpy-style type promotion.

``tensorpotential._tf_options`` enables ``dtype_conversion_mode="all"``, which the model code relies on. Under
that mode every operation of a Keras 3 optimizer fails with ``ValueError: Attempt to convert a value
(<class 'bool'>) with an unsupported type (<class 'type'>) to a Tensor``: the promotion rules ask whether an
operand ``== bool`` (and ``== int``, ``== float``), and ``keras.Variable.__eq__`` builds a tensor from the type
instead of answering ``False``.

:func:`install_variable_comparison_fix` makes ``keras.Variable == <type>`` answer ``False`` and changes nothing
else. It is a pure function, installed once for the process, so it is safe in the replica threads of a
``MirroredStrategy`` (an earlier attempt switched the promotion mode off around the optimizer step; the mode is
a process-wide setting, and a replica tracing the model in another thread then failed). Importing this module
installs the fix. When Keras no longer has the class or the method it patches,
:class:`KerasVariableUnavailableError` says so: nothing falls back silently.
"""

from __future__ import annotations

from typing import Any

from tensorpotential import _tf_options  # noqa: F401
import keras

PATCH_MARK = "_tensorpotential_type_comparison_fix"


class KerasVariableUnavailableError(RuntimeError):
    """``keras.Variable`` or its ``__eq__`` is missing, so the comparison fix cannot be installed."""


def install_variable_comparison_fix() -> None:
    """Make ``keras.Variable == <a Python type>`` answer ``False`` instead of raising; idempotent.

    Raises
    ------
    KerasVariableUnavailableError
        If ``keras.Variable`` or ``keras.Variable.__eq__`` does not exist in the installed Keras.
    """
    variable = getattr(keras, "Variable", None)
    original = getattr(variable, "__eq__", None)
    if variable is None or not callable(original):
        raise KerasVariableUnavailableError(
            "keras.Variable.__eq__ is not available in this Keras version. tensorpotential.keras_adapter patches "
            "it so that Keras 3 optimizers work under the numpy-style type promotion; adapt the module to this "
            "Keras version."
        )
    if getattr(original, PATCH_MARK, False):
        return

    def __eq__(self: Any, other: Any) -> Any:  # noqa: N807
        if isinstance(other, type):
            return False
        return original(self, other)

    setattr(__eq__, PATCH_MARK, True)
    variable.__eq__ = __eq__


install_variable_comparison_fix()
