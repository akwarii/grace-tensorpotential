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

Other Keras 3 facts found by DEPS3 (moved here from ``CLAUDE.md``)
-------------------------------------------------------------------
* Measured on the replica race: with the mode switched off around the optimizer step, ``gracemaker -m``
  (``tests/test_distrib.py``) failed with ``x and y must have the same dtype``.
* Under a strategy Keras 3 cannot create the optimizer variables inside the traced step:
  ``TensorPotential.prepare_for_training()`` builds the optimizer once the trainable set, the checkpoint and
  ``reset_optimizer`` are final.
* A Keras 3 EMA average is zero until the first update, so a swap needs ``iterations > 0`` (``_ema_ready``), not
  merely the existence of the slots. Keras 3 starts the EMA at the weights after the first step (legacy Keras 2
  started from the initial weights): the two averages differ by ``momentum**n (w_0 - w_1)``, a documented change.
* Adam in both Keras lines casts its betas through float32 (``cast(python float, float64)`` goes through a float32
  tensor), so the bias correction uses ``float32(0.9)`` and ``float32(0.999)``: an oracle for it must do the same
  (``tests/test_tensorpot.py::_through_float32``) to reach 1e-12.
* A Keras 3 optimizer saves ``_iterations`` and ``_learning_rate`` under their own keys and the other slots at
  ``_variables/<n>`` from n = 2 (legacy: from 1), so a legacy checkpoint's slots restore at shifted positions
  without an error when shapes agree. ``load_checkpoint`` compares the layout first (``optimizer_layout.py``) and
  resets the optimizer with a ``LegacyOptimizerStateWarning``. The fixture is ``tests/legacy_keras_checkpoint/``
  (written by ``make_fixture.py`` on the tree before DEPS3, not regenerated).
"""

from __future__ import annotations

from typing import Any

from tensorpotential import _tf_options  # noqa: F401
import keras

from tensorpotential.errors import KerasVariableUnavailableError

PATCH_MARK = "_tensorpotential_type_comparison_fix"


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
