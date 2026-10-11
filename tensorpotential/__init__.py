import os
import warnings

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("tensorpotential")
except (ImportError, PackageNotFoundError):
    # If the package is not installed, don't crash.
    __version__ = "unknown"


LEGACY_KERAS_VARIABLE = "TF_USE_LEGACY_KERAS"
LEGACY_KERAS_TRUE_VALUES = ("1", "true", "True")  # the values for which TensorFlow loads legacy Keras


def _warn_if_legacy_keras_is_requested():
    """Warn when the environment still asks TensorFlow for legacy Keras.

    The package uses Keras 3 and neither sets nor reads ``TF_USE_LEGACY_KERAS``; ``os.environ`` is left alone.
    With the variable set, TensorFlow loads legacy Keras, which is no longer a dependency.
    """
    value = os.environ.get(LEGACY_KERAS_VARIABLE)
    if value in LEGACY_KERAS_TRUE_VALUES:
        warnings.warn(
            f"{LEGACY_KERAS_VARIABLE}={value!r} is set. {__name__} uses Keras 3 and no longer needs it; "
            "with it TensorFlow loads legacy Keras (tf_keras), which is not installed with this package "
            f"(the import fails without it). Remove it from the environment: unset {LEGACY_KERAS_VARIABLE}",
            RuntimeWarning,
            stacklevel=2,
        )


def _configure_tf_options(verbose=True):
    """
    Globally disables TensorFloat-32 execution for accurate mathematical operations.
    Runs once, when the first TensorFlow-side module imports ``tensorpotential._tf_options``.
    """
    try:
        import tensorflow as tf

        try:
            tf.experimental.numpy.experimental_enable_numpy_behavior(
                dtype_conversion_mode="all"
            )
        except TypeError:
            # Fallback for older TF versions or those that don't support the kwarg
            tf.experimental.numpy.experimental_enable_numpy_behavior()
            
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if verbose:
            print(f"[{__name__}] Info: tf.experimental.numpy behavior enabled.")
            print(f"[{__name__}] Info: TensorFloat-32 execution disabled.")
    except ImportError:
        pass


# Run immediately on import. This does not import TensorFlow: the TensorFlow options are applied by
# ``tensorpotential._tf_options``, which every TensorFlow-side module imports before TensorFlow.
_warn_if_legacy_keras_is_requested()

from tensorpotential.core.lazy import lazy_exports  # noqa: E402

__all__ = ["TensorPotential", "TPModel", "LossFunction", "L2Loss"]

# The classes are imported on first access (PEP 562), so that ``import tensorpotential`` works
# without TensorFlow; a missing TensorFlow raises an ImportError that names the ``tf`` extra.
__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "TensorPotential": "tensorpotential.tensorpot",
        "TPModel": "tensorpotential.tpmodel",
        "LossFunction": "tensorpotential.loss",
        "L2Loss": "tensorpotential.loss",
    },
)
