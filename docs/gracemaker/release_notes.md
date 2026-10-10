# Release notes

## Keras 3 replaces legacy Keras 2

The package now trains with Keras 3, the version that TensorFlow 2.20 installs, and no longer needs `tf_keras`.

* **`tf_keras` is no longer installed** by `pip install "tensorpotential[tf]"`, and the package no longer sets
  `TF_USE_LEGACY_KERAS` (it leaves `os.environ` alone). If the variable is still exported (for example from a shell
  rc file or a job script written for an earlier version), importing the package warns; remove it with
  `unset TF_USE_LEGACY_KERAS`. With the variable set, TensorFlow tries to load `tf_keras`, which is no longer there.
* **Old checkpoints still load.** The model weights, the step and the epoch of a checkpoint written by an earlier
  version are restored exactly. Its **optimizer state is not**: Keras 3 stores the Adam moments (and, with
  `use_ema`, the moving averages) in another order, and restoring them would assign them to the wrong variables. The
  optimizer starts from scratch, with a `LegacyOptimizerStateWarning` and a line in the log, and the mid-epoch
  resume state of the checkpoint is cleared (as `reset_optimizer` does). The same happens for a checkpoint written with other optimizer
  options than the ones in use (for example `amsgrad` switched).
* **The moving average (`use_ema`) starts differently.** Keras 3 sets the average to the weights after the first
  step, where Keras 2 started from the initial weights. After `n` steps the two averages differ by
  `momentum**n * (w_0 - w_1)`, a gap that decays geometrically and is negligible after a few hundred steps.
  The Adam updates, the weight decay and the learning-rate schedules are unchanged (verified bit for bit on a
  small model).
* **The `reduce_on_plateau` callback no longer fails** with `TypeError: 'NoneType' object is not callable` (the error
  that the legacy-Keras variable used to work around), and its default mode still minimises a loss.
