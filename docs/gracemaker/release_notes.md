# Release notes

## Importing the package no longer configures logging

Importing a library module (`tensorpotential.data.datasets`, the weighting and plotting modules, the training modules of `tensorpotential.cli`) used to call `logging.basicConfig`, so a program that
merely imported them had its root logger reconfigured. The console commands (`gracemaker`, `grace_utils`, `grace_collect`, `grace_predict`, `extxyz2df`, `df2extxyz`) now configure it themselves when they run, with the same format and level. A script
that imports these modules and relied on that side effect should call `logging.basicConfig` itself; `grace_uq build` now logs in its own format (`LEVEL: message`) instead of the one the datasets module had set.

The code that lived in `tensorpotential/cli/data.py`, `scripts/_kokkos_export.py` and in `scripts/grace_preprocess.py` and `scripts/grace_utils.py` is now in `tensorpotential/data/datasets.py`, `export_kokkos.py`,
`data/preprocess.py` and `model_ops.py`; the old import paths keep working.

## Model files are checked on load

A `model.yaml` and a downloaded model archive are untrusted input, and they are now treated so.

* **A `__cls__` entry is looked up, never executed.** It must be a dotted path (`package.module.Name`) that resolves to a
  subclass of the instruction class being loaded; anything else (a statement, a function, a class that is not an
  instruction) stops the load with a `ValueError` before the object is called. Instruction classes defined in your own
  package keep working. A name that a module does not define is now a `ValueError`, no longer an `ImportError`.
* **A downloaded archive may not hold links or special files that point outside its folder**; the check also applies to
  Python 3.11.0 to 3.11.3, which lack the `tarfile` data filter, by refusing every link and special file.

## A single `grace` command

`grace <subcommand>` runs the console commands of the package: `grace train` is `gracemaker`, `grace uq build` is
`grace_uq build`, and likewise `grace utils`, `grace models`, `grace preprocess`, `grace predict`, `grace collect`,
`grace dashboard`, `grace extxyz2df` and `grace df2extxyz`. See [the command line](cli.md).

* **The old names keep working** and print the same output, with the same options, defaults and exit codes; only the
  program name in usage lines differs (`grace uq build` through `grace`, `grace_uq build` through the old name).
* **`grace`, `grace --help` and an unknown sub-command need no TensorFlow**; a sub-command that does says so with the
  `pip install "tensorpotential[tf]"` command before it imports anything.
* **Output on a terminal is styled** with [rich](https://rich.readthedocs.io/); in a pipe, a file or a SLURM log it is the
  same plain text as before, and `NO_COLOR` turns the colour off.
* **`grace_dashboard` has an extra:** `pip install "tensorpotential[dashboard]"` installs Flask.

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
* **Restarting a run that uses `use_ema` resumes from the training weights.** A checkpoint stores the averaged
  weights in the model and the training weights in the optimizer; on a restart they are swapped back before
  training continues. With Keras 2 that swap was silently skipped (the optimizer was not built yet), so a
  restarted run continued from the averaged weights and lost part of its progress. A restart now continues
  exactly where the interrupted run stood (the next step is identical).
