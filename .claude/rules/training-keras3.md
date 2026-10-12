---
paths:
  - "tensorpotential/tensorpot.py"
  - "tensorpotential/keras_adapter.py"
  - "tensorpotential/optimizer_layout.py"
  - "tensorpotential/_tf_options.py"
  - "tensorpotential/loss.py"
  - "tests/test_tensorpot.py"
  - "tests/legacy_keras_checkpoint/**"
---

# Training and Keras 3

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- **Keras 3 (DEPS3).** Under `dtype_conversion_mode="all"` (set by `_tf_options`) every Keras 3 optimizer operation fails unless `tensorpotential/keras_adapter.py` is imported (`tensorpot.py` does); never switch the promotion mode off around the optimizer step
  (process-wide: `MirroredStrategy` replica threads race). Under a strategy build the optimizer with `TensorPotential.prepare_for_training()`; an EMA swap needs `iterations > 0`. Adam casts its betas through float32 (an oracle must too: `tests/test_tensorpot.py::_through_float32`);
  a legacy Keras 2 checkpoint resets the optimizer with a `LegacyOptimizerStateWarning` (`optimizer_layout.py`). Reasons and measurements: docstring of `keras_adapter.py`.
