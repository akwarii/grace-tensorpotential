---
paths:
  - "tensorpotential/core/**"
  - "tensorpotential/torch_backend/**"
  - "tests_torch/**"
  - "tests/test_import_gates.py"
  - "tests/test_tf_options.py"
  - "tests/core/**"
  - "tensorpotential/_tf_options.py"
  - "tensorpotential/__init__.py"
  - "tensorpotential/calculator/__init__.py"
---

# The TensorFlow-free packages and the import gates

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

`core/` and `torch_backend/` must import without TensorFlow (import contract: `uv run --frozen --no-sync lint-imports`; TF-free modules: `core/`, `constants`, `poly`, `couplings`, `foundation_models`); the TF extractor lives outside them.

- **The package does not import TensorFlow.** `tensorpotential/__init__.py` and `calculator/__init__.py` resolve public names on first access (`core/lazy.py`); the TF options (`_configure_tf_options`) run when a TF-side module imports `tensorpotential._tf_options`.
  **A new module with a module-level `import tensorflow` starts with `from tensorpotential import _tf_options  # noqa: F401`** (`tests/test_tf_options.py` scans for it; `compat/pace` is exempt). A module that must stay TF-free goes in `source_modules` of `[tool.importlinter]` and in `TF_FREE_MODULES` of `tests/test_import_gates.py`.
- The modules of `torch_backend/` need no listing: `tests_torch/test_package_structure.py` walks the package, imports every module with TensorFlow refused and the torch-free layers (errors, spec, plans, weights, protocol) with torch refused too, and checks that every exception type lives in `torch_backend/errors.py` and every dataclass is frozen. The layers contract of the backend is in `pyproject.toml`.
- `tests/test_import_gates.py` imports every module in its own interpreter and fails when a name that dead-code tools cannot see stops resolving (`from tensorpotential.X import name`, `__cls__` strings, the `__getattr__` shims); shrink its allow-lists (`KNOWN_ABSENT_PACKAGES`, `KNOWN_STALE_SOURCES`, `BASELINED_IMPORT_FAILURES`), never grow them silently.
- **Import `torch` (and `torch_sim`) before `tensorflow` in one process:** `import tensorflow, triton` segfaults on triton 3.8.0 / TF 2.20.
