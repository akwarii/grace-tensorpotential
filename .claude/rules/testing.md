---
paths:
  - "tests/**"
  - "tests_torch/**"
  - "tools/tests/**"
  - "baselines/**"
  - "pytest.ini"
  - "conftest.py"
---

# Testing conventions

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

Procedures, the two layers, planted mutants, mock policy and speed tips: `grace-torch-tests` skill. Baseline pitfalls (probe checkpoint keys, junit and AST, snapshot reproducibility): `baselines/README.md`.

- A function or class you **modify** must already be covered at 90% (line and branch). Below that, write characterization tests first, green on the unmodified code, in their own commit, then change the code. Comment-only and annotation-only edits are exempt.
- Every unit gets a **logic** layer and, where physics applies, a **physical values** layer from an oracle that does not call the unit under test.
- **A test goes in the file named after the source module** (`tests/test_<module>.py` for `tensorpotential/<...>/<module>.py`, e.g. `tests/test_tp_model.py`); `tests/` stays flat, except `tensorpotential/core/` (TF-free shared code) whose tests
  live in `tests/core/test_<module>.py` (owner, 2026-10-04). Do not group tests by issue or kind of change; do not move existing test files unless an issue says so (owner, 2026-10-03).
- Prefer real objects to mocks; a mock or `monkeypatch` only at an external boundary (network, clock, absent hardware), with a comment saying what it replaces, never for the unit under test or a numeric path.
- Tolerances come from one named table (`tests/tolerances.py`, `numpy.isclose` semantics `atol + rtol*|reference|`); never inline a number, never widen a tolerance to make a test pass.
- Tests depend neither on the working directory nor on each other's output: `tmp_path`, per-worker directories, fixed seeds, no network. Random-weight models only; real or foundation-model weights are HPC-only.
- Markers: `slow` (30 s or more, registered in `pytest.ini`; the root `conftest.py` starts them first under `-n`), `gpu`, `hpc`, `tf` (these three not registered yet); unavailable capabilities skip locally but fail in CI jobs that require them. Nothing is skipped by default.
- A model, calculator or result built once and shared by tests is read-only for them: fingerprint it when built and compare when its scope ends (`tests/shared_models.py`: `weights_fingerprint`, `CuTwoLayerModels`). A test that saves, trains or edits a model builds its own.
- TF numerics are not bit-reproducible across processes (about one ulp on some float64 intermediates): compare through `tools/oracle_snapshot.py compare` (a named scaled tolerance per precision), never with exact float64 equality.
