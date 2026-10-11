---
paths:
  - ".github/**"
  - "tests/.test_durations"
  - "pyproject.toml"
  - "uv.lock"
  - "tests/test_packaging.py"
  - "tools/wheel_smoke.py"
  - "tools/resolve_extras.py"
---

# CI and packaging

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- **TensorFlow is the `tf` extra, not a base dependency (D12).** `tensorpotential/_tf_options.py` calls `core.backends.require_backend("tf")` first, so every TF-side module and console script stops with the `pip install 'tensorpotential[tf]'` hint without TF
  (`grace_dashboard` also needs `flask`); the `dev` group includes the extra. **Import `torch` (and `torch_sim`) before `tensorflow` in one process**: `import tensorflow, triton` segfaults on triton 3.8.0 / TF 2.20.
- **CI limits (CI2).** The hosted runner has 16 GB and a TensorFlow xdist worker's memory only grows, so `test-tf` is four `pytest-split` shards with 2 workers each; every pytest in CI goes through `.github/actions/run-pytest`.
  Workflows, measurements, package discovery, the regeneration commands: `.github/CI.md`.

```bash
# CI runs the suite as 4 pytest-split shards balanced by tests/.test_durations (regenerating it, running a shard locally: .github/CI.md)
uv run --frozen --no-sync pytest --noconftest tests/test_packaging.py   # wheel contents and extras; also tools/wheel_smoke.py, tools/resolve_extras.py (CI runs them: .github/CI.md)
```
