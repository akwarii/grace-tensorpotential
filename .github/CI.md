# CI of the fork

What the workflows do, why they are shaped this way, and the commands behind them. The rules every agent needs are in `CLAUDE.md`; this file holds the detail.

## Workflows (`.github/workflows/`)

| File | Jobs |
|---|---|
| `lint.yaml` | ruff and ty on the new packages, the lint ratchet, the import contract, registry completeness, the tests of the packaging tools |
| `tests.yaml` | `test-core` (no TensorFlow), `test-torch` (pandas 2.3 and 3.0), `test-tf` (four shards); also nightly |
| `wheel.yaml` | `wheel`, `wheel-smoke`, `extras-resolve` |
| `nightly.yml` | both stacks in one environment, torch-sim main |
| `upstream-completeness.yml` | `tools/instruction_ast.py check` against upstream master; scheduled, so it runs only when Actions is on and the file is on the default branch |

Every pytest in CI goes through `.github/actions/run-pytest` (per-test `--timeout=900` from `pytest-timeout`, workers, markers, shard). Every job has `timeout-minutes`; `cancel-in-progress` is for pull requests only. A new job that runs pytest uses the action; a new heavy test file needs no edit.

## The hosted runner has 16 GB and a TensorFlow worker's memory only grows (CI2)

One unchanged `test-tf` job (4 workers, whole suite) climbed from 8 GB to 15 GB used and died with "The runner has received a shutdown signal" (no failing test, no kernel line when it survives) at 94 to 97% of the suite. A single runner with 2 workers on the whole suite still peaked at 14.9 GB, so the fix is **fewer tests per runner**: `test-tf` is four `pytest-split` groups with 2 workers each (peak memory used 6.7, 6.5, 6.0 and 5.9 GB, 5 to 7 minutes each; three groups had peaked at 7.1, 11.5 and 9.4 GB, too close to 16 GB). Disk was never the problem (80 GB free after the suite), so there is no free-disk step.

The shards are balanced by `tests/.test_durations`. After a large change regenerate it on an idle machine (times under `-n` are inflated but proportional, which is all the split needs), commit it, and run a group as CI does:

```bash
uv run --frozen --no-sync pytest tests -n 2 --store-durations --durations-path=tests/.test_durations <same --ignore options as the full run>
uv run --frozen --no-sync pytest tests --splits 4 --group 1 --splitting-algorithm=least_duration --durations-path=tests/.test_durations -n 2 --timeout=900 <same --ignore options>
```

## Packaging (CI1, D12)

- TensorFlow is the `tf` extra, not a base dependency: `pip install tensorpotential` has none; extras are `tf`, `torch`, `torch-sim`, `all`; `[dependency-groups] dev` includes `tensorpotential[tf]`.
- `tensorpotential/_tf_options.py` calls `core.backends.require_backend("tf")` first (it checks with `find_spec`, never imports TF), so every TF-side module, hence the console scripts `gracemaker`, `grace_predict`, `grace_preprocess`, `grace_utils` and `extxyz2df`, stops with the `pip install 'tensorpotential[tf]'` hint in a TF-free environment (a script needs no guard of its own; `grace_dashboard` also needs `flask`, which no extra installs).
- Package discovery is `[tool.setuptools.packages.find]` (`tensorpotential*`); an unscoped `find_packages()` would ship the tracked `tests` package.
- TF and torch resolve together in `uv.lock`, but **import `torch` (and `torch_sim`) before `tensorflow` in one process**: `import tensorflow, triton` segfaults on triton 3.8.0 / TF 2.20 (finding on the CI1 issue).
- Checks: `pytest --noconftest tests/test_packaging.py` (builds the wheel in a scratch copy, or `TENSORPOTENTIAL_WHEEL=<wheel>`: file list against `baselines/wheel_files_upstream.txt`, metadata, extras; `--noconftest` because `tests/conftest.py` imports TensorFlow), `python tools/wheel_smoke.py <venv>/bin/python --expect tf|no-tf` (clean venv with the wheel: core imports without TF, `--help` of the ten console scripts), `python tools/resolve_extras.py [EXTRA]` (resolves the extras with uv). CI runs the last two.

## Runtime of the suite

Measured on 14 cores and 30 GB, idle machine, about 2,640 tests, 2026-10-09: the full suite takes about 15 minutes with `-n 4` (19 with `--cov`); the longest test, `test_distrib`, takes about 4 minutes, and the total, not one test, bounds the run at `-n 4`.
An older figure for `-m "not slow"` (4 to 5 minutes at 1,340 tests) was not re-measured. A second suite running next to it about doubles the time, so compare timings only between back-to-back runs. The histogram PNGs of `load_and_prepare_datasets` cost about 1 s of a 15 to 48 s training run, so they are not worth a switch.
Re-measure before relying on these numbers.
