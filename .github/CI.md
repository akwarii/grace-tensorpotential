# CI of the fork

What the workflows do, why they are shaped this way, and the commands behind them. The rules every agent needs are in `CLAUDE.md`; this file holds the detail.

## Workflows (`.github/workflows/`)

| File | Jobs |
|---|---|
| `lint.yaml` | ruff and ty on the new packages, the lint ratchet, the import contract, `deptry`, the dead-code ratchet, registry completeness, the tests of the packaging, CI and dead-code tools, the clone ratchet, the divergence ledger |
| `tests.yaml` | `test-core` (no TensorFlow), `test-torch` (pandas 2.3 and 3.0), `test-tf` (four shards, each uploads its coverage data), `quality-gates` (combines the coverage); also nightly |
| `durations.yaml` | `Durations` (CI4): weekly and on request, the four shards store their durations and `tools/merge_durations.py` merges them into the artifact `test-durations`; read-only token, never pushes (section "The shard durations") |
| `wheel.yaml` | `wheel`, `wheel-smoke`, `extras-resolve` |
| `nightly.yml` | both stacks in one environment, torch-sim main, `quality-report` (`tools/quality_report.py check` on the coverage and junit files of the Tests run of the same commit; the table goes to the job summary, a worse metric fails the job) |
| `upstream-completeness.yml` | `tools/instruction_ast.py check` against upstream master; scheduled, so it runs only when Actions is on and the file is on the default branch |

Every pytest in CI goes through `.github/actions/run-pytest` (per-test `--timeout=900` from `pytest-timeout`, workers, markers, shard). Every job has `timeout-minutes`; `cancel-in-progress` is for pull requests only. A new job that runs pytest uses the action; a new heavy test file needs no edit.

## Required checks and what CI does not run (CI3)

The job names a pull request must pass; the owner sets them as required checks in the repository settings (an agent never changes them):

| Workflow | Jobs | Covers |
|---|---|---|
| Lint | `lint` | strict ruff and ty on the new packages, lint ratchet, import contract, `deptry`, dead-code ratchet (`tools/dead_code.py`), registry completeness, packaging- and dead-code-tool tests, clone ratchet (`tools/check_clones.py`), divergence ledger (`tools/divergence.py`, against a read-only fetch of the public upstream master: no secret, no push permission) |
| Tests | `test-core`, `test-torch (2.3.3)`, `test-torch (3.0.3)`, `test-tf (group 1 of 4)` to `test-tf (group 4 of 4)` | the suites |
| Tests | `quality-gates` | combines the coverage data of the four shards, runs `tools/coverage_ratchet.py check` and, on a pull request, `tools/check_touched_coverage.py --base origin/<base branch>` (touched units at 90%) |
| Wheel | `wheel`, `wheel-smoke`, `extras-resolve` | packaging |

`quality-gates` needs the four shards to pass, so a failing shard hides its verdict. Each shard writes `.coverage.group-N` (`COVERAGE_FILE`) and uploads it, with `junit.group-N.xml`, as the artifact `coverage-group-N`; the combine step needs the checkout path of the shards, which is the same on every hosted job.

CI does **not** run: `python tools/board.py lint` and every other project-board query (a token with project scope; it stays in the agent procedure), the real-weight and foundation-model tests (HPC only), any GPU test, the comparisons with `baselines/` (`tools/junit_outcomes.py`, `tools/ast_manifest.py`; run them locally for a gate), and the scheduled `Nightly` and `Upstream completeness` jobs on a pull request.

## Python versions (DEPS4)

`requires-python` is `>=3.12` (floor raised on 2026-10-12, CI3). Every job runs 3.12, so the floor is covered by `test-core`, `test-torch`, `test-tf`, `lint` and the nightly jobs; `wheel-smoke` and `extras-resolve` also run 3.13, the newest version the project claims. Nothing runs the TensorFlow suite or `test-core` on 3.13, and nothing on 3.14. `upstream-completeness.yml` uses 3.11 because it only runs `tools/instruction_ast.py` on upstream master (no dependencies), which still supports older versions. A matrix on `test-core` would rename the required check `test-core` (to `test-core (3.12)`), which is the owner's to change in the repository settings, so the job stays as it is.

## The hosted runner has 16 GB and a TensorFlow worker's memory only grows (CI2)

One unchanged `test-tf` job (4 workers, whole suite) climbed from 8 GB to 15 GB used and died with "The runner has received a shutdown signal" (no failing test, no kernel line when it survives) at 94 to 97% of the suite. A single runner with 2 workers on the whole suite still peaked at 14.9 GB, so the fix is **fewer tests per runner**: `test-tf` is four `pytest-split` groups with 2 workers each (peak memory used 6.7, 6.5, 6.0 and 5.9 GB, 5 to 7 minutes each; three groups had peaked at 7.1, 11.5 and 9.4 GB, too close to 16 GB). Disk was never the problem (80 GB free after the suite), so there is no free-disk step.

## The shard durations (CI2, CI3, CI4)

The shards are balanced by `tests/.test_durations` (`pytest-split`, `--splitting-algorithm=least_duration`). The file ages with every merge that adds tests, and `pytest-split` gives an unknown test the mean duration, so the shards drift apart (right after CI3 the file did not know 452 of 3,759 tests, 12%). **Regenerate it with the `Durations` workflow, not on a developer machine**: CI3 measured that durations stored on a developer machine are no better than the old file (slowest shard with coverage 583 s against 560 s), while durations stored on the hosted runner with coverage on, the configuration CI runs, are the only ones that matched the runner.

- `Tests` prints a `::warning::` annotation (a warning, never a failure: a stale file costs time, not correctness) when more than 10% of the collected tests have no entry in the file. It is counted by `tools/merge_durations.py stale` in the job of shard 1, from `pytest --collect-only -q` (the action's `log-file` input keeps the ids).
- `Durations` (`durations.yaml`) runs weekly (Sunday 03:30 UTC) and on request (`gh workflow run durations.yaml`, or the Actions tab). It runs the four shards as `Tests` does (same action, two workers, coverage on) with `--store-durations --clean-durations`, so each shard file holds exactly its own tests, and a last job runs `tools/merge_durations.py merge`: the shard values replace the committed ones, the entries of tests that no shard ran any more are dropped, and the job summary lists the tests in the file, the new and the dropped ones, and the share of the total duration that moved by more than 20%. The result is the artifact `test-durations` (30 days).
- The workflow has a read-only token: it neither pushes nor opens a pull request. Download the artifact (`gh run download <run-id> -n test-durations`), put the file at `tests/.test_durations` on a work branch (`ci-<slug>`, one commit `chore: regenerate the test durations`) and open the pull request. A failing test in any shard stops the merge (the shard files must be complete), so fix or rerun first.
- A scheduled or dispatched workflow starts only from the default branch, so `Durations` runs on its schedule and on request only once `durations.yaml` is on the default branch of the fork (the rule recorded for `nightly.yml`; `torch-backend` is the integration branch, not the default). Its first run was made from the pull request of CI4 with a temporary `pull_request` trigger, a trick to reuse for any change to the workflow itself.
- Not done: splitting or pinning `test_distrib` (one test of about 176 s that keeps its shard the slowest; `pytest-split` balances the sum of the durations, not the makespan of two xdist workers).

To run a shard locally as CI does (a developer machine's durations are fine for this, not for the committed file):

```bash
uv run --frozen --no-sync pytest tests --splits 4 --group 1 --splitting-algorithm=least_duration --durations-path=tests/.test_durations -n 2 --timeout=900 <same --ignore options>
```

## Locally

**Locally too (CLI1):** a whole-suite `-n 4` run needs more than 18 GB of a 30 GB machine (workers of 4 to 6 GB each, `tests/test_integration_test.py` the largest consumer), and with a browser open it took the desktop session down at 94% (no OOM line in the journal). For a full or coverage run use the CI shards one after another (`--splits 4 --group N --splitting-algorithm=least_duration --durations-path=tests/.test_durations`, `--cov-append` with one `COVERAGE_FILE`, junit files merged by hand) inside `systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=2G`, so that an overrun kills the test run and not the session. Every run of `predict_main` / `TPCalculator` keeps about 350 MB of TensorFlow state that `clear_session`, `gc` and `malloc_trim` do not return; and under `--dist load` a module fixture is rebuilt on every worker that gets one of its tests, so a heavy run belongs inside one test. To find the culprit, log the resident memory after each test with a small plugin (`/proc/<pid>/status`, `VmRSS`) and run the files serially.

## Packaging (CI1, D12)

- TensorFlow is the `tf` extra, not a base dependency: `pip install tensorpotential` has none; extras are `tf`, `torch`, `torch-sim`, `dashboard` (Flask, for `grace_dashboard`), `all`; `[dependency-groups] dev` includes `tensorpotential[tf,dashboard]`.
- `tensorpotential/_tf_options.py` calls `core.backends.require_backend("tf")` first (it checks with `find_spec`, never imports TF), so every TF-side module, hence the console scripts `gracemaker`, `grace_predict`, `grace_preprocess`, `grace_utils` and `extxyz2df`, stops with the `pip install 'tensorpotential[tf]'` hint in a TF-free environment (a script needs no guard of its own; `grace_dashboard` needs `flask`, which the `dashboard` extra installs).
- Package discovery is `[tool.setuptools.packages.find]` (`tensorpotential*`); an unscoped `find_packages()` would ship the tracked `tests` package.
- TF and torch resolve together in `uv.lock`, but **import `torch` (and `torch_sim`) before `tensorflow` in one process**: `import tensorflow, triton` segfaults on triton 3.8.0 / TF 2.20 (finding on the CI1 issue).
- Checks: `pytest --noconftest tests/test_packaging.py` (builds the wheel in a scratch copy, or `TENSORPOTENTIAL_WHEEL=<wheel>`: file list against `baselines/wheel_files_upstream.txt`, metadata, extras; `--noconftest` because `tests/conftest.py` imports TensorFlow), `python tools/wheel_smoke.py <venv>/bin/python --expect tf|no-tf` (clean venv with the wheel: core imports without TF, `--help` of the eleven console scripts), `python tools/resolve_extras.py [EXTRA]` (resolves the extras with uv). CI runs the last two.

## Runtime of the suite

Measured on 14 cores and 30 GB, idle machine, about 2,640 tests, 2026-10-09: the full suite takes about 15 minutes with `-n 4` (19 with `--cov`); the longest test, `test_distrib`, takes about 4 minutes, and the total, not one test, bounds the run at `-n 4`.
An older figure for `-m "not slow"` (4 to 5 minutes at 1,340 tests) was not re-measured. A second suite running next to it about doubles the time, so compare timings only between back-to-back runs. The histogram PNGs of `load_and_prepare_datasets` cost about 1 s of a 15 to 48 s training run, so they are not worth a switch.
Re-measure before relying on these numbers.
