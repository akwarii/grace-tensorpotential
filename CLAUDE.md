# CLAUDE.md

GRACE / `tensorpotential`: machine-learning interatomic potentials (Graph Atomic Cluster Expansion) with the `gracemaker` training CLI, an ASE calculator,
UQ tools and foundation-model loading. The library is TensorFlow 2 (Keras legacy mode); a PyTorch backend that loads TF-trained models is being added next
to it. Core abstractions: an *instruction graph* (`TPInstruction` subclasses described in a `model.yaml`) run by `TPModel`, trained by `TensorPotential`,
served by `TPCalculator`.

## Precedence

Everything below is a default. **The user's instructions, and the issue you were asked to work on, take precedence over this file and the skills.** An issue
states the agreed scope of one task: what may change, what must not, how it is verified. If it (or the user) asks for something this file discourages (for
example restructuring legacy code in a clean-up milestone), do what the issue says within its scope, keep the safeguards it names (coverage,
behaviour-preserving checks, baselines), and note the conflict in a comment on the issue. If the user and an issue disagree, ask.

## Development commands

```bash
uv sync --group dev          # .venv from the committed uv.lock; the dev group has pytest, pytest-cov, pytest-xdist, ruff, ty and the tf extra (TF is not a base dependency)

# Tests: from the repository root; they write nothing into the tree. Under -n N every worker gets cores/N TF and OpenMP threads (conftest.py).
uv run --frozen --no-sync pytest tests -q -n 4 --dist load \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py   # fast loop: add -m "not slow"
uv run --frozen --no-sync pytest tests/test_instructions.py -vv    # one file, serial
uv run --frozen --no-sync pytest tests_torch -q                     # torch-backend tests, TF-free, seconds

# CI shards the suite into 4 groups (pytest-split, balanced by tests/.test_durations). After a large change regenerate that file on an idle machine (times under -n are inflated but proportional, which is all the split needs), commit it, and run a group as CI does:
uv run --frozen --no-sync pytest tests -n 2 --store-durations --durations-path=tests/.test_durations <same --ignore options>
uv run --frozen --no-sync pytest tests --splits 4 --group 1 --splitting-algorithm=least_duration --durations-path=tests/.test_durations -n 2 --timeout=900 <same --ignore options>

# tests_torch/structures/structures.json/.xyz are written by python -m tests_torch.structures.build_structures (test_build_structures.py fails on a difference).
# Golden fixtures (tests_torch/golden/README.md): the generator is the only thing allowed to write them, from a committed tree.
uv run --frozen --no-sync python tools/make_golden.py write --tier tiny   # tests_torch/golden/ (committed); --tier faithful: tests_torch/fixtures/ (npz git-ignored)
uv run --frozen --no-sync python tools/make_golden.py verify tests_torch/golden   # also: compare DIR_A DIR_B, cells
uv run --frozen --no-sync pytest tools/tests/test_make_golden.py -q   # the generator (TF, about 2 minutes); tests_torch/test_golden_fixtures.py checks the committed files without TF

# Lint, format, types (dev group pins ruff==0.16.7, ty==0.0.84; ty is pre-1.0)
uv run --frozen --no-sync ruff check PATH                  # strict set in the new packages; elsewhere E, F, ERA001
uv run --frozen --no-sync ruff format --preview NEW_FILE   # NEW files only, never reformat existing files
uv run --frozen --no-sync ty check NEW_PACKAGE             # strict in the new packages; [[tool.ty.overrides]] relax legacy
uv run --frozen --no-sync python tools/lint_ratchet.py check   # legacy ruff/ty counts per (file, rule) may not rise; `record` after a drop (a rise needs --allow-rise)
uv run --frozen --no-sync lint-imports                     # import contract: TF-free modules (core/, constants, poly, couplings, foundation_models) never reach tensorflow
uv run --frozen --no-sync python tools/divergence.py check --pr-branches   # every upstream file the fork modifies has a row in tools/divergence.yaml (same PR; drop it when the file equals upstream again); no fork-only path on a pr/U* branch
uv run --frozen --no-sync python tools/instruction_ast.py check   # every TPInstruction subclass has a registry entry and pinned defaults equal the source; `table` prints the constructor table
uv run --frozen --no-sync pytest --noconftest tests/test_packaging.py   # wheel built in a scratch copy (or TENSORPOTENTIAL_WHEEL=<wheel>): file list vs baselines/wheel_files_upstream.txt, metadata, extras
python tools/wheel_smoke.py <venv>/bin/python --expect tf|no-tf   # clean venv with the wheel: core imports without TF, --help of the ten console scripts; tools/resolve_extras.py [EXTRA] resolves the extras (CI runs both)
prek install                                               # hooks on changed files (.pre-commit-config.yaml): strict ruff, ty, ratchet

# Coverage and baselines (procedures: skills grace-torch-tests, grace-torch-goldens; baselines/README.md)
uv run --frozen --no-sync pytest tests -q -n 4 --dist load --cov=tensorpotential --cov-branch --cov-report=json:cov.json <same --ignore options>
python tools/check_touched_coverage.py --coverage cov.json [--base origin/torch-backend] [--unit PATH::Class.method]   # touched units: 90%
python tools/coverage_ratchet.py check cov.json            # a file fails when its share falls below baselines/coverage_baseline.json and its uncovered count rises
python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json   # also tools/ast_manifest.py and tools/check_clones.py
```

**Runtime** (14 cores, 30 GB, about 1,340 tests, 2026-10-03): the full suite takes about 10 minutes with `-n 4` on an idle machine (12 with `--cov`), about 20 serially;
`-m "not slow"` 4 to 5 minutes; a second suite next to it about doubles that. Compare timings only between back-to-back runs.
Not in a normal run: `test_structured_grid.py` (skipped on public master: needs the non-existent `tensorpotential.experimental`; ignored above only so counts match `baselines/`) and
`test_foundation_model_regression.py` (needs foundation-model weights, HPC only). A test writes only into `tmp_path` or a scratch directory; `git status` is clean after a run.

## Code style

- Python 3.11+, `from __future__ import annotations` in new modules, type hints on everything new (checked by `ty`).
- **Legacy code stays as it is**: no repo-wide reformat, no file splitting (`instructions/compute.py` is about 4,900 lines and upstream merges depend on its shape), no renames; improve a unit only
  while you are already changing it. An issue whose job is to change legacy code (clean-up, quality, deduplication, packaging, performance) says what may change; the coverage and comparison rules govern it and it is not widened.
- New packages get the strict ruff set (I, UP, B, SIM, C4, RUF, PD, NPY, PIE, PLE, PLW, PERF, RET, PTH, T20, ERA, W, complexity, docstring, exception, security) from the **single ruff configuration in
  `pyproject.toml`** (no `ruff.toml`; `per-file-ignores` switch the strict families off outside the new packages, listed there and in `tools/lint_ratchet.py`).
- Numpy-convention docstrings on new public API. No commented-out code, no `print` in library code (`logging.getLogger(__name__)`), no bare `except`, no `TODO` without an issue link, no mutable or shared default arguments.
- Small functions (complexity at most 10, at most 6 arguments); pure functions and frozen dataclasses for specifications. Errors are typed, actionable and raised early; nothing falls back silently.
- Physics names (`N806`, `E741`) are allowed. Units: eV, eV/Angstrom, eV/Angstrom^3.

## Testing conventions

Procedures, the two layers, planted mutants, mock policy and speed tips: `grace-torch-tests` skill.

- A function or class you **modify** must already be covered at 90% (line and branch). Below that, write characterization tests first, green on the unmodified code, in their own commit, then change the code. Comment-only and annotation-only edits are exempt.
- Every unit gets a **logic** layer and, where physics applies, a **physical values** layer from an oracle that does not call the unit under test.
- **A test goes in the file named after the source module** (`tests/test_<module>.py` for `tensorpotential/<...>/<module>.py`, e.g. `tests/test_tp_model.py`); `tests/` stays flat, except `tensorpotential/core/` (TF-free shared code) whose tests
  live in `tests/core/test_<module>.py` (owner, 2026-10-04). Do not group tests by issue or kind of change; do not move existing test files unless an issue says so (owner, 2026-10-03).
- Prefer real objects to mocks; a mock or `monkeypatch` only at an external boundary (network, clock, absent hardware), with a comment saying what it replaces, never for the unit under test or a numeric path.
- Tolerances come from one named table (`tests/tolerances.py`, `numpy.isclose` semantics `atol + rtol*|reference|`); never inline a number, never widen a tolerance to make a test pass.
- Tests depend neither on the working directory nor on each other's output: `tmp_path`, per-worker directories, fixed seeds, no network. Random-weight models only; real or foundation-model weights are HPC-only.
- Markers: `slow` (30 s or more, registered in `pytest.ini`; the root `conftest.py` starts them first under `-n`), `gpu`, `hpc`, `tf` (these three not registered yet); unavailable capabilities skip locally but fail in CI jobs that require them. Nothing is skipped by default.
- A model, calculator or result built once and shared by tests is read-only for them: fingerprint it when built and compare when its scope ends (`tests/shared_models.py`: `weights_fingerprint`, `CuTwoLayerModels`). A test that saves, trains or edits a model builds its own.

## Architecture

```
ASE Atoms -> TPAtoms / GeometricalDataBuilder (neighbour list) -> TPModel(instructions) -> energy, forces, virial -> TPCalculator
```

- `instructions/`: instruction classes (`compute.py` geometry, basis, products, reductions; `output.py` targets and shifts). Each has `build` (variables, index tables) and `frwrd`; its constructor arguments are captured into the saved `model.yaml`.
- `tpmodel.py`: `TPModel` runs the instructions in file order and derives forces and virial from the pair energies. `tensorpot.py`: `TensorPotential` (training, checkpoints). Also `loss.py`, `metrics.py`, `utils.py`, `poly.py`, `export.py`.
- `functions/`: Clebsch-Gordan and coupling tables (`couplings.py`), radial functions, activations. `data/`: dataframes and TF datasets (`databuilder.py`, `process_df.py`), neighbour lists (matscipy by default), streaming.
- `calculator/`: `TPCalculator` (ASE), foundation-model loading. `cli/`, `scripts/`: `gracemaker`, `grace_preprocess`, `grace_predict`, ... `potentials/`: presets. `extra/`: model generators. `uq/`: GMM-based uncertainty.
- `compat/pace/`: legacy, untested, **out of scope: no change is allowed to it** (owner, 2026-10-01), whatever the issue; report findings, never fix or remove them.
- The PyTorch backend lives in `torch_backend/` with TF-free shared code in `core/`; both must import without TensorFlow (import contract); the TF extractor lives outside them.
- `tests/` (run from the repository root), `baselines/` and `tools/` (untouched-tree baselines and the tools that compare with them), `docs/`.

Key dependencies: `tensorflow~=2.20` + `tf_keras` (the package sets `TF_USE_LEGACY_KERAS=1`), `numpy`, `pandas<3` (see Gotchas), `ase`, `matscipy`, `sympy`/`scipy`, `pyyaml`; extras `torch`, `torch-sim`.

## Gotchas

Baseline pitfalls (probe checkpoint keys, junit and AST, snapshot reproducibility): `baselines/README.md` ("Pitfalls"); units, signs, force and virial formulas: `grace-torch-numerics` skill; golden fixtures: `tests_torch/golden/README.md`.

- `capture_init_args` writes constructor defaults into the saved `model.yaml`: a default is part of every saved model, so changing a constructor signature or default changes existing models (persisted API). It is a **class** decorator.
- **A new `TPInstruction` subclass needs a registry entry** (`tests/test_registry.py`, `tools/instruction_ast.py check`, also a scheduled job on upstream master that runs only when Actions is on and the workflow file is on the default branch):
  neither in `SUPPORTED_OPTIONS` (with a spec sheet) nor in `REJECTED_CLASSES` (with a reason) of `torch_backend/spec/options.py`, or a pinned default that differs from the source, fails. A class that inherits its constructor reports the parent's parameters.
- Output instructions overwrite `input_data[target.name]` **in place** (`instructions/output.py`); instruction order in the yaml matters and a dump must snapshot the key before and after.
- `tpmodel.__getattr__` and `instructions.__getattr__` look unused but saved yamls import classes through them; dead-code tools flag them, keep them.
- `pandas<3` stays pinned (decision D11) although the suite is green on pandas 3.0.3 and 2.3.3 (DEPS1: `_split_rows`); pickles written with pandas 3 cannot be read by pandas 2. `pd.concat(..., copy=False)` in `cli/data.py` still warns on pandas 3.
  A second pandas environment (for the pandas 3 run) is built as in `baselines/README.md`, with its `bin` on `PATH` (tests start `grace_preprocess` by name).
- TF numerics are not bit-reproducible across processes (some float64 intermediates of `large_base` differ by about one ulp): compare through `tools/oracle_snapshot.py compare` (a named scaled tolerance per precision), never with exact float64 equality.
  `baselines/oracle_snapshot_wide.npz` adds float32, `lm_first`, `dense_nbr`, edge structures and presets.
- `GeometricalDataBuilder.extract_from_ase_atoms` edits its argument: `enforce_pbc` gives a non-periodic `Atoms` a cell, centres it and sets `pbc=True` (also overriding `pbc=(T,T,F)`), so pass a copy when the structure is reused
  (stress would otherwise be divided by an invented volume). An atom with no neighbour gets one dummy bond `(i, 0)` far outside the cutoff, so it never has zero bonds; a zero third cell vector with partial `pbc` raises `LinAlgError`.
  Pinned by `tests/test_databuilder.py`, `tests/test_utils.py`; brute-force reference `tests/neighbour_oracle.py`.
- Golden fixtures: the library imported is the one on `PYTHONPATH` (read the `library:` line the generator prints); the manifest records `library_dirty` and `generator_dirty` and `tests_torch/test_golden_fixtures.py` fails on a modified tree, so commit the generator first.
  Never write the manifest with sorted keys (its instruction order is the execution order). A reload test must start from a seed other than the fixture's. More in the golden README.
- `lm_first=True` fails on yamls whose output is `MLPOut2ScalarTarget` (no transpose, unlike `LinMLPOut2ScalarTarget`); `model_grace.yaml` is one. `CollectInvarBasis` cannot be instantiated as shipped (abstract `upd_init_args_new_elements` missing); tests subclass it with a stub.
  `ConstantScaleShiftTarget` sorts `atomic_shift_map` by key, so keys must be element indices.
- **Worktrees.** A worktree has no `.venv` (git-ignored): link it from the main checkout (`uv.lock` is tracked). Never run `uv sync` there: it repoints the shared `.venv`'s editable `tensorpotential` install to the worktree (repair with `uv sync --frozen --group dev` in the main checkout);
  `uv run --frozen --no-sync` and `.venv/bin/<tool>` are safe. Run with `PATH=<worktree>/.venv/bin:$PATH` (subprocess tests call `grace_preprocess`) and `PYTHONPATH=$PWD`, print `tensorpotential.__file__` once (the editable install otherwise resolves to the main tree),
  give `ty` the environment with `--python <main>/.venv`, use `--cov=tensorpotential`, and in a scratch script import `tensorpotential` before `tensorflow` (or Keras 3 is used).
- **The hosted runner has 16 GB and a TensorFlow xdist worker's resident memory only grows (CI2).** One unchanged `test-tf` job (4 workers, whole suite) climbed from 8 GB to 15 GB used and died with "The runner has received a shutdown signal"
  (no failing test, no kernel line when it survives) at 94 to 97% of the suite; a single runner with 2 workers on the whole suite still peaked at 14.9 GB, so the fix is **fewer tests per runner**: `test-tf` is four `pytest-split` groups with 2 workers each (peak memory used 6.7, 6.5, 6.0 and 5.9 GB, 5 to 7 minutes each; three groups peaked at 7.1, 11.5 and 9.4 GB, too close to 16 GB).
  Every pytest in CI goes through `.github/actions/run-pytest` (per-test `--timeout=900` from `pytest-timeout`, workers, markers, shard); every job has `timeout-minutes`; `cancel-in-progress` is for pull requests only. A new heavy test file needs no edit; a new CI job that runs pytest uses the action.
  Disk was never the problem (80 GB free after the suite), so there is no free-disk step.
- **TensorFlow is the `tf` extra, not a base dependency (D12).** `pip install tensorpotential` has none; extras are `tf`, `torch`, `torch-sim`, `all`; the `dev` group includes `tensorpotential[tf]`. `tensorpotential/_tf_options.py` calls `core.backends.require_backend("tf")` first
  (`find_spec`, never imports TF), so every TF-side module, hence `gracemaker`, `grace_predict`, `grace_preprocess`, `grace_utils` and `extxyz2df`, stops with the `pip install 'tensorpotential[tf]'` hint without TF (`grace_dashboard` also needs `flask`, which no extra installs).
  Package discovery is `[tool.setuptools.packages.find]` (`tensorpotential*`); an unscoped `find_packages()` would ship the tracked `tests` package. TF and torch resolve together in `uv.lock`, but **import `torch` (and `torch_sim`) before `tensorflow` in one process**:
  `import tensorflow, triton` segfaults on triton 3.8.0 / TF 2.20. CI: `.github/workflows/` `lint.yaml`, `tests.yaml` (test-core without TF, test-torch, test-tf; also nightly), `wheel.yaml` (wheel, wheel-smoke, extras-resolve), `nightly.yml` (both stacks, torch-sim main).
- **The package does not import TensorFlow.** `tensorpotential/__init__.py` and `calculator/__init__.py` resolve public names on first access (`core/lazy.py`); the TF options (`_configure_tf_options`) run when a TF-side module imports `tensorpotential._tf_options`.
  **A new module with a module-level `import tensorflow` starts with `from tensorpotential import _tf_options  # noqa: F401`** (`tests/test_tf_options.py` scans for it; `compat/pace` is exempt). A module that must stay TF-free goes in `source_modules` of `[tool.importlinter]` and in `TF_FREE_MODULES` of `tests/test_import_gates.py`.
- `tests/test_import_gates.py` imports every module first in its own interpreter (import cycles the old preloading hid now show) and fails when a name that dead-code tools cannot see stops resolving (`from tensorpotential.X import name`, `__cls__` strings, the two `__getattr__` shims).
  Its allow-lists (`KNOWN_ABSENT_PACKAGES`, `KNOWN_STALE_SOURCES`, `BASELINED_IMPORT_FAILURES`) name what is already broken: shrink them, never grow them silently.

## Working agreements

Procedure (board commands, PR text, findings, Definition of Done): `grace-torch-ticket` skill; these rules hold even without it.

- **You may commit, and once the work of your issue is ready you open a draft pull request into `torch-backend` yourself and ask the user to review it. You never merge it, mark it ready for review or turn on auto-merge: the user does.** *Ready* means: the exit criterion is met,
  the checks the issue names are green, and the description from `python tools/board.py pr-body <ID>` passes `pr-check`. Commit on the issue's work branch (one concern per commit, never on `torch-backend` or `master`), push only that branch to the fork,
  then `python tools/board.py status <ID> "PR Open"` and tell the user the PR number and what to look at. Nothing is pushed to, or opened against, the upstream `ICAMS` repositories without an explicit go for that unit (no `pr/U*` branch on your own; the `upstream` remote has its push URL disabled).
- **One agent, one issue, one git worktree** (`board.py` rewrites the whole issue body when it ticks a box). Never switch branches in, or run a branch-changing command on, a tree another agent or the user is using. After the merge and `board.py done`, remove the worktree without `--force`
  (if it refuses, report the uncommitted work); leave the remote branch unless the user says otherwise.
- **Check that the machine is idle before any test, coverage, baseline, `gracemaker` or other TensorFlow job:** `uptime` and `ps -eo pid,etimes,args | grep '[p]ytest\|[g]racemaker'`. It is **busy** when another test or training process runs (any worktree, any agent) or the 1-minute load is above 4 (14 cores).
  When busy, start no full, `-n` or coverage run; do work that needs no CPU (edits, `ruff`, `ty`, one test file serially), look again later and tell the user when waiting blocks you. One suite at a time, also your own. Never kill another agent's process; never use a busy-machine run for a timing or baseline claim.
- Ask before any download (file, source, size) and before any other outward action (creating or closing issues, repository settings); the draft PR of your own issue is covered above. New issues start only on the user's go, from `.github/ISSUE_TEMPLATE/milestone.md`.
  Development dependencies (test, coverage, parallelism, lint, typing, mutation tools) may be added to the `dev` group; runtime dependencies are decisions.
- **An issue is resolved only once a PR with `Refs #<issue>` is merged into `torch-backend` on this fork** (`tools/board.py done` checks; `--waive-pr` only when the user says so). Follow the issue's protocol: Todo to In Progress when starting, unexpected findings commented on the issue,
  and the **Definition of Done boxes ticked every time a task is finished** (`tools/board.py check`), with evidence.
- **Mind the GraphQL budget** (5,000 points an hour for the whole account, shared by all agents): use `tools/board.py`, never `gh project item-list` / `field-list` (about 100 and 150 points a call), never loops of `gh` calls; `python tools/board.py budget` shows what is left.
- **Keep the issues current.** When a task changes something an issue mentions (path, command, count, decision, tool, dependency line, exit criterion), update that issue in the same task; run `python tools/board.py lint` and fix every `STALE` finding.
- **The fork is public: sanitise everything you post** (issues, comments, PR text, commit messages). `tools/board.py` sanitises what it posts; pass other text through `python tools/board.py sanitise FILE`. No personal data, institution names, e-mail addresses, absolute local paths or research-application details;
  no bare `#N` that is not an issue of this repository; no `@mentions` outside code spans; no links to other repositories' issues or pull requests (back-links). Sanitise the existing text of an issue you touch when it needs it.
- `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` are fork-only: committed on `torch-backend`, never on an upstream branch (`pr/U*`); check with `python tools/check_pr_branch.py` before any upstream PR text is drafted.
- Ask for GPU access before any GPU benchmark; never report a GPU result from a CPU run. Use the scratch directory for temporary files, not the repository.
- Verify compatibility claims with warnings unmuted (`-W always`) and across all entry points (scripts, CLI, data pipeline, tests); state what a check does not cover.

Anytime we learn something that could help future sessions, add it to this file or the matching skill: non-obvious gotchas, subtle bugs under specific conditions, repeated corrections of coding agents.
**Creating a new skill (a directory under `.claude/skills/`) needs the user's validation first**: propose name, description and outline (in the conversation or as an issue comment), create it once the user agrees. The same holds for restructuring existing skills.
