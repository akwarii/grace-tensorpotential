# CLAUDE.md

GRACE / `tensorpotential`: machine-learning interatomic potentials (Graph Atomic Cluster Expansion) with the `gracemaker` training CLI,
an ASE calculator, UQ tools and foundation-model loading. The library is TensorFlow 2 (Keras legacy mode); a PyTorch backend that loads
TF-trained models is being added next to it. Core abstractions: an *instruction graph* (`TPInstruction` subclasses described in a
`model.yaml`) executed by `TPModel`, trained by `TensorPotential`, and served by `TPCalculator`.

## Precedence

Everything below is a default. **The user's instructions, and the issue you were asked to work on, take precedence over this file and over the
skills.** An issue states the agreed scope of one task: what may change, what must not, and how it is verified. If it (or the user) asks for
something this file discourages, for example reformatting or restructuring legacy code in a clean-up milestone, do what the issue says within its
scope, keep the safeguards it names (coverage, behaviour-preserving checks, baselines), and note the conflict in a comment on the issue. If the
user and an issue disagree, ask.

## Development commands

```bash
# Environment (creates .venv from uv.lock; TensorFlow 2.20 on CPU works locally, no GPU on most dev machines)
uv sync --group dev          # the dev group holds pytest, pytest-cov, pytest-xdist, ruff and ty

# Tests: run from the repository root (they do not depend on the working directory and write nothing into the tree)
# (under -n N, every worker gets cores/N TensorFlow and OpenMP threads from conftest.py; variables you set yourself win)
uv run --frozen --no-sync pytest tests -q -n 4 --dist load \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py
uv run --frozen --no-sync pytest tests/test_instructions.py -vv      # one file, serial
uv run --frozen --no-sync pytest tests -q -n 4 --dist load -m "not slow" \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py   # fast development loop, about 4 to 5 minutes (TEST4)

# Lint / format / types (dev group pins ruff==0.16.7 and ty==0.0.84; ty is pre-1.0, expect rule changes when bumping)
uv run --frozen --no-sync ruff check path/to/file.py       # strict set in the new packages; elsewhere E, F, ERA001
uv run --frozen --no-sync ruff format --preview path/to/new_file.py      # NEW files only, never reformat existing files
uv run --frozen --no-sync ty check path/to/new_package     # strict in the new packages; [[tool.ty.overrides]] relax legacy
uv run --frozen --no-sync python tools/lint_ratchet.py check    # legacy ruff/ty counts per (file, rule) may not rise
uv run --frozen --no-sync python tools/lint_ratchet.py record   # after a drop: records the lower baseline (a rise needs --allow-rise)
prek install                  # hooks on the changed files (.pre-commit-config.yaml): strict ruff, ty, ratchet

# Coverage gate (TEST3): a full run with branch coverage takes about 21 minutes with -n 4
uv run --frozen --no-sync pytest tests -q -n 4 --dist load --cov=tensorpotential --cov-branch --cov-report=json:cov.json \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py
python tools/check_touched_coverage.py --coverage cov.json [--base origin/torch-backend]  # every def/method/class body the diff touches: 90%
python tools/check_touched_coverage.py --coverage cov.json --unit path/to/file.py::Class.method   # name a unit whatever the diff says
python tools/coverage_ratchet.py check cov.json            # no file may cover a smaller share than baselines/coverage_baseline.json

# Compare against the untouched-tree baselines (see baselines/README.md)
python tools/ast_manifest.py check baselines/ast_manifest.json
python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json
python tools/check_clones.py check                  # duplicated functions may not increase (baselines/clone_baseline.json)
```

The full suite takes about 20 minutes serially and about 8 minutes with `-n 4` (14 cores, 30 GB; idle machine, 1,277 tests, see `cleanup_report.md` section 8). Two test files are not part of a normal
run: `test_structured_grid.py` is skipped on public master (it needs the non-existent `tensorpotential.experimental`; it is ignored above only so that
counts match `baselines/`), and `test_foundation_model_regression.py` needs foundation-model weights, which are only available on the HPC.
A test writes only into `tmp_path` (or a scratch directory), never into the working directory or the source tree; `git status` is clean after a run.

## Code style

- Python 3.11+, `from __future__ import annotations` in new modules, type hints on everything new (checked by `ty`).
- **By default legacy code stays as it is**: no repo-wide reformat (`ruff format` would rewrite most files), no file splitting (`instructions/compute.py`
  is about 4,900 lines and upstream merges depend on its shape), no renames, and improve a unit only while you are already changing it. The exception
  is an issue whose job is to change legacy code (clean-up, code quality, deduplication, packaging, performance): it says what may change and how,
  and the coverage and comparison rules below then govern the change. Do not widen the change beyond what the issue names.
- New packages get the strict ruff set (I, UP, B, SIM, C4, RUF, PD, NPY, PIE, PLE, PLW, PERF, RET, PTH, T20, ERA, W and the
  complexity, docstring, exception and security families) through the **single ruff configuration in `pyproject.toml`** (no `ruff.toml` anywhere: the strict families are
  selected for the whole repository and switched off by `per-file-ignores` for everything outside the new packages, which are listed there and in `tools/lint_ratchet.py`).
- Docstrings on new public API use the numpy convention. No commented-out code, no `print` in library code (use `logging.getLogger(__name__)`),
  no bare `except`, no `TODO` without an issue link, no mutable or shared default arguments.
- Small functions (complexity at most 10, at most 6 arguments). Prefer pure functions and frozen dataclasses for specifications. Errors are
  typed, actionable and raised early; nothing falls back silently.
- Physics names (`N806`, `E741`) are allowed. Units are eV, eV/Angstrom, eV/Angstrom^3 throughout.

## Testing conventions

- A function or class you **modify** must already be covered at 90% or more (line and branch). Below that, write characterization tests
  first, green on the unmodified code, in their own commit, then change the code. Comment-only and annotation-only edits are exempt.
- Tests have two layers: **logic** (branches, errors, shapes, edge cases) and **physical values** from an oracle that does not call the unit
  under test: finite-difference forces and stress, rotation/translation/permutation invariance, extensivity, sympy or scipy references,
  hand-computed numbers. A refactor may change the logic and the physics tests must still pass.
- Prefer real objects to mocks. A mock, stub or `monkeypatch` is acceptable at an external boundary (network, clock, absent hardware) with a
  comment saying what it replaces; never for the unit under test or for a numeric or physical path.
- Tolerances come from one named table (`tests/tolerances.py`; `numpy.isclose` semantics: `atol + rtol*|reference|`); never inline a number, never widen a tolerance to
  make a test pass.
- Tests must not depend on the working directory or on each other's output: use `tmp_path`, per-worker directories, fixed seeds, no network.
- Markers: `slow` (30 s or more; registered in `pytest.ini`; put `@pytest.mark.slow` on a test of that size, 16 so far; under `-n` the root `conftest.py` starts the marked tests first, `--slow-first=off` disables that), `gpu`, `hpc`, `tf` (these three are not registered yet); unavailable capabilities skip locally but fail in CI jobs that require them. Nothing is skipped by default; `-m "not slow"` is the fast loop.
- A model, calculator or result built once and used by several tests is read-only for them: fingerprint it when it is built and compare when its scope ends (`tests/shared_models.py`: `weights_fingerprint`, `CuTwoLayerModels`). A test that saves, trains or edits a model builds its own.
- Timings: wall-clock figures are comparable only between runs made back to back on a machine with no other test run (another agent's suite in a second worktree about doubled the time of the same tests); under `-n N` the tests themselves run slower (about 38% at N = 4), so compare wall time to wall time and summed test times to summed test times.
- Random-weight models only; real or foundation-model weights are HPC-only.

## Architecture

```
ASE Atoms -> TPAtoms / GeometricalDataBuilder (neighbour list) -> TPModel(instructions) -> energy, forces, virial -> TPCalculator
```

- `tensorpotential/instructions/`: the instruction classes (`compute.py` geometry, basis, products, reductions; `output.py` targets and shifts).
  Each has `build` (variables, index tables), `frwrd` (forward), and its constructor arguments are captured into the saved `model.yaml`.
- `tensorpotential/tpmodel.py`: `TPModel` runs the instructions in file order and derives forces and virial from the pair energies.
  `tensorpot.py`: `TensorPotential` (training, checkpoints). `loss.py`, `metrics.py`, `utils.py`, `poly.py`, `export.py`.
- `functions/`: Clebsch-Gordan generation and coupling tables (`couplings.py`), radial functions, activations.
- `data/`: dataframes and TF datasets (`databuilder.py`, `process_df.py`), neighbour lists (matscipy by default), streaming.
- `calculator/`: `TPCalculator` (ASE), foundation-model loading. `cli/`, `scripts/`: `gracemaker`, `grace_preprocess`, `grace_predict`, ...
- `potentials/`: presets. `extra/`: model generators. `uq/`: GMM-based uncertainty. `compat/pace/`: legacy, untested, **out of scope: no change is allowed to it** (owner, 2026-10-01), whatever the issue; report findings, never fix or remove them.
- The PyTorch backend is being added in `tensorpotential/torch_backend/` with TF-free shared code in `tensorpotential/core/` (not created yet); both must import without
  TensorFlow (enforced by an import contract), and the TF extractor lives outside them.
- `tests/` (run from the repository root), `baselines/` and `tools/` (untouched-tree baselines and the tools that compare against them), `docs/`.

## Key dependencies

`tensorflow~=2.20` with `tf_keras` (the package sets `TF_USE_LEGACY_KERAS=1` on import), `numpy`, `pandas<3` (see Gotchas), `ase`, `matscipy`,
`sympy`/`scipy` (coupling tables), `pyyaml`. Optional extras (packaging in progress): `torch`, `torch-sim`, `vesin`.

## Gotchas

- `capture_init_args` writes constructor defaults into the saved `model.yaml`, so a default value is part of every saved model. Changing a
  constructor signature or default changes models that already exist; treat them as persisted API.
- Output instructions overwrite `input_data[target.name]` **in place** (`instructions/output.py`); instruction order in the yaml matters, and
  a dump must snapshot the key before and after.
- `tpmodel.__getattr__` and `instructions.__getattr__` look unused but saved yamls import classes through them. Dead-code tools flag them; keep them.
- pandas 3 breaks `data/databuilder.py` (`np.array_split(<DataFrame>)`), which is why `pandas<3` is pinned; pickles written with pandas 3 cannot
  be read by pandas 2.
- TF numerics are not bit-reproducible across processes: some float64 intermediates (`large_base`: `YI`, `B`, `BB`, ...) differ by about one ulp
  between runs. Compare through `tools/oracle_snapshot.py compare` (scaled tolerance), never with exact equality.
- A worktree has no `.venv` (git-ignored): link `.venv` and `uv.lock` from the main checkout. Never run `uv sync` there: it repoints the editable `tensorpotential`
  install of the shared `.venv` to the worktree (repair with `uv sync --frozen --group dev` in the main checkout). `uv run --frozen --no-sync` and `.venv/bin/<tool>` are safe.
- `tests/test_import_gates.py` fails when a name that dead-code tools cannot see stops resolving (`from tensorpotential.X import name` in code, tests, docs and notebooks, `__cls__` strings,
  every module imported on its own in a fork of one interpreter that has imported the package, the two `__getattr__` shims). Its allow-lists (`KNOWN_ABSENT_PACKAGES`, `KNOWN_STALE_SOURCES`, `BASELINED_IMPORT_FAILURES`) name what is already broken; shrink them, never grow them silently.
- `test_graph_split.py` writes `temp_saved_model_test/` into the working directory and removes it at the end; a killed run leaves it behind (untracked), delete it before committing.
- A coverage run writes `.coverage*` into the working directory; another `--cov` run started in the same directory while a full-suite run is going (even of a tool test) is combined into its report (`tools/check_clones.py` appeared in a library report that way). Run one coverage job per worktree, or filter the report before `coverage_ratchet.py record`.
- A new worktree has no `uv.lock` (it is git-excluded): copy it from another tree, then `uv sync --frozen --offline`.
- Forces are `-dE/d(bond_vector)` with `F = segment_sum(pair_f, ind_j) - segment_sum(pair_f, ind_i)`; virial is `sum(pair_f (x) D)`; the ASE stress is
  `-virial / V` with Voigt reorder `[0, 1, 2, 5, 4, 3]`.
- The TF calculator's `enforce_pbc` edits the caller's `Atoms` in place and makes every axis periodic; new code must not copy that behaviour silently.
- A git worktree has no `.venv`: run with the main checkout's interpreter and `PYTHONPATH=$PWD` (print `tensorpotential.__file__` once, because the
  editable install otherwise resolves to the main tree), and give `ty` the environment with `--python <main>/.venv`. `--cov=<dotted module path>` failed once at
  conftest import with an ImportError (cause not isolated); use `--cov=tensorpotential`.
- `CollectInvarBasis` cannot be instantiated as shipped (it lacks the abstract `upd_init_args_new_elements`); `tests/test_spbf_layout_opt.py` and
  `tests/test_function_reduce_builds.py` subclass it with a stub. `ConstantScaleShiftTarget` sorts `atomic_shift_map` by key, so keys must be element indices.
- `FCRight2Left.build` draws random numbers for `w_right` even with `init_vars="zeros"`; `compat/pace` has a latent `NameError` (`rankmax`).
- Stage-style work (cleanup, packaging, torch backend) is tracked as issues on the fork's board; use the skills in `.claude/skills/` for the
  procedure (`grace-torch`, `grace-torch-ticket`, `grace-torch-tests`, `grace-torch-goldens`, `grace-torch-numerics`).

## Working agreements

- **You may commit, and once the work of your issue is ready you open a draft pull request for it yourself and ask the user to review it. You never merge it, mark it ready for review or turn on
  auto-merge: the user does.** Commit on the work branch of the issue (one concern per commit, imperative message with `feat:`, `fix:`, `refactor:`, `test:`, `docs:` or `chore:`, test commits before
  the change they protect, never on `torch-backend` or `master`). *Ready* means: the exit criterion is met, the checks the issue names are green, and the description made with `python tools/board.py
  pr-body <ID>` passes `pr-check`. Push that branch (only that branch) to the fork when you open the draft PR into `torch-backend` and to update it after review; then `python tools/board.py status <ID>
  "PR Open"` and tell the user the PR number and what to look at. Nothing is ever pushed to, or opened against, the upstream `ICAMS` repositories without an explicit go for that unit (so no `pr/U*`
  branch is pushed or opened on your own); the `upstream` remote has its push URL disabled.
- **Several agents may work at the same time, so each works in its own git worktree, and removes it when its pull request is merged.** Create it from the integration branch,
  `git worktree add -b <id>-<slug> ../<repository>-<id> origin/torch-backend`, then run `git branch --unset-upstream` in it (git sets the upstream to `torch-backend`, and a plain push would
  update that branch). Never switch branches in, or run a branch-changing command on, a tree that another agent or the user is using. One agent per issue: `board.py` rewrites the whole issue body
  when it ticks a box, so two agents on the same issue overwrite each other. After the merge, and once `board.py done` has run: `git worktree remove ../<repository>-<id>` (without `--force`; if it
  refuses, the tree holds uncommitted work, so report it) and `git branch -d <branch>`. Leave the remote branch unless the user says to delete it.
- Ask before any download (file, source, size) and before any other outward action (creating or closing issues, repository settings); the draft PR for your own issue is covered by the rule above.
  Development dependencies (test, coverage, parallelism, lint, typing, mutation tools) may be added to the `dev` group; runtime dependencies are decisions.
- New issues (only on the user's go) start from `.github/ISSUE_TEMPLATE/milestone.md`; see "Creating an issue" in the `grace-torch-ticket` skill.
- Pull-request descriptions use the fork's template (`.github/PULL_REQUEST_TEMPLATE/torch-backend.md`, generated with `python tools/board.py pr-body <ID>` and checked with
  `pr-check`); the rules for writing the summary are in the `grace-torch-ticket` skill.
- **An issue is resolved only once a pull request that references it (`Refs #<issue>`) is merged into `torch-backend` on this fork.** `tools/board.py done` checks that and
  refuses otherwise; `--waive-pr` only when the user says so (for example for a gate or a repository setting). The user merges.
- Work on an issue follows its protocol (`grace-torch-ticket` skill): Todo to In Progress when starting, unexpected findings commented on the issue, and the
  **Definition of Done checkboxes ticked every time a task of the issue is finished** (`tools/board.py check`), with evidence.
- Ask for GPU access before any GPU benchmark; do not report a GPU result from a CPU run.
- Use the scratch directory for temporary files, not the repository.
- **Keep the issues current: fix stale text.** When a task changes something an issue mentions (a path, command, count, decision, tool, dependency line or
  exit criterion), update that issue in the same task; run `python tools/board.py lint` and fix every `STALE` finding, including ones you did not cause.
- **The fork is public: sanitise everything you post** (issues, comments, PR descriptions, commit messages). `tools/board.py` sanitises automatically what it posts;
  pass any other text through `python tools/board.py sanitise FILE`. Sanitising means: no personal data, institution names, e-mail addresses, absolute local
  paths or research-application details; no bare `#N` that is not an issue of this repository; no `@mentions` outside code spans; no links to other
  repositories' issues or pull requests (they create back-links there). When an issue you touch needs it, sanitise its existing text too.
- `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` are fork-only. They are committed on `torch-backend` but must never appear on an upstream branch
  (`pr/U*`); check with `python tools/check_pr_branch.py` before any upstream PR text is drafted.
- Verify compatibility claims with warnings unmuted (`-W always`) and across all entry points (scripts, CLI, data pipeline, tests), and state
  what a check does not cover.

Anytime we learn something that could be beneficial in future coding sessions, add it to this file or to the matching existing skill:
gotchas that are not obvious, subtle bugs that appear under specific conditions, and repeated corrections made to the output of coding agents.
**Creating a new skill (a new directory under `.claude/skills/`) is allowed only after the user has validated it**: propose its name, description and outline first
(in the conversation or as a comment on the issue), and create it once the user agrees. The same holds for restructuring existing skills.
