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

# Tests: run from the repository root; they write nothing into the tree. Under -n N every worker gets cores/N TF and OpenMP threads (conftest.py).
uv run --frozen --no-sync pytest tests -q -n 4 --dist load \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py
uv run --frozen --no-sync pytest tests/test_instructions.py -vv      # one file, serial
# fast loop: add -m "not slow" to the full command

# Lint / format / types (dev group pins ruff==0.16.7 and ty==0.0.84; ty is pre-1.0, expect rule changes when bumping)
uv run --frozen --no-sync ruff check path/to/file.py       # strict set in the new packages; elsewhere E, F, ERA001
uv run --frozen --no-sync ruff format --preview path/to/new_file.py      # NEW files only, never reformat existing files
uv run --frozen --no-sync ty check path/to/new_package     # strict in the new packages; [[tool.ty.overrides]] relax legacy
uv run --frozen --no-sync python tools/lint_ratchet.py check    # legacy ruff/ty counts per (file, rule) may not rise; `record` after a drop (a rise needs --allow-rise)
prek install                  # hooks on the changed files (.pre-commit-config.yaml): strict ruff, ty, ratchet

# Coverage and baselines (procedures: skills grace-torch-tests and grace-torch-goldens; baselines/README.md)
uv run --frozen --no-sync pytest tests -q -n 4 --dist load --cov=tensorpotential --cov-branch --cov-report=json:cov.json <same --ignore options>
python tools/check_touched_coverage.py --coverage cov.json [--base origin/torch-backend] [--unit path/to/file.py::Class.method]   # touched units: 90%
python tools/coverage_ratchet.py check cov.json       # a file fails when its share falls below baselines/coverage_baseline.json and its uncovered count rises
python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json     # also tools/ast_manifest.py and tools/check_clones.py
```

**Rough runtime** (14 cores, 30 GB, about 1,340 tests; measured 2026-10-03, re-measure before relying on it): the full suite takes about 10 minutes with `-n 4` on an idle machine (12 with `--cov`) and about 20 serially; `-m "not slow"` about 4 to 5 minutes. Another suite running in a second worktree about doubles that (see the idle-machine rule in Working agreements). Compare timings only between back-to-back runs (wall time to wall time).

Two test files are not part of a normal run: `test_structured_grid.py` is skipped on public master (it needs the non-existent `tensorpotential.experimental`; it is ignored above only so that
counts match `baselines/`), and `test_foundation_model_regression.py` needs foundation-model weights, which are only available on the HPC.
A test writes only into `tmp_path` (or a scratch directory), never into the working directory or the source tree; `git status` is clean after a run.

## Code style

- Python 3.11+, `from __future__ import annotations` in new modules, type hints on everything new (checked by `ty`).
- **By default legacy code stays as it is**: no repo-wide reformat (`ruff format` would rewrite most files), no file splitting (`instructions/compute.py`
  is about 4,900 lines and upstream merges depend on its shape), no renames, and improve a unit only while you are already changing it. The exception
  is an issue whose job is to change legacy code (clean-up, code quality, deduplication, packaging, performance): it says what may change and how,
  and the coverage and comparison rules below then govern the change. Do not widen the change beyond what the issue names.
- New packages get the strict ruff set (I, UP, B, SIM, C4, RUF, PD, NPY, PIE, PLE, PLW, PERF, RET, PTH, T20, ERA, W and the complexity, docstring, exception and security families) through the **single ruff configuration in `pyproject.toml`**
  (no `ruff.toml`; the strict families are selected repo-wide and switched off by `per-file-ignores` outside the new packages, which are listed there and in `tools/lint_ratchet.py`).
- Docstrings on new public API use the numpy convention. No commented-out code, no `print` in library code (use `logging.getLogger(__name__)`),
  no bare `except`, no `TODO` without an issue link, no mutable or shared default arguments.
- Small functions (complexity at most 10, at most 6 arguments). Prefer pure functions and frozen dataclasses for specifications. Errors are
  typed, actionable and raised early; nothing falls back silently.
- Physics names (`N806`, `E741`) are allowed. Units are eV, eV/Angstrom, eV/Angstrom^3 throughout.

## Testing conventions

Procedures, planted mutants and speed tips are in the `grace-torch-tests` skill.

- A function or class you **modify** must already be covered at 90% or more (line and branch). Below that, write characterization tests
  first, green on the unmodified code, in their own commit, then change the code. Comment-only and annotation-only edits are exempt.
- **A test goes in the file named after the source module it covers** (`tests/test_<module>.py` for `tensorpotential/<...>/<module>.py`, for example `tests/test_tp_model.py` for `tpmodel.py`, `tests/test_process_df.py` for `data/process_df.py`); `tests/` stays flat. Do not group tests by the issue or the kind of change that added them, and do not move existing test files unless an issue says so (owner, 2026-10-03).
- Two layers: **logic** (branches, errors, shapes, edge cases) and **physical values** from an oracle that does not call the unit under test
  (finite differences, rotation/translation/permutation invariance, extensivity, sympy or scipy references, hand-computed numbers). A refactor may change the logic and the physics tests must still pass.
- Prefer real objects to mocks; a mock or `monkeypatch` only at an external boundary (network, clock, absent hardware), with a comment saying what it replaces, never for the unit under test or a numeric path.
- Tolerances come from one named table (`tests/tolerances.py`; `numpy.isclose` semantics: `atol + rtol*|reference|`); never inline a number, never widen a tolerance to make a test pass.
- Tests must not depend on the working directory or on each other's output: `tmp_path`, per-worker directories, fixed seeds, no network. Random-weight models only; real or foundation-model weights are HPC-only.
- Markers: `slow` (30 s or more, registered in `pytest.ini`; the root `conftest.py` starts them first under `-n`), `gpu`, `hpc`, `tf` (these three are not registered yet); unavailable capabilities skip locally but fail in CI jobs that require them. Nothing is skipped by default.
- A model, calculator or result built once and used by several tests is read-only for them: fingerprint it when it is built and compare when its scope ends (`tests/shared_models.py`: `weights_fingerprint`, `CuTwoLayerModels`). A test that saves, trains or edits a model builds its own.

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
- `np.array_split(<DataFrame>)` broke `data/databuilder.py` on pandas 3; DEPS1 fixed it (`_split_rows`), and the suite is green on pandas 3.0.3 and 2.3.3, but `pandas<3` stays pinned
  (decision D11) and pickles written with pandas 3 cannot be read by pandas 2. `pd.concat(..., copy=False)` in `cli/data.py` still raises `Pandas4Warning` on pandas 3. A second pandas environment:
  `UV_PROJECT_ENVIRONMENT=<dir> uv sync --frozen --offline` then `uv pip install --offline --python <dir>/bin/python pandas==3.0.3`; put its `bin` on `PATH` (the tests start `grace_preprocess` by name).
- TF numerics are not bit-reproducible across processes: some float64 intermediates (`large_base`: `YI`, `B`, `BB`, ...) differ by about one ulp
  between runs (float32-parameter snapshots did not vary in seven repeats). Compare through `tools/oracle_snapshot.py compare` (a named scaled tolerance per precision), never with exact
  equality for float64. `baselines/oracle_snapshot_wide.npz` adds float32, `lm_first`, `dense_nbr`, edge structures and presets (`baselines/README.md`); a `git archive` copy of a tag is imported with `PYTHONPATH` from a directory outside both trees.
- `GeometricalDataBuilder.extract_from_ase_atoms` edits its argument: `enforce_pbc` gives a non-periodic `Atoms` a cell, centres it and sets `pbc=True`. Pass a copy when the structure is reused (stress would otherwise be divided by an invented volume).
- `GeometricalDataBuilder.extract_from_ase_atoms`: an atom with no neighbour gets one dummy bond `(i, 0)` (vector far outside the cutoff, scrambled across atoms for several isolated atoms in an unequal-sided cell), so it never has zero bonds; a zero third cell vector with partial `pbc` raises `LinAlgError`; `pbc=(T,T,F)` is overridden by `enforce_pbc`. All pinned by tests (`tests/test_databuilder.py`, `tests/test_utils.py`); the brute-force reference is `tests/neighbour_oracle.py`.
- Checkpoint keys are not `variable.name`: a variable is saved as `model/instructions/<instruction>/<attribute path>/.ATTRIBUTES/VARIABLE_VALUE` (`A1/w_left_FC:0` is saved under `.../A1/w_left/...`); `tools/grace_probe.py` writes both (`baselines/probes/`). Each model also has two non-float variables (`Z/element_map_symbols`, a string, and `Z/element_map_index`, int32). With float32 parameters the radial basis, the bond vectors and the spherical harmonics still compute in float64 and `A`/`YI`/`R` cast down; with float64 parameters the swish `beta` is a float32 literal (value 1.0) cast up. Timings of the probe are quotable only when `timings_foreign_cpu_cores` in the JSON is near 0.
- `lm_first=True` fails on yamls whose output is `MLPOut2ScalarTarget` (it has no transpose for it, unlike `LinMLPOut2ScalarTarget`); `model_grace.yaml` is such a model.
- **Worktrees.** A worktree has no `.venv` or `uv.lock` (both git-ignored): link `.venv` and copy `uv.lock` from the main checkout. Never run `uv sync` there: it repoints the editable
  `tensorpotential` install of the shared `.venv` to the worktree (repair with `uv sync --frozen --group dev` in the main checkout); `uv run --frozen --no-sync` and `.venv/bin/<tool>` are safe.
  Run with `PATH=<worktree>/.venv/bin:$PATH` (subprocess tests call `grace_preprocess`) and `PYTHONPATH=$PWD`, and print `tensorpotential.__file__` once (the editable install otherwise resolves to the main tree);
  give `ty` the environment with `--python <main>/.venv`; use `--cov=tensorpotential`; in a scratch script import `tensorpotential` before `tensorflow`, or Keras 3 is used.
- `tests/test_import_gates.py` fails when a name that dead-code tools cannot see stops resolving (`from tensorpotential.X import name` anywhere, `__cls__` strings, every module imported on its own, the two `__getattr__` shims).
  Its allow-lists (`KNOWN_ABSENT_PACKAGES`, `KNOWN_STALE_SOURCES`, `BASELINED_IMPORT_FAILURES`) name what is already broken; shrink them, never grow them silently.
- A coverage run writes `.coverage*` into the working directory; a second `--cov` run in the same directory while a full run is going is combined into its report. Run one coverage job per worktree, or filter the report before `coverage_ratchet.py record`.
- `CollectInvarBasis` cannot be instantiated as shipped (it lacks the abstract `upd_init_args_new_elements`); tests subclass it with a stub. `ConstantScaleShiftTarget` sorts `atomic_shift_map` by key, so keys must be element indices.
- `tools/junit_outcomes.py summarize` needs `--log <pytest log>` with `-rX` lines to know an XPASS; without it `test_construct_batches_multiple_db` reads as `xpassed -> passed` in `compare` (not a change). `baselines/ast_manifest.json` is the untouched tree: to check that a branch changes only some files, `ast_manifest.py write --rev origin/torch-backend <file>` first and `check` against that.
- Units, signs, the force, virial and stress formulas, and the TF calculator's `enforce_pbc` behaviour are in the `grace-torch-numerics` skill.
- Stage-style work (cleanup, packaging, torch backend) is tracked as issues on the fork's board; use the skills in `.claude/skills/` for the
  procedure (`grace-torch`, `grace-torch-ticket`, `grace-torch-tests`, `grace-torch-goldens`, `grace-torch-numerics`).

## Working agreements

The procedure (board commands, PR text, findings, Definition of Done) is in the `grace-torch-ticket` skill; these are the rules that hold even without it.

- **You may commit, and once the work of your issue is ready you open a draft pull request into `torch-backend` yourself and ask the user to review it. You never merge it, mark it ready for review or turn on
  auto-merge: the user does.** *Ready* means: the exit criterion is met, the checks the issue names are green, and the description made with `python tools/board.py pr-body <ID>` passes `pr-check`.
  Commit on the work branch of the issue (one concern per commit, imperative message with `feat:`, `fix:`, `refactor:`, `test:`, `docs:` or `chore:`, test commits before the change they protect, never on `torch-backend` or `master`);
  push only that branch to the fork; then `python tools/board.py status <ID> "PR Open"` and tell the user the PR number and what to look at. Nothing is ever pushed to, or opened against, the upstream `ICAMS` repositories without an explicit go for that unit
  (so no `pr/U*` branch is pushed or opened on your own); the `upstream` remote has its push URL disabled.
- **Each agent works in its own git worktree** (`git worktree add -b <id>-<slug> ../<repository>-<id> origin/torch-backend`, then `git branch --unset-upstream` so a plain push cannot update `torch-backend`), one agent per issue
  (`board.py` rewrites the whole issue body when it ticks a box). Never switch branches in, or run a branch-changing command on, a tree that another agent or the user is using. After the merge and `board.py done`:
  `git worktree remove` (no `--force`; if it refuses, report the uncommitted work) and `git branch -d`; leave the remote branch unless the user says otherwise.
- **Check that the machine is idle before you run tests** (a pytest run of any size under `-n`, a coverage run, a baseline run, a `gracemaker` training or any other TensorFlow job): run `uptime` and
  `ps -eo pid,etimes,args | grep '[p]ytest\|[g]racemaker'` first. The machine is **busy** when another test or training process is running (in any worktree, yours or another agent's) or the 1-minute load average is above 4 (14 cores).
  When it is busy, start no full, `-n` or coverage run; do work that needs no CPU (edits, `ruff`, `ty`, one test file serially), look again later, and tell the user when waiting blocks you. Run one suite at a time, also your own.
  Never kill another agent's process, and never use a run made on a busy machine for a timing or baseline claim. Details and pitfalls: "Speed" in the `grace-torch-tests` skill.
- Ask before any download (file, source, size) and before any other outward action (creating or closing issues, repository settings); the draft PR for your own issue is covered above. New issues start only on the user's go, from `.github/ISSUE_TEMPLATE/milestone.md`.
  Development dependencies (test, coverage, parallelism, lint, typing, mutation tools) may be added to the `dev` group; runtime dependencies are decisions.
- **An issue is resolved only once a pull request that references it (`Refs #<issue>`) is merged into `torch-backend` on this fork.** `tools/board.py done` checks that and
  refuses otherwise; `--waive-pr` only when the user says so. Work follows the issue's protocol: Todo to In Progress when starting, unexpected findings commented on the issue,
  and the **Definition of Done checkboxes ticked every time a task of the issue is finished** (`tools/board.py check`), with evidence.
- **Keep the issues current: fix stale text.** When a task changes something an issue mentions (a path, command, count, decision, tool, dependency line or exit criterion), update that issue in the same task; run `python tools/board.py lint` and fix every `STALE` finding.
- **The fork is public: sanitise everything you post** (issues, comments, PR descriptions, commit messages). `tools/board.py` sanitises automatically what it posts;
  pass any other text through `python tools/board.py sanitise FILE`. Sanitising means: no personal data, institution names, e-mail addresses, absolute local
  paths or research-application details; no bare `#N` that is not an issue of this repository; no `@mentions` outside code spans; no links to other
  repositories' issues or pull requests (they create back-links there). When an issue you touch needs it, sanitise its existing text too.
- `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` are fork-only. They are committed on `torch-backend` but must never appear on an upstream branch
  (`pr/U*`); check with `python tools/check_pr_branch.py` before any upstream PR text is drafted.
- Ask for GPU access before any GPU benchmark; do not report a GPU result from a CPU run. Use the scratch directory for temporary files, not the repository.
- Verify compatibility claims with warnings unmuted (`-W always`) and across all entry points (scripts, CLI, data pipeline, tests), and state what a check does not cover.

Anytime we learn something that could be beneficial in future coding sessions, add it to this file or to the matching existing skill:
gotchas that are not obvious, subtle bugs that appear under specific conditions, and repeated corrections made to the output of coding agents.
**Creating a new skill (a new directory under `.claude/skills/`) is allowed only after the user has validated it**: propose its name, description and outline first
(in the conversation or as a comment on the issue), and create it once the user agrees. The same holds for restructuring existing skills.
