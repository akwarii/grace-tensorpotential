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
uv sync

# Tests: run from the repository root (they do not depend on the working directory and write nothing into the tree)
uv run --frozen --no-sync --with pytest --with pytest-xdist pytest tests -q -n 4 --dist load \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py
uv run --frozen --no-sync --with pytest pytest tests/test_instructions.py -vv      # one file, serial

# Lint / format / types (versions pinned: ruff 0.16.7; ty is pre-1.0)
uvx ruff@0.16.7 check path/to/file.py
uvx ruff@0.16.7 format --preview path/to/new_file.py      # NEW files only, never reformat existing files
uv tool run ty check path/to/new_package

# Compare against the untouched-tree baselines (see baselines/README.md)
python tools/ast_manifest.py check baselines/ast_manifest.json
python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json
```

The full suite takes about 32 minutes serially and about 10 minutes with `-n 4` (14 cores, 30 GB). Two test files are not part of a normal
run: `test_structured_grid.py` is skipped on public master (it needs the non-existent `tensorpotential.experimental`; it is ignored above only so that
counts match `baselines/`), and `test_foundation_model_regression.py` needs foundation-model weights, which are only available on the HPC.
A test writes only into `tmp_path` (or a scratch directory), never into the working directory or the source tree; `git status` is clean after a run.

## Code style

- Python 3.10+, `from __future__ import annotations` in new modules, type hints on everything new (checked by `ty`).
- **By default legacy code stays as it is**: no repo-wide reformat (`ruff format` would rewrite most files), no file splitting (`instructions/compute.py`
  is about 4,900 lines and upstream merges depend on its shape), no renames, and improve a unit only while you are already changing it. The exception
  is an issue whose job is to change legacy code (clean-up, code quality, deduplication, packaging, performance): it says what may change and how,
  and the coverage and comparison rules below then govern the change. Do not widen the change beyond what the issue names.
- New packages get the strict ruff set (I, UP, B, SIM, C4, RUF, PD, NPY, PIE, PLE, PLW, PERF, RET, PTH, T20, ERA, W and the
  complexity, docstring, exception and security families) through a nested `ruff.toml` **inside the new package only**, never in `tensorpotential/`
  itself (a nested config applies to every file below its directory).
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
- Tolerances come from one named table (`numpy.isclose` semantics: `atol + rtol*|reference|`); never inline a number, never widen a tolerance to
  make a test pass.
- Tests must not depend on the working directory or on each other's output: use `tmp_path`, per-worker directories, fixed seeds, no network.
- Markers: `slow` (30 s or more), `gpu`, `hpc`, `tf`; unavailable capabilities skip locally but fail in CI jobs that require them.
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
- Forces are `-dE/d(bond_vector)` with `F = segment_sum(pair_f, ind_j) - segment_sum(pair_f, ind_i)`; virial is `sum(pair_f (x) D)`; the ASE stress is
  `-virial / V` with Voigt reorder `[0, 1, 2, 5, 4, 3]`.
- The TF calculator's `enforce_pbc` edits the caller's `Atoms` in place and makes every axis periodic; new code must not copy that behaviour silently.
- `FCRight2Left.build` draws random numbers for `w_right` even with `init_vars="zeros"`; `compat/pace` has a latent `NameError` (`rankmax`).
- Stage-style work (cleanup, packaging, torch backend) is tracked as issues on the fork's board; use the skills in `.claude/skills/` for the
  procedure (`grace-torch`, `grace-torch-ticket`, `grace-torch-tests`, `grace-torch-goldens`, `grace-torch-numerics`).

## Working agreements

- **You may commit, but you may not open or merge a pull request unless you are told to.** Commit on the work branch of the issue (one concern per commit, imperative message with
  `feat:`, `fix:`, `refactor:`, `test:`, `docs:` or `chore:`, test commits before the change they protect, never on `torch-backend` or `master`). Push a work branch to the fork only when told
  to push or to open the PR. **Opening a pull request needs an explicit instruction from the user, and so does merging one** (the user merges). Nothing is ever pushed to, or opened against,
  the upstream `ICAMS` repositories without an explicit go for that unit; the `upstream` remote has its push URL disabled.
- Ask before any download (file, source, size) and before any outward action (creating or closing issues, PRs, repository settings).
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
