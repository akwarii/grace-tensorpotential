# CLAUDE.md

GRACE / `tensorpotential`: machine-learning interatomic potentials (Graph Atomic Cluster Expansion) with the `gracemaker` training CLI (also `grace train`: one `grace <subcommand>` command runs all the console tools), an ASE calculator,
UQ tools and foundation-model loading. The library is TensorFlow 2 with Keras 3; a PyTorch backend that loads TF-trained models is being added next
to it. Core abstractions: an *instruction graph* (`TPInstruction` subclasses described in a `model.yaml`) run by `TPModel`, trained by `TensorPotential`,
served by `TPCalculator`.

## Precedence

Everything below is a default. **The user's instructions, and the issue you were asked to work on, take precedence over this file and the skills.** An issue
states the agreed scope of one task: what may change, what must not, how it is verified. If it (or the user) asks for something this file discourages (for
example restructuring legacy code in a clean-up milestone), do what the issue says within its scope, keep the safeguards it names (coverage,
behaviour-preserving checks, baselines), and note the conflict in a comment on the issue. If the user and an issue disagree, ask.

## Working agreements

Procedures: board commands, findings, Definition of Done: `grace-torch-ticket`; branch, tidy-up pass, PR text, commits: `grace-torch-pr`. These rules hold even without them.

- **You may commit, and once the work of your issue is ready you open a draft pull request into `torch-backend` yourself and ask the user to review it. You never merge it, mark it ready for review or turn on auto-merge: the user does.** *Ready* means: the exit criterion is met,
  the checks the issue names are green, the tidy-up pass below is done, the branch contains the current `torch-backend`, and the description from `python tools/board.py pr-body <ID>` passes `pr-check`. Commit on the issue's work branch (one concern per commit, never on `torch-backend` or `master`), push only that branch to the fork,
  then `python tools/board.py status <ID> "PR Open"` and tell the user the PR number and what to look at. Nothing is pushed to, or opened against, the upstream `ICAMS` repositories without an explicit go for that unit (no `pr/U*` branch on your own; the `upstream` remote has its push URL disabled).
- **A planning gap (an exit criterion or Definition of Done box that the issue itself makes unreachable, e.g. a unit below 90% coverage to modify while new tests are out of scope) is a question to the user, not your decision.** First look through the other open issues (`python tools/board.py list`, `board.py context <ID>`, or one `gh issue list --repo akwarii/grace-tensorpotential --search "<unit or file>" --state open`) for one that covers the missing work or must run first: then it is an ordering problem.
  Ask with the gap, what you found, the options (run the other issue first, widen the scope, split, waive, change the exit) and your recommendation, in chat and as `board.py finding`. Until the answer do not widen the scope, tick or `na` the box, use `--waive` / `--waive-pr`, or lower a threshold; continue with what the gap does not touch.
- **Tidy up your code before you open the PR:** four reviews of the diff against `origin/torch-backend`, one per lens (reuse, simplification, efficiency: only what a measurement shows, R7; altitude), as subagents or with the `simplify` skill. Fixes go in `refactor:` commits, the outcome per lens in the PR's Review focus; a lens touches only what the PR already changes (the rest is a finding), and a documentation-only diff is exempt. Procedure: `grace-torch-pr`.
- **Merge `origin/torch-backend` into your branch (a merge, never a rebase) before you open the PR, and watch CI after every push:** `gh pr checks <n> --repo akwarii/grace-tensorpotential` about every 5 minutes, never `--watch` or a loop (GraphQL budget), until every job has finished. Fix a red job on the branch (a finding on the issue when your diff did not cause it), merge again when the PR stops merging cleanly, and tell the user the result. Procedure: `grace-torch-pr`.
- **One agent, one issue, one git worktree** (`board.py` rewrites the whole issue body when it ticks a box). Never switch branches in, or run a branch-changing command on, a tree another agent or the user is using. After the merge and `board.py done`, remove the worktree without `--force`
  (if it refuses, report the uncommitted work); leave the remote branch unless the user says otherwise.
- **Check that the machine is idle before any test, coverage, baseline, `gracemaker` or other TensorFlow job:** `uptime` and `ps -eo pid,etimes,args | grep '[p]ytest\|[g]racemaker'`. **Busy** = another test or training process runs (any worktree) or the 1-minute load is above 4 (14 cores).
  When busy, start no full, `-n` or coverage run; do work that needs no CPU, look again later and tell the user when waiting blocks you. One suite at a time, also your own. Never kill another agent's process; never use a busy-machine run for a timing or baseline claim.
- Ask before any download (file, source, size) and before any other outward action (creating or closing issues, repository settings); the draft PR of your own issue is covered above. New issues start only on the user's go, from `.github/ISSUE_TEMPLATE/milestone.md`.
  Development dependencies (test, coverage, parallelism, lint, typing, mutation tools) may be added to the `dev` group; runtime dependencies are decisions.
- **An issue is resolved only once a PR with `Refs #<issue>` is merged into `torch-backend` on this fork** (`tools/board.py done` checks; `--waive-pr` only when the user says so). Follow the issue's protocol: Todo to In Progress when starting, unexpected findings commented on the issue,
  and the **Definition of Done boxes ticked every time a task is finished** (`tools/board.py check`), with evidence.
- **Mind the GraphQL budget** (5,000 points an hour for the whole account, shared by all agents): use `tools/board.py`, never `gh project item-list` / `field-list` (about 100 and 150 points a call), never loops of `gh` calls; `python tools/board.py budget` shows what is left.
- **Keep the issues current.** When a task changes something an issue mentions (path, command, count, decision, tool, dependency line, exit criterion), update that issue in the same task; run `python tools/board.py lint` and fix every `STALE` finding.
- **The fork is public: sanitise everything you post** (issues, comments, PR text, commit messages). `tools/board.py` sanitises what it posts; pass other text through `python tools/board.py sanitise FILE`. No personal data, institution names, e-mail addresses, absolute local paths or research-application details;
  no bare `#N` that is not an issue of this repository; no `@mentions` outside code spans; no links to other repositories' issues or pull requests (back-links). Sanitise the existing text of an issue you touch when it needs it.
- `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` are fork-only: committed on `torch-backend`, never on an upstream branch (`pr/U*`); check with `python tools/check_pr_branch.py` before any upstream PR text is drafted.
- **A Bash hook enforces part of these rules** (`tools/guard_bash.py`, `.claude/settings.json`, new sessions): it blocks GraphQL project listing, `uv sync` in a worktree, commits, merges and pushes on `torch-backend` or `master`, `gh pr merge` / `ready`, `gh pr create` without `--draft`, `gh pr` / `gh issue` writes without `--repo` or on the ICAMS repositories,
  `git worktree remove --force`, killing a test process, and heavy runs on a busy machine. Only when the user told you to run such a command, prefix it with `GRACE_HOOK_OVERRIDE=<rule>` (the block message names it).
- Ask for GPU access before any GPU benchmark; never report a GPU result from a CPU run. Use the scratch directory for temporary files, not the repository.
- Verify compatibility claims with warnings unmuted (`-W always`) and across all entry points (scripts, CLI, data pipeline, tests); state what a check does not cover.

## Where to look

| For | Look at |
|---|---|
| Board commands, findings, Definition of Done, creating and closing issues | skill `grace-torch-ticket` |
| Branch and worktree, tidy-up pass, PR text, commit messages, upstream units | skill `grace-torch-pr` |
| The rules of the torch backend (R1 to R9), the fork's issues | skill `grace-torch` |
| Coverage rule, test layers, planted mutants, parallel runs | skill `grace-torch-tests`, rule `.claude/rules/testing.md` |
| Baselines, golden fixtures, tolerances | skill `grace-torch-goldens`, `baselines/README.md`, `tests_torch/golden/README.md` |
| Units, signs, dtypes, derivatives | skill `grace-torch-numerics` |
| CI, required checks, packaging, runner memory, runtime of the suite | `.github/CI.md` |
| Spec sheets and the instruction registry | `docs/torch_backend/spec/README.md` |
| Area gotchas (instructions, data, Keras 3, TF-free packages, golden, CI) | `.claude/rules/*.md`: each loads when you read or edit a file its `paths` match |

## Architecture

```
ASE Atoms -> TPAtoms / GeometricalDataBuilder (neighbour list) -> TPModel(instructions) -> energy, forces, virial -> TPCalculator
```

- `instructions/`: instruction classes (`compute.py` geometry, basis, products, reductions; `output.py` targets and shifts). Each has `build` (variables, index tables) and `frwrd`; its constructor arguments are captured into the saved `model.yaml`.
- `tpmodel.py`: `TPModel` runs the instructions in file order and derives forces and virial from the pair energies. `tensorpot.py`: `TensorPotential` (training, checkpoints). Also `loss.py`, `metrics.py`, `utils.py`, `poly.py`, `export.py`.
- `functions/`: Clebsch-Gordan and coupling tables (`couplings.py`), radial functions, activations. `data/`: dataframes and TF datasets (`databuilder.py`, `process_df.py`), neighbour lists (matscipy by default), streaming.
- `calculator/`: `TPCalculator` (ASE), foundation-model loading. `cli/`, `scripts/`: command code only (CLI2), no library module imports them (import contract): `grace` (`scripts/grace.py`), `gracemaker`, `grace_preprocess`, `grace_predict`, ...; shared `cli/console.py` and `cli/prompts.py`. Their library side: `data/datasets.py`, `data/preprocess.py`, `model_ops.py`, `export_kokkos.py` (old paths re-export); a command's `main` configures logging, a library module never does. `potentials/`: presets. `extra/`: model generators. `uq/`: GMM-based uncertainty.
- `compat/pace/`: legacy, untested, **out of scope: no change is allowed to it** (owner, 2026-10-01), whatever the issue; report findings, never fix or remove them.
- The PyTorch backend lives in `torch_backend/` with TF-free shared code in `core/`; both must import without TensorFlow (import contract); the TF extractor lives outside them.
- `tests/` (run from the repository root), `baselines/` and `tools/` (untouched-tree baselines and the tools that compare with them), `docs/`.

Key dependencies: `tensorflow~=2.20` (Keras 3 comes with it; the package neither sets nor reads the legacy-Keras variable and warns when it is exported), `numpy`, `pandas<3` (see Gotchas), `ase`, `matscipy`, `sympy`/`scipy`, `pyyaml`; extras `torch`, `torch-sim`.

## Commands

```bash
uv sync --group dev          # .venv from the committed uv.lock; the dev group has pytest, pytest-cov, pytest-xdist, ruff, ty and the tf extra (TF is not a base dependency)

# Tests: from the repository root; they write nothing into the tree. Under -n N every worker gets cores/N TF and OpenMP threads (conftest.py).
uv run --frozen --no-sync pytest tests -q -n 4 --dist load \
    --ignore=tests/test_structured_grid.py --ignore=tests/test_foundation_model_regression.py   # fast loop: add -m "not slow"
uv run --frozen --no-sync pytest tests/test_instructions.py -vv    # one file, serial
uv run --frozen --no-sync pytest tests_torch -q                     # torch-backend tests, TF-free, seconds

# Checks (dev group pins ruff==0.16.7, ty==0.0.84; ty is pre-1.0)
uv run --frozen --no-sync ruff check PATH                  # strict set in the new packages; elsewhere E, F, ERA001
uv run --frozen --no-sync ruff format --preview NEW_FILE   # NEW files only, never reformat existing files
uv run --frozen --no-sync ty check NEW_PACKAGE             # strict in the new packages; [[tool.ty.overrides]] relax legacy
uv run --frozen --no-sync python tools/lint_ratchet.py check   # legacy counts may not rise, a new file (not in baselines/legacy_files.txt) gets the strict set; `record` after a drop, `--moved OLD NEW` after a move (also in tools/coverage_ratchet.py)
uv run --frozen --no-sync lint-imports                     # import contract: TF-free modules never reach tensorflow
uv run --frozen --no-sync deptry tensorpotential           # declared against imported dependencies (DEPS4; whitelist in pyproject.toml)
uv run --frozen --no-sync python tools/dead_code.py check  # vulture findings may not rise per file (DEPS4; baselines/dead_code.json, whitelist tools/dead_code_whitelist.txt)
uv run --frozen --no-sync python tools/divergence.py check --pr-branches   # every upstream file the fork modifies has a row in tools/divergence.yaml (same PR); no fork-only path on a pr/U* branch
prek install                                               # hooks on changed files (.pre-commit-config.yaml): strict ruff, ty, ratchet

# Coverage and baselines (procedures: skills grace-torch-tests, grace-torch-goldens; baselines/README.md)
uv run --frozen --no-sync pytest tests -q -n 4 --dist load --cov=tensorpotential --cov-branch --cov-report=json:cov.json <same --ignore options>
python tools/check_touched_coverage.py --coverage cov.json [--base origin/torch-backend] [--unit PATH::Class.method]   # touched units: 90%
python tools/coverage_ratchet.py check cov.json            # a file fails when its share falls below baselines/coverage_baseline.json and its uncovered count rises
python tools/quality_report.py check --ratchet-coverage   # one table of the quality metrics against baselines/quality.json (TOOL4; nightly job)
python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json   # also tools/ast_manifest.py and tools/check_clones.py
```

Not in a normal run: `test_structured_grid.py` (skipped on public master: needs the non-existent `tensorpotential.experimental`; ignored above only so counts match `baselines/`) and `test_foundation_model_regression.py` (needs foundation-model weights, HPC only).
A test writes only into `tmp_path` or a scratch directory; `git status` is clean after a run. The full suite takes about 15 minutes at `-n 4` on an idle machine: run it in the background.
**A whole-suite `-n 4` run needs over 18 GB and has taken the desktop session down:** run the CI shards one after another inside `systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=2G` (`.github/CI.md`, "Locally").

**Required checks of a pull request** (CI3; the owner sets them, an agent never does; detail and what CI does not run: `.github/CI.md`): `lint`, `test-core`, `test-torch` (two pandas versions), `test-tf` (4 groups), `quality-gates`, `wheel`, `wheel-smoke`, `extras-resolve`. `board.py lint`, the real-weight tests and any GPU test are not in CI.

## Code style

- Python 3.12+, `from __future__ import annotations` in new modules, type hints on everything new (checked by `ty`).
- **Legacy code stays as it is**: no repo-wide reformat, no file splitting (`instructions/compute.py` is about 4,900 lines and upstream merges depend on its shape), no renames; improve a unit only
  while you are already changing it. An issue whose job is to change legacy code (clean-up, quality, deduplication, packaging, performance) says what may change; the coverage and comparison rules govern it and it is not widened.
- New packages get the strict ruff set from the **single ruff configuration in `pyproject.toml`**. **Strict by default (TOOL3):** a **new file** anywhere (not in `baselines/legacy_files.txt`: `cli/`, `data/`, `tests/`, `tools/`, ...) must have no ruff finding (docstrings and plain `assert` are allowed in tests) and no ty finding; a legacy file may not raise its counts. A plain `ruff check` stays quiet on legacy code, so run `tools/lint_ratchet.py check` (its docstring has the rules).
- Numpy-convention docstrings on new public API. No commented-out code, no `print` in library code (`logging.getLogger(__name__)`), no bare `except`, no `TODO` without an issue link, no mutable or shared default arguments.
- Small functions (complexity at most 10, at most 6 arguments); pure functions and frozen dataclasses for specifications. Errors are typed, actionable and raised early; nothing falls back silently.
- Physics names (`N806`, `E741`) are allowed. Units: eV, eV/Angstrom, eV/Angstrom^3.

## Testing (the rules that always apply)

- A function or class you **modify** must already be covered at 90% (line and branch). Below that, write characterization tests first, green on the unmodified code, in their own commit, then change the code. Comment-only and annotation-only edits are exempt.
- Every unit gets a **logic** layer and, where physics applies, a **physical values** layer from an oracle that does not call the unit under test. Never widen a tolerance to make a test pass; real or foundation-model weights are HPC-only.
- Conventions (file naming, tolerances, markers, shared models, no network) and procedures: `.claude/rules/testing.md` and the `grace-torch-tests` skill.

## Gotchas that apply everywhere

Area gotchas (instructions and `model.yaml`, data builder and pandas, training and Keras 3, TF-free packages and import gates, golden fixtures, CI and packaging) are in `.claude/rules/`; they load when you read or edit a matching file, and **a path-scoped rule is not re-injected after a compaction** until you touch a matching file again.

- **The package does not import TensorFlow.** `tensorpotential/__init__.py` and `calculator/__init__.py` resolve public names on first access (`core/lazy.py`); the TF options (`_configure_tf_options`) run when a TF-side module imports `tensorpotential._tf_options`.
  **A new module with a module-level `import tensorflow` starts with `from tensorpotential import _tf_options  # noqa: F401`** (`tests/test_tf_options.py` scans for it; `compat/pace` is exempt). A module that must stay TF-free goes in `source_modules` of `[tool.importlinter]` and in `TF_FREE_MODULES` of `tests/test_import_gates.py`.
- **Worktrees.** Link `.venv` from the main checkout (git-ignored). Never run `uv sync` in a worktree (`uv run --frozen --no-sync` and `.venv/bin/<tool>` are safe): it repoints the shared editable install (repair: `uv sync --frozen --group dev` in the main checkout; to sync the shared venv safely: `UV_PROJECT_ENVIRONMENT=<main>/.venv uv sync --frozen --group dev --no-install-project --inexact`).
  Run with `PATH=<worktree>/.venv/bin:$PATH` and `PYTHONPATH=$PWD`, print `tensorpotential.__file__` once, give `ty` `--python <main>/.venv`, and in a scratch script import `tensorpotential` before `tensorflow`.
- **Shell.** The shell is zsh: an unquoted `$VAR` is not split into words (use an array), `==` starts a command substitution, and a heredoc together with `$(...)` in one call can hang the tool (write scripts with the Write tool).

**Adding to this file.** A lesson goes here only when an agent working on something else could trip on it and a rule or one command prevents it: at most three lines. What is tied to one module, tool, test file or issue goes in the document next to that code (module docstring, a README, `.github/CI.md`, the matching `.claude/rules/` file or skill) with at most a one-line pointer here; evidence goes in the issue (`board.py finding`).
Before adding, extend or delete a bullet; delete one when its warning is fixed. Never a measurement log or a dated note. Each agent file has a size budget (`tools/tests/test_agent_file_budget.py`, run by the `lint` job); the user raises it, not you. A rule that must hold after a compaction stays here (path-scoped rules are not re-injected); one that must always hold is better enforced by a hook or a check.
**Creating a new skill (a directory under `.claude/skills/`) needs the user's validation first**: propose name, description and outline (in the conversation or as an issue comment), create it once the user agrees. The same holds for restructuring existing skills and for adding a file to `.claude/rules/`.
