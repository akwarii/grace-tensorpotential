# Cleanup report: findings that are reported, not fixed (CLEAN4)

Status: **ready for review in the pull request**. No source file was edited; this report only records what the tree contains, what was measured, and the rules that govern the next removals and upstream merges.

**Scope rule (owner, 2026-10-01): `tensorpotential/compat/pace/` is out of scope and no change to it is allowed**, by this issue or by any other: no edit, deletion, reformatting or fix, including the findings below that sit inside it (F1, F2, F4 and part of F3). It is reported here and left as it is.

Measured on 2026-10-01 on `torch-backend` at `c313e3e` (after CLEAN2), CPU, Python 3.12.3, `ruff 0.16.7 --isolated`, TensorFlow 2.20. Commands are given so each number can be reproduced.

## 1. Findings

| # | Finding | Count | Where | Action taken |
|---|---|---:|---|---|
| F1 | Latent `NameError` (`rankmax` instead of the parameter `ranksmax`) | 1 | `compat/pace/potentials/ace.py:975` | none; out of scope (`compat/pace/`) |
| F2 | Bare `except:` | 8 | all in `compat/pace/` | none; out of scope (`compat/pace/`) |
| F3 | `TODO`/`FIXME`/`XXX`/`HACK` markers | 52 | 47 in live code, 5 in `compat/pace/`; none links an issue | none; the 5 in `compat/pace/` are out of scope |
| F4 | Star imports | 3 + 1 | 3 in `compat/pace/`, 1 in a notebook | none; the 3 in `compat/pace/` are out of scope |
| F5 | `compat/pace/` is imported by nothing outside itself | 17 files, 4,399 lines | `tensorpotential/compat/pace/` | none; out of scope, no change allowed (F5) |

### F1. `rankmax` NameError

`ConfigBasis.set_basis_from_list_` (note the trailing underscore) loops `for r in range(1, rankmax)` but its parameter is `ranksmax`:

```python
def set_basis_from_list_(self, bbasisfunc, ranksmax, nelem):
    for r in range(1, rankmax):            # ace.py:975, F821
```

Reproduced: calling the method raises `NameError: name 'rankmax' is not defined`. **It is unreachable today**: nothing calls `set_basis_from_list_` (`grep -rn set_basis_from_list_` finds only the definition); the live constructor calls `set_basis_from_list` (no underscore, `ace.py:963`), which is correct. `set_basis_from_df` has a parameter named `rankmax` and is fine. Ruff's only real F821 in the package is this line (the other two F821 are `plt` in an example notebook). The package has no test and the file is out of scope for every edit (D5), so the line is left as it is.

Reproduce: `uvx ruff@0.16.7 check --isolated --select F821 tensorpotential`.

### F2. Eight bare `except:`

All eight are in `compat/pace/` (E722): `fit.py:98, 332, 667`, `fitmetrics.py:192`, `utils/tensorcalc.py:26`, `utils/utilities.py:32, 246, 291`. The live code has none. Not fixed: `compat/pace/` is not edited. If the package is kept, each needs a named exception type and a log line.

Reproduce: `uvx ruff@0.16.7 check --isolated --select E722 --exclude tools --statistics .`.

### F3. 52 TODO-style markers

`grep -rnE '\b(TODO|FIXME|XXX|HACK)\b' --include='*.py' tensorpotential`, by file: `scripts/grace_preprocess.py` 10, `cli/train.py` 7, `instructions/compute.py` 6, `cli/gracemaker.py` 3, `cli/data.py` 3, `instructions/base.py` 3, `tensorpot.py` 3, `tpmodel.py` 2, and one each in `constants.py`, `data/process_df.py`, `data/databuilder.py`, `instructions/output.py`, `extra/gen_tensor/loss.py`, `extra/gen_tensor/ewald.py`, `potentials/presets.py`, `functions/nn.py`, `functions/couplings.py`, `functions/__init__.py`, `cli/metrics.py`, and 5 in `compat/pace/`. None carries an issue link, so none satisfies the rule "no `TODO` without an issue link" for new code; legacy markers are not touched (the rule is for code written from now on).

Markers that point at behaviour rather than at style, to be looked at when the unit is next modified:

- `scripts/grace_preprocess.py:720-725`: six markers (reference energy, `stress_units`, shift from `lstsq`, element map, weighting, user cutoff) next to the disabled `stress_units` wiring (class E below).
- `cli/gracemaker.py:652`: "first save with jit will convert function to JIT forever" (next to the disabled LoRA path).
- `instructions/base.py:210, 217`: "call `pre_serialize` / `pre_deserialize`?" in the (de)serialisation of the instruction graph.
- `instructions/compute.py:2051`: "This must not be done" (no further text).
- `functions/nn.py:323`: "apply only if `self.activation`. Code above is always executed".

### F4. Star imports

Three in `compat/pace/` (`fit.py:8`, `tensorpot.py:3`, `utils/utilities.py:5`, all `from tensorpotential.compat.pace.constants import *`) and one in `examples/grace/visualize_metrics.ipynb` (cell 6, `from tensorpotential.utils import *`; the same notebook also uses an undefined `plt` twice, F821). The live library has none. The issue text says four, which is the ruff F403 count over notebooks and Python files together; there are three in `.py` files. Decision D5(d) keeps star imports out of the clean-up.

Reproduce: `uvx ruff@0.16.7 check --isolated --select F403,F821 --exclude tools .`.

### F5. `compat/pace/` is imported by nothing outside itself

Evidence (all on the current tree):

- `grep -rnI 'compat.pace\|compat/pace\|tensorpotential.compat' . --exclude-dir=.venv --exclude-dir=.git` outside `tensorpotential/compat/pace/` matches only `CLAUDE.md`, `cleanup_triage.md` and the planning/tooling files (`plan/`, which are not shipped).
- No `tests/` file mentions it (the three matches of "compat" are the words "compatible" / "backward compat").
- No `importlib.import_module`, string import or saved-yaml class lookup reaches it: the dynamic imports of the library (`extra/presets.py`, `uq/cli/build/data_resolve.py`, `scripts/grace_uq.py`, `cli/prepare.py`) resolve other modules, and `tpmodel.__getattr__` resolves instruction classes, none of which lives in `compat/pace/`.
- No console script of `pyproject.toml` (`[project.scripts]`) points into it.
- It does not import anything else from `tensorpotential` (the grep for `from tensorpotential.<module>` outside `compat` finds nothing), so removing it cannot break a dependency of its own.
- It **is shipped**: `setup.py` uses `find_packages(include=["tensorpotential", "tensorpotential.*"])`, so the wheel contains it, and `import tensorpotential.compat.pace` and each of its modules import without error today (TensorFlow is imported, no other package is needed).
- It has no coverage: no test exercises it.

Conclusion: **out of scope, no change allowed** (owner decision of 2026-10-01, on top of D5(b)). The evidence above is recorded only so that nobody has to re-derive it; it is not a proposal to remove the package, and no retirement issue is planned.

## 2. Class E register (disabled wiring of documented or user-visible options)

Copied from `cleanup_triage.md` (CLEAN1). Locations are re-measured on the current tree: CLEAN2 deleted comment blocks above these lines, so they moved by a few lines; the id is the stable key. Default action: keep. Each row needed a decision from the owner; they were taken on 2026-10-01 and are in 2.1.

| id | location (current tree) | what is disabled | evidence | decision to take |
|---|---|---|---|---|
| `5a978b0c` | `tensorpotential/cli/gracemaker.py:643-645` | LoRA finalisation before `--save-model` | `docs/gracemaker/inputfile.md:74` says "LORA (experimental, not supported)"; `enable_lora_adaptation` / `finalize_lora_update` are live in `functions/nn.py`, `instructions/base.py`, `instructions/compute.py`, `tpmodel.py` | retire the LoRA path as a whole, or finish it |
| `8acc3b08` | `tensorpotential/cli/gracemaker.py:659-671` | `potential: lora` / `reduce_lora` input options (activation and reduction, with model re-save) | same docs lines 75-76 (option examples); no test exercises it | same |
| `bfc0922a` | `tensorpotential/cli/gracemaker.py:741-743` | LoRA finalisation before the final save | same | same |
| `1c5d0a40` | `tensorpotential/functions/radial.py:98-100` | `GaussianRadialBasisFunction(normalized=...)` stores nothing and normalises nothing | the argument is accepted at `radial.py:83`; the class is reached through the captured kwargs of `RadialBasis(basis_type='Gaussian', ...)`, so `normalized` can be in saved `model.yaml` files | remove the argument (a model-format change, see the persisted-API gotcha) or restore the code |
| `ca55d8af` | `tensorpotential/functions/radial.py:111-117` | same, `build` | same | same |
| `caafca75` | `tensorpotential/functions/radial.py:122-127` | same, `compute_basis` | same | same |
| `cf89403b` | `tensorpotential/scripts/grace_preprocess.py:713` | `stress_units` in `grace_preprocess` | wired in `cli/data.py` (gracemaker) and documented in `docs/gracemaker/inputfile.md:25`; commented in `grace_preprocess.py:713` with a TODO at `:721`: preprocessing always uses eV/A3 | wire it or document the eV/A3 limitation of `grace_preprocess` |
| `d1ac63dd` | `tensorpotential/scripts/grace_utils.py:491` | `--aux` selection of `grace_utils aux_model` (read of `args.aux`) | documented in `docs/gracemaker/utilities.md:228` and `:347` (`--aux energy_only parallel_2L`); the `add_argument('--aux')` is commented at `:708-713`; today only `compute_energy` is added | restore the argument or correct the docs |
| `f0d17646` | `tensorpotential/scripts/grace_utils.py:503-505` | `compute_local` aux function (`ComputeStructureEnergyAndForcesAndVirial(local=True)`) | listed with `--aux` in the docs and the argparse help text | same |
| `fb708616` | `tensorpotential/scripts/grace_utils.py:708-713` | the `--aux` argparse entry itself (default `parallel_2L energy_only compute_local`) | same | same |
| `94abfc6e` | `tests/test_integration_test.py:328-363` | test `test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop` (MLP model, ef/lr switch, early stop; 6 epochs of numeric references) | `tests/MoNbTaW-MLP/input.yaml` is tracked | re-baseline the references and re-enable, or retire with the input |
| `0bb0b394` | `tests/test_integration_test.py:501-537` | test `test_MoNbTaW_GRACE_2L_MP` (2L message-passing model; 3 epochs of numeric references) | `tests/MoNbTaW-GRACE/input_2L_MP.yaml` is tracked | same |

Grouped by the decision they need: **LoRA** (3 rows, one decision), **Gaussian `normalized`** (3 rows, one decision, touches the model format), **`stress_units` in `grace_preprocess`** (1), **`grace_utils aux_model --aux`** (3 rows, one decision; the documentation currently describes behaviour the code does not have), **two disabled integration tests** (2). Five decisions in total, taken in 2.1 and **not carried out here**; CLEAN1 recorded that a decision is needed, CLEAN2 left all 12 blocks in place (verified: the 12 blocks are present above).

### 2.1 Owner decisions (2026-10-01)

| Decision | Blocks | Outcome | What carrying it out means (not done in CLEAN4) |
|---|---|---|---|
| E1 LoRA | `5a978b0c`, `8acc3b08`, `bfc0922a` | **finish it** | Re-enable the three blocks in `cli/gracemaker.py` and make `lora` / `reduce_lora` work end to end (activation, reduction, re-save of the model yaml). Touches live TF code in `functions/nn.py`, `instructions/base.py`, `instructions/compute.py`, `tpmodel.py` (each modified unit at 90% coverage first), needs a test with physical oracles (a reduced LoRA model reproduces the activated one), and the docs lose "not supported". The TODO at `gracemaker.py:652` (the first jitted save converts the function for good) is part of it. |
| E2 Gaussian `normalized` | `1c5d0a40`, `ca55d8af`, `caafca75` | **mark as deprecated** | Keep accepting the argument (saved yamls stay loadable, no model-format change) and emit a `DeprecationWarning` when it is passed; document that it has no effect. The commented code stays or goes with that change. |
| E3 `stress_units` | `cf89403b` | **wire it** | `grace_preprocess` passes `stress_units` through like `gracemaker` does (`cli/data.py`); a test converts GPa / kbar / -kbar input and compares with eV/A3 by hand-computed factors. Changes preprocessing output for users who set the option. |
| E4 `--aux` | `d1ac63dd`, `f0d17646`, `fb708616` | **correct the docs** | `docs/gracemaker/utilities.md` (lines 228 and 347) and the argparse help describe only what `aux_model` does today (`compute_energy`); the three commented blocks are then dead code and can be deleted in the same change. Documentation only. |
| E5 disabled tests | `94abfc6e`, `0bb0b394` | **re-enable** | Uncomment the two tests and re-baseline their numeric references (6 and 3 epochs of metrics); needs a training run on this machine. Regenerating references is its own change, never inside a feature PR. |

None of these fits CLEAN4, whose scope is report-only (no behaviour change, no new feature, no other issue's work). They are tracked as follow-up issues, all depending on GATE-CLEAN so that the gate's comparison with the untouched baseline stays meaningful: E1 in LORA1 (#103), E3 in DATA1 (#104), E2 and E4 in CLEAN5 (#105), E5 in TEST5 (#106). This PR records the decisions only.

## 3. Rules

### 3.1 Upstream-sync procedure

State today: `origin` is the fork; `upstream` is the ICAMS repository with its push URL disabled. `git rev-list --left-right --count upstream/master...origin/torch-backend` gives `0 26` (upstream has nothing the fork lacks; the fork has 26 commits more).

Incoming (upstream changes into the fork):

1. `git fetch upstream`, then the rev-list count above. If the left number is 0, there is nothing to do.
2. Otherwise branch `sync-upstream-<yyyymmdd>` from `origin/torch-backend` and **merge** `upstream/master` (a merge keeps the history that a later `pr/U*` squash-merge needs; no rebase of published branches). Conflicts are expected where upstream edits near a block that CLEAN2 deleted or a file the fork changed: keep upstream's behaviour, re-apply the fork's removal only if the deleted block is still dead.
3. Verify before the sync PR is opened, from the repository root: the full suite with `-n 4 --dist load` compared with `baselines/outcomes_pd2.json` through `tools/junit_outcomes.py compare`; the AST manifest check for the files upstream changed (every `changed:` line must be explained by the upstream diff); the oracle snapshot if `instructions/`, `functions/`, `tpmodel.py` or `poly.py` changed. A sync is one PR into `torch-backend`, opened and merged by the owner.
4. Never push to `master` of the fork's `origin` as a way to track upstream, and never push to `upstream`.

Outgoing (a unit offered to ICAMS), as decided in D6 and D17:

1. Cut `pr/U<n>-<slug>` from `upstream/master`, never from `torch-backend`; merge that branch into `torch-backend` at once so the fork never waits.
2. `python tools/check_pr_branch.py` must pass (no `CLAUDE.md`, `.claude/`, `tools/`, `baselines/`, `plan/`, `tests_torch/`, `tensorpotential/torch_backend/`, `tensorpotential/core/`, ... in the diff against `upstream/master`).
3. No AI co-author trailer in commits of `pr/U*` branches (D17e); PR text for upstream is drafted locally, passed through `python tools/board.py sanitise`, and shown to the owner first.
4. Nothing is opened, pushed or commented upstream without an explicit go for that unit; the owner checks the licence clause of the upstream PR template with the institution before the first PR (D6); at most two upstream PRs are open at a time.
5. If upstream merges the unit (squash), merging `upstream/master` is a no-op or a trivial conflict; if upstream changes it during review, rebase the fork's copy onto the merged version.

Consequence of CLEAN2 for later syncs (an expectation, not yet observed because upstream has not moved): the 169 deleted blocks and the other clean-up edits are fork-only (the U6 upstream units were skipped, so nothing replays them upstream), hence a future merge can conflict wherever upstream edits next to a deleted block. The divergence ledger of BOARD2 (D17b) will list fork-modified upstream files; until it exists, `git diff --stat upstream/master...origin/torch-backend -- tensorpotential` is the list.

### 3.2 Removal protocol for any future retirement of code

Adapted from the retirement gates (RET-1 to RET-6) that MACE used for its legacy code; applies to deleting a module, a class, an option or a package (for example a class-E item above; **never `compat/pace/`**, which is out of scope). A comment-only deletion follows CLEAN1 and CLEAN2 instead (triage table, AST manifest equal).

1. **The replacement is the default and its parity is green.** A retirement names what replaces the code and shows the comparison (suite outcome equal to `baselines/`, oracle snapshot within tolerance when numerics are touched) before the deletion, never in the same commit.
2. **Nothing live imports the code.** Evidence is an import-resolution check (every `import` / `from` in the tree resolves after the change) and an import smoke (`python -W always -c "import <every module>"`), plus the dynamic routes of this repository: `importlib.import_module` calls, `tpmodel.__getattr__` and `instructions.__getattr__` (saved `model.yaml` files import classes by name through them, so a class deletion is a model-format break), console scripts of `pyproject.toml`, and entry points.
3. **An exact deletion set and a not-in-set list.** A machine-readable list of the files, classes and functions that go (ids, not line numbers), and the list of things that look related and stay, in the PR. The AST manifest and the pragma/shebang guard cover what the AST check cannot see (comments).
4. **Every removed test has a named successor**, or a stated reason that none is needed; the junit comparison against `baselines/outcomes_pd2.json` explains each missing id.
5. **A dependents sweep for everything shipped in the wheel**: `find_packages` and `package_data` in `setup.py`, `[project.scripts]`, the documentation (`docs/`), `examples/` and notebooks, the Docker and CI files, and users' saved models (the persisted-API gotcha: a constructor argument or default is part of every saved model).
6. **Persisted formats and public options get a deprecation step** (an actionable error or warning naming the replacement) before they disappear; nothing falls back silently.
7. One concern per PR and the owner decides. `compat/pace/` is outside this protocol: no change is allowed to it.

## 4. Decisions this report needs from the owner

1. **Class E**: decided (2.1); the follow-up work is tracked in LORA1, DATA1, CLEAN5 and TEST5 (2.1).
2. **Upstream issues (U10)**: whether to report F1 (reporting does not change the code) and the class-E documentation mismatch (`--aux`) as issue texts to the maintainers; nothing is sent without an explicit go (D6).
3. **Wall-time clauses of TEST4 (section 8)**: serial 73.2% of the baseline (target 60%) and `-n 4` 8 min 00 s (target 7 min) are not met, and the rest cannot be reached without touching the 19 integration runs (44% of the serial time; each has its own configuration and reference metrics, so they cannot be merged and a golden may not be regenerated). Options: (a) accept the measured 73.2% and 8 min 00 s and close TEST4 with the fast tier (4 min 14 s with `-n 4`) (`-m "not slow"`) as the development loop; (b) cut the fixed cost of a `gracemaker` run in the library (tf.function tracing and autograph conversion, about 9 s of 24 s, and the SavedModel export, about 3 s), which is a library change for a separate issue; (c) move the two clauses to that issue.

## 5. Parallel test execution (TEST2)

Measured on 2026-10-01 and 2026-10-02 on the TEST2 branch (CPU, 14 cores, 30 GB, pandas 2.3.3, TensorFlow 2.20, pytest-xdist 3.8.0). Every run is the full suite from the repository root on a **read-only bind mount** of the tree (`unshare -rm`, then `mount --bind` and `mount -o remount,ro,bind`; `/tmp` stays writable), `--dist load`, with `tests/test_structured_grid.py` and `tests/test_foundation_model_regression.py` ignored as in `baselines/`. Each worker gets `cores / N` TensorFlow and OpenMP threads (root `conftest.py`, `tests/thread_budget.py`). The serial reference is the 32 min 19 s measured on 2026-10-01 (`baselines/outcomes_pd2.json` holds the outcomes). The machine was not idle: other applications held 9 to 11 GB before each run.

| N | threads per worker | wall time | single-process peak RSS | peak system memory above start | outcome per test id vs `outcomes_pd2.json` |
|---:|---:|---:|---:|---:|---|
| 2 | 7 | 15 min 20 s | 10.5 GB | 15.6 GB | identical |
| 4 (run 1) | 3 | 11 min 51 s | 6.6 GB | 17.7 GB | identical |
| 4 (run 2) | 3 | 11 min 38 s | 6.7 GB | 16.1 GB | identical |
| 4 (run 3) | 3 | 11 min 54 s | 6.6 GB | 16.9 GB | identical |
| 4 (run 4, idle machine) | 3 | 9 min 37 s | 6.7 GB | 17.9 GB | identical |
| 4 (run 5, idle machine) | 3 | 10 min 25 s | 6.6 GB | 17.2 GB | identical |
| 4 (run 6, idle machine) | 3 | 11 min 53 s | 5.8 GB | 16.4 GB | identical |
| 4, `--dist worksteal` (idle machine) | 3 | 10 min 18 s | 6.7 GB | 20.1 GB | identical |
| 6 | 2 | 10 min 08 s | 5.5 GB | 18.3 GB | identical |
| 8 | 1 | 13 min 09 s | 5.0 GB | 19.5 GB | identical |

Every run: 709 passed, 6 skipped, 2 xfailed, 1 xpassed, 0 failed (710 passed in the two idle-machine runs, which include the xdist-only thread-budget check added afterwards; `tools/junit_outcomes.py compare` reports no changed test; the only differences are the 5 run-isolation tests of TEST1 and the 18 thread-budget tests added since the baseline). A first `-n 4` run on the same mount before a machine restart took 9 min 58 s.

Reading of the numbers:

- **The target of a third of the serial time (10 min 46 s) is met in two of three idle runs at N = 4, so the exit criterion of TEST2 uses 40% of the serial time (12 min 55 s), which every N = 4 run satisfied (slowest 11 min 54 s); by decision of the owner (2026-10-02) the work on the longest tests moves to TEST4.** Three consecutive runs on a machine with nothing else running took 9 min 37 s, 10 min 25 s and 11 min 53 s (the last is 1 min 07 s above the target); a run before the restart took 9 min 58 s, and three runs made with other applications running took 11 min 38 s to 11 min 54 s. The spread is in the tests, not in the scheduling: the summed test time of the three idle runs was 1,392 s, 1,826 s and 1,556 s for the same tests (the longest, `test_compute_distributed_data_and_distrib_fit`, took 205 s, 246 s and 243 s), so the machine itself varies by up to 30% from run to run; the cause (clock speed, memory pressure, background processes) was not isolated. N = 6 took 10 min 08 s once (other applications running); it was not repeated.
- N = 8 is slower than N = 6. At N = 8 the system came within 1.5 GB of the 30 GB (peak 28.7 GB used, which includes about 9 GB of other applications), so memory pressure is the probable cause; not verified.
- Per-test durations were not analysed for these runs. The earlier profile (28 tests of 20 s or more hold 1,517 s of 1,939 s; the two longest take 243 s and 189 s) means the wall time cannot fall below about 4 min whatever N is.
- Memory: a worker is large (the single largest process peaked at 5 to 10 GB); N = 6 is the highest count that leaves headroom on this machine.

**Coverage under xdist (subset).** `pytest --cov=tensorpotential --cov-branch` on every test file except `test_integration_test.py`, `test_distrib.py` and `test_uq_integration.py` (the three heaviest; about 590 s of the 1,869 s of test time), serial and with `-n 4 --dist load`, on the read-only mount. Per-file line and branch coverage of the 115 measured files is identical in the two runs (largest difference 0.0 percentage points; total 52.39% in both), well inside the 0.1 point limit. The per-test outcomes are identical except for one test that is skipped serially by design, `test_xdist_worker_received_its_budget_before_tensorflow_started` (685 passed, 2 skipped in parallel; 684 passed, 3 skipped serially). Wall time with coverage: 6 min 14 s with `-n 4` (the other agent's `-n 4` run overlapped part of it), 11 min 55 s serial (machine otherwise idle). Subprocess coverage (`gracemaker` runs started by tests) is not collected in either run.

**`--dist worksteal` against `load`** (N = 4, back to back on an idle machine, same tree): `worksteal` 10 min 18 s, `load` 9 min 37 s, identical outcomes. One pair of runs, so a 41 s difference is not conclusive, but `worksteal` is not faster, and `--dist load` stays the documented option.

**Not measured:** coverage for the full suite, including the three excluded files (they are the longest tests and the only ones that start subprocesses). By decision of the owner (2026-10-02) the exit criterion of TEST2 is limited to the subset above, and TEST3 checks the full suite once when it records its coverage baseline; the effect of the thread budget alone (the default of all threads per worker was not timed on the read-only mount). The earlier timings were taken while other work ran on the machine, so differences of a minute are within the noise.

**Write audit.** The first read-only run found one write into the working directory: `test_graph_split.py::TestGraphSplitSaveReload::test_split_model_save_reload` saved `temp_saved_model_test` there (now `tmp_path`). `chmod -R a-w` is not a valid way to make the tree read-only for this check: `shutil.copy` and `copytree` propagate the mode and then fail on the copy; a read-only bind mount does not.

## 6. Coverage gate and characterization tests (TEST3)

Measured on 2026-10-02 on the TEST3 branch (CPU, 14 cores, 30 GB, pandas 2.3.3, TensorFlow 2.20, coverage.py 7.16.2). The library (`tensorpotential/`) is not changed by this issue: it adds two tools, a tolerance table, six test files and the baseline.

**Tools.** `tools/check_touched_coverage.py` reads a coverage.py JSON report (branch mode) and prints one line per unit, `(statements + branches executed) / (statements + branches)`, exit status 1 under 90%. A unit is a `def`, a method or a class body; a nested `def` is its own unit; module-level statements are not gated. A unit is touched when it is new or its AST differs from the base after dropping comments, docstrings and annotations (the normalised mode of `ast_manifest.py`). A file no test imports counts as 0%. `--unit path::Qualname` names units whatever the diff says. `tools/coverage_ratchet.py record|check` keeps `baselines/coverage_baseline.json` (executed and total statements plus branches per file, 115 files, 56.16% in all) from falling.

**Units of work item (c), before and after.** "Before" is the full suite at `3d8e4ad`; "after" is the full suite with the new tests (`pytest tests -n 4 --dist load --cov=tensorpotential --cov-branch`); "new tests alone" is the new test file of the unit run by itself.

| Unit | Before | New tests alone | After |
|---|---:|---:|---:|
| `estimate_n_buckets` | 73.9% | 100% | 100% |
| `extract_const_shift_scale` | 60.0% | 100% | 100% |
| `grace_2` | 61.2% | 100% | 100% |
| `TPCalculator.__init__` | 84.6% | 98.2% | 100% |
| `FunctionReduce.build` | 71.0% | 100% | 100% |
| `FunctionReduceN.build` | 60.4% | 100% | 100% |
| `CollectInvarBasis.build` | 85.7% | 100% | 100% |
| `FunctionReduceParticular.build` | 88.5% | 100% | 100% |

`TensorPotential._set_default_loss_specs` is dropped from the list (owner, 2026-10-02): it exists only in `compat/pace/`, which no issue may change. The new tests pass on the tag `pre-cleanup` (184 passed, library of the tag) and on HEAD. The full suite with the new tests: 1025 passed, 6 skipped, 2 xfailed, 1 xpassed, 0 failed in 12 min 13 s with `-n 4`; `tools/junit_outcomes.py compare` against `baselines/outcomes_pd2.json` shows no changed or removed test, only added ones (the 184 tests of this issue and those of earlier issues).

**Test layers.** Logic: branches, errors, shapes, options. Physics, from oracles that do not call the unit: hand-counted histories, weight columns and output-norm maps from the coupling selection rules (including the exchange symmetry that removes odd `L` for `l1 == l2`); initialisers against their stated standard deviation with a sampling bound; rotation of the built reducers with real Wigner matrices; for `grace_2`, `R T R^T` covariance of the rank-2 output, translation, permutation, energy invariance and finite-difference forces on the real float64 model; for the calculator, isolated-atom energies beyond the extracted cutoff, per-pair cutoffs, rigid-motion invariance and finite-difference forces. All tolerances come from `tests/tolerances.py`. Mock-like uses: three `monkeypatch.setitem(sys.modules, ...)` in `test_calculator_init.py`, each standing in for a package that is absent from this tree (`tensorpotential.experimental`, the `gen_tensor` data builders), with a comment; one stub subclass of `CollectInvarBasis` (see below).

**Planted mutants** (hand-written, applied in a scratch copy of the library, the test file of the unit run with `-x`): 41 mutants, all caught, none survived.

| Unit | Id | Kind | Mutation | Result |
|---|---|---|---|---|
| `estimate_n_buckets` | EN1 | logic | `overhead <= budget` becomes `<` | caught |
| | EN2 | logic | `total_real_neigh == 0` becomes `!= 0` | caught |
| | EN3 | physics | bucket padded to its mean instead of its largest batch | caught |
| | EN4 | physics | overhead `- 1.0` becomes `- 0.9` | caught |
| | EN5 | logic | fall-through returns 1 instead of `min(batches, 32)` | caught |
| `extract_const_shift_scale` | XS1 | logic | `not isinstance(shift, (float, int))` inverted | caught |
| | XS2 | logic | shift map no longer flattened | caught |
| | XS3 | physics | scale and shift swapped in the result | caught |
| | XS4 | logic | `ConstantScaleShiftTarget` lookup inverted | caught |
| `grace_2` | G21 | logic | `max_order > 2` becomes `>= 2` | caught |
| | G22 | logic | rank guard `< 3` becomes `<= 3` | caught |
| | G23 | physics | `scale=constant_out_scale` becomes `1.0` | caught |
| | G24 | physics | parity of the rank-1 reduction `-1` becomes `+1` | caught |
| | G25 | physics | second-layer product `Lmax=3` becomes `2` (dropped terms) | caught |
| `TPCalculator.__init__` | TC1 | logic | `"uniform"` removed from the accepted modes | caught |
| | TC2 | logic | ensemble data-key check inverted | caught |
| | TC3 | physics | ensemble cutoff `max` becomes `min` | caught |
| | TC4 | physics | ensemble pair-cutoff matrix `max` becomes `min` | caught |
| | TC5 | physics | calculator cutoff from the matrix `max` becomes `min` | caught |
| | TC6 | physics | pair-cutoff keys built with the elements swapped | caught |
| | TC7 | logic | geometry builder always segment-sum (`dense_nbr=False`) | caught |
| | TC8 | logic | ensemble engine capability `and` becomes `or` | caught |
| | TC9 | logic | cutoff-mismatch message disabled | caught |
| `FunctionReduce.build` | FR1 | logic | atom-type axis dropped from the weight shape | caught |
| | FR2 | physics | initial standard deviation `sqrt(2/(n_in w))` becomes `sqrt(1/(n_in w))` | caught |
| | FR3 | logic | `is_built` guard removed | caught |
| | FR4 | physics | `1/n_instr` constant becomes `n_instr` | caught |
| | FR5 | physics | output rows of the scatter table reversed (breaks equivariance) | caught |
| `FunctionReduceN.build` | FN1 | physics | norm `scale/sqrt(n_in)` becomes `scale*sqrt(n_in)` | caught |
| | FN2 | physics | unnormalised init `1/sqrt(n_in)` becomes `1/n_in` | caught |
| | FN3 | physics | uniform init lower bound `-s` becomes `0` | caught |
| | FN4 | logic | output-norm map reshaped `[-1, 1]` instead of `[-1, 1, 1]` | caught |
| | FN5 | logic | LoRA branch disabled | caught |
| | FN6 | physics | contribution count `+= 1` becomes `+= 2` (output-norm map) | caught |
| `CollectInvarBasis.build` | CI1 | logic | `is_built` guard inverted | caught |
| | CI2 | physics | recorded dtype always `float32` | caught |
| | CI3 | logic | `max(ls_max) == 0` assertion weakened to `>= 0` | caught |
| `FunctionReduceParticular.build` | FP1 | logic | `out_norm` branch inverted | caught |
| | FP2 | physics | `norm = 1/sqrt(n_in)` becomes `1/n_in` | caught |
| | FP3 | physics | initial standard deviation `1.0` becomes `0.5` | caught |
| | FP4 | physics | contribution count `+= 1` becomes `+= 2` | caught |

The two tools have their own tests: `check_touched_coverage.py` 36 tests, 99% branch coverage, 23 mutants caught; `coverage_ratchet.py` 15 tests, 98%, 12 of 13 mutants caught (the survivor, `<` against `<=` on the fall test, is equivalent behind the `1e-9` float slack).

**The gate turns red on an untested edit.** On a scratch branch cut from this one, `extrapolate_series` (`scripts/grace_dashboard.py`, never executed by the suite) got one changed constant and `estimate_n_buckets` got a comment-only edit; against the full-suite report of `3d8e4ad`:

```
FAIL   0.0%  tensorpotential/scripts/grace_dashboard.py::extrapolate_series  (0/42, L233-L278)
1 unit(s) checked, 1 under 90%, 0 problem(s)        (exit status 1; the comment-only edit is exempt)
```

**Edits moved here from QUAL1 (work item (d)).** Five QUAL1(c,d) sites sit in units the suite never executes, so by the coverage rule they are not edited and are reported: `scripts/grace_dashboard.py::extract_n_params` (silent `except` at line 108, 0.0%, 0/18), `scripts/grace_dashboard.py::extrapolate_series` (line 272, 0.0%, 0/42), `uq/cli/build/master.py::run_master` (lines 152 and 615, 0.3%, 1/361) and `scripts/grace_collect.py::main` (the `map(lambda ...)` at line 252, 0.0%, 0/108). The other QUAL1 sites are in units now at 100% or already above 90% (`extract_const_shift_scale`, `TPCalculator.__init__`, `compute_compositions` 93.8%, `ElasticBatchIterator.__next__` 94.4%, `get_preset` 100%).

**What the numbers do not cover.** Subprocess coverage (`gracemaker` started by tests) is collected in no run. **Serial against parallel, full suite** (the check TEST2 left to this issue): the same tree run serially (1024 passed, 7 skipped, 2 xfailed, 1 xpassed, 31 min 40 s) and with `-n 4 --dist load` (1025 passed, 6 skipped, 12 min 13 s) gives identical per-file coverage for all 115 files, including `test_integration_test`, `test_distrib` and `test_uq_integration` (largest difference 0.0 percentage points; 56.16% in both); the one outcome that differs is the xdist-only thread-budget test, skipped serially by design. Units not in the list keep their own coverage; the global figure is 56.16% (it was 55.28% at `3d8e4ad`). Findings: the Appendix E table dump is not in the repository, so the table oracles are derived by hand; `CollectInvarBasis` cannot be instantiated as shipped (abstract `upd_init_args_new_elements` missing); `estimate_n_buckets` fails on pandas 3 and emits a `FutureWarning` from `np.array_split` on pandas 2.3.3 (its tests are marked xfail on pandas 3, reason DEPS1).

## 7. Deduplication (QUAL2)

Measured on 2026-10-02 on the QUAL2 branch (CPU, 14 cores, 30 GB, pandas 2.3.3, TensorFlow 2.20) against `torch-backend` at `b8a0104`.

**Tool.** `tools/check_clones.py` (`check`, `record`, `list`) finds *clone groups*: functions of 8 lines or more (from `def` to the last line) whose bodies, without the docstring and the signature, have the same AST. A group is **identical** when the bodies are equal, **renamed** when they become equal after abstracting every identifier (variables, attributes, arguments, keywords, nested names; in `tests/` also the constant values, which is what `pytest.mark.parametrize` abstracts). A body that is only `pass` or `...` is a stub and never counts. `tensorpotential/` is scanned without `compat/pace/`. The check is function-level, so a clone inside a large function is not seen. The ratchet sums both kinds per tree and lets none of groups, functions and redundant lines (lines of all members but the largest) rise above `baselines/clone_baseline.json`. The tool has 34 tests (99% branch coverage) and 20 planted mutants, all caught.

| Tree | Kind | Groups before | Functions before | Redundant lines before | Groups after | Functions after | Redundant lines after |
|---|---|---:|---:|---:|---:|---:|---:|
| `tensorpotential/` | identical | 5 | 14 | 112 | 4 | 10 | 88 |
| `tensorpotential/` | renamed | 5 | 14 | 114 | 5 | 14 | 114 |
| `tests/` | identical | 3 | 6 | 40 | 0 | 0 | 0 |
| `tests/` | renamed | 11 | 28 | 486 | 0 | 0 | 0 |

The numbers of the issue text (library 6 identical groups / 16 functions and 9 renamed groups / 26 functions; tests 3 and 13 groups) came from a throw-away script that was not kept; the identical-body counts agree except for two stubs (`IDatasetPlotter.plot` and `TPAtomsDataContainer.__init__`, bodies `pass` behind long signatures, which the issue called "plain attribute assignments"), the renamed counts cannot be reproduced exactly. All groups the issue names are found.

**Library groups.**

| Group | Decision | Reason |
|---|---|---|
| `sizeof_fmt` in `cli/data.py`, `scripts/df2extxyz.py`, `scripts/extxyz2df.py`, `scripts/grace_preprocess.py` | **merged** (upstream unit U13) | one function in the new TF-free module `tensorpotential/formatting.py`; three modules import it, the copy in `grace_preprocess.py` had no caller and is deleted; output identical (new tests: 236 cases, 100% line and branch coverage of the module, 9 of 9 planted mutants caught); no script gained a TensorFlow import |
| `loss.py`: `WeightedOffsetEnergyHuberLoss`, `WeightedHuberEnergyPerAtomLoss`, `WeightedHuberVirialLoss` (identical); six `Weighted{SSE,MAE}*Loss` constructors and the two `WeightedPiecewiseLinear{Force,Stress}Loss` constructors (renamed) | report only | each constructor restates the signature of its parent with its own defaults and forwards it; these are public API, and `capture_init_args` writes the defaults into saved `model.yaml` files, so merging them would change existing models |
| `instructions/output.py`: `TrainableShiftTarget.__init__`, `TrainableShiftTarget_v2.__init__` against `WeightedSSEForceLoss.__init__` and `HuberLoss.__init__` (renamed) | report only | new in this table: short constructors (a signature, a `super().__init__` call and one assignment) of unrelated classes that happen to have the same shape; public constructors with persisted defaults |
| `FunctionReduce.drop_unused` / `FunctionReduceN.drop_unused`, `simplify_collected_tensors` (same two), `prepare_variables_for_selected_elements` (`FunctionReduce`, `FunctionReduceN`, `FunctionReduceParticular`) in `instructions/compute.py` | report only | merging needs a shared base or helper in the classes CPU1 edits, and these methods are 3 to 14% covered, so the characterisation tests would cost more than half a day |
| `TensorPotential.distributed_train_step` / `distributed_test_step` | report only | distributed code that cannot be executed on this machine |

**Test groups (work item (c)).** All 14 test groups are folded; the test ids are unchanged (the folded tests stay as thin functions that call a shared helper, so the id of every test and its outcome in `baselines/outcomes_pd2.json` stay as they were).

| Group | Fold |
|---|---|
| six `test_MoNbTaW_LINEAR_lr_*`, `test_MoNbTaW_LINEAR_virial`/`_stress`, `test_MoNbTaW_FS_ef_switch`/`_HEA25`, `test_MoNbTaW_GRACE_1L`/`_bond_cutoff_and_zbl` (`test_integration_test.py`) | the reference metrics move unchanged into one table `REFERENCE`; each test calls `_run_reference(key)` |
| force and stress loss pairs (`test_loss_piecewise_linear.py`) | `_force_loss_value`, `_stress_loss_value`; the expected values and tolerances stay in the tests |
| `_build_equivariant_rms_norm_test_data` / `_build_equivariant_gate_test_data`, and the two shape tests (`test_instructions.py`) | one `_build_aa_equivariant_test_data` and `_assert_output_shape_matches_input` |
| the two `build` closures of the `cp_lL` lm-first tests (`test_spbf_layout_opt.py`) | `_cp_lL_builder` |
| the two `structure_gen` closures (`test_streaming_pipeline.py`) | `_structure_gen` |
| five wizard pairs (`test_wizard.py`) | helpers `_assert_apply_state_bfgs_family`, `_use_switch_after_section_loss`, `_ask_foundation_model_selecting`, `_maybe_ask_fp64_variant_answering` |

Checks: the collected test ids are identical before and after (1,271 ids; `pytest --collect-only`, sorted, `diff` empty). For the integration tests, whose bodies are data, a scratch script imported the old and the new module with `general_integration_test` replaced by a recorder and compared the keyword arguments of the 22 calls: equal for all; the same script reports a difference when one number is changed in the table. Seven library mutants (two in `piecewise_linear`, four in `cli/wizard.py`, one in `_ask_foundation_model`) were run against the old and the new `test_loss_piecewise_linear.py` and `test_wizard.py`: the same tests fail in both for each of them (six caught by both; the seventh, `models_in_size[0]` to `[-1]`, survives both: an existing gap, not changed here).

**Full suite.** `pytest tests -n 4 --dist load --cov=tensorpotential --cov-branch` from the repository root, `test_structured_grid.py` and `test_foundation_model_regression.py` ignored as in `baselines/`: 1262 passed, 6 skipped, 2 xfailed, 1 xpassed, 0 failed in 23 min 4 s (the machine ran other work, and coverage is on). `tools/junit_outcomes.py compare baselines/outcomes_pd2.json` reports no changed and no removed test, only 576 added ones (those of the earlier issues and the 236 of `test_sizeof_fmt.py`). Per-file coverage against `baselines/coverage_baseline.json`: identical for 111 of the 115 files; the four that differ are the four edited by the `sizeof_fmt` merge. `cli/data.py` falls from 48.12% to 47.84% (474/985 to 465/972) because nine covered statements and branches of the function moved to `formatting.py` (17 of 17 covered) together with four uncovered ones; `df2extxyz.py`, `extxyz2df.py` rise because `test_sizeof_fmt.py` now imports them; `grace_preprocess.py` loses the dead copy. The baseline was re-recorded with `--allow-fall` for that relocation (116 files, 56.34%; it was 56.16%). Library units are otherwise untouched: `ast_manifest.py check` lists the same files as before plus the four. A concurrent `--cov` run of the tool tests wrote `tools/check_clones.py` into the combined data of this run; it was removed from the report before recording. **Pandas 3.0.3** (environment rebuilt offline as in `baselines/README.md`, same command without coverage): 1184 passed, 27 failed, 6 skipped, 51 xfailed, 3 xpassed in 10 min 19 s; the 27 failures are the baselined ones (`np.array_split(<DataFrame>)`, DEPS1), `junit_outcomes.py compare baselines/outcomes_pd3.json` reports no changed and no removed test, only 576 added.

## 8. Test-suite runtime beyond parallelism (TEST4)

Measured on 2026-10-02 (CPU, 14 cores, 30 GB, pandas 2.3.3, TensorFlow 2.20, pytest-xdist 3.8.0), full suite from the repository root, `tests/test_structured_grid.py` and `tests/test_foundation_model_regression.py` ignored as in `baselines/`, thread budget of `conftest.py` active under `-n`. **Before** is `torch-backend` at `5a93aaa` (QUAL2 merged), **after** is the branch of this issue at `69f6265`; each pair was run back to back, in a separate worktree each, with no other test run on the machine (checked with `ps` before every run; the machine-wide load average was 2 to 5 from the preceding run).

**Measurement conditions matter.** A first serial run (37 min 13 s) was taken while another agent ran its own suites in a second worktree and is discarded: `test_distrib` took 254 s in it and 113 s alone, `test_savedmodel_dual_dense_signature` 45 s against 19 s. The wall-clock spread that TEST2 left unexplained (9 min 37 s to 11 min 53 s for the same tree) is probably the same cause. Compare wall-clock figures only between runs made on an idle machine, back to back.

| Run | Before | After | Ratio |
|---|---:|---:|---:|
| serial | 27 min 34 s (1654.67 s; 28 min 34 s with start-up) | 20 min 11 s (1211.65 s; 21 min 13 s) | 73.2% |
| `-n 4 --dist load` | 15 min 36 s (936.02 s) | 8 min 00 s (480.86 s; 8 min 06 s) | 51.4% |
| `-n 4 --dist load -m "not slow"` (fast tier, 1,147 tests; measured when the import-gate tests were still marked slow and so left out, they are in it now, about 30 s more per worker, not re-measured) | | 4 min 14 s (254.71 s; 4 min 20 s) | |
| `-n 4 --dist load --slow-first=off` (at `7603b2a`, see item 5) | | 11 min 24 s (684.96 s) | |
| largest process (serial / `-n 4`) | 14.5 GB / 7.7 GB | 14.1 GB / 7.2 GB | |

An earlier pair on `7603b2a` (before the UQ calculator sharing was undone, see item 4) gave 20 min 00 s serial and 7 min 45 s at `-n 4`; the two ends of that change differ by 11 s and 15 s, within the run-to-run spread, which was not measured with repeats.

Outcomes: before 1261 passed, 7 skipped, 2 xfailed, 1 xpassed (serial) and 1262 passed, 6 skipped (the xdist-only test skips serially); after 1277 (serial) and 1278 (`-n 4`) passed with the same skips, xfails and xpass. `tools/junit_outcomes.py compare` of before against after, serial and `-n 4`: no changed and no removed test id, 16 added (the new tests listed below); against `baselines/outcomes_pd2.json` the only difference is `test_construct_batches_multiple_db`, an XPASS that a log without `-rX` records as a pass.

**Reading against the exit criterion.**

| Clause | Result |
|---|---|
| serial at most 60% of the baseline | **not met**: 73.2% (16 min 33 s would be 60% of 27 min 34 s) |
| `-n 4` at most 7 min | **not met**: 8 min 00 s, 60 s over (7 min 45 s in the earlier run) |
| no `-n 4` run above a third of the serial time (9 min 11 s of the new baseline) | met in both runs made (481 s and 465 s against 551 s); three consecutive runs not made (the clause is redundant with the 7 min target by decision of 2026-10-02) |
| same pass, skip, xfail, xpass counts and ids | met (above) |
| branch coverage per file not below the baseline | met: `coverage_ratchet.py check` on a full `-n 4` run with coverage (1278 passed, 12 min 27 s): 116 files in the baseline, 0 fell, 0 gone, 0 new |
| TEST3 planted mutants still caught | the nine of `TPCalculator.__init__` (TC1 to TC9) are checked by `test_calculator_init.py`, which this issue edited: all nine caught; the test files of the other 32 are not touched |
| every cut listed, no assertion removed | met: no cut was made (items c and d below); every assertion of an edited test is kept |
| `slow` list committed | the 16 tests carry `@pytest.mark.slow` (no separate list) |
| mock listing before and after | below: 160 places, unchanged |

**Where the time is (sum of test times in the serial runs, seconds).**

| File | Before | After | `-n 4` before | `-n 4` after |
|---|---:|---:|---:|---:|
| `test_integration_test.py` (22 tests) | 734 | 704 | 746 | 711 |
| `test_import_gates.py` | 423 | 46 | 636 | 138 |
| `test_distrib.py` | 111 | 107 | 204 | 200 |
| `test_uq_integration.py` | 72 | 75 | 127 | 86 |
| `test_instructions.py` | 55 | 37 | 71 | 96 |
| `test_calculator.py` | 54 | 51 | 103 | 102 |
| `test_databuilder.py` | 45 | 43 | 33 | 50 |
| `test_graph_split.py` | 39 | 41 | 92 | 76 |
| `test_uq_features.py` | 28 | 27 | 71 | 64 |
| `test_fm_shift_auto.py` | 19 | 10 | 25 | 19 |
| `test_calculator_init.py` | 16 | 12 | 45 | 22 |
| rest | 57 | 58 | 132 | 136 |

The integration tests are 44% of the serial time and no change in this issue touches them (see item (c)). Under `-n 4` the same tests take about 38% longer than serially (sum of test times 2,284 s against 1,652 s before; 1,701 s against 1,210 s after), so four workers give a speed-up of 1.77, not 4; the cause (memory bandwidth, three threads per worker) was not isolated.

**What was changed, one commit each.**

1. `test_import_gates.py`: gate 1 imported each of the 115 modules in its own interpreter and paid the TensorFlow import each time. One interpreter now imports `tensorpotential` (a fresh `import tensorpotential.X` always runs the package `__init__` first) and forks one child per module, so each module is still imported on its own; the test ids are the same. 423 s to 46 s serial. Planted mutants (a missing third-party package, a missing name in a sibling module) are each caught by exactly the affected module's test; a logic test covers a missing module and a module found outside the expected root. The module-scoped fixture runs once per worker, about 30 s.
2. Slow tier: `slow` registered in `pytest.ini`; the 16 tests of 30 s or more (serial reference run) carry `@pytest.mark.slow`; `-m "not slow"` is the quick loop (1268 of 1284 collected tests; the 115 import-gate tests are in it, they take a fraction of a second each and the fixture they share costs about 30 s once). They are deliberately not marked: marked slow, the hook dealt them over all four workers and every worker ran that fixture (see item 5).
3. `test_fm_shift_auto.py`: the shift is computed once per preset instead of in three tests (23.9 s to 14.0 s for the file alone).
4. Shared read-only models: the default one-layer model of `test_calculator_init.py` (14 builds to 1), the Cu two-layer models of `test_calculator.py` and `test_databuilder.py` (5 of 8 builds). A model that is saved, exported or edited, and the `"reshape"` spelling of `dense_nbr`, keep their own build. Each shared object is fingerprinted when built (name, dtype, shape, values of every variable; string variables by content) and compared when its scope ends, so a test that changes it makes the run fail; the first version of the fingerprint hashed the pointers of a string variable and was caught by that very check, now fixed and tested (four tests, three planted mutants caught). Lint ratchet: two E702 fewer in `test_calculator.py`. **Sharing the UQ feature calculator was tried and undone**: `test_uq_features.py` and `test_uq_gmm.py` were the only callers of `setup_feature_calculator` without `param_dtype`, so the shared (explicit-dtype) calculator stopped running the dtype inference of `metadata_utils.resolve_param_dtype`, and the coverage ratchet reported `metadata_utils.py` falling from 72.73% to 50.00%. The two tests are back to their previous form.
5. `--slow-first` (default `deal`): `pytest-xdist --dist load` hands the collected list to the workers in blocks (the first block to the first worker). The hook in the root `conftest.py` deals the tests marked `slow` round-robin, in collection order, over the first block of each worker (`tests/slow_first.py`); all at the front would all go to worker 0. `--slow-first=off` restores the collection order; serial runs are unchanged. Logic tests (the block size against hand-computed values, a real two-worker session) and seven planted mutants (stride, block size, one-worker guard, padding, dropped or reversed tests) are all caught. Measured at `7603b2a`, where the order came from a duration-sorted list: 7 min 45 s with `deal`, 11 min 24 s with `off` (one pair). The list was then replaced by explicit marks; the first version also marked the 115 import-gate tests, and an A/B against the duration-sorted commit (`-n 4`, alternating, same machine state, which was slower than during the earlier runs) gave 9 min 28 s and 9 min 31 s for the earlier commit and 10 min 13 s and 10 min 48 s for the marked-gate version: the summed time of `test_import_gates.py` went from 162 s to 290 s, because every worker that received some of the dealt gate tests ran the 30 s fixture. With the gate tests unmarked a second A/B gave 563 s and 563 s for the earlier commit and 570 s and 563 s for the final one, i.e. no difference. (Before the A/B, five `-n 4` runs of the code with explicit marks took 9 min 54 s to 12 min 15 s with other work on the machine, and a scheduler simulation put the marks equal to the list; the simulation did not model the per-worker fixture and missed this effect.)

**Item (c): merging runs that differ only in the assertions.** No candidate exists. The 22 integration tests (19 run, 3 skipped) each use their own input file; the files differ from `MoNbTaW-LINEAR/input.yaml` in the loss, the optimiser, the scheduler, the dtype, the number of structures or epochs, and each test compares different reference metrics. Merging two would mean one run and one set of reference values, i.e. regenerating a golden reference, which this issue may not do. `test_MoNbTaW_FS_restart` is already one test with several runs. The other repeats were of models and results, not of runs, and were shared (item b). QUAL2 had been merged first, as the issue asked.

**Item (d): reducing work.** Nothing was cut. A `gracemaker` run of `MoNbTaW-LINEAR` takes about 24 s, of which the first epoch (tf.function tracing and autograph conversion) is about 9 s, the SavedModel export about 3 s and the TensorFlow import about 4 s; the second epoch takes 1 s. The cost is per run, not per epoch, and the reference metrics depend on the two epochs, so fewer epochs would change what is compared.

**Item (f): mocks, stubs and `monkeypatch`.** Counted with an AST scan (calls of `monkeypatch.*`, `Mock`, `MagicMock`, `patch`, and classes named `Fake*`, `Stub*`, `Mock*`) over `tests/` and `tools/tests/`: **160 places in 15 files before, 160 after, the same places** (this issue adds none). By what they replace:

| Files | Places | Replaces |
|---|---:|---|
| `test_wizard.py`, `test_gen_tensor_wizard.py` | 96 | the interactive prompts (`_ask_select`, `_ask_text`, `_ask_confirm`, `_ask_path`, `builtins.input`), the terminal output helpers and the presence of the optional `questionary`: user input and console, no numeric path |
| `test_origin_sidecar.py` | 17 | the download of foundation models (`get_or_download_model`, `check_origin`), the cache directories, the model registry and one environment variable: network and file-system boundary |
| `tools/tests/test_board.py`, `test_check_pr_branch.py` | 20 | the `gh` command line, `subprocess.run`, `sys.argv`, `git diff`: external service, fork-only tools |
| `test_uq_cli.py`, `test_detect_multigpu_intent.py` | 7 | working directory, `sys.path`, one environment variable: environment, no logic replaced |
| `test_thread_budget.py`, `test_uq_common.py` | 8 | `os.cpu_count` (the hardware) and environment variables |
| `test_grace_utils_export.py` | 4 | `sys.argv` of a command-line entry point |
| `test_calculator_init.py` | 2 | `sys.modules` entry for the absent `tensorpotential.experimental` package |
| `test_instructions.py`, `test_uq_features.py`, `test_structured_grid.py` | 4 | small stand-ins for a metadata object (`MockOrigin`, `_Stub`, `_StubInstr`) |
| `test_intra_epoch_checkpoint.py` | 2 | **the training object** (`MagicMock` for `TensorPotential` in `train_one_epoch`) |

The last row is the only place that stands in for something the rule discourages: the unit under test is the fast-forward logic of `train_one_epoch`, and the mock replaces the training step that does the numerics, with no number asserted; recorded here and not changed (a real `TensorPotential` would turn a control-flow test into a training run).

**Item (g): tail latency.** Dealing the slow tests is the whole change (item 5 above). `test_distrib.py` was not split: its three stages cost 25 s (data), 43 s and 42 s (two `gracemaker -m` fits) when run alone, 113 s together, and a split would lower the longest single test but not the total, which is what bounds the wall time here (sum of test times 1,718 s over four workers is 430 s of the 465 s). The second fit exists to check `TF_CONFIG` handling and could be shortened with a smaller input; not done, because the first epoch's tracing, not the epoch count, dominates it.

**Found on the way.** Two things that the coverage ratchet exposed and the timing comparison alone would have hidden: (1) the UQ calculator sharing above; (2) `TPModel.__repr__` (`tpmodel.py` 64.57% to 64.44%) was run by nothing except TensorFlow's "function retraced too often" warning, which formats `repr(model)`; `test_fm_shift_auto` stopped triggering it once the shift is computed once, so `test_repr_numbers_the_instructions_in_file_order` now pins it directly. Both are the reason the coverage clause is checked on the final tree.

**Identified, not done (each under 10 s of the serial time, risk higher than the gain).** The same `get_gmm_uq_calculator` build in two tests of `test_uq_integration.py` (6 to 8 s; ASE caches results per `Atoms`, so the second test could skip the computation it is meant to exercise); `tensor_model` of `test_grace_2_tensor_preset.py` (3 builds, 4 to 5 s); the padding tests of `test_calculator.py` (4 to 6 s); the `two_layer` models of `test_calculator_init.py` (4 to 5 s); one exported SavedModel shared by the dual-signature and the export-flag tests (about 10 s, but the flag test asserts the state of the model it exported itself).

**What the numbers do not cover.** One `-n 4` run per configuration (the three-consecutive-runs clause was not repeated); the `--slow-first` comparison is one pair, and the timings of the first table were taken at `69f6265`, before the list was replaced by explicit marks (item 5: the final commit equals it within 7 s in an A/B on a slower machine state, its absolute time was measured once more on 2026-10-03 on a quiet machine, mean load from other processes 0.26 cores: 9 min 54 s (593.73 s, 1275 passed, 6 skipped, 2 xfailed, 1 xpassed, no changed or removed test id against the baseline). That is 114 s above the 8 min 00 s of the day before for the same tests, and the earlier commit also took 9 min 28 s in the A/B of that day, so the machine itself ran slower on 2026-10-03; absolute `-n 4` times are comparable only within one session, and the 51% ratio of the first table (a back-to-back pair) is the figure that carries; pandas 3 was not run (no test or library change depends on it); nothing was run on GPU or HPC. The fixture of the import gate runs once per xdist worker (about 30 s each), so it costs more at `-n 8`. The coverage run of the final tree took 12 min 27 s with `-n 4` (QUAL2 recorded 23 min 4 s, with other work on the machine).

## 9. Code quality without architectural change (QUAL1)

Measured on 2026-10-03 on the QUAL1 branch (CPU, `ruff 0.16.7 --isolated`, TensorFlow 2.20) against `torch-backend` at `39a3c86`.

**Edits.** `ruff --select B006,B008,B904,B023,RUF013,S110`, library without `compat/pace/`, before and after:

| Rule | Item | Before | After | Where it stays |
|---|---|---:|---:|---|
| RUF013 | (a) implicit `Optional` rewritten as `X \| None` | 157 in 19 files | 0 | |
| B008 | (b) default-argument instances of `ComputeStructureEnergyAndForcesAndVirial` and `ComputeBatchEnergyAndForces` | 4 | 0 | |
| B006 | (b) `grace_2(n_rad_max=[32, 42])` | 1 | 0 | |
| B904 | (c) `raise ... from err` | 5 | 0 | 3 in `compat/pace/` (out of scope) |
| S110 | (c) `except: pass` | 2 of 6 | 0 | 4 in units the suite does not run (below) |
| B023 | (d) loop variable in a `map(lambda ...)` | 1 of 2 | 0 | `grace_collect.main` (below) |

(a) is annotation-only: for each of the 19 files the AST with annotations removed equals the AST of `HEAD`. `TensorPotential.__init__`, `TPModel.__init__` and `grace_2` treat `None` as "build the default now"; the values are unchanged (widths 32 and 42, the same two function classes), and an explicit argument is kept by identity. A caller that used to pass `None` for a function on purpose would now get the default, and no caller in the tree does.

**Edits that moved here (units the suite does not execute, coverage rule).** `scripts/grace_dashboard.py::extract_n_params` (silent `except`, 0/18) and `extrapolate_series` (0/42; the module cannot be imported without `flask`, which is not a declared dependency), `uq/cli/build/master.py::run_master` (two silent `except`, 1/361) and `scripts/grace_collect.py::main` (late-binding lambda at line 252, 0/108) are not edited. Their ruff hits are the only ones outside `compat/pace/` that remain.

**Report only (item (e)).**

- `FCRight2Left.build` draws normal random numbers for `w_right` when `init_vars="zeros"` (`instructions/compute.py:3755-3760`; the `w_left` branch above it uses `tf.zeros`). The weights are not zero and consume random state. Not changed: it would change the weights of models built with `init_vars="zeros"`.
- The `rankmax` NameError is F1 above and sits in `compat/pace/` (out of scope).
- TODO-style markers: 52 in total (`grep -rnE '\b(TODO|FIXME|XXX|HACK)\b' --include='*.py' tensorpotential`), 48 outside and 4 inside `compat/pace/`; F3 counted 47 and 5.
- `print` calls: `ruff --select T201` gives 365 (346 outside `compat/pace/`, 19 inside); the issue text said 53. Outside `cli/`, `scripts/`, the wizards and `compat/pace/` there are 36 (`calculator/foundation_models.py` 15, `calculator/asecalculator.py` 11, `utils.py` 4, `__init__.py` 4, one each in `uq/feature_extraction.py` and `data/process_df.py`). They are user-facing messages and are left on purpose.

**Side effects.** (1) Honest `X | None` annotations make `ty` report two more `no-matching-overload` findings, `' x '.join(self.coupling_origin)` in two `__repr__` methods of `compute.py` (the parameter can be `None`); the owner approved recording the rise in `baselines/lint_ratchet.json` (2 to 4). (2) The minimum Python version is now 3.11 (`pyproject.toml`; it was `>=3.9`), by decision of the owner; the ruff target of the new packages follows.

## 10. pandas 3 readiness (DEPS1)

Measured on 2026-10-03 on the DEPS1 branch (cut from `91103a8`), CPU, Python 3.12.3, `pytest -n 4 --dist load` with the two ignores of `baselines/README.md`. pandas 2.3.3 is the project environment, pandas 3.0.3 is a second environment built offline from the same lock (`baselines/README.md`).

**Change.** The two `np.array_split(<DataFrame>, n)` calls of `data/databuilder.py` (in `estimate_n_buckets` and `split_batches_into_buckets`) go through one private helper, `_split_rows`, that splits the row positions and takes `df.iloc[ix]`; the chunks are the same as before (sizes and row labels pinned by `tests/test_split_batches_into_buckets.py`, written first and green on the old code). The blanket `FutureWarning` and `DeprecationWarning` filters of `scripts/grace_predict.py` are replaced by one filter for the `DeprecationWarning` that ASE raises through NumPy 2.5 (`Setting the shape on a NumPy array`, 3 per two structures); with every filter removed and `-W always`, that was the only warning of a `grace_predict` run, and no pandas warning occurred.

| Run | passed | failed | skipped | xfailed | xpassed |
|---|---:|---:|---:|---:|---:|
| pandas 2.3.3 | 1282 | 0 | 6 | 2 | 1 |
| pandas 3.0.3 | 1282 (1281 in the full run, `test_distrib` re-run, see below) | 0 | 6 | 2 | 1 |
| pandas 2.3.3, `PANDAS_COPY_ON_WRITE=warn` | 1282 | 0 | 6 | 2 | 1 |
| pandas 2.3.3, `-W error::FutureWarning -W error::DeprecationWarning` | 1183 | 66 (+33 errors) | 6 | 2 | 1 |

`test_distrib` failed in the first pandas 3 run with exit status 127: the test starts `grace_preprocess` by name and the environment's `bin` was not on `PATH` (the interpreter was called by path). With `PATH` set it passes (1 passed, 1 min 54 s). The 27 failures of the pandas 3 baseline (`outcomes_pd3.json`) are all gone, 26 through the fix and the 27th (`test_distrib`) through the same fix in stage 3.

**Triage of the warn-mode runs.**

- `PANDAS_COPY_ON_WRITE=warn`: no test fails and the log holds no pandas warning (0 occurrences of "pandas", `ChainedAssignment` or `SettingWithCopy`); the library does not rely on chained assignment or on views of a frame.
- `-W error`: all 100 error lines are `DeprecationWarning`s from outside pandas, none from pandas: 93 NumPy 2.5 "Setting the shape on a NumPy array" (raised in `ase/atoms.py`), 5 "`__array__` implementation doesn't accept a copy keyword" (`instructions/compute.py:477-514`, TensorFlow tensors passed to `np.array`), 2 `importlib.resources.open_text` (`cli/wizard.py:1043`). They affect NumPy 2.5 readiness, not pandas 3, and are not fixed here.
- pandas 3.0.3 emits one warning of its own in the whole suite: `Pandas4Warning` for `pd.concat(..., copy=False)` at `cli/data.py:957` (inside `load_and_prepare_datasets`, a 550-line function; `cli/data.py` is covered 48% in `baselines/coverage_baseline.json`, so it may not be modified before characterization tests exist). `Pandas4Warning` derives from `DeprecationWarning`, not from `FutureWarning`. Reported on the issue as a finding, not fixed here.
- `FutureWarning`: the only ones left on pandas 2.3.3 are ASE's `FiniteDifferenceCalculator` notices in `tests/test_calculator.py`; none comes from pandas. The 48 `DataFrame.swapaxes` warnings of `estimate_n_buckets` are gone.

**Pipeline.** `grace_preprocess` stages 1 to 4 on `tests/data/MoNbTaW_train50.pkl.gz` (strategy `neighbours`, batch 400, 3 buckets, cutoff 5) give the same stage 2 table hash, the same 46 stage 3 batches (hash of every tensor of every batch) and the same `stats.json` on pandas 2.3.3 and 3.0.3.

**What this does not cover.** `grace_collect` and `gracemaker` data loading on real user datasets were not run with pandas 3 beyond what the suite exercises; `df2extxyz` ran on both versions with `-W error::FutureWarning`; a pickle written by pandas 3 cannot be read by pandas 2 (unchanged); the pin `pandas<3` is not relaxed here (decision D11, the relaxation belongs to the packaging issues); nothing ran on GPU or HPC.

## Provenance

Counts re-measured on the tree named at the top; the F1 reproduction was run with the project environment. Compared with the issue text: star imports are 3 in `.py` files (4 with the notebook), everything else (1 `NameError`, 8 bare `except`, 52 markers, the unimported package) matches.

Decided after the first draft: `compat/pace/` is out of scope with no change allowed (owner, 2026-10-01); the draft's open question about retiring it is withdrawn.
