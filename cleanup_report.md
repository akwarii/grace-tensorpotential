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

## Provenance

Counts re-measured on the tree named at the top; the F1 reproduction was run with the project environment. Compared with the issue text: star imports are 3 in `.py` files (4 with the notebook), everything else (1 `NameError`, 8 bare `except`, 52 markers, the unimported package) matches.

Decided after the first draft: `compat/pace/` is out of scope with no change allowed (owner, 2026-10-01); the draft's open question about retiring it is withdrawn.
