# Cleanup report: findings that are reported, not fixed (M0.5)

Status: **draft for review**. No source file was edited; this report only records what the tree contains, what was measured, and the rules that govern the next removals and upstream merges.

Measured on 2026-10-01 on `torch-backend` at `c313e3e` (after M0.3), CPU, Python 3.12.3, `ruff 0.16.7 --isolated`, TensorFlow 2.20. Commands are given so each number can be reproduced.

## 1. Findings

| # | Finding | Count | Where | Action taken |
|---|---|---:|---|---|
| F1 | Latent `NameError` (`rankmax` instead of the parameter `ranksmax`) | 1 | `compat/pace/potentials/ace.py:975` | none; reported (F1) |
| F2 | Bare `except:` | 8 | all in `compat/pace/` | none; reported (F2) |
| F3 | `TODO`/`FIXME`/`XXX`/`HACK` markers | 52 | 47 in live code, 5 in `compat/pace/`; none links an issue | none; reported (F3) |
| F4 | Star imports | 3 + 1 | 3 in `compat/pace/`, 1 in a notebook | none; reported (F4) |
| F5 | `compat/pace/` is imported by nothing outside itself | 17 files, 4,399 lines | `tensorpotential/compat/pace/` | none; removal candidate (F5) |

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

Conclusion: a removal candidate, **not done** (D5(b), and M0.5's scope). It would satisfy gates 2 and 5 of the removal protocol below; gates 1 (a replacement is default), 3 (an exact deletion set) and 4 (no tests to replace) are for the owner to decide, together with whether a user of the legacy PACE-fitting workflow exists whom the maintainers want to hear first (a question for upstream, not for the fork).

## 2. Class E register (disabled wiring of documented or user-visible options)

Copied from `cleanup_triage.md` (M0.2). Locations are re-measured on the current tree: M0.3 deleted comment blocks above these lines, so they moved by a few lines; the id is the stable key. Default action: keep. Each row needs a decision from the owner (retire, or complete).

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

Grouped by the decision they need: **LoRA** (3 rows, one decision), **Gaussian `normalized`** (3 rows, one decision, touches the model format), **`stress_units` in `grace_preprocess`** (1), **`grace_utils aux_model --aux`** (3 rows, one decision; the documentation currently describes behaviour the code does not have), **two disabled integration tests** (2). Five decisions in total. None of them is taken here; M0.2 recorded that a decision is needed, M0.3 left all 12 blocks in place (verified: the 12 blocks are present above).

## 3. Rules

### 3.1 Upstream-sync procedure

State today: `origin` is the fork; `upstream` is the ICAMS repository with its push URL disabled. `git rev-list --left-right --count upstream/master...origin/torch-backend` gives `0 26` (upstream has nothing the fork lacks; the fork has 26 commits more).

Incoming (upstream changes into the fork):

1. `git fetch upstream`, then the rev-list count above. If the left number is 0, there is nothing to do.
2. Otherwise branch `sync-upstream-<yyyymmdd>` from `origin/torch-backend` and **merge** `upstream/master` (a merge keeps the history that a later `pr/U*` squash-merge needs; no rebase of published branches). Conflicts are expected where upstream edits near a block that M0.3 deleted or a file the fork changed: keep upstream's behaviour, re-apply the fork's removal only if the deleted block is still dead.
3. Verify before the sync PR is opened, from the repository root: the full suite with `-n 4 --dist load` compared with `baselines/outcomes_pd2.json` through `tools/junit_outcomes.py compare`; the AST manifest check for the files upstream changed (every `changed:` line must be explained by the upstream diff); the oracle snapshot if `instructions/`, `functions/`, `tpmodel.py` or `poly.py` changed. A sync is one PR into `torch-backend`, opened and merged by the owner.
4. Never push to `master` of the fork's `origin` as a way to track upstream, and never push to `upstream`.

Outgoing (a unit offered to ICAMS), as decided in D6 and D17:

1. Cut `pr/U<n>-<slug>` from `upstream/master`, never from `torch-backend`; merge that branch into `torch-backend` at once so the fork never waits.
2. `python tools/check_pr_branch.py` must pass (no `CLAUDE.md`, `.claude/`, `tools/`, `baselines/`, `plan/`, `tests_torch/`, `tensorpotential/torch_backend/`, `tensorpotential/core/`, ... in the diff against `upstream/master`).
3. No AI co-author trailer in commits of `pr/U*` branches (D17e); PR text for upstream is drafted locally, passed through `python tools/board.py sanitise`, and shown to the owner first.
4. Nothing is opened, pushed or commented upstream without an explicit go for that unit; the owner checks the licence clause of the upstream PR template with the institution before the first PR (D6); at most two upstream PRs are open at a time.
5. If upstream merges the unit (squash), merging `upstream/master` is a no-op or a trivial conflict; if upstream changes it during review, rebase the fork's copy onto the merged version.

Consequence of M0.3 for later syncs (an expectation, not yet observed because upstream has not moved): the 169 deleted blocks and the other clean-up edits are fork-only (the U6 upstream units were skipped, so nothing replays them upstream), hence a future merge can conflict wherever upstream edits next to a deleted block. The divergence ledger of M1.11 (D17b) will list fork-modified upstream files; until it exists, `git diff --stat upstream/master...origin/torch-backend -- tensorpotential` is the list.

### 3.2 Removal protocol for any future retirement of code

Adapted from the retirement gates (RET-1 to RET-6) that MACE used for its legacy code; applies to deleting a module, a class, an option or a package (for example `compat/pace/`, or a class-E item above). A comment-only deletion follows M0.2 and M0.3 instead (triage table, AST manifest equal).

1. **The replacement is the default and its parity is green.** A retirement names what replaces the code and shows the comparison (suite outcome equal to `baselines/`, oracle snapshot within tolerance when numerics are touched) before the deletion, never in the same commit.
2. **Nothing live imports the code.** Evidence is an import-resolution check (every `import` / `from` in the tree resolves after the change) and an import smoke (`python -W always -c "import <every module>"`), plus the dynamic routes of this repository: `importlib.import_module` calls, `tpmodel.__getattr__` and `instructions.__getattr__` (saved `model.yaml` files import classes by name through them, so a class deletion is a model-format break), console scripts of `pyproject.toml`, and entry points.
3. **An exact deletion set and a not-in-set list.** A machine-readable list of the files, classes and functions that go (ids, not line numbers), and the list of things that look related and stay, in the PR. The AST manifest and the pragma/shebang guard cover what the AST check cannot see (comments).
4. **Every removed test has a named successor**, or a stated reason that none is needed; the junit comparison against `baselines/outcomes_pd2.json` explains each missing id.
5. **A dependents sweep for everything shipped in the wheel**: `find_packages` and `package_data` in `setup.py`, `[project.scripts]`, the documentation (`docs/`), `examples/` and notebooks, the Docker and CI files, and users' saved models (the persisted-API gotcha: a constructor argument or default is part of every saved model).
6. **Persisted formats and public options get a deprecation step** (an actionable error or warning naming the replacement) before they disappear; nothing falls back silently.
7. One concern per PR, the owner decides, and `compat/pace/` stays untouched until a retirement PR names it.

## 4. Decisions this report needs from the owner

1. **`compat/pace/`**: keep untouched (the default), or open a retirement issue following 3.2? The evidence is in F5.
2. **Class E**: five decisions (LoRA, Gaussian `normalized`, `stress_units` in `grace_preprocess`, `grace_utils aux_model --aux`, the two disabled integration tests); a model-format break is involved only in the Gaussian one.
3. **Upstream issues (U10)**: whether to report F1 and the class-E documentation mismatch (`--aux`) as issue texts to the maintainers; nothing is sent without an explicit go (D6).

## Provenance

Counts re-measured on the tree named at the top; the F1 reproduction was run with the project environment. Compared with the issue text: star imports are 3 in `.py` files (4 with the notebook), everything else (1 `NameError`, 8 bare `except`, 52 markers, the unimported package) matches.
