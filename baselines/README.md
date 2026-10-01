# Baselines of the untouched tree (M0.1)

Taken on 2026-09-30 from tag `pre-cleanup` (`cc1bb38`, upstream 0.6.1) on this machine (CPU, TensorFlow 2.20.0,
numpy 2.5.3, ASE 3.29.0, Python 3.12.3). Stage 0 must leave all of this unchanged (gate G0).

| File | What | Check against it |
|---|---|---|
| `ast_manifest.json` | sha256 of the AST of each of the 165 tracked `.py` files of the untouched tree; fork-only paths (the prefixes of `tools/check_pr_branch.py`: `tools/`, `tests_torch/`, `tensorpotential/core/`, ...) are ignored in `check`, `--exclude PREFIX` adds more | `python tools/ast_manifest.py check baselines/ast_manifest.json` |
| `junit_pd2.xml`, `outcomes_pd2.json` | full suite, pandas 2.3.3 | see below |
| `junit_pd3.xml`, `outcomes_pd3.json` | full suite, pandas 3.0.3 | see below |
| `oracle_snapshot.npz` (git-ignored, 52 MB) and `oracle_snapshot.meta.json` | TensorFlow numerics of the three test yamls | `python tools/oracle_snapshot.py compare baselines/oracle_snapshot.npz new.npz` |
| `lint_ratchet.json` | ruff (E, F, ERA001, F401, F841, F811) and ty findings of legacy code per file and rule, with the tool versions (M0.7); the strict packages have no baseline | `python tools/lint_ratchet.py check` |

The plan said 183 `.py` files; `git ls-files '*.py'` gives 165 (243 tracked files in total, as the plan also says), so the
manifest covers 165.

## Suite baselines

Run from inside `tests/` (the untouched tree depends on the working directory; since M0.9 the suite gives the same
outcomes from the repository root, where the paths of the `--ignore` options below read `tests/<file>`), on a `git archive` copy of the tag, in a network-less
namespace (`unshare -rn`), with `--ignore=test_structured_grid.py` (imports the non-existent `tensorpotential.experimental`)
and `--ignore=test_foundation_model_regression.py` (needs foundation weights, HPC only), `-rxX`, pytest 9.1.1 as an overlay.

| pandas | passed | failed | skipped | xfailed | xpassed | time |
|---|---|---|---|---|---|---|
| 2.3.3 | 686 | 0 | 6 | 2 | 1 | 33 min |
| 3.0.3 | 659 | 27 | 6 | 3 | 0 | 13 min |

The 27 failures on 3.0.3 are the `np.array_split(<DataFrame>)` break of `data/databuilder.py` (26 `IndexError`s in
`test_integration_test` (19), `test_databuilder` (5), `test_bucketing_heuristic`, `test_gen_tensor_integration`) plus
`test_distrib`, whose shell script runs `grace_preprocess`; M0.6 fixes them. `test_construct_batches_multiple_db` is an
XPASS on 2.3.3 and an XFAIL on 3.0.3. The six skips: three need the trained model `MoNbTaW-GRACE/seed/42` (not in the
repository), two are marked "skip custom model" and one "Deprecated due to new API". The two XFAILs of
`test_activate_reduce_*` are unconditional `xfail` marks (no reason given) and occur on both versions. Times are
wall-clock with both runs, other work and a foreign pytest process sharing the 14 cores; they are not benchmarks.

Compare a later run with `python tools/junit_outcomes.py summarize junit.xml new.json --log pytest.log` and
`python tools/junit_outcomes.py compare baselines/outcomes_pd2.json new.json` (exit status 1 on any change). Pass the log,
otherwise an XPASS reads as a pass. Junit class names carry the rootdir (`tests.test_a`), the log does not; the tool
matches by suffix and refuses an ambiguous id.

Rebuild the pandas 3 environment (offline, from the same lock; `uv.lock` stays unchanged):

```bash
UV_PROJECT_ENVIRONMENT=/path/to/env_pd3 uv sync --frozen --offline
uv pip install --offline --python /path/to/env_pd3/bin/python pandas==3.0.3
```

## Numeric oracle snapshot

`tools/oracle_snapshot.py` builds the float64 TensorFlow model of `model_grace.yaml`, `model_grace_2L_omat.yaml` and
`model_grace_2L_omat_large_base.yaml`, overwrites every **trainable** variable with seeded values (the cutoff and element
maps are constants and stay), and records for three structures of `tests/data/MoNbTaW_test50.pkl.gz` (Nb2, Mo7W9, Ta26,
periodic) the energy, atomic energies, forces, virial, stress and every tensor the instructions write into the input
dictionary: 369 arrays, one thread, CPU. Its tests check forces and stress against central finite differences of the
energy (an oracle that shares no code with the analytic gradient) and translation invariance.

**Reproducibility, measured.** Nineteen snapshots of the unchanged library (both pandas versions; sequential and
concurrent runs; with and without ASLR, with `OMP_NUM_THREADS=1`, with oneDNN off) give identical energies, atomic energies, forces, virials and stresses in every
case. 22 of the 369 arrays, all intermediate tensors of the `large_base` model (`YI`, `B`, `B1`, `BB`, `BB1`, `BB2`,
`BBB`, `BBBB`), take one of two values that differ by about one ulp (largest difference `8e-28`, `8e-18` of the array's
largest element). The variation is between processes and does not follow the pandas version, ASLR, `OMP_NUM_THREADS` or
`TF_ENABLE_ONEDNN_OPTS`; its cause was not pinned down. Hence `compare` defaults to `--scale-rtol 1e-12` (tolerance
relative to the largest element of each array; `0` gives an exact comparison). Planted checks: a swapped index in the
virial of `tpmodel.py` (scratch clone) is flagged in `stress` and `virial` only, with a maximum scaled difference of `0.069`.
