# Baselines of the untouched tree (SAFE1)

Taken on 2026-09-30 from tag `pre-cleanup` (`cc1bb38`, upstream 0.6.1) on this machine (CPU, TensorFlow 2.20.0,
numpy 2.5.3, ASE 3.29.0, Python 3.12.3). Stage 0 must leave all of this unchanged (gate GATE-CLEAN).

| File | What | Check against it |
|---|---|---|
| `ast_manifest.json` | sha256 of the AST of each of the 165 tracked `.py` files of the untouched tree; fork-only paths (the prefixes of `tools/check_pr_branch.py`: `tools/`, `tests_torch/`, `tensorpotential/core/`, ...) are ignored in `check`, `--exclude PREFIX` adds more | `python tools/ast_manifest.py check baselines/ast_manifest.json` |
| `junit_pd2.xml`, `outcomes_pd2.json` | full suite, pandas 2.3.3 | see below |
| `junit_pd3.xml`, `outcomes_pd3.json` | full suite, pandas 3.0.3 | see below |
| `oracle_snapshot.npz` (git-ignored, 52 MB) and `oracle_snapshot.meta.json` | TensorFlow numerics of the three test yamls | `python tools/oracle_snapshot.py compare baselines/oracle_snapshot.npz new.npz` |
| `oracle_snapshot_wide.npz` (git-ignored, about 220 MB) and `oracle_snapshot_wide.meta.json` | the same plus float32 parameters, layout options, edge structures and presets: 4,130 arrays (SAFE2) | `python tools/oracle_snapshot.py compare baselines/oracle_snapshot_wide.npz new.npz` |
| `lint_ratchet.json` | ruff (E, F, ERA001) and ty findings of legacy code per file and rule, with the tool versions (TOOL1); the strict packages have no baseline | `python tools/lint_ratchet.py check` |
| `coverage_baseline.json` | executed and total statements plus branches of each of the 116 library files from a full-suite run with branch coverage on `torch-backend` **with** the TEST3 tests and QUAL2 (not the untouched tree; 56.34% in all, coverage.py 7.16.2) | `python tools/coverage_ratchet.py check cov.json` (`cov.json` from `pytest tests -n 4 --dist load --cov=tensorpotential --cov-branch --cov-report=json:cov.json`); a file fails only when its covered share falls **and** its uncovered statements plus branches rise (a refactor that removes covered code passes; TOOL2); `record` refuses such a fall unless `--allow-fall` |
| `clone_baseline.json` | groups, functions and redundant lines of the duplicated functions (8 lines or more; identical bodies, and bodies equal after abstracting identifiers) of `tensorpotential/` without `compat/pace/` and of `tests/` (QUAL2); the totals of a tree may not rise | `python tools/check_clones.py check` |
| `probes/probe_<yaml>.json` | SPEC2: variable names, shapes, dtypes and checkpoint keys of the three test yamls (random weights, float64 and float32 parameters), the tensor attributes with the dtype they are created in, a per-instruction graph trace (float dtype of the operations, casts, creation against use of each captured value) and neighbour-list against model timings; each timing section carries `timings_foreign_cpu_cores` (near 0: idle machine) | `python tools/grace_probe.py write baselines/probes` (the real-checkpoint probe SPEC2b: `--yamls model.yaml --label NAME --checkpoint PREFIX --dtypes float32`); timings are quotable only when `timings_foreign_cpu_cores` is near 0 |
| `wheel_files_upstream.txt` | the file list of the wheel built from `upstream/master` (`cc1bb38`, 0.6.1, setup.py packaging): 122 files, one per line, sorted (CI1) | `tests/test_packaging.py` compares the wheel of this tree with it; regenerate with `git archive upstream/master \| tar -x -C DIR && (cd DIR && uv build --wheel --out-dir OUT) && unzip -Z1 OUT/*.whl \| sort` |

The plan said 183 `.py` files; `git ls-files '*.py'` gives 165 (243 tracked files in total, as the plan also says), so the
manifest covers 165.

## Suite baselines

Run from inside `tests/` (the untouched tree depends on the working directory; since TEST1 the suite gives the same
outcomes from the repository root, where the paths of the `--ignore` options below read `tests/<file>`), on a `git archive` copy of the tag, in a network-less
namespace (`unshare -rn`), with `--ignore=test_structured_grid.py` (imports the non-existent `tensorpotential.experimental`)
and `--ignore=test_foundation_model_regression.py` (needs foundation weights, HPC only), `-rxX`, pytest 9.1.1 as an overlay.

| pandas | passed | failed | skipped | xfailed | xpassed | time |
|---|---|---|---|---|---|---|
| 2.3.3 | 686 | 0 | 6 | 2 | 1 | 33 min |
| 3.0.3 | 659 | 27 | 6 | 3 | 0 | 13 min |

The 27 failures on 3.0.3 are the `np.array_split(<DataFrame>)` break of `data/databuilder.py` (26 `IndexError`s in
`test_integration_test` (19), `test_databuilder` (5), `test_bucketing_heuristic`, `test_gen_tensor_integration`) plus
`test_distrib`, whose shell script runs `grace_preprocess`; DEPS1 fixes them. `test_construct_batches_multiple_db` is an
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

## Wide numeric oracle snapshot (SAFE2)

`oracle_snapshot_wide.npz` is the first snapshot widened to the paths that the clean-up could change without the first one
noticing. It was recorded by the same tool on a `git archive` copy of the tag `pre-cleanup` (the tool of the SAFE2 branch run
with `PYTHONPATH` on the copy; `tensorpotential.__file__` printed to make sure the copy is imported), one thread, CPU, TensorFlow 2.20.0.
Keys are `<model>/<case>/<quantity>`; the 369 keys of `oracle_snapshot.npz` are in it (compared with the old file: no key
outside the tolerance, largest difference `8e-28`). `oracle_snapshot_wide.meta.json` lists the models, structures, variables and versions.

| Model label | What | Arrays |
|---|---|---|
| `model_grace`, `model_grace_2L_omat`, `model_grace_2L_omat_large_base` | float64, as before; cases `s0`-`s2` as before | 369 + 4 edge cases each |
| `<yaml>.f32` | the same three yamls with float32 parameters and float64 inputs (the precision of the foundation models) | 3 models |
| `<yaml>.lm_first` | `lm_first: true` set in the serialised yaml on every instruction that takes it | 2 models; `model_grace` is skipped (see below) |
| `<yaml>.dense` | `dense_nbr: true` on the equivariant single-particle basis, evaluated in the dense bond layout | 3 models |
| `preset.LINEAR`, `preset.FS`, `preset.GRACE_1LAYER_v2_25`, `preset.GRACE_2LAYER_v2_25` | small versions (settings in `PRESET_SETTINGS`) with the elements Mo, Nb, Ta, W; float64 | 4 models |

Cases: `s0`, `s1`, `s2` are Nb2, Mo7W9 and Ta26 of the test data. `isolated` is one Mo atom (the data builder adds one dummy bond beyond
the cutoff and the energy of the test models is zero); `dimer` is Mo-Nb at 2.6 A without a cell (the stress is recorded as zero, the virial is not);
`slab` is four Nb atoms with 20 A of vacuum, periodic in x and y (the data builder switches periodicity on in all directions: `enforce_pbc`);
`selfimage` is Mo-W in a 2.7 x 2.9 x 3.1 A cell, so that bonds join an atom to its own image (`i == j`, non-zero shift), with generic positions so that the forces do not cancel.
In all, 4,130 arrays: 15 models of 196 to 329 arrays each.

`lm_first` is skipped for `model_grace`: its output instruction `MLPOut2ScalarTarget` (`instructions/output.py`) reads `[:, :, 0]` without the
`lm_first` transpose, so the model fails with a `MatMul` shape error. The skip is stored in `UNSUPPORTED` and in the metadata, so it is visible, not silent.
The library is not changed by SAFE2.

**Evaluation detail.** The data builder edits its argument (`enforce_pbc` gives a non-periodic structure a cell and centres it), so the tool
evaluates a copy; the first snapshot's three structures are periodic and were not affected.

**Tolerance rows.** `compare` takes the tolerance from the named table `TOLERANCE_ROWS` of the tool, by the precision of the key (`.f32` suffix of the model
label: float32, else float64), relative to the largest element of each array: float64 `1e-12` (as before), float32 `0` (exact). `--scale-rtol` overrides both.

**Repeat spread.** `python tools/oracle_snapshot.py spread run1.npz ... run7.npz` reports, per precision and quantity, the largest range over the repeats divided by the
largest element of the first. Seven snapshots of the untouched tree (six started together, one alone; one thread each): every float32 quantity (52 groups,
including energies, forces, virials, stresses and all intermediate tensors) has a spread of exactly 0. The float64 spread is at most `1.9e-17` (`YI` of `large_base`, case
`selfimage`; 8 of 56 groups nonzero, all intermediate tensors of `large_base`), as in the first snapshot: energies, forces and stresses never moved. Hence the exact float32 row.

Record and compare (one thread; the file is large, so keep it outside the repository or in this git-ignored directory):

```bash
python tools/oracle_snapshot.py write new.npz              # about 75 s, 220 MB, also writes new.meta.json
python tools/oracle_snapshot.py compare baselines/oracle_snapshot_wide.npz new.npz
```

To record on the untouched tree: `git archive pre-cleanup | tar -x -C /scratch/pre-cleanup`, then run the tool with
`PYTHONPATH=/scratch/pre-cleanup` from a directory outside both trees.

## Pitfalls

- `tools/junit_outcomes.py summarize` needs `--log <pytest log>` with `-rX` lines to know an XPASS; without it `test_construct_batches_multiple_db` reads as `xpassed -> passed` in `compare` (not a change).
- `baselines/ast_manifest.json` is the untouched tree: to check that a branch changes only some files, `ast_manifest.py write --rev origin/torch-backend <file>` first and `check` against that.
- A `git archive` copy of a tag is imported with `PYTHONPATH` from a directory outside both trees; print `tensorpotential.__file__` to be sure which tree is imported.
- Checkpoint keys are not `variable.name`: a variable is saved as `model/instructions/<instruction>/<attribute path>/.ATTRIBUTES/VARIABLE_VALUE` (`A1/w_left_FC:0` is saved under `.../A1/w_left/...`); `tools/grace_probe.py` writes both (`probes/`).
  Each model also has two non-float variables (`Z/element_map_symbols`, a string, and `Z/element_map_index`, int32).
- With float32 parameters the radial basis, the bond vectors and the spherical harmonics still compute in float64 and `A`/`YI`/`R` cast down; with float64 parameters the swish `beta` is a float32 literal (value 1.0) cast up.
- Timings of the probe are quotable only when `timings_foreign_cpu_cores` in the JSON is near 0.
