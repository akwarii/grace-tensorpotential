---
name: grace-torch-tests
description: How to test changes in this repository - the 90% coverage rule for modified code, characterization tests, the two test layers (logic and physical values), planted mutants, parallel and fast test runs, and the guideline on mocks. Use before modifying any existing function or class, when writing or reviewing tests, when a test is slow or flaky, or when asked whether something is well tested.
---

# grace-torch-tests

## The coverage rule (modified code)

A function, method or class you are about to **modify** must already be covered at **90% or more**, branches included, by the existing suite. Unit =
a def or method (a class body counts when a class attribute changes). Coverage = (statements + branches executed) / (statements + branches).

1. Measure the unit on the unmodified code (branch coverage; `pytest --cov=tensorpotential --cov-branch --cov-report=json:cov.json -n 4`, then
   `python tools/check_touched_coverage.py --coverage cov.json --unit path/to/file.py::Class.method`; without `--unit` it checks every unit a diff against
   `--base` touches). A unit counts as touched when its AST differs after dropping comments, docstrings and annotations; a nested def is its own unit and
   module-level statements are not gated. A file that no test imports counts as 0%.
2. Below 90%: write **characterization tests first**, check that they pass on the code before the change (the `pre-cleanup` tag or the parent commit),
   commit them separately as `test:`.
3. Make the change. Re-run the check; it must report 90% or more for every touched unit.
4. Exempt: comment-only, docstring-only and annotation-only edits (AST identical), and test code itself.
5. Not modified: units the suite cannot execute here (multi-GPU, distributed, HPC-only), and units whose tests would cost more than half a day for a
   cosmetic fix. Report them on the issue instead.

New code in `core/` and `torch_backend/` is held to 90% (branch) per package in CI. `# pragma: no cover` needs a reason in the comment. Raising coverage
beyond the touched units is welcome; a file's coverage never falls.

## Two layers: logic and physics

Coverage shows that code ran, not that anything was checked. Every unit gets both layers where physics applies:

- **Logic layer**: each branch, each raised error (type and message), shapes and dtypes, empty and single-element inputs, option values, no mutation of arguments.
- **Physics layer**: values that hold whatever the implementation, from an oracle that neither calls nor shares code with the unit under test. A refactor may
  change the logic and the test must still pass; a change that breaks the physics must fail it.

| Family | Physical or value checks |
|---|---|
| Spherical harmonics | orthonormality by quadrature, parity `(-1)^l`, rotation covariance against scipy/sympy |
| Radial basis, cutoff | Chebyshev recurrence against `numpy.polynomial`; envelope is 1 at 0 and reaches 0 at `rc` smoothly |
| Clebsch-Gordan, plans | selection rules, orthogonality, sympy values, the committed Kokkos tables as a third source |
| Contractions | equivariance (rotate input, output rotates with the Wigner matrices), invariant scalars, neighbour permutation |
| Energy model | rotation, translation, permutation invariance; extensivity (two far copies give twice the energy); isolated atom energy |
| Forces | central finite difference of the energy; `sum(F) = 0`; no net torque for a cluster |
| Virial, stress | finite-strain derivative of the energy; symmetry; sign and units fixed by a hand case |
| Neighbour lists | completeness against a brute-force oracle; invariance under lattice translation; strict `d < rc` |
| Data pipeline | hand-computed numbers on toy frames; reference-energy subtraction conserves the total |
| I/O | npz and yaml round trips exact; unknown keys and shape mismatches raise |

Non-physical units (argument parsing, plumbing) need the logic layer only.

**Planted mutants.** For each new test file apply at least one logic mutant (a flipped comparison or branch) and one physics mutant (a wrong sign, factor,
index or dropped term) in a scratch copy: every one must make a test fail. A survivor means the test is rewritten, not the mutant dropped. An automated
mutation tool may be added to the `dev` group; say which and from where before downloading it.

## Mocks are a guideline, not a rule

Prefer real objects. A mock, stub or monkeypatch is acceptable to replace an external boundary (network, clock, absent hardware, a slow download) or
something that cannot run here, with a comment on what is replaced and why. Never for the unit under test or for a numeric or physical path. Keep a
count of mock uses before and after when you change a test file.

## Speed

- Run the suite from the repository root (`pytest tests`; it also passes from inside `tests/`). Parallel: `pytest -n 4 --dist load` (pytest-xdist, `dev` group): 10 to 12 min instead of 32 min on 14 cores with the identical outcome (`--dist worksteal` was not faster).
  Choose the worker count from memory (TF processes are large); the root `conftest.py` gives each worker `cores / N` threads before TensorFlow is first imported (`tests/thread_budget.py`; it must stay out of `tests/conftest.py`, where it would add E402 findings; overridable by setting `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` or `OMP_NUM_THREADS`).
- **Check that the machine is idle first** (rule in `CLAUDE.md`). `uptime` gives the load average; `ps -eo pid,etimes,args | grep '[p]ytest\|[g]racemaker'` lists test and training processes (the bracket keeps the grep itself out of the list). A shell whose own command line merely contains the word
  "pytest" is not a test run, so look at the executable (`python ... -m pytest`, or the xdist workers). `pgrep -f` matches the shell that runs it, so a wait loop on it never ends, and `pkill -f <script name>` kills your own shell: do not use either. Busy means another test or
  training process is running or the 1-minute load average is above 4 on 14 cores (the figure comes from a suite at `-n 4` taking about twice as long next to another one; adjust it if the machine changes). While it is busy, do the cheap work (edits, `ruff`, `ty`, a single test file in one process) and
  start the full run when the other has finished; if the wait blocks the task, say so. When a number depends on timing, note the load average before and after the run.
- Tests must be safe in parallel: `tmp_path` or a per-worker directory for every file written, no test reads another test's output, no fixed ports.
- **Check that nothing writes into the source tree** by running the suite on a read-only mount, not with `chmod -R a-w` (that makes `shutil.copy` and `copytree` copies read-only too, so tests that copy a fixture and edit it fail for the wrong reason):

  ```bash
  mkdir -p "$COPY"; git ls-files -co --exclude-standard -z | xargs -0 -I{} cp --parents {} "$COPY"/     # a copy of the tree, outside the repository
  mkdir -p "$MNT"; unshare -rm bash -c "mount --bind $COPY $MNT && mount -o remount,ro,bind $MNT && cd $MNT && \
      PATH=/path/to/.venv/bin:\$PATH PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -n 4 -p no:cacheprovider"
  ```

  `/tmp` stays writable, so `tmp_path` works; `PATH` must hold the venv's `bin` (the tests call `gracemaker` and `grace_preprocess`); the copy is imported, not the editable install. Do not run `uv sync` for this (see the worktree rule in `CLAUDE.md`).
- Wall times on this machine vary by up to 30% between identical runs (summed test time of the same tests: 1,392 s to 1,826 s), so compare timings only between runs made back to back with nothing else running, and say how many runs a number comes from. A test that only makes sense under xdist (it checks the worker environment) skips serially by design; mention that when comparing counts with a serial run.
- Share real expensive objects (built models, prepared data, trained tiny models) through module- or session-scoped fixtures when tests only read them,
  with a check that they are not mutated. Merge training runs that differ only in what is asserted afterwards, keeping every assertion. Reduce work
  only where the assertion does not depend on it, and list each cut; never relax a tolerance for speed.
- `@pytest.mark.slow` for tests of 30 s or more gives a fast loop (`-m "not slow"`); CI and gates run everything, and nothing is skipped by default.
- Compare a run with the baselines: same counts, same outcome per test id (`tools/junit_outcomes.py compare`).
