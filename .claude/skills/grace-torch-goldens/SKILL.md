---
name: grace-torch-goldens
description: Work with the numerical oracles of this repository - the untouched-tree baselines in `baselines/` (AST manifest, junit outcomes, TensorFlow numeric snapshot) and the golden fixtures the torch twins are compared with. Use when comparing a run with a baseline, adding or reading a golden, choosing a tolerance, debugging a mismatch between TF and torch, or when asked to regenerate a reference. Also use before touching anything that changes energies, forces or stress.
---

# grace-torch-goldens

The torch backend is measured against TensorFlow numbers that are committed or reproducible. Nothing is regenerated as a side effect of another change.

## What exists

`baselines/` is the untouched tree (tag `pre-cleanup` = master cc1bb38 = upstream 0.6.1) plus the later ratchets; `baselines/README.md` lists every file with the command that checks it and its pitfalls
(AST manifest, junit outcomes for pandas 2 and 3, the narrow and wide TensorFlow numeric snapshots, lint, coverage and clone ratchets, probes). The tools are in `tools/`.

Use the same `--ignore` list as the baseline (`tests/test_structured_grid.py`, `tests/test_foundation_model_regression.py`) when comparing counts: an unfiltered run
reports more skips. Pass the pytest log to `summarize`, otherwise an XPASS reads as a pass.

Golden fixtures for the twins (TF-generated, random weights, fp64 and fp32, every instruction output, index tables, energies, forces, virial, stress, a manifest of
versions and seeds) come in two tiers: tiny anchors under 1 MB committed in `tests_torch/golden/` (so CI needs no TF) and faithful larger ones generated
locally into `tests_torch/fixtures/` (the npz files git-ignored, manifests and yamls committed). The generator, `tools/make_golden.py`, is the only thing allowed to
rewrite either; `tests_torch/golden/README.md` has the key layout (`<case>/in|ins|out_before|out_after|res/...`, `tables/...`, weights keyed `<instruction>/<attribute path>`),
the option pairs (`*_dense`, `*_lm_first`: same weights, option on), the species relabelling and the sizes. A twin test reads them with numpy only
(`np.load(..., allow_pickle=False)`); `python tools/make_golden.py verify tests_torch/golden` reloads them in TF. Compare two runs of the generator (for example from
the `pre-cleanup` tag and from the head) with `compare`, which uses the rows of `tools/oracle_snapshot.py`.

## Tolerances

One table of named rows, used exactly as `numpy.isclose`: `|a - b| <= atol + rtol * |b|` with `b` the reference (an absolute floor is needed because force
components are often ~1e-16). Pick a row by name; never inline a number. **Tolerances are edit-locked**: changing one is a separate, justified, reviewed
change, never part of a feature change and never the fix for a failing test. The row for fp64 comparisons must not be tighter than the TF process-to-process
variation (about one ulp on some intermediates); fp32 rows come from the measured fp32-vs-fp64 spread of TF itself.

**Bit-exactness holds only within one process.** Across processes or against a committed file compare through the table or `--scale-rtol`.

## When a comparison fails

The default assumption is that your change is wrong.

1. Reproduce and read the actual numbers: a mismatch at 1e-16 and one at 1e-3 are different bugs.
2. Bisect with per-instruction outputs (the twins are tested in isolation with teacher forcing: each receives TF's recorded inputs), so a failure names one class.
3. Suspect shared global state and dtype before the physics: TF32 or default-dtype settings, a float32 constant promoting a float64 tensor, a layout option
   (`dense_nbr`, `lm_first`), the order of instructions (output instructions overwrite their target in place).
4. If the numbers really disagree and your code is right, you found a discrepancy in the oracle: record it as a finding on the issue. Do not encode a
   workaround and do not average two behaviours.

What is never a fix: widening a tolerance, regenerating a reference, marking the test xfail, or adding a cast that makes the numbers line up.

## Planted-bug check

A comparison is only trustworthy if it can fail. After writing one, plant a bug in a scratch copy (for example swap an index in the virial of `tpmodel.py`)
and confirm the comparison flags it in the expected quantity only.
