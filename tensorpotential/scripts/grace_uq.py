#!/usr/bin/env python3
"""``grace_uq`` CLI dispatcher.

Subcommands:
    build   - parallel GMM-UQ artifact generation
    info    - print a summary of an existing UQ .npz artifact
    predict - run E/F/S + per-atom gamma over a dataset
    select  - pick N structures from a candidate pool
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from tensorpotential.cli.console import print_ansi
from tensorpotential.scripts.grace import UQ_COMMANDS, dispatch


def _dispatch(argv: list[str], prog: str = "grace_uq") -> int:
    return dispatch(
        UQ_COMMANDS,
        argv,
        prog=prog,
        print_help=lambda: _print_top_help(prog),
    )


def _print_top_help(prog: str = "grace_uq"):
    b, c, g, r = "\033[1m", "\033[36m", "\033[32m", "\033[0m"  # bold, cyan, green, reset

    text = (
        f"""{b}grace_uq{r} <subcommand> [options]

{b}Subcommands:{r}
  {c}build{r}    Parallel GMM-UQ artifact generation (3 steps + optional SavedModel export).
  {c}info{r}     Summarize a UQ .npz artifact (D, K, elements, thresholds, diagnostics).
  {c}predict{r}  Run E/F/S + per-atom gamma over a dataset; multi-worker; pkl.gz/extxyz.
  {c}select{r}   Pick N structures from a candidate pool (random/fps; gamma window;
           per-element stratification).

Run `grace_uq <subcommand> --help` for subcommand-specific options.


{b}Examples{r}
--------

{b}1. Building UQ artifacts:{r}

 * Build UQ artifacts (default):
{g}    grace_uq build{r}

 * Build UQ artifacts from a training set:
{g}    grace_uq build --model-yaml model.yaml \\
                   --checkpoint checkpoints/checkpoint.best_test_loss.index \\
                   --train-data training_set.pkl.gz  \\
                   --n-workers 4{r}

 * Re-export a SavedModel from existing artifacts (no training data needed):
{g}    grace_uq build --model-yaml model.yaml \\
                   --checkpoint checkpoint \\
                   --artifact-path UQ/gmm_artifacts.npz{r}

{b}2. Inspecting UQ artifacts:{r}

 * Inspect an artifact:
{g}    grace_uq info{r}

{b}3. Predicting with uncertainties{r}

  * Predict on a candidate pool using the exported SavedModel (multi-worker):
{g}    grace_uq predict --model saved_model/ \\
                     --dataset candidates.pkl.gz \\

  * Predict using model.yaml + checkpoint + artifact (no SavedModel):
{g}    grace_uq predict --model model.yaml \\
                     --checkpoint checkpoint \\
                     --artifact UQ/gmm_artifacts.npz \\
                     --dataset candidates.pkl.gz \\
                     --save-features{r}

  * Predict and save as extxyz, dropping the ase_atoms column from pkl:
{g}    grace_uq predict --model saved_model/ \\
                     --dataset candidates.pkl.gz \\
                     --output predicted.xyz{r}

{b}4. Selecting structures{r}

  * Simplest case — point at a SavedModel + raw dataset; predict + select in one go:
{g}    grace_uq select --model saved_model/ \\
                    --dataset candidates.pkl.gz \\
                    -n 200{r}

  * Select 200 most-extrapolatory structures by Mini-batch FPS, gamma > 1.5:
{g}    grace_uq select --uq UQ/gmm_artifacts.npz \\
                    --predicted predicted.pkl.gz \\
                    -n 200 --strategy fps-extrap \\
                    --gamma-min 1.5 \\
                    --output selected.pkl.gz{r}

  * Random selection over the full pool, no stratification:
{g}    grace_uq select --uq UQ/gmm_artifacts.npz \\
                    --predicted predicted.pkl.gz \\
                    -n 100 --strategy random-all \\
                    --no-element-stratified \\
                    --output selected.pkl.gz{r}

  * Select from a raw (un-predicted) dataset — runs predict first, then selects:
{g}    grace_uq select --uq UQ/gmm_artifacts.npz \\
                    --dataset candidates.pkl.gz \\
                    --model saved_model/ \\
                    -n 200 --strategy fps-extrap \\
                    --n-workers 4 --gpus 0,1,2,3 \\
                    --output selected.pkl.gz{r}

  * Bound the gamma window (only mildly extrapolatory atoms):
{g}    grace_uq select --uq UQ/gmm_artifacts.npz \\
                    --predicted predicted.pkl.gz \\
                    -n 200 --strategy fps-extrap \\
                    --gamma-min 1.0 --gamma-max 3.0 \\
                    --output selected.pkl.gz{r}
"""
    )
    print_ansi(text.replace("grace_uq", prog))


def main(argv: Sequence[str] | None = None, *, prog: str = "grace_uq") -> int:
    return _dispatch(list(sys.argv[1:] if argv is None else argv), prog)


if __name__ == "__main__":
    raise SystemExit(main())
