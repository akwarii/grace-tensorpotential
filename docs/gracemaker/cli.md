# Command line

## The `grace` command

`grace <subcommand> [options]` is the one entry point of the command-line tools. Each sub-command runs the same code as
the console script of the same purpose, with the remaining arguments; the old names (`gracemaker`, `grace_utils`, ...) are
still installed and print the same output.

| `grace` sub-command | Old command | What it does |
| :--- | :--- | :--- |
| `grace train` | `gracemaker` | fit a potential from an `input.yaml` ([input file](inputfile.md), [quick start](quickstart.md)) |
| `grace preprocess` | `grace_preprocess` | precompute a dataset for fitting ([utilities](utilities.md#grace_preprocess)) |
| `grace predict` | `grace_predict` | predict energies, forces and stresses of a dataset ([utilities](utilities.md#grace_predict)) |
| `grace utils` | `grace_utils` | convert, reduce, summarize and export models ([utilities](utilities.md#grace_utils)) |
| `grace models` | `grace_models` | list, inspect and download foundation models ([utilities](utilities.md#grace_models)) |
| `grace uq` | `grace_uq` | uncertainty quantification: `build`, `info`, `predict`, `select` ([UQ](uq.md)) |
| `grace collect` | `grace_collect` | collect VASP calculations into a dataframe ([utilities](utilities.md#grace_collect)) |
| `grace extxyz2df` | `extxyz2df` | convert an extxyz file into a dataframe ([utilities](utilities.md#extxyz2df)) |
| `grace df2extxyz` | `df2extxyz` | convert a dataframe into an extxyz file ([utilities](utilities.md#df2extxyz)) |
| `grace dashboard` | `grace_dashboard` | browse the results of fits in a web dashboard ([utilities](utilities.md#grace_dashboard)) |

```bash
grace                          # list the sub-commands
grace train --help             # options of a sub-command
grace uq build --help          # a sub-command of a sub-command
grace utils reduce_elements --help
grace models list
```

The usage lines read `grace uq build` when the command is started through `grace` and `grace_uq build` through the old
name; options, defaults, output and exit codes are the same. Nothing is imported until a sub-command runs, so `grace`,
`grace --help` and an unknown sub-command answer at once, also in an environment without TensorFlow. A sub-command that
needs TensorFlow (`train`, `preprocess`, `predict`, `utils`, `extxyz2df` and the `uq` sub-commands) says so, with the
`pip install "tensorpotential[tf]"` command, before it imports anything.

Output is plain text when it is not a terminal (a pipe, a file, a SLURM log) and styled with
[rich](https://rich.readthedocs.io/) on a terminal; set `NO_COLOR` to turn the colour off.

!!! note "Another program called `grace`"
    The Grace 2D plotting program installs an executable of the same name. Inside the environment of this package the
    command of the package is the one that runs when the `bin` directory of the environment comes first on `PATH`;
    otherwise use the old names, which are unique.

## `grace train` (`gracemaker`)

```
usage: grace train [-h] [-l LOG] [-m] [-rl] [-r] [-rs RESTART_SUFFIX] [--no-intra-epoch-redo]
                   [-p POTENTIAL] [-s] [-sf] [-e] [-nj] [--seed SEED] [-cm] [-t]
                   [-cn CHECKPOINT_NAME] [--reset-epoch-and-step]
                   [input]

Fitting utility for (graph) atomic cluster expansion potentials.

positional arguments:
  input                 input YAML file, default: input.yaml

options:
  -h, --help            show this help message and exit
  -l LOG, --log LOG     log filename, default: log.txt
  -m, --multigpu        Single host/multi GPU distributed fit
  -rl, --restart-latest
                        Restart from latest checkpoint (use separately from -r/-rs)
  -r, --restart-best-test
                        Restart from latest best test checkpoint (use separately from -rs/-rl)
  -rs RESTART_SUFFIX, --restart-suffix RESTART_SUFFIX
                        Suffix of checkpoint to restart from, i.e. .epoch_10 (use separately from
                        -r/-rl)
  --no-intra-epoch-redo
                        When resuming from a mid-epoch checkpoint (intra_epoch_save=True), skip
                        the interrupted epoch's remaining batches entirely and bump the step
                        counter to the nominal end of that epoch. This keeps the LR scheduler
                        aligned with the original maxiter budget at the cost of under-training by
                        (nominal_end - saved_step) gradient updates. Default is to fast-forward
                        the iterator and train only the unseen tail, which is usually cheaper and
                        preserves full training amount.
  -p POTENTIAL, --potential POTENTIAL
                        Potential configuration to load, model.yaml file
  -s, --save-model      Export model as TF saved model
  -sf, --save--fs       Export FS model as yaml to be loaded in CPP
  -e, --eager           Eager graph execution
  -nj, --no-jit         No JIT
  --seed SEED           Random seed (will overwrite value from input.yaml)
  -cm, --check-model    Check model consistency, without performing fit
  -t, --template        Generate a template 'input.yaml' file by dialog
  -cn CHECKPOINT_NAME, --checkpoint-name CHECKPOINT_NAME
                        Explicit name of the checkpoint (omit .index suffix)
  --reset-epoch-and-step
                        Reset epoch and step counters from prev. runs
```
