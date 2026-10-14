# GPU benchmark of the TensorFlow models (QUAL9)

The CPU profile of the TensorFlow side (see issue QUAL9) ranks the model classes by cost on a laptop-class CPU.
The same ranking on a GPU decides which classes are worth optimizing, so this sweep measures it on one
NVIDIA A100 (40 GB). Nothing here is a result: the sweep produces files that are read afterwards, and a number
from a CPU run must never be labelled a GPU number (every tool stops when TensorFlow sees no GPU).

## What it measures

| step | tool | answers |
|---|---|---|
| 1 | `tools/profile_instructions.py` | cost of each model class (forward, forward and backward) alone, and of the whole fused model, for the 3 test models, float64 and float32, `none` and `dense_nbr` bond layouts, 432 and 1,458 atoms |
| 2 | `tools/bench_cpu.py run --device gpu` | calculator start-up, 100 evaluations through a SavedModel, a tiny training run (first step = compilation), peak GPU memory |
| 3 | `gracemaker` on `tests/MoNbTaW-GRACE/input_1L.yaml` and `input_2L.yaml` | end-to-end phases (compile, iterations, final save) at batch sizes 10 and 25, from the log timestamps |

## Run it

1. Environment on the HPC (Linux; the `tf` extra brings TensorFlow 2.20 with its own CUDA libraries, no CUDA module is needed):

   ```bash
   python -m venv grace-venv && source grace-venv/bin/activate
   pip install -e '.[tf]'            # from the repository checkout, at the commit to be measured
   python -c "import tensorpotential, tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
   ```

   The last line must list one `GPU:0`. If it lists none, stop: the driver or the library path is wrong.

2. Submit one job on **one A100, exclusive node** (shared nodes make timings meaningless): copy
   `tools/slurm/gpu_bench.sbatch`, fill the `<placeholders>` (partition, account, paths, modules), then

   ```bash
   sbatch tools/slurm/gpu_bench.sbatch
   ```

   or, in an interactive allocation, `bash tools/slurm/gpu_bench.sh gpu_bench_out`.
   It takes about 1 to 2 hours. `SUPERCELLS="2"` and `REPEATS=5` shorten it; `ITERS` sets the `gracemaker` iterations.

3. Send back `gpu_bench_<job id>.tar.gz` (or the directory). It holds `environment.txt` (GPU name, driver,
   TensorFlow, commit), one JSON and one text table per step, `failures.txt` if a step failed, and the `gracemaker` runs with
   their `seed/42/log.txt`. Do not edit the files.

## The architecture comparison (`gpu_architectures.sh`)

The same comparison as the CPU one (FS, 1-, 2- and 3-layer foundation architectures from
`tests/data/model_yamls/`: which classes cost most, how time and memory grow), plus how far each model scales on 40 GB.

```bash
sbatch --export=ALL,SCRIPT=gpu_architectures.sh tools/slurm/gpu_bench.sbatch    # or: bash tools/slurm/gpu_architectures.sh OUT
```

Per architecture it runs (1) the per-class profile at 54 and 432 atoms in float64 and float32 with inference gradients
(forces and virial only) and once in float64 with training gradients, then (2) the whole model alone (time, peak GPU memory) at
supercells 1 to 8 (54 to 27,648 atoms), float64 and float32, default and `dense_nbr` layouts, each size in its own process:
the first size that fails (out of memory) stops the larger ones for that configuration and is listed in `failures.txt`; that
size is the answer to "how many atoms fit", not an error. About 2 to 4 hours. `ARCHS` (model names, or paths to other
`model.yaml` files), `CELLS` and `REPEATS` change the sweep. It ends with `summary.md` (one row per run: time, time per atom,
peak GPU memory and KB per bond, top classes); `python tools/summarise_profiles.py DIR` rebuilds it from the JSON files.

## Reading the `gracemaker` phases

```bash
grep -E "Start GRACEmaker|Start Adam|Iteration #(1|2|30)/|Loading best|Saving model" gpu_bench_*/gracemaker_input_2L_b10/seed/42/log.txt | cut -c1-80
```

The gaps are: data and model set-up, iteration 1 (tracing and compilation), a steady iteration, the final save.

## Notes

- Timings are medians of `REPEATS` calls after a compiling call and end with a copy to the host, so they include the transfer.
- `--option dense_nbr` changes the bond layout (padded per-atom neighbours); results equal the default layout to round-off.
- TensorFloat-32 is off in this code base (the log says so), so float32 runs are true float32.
- Peak GPU memory of the largest cases (1,458 atoms, float64, `large_base`) can approach the 40 GB; an out-of-memory step is recorded in
  `failures.txt` and is itself a result, not an error to fix.
- To test the scripts on a machine without a GPU: `DEVICE=cpu YAMLS=model_grace.yaml SUPERCELLS=1 REPEATS=1 ITERS=2 bash tools/slurm/gpu_bench.sh out` and `DEVICE=cpu CELLS=1 REPEATS=1 ARCHS=GRACE-FS-OMAT bash tools/slurm/gpu_architectures.sh out`.
