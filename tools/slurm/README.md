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
- To test the scripts on a machine without a GPU: `DEVICE=cpu YAMLS=model_grace.yaml SUPERCELLS=1 REPEATS=1 ITERS=2 bash tools/slurm/gpu_bench.sh out`.
