#!/usr/bin/env bash
# GPU benchmark sweep of the TensorFlow GRACE models (QUAL9). Run it on a node with one GPU, from a checkout
# whose environment has TensorFlow with CUDA (`pip install -e '.[tf]'`); see tools/slurm/README.md.
#
#   bash tools/slurm/gpu_bench.sh [OUTPUT_DIR]
#
# Environment: PYTHON (default python), SUPERCELLS (default "2 3": 432 and 1458 atoms), REPEATS (default 7),
# YAMLS (default the three test models), ITERS (gracemaker iterations, default 30),
# DEVICE (default gpu; "cpu" only to test this script on a machine without a GPU, with SUPERCELLS=1 REPEATS=1).
# A failing step is recorded in OUTPUT_DIR/failures.txt and the sweep goes on.
set -u
cd "$(dirname "$0")/../.."
PYTHON="$(command -v "${PYTHON:-python}")" || { echo "python not found: set PYTHON" >&2; exit 2; }
SUPERCELLS="${SUPERCELLS:-2 3}"
REPEATS="${REPEATS:-7}"
DEVICE="${DEVICE:-gpu}"
[ "$DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=-1
OUT="${1:-gpu_bench_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
YAMLS="${YAMLS:-model_grace.yaml model_grace_2L_omat.yaml model_grace_2L_omat_large_base.yaml}"
ITERS="${ITERS:-30}"

step() {  # step NAME COMMAND...: run, log, and keep going on failure
  local name="$1"; shift
  echo "== $name" | tee -a "$OUT/steps.log"
  if ! "$@" > "$OUT/$name.txt" 2> "$OUT/$name.err"; then
    echo "$name" >> "$OUT/failures.txt"
    echo "   FAILED (see $OUT/$name.err)" | tee -a "$OUT/steps.log"
  fi
}

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "git: $(git rev-parse HEAD) $(git status --short | wc -l) modified file(s)"
  "$PYTHON" -c "import tensorflow as tf; print('tensorflow', tf.__version__); print('gpus', tf.config.list_physical_devices('GPU'))"
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
  command -v nvidia-smi > /dev/null && nvidia-smi topo -m | head -5
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"
if [ "$DEVICE" = gpu ] && ! grep -q "GPU:0" "$OUT/environment.txt"; then
  echo "TensorFlow sees no GPU: stop (nothing below would be a GPU measurement)" >&2
  exit 2
fi

# 1. Per-class profile of each model: float64 and float32, both bond layouts, two sizes.
for dtype in float64 float32; do
  for option in none dense_nbr; do
    for cell in $SUPERCELLS; do
      step "instructions_${dtype}_${option}_s${cell}" \
        "$PYTHON" tools/profile_instructions.py $YAMLS --device "$DEVICE" --dtype "$dtype" --option "$option" \
        --supercell "$cell" --repeats "$REPEATS" --json "$OUT/instructions_${dtype}_${option}_s${cell}.json"
    done
  done
done

# 2. Start-up, a calculator on a SavedModel, a tiny training run (summary table with GPU peak memory).
step bench_gpu "$PYTHON" tools/bench_cpu.py run --device "$DEVICE" --allow-busy --repeats 5 --wait-idle 0 \
  --cases calc_init:model_grace.yaml calc_init:model_grace_2L_omat.yaml calc_eval train_step train_run \
  --out "$OUT/bench_gpu.json"

# 3. gracemaker end to end: 1- and 2-layer inputs of the tests, 30 iterations, two batch sizes, jit on.
for input in input_1L input_2L; do
  for batch in 10 25; do
    run="$OUT/gracemaker_${input}_b${batch}"
    mkdir -p "$run"
    sed -e "s#\.\./data/#$PWD/tests/data/#g" \
        -e 's/train_size: [0-9]*/train_size: 50/; s/test_size: [0-9]*/test_size: 50/' \
        -e "s/maxiter: [0-9]*/maxiter: $ITERS/; s/jit_compile: False/jit_compile: True/" \
        -e "s/^  batch_size: [0-9]*/  batch_size: $batch/; s/^  test_batch_size: [0-9]*/  test_batch_size: 50/" \
        "tests/MoNbTaW-GRACE/$input.yaml" > "$run/input.yaml"
    ( cd "$run" && step_start=$(date +%s) && "$PYTHON" -m tensorpotential.scripts.gracemaker input.yaml \
        > run.out 2> run.err; echo "wall_s $(( $(date +%s) - step_start ))" > wall.txt ) \
      || echo "gracemaker_${input}_b${batch}" >> "$OUT/failures.txt"
  done
done

tar -czf "$OUT.tar.gz" "$OUT" && echo "wrote $OUT.tar.gz"
[ -f "$OUT/failures.txt" ] && { echo "failed steps:"; cat "$OUT/failures.txt"; exit 1; }
exit 0
