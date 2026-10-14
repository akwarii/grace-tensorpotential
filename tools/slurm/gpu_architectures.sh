#!/usr/bin/env bash
# The architecture comparison of QUAL9 (FS, 1-, 2- and 3-layer foundation models) on one GPU.
# Same measurements as the CPU comparison, plus how far each model scales in memory.
#
#   bash tools/slurm/gpu_architectures.sh [OUTPUT_DIR]
#
# Steps, per architecture:
#   1. per-class profile at 1 and 2 supercells (54 and 432 atoms): float64 and float32, inference gradients
#      (`--grads inputs`, forces and virial only), and float64 with `--grads all` (what training adds);
#   2. whole model only (time and peak GPU memory) for growing sizes, each size in its own process so that an
#      out-of-memory failure ends only that size: float64 and float32, default and dense_nbr bond layouts.
#      A size that fails stops the sizes above it for that configuration (recorded in failures.txt).
# Then the summary table (tools/summarise_profiles.py) is written to OUTPUT_DIR/summary.md.
#
# Environment: PYTHON (default python), REPEATS (default 5), CELLS (default "1 2 3 4 5 6 8": 54 to 27,648 atoms),
# DEVICE (default gpu; "cpu" only to test this script without a GPU, with CELLS="1" REPEATS=1 ARCHS=GRACE-FS-OMAT).
# ARCHS: model names of tests/data/model_yamls without .yaml (default the four architectures); a path to another
# model.yaml may be given instead.
set -u
cd "$(dirname "$0")/../.."
PYTHON="$(command -v "${PYTHON:-python}")" || { echo "python not found: set PYTHON" >&2; exit 2; }
REPEATS="${REPEATS:-5}"
CELLS="${CELLS:-1 2 3 4 5 6 8}"
DEVICE="${DEVICE:-gpu}"
ARCHS="${ARCHS:-GRACE-FS-OMAT GRACE-1L-OMAT-large-base GRACE-2L-OMAT-large-base GRACE-3L-OMAT-large}"
[ "$DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=-1
OUT="${1:-gpu_architectures_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

yaml_of() { [ -f "$1" ] && echo "$1" || echo "tests/data/model_yamls/$1.yaml"; }
run() {  # run NAME COMMAND...: log; return the command's status
  local name="$1"; shift
  echo "== $name" | tee -a "$OUT/steps.log"
  "$@" > "$OUT/$name.txt" 2> "$OUT/$name.err"
}

{
  echo "date: $(date -Is)"; echo "host: $(hostname)"
  echo "git: $(git rev-parse HEAD) $(git status --short | wc -l) modified file(s)"
  "$PYTHON" -c "import tensorflow as tf; print('tensorflow', tf.__version__); print('gpus', tf.config.list_physical_devices('GPU'))"
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"
if [ "$DEVICE" = gpu ] && ! grep -q "GPU:0" "$OUT/environment.txt"; then
  echo "TensorFlow sees no GPU: stop (nothing below would be a GPU measurement)" >&2
  exit 2
fi

for arch in $ARCHS; do
  yaml="$(yaml_of "$arch")"
  name="$(basename "$arch" .yaml)"
  # 1. per-class profiles
  for dtype in float64 float32; do
    for cell in 1 2; do
      run "classes_${name}_${dtype}_s${cell}" "$PYTHON" tools/profile_instructions.py "$yaml" --device "$DEVICE" \
        --dtype "$dtype" --supercell "$cell" --grads inputs --repeats "$REPEATS" --json "$OUT/classes_${name}_${dtype}_s${cell}.json" \
        || echo "classes_${name}_${dtype}_s${cell}" >> "$OUT/failures.txt"
    done
  done
  run "classes_${name}_float64_s1_training" "$PYTHON" tools/profile_instructions.py "$yaml" --device "$DEVICE" \
    --dtype float64 --supercell 1 --grads all --repeats "$REPEATS" --json "$OUT/classes_${name}_float64_s1_training.json" \
    || echo "classes_${name}_float64_s1_training" >> "$OUT/failures.txt"
  # 2. whole model, growing sizes
  for dtype in float64 float32; do
    for option in none dense_nbr; do
      for cell in $CELLS; do
        tag="whole_${name}_${dtype}_${option}_s${cell}"
        if ! run "$tag" "$PYTHON" tools/profile_instructions.py "$yaml" --device "$DEVICE" --dtype "$dtype" \
            --option "$option" --supercell "$cell" --whole-only --repeats "$REPEATS" --json "$OUT/$tag.json"; then
          echo "$tag (stopping larger sizes for ${name} ${dtype} ${option})" >> "$OUT/failures.txt"
          break
        fi
      done
    done
  done
done

"$PYTHON" tools/summarise_profiles.py "$OUT" --out "$OUT/summary.md" > /dev/null 2>> "$OUT/steps.log"
tar -czf "$OUT.tar.gz" "$OUT" && echo "wrote $OUT.tar.gz"
[ -f "$OUT/summary.md" ] && cat "$OUT/summary.md"
[ -f "$OUT/failures.txt" ] && { echo "failed or out-of-memory steps:"; cat "$OUT/failures.txt"; }
exit 0
