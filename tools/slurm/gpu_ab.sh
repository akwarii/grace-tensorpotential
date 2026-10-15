#!/usr/bin/env bash
# Before/after comparison of two commits on one GPU (QUAL10): tools/gpu_ab.py with the site's Python.
#
#   bash tools/slurm/gpu_ab.sh OUT [--base REF] [--head REF] [--list | --step NAME ... | --summary]
#
# Restartable: run the same command again after an interruption; finished runs are kept in OUT.
# In a Slurm array job (SLURM_ARRAY_TASK_ID set, no --step given) the task runs the step whose
# 0-based position in `--list` is its task id, so N tasks on N GPUs run N steps at the same time;
# `bash tools/slurm/gpu_ab.sh OUT --list | wc -l` gives the number of steps for --array=0-(N-1).
#
# Environment: PYTHON (default python).
set -eu
cd "$(dirname "$0")/../.."
PYTHON="$(command -v "${PYTHON:-python}")" || { echo "python not found: set PYTHON" >&2; exit 2; }
[ $# -ge 1 ] || { echo "usage: bash tools/slurm/gpu_ab.sh OUT [options]" >&2; exit 2; }
OUT="$1"; shift

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ " $* " != *" --step "* ]] && [[ " $* " != *" --list "* ]] && [[ " $* " != *" --summary "* ]]; then
  STEP="$("$PYTHON" tools/gpu_ab.py "$OUT" "$@" --list | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | cut -f1)"
  [ -n "$STEP" ] || { echo "no step at position ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
  exec "$PYTHON" tools/gpu_ab.py "$OUT" "$@" --step "$STEP"
fi
exec "$PYTHON" tools/gpu_ab.py "$OUT" "$@"
