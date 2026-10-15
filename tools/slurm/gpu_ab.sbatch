#!/usr/bin/env bash
# Before/after GPU comparison of QUAL10 (tools/gpu_ab.py). Fill the <placeholders> for your site.
#
# One GPU, every step in turn (restartable: submit again to resume):
#   sbatch tools/slurm/gpu_ab.sbatch
# Several GPUs, one step per array task (48 steps by default; see `bash tools/slurm/gpu_ab.sh OUT --list`):
#   sbatch --array=0-47%4 tools/slurm/gpu_ab.sbatch      # at most 4 GPUs at a time
# then, once all tasks are done:  bash tools/slurm/gpu_ab.sh OUT --summary
#
#SBATCH --job-name=grace-gpu-ab
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<account>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=gpu_ab_%A_%a.log

set -eu
# module load <cuda-or-python modules of your site>   (TensorFlow brings its own CUDA with the `tf` extra)
source <path-to-venv>/bin/activate
cd <path-to-checkout>
# OUT is shared by every task of the comparison: keep it the same across resubmissions.
OUT="${OUT:-gpu_ab_results}"
BASE="${BASE:-df8d556}"
HEAD="${HEAD:-HEAD}"
# ARGS: further options of tools/gpu_ab.py, e.g.  --export=ALL,ARGS="--cell 3 --pairs 7"
# shellcheck disable=SC2086
bash tools/slurm/gpu_ab.sh "$OUT" --base "$BASE" --head "$HEAD" ${ARGS:-}
