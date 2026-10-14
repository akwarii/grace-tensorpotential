#!/usr/bin/env bash
# One A100 (40 GB) for a GPU benchmark sweep (gpu_bench.sh or gpu_architectures.sh). Fill the <placeholders> for your site, then:  sbatch tools/slurm/gpu_bench.sbatch
#SBATCH --job-name=grace-gpu-bench
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<account>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=gpu_bench_%j.log

set -eu
# module load <cuda-or-python modules of your site>   (TensorFlow brings its own CUDA with the `tf` extra)
source <path-to-venv>/bin/activate
cd <path-to-checkout>
# SCRIPT: gpu_bench.sh (default) or gpu_architectures.sh; choose with  sbatch --export=ALL,SCRIPT=gpu_architectures.sh ...
SCRIPT="${SCRIPT:-gpu_bench.sh}"
bash "tools/slurm/${SCRIPT}" "${SCRIPT%.sh}_${SLURM_JOB_ID}"
