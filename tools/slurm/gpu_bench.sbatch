#!/usr/bin/env bash
# One A100 (40 GB) for the GPU benchmark sweep. Fill the <placeholders> for your site, then:  sbatch tools/slurm/gpu_bench.sbatch
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
bash tools/slurm/gpu_bench.sh "gpu_bench_${SLURM_JOB_ID}"
