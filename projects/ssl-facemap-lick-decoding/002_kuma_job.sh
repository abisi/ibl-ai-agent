#!/bin/bash
# Facemap lick decoding (Kuma): one task per session clips video. Submit with the array size, e.g.
#   sbatch --array=0-240%60 002_kuma_job.sh
# l40s limits: <= 8 cores per GPU, <= 5.9 GB per core.
#SBATCH --job-name=facemap_full
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=4:00:00
#SBATCH --output=/work/lsens/bisi/facemap_full/logs/facemap_%A_%a.out

set -euo pipefail
cd /work/lsens/bisi/facemap_full
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
/work/lsens/bisi/facemap_pilot/venv_facemap/bin/python -u code/002_kuma_facemap.py "$SLURM_ARRAY_TASK_ID"
