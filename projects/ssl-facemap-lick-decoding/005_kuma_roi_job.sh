#!/bin/bash
# Facemap lick decoding, ROI step 2 (Kuma): one task per ROI video (4 per session). Submitted by 005_kuma_roi_submit.sh.
# l40s limits: <= 8 cores per GPU, <= 5.9 GB per core.
#SBATCH --job-name=facemap_roif
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=2:00:00
#SBATCH --output=/work/lsens/bisi/facemap_roi_full/logs/facemap_%A_%a.out

set -euo pipefail
cd /work/lsens/bisi/facemap_roi_full
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
/work/lsens/bisi/facemap_pilot/venv_facemap/bin/python -u code/005_kuma_roi_facemap.py "$SLURM_ARRAY_TASK_ID"
