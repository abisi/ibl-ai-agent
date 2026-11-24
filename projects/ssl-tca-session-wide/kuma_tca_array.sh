#!/bin/bash
# Kuma SLURM array: one session per task, 006g_tca_gpu.py on one L40S GPU.
# Tensors in from the NAS (005 output), results back to the NAS (artifacts/tca_aligned_gpu/<sid>) with rclone.
# Submit from /work/lsens/bisi/tca_gpu: sbatch --array=0-<n-1> kuma_tca_array.sh   (sessions.txt: one session per line)
#SBATCH -p l40s
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 01:00:00
#SBATCH -o /work/lsens/bisi/tca_gpu/logs/tca_%A_%a.out
set -euo pipefail
cd /work/lsens/bisi/tca_gpu
module load rclone/1.65.2
NAS=nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-tca-session-wide/artifacts
SID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sessions.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${SID} on $(hostname)"
rclone copy "${NAS}/aligned/${SID}" "data/${SID}"
venv/bin/python 006g_tca_gpu.py --session "${SID}" --data-root data --out-root out "$@"
rclone copy "out/${SID}" "${NAS}/tca_aligned_gpu/${SID}"
echo "done ${SID}"
