#!/bin/bash
# Facemap lick decoding, ROI step 2 (Kuma): fetch the ROI videos, run Facemap (array, <= 80 at once), copy the small
# outputs back. Run on the Kuma login node with the number of ROI videos:  bash code/005_kuma_roi_submit.sh <n_videos>
set -euo pipefail
N=$1
B=/work/lsens/bisi/facemap_roi_full
NAS=nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-facemap-lick-decoding/roi
cd $B
fetch=$(sbatch --parsable --job-name=fmrf_fetch --partition=mig12gb --gres=gpu:1 --cpus-per-task=2 --mem=8G \
  --time=2:00:00 --output=$B/logs/fetch_%j.out --wrap "module load rclone/1.65.2 && rclone copy $NAS/videos $B/videos \
  --include '*.mp4' --exclude '*.tmp.mp4' --transfers 16 && test \$(ls $B/videos/*.mp4 | wc -l) -eq $N")
run=$(sbatch --parsable --dependency=afterok:$fetch --array=0-$((N - 1))%80 code/005_kuma_roi_job.sh)
back=$(sbatch --parsable --dependency=afterany:$run --job-name=fmrf_back --partition=mig12gb --gres=gpu:1 \
  --cpus-per-task=2 --mem=8G --time=2:00:00 --output=$B/logs/back_%j.out --wrap "module load rclone/1.65.2 && \
  rclone copy $B/results $NAS/facemap --include '/*_facemap_small.npz' --transfers 16 && \
  touch $B/results/BACK_DONE && rclone copy $B/results $NAS/facemap --include '/BACK_DONE'")
echo "fetch $fetch -> facemap array $run (0-$((N - 1))) -> copy back $back"
