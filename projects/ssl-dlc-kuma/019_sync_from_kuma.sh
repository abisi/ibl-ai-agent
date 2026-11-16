#!/bin/bash
# Copy the merged-training results needed by 019_training_summary.py from Kuma to the NAS
# (DLCProjects/merged_training_2026-11-09/kuma/): split + ground truth, learning_stats per shuffle, evaluation
# predictions per shuffle (*.h5 and the DLC scores), and the full-video pilots (023, 026: filtered h5, timing, GPU
# utilisation). Rerun whenever another architecture is evaluated or a pilot finishes.
# Run on Kuma: bash /work/lsens/bisi/dlc_merged/scripts/019_sync_from_kuma.sh
set -euo pipefail
module load rclone/1.65.2
P=/work/lsens/bisi/dlc_merged/Tracking2D-Axel-2026-11-04
DST=nas-sv:Petersen-Lab/analysis/Axel_Bisi/DLCProjects/merged_training_2026-11-09/kuma
rclone copyto /work/lsens/bisi/dlc_merged/analysis/split_ground_truth.h5 $DST/split_ground_truth.h5
for s in 1 2 3 4 5 6; do
  rclone copyto $P/dlc-models-pytorch/iteration-0/Tracking2DNov4-trainset90shuffle$s/train/learning_stats.csv \
    $DST/learning_stats_shuffle$s.csv
  E=$P/evaluation-results-pytorch/iteration-0/Tracking2DNov4-trainset90shuffle$s
  if [ -d "$E" ]; then rclone copy "$E" $DST/eval_shuffle$s --include "*.h5" --include "*.csv"; fi
done
for d in pilot_inference pilot_models; do
  [ -d /work/lsens/bisi/dlc_merged/$d ] && rclone copy /work/lsens/bisi/dlc_merged/$d $DST/$d --include "*_filtered.h5" --include "*_timing.json" --include "*_gpu.csv"
done
rclone ls $DST | wc -l
