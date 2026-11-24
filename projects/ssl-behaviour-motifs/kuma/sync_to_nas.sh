#!/bin/bash
# Kuma -> NAS: compact results only (state sequences, summaries, sweeps, frames, movies, configs, logs); no checkpoints,
# no lightning-action prediction CSVs / runs (the *_probs.npz hold the predictions)
module load rclone/1.65.2 2>/dev/null
cd /work/lsens/bisi/behaviour_motifs
R=nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-behaviour-motifs/kuma
rclone copy results $R --transfers 8 \
  --exclude "**/pred/**" --exclude "**/run/**" --exclude "**/checkpoint.h5" --exclude "video/clips/**" \
  --include "*.npz" --include "*.json" --include "f1_heldout*.csv" --include "sweep.csv" --include "*.png" --include "*.mp4" \
  --include "config.yml" --include "config.yaml" --exclude "*"
rclone copy logs $R/logs --include "*.log"
rclone copy scripts $R/scripts_kuma_copy
echo synced
