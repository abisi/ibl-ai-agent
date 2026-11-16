#!/bin/bash
# Facemap lick decoding (Kuma): fetch the clips videos from the NAS, run Facemap (array, one task per session), send
# the small outputs back. Run on the Kuma login node once step 1 (haas) has finished, with the number of clips videos:
#   bash /work/lsens/bisi/facemap_full/code/002_kuma_submit.sh <n_videos>
# All Kuma jobs need a GPU; MIG slices allow <= 2 cores (used for the rclone jobs).
set -euo pipefail
N=$1
B=/work/lsens/bisi/facemap_full
NAS=nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-facemap-lick-decoding
cd $B

fetch=$(sbatch --parsable --job-name=fmf_fetch --partition=mig12gb --gres=gpu:1 --cpus-per-task=2 --mem=8G \
  --time=4:00:00 --output=$B/logs/fetch_%j.out --wrap "module load rclone/1.65.2 && rclone copy $NAS/clips $B/clips \
  --include '*_clips.mp4' --transfers 8 --log-level INFO && test \$(ls $B/clips/*_clips.mp4 | wc -l) -eq $N")
run=$(sbatch --parsable --dependency=afterok:$fetch --array=0-$((N - 1))%60 code/002_kuma_job.sh)
back=$(sbatch --parsable --dependency=afterany:$run --job-name=fmf_back --partition=mig12gb --gres=gpu:1 \
  --cpus-per-task=2 --mem=8G --time=4:00:00 --output=$B/logs/back_%j.out --wrap "module load rclone/1.65.2 && \
  rclone copy $B/results $NAS/facemap --include '/*_facemap_small.npz' --transfers 8 --log-level INFO && \
  rclone copy $B/logs $NAS/facemap/kuma_logs && touch $B/results/BACK_DONE && rclone copy $B/results $NAS/facemap \
  --include '/BACK_DONE'")
echo "fetch $fetch -> facemap array $run (0-$((N - 1))) -> copy back $back"
