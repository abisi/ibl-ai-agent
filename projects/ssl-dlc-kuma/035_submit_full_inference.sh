#!/bin/bash
# Submit the full re-inference (user 2026-11-10): manifests from the NAS (032), one array per tier, longest first,
# at most 32 GPUs at once in total (S %18, M %10, L %4; change live with scontrol update ArrayTaskThrottle=<n> JobId=<id>),
# then 034 after all of them (and the MH007 side job, 4884034) finish.
set -eu
module load rclone/1.65.2
ROOT=/work/lsens/bisi/dlc_merged/inference_cspnext_m_330
S=/work/lsens/bisi/dlc_merged/scripts
mkdir -p $ROOT/manifests
rclone copy nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/inference_cspnext_m_330 $ROOT/manifests --include "manifest_*.tsv"
declare -A TIME=([S]=02:00:00 [M]=03:00:00 [L]=04:30:00 [XL]=06:00:00) THR=([S]=18 [M]=10 [L]=4 [XL]=2)
DEPS=""
for T in L M S XL; do
  F=$ROOT/manifests/manifest_$T.tsv
  N=$(($(wc -l < "$F") - 1))
  [ "$N" -le 0 ] && continue
  J=$(sbatch --parsable --array=0-$((N - 1))%${THR[$T]} --time=${TIME[$T]} --job-name=dlc_full_$T --export=ALL,TIER=$T $S/033_full_inference.sbatch)
  echo "tier $T: $N videos, job $J"
  DEPS="$DEPS:$J"
done
F=$(sbatch --parsable --job-name=dlc_full_finalize --partition=mig12gb --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=06:00:00 \
  --dependency=afterany$DEPS:4884034 --output=/work/lsens/bisi/dlc_merged/logs/%x_%j.out $S/034_finalize_inference.sh)
echo "finalize job $F (afterany$DEPS:4884034)"
