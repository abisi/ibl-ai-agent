#!/bin/bash
# Regenerate every figure / table in DLCProjects/merged_training_2026-11-09 (user 2026-11-10): training summary (019),
# unlabelled confident crops (020), unseen-mice and general benchmarks (021, 022), pilot vs old (024), before / after
# (025), the model choice (027) and the 5 failed top views (030). Run on haas after 019_sync_from_kuma.sh (Kuma): bash 028_refresh_all.sh
cd "$(dirname "$0")"
PY=~/code/ibl-ai-agent/.venv/bin/python
for s in 019_training_summary 020_unlabelled_confident 021_unseen_mice_benchmark 022_general_benchmark \
         024_pilot_vs_old 025_pilot_before_after 027_model_choice 030_failed_top_before_after; do
  nice -n 10 $PY $s.py > /tmp/bisi_$s.log 2>&1
  echo "$s exit $?"
done
