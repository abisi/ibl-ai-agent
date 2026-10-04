#!/bin/bash
# Full run (user 2026-09-29; 20 mice x 10 neurons per session): 002 for R+ then R-, all areas, 3 targets, then 003 figures/summaries (SD bands).
# 100 iterations x 10 shifts are PILOT counts; rerunning with a larger n_iterations resumes and only adds iterations.
set -u
cd ~/code/ibl-ai-agent
P=projects/ssl-pseudopopulation-area-decoding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSL_PSEUDO_N_WORKERS=60 SSL_PSEUDO_N_NEURONS=10 SSL_PSEUDO_N_MICE=20
for c in R+ R-; do
  echo "=== 002 $c start $(date)"
  .venv/bin/python -u $P/exploratory-analyses/002_pseudopop_decoding.py "$c" all 100
  echo "=== 002 $c done $(date)"
done
for c in R+ R-; do
  for tg in hitmiss modality_stim modality_lick; do
    .venv/bin/python $P/exploratory-analyses/003_summarize_pseudopop.py $tg "$c"
  done
done
echo "=== ALL DONE $(date)"
