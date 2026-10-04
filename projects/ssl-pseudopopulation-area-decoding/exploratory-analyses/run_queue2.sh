#!/bin/bash
# Queue v2 (2026-09-30, after the 009 out-of-memory crash): 009 window permutation (500) with whole brain in its own
# 3-worker process (own output file, SSL_PERM_OUT_TAG=_wholebrain) and the 14 area groups in a 45-worker process;
# then the 002 extension to 1000 iterations (run_1000.sh, resumes).
set -u
cd ~/code/ibl-ai-agent
P=projects/ssl-pseudopopulation-area-decoding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
AREAS="Motor areas,Frontal areas,Somatosensory-orofacial,Somatosensory-whisker,Auditory areas,Retrosplenial areas,Posterior parietal areas,Insular areas,Hippocampus,Striatum,Pallidum,Lateral septal complex,Thalamus,Midbrain"
echo "=== 009 v2 start $(date)" >> $P/artifacts/queue_log.txt
SSL_PERM_N_WORKERS=45 .venv/bin/python -u $P/exploratory-analyses/009_cohort_perm_windows.py 500 20 "$AREAS" > $P/artifacts/009_perm_areas_log.txt 2>&1 &
SSL_PERM_N_WORKERS=3 SSL_PERM_OUT_TAG=_wholebrain .venv/bin/python -u $P/exploratory-analyses/009_cohort_perm_windows.py 500 20 "All units" > $P/artifacts/009_perm_wholebrain_log.txt 2>&1 &
wait
echo "=== 009 v2 done / 1000 start $(date)" >> $P/artifacts/queue_log.txt
bash $P/exploratory-analyses/run_1000.sh
echo "=== queue2 done $(date)" >> $P/artifacts/queue_log.txt
