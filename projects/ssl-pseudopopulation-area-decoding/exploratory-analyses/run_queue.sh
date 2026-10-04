#!/bin/bash
# Queue (user 2026-09-30), one job at a time because of memory (009 alone ~385 GB):
#  1. pooled-cohort (R+ and R- together) stimulus decoding, modality_stim + modality_lick, whole brain + 14 areas,
#     100 iterations (as the per-cohort figures);
#  2. 009 window permutation, 500 permutations (resumes the saved rows);
#  3. 002 extension to 1000 iterations per cohort (run_1000.sh).
set -u
cd ~/code/ibl-ai-agent
P=projects/ssl-pseudopopulation-area-decoding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSL_PSEUDO_N_NEURONS=10 SSL_PSEUDO_N_MICE=20
AREAS="whole_brain,Motor areas,Frontal areas,Somatosensory-orofacial,Somatosensory-whisker,Auditory areas,Retrosplenial areas,Posterior parietal areas,Insular areas,Hippocampus,Striatum,Pallidum,Lateral septal complex,Thalamus,Midbrain"
echo "=== pooled start $(date)" >> $P/artifacts/queue_log.txt
SSL_PSEUDO_N_WORKERS=45 SSL_PSEUDO_CHUNK=20 .venv/bin/python -u $P/exploratory-analyses/002_pseudopop_decoding.py pooled "$AREAS" 100 modality_stim,modality_lick > $P/artifacts/002_pooled_log.txt 2>&1
echo "=== pooled done / 009 start $(date)" >> $P/artifacts/queue_log.txt
SSL_PERM_N_WORKERS=60 .venv/bin/python -u $P/exploratory-analyses/009_cohort_perm_windows.py 500 20 all >> $P/artifacts/009_perm_log.txt 2>&1
echo "=== 009 done / 1000 start $(date)" >> $P/artifacts/queue_log.txt
bash $P/exploratory-analyses/run_1000.sh
echo "=== queue done $(date)" >> $P/artifacts/queue_log.txt
