#!/bin/bash
# 002 extension to 1000 iterations (user 2026-09-29), both cohorts in parallel, low priority (niced) behind the 009
# window permutation. Whole brain in its own process with few workers (each whole-brain task holds all units of a
# cohort in memory; 18 concurrent loads per cohort risked running haas out of memory). Resumes: only missing (area, rep).
set -u
cd ~/code/ibl-ai-agent
P=projects/ssl-pseudopopulation-area-decoding
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSL_PSEUDO_N_NEURONS=10 SSL_PSEUDO_N_MICE=20 SSL_PSEUDO_CHUNK=50
AREAS="Motor areas,Frontal areas,Somatosensory-orofacial,Somatosensory-whisker,Auditory areas,Retrosplenial areas,Posterior parietal areas,Insular areas,Hippocampus,Striatum,Pallidum,Lateral septal complex,Thalamus,Midbrain"
for c in R+ R-; do
  SSL_PSEUDO_N_WORKERS=26 nice -n 10 .venv/bin/python -u $P/exploratory-analyses/002_pseudopop_decoding.py "$c" "$AREAS" 1000 > $P/artifacts/002_iter1000_${c}_areas_log.txt 2>&1 &
  SSL_PSEUDO_N_WORKERS=3 nice -n 10 .venv/bin/python -u $P/exploratory-analyses/002_pseudopop_decoding.py "$c" whole_brain 1000 > $P/artifacts/002_iter1000_${c}_wholebrain_log.txt 2>&1 &
done
wait
echo "=== 1000 DONE $(date)" > $P/artifacts/002_iter1000_done.txt
