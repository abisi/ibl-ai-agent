#!/bin/bash
# Run 01_screen_probe.py on every inventory row, ~96 parallel workers (haas, nice 10). Logs in <OUT>/logs/.
OUT=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/screen
P=/home/bisi/anaconda3/envs/brain_wide_analysis/bin/python
cd "$(dirname "$0")"
N=$(($(wc -l < $OUT/tables/00_inventory.csv) - 2))
seq 0 $N | OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P ${WORKERS:-96} -I{} sh -c "nice -n 10 $P 01_screen_probe.py {} > $OUT/logs/probe_{}.log 2>&1 || echo FAIL {}"
echo done > $OUT/logs/done
