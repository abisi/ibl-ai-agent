set -u
cd ~/code/ibl-ai-agent; H=projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSL_DECODE_N_WORKERS=4
for spec in "modality lick whole_brain" "modality lick area_group" "hitmiss stim whole_brain" "modality lick area_acronym_custom" "hitmiss stim area_acronym_custom"; do
  set -- $spec
  tag="${1}_${2}_${3}"
  echo "=== $tag start $(date)" >> $H/024d_fillin_log.txt
  SSL_FILLIN_TARGETS=$H/024_fillin_stalenull_${tag}.parquet .venv/bin/python -u $H/024_master_sweep.py $1 $2 learning $3 >> $H/024d_fillin_log.txt 2>&1
  echo "=== $tag done $(date) exit $?" >> $H/024d_fillin_log.txt
done
echo "=== ALL DONE $(date)" >> $H/024d_fillin_log.txt
