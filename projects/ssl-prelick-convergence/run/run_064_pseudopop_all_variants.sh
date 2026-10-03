# Rerun 064 pseudo-population hierarchical bootstrap for all variants (SL first); single-threaded BLAS per worker.
cd ~/code/unit_spikes_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses/064_pseudopop_hbootstrap.py
for ref in sl fa; do for pop in all learners; do
  echo "== $ref $pop $(date)"; PRELICK_REF=$ref PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python $S --population $pop --n-proc 100 2>&1 | grep -v Warning | tail -3; echo "exit $?"
done; done
echo CHAIN DONE
