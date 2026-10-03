# Rebuild 062 figures (Fig1-5, supplements, COSYNE, captions, stats) for FA/SL x all/learners on haas.
cd ~/code/unit_spikes_analysis
S=~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses/062_pub_convergence_figures.py
for ref in fa sl; do for pop in all learners; do
  echo "== $ref $pop"; PRELICK_REF=$ref PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python $S --population $pop 2>&1 | grep -v Warning | tail -5
done; done
echo CHAIN DONE
