# rebuild figures + article sources for the given refs/pops: cR.sh "<refs>" "<pops>"
B=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4
cd ~/code/unit_spikes_analysis
S=~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses/062_pub_convergence_figures.py
for ref in $1; do d=$B/ssl-prelick-convergence/across_days/$ref
  for pop in $2; do
    echo "== $ref $pop"; PRELICK_REF=$ref PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python $S --population $pop 2>&1 | grep -v "Warning\|anova_lm\|wald_test\|warn" | tail -2
    ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses/build_article.py $d/publication/$pop $ref $pop
  done; done
echo RDONE
