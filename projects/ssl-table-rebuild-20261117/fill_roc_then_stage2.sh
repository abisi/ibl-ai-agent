#!/bin/bash
# 2026-11-17: instead of rerunning ROC for the 4 realigned sessions (labels patched: patch_roc_labels.py,
# patch_roc_fix_unmatched.py), compute ROC + pre-lick ROC only where the new NWB has units without ROC rows (AB156
# 370 units, MH023 1 unit), in parallel; then wait for the datasets / derived tables (s2_done) and the end of
# rebuild_chain.sh (which stops before stage 2 because s1_done is removed), then run stage2.sh.
set -u
R=/mnt/lsens-analysis/Axel_Bisi; B=$R/combined_results_ks4; N=$R/NWB_ks4; W=$B/ssl-table-rebuild-20261117
LOG=$W/logs; STATUS=$LOG/status.txt
log() { echo "$(date '+%F %T') $*" >> $STATUS; }
PYU=$HOME/code/unit_spikes_analysis/.venv/bin/python
EA=$HOME/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses
export PYTHONPATH=$R/Github/ephys_utilities:$HOME/code/NWB_reader:$HOME/code/unit_spikes_analysis
rm -f $W/s1_done
for sd in AB156_20250206_121313:whisker_0 MH023_20250318_104610:whisker_2; do
  s=${sd%%:*}; day=${sd##*:}; m=${s%%_*}
  ( log "START roc_full_$s"; nice -n 5 $PYU $HOME/rebuild_20261106/001b_roc_session.py $N/$s.nwb $B/$m/$day/roc_analysis > $LOG/roc_full_$s.log 2>&1
    log "END   roc_full_$s exit=$?" ) &
done
wait
cd $EA
for ref in fa sl; do
  ( log "START prelick_roc_$ref"; PRELICK_REF=$ref $PYU 051_roc_prelick.py --sessions AB156_20250206_121313 MH023_20250318_104610 --n-proc 4 > $LOG/prelick_roc_$ref.log 2>&1
    log "END   prelick_roc_$ref exit=$?" ) &
done
wait
touch $W/s1_done
log "stage 1 complete (labels patched, AB156 + MH023 ROC recomputed)"
until [ -e $W/s2_done ] && ! pgrep -f "[r]ebuild_chain.sh" > /dev/null; do sleep 60; done
[ -e $W/s2_done ] || { log "s2 not done: stage 2 NOT started"; exit 1; }
exec bash $W/stage2.sh
