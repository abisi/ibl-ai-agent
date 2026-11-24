#!/bin/bash
# Stage 2 of rebuild_chain.sh alone (stage 1 finished separately on 2026-11-17: ROC labels patched, AB156 + MH023 ROC rerun).
set -u
R=/mnt/lsens-analysis/Axel_Bisi; B=$R/combined_results_ks4; N=$R/NWB_ks4
W=$B/ssl-table-rebuild-20261117; LOG=$W/logs; BK=$W/pre_rebuild; mkdir -p $LOG $BK
STATUS=$LOG/status.txt
REPO=$HOME/code/ibl-ai-agent; EA=$REPO/projects/ssl-prelick-convergence/exploratory-analyses
PYU=$HOME/code/unit_spikes_analysis/.venv/bin/python
PYA=$REPO/.venv/bin/python
PYB=$HOME/anaconda3/envs/brain_wide_analysis/bin/python
EU=$R/Github/ephys_utilities                       # NAS checkout (allen_utils fixes, exclude_intervals)
log() { echo "$(date '+%F %T') $*" >> $STATUS; }
run() { local name=$1; shift; log "START $name"; "$@" > $LOG/$name.log 2>&1; local rc=$?; log "END   $name exit=$rc"; return $rc; }
log "STAGE 2 START (stage2.sh)"
# ---------------- stage 2
cd ~/code/unit_spikes_analysis
export PYTHONPATH=$EU:$HOME/code/NWB_reader:.
mkdir -p $BK/_cache $BK/_roc_stage_analysis $BK/tables $BK/across_days
cp -a $B/rastermap_variants/_cache/tables_all_days_v2.pkl $B/rastermap_variants/_cache/tables_all_days_v2_provenance.json $BK/_cache/
cp -a $B/_roc_stage_analysis/{units,roc_long,unit_info_all_days_v2}.parquet $B/_roc_stage_analysis/sessions_cohort_check.csv $BK/_roc_stage_analysis/
cp -a $B/ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units* $BK/tables/
for d in fa sl_stable; do cp -a $B/ssl-prelick-convergence/across_days/$d $BK/across_days/; done
log "backups done: $(du -sh $BK | cut -f1)"

run v2 $PYU $REPO/projects/ssl-prelick-convergence/data_prep/build_unit_table_all_days.py || exit 1
run 045 $PYU $EA/045_roc_stage_table.py || exit 1
( cd $REPO && run 137 $PYU projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/137_stable_units.py ) || exit 1
( PRELICK_REF=fa PRELICK_UNITS=all run 053_fa $PYU $EA/053_roc_prelick_table.py ) &
( PRELICK_REF=sl PRELICK_UNITS=stable run 053_sl_stable $PYU $EA/053_roc_prelick_table.py ) &
wait
export PRELICK_REF=sl PRELICK_UNITS=stable
run 080 $PYU $EA/080_single_unit_geometry.py --n-proc 60
run 081 $PYU $EA/081_single_unit_classes.py --n-proc 60 || exit 1
run 127 $PYU $EA/127_unified_single_neuron.py --n-proc 50 || exit 1
for pop in learners all; do
  ( for bl in shared raw orig; do run 128_${bl}_signed_$pop $PYU $EA/128_unified_figure.py --baseline $bl --measure signed --pop $pop; done
    run 128_shared_abs_$pop $PYU $EA/128_unified_figure.py --baseline shared --measure abs --pop $pop
    run 128_shared_signed_${pop}_ddiff $PYU $EA/128_unified_figure.py --baseline shared --measure signed --dmode diff --pop $pop ) &
  ( run 129_$pop $PYU $EA/129_cd_along_orthogonal.py --n-proc 40 --pop $pop ) &
  ( run 126_$pop $PYU $EA/126_jaw_control.py --n-proc 30 --pop $pop ) &
done
wait
run prelick_report $PYU $REPO/projects/ssl-prelick-convergence/report/build_report.py
log "CHAIN DONE"
