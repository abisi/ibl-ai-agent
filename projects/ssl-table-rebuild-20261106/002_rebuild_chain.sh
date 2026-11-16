#!/bin/bash
# 002 -- Rebuild of the shared SSL tables and the pre-lick chain from the current NWB_ks4 (user 2026-11-06), on haas.
#   wait 001 (per-session reruns) -> back up the outputs -> all-days unit cache v2 -> 045 ROC stage table
#   -> wait for the new ssl_ephys / derived tables on haas (003, flag file) -> 137 stable units
#   -> 053 (fa all, sl all, sl stable) -> 080 -> 081 -> 127 -> 128 (variants) -> 129 -> 126 -> report build_report.py
# ephys_utilities: the NAS checkout (PYTHONPATH), not ~/code/ephys_utilities (stale clone: 5071dbb + local edits, returns the
# uncorrected spike times from combine_ephys_nwb, no exclude_intervals).
# Run (haas): nohup bash 002_rebuild_chain.sh > 002.log 2>&1 &     status: $LOG/status.txt
set -u
B=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4
BK=$B/ssl-table-rebuild-20261106/pre_rebuild
LOG=$B/ssl-table-rebuild-20261106/logs; mkdir -p $LOG $BK
STATUS=$LOG/status.txt
FLAG=$HOME/rebuild_20261106/derived_ready.flag
REPO=$HOME/code/ibl-ai-agent
EA=$REPO/projects/ssl-prelick-convergence/exploratory-analyses
cd ~/code/unit_spikes_analysis
PY=./.venv/bin/python
export PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:$HOME/code/NWB_reader:.

run() {  # run <name> <script> [args...]
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  $PY "$@" > $LOG/$name.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name exit=$rc" >> $STATUS
  return $rc
}

echo "$(date '+%F %T') CHAIN START" >> $STATUS
until grep -q "TIER0 FINISHED" $HOME/rebuild_20261106/001.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') 001 finished" >> $STATUS
for f in AB155/whisker_0/roc_analysis/AB155 MH038/whisker_0/roc_analysis/MH038 MH068/whisker_3/roc_analysis/MH068          MH069/whisker_3/roc_analysis/MH069 MH064/whisker_3/roc_analysis/MH064 MH065/whisker_2/roc_analysis/MH065; do
  for t in _roc_results_new.csv _roc_prelick_results.csv _roc_prelick_sl_results.csv; do
    [ $B/$f$t -nt $HOME/rebuild_20261106/001b_roc_session.py ] || { echo "$(date '+%F %T') MISSING new $f$t: stop" >> $STATUS; exit 1; }
  done
done

# backups (single files: rename within the share; result folders: copy)
mkdir -p $BK/_cache $BK/_roc_stage_analysis $BK/tables $BK/across_days
mv $B/rastermap_variants/_cache/tables_all_days_v2.pkl $B/rastermap_variants/_cache/tables_all_days_v2_provenance.json $BK/_cache/
cp -a $B/_roc_stage_analysis/{units,roc_long,unit_info_all_days_v2}.parquet $B/_roc_stage_analysis/sessions_cohort_check.csv $BK/_roc_stage_analysis/
cp -a $B/ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units* $BK/tables/
for d in fa sl sl_stable; do cp -a $B/ssl-prelick-convergence/across_days/$d $BK/across_days/; done
echo "$(date '+%F %T') backups done: $(du -sh $BK | cut -f1)" >> $STATUS

run v2 $REPO/projects/ssl-prelick-convergence/data_prep/build_unit_table_all_days.py || exit 1
run 045 $EA/045_roc_stage_table.py || exit 1

until [ -e $FLAG ]; do sleep 120; done
echo "$(date '+%F %T') new ssl_ephys / derived tables on haas" >> $STATUS
( cd $REPO && $PY projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/137_stable_units.py > $LOG/137.log 2>&1 )
echo "$(date '+%F %T') END   137 exit=$?" >> $STATUS

PRELICK_REF=fa PRELICK_UNITS=all run 053_fa $EA/053_roc_prelick_table.py
# 053_sl (all units) writes across_days/sl = the 2026-10-07 record (RECORD.md, do not overwrite): 2026-11-07 its 4 new
# files were moved to across_days/sl_rebuild_20261106_all/ and the record restored from $BK. Do not rerun this line.
PRELICK_REF=sl PRELICK_UNITS=all run 053_sl $EA/053_roc_prelick_table.py
export PRELICK_REF=sl PRELICK_UNITS=stable
run 053_sl_stable $EA/053_roc_prelick_table.py || exit 1
run 080 $EA/080_single_unit_geometry.py --n-proc 40
run 081 $EA/081_single_unit_classes.py --n-proc 40 || exit 1
run 127 $EA/127_unified_single_neuron.py --n-proc 25 || exit 1
for pop in learners all; do
  for bl in shared raw orig; do run 128_${bl}_signed_$pop $EA/128_unified_figure.py --baseline $bl --measure signed --pop $pop; done
  run 128_shared_abs_$pop $EA/128_unified_figure.py --baseline shared --measure abs --pop $pop
  run 128_shared_signed_${pop}_ddiff $EA/128_unified_figure.py --baseline shared --measure signed --dmode diff --pop $pop
  run 129_$pop $EA/129_cd_along_orthogonal.py --n-proc 25 --pop $pop
  run 126_$pop $EA/126_jaw_control.py --n-proc 20 --pop $pop
done
run report $REPO/projects/ssl-prelick-convergence/report/build_report.py
echo "$(date '+%F %T') CHAIN DONE" >> $STATUS
