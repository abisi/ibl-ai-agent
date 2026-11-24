#!/bin/bash
# Rebuild after the 2026-11-18 changes (all on haas; template: projects/ssl-table-rebuild-20261117).
#   Target labels: MH034_20250514_104756 and MH034_20250516_102648 imec3 PPC -> wS1 (sheet joint_probe_insertion_info,
#   PPC-near-wS1 review projects/ssl-ppc-target-review). Needs the two NWBs reconverted (target stored in the NWB).
#   allen_utils 7ae712d (pushed 2026-11-18): SSp-un -> SSp-body (was SSp-oro); handle_ppc before the oro/body handlers,
#   so SSp-tr / SSp-un on PPC-targeted probes -> PPC (review projects/ssl-ssp-un-review).
#   No spike / unit content changes: no ROC or pre-lick ROC rerun (ROC is per unit, labels joined downstream).
# Step 0: wait until no barrelreg job runs (another session's wS1 registration tests; the 11-17 parallel run exhausted
#   haas memory). Step 1: reconvert + check + swap the 2 NWBs (reconvert_targets_20261118.py); stop unless both swapped.
# Stage 1: S2 ssl_ephys + ssl_behavior datasets (NAS output, haas symlinks swapped) then the derived tables | S3
#   data_summary unit-table caches (3 populations, --reload). Stage 2: all-days unit cache v2 -> 045 ROC stage table ->
#   137 stable units -> 053 (fa all, sl stable) -> 080 -> 081 -> 127 -> 128 variants -> 129 -> 126 -> prelick report.
# Run (haas): mkdir -p ~/rebuild_20261118 && cp <W>/rebuild_chain.sh <W>/reconvert_targets_20261118.py ~/rebuild_20261118/
#   nohup setsid bash ~/rebuild_20261118/rebuild_chain.sh > ~/rebuild_20261118/chain.log 2>&1 < /dev/null &
#   status: $LOG/status.txt
set -u
R=/mnt/lsens-analysis/Axel_Bisi; B=$R/combined_results_ks4; N=$R/NWB_ks4
W=$B/ssl-table-rebuild-20261118; LOG=$W/logs; BK=$W/pre_rebuild; mkdir -p $LOG $BK
STATUS=$LOG/status.txt
REPO=$HOME/code/ibl-ai-agent; EA=$REPO/projects/ssl-prelick-convergence/exploratory-analyses
PYU=$HOME/code/unit_spikes_analysis/.venv/bin/python
PYA=$REPO/.venv/bin/python
PYB=$HOME/anaconda3/envs/brain_wide_analysis/bin/python
EU=$R/Github/ephys_utilities                       # NAS checkout (7ae712d + uncommitted exclude_intervals)
log() { echo "$(date '+%F %T') $*" >> $STATUS; }
run() { local name=$1; shift; log "START $name"; "$@" > $LOG/$name.log 2>&1; local rc=$?; log "END   $name exit=$rc"; return $rc; }
rm -f $W/s2_done
log "CHAIN START"

# ---------------- step 0: wait for the barrel-registration jobs
while pgrep -f "[b]arrelreg" > /dev/null; do sleep 120; done
log "no barrelreg job running"

# ---------------- step 1: reconvert the two MH034 NWBs
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
for s in MH034_20250514_104756 MH034_20250516_102648; do
  ( cd ~ && ~/code/nwb_converter_env/bin/python $HOME/rebuild_20261118/reconvert_targets_20261118.py $s > $LOG/reconv_$s.log 2>&1 ) &
done
wait
unset NPY_DISABLE_CPU_FEATURES
grep -h "SWAPPED IN\|NOT swapped\|FAILED\|target old" $LOG/reconv_*.log >> $STATUS
n=$(grep -l "SWAPPED IN" $LOG/reconv_*.log | wc -l)
[ "$n" = 2 ] || { log "reconversion: $n / 2 swapped in, rebuild NOT started"; exit 1; }
log "reconversion: 2 / 2 swapped in"

# ---------------- stage 1
s2() {  # datasets on the NAS, haas repo symlinks swapped, then the derived tables
  cd $REPO
  ( run ds_ephys $REPO/.venv/bin/ibl-ai-agent build-ssl-ephys-dataset --nwb-root $N --output-root $W/datasets ) &
  ( run ds_behavior $REPO/.venv/bin/ibl-ai-agent build-ssl-behavior-dataset --nwb-root $N --output-root $W/datasets ) &
  wait
  for d in ssl_ephys ssl_behavior; do
    [ -d $W/datasets/$d/1.0.0 ] || { log "dataset $d missing: stop s2"; return 1; }
  done
  for d in ssl_ephys ssl_behavior; do
    L=$REPO/reports/datasets/$d/1.0.0
    [ -L $L ] && mv $L $REPO/reports/datasets/$d/1.0.0_pre_20261118      # symlink to the 20261117 NAS copy
    ln -s $W/datasets/$d/1.0.0 $L
  done
  log "datasets swapped in (old symlinks: reports/datasets/*/1.0.0_pre_20261118)"
  [ -d $BK/derived ] || cp -r --no-preserve=all $REPO/reports/ssl_analysis/derived $BK/derived
  for s in extract_ssl_target_region extract_ssl_mouse_reference compute_ssl_unit_coverage compute_ssl_quality_label \
           build_ssl_area_labels build_ssl_analysis_units compute_ssl_trial_window_rates build_ssl_lmm_table \
           compute_ssl_modulation_index compute_ssl_psth; do
    run derived_$s $PYA scripts/$s.py || { log "derived $s FAILED: stop s2"; return 1; }
  done
  touch $W/s2_done
}
s3() {  # data_summary unit-table caches
  cd $REPO/projects/ssl-data-summary
  for p in day0_learners all_days_no_day0_nonlearners all_mice; do
    PYTHONPATH=$EU:$R/NWB_reader:$R/brain_wide_analysis run data_summary_$p $PYB 001_unit_quality_by_area.py $p --reload
  done
}
s2 & s3 &
wait
[ -e $W/s2_done ] || { log "stage 1 incomplete: stop"; exit 1; }
log "STAGE 1 DONE"

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
