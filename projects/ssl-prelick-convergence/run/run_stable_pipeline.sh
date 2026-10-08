#!/bin/bash
# Full ssl-prelick-convergence pipeline on stable units (user 2026-10-07), reference SL, both populations.
# Results: combined_results_ks4/ssl-prelick-convergence/{across_days,within_day}/sl_stable/ (the good + mua results in sl/
# stay untouched as records). 051 (per-event rates + per-unit ROC) is unit-set independent and is not rerun; 053 rebuilds
# prelick_units.parquet with non-stable units relabelled "unstable". 077 (quiet windows + behaviour, unit independent)
# is not rerun: its performance table is copied. Part I and Part II run as two parallel chains after 053 (learner mice come from the behaviour table, not from 061 outputs).
# Usage (haas):  nohup bash run_stable_pipeline.sh > ~/stable_run.log 2>&1 &
# Status: one line per step in $STATUS (start / end / exit code).
set -u
cd ~/code/unit_spikes_analysis
export PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.
PY=./.venv/bin/python
EA=~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses
WD=$EA/within_day
B=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence
LOG=$B/run_logs/sl_stable; mkdir -p $LOG
STATUS=$LOG/status.txt

run() {  # run <name> <script> [args...]
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  $PY "$@" > $LOG/$name.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name exit=$rc" >> $STATUS
}

echo "$(date '+%F %T') PIPELINE START (PRELICK_UNITS=stable)" >> $STATUS
run 053 $EA/053_roc_prelick_table.py
mkdir -p $B/across_days/sl_stable/extras $B/within_day/sl_stable/cosyne   # 010 writes into cosyne/ before 005 creates it
cp $B/across_days/sl/extras/performance_sessions.csv $B/across_days/sl_stable/extras/ 2>/dev/null
# 061 runs in part1 after 057 / 059 (needs lambda_sessions.csv)

part1() {
  # 054 / 055 / 058 dropped 2026-10-08: superseded summaries (need the removed long_rt variant)
  run 056 $EA/056_roc_prelick_psth.py
  run 057 $EA/057_roc_prelick_lambda.py
  run 060 $EA/060_roc_prelick_lambda_lda.py
  run 063 $EA/063_decoder_sweep.py
  run 065 $EA/065_generalizing_units_location.py
  run 066 $EA/066_area_attrition.py
  run 067 $EA/067_generalizing_density_maps.py
  run 068 $EA/068_decoder_linear_shift.py --n-proc 40
  run 069 $EA/069_decoder_transfer_shift.py --n-proc 40
  run 059 $EA/059_roc_prelick_transfer.py     # after 069 (decoder panels)
  run 061 $EA/061_roc_prelick_learners.py     # after 057 and 059
  run 071 $EA/071_sl_timing_control.py
  run 074 $EA/074_geometry_normalisation.py
  run 076 $EA/076_session_geometry_extras.py --n-proc 30
  for pop in learners all; do
    run 075_$pop $EA/075_supp_selectivity_angles.py --population $pop
    run 078_$pop $EA/078_supp_shift_cd_quiet_performance.py --population $pop
    run 079_$pop $EA/079_supp_converging_sign_decoder_quality.py --population $pop
  done
  ( export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    for pop in learners all; do run 064_$pop $EA/064_pseudopop_hbootstrap.py --population $pop --n-proc 60; done )
  for pop in learners all; do run 062_$pop $EA/062_pub_convergence_figures.py --population $pop; done
  run 080 $EA/080_single_unit_geometry.py --n-proc 40
  run 081 $EA/081_single_unit_classes.py --n-proc 40
  run 082 $EA/082_neuron_class_psth.py
  run 083 $EA/083_cosyne_across_days.py
  echo "$(date '+%F %T') PART I DONE" >> $STATUS
}

part2() {
  run w001 $WD/001_within_session_halves.py --n-proc 50
  run w002 $WD/002_trial_slopes.py --n-proc 30
  SSL_MIN_WH=8 run w002_wh8 $WD/002_trial_slopes.py --n-proc 30
  run w004 $WD/004_halves_psth.py --n-proc 30
  run w009 $WD/009_mixed_model.py --n-proc 50
  run w003 $WD/003_epoch_comparison.py --n-proc 50
  run w003_n4 $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4
  run w003_u150x10 $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4 --n-units 150 --unit-draws 10
  run w003_u50x10 $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4 --n-units 50 --unit-draws 10
  run w003_u100x10 $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4 --n-units 100 --unit-draws 10
  run w003_u400x5 $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4 --n-units 400 --unit-draws 5
  run w003_uall $WD/003_epoch_comparison.py --n-proc 50 --n-fix 4 --n-units 0
  run w010 $WD/010_decoder_schemes.py
  run w008 $WD/008_cosyne_timeline_expanded.py
  run w015 $WD/015_area_cd_drift.py --n-proc 30
  for pop in learners all; do
    for s in 011_cd_trajectories 012_psth_halves_groups 013_within_session_change 014_cohort_specificity 016_day0_predicts_expert 019_cosyne_day0_v2; do
      run w${s:0:3}_$pop $WD/$s.py --population $pop
    done
  done
  run w017 $WD/017_area_cd_drift_views.py
  run w020 $WD/020_area_cd_panels.py
  run w022 $WD/022_wh_trial_trajectories.py
  run w023 $WD/023_cd_cohorts_by_stage.py
  run w024 $WD/024_drift_compact.py
  run w025 $WD/025_cd_normalisation.py --n-proc 40
  echo "$(date '+%F %T') PART II DONE" >> $STATUS
}

part1 &
P1=$!
part2 &
P2=$!
wait $P1 $P2
run 084 $EA/084_along_vs_orthogonal.py          # needs 057 (part 1) and w025 (part 2)
run w021 $WD/021_figure_index.py
echo "$(date '+%F %T') PIPELINE DONE" >> $STATUS
