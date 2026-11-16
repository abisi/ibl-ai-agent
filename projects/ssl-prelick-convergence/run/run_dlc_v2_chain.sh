#!/bin/bash
# DLC v2 rerun of the jaw / kinematics chain (2026-11-12; user: "ALL analyses should use what is in NWB_ks4"): every DLC series and
# jaw onset from ephys_utilities dlc_helpers (NAS checkout first on PYTHONPATH). Writes only v2 names (dlc_version.py):
#   across_days/sl_stable/{dlc_passive_pre_v2, prelick_kinematics_v2, jaw_control_v2, publication_v2}, single_neuron_signed/*_v2,
#   across_days/sl_stable_jaw2 + _jawc2, within_day/sl_stable_jaw2 + _jawc2, per-session *_roc_prelick_sl_jaw2 / _sl_jawc2_trials.npz.
# Usage (haas): nohup setsid bash run_dlc_v2_chain.sh > $B/logs_v2_20261112/chain.log 2>&1 < /dev/null &
# Status: one line per step in $LOG/status.txt. NP = processes per step (two steps run in parallel at most).
set -u
NP=${NP:-18}
cd ~/code/unit_spikes_analysis
export PRELICK_REF=sl PRELICK_UNITS=stable PRELICK_DLC=v2 PRELICK_SELECT=all
export PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:/mnt/lsens-analysis/Axel_Bisi/NWB_reader:.
PY="nice -n 10 ./.venv/bin/python"
EA=~/code/ibl-ai-agent/projects/ssl-prelick-convergence/exploratory-analyses
WD=$EA/within_day
B=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence
LOG=$B/logs_v2_20261112; mkdir -p $LOG
STATUS=$LOG/status.txt

run() {  # run <name> <env assignments...> -- <script> [args...]
  local name=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  env "${envs[@]}" $PY "$@" > $LOG/$name.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name exit=$rc" >> $STATUS
}

echo "$(date '+%F %T') CHAIN START (PRELICK_DLC=v2, NP=$NP)" >> $STATUS
# stage 1: jaw onsets (pre, centered) in parallel
run 118_jaw2 PRELICK_ALIGN=lick -- $EA/118_jaw_onset_events.py --n-proc $NP &
run 118_jawc2 PRELICK_ALIGN=lick PRELICK_JAW_WIN=centered -- $EA/118_jaw_onset_events.py --n-proc $NP &
wait
# stage 2: DLC series (passive pre, pre-lick kinematics) || jaw-aligned across days (087) for both alignments
( run 113_v2 PRELICK_ALIGN=lick -- $EA/113_dlc_passive_pre.py --n-proc $NP
  run 116_v2 PRELICK_ALIGN=lick -- $EA/116_prelick_kinematics.py --n-proc $NP
  run 126_v2 PRELICK_ALIGN=lick -- $EA/126_jaw_control.py --n-proc $NP
  run 125_jaw_v2 PRELICK_ALIGN=lick -- $EA/125_single_neuron_figure.py --jaw-only ) &
( run 087_jaw2 PRELICK_ALIGN=jaw2 -- $EA/087_across_days_harmonised.py --n-proc $NP
  run 087_jawc2 PRELICK_ALIGN=jawc2 -- $EA/087_across_days_harmonised.py --n-proc $NP
  run 026wd_jaw2 PRELICK_ALIGN=jaw2 -- $WD/026_cd_shift_null.py --n-proc $NP
  run 026wd_jawc2 PRELICK_ALIGN=jawc2 -- $WD/026_cd_shift_null.py --n-proc $NP
  run 120_jaw2 PRELICK_ALIGN=jaw2 -- $EA/120_cd_jaw_timecourse.py --n-proc $NP
  run 123_jaw2 PRELICK_ALIGN=jaw2 -- $EA/123_single_neuron_signed.py --n-proc $NP
  run 124_v2 PRELICK_ALIGN=lick -- $EA/124_signed_selectivity_timecourse.py ) &
wait
run 115_v2 PRELICK_ALIGN=lick -- $EA/115_cosyne_combined.py
echo "$(date '+%F %T') CHAIN END" >> $STATUS
