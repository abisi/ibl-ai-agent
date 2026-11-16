#!/bin/bash
# 001 -- Per-session results for the sessions whose NWB content changed since they were computed (000):
#   AB155_20250208_175103 whisker_0  joined session (units re-sorted, passive post): ROC + pre-lick ROC (drift: 020)
#   MH038_20250524_105329 whisker_0  396 trials instead of 79 (reconverted 2026-11-02): drift + ROC + pre-lick ROC
#   MH068_20260125_124921 whisker_3  ephys since the 2026-11 reconversion, never tested: drift + ROC + pre-lick ROC
#   MH069_20260125_094519 whisker_3  idem (imec0 aligned 2026-11-03)
#   MH064_20260117_105034 whisker_3  ephys, never tested: drift + ROC + pre-lick ROC
#   MH065_20260116_174248 whisker_2  idem
# Previous files are moved to <folder>/_pre_rebuild_20261106/ (nothing deleted). Waits for 020 (AB155 drift test).
# Run (haas): nohup bash 001_tier0_reruns.sh > 001.log 2>&1 &   [--roc-only: skip the wait, the moves and the drift tests]
# 2026-11-06: first run without the NWB_reader PYTHONPATH -> ROC steps failed; rerun with --roc-only.
set -u
R=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4
N=/mnt/lsens-analysis/Axel_Bisi/NWB_ks4
PY=$HOME/code/unit_spikes_analysis/.venv/bin/python
export PYTHONPATH=$HOME/code/NWB_reader:$HOME/code/unit_spikes_analysis   # roc_utils_new imports NWB_reader_functions
REPO=$HOME/code/ibl-ai-agent
SESS="AB155_20250208_175103:whisker_0 MH038_20250524_105329:whisker_0 MH068_20260125_124921:whisker_3 MH069_20260125_094519:whisker_3 MH064_20260117_105034:whisker_3 MH065_20260116_174248:whisker_2"

ROC_ONLY=0; [ "${1:-}" = "--roc-only" ] && ROC_ONLY=1
if [ $ROC_ONLY = 0 ]; then
while pgrep -f "020_drift_test_final.py" >/dev/null; do sleep 60; done
echo "== $(date +%T) 020 finished"

echo "== move previous results aside"
for d in AB155/whisker_0/roc_analysis MH038/whisker_0/roc_analysis MH038/whisker_0/single_neuron_motion_shift_test; do
  b=$R/$d/_pre_rebuild_20261106; mkdir -p $b
  find $R/$d -maxdepth 1 -type f \( -name "*_roc_*" -o -name "*motion_shift_test_results.csv" \) -exec mv {} $b/ \;
  echo "$d: $(ls $b | wc -l) files -> _pre_rebuild_20261106"
done

echo "== $(date +%T) drift tests (missing sessions of MH038 MH064 MH065 MH068 MH069)"
cd $REPO/projects/ssl-prelick-convergence/data_prep
$PY run_missing_drift_tests.py --mice MH038 MH064 MH065 MH068 MH069 --n-workers 20 2>&1 | grep -v -i warn

fi
echo "== $(date +%T) main ROC"
for sd in $SESS; do
  s=${sd%%:*}; day=${sd##*:}; m=${s%%_*}; out=$R/$m/$day/roc_analysis; mkdir -p $out
  $PY $HOME/rebuild_20261106/001b_roc_session.py $N/$s.nwb $out 2>&1 | grep -v -i warn | tail -3
done

echo "== $(date +%T) pre-lick ROC (fa, sl)"
cd $REPO/projects/ssl-prelick-convergence/exploratory-analyses
ids=$(for sd in $SESS; do echo -n "${sd%%:*} "; done)
for ref in fa sl; do
  PRELICK_REF=$ref $PY 051_roc_prelick.py --sessions $ids --n-proc 6 2>&1 | grep -E "^\[051\]|ALL DONE|Error|Traceback"
done
echo "== $(date +%T) TIER0 FINISHED"
