#!/bin/bash
# 003 -- (local PC, Git Bash) after the ssl_ephys / ssl_behavior rebuild (scripts/run_ssl_ks4_build.ps1): rebuild the derived
# tables in reports/ssl_analysis/derived, stage datasets + derived on the NAS, swap them into the haas repo copy (old haas
# copies moved to the NAS backup) and write the flag that 002 waits for.
# Order: target region -> mouse reference -> unit coverage -> quality labels -> area labels (+ quality merge) -> analysis units
# -> trial-window rates -> LMM table -> modulation index -> passive PSTHs.
# Run: nohup bash projects/ssl-table-rebuild-20261106/003_derived_and_sync.sh > <log> 2>&1 &
set -u
REPO=/c/Users/bisi/Github/int-brain-lab/ibl-ai-agent
cd $REPO
PY=$REPO/.venv/Scripts/python.exe
D=reports/ssl_analysis/derived
STAGE=/m/analysis/Axel_Bisi/combined_results_ks4/ssl-table-rebuild-20261106/staging
HB=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-table-rebuild-20261106/pre_rebuild/haas_reports

START_AT=${1:-}   # resume from this step (skips the wait and the derived backup)
if [ -z "$START_AT" ]; then
until ls -t reports/ssl_build_logs/ssl_ks4_build_*.log 2>/dev/null | head -1 | xargs grep -q "SSL KS4 build run finished"; do sleep 120; done
L=$(ls -t reports/ssl_build_logs/ssl_ks4_build_*.log | head -1)
grep -q "FAILED" $L && { echo "dataset build FAILED, see $L"; exit 1; }
echo "$(date +%T) datasets rebuilt ($L)"

cp -a $D ${D}_pre_rebuild_20261106
fi
skip=1; [ -z "$START_AT" ] && skip=0
for s in extract_ssl_target_region extract_ssl_mouse_reference compute_ssl_unit_coverage compute_ssl_quality_label \
         build_ssl_area_labels build_ssl_analysis_units compute_ssl_trial_window_rates build_ssl_lmm_table          compute_ssl_modulation_index compute_ssl_psth; do
  [ "$s" = "$START_AT" ] && skip=0; [ $skip = 1 ] && continue
  echo "$(date +%T) START $s"
  $PY scripts/$s.py > $REPO/reports/ssl_build_logs/rebuild_20261106_$s.log 2>&1 || { echo "$s FAILED"; exit 1; }
  echo "$(date +%T) END   $s"
done

echo "$(date +%T) staging on the NAS"
rm -rf $STAGE; mkdir -p $STAGE/datasets/ssl_ephys $STAGE/datasets/ssl_behavior $STAGE/ssl_analysis
cp -r reports/datasets/ssl_ephys/1.0.0 $STAGE/datasets/ssl_ephys/
cp -r reports/datasets/ssl_behavior/1.0.0 $STAGE/datasets/ssl_behavior/
cp -r $D $STAGE/ssl_analysis/derived

echo "$(date +%T) swapping into haas"
ssh -n -o BatchMode=yes haas "set -e; S=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-table-rebuild-20261106/staging
  R=~/code/ibl-ai-agent/reports; mkdir -p $HB/datasets/ssl_ephys $HB/datasets/ssl_behavior $HB/ssl_analysis
  mv \$R/datasets/ssl_ephys/1.0.0 $HB/datasets/ssl_ephys/; mv \$R/datasets/ssl_behavior/1.0.0 $HB/datasets/ssl_behavior/
  cp -a \$R/ssl_analysis/derived $HB/ssl_analysis/
  cp -r \$S/datasets/ssl_ephys/1.0.0 \$R/datasets/ssl_ephys/; cp -r \$S/datasets/ssl_behavior/1.0.0 \$R/datasets/ssl_behavior/
  rm -rf \$R/ssl_analysis/derived; cp -r \$S/ssl_analysis/derived \$R/ssl_analysis/derived
  touch ~/rebuild_20261106/derived_ready.flag; echo haas swap done"
echo "$(date +%T) 003 DONE"
# 2026-11-06 run: the swap stopped after the backups (mv/cp -a onto the NAS cannot preserve permissions -> exit 1 under
# set -e). Finished by hand: staging renamed to haas_reports_20261106 (NAS), haas reports/datasets/*/1.0.0 are symlinks to it
# (15.6 GB, haas / had 18 GB free), derived copied with cp -r --no-preserve=all, flag touched.
