#!/bin/bash
# Stop the first rebuild run (it exhausted haas memory), reconvert the 4 NWBs of the round-2 target changes (checked swap,
# backups in NWB_ks4_backup_20261117_round2), then restart rebuild_chain.sh. Run (haas):
#   nohup setsid bash ~/rebuild_20261117/round2_then_rebuild.sh > ~/rebuild_20261117/round2.log 2>&1 < /dev/null &
set -u
W=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-table-rebuild-20261117
for p in rebuild_chain.sh "ibl-ai-agent build-ssl" 001b_roc_session.py 001_unit_quality_by_area.py 051_roc_prelick.py; do pkill -f "$p"; done
sleep 10; echo "$(date '+%F %T') first run stopped" >> $W/logs/status.txt
cp $W/reconvert_targets_20261117.py ~/reconvert_targets_20261117.py
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
for s in MH035_20250514_142713 MH035_20250515_161921 MH062_20260111_112441 MH022_20250309_161625; do
  ( cd ~ && ~/code/nwb_converter_env/bin/python ~/reconvert_targets_20261117.py $s > ~/reconv_round2_$s.out 2>&1 ) &
done
wait
grep -h "SWAPPED IN\|NOT swapped\|FAILED\|target old" ~/reconv_round2_*.out | tee -a $W/logs/round2_reconversion.txt
n=$(grep -l "SWAPPED IN" ~/reconv_round2_*.out | wc -l)
echo "$(date '+%F %T') round-2 reconversions swapped in: $n / 4" >> $W/logs/status.txt
[ "$n" = 4 ] || { echo "$(date '+%F %T') not all swapped: rebuild NOT restarted" >> $W/logs/status.txt; exit 1; }
cp $W/rebuild_chain.sh ~/rebuild_20261117/rebuild_chain.sh
exec bash ~/rebuild_20261117/rebuild_chain.sh > ~/rebuild_20261117/chain_restart.log 2>&1
