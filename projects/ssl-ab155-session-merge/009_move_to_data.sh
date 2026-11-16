#!/bin/bash
# 009 -- Copy the joined AB155 day-0 sorting (run A = run 4819418: DREDge before sorting + Kilosort 4 drift correction;
# user 2026-11-05) into the analysis data folder that the NWB converter reads, keeping everything that was there.
# Runs on haas (lsens-analysis mounted at /mnt/lsens-analysis). The joined outputs are copied (user 2026-11-06), so
# Misc/SessionToConcat keeps its copy until the merged data is checked; the backup in step 1 is a rename (same share).
# Default: dry run (prints the commands). Apply: bash 009_move_to_data.sh --apply
#
# 1. Back up the current g0-only processed Ephys folder (1.4 TB) by renaming it to Ephys_g0only_backup_20261106 (the
#    converter only reads <session>/Ephys, so the backup is invisible to it). Nothing deleted.
# 2. Copy the joined run A outputs (SessionToConcat/..._joined/Ephys/catgt_AB155_g0: kilosort4 incl. Bombcell, C_Waves,
#    ibl_format, dredge, TPrime synced spike times, sync_event_times, sc_offsets, NIDQ meta and event files; 68 GB,
#    verified against Kuma) to <session>/Ephys/catgt_AB155_g0. concat_figure/ and nwb_test/ stay in SessionToConcat.
#    Not included: the supercat binaries (their source stays in lsens-data Recording/.../Ephys/AB155_g0, 399 GB) and the
#    preprocessed caches (temp_wh.dat, traces_cached_seg0.raw; on Kuma /work only).
# 3. Probe alignment: each joined imec folder holds ibl_format/channel_locations.json copied from the g0-only
#    alignment (same insertion, channel-indexed; copied for the test NWB). Kept as is unless re-aligned.
# 4. DLC: joined DLC per camera = part 1 (DLCOutput/DLC_Day0, 17-50-5x, 595,292 rows) + part 2 (18-55-xx, 77,288
#    rows) concatenated in time order (672,580 rows vs 672,582 synced camera frames), written under the part-1 file name
#    (the joined video keeps the part-1 name, lsens-data Recording/.../Video, 309.9 GB). The current part-1-only h5
#    files in <session>/Video are renamed *_part1only_backup_20261106.h5 first.
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
run() { echo "+ $*"; if [ $APPLY = 1 ]; then "$@"; fi; }
A=/mnt/lsens-analysis/Axel_Bisi
S=$A/data/AB155/AB155_20250208_175103
J=$A/Misc/SessionToConcat/AB155_20250208_175103_joined
DLC=$A/DLCOutput/DLC_Day0
TAG=20261106

# preflight
[ -d "$S/Ephys/catgt_AB155_g0" ] || { echo "missing $S/Ephys/catgt_AB155_g0"; exit 1; }
[ -d "$J/Ephys/catgt_AB155_g0" ] || { echo "missing $J/Ephys/catgt_AB155_g0"; exit 1; }
[ ! -e "$S/Ephys_g0only_backup_$TAG" ] || { echo "backup already exists: $S/Ephys_g0only_backup_$TAG"; exit 1; }
for p in 0 1 2 3 4; do
  for f in kilosort4/sorter_output/spike_clusters.npy kilosort4/sorter_output/cluster_bc_unitType.tsv \
           kilosort4/sorter_output/cwaves/waveform_metrics.csv kilosort4/sorter_output/bombcell/templates._bc_qMetrics.parquet \
           ibl_format/channel_locations.json AB155_g0_imec${p}_kilosort4_spike_times_sec_sync.npy; do
    [ -e "$J/Ephys/catgt_AB155_g0/AB155_g0_imec$p/$f" ] || { echo "missing imec$p/$f"; exit 1; }
  done
done
echo "preflight ok (apply=$APPLY)"

# 1-2
run mv "$S/Ephys" "$S/Ephys_g0only_backup_$TAG"
run mkdir "$S/Ephys"
# rsync, not cp -a: the NAS share refuses to preserve times, and preprocess/ (Kuma caches backed up into
# SessionToConcat from 2026-11-06, 157 GB per probe) must not be copied
run rsync -r --size-only --exclude "preprocess/" "$J/Ephys/catgt_AB155_g0/" "$S/Ephys/catgt_AB155_g0/"
run sh -c "echo 'AB155 day 0 joined (supercat g0 + g1 passive post), run 4819418 (DREDge + Kilosort 4 drift correction), copied $TAG from Misc/SessionToConcat; g0-only processing kept in ../Ephys_g0only_backup_$TAG' > '$S/Ephys/README_joined.txt'"

# 4
for v in "side 25-02-08 17-50-56|side 25-02-08 18-55-50" "top 25-02-08 17-50-54|top 25-02-08 18-55-49"; do
  p1="AB155_${v%%|*}DLC_Resnet50_Tracking2DAug10shuffle3_snapshot_best-90_filtered.h5"
  p2="AB155_${v##*|}DLC_Resnet50_Tracking2DAug10shuffle3_snapshot_best-90_filtered.h5"
  [ -e "$S/Video/$p1" ] && run mv "$S/Video/$p1" "$S/Video/${p1%.h5}_part1only_backup_$TAG.h5"
  run /home/bisi/code/ibl-ai-agent/.venv/bin/python -c "
import pandas as pd
a = pd.read_hdf('$DLC/$p1'); b = pd.read_hdf('$DLC/$p2')
assert list(a.columns) == list(b.columns), 'DLC columns differ between parts'
j = pd.concat([a, b], ignore_index=True)
j.to_hdf('$S/Video/$p1', key='df_with_missing', mode='w')
print('$p1', len(a), '+', len(b), '=', len(j))
"
done
echo done
