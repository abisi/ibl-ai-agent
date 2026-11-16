#!/bin/bash
# submit.sh -- freeze config.yaml + the run table, then submit one stage (as the pipeline's submit_ephys.sh).
# Usage: ./submit.sh <stage> [--slice] [extra sbatch args, e.g. --dependency=afterok:123]
#   stage: stage | sort-container | sort-host | sort-ibl | sort-dartsort | post | figures
# --slice: 5-min test slice of the first recording (outputs in *_slice folders), one run per sorter.
set -euo pipefail
STAGE=${1:?stage}; shift
SLICE=""
if [ "${1:-}" = "--slice" ]; then SLICE="--slice"; shift; fi
CODE=/home/bisi/code/npworkshop_figures
OUT=/work/lsens/bisi/data/npworkshop
SIF=/home/bisi/code/ephys_preprocessing/singularity/ephys-pipeline_latest.sif
mkdir -p $OUT/logs $OUT/config_snapshots
TS=$(date +%Y%m%d_%H%M%S)
FROZEN=$OUT/config_snapshots/config_${TS}.yaml
cp $CODE/config.yaml $FROZEN
PYC="singularity exec --bind /work/lsens/bisi --bind $CODE $SIF python3.11"
# run table snapshot + array sizes
$PYC -c "
import sys; sys.path.insert(0, '$CODE'); import npw, json
cfg = npw.load_config('$FROZEN'); sl = '$SLICE' == '--slice'
rows = npw.run_table(cfg, sl)
json.dump(rows, open('$OUT/config_snapshots/runs_${TS}.json', 'w'), indent=1)
for env in ('container', 'npw-host', 'npw-ibl', 'npw-dartsort'):
    print(env, sum(r['env'] == env for r in rows))
print('post', len(npw.all_runs(cfg, sl)))
" 2>/dev/null | grep -v -i warn > $OUT/config_snapshots/counts_${TS}.txt
n() { awk -v k=$1 '$1==k {print $2}' $OUT/config_snapshots/counts_${TS}.txt; }
EXP="ALL,NPW_CONFIG=$FROZEN,NPW_SLICE=$SLICE"
case $STAGE in
  stage)          sbatch --export=$EXP "$@" $CODE/sbatch/stage_inputs.sbatch ;;
  sort-container) sbatch --export=$EXP --array=1-$(n container)%8 "$@" $CODE/sbatch/sort_container.sbatch ;;
  sort-host)      sbatch --export=$EXP,NPW_ENV=npw-host --job-name=npw-host --array=1-$(n npw-host)%6 "$@" $CODE/sbatch/sort_env.sbatch ;;
  sort-ibl)       sbatch --export=$EXP,NPW_ENV=npw-ibl --job-name=npw-ibl --array=1-$(n npw-ibl)%3 "$@" $CODE/sbatch/sort_env.sbatch ;;
  sort-dartsort)  sbatch --export=$EXP,NPW_ENV=npw-dartsort --job-name=npw-dartsort --array=1-$(n npw-dartsort)%3 "$@" $CODE/sbatch/sort_env.sbatch ;;
  post)           sbatch --export=$EXP --array=1-$(n post)%12 "$@" $CODE/sbatch/postprocess.sbatch ;;
  figures)        sbatch --export=$EXP "$@" $CODE/sbatch/figures.sbatch ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "frozen config: $FROZEN"
cat $OUT/config_snapshots/counts_${TS}.txt
