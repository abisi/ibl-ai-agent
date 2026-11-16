#!/bin/bash
# 00 -- uv envs for the sorters that are not in the pipeline container (login node; downloads only, no compute).
# Pinned versions (all released >= 2 weeks before 2026-11-12); SpikeInterface pinned to the container's 0.104.3 so
# every env reads the same input folder and writes the same sorting format.
#   npw-host:     spikeinterface 0.104.3 + mountainsort5 0.5.10 + spython (SI's apptainer runner for Kilosort 2.5)
#   npw-ibl:      ibl-sorter git tag 1.13.0 + torch (cu128 wheels) + cupy-cuda12x
#   npw-dartsort: dartsort 0.5.31 (Python 3.12) + torch
set -euo pipefail
E=/work/lsens/bisi/npworkshop_envs
UV=/usr/local/bin/uv
export UV_CACHE_DIR=$E/uv_cache
mkdir -p $E
cd $E

$UV venv --python 3.11 npw-host
$UV pip install --python npw-host/bin/python "spikeinterface[full]==0.104.3" "mountainsort5==0.5.10" hdbscan spython pyyaml "six==1.17.0"

$UV venv --python 3.12 npw-ibl
[ -d ibl-sorter ] || git clone --branch 1.13.0 --depth 1 https://github.com/int-brain-lab/ibl-sorter.git
$UV pip install --python npw-ibl/bin/python torch --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python npw-ibl/bin/python cupy-cuda12x -e ./ibl-sorter "spikeinterface==0.104.3" pyyaml
# the resolver otherwise picks boto3 0.0.3 / botocore 0.76 / six 1.7 (one-api import fails on Python 3.12)
$UV pip install --python npw-ibl/bin/python "boto3>=1.35" "botocore>=1.35" "six==1.17.0"

$UV venv --python 3.12 npw-dartsort
$UV pip install --python npw-dartsort/bin/python torch --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python npw-dartsort/bin/python "dartsort==0.5.31" "spikeinterface==0.104.3" pyyaml

# Kilosort 2.5 image (SpikeInterface's compiled MATLAB-runtime build), cached for the jobs
export APPTAINER_CACHEDIR=$E/apptainer_cache
for env in npw-host npw-ibl npw-dartsort; do
  echo "== $env"; $env/bin/python -c "import spikeinterface, sys; print(sys.version.split()[0], 'SI', spikeinterface.__version__)"
done
npw-ibl/bin/python -c "from iblsorter.utils import cuda_installation_test; print('iblsorter import ok')"
npw-dartsort/bin/python -c "import dartsort; print('dartsort', dartsort.__version__)"
