#!/bin/bash
# uv env for brain-3d-render on Kuma (login node, downloads only). Versions released >= 2 weeks before 2026-11-12.
set -euo pipefail
E=/work/lsens/bisi/brain3d/env
export UV_CACHE_DIR=/work/lsens/bisi/npworkshop_envs/uv_cache
/usr/local/bin/uv venv --python 3.12 $E
/usr/local/bin/uv pip install --python $E/bin/python "pyvista==0.49.1" "vtk==9.7.1" "tifffile==2026.10.15" \
    "imageio==2.37.0" "imageio-ffmpeg==0.7.0" h5py pyyaml numpy
$E/bin/python -c "import vtk, pyvista; print('vtk', vtk.vtkVersion.GetVTKVersion(), 'pyvista', pyvista.__version__)"
