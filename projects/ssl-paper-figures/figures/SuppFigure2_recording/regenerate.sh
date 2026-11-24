#!/usr/bin/env bash
# Supp. Fig. 2 (probe localization + preprocessing yield): regenerate every computed panel source (spec.yaml).
# The figure itself is assembled by hand; this only rewrites the panel files, in place (dated names for p2 a, b).
# Run on haas from anywhere: bash regenerate.sh [p1b p1d p1e p2a p2b p2cd]   (no argument = all)
set -euo pipefail
AGENT=~/code/ibl-ai-agent/projects
PY=~/code/unit_spikes_analysis/.venv/bin/python
NAS=/mnt/lsens-analysis/Axel_Bisi
ALL="${*:-}"
run() { [ -z "$ALL" ] || [[ " $ALL " == *" $1 "* ]]; }

if run p1b; then   # page 1 b: entry points on the atlas surface (4d ... _plain); copy into tracks/entry_points_surface
  (cd $AGENT/ssl-anatomy-tracks && export PYTHONPATH=$NAS/Github/ephys_utilities:. &&
   $PY insertion_points.py && $PY entry_crossref.py && $PY entry_surface_map.py)
  cp $NAS/combined_results_ks4/anatomy/_viz_trials_20261116/4d_entry_points_atlas_surface* \
     $NAS/combined_results_ks4/anatomy/tracks/entry_points_surface/
fi
if run p1d; then   # page 1 d: wS1 barrel registration example track
  (cd ~/code/wS1_barrel_registration && .venv/bin/python scripts/paper_figure.py \
     -c $NAS/combined_results_ks4/ws1-barrel-registration/D2_final/config.toml AB164 \
     --tracks /raid0/lmsmith/local_barrel_reg/D2_neuropixels/tracks/tracks_summary.csv --track imec2)
fi
if run p1e; then   # page 1 e: 3D probe-track renders (the script's __main__ sweep includes reward_group / day0 / sagittal, top)
  (cd $NAS/Github/ephys_utilities/ephys_utilities/allen_utils && ~/anaconda3/envs/brainrender/bin/python \
     generate_probe_tracks_visualization_colors_single_parallel.py)
fi
if run p2a; then   # page 2 a: motion correction example AB150 imec1 (density style, preview mode -> _explore/)
  (cd $AGENT/ssl-paper-figures/figures/SuppFigure2_recording && $PY motion_example.py density \
     $NAS/data/AB150/AB150_20241219_144646/Ephys/catgt_AB150_g0/AB150_g0_imec1 1442.67 2325 AB150)
fi
if run p2b; then   # page 2 b: unit (in)stability types
  (cd $AGENT/ssl-paper-figures && PYTHONPATH=~/code/unit_spikes_analysis $PY explore_unit_instability.py)
fi
if run p2cd; then  # page 2 c, d: area x cohort neuron-count grid, all mice (brain_wide_analysis env; see 001 docstring)
  (cd $AGENT/ssl-data-summary && python 001_unit_quality_by_area.py all_mice area_acronym_count_grid)
fi
