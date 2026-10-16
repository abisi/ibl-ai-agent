"""Shared paths of ssl-rl-behaviour-model.
Model code (rl_independent / rl_shared / rl_mixed, rl_data_utils) is vendored from github.com/abisi/rl_model_auditory_whisker (branch
shared-pretraining); rl_joint holds the joint fit with one pretraining parameter set for R+ and R-.
Outputs: combined_results_ks4/ssl-rl-behaviour-model/{tables,targets,fits,figures}.
Run on haas (all compute there): cd <this folder> && OMP_NUM_THREADS=1 PYTHONPATH=~/code/NWB_reader ~/code/unit_spikes_analysis/.venv/bin/python ...
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-rl-behaviour-model"
TAB, TARGETS, FITS, FIG = ROOT / "tables", ROOT / "targets", ROOT / "fits", ROOT / "figures"
for _d in (TAB, TARGETS, FITS, FIG):
    _d.mkdir(parents=True, exist_ok=True)
TRIALS = TAB / "001_trials.csv"
NWB_DIR = axel_bisi_root() / "NWB_combined"
SHEET_REL = ("share_internal", "Axel_Bisi_Share", "dataset_info", "joint_mouse_reference_weight.xlsx")


def reference_sheet() -> Path:
    """joint mouse reference sheet (haas /mnt/share_internal/..., Windows M:/share_internal/...)"""
    for base in (Path("/mnt"), Path("M:/")):
        p = base.joinpath(*SHEET_REL)
        if p.exists():
            return p
    raise FileNotFoundError("joint_mouse_reference_weight.xlsx not found under /mnt or M:/")
