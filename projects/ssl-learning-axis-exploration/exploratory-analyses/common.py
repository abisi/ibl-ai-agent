"""Shared paths of ssl-learning-axis-exploration. The statistics and plotting helpers, population rule and project paths are those of
ssl-within-day-two-axes (re-exported, so both projects use one implementation); inputs from that project (per-event table 008) are read
from TAB_WD; outputs go to combined_results_ks4/ssl-learning-axis-exploration/{tables,figures}.
Run on haas with the ssl-prelick-convergence environment (see ssl-within-day-two-axes/exploratory-analyses/common.py).
"""
from __future__ import annotations

import importlib.util as _u
import sys
from pathlib import Path as _P

_SRC = _P(__file__).resolve().parents[2] / "ssl-within-day-two-axes" / "exploratory-analyses" / "common.py"
sys.path.insert(0, str(_SRC.parent))
_spec = _u.spec_from_file_location("_wd_common", _SRC)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})

TAB_WD = _mod.TAB                                                   # inputs from ssl-within-day-two-axes (008 per-event table)
ROOT = _mod.axel_bisi_root() / "combined_results_ks4" / "ssl-learning-axis-exploration"
TAB, FIG = ROOT / "tables", ROOT / "figures"
for _d in (TAB, FIG):
    _d.mkdir(parents=True, exist_ok=True)
