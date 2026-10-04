"""Figures for `095_learning_trial_split_shiftnull.py` (user request
2026-09-24: redo of the pre/post learning_trial hit/miss decoding with the
linear shift null, three feature windows, terminal disengagement dropped).
Same figure set as `091_learning_trial_split_figures.py` (027/029/034-style,
both population scopes), generated once per window by reusing 091's
functions with 095's inputs: "own null" now = the session-level linear
shift null (`nullmean_<ep>_<matched|full>`).

Outputs, per window W in {sensory, baseline, sensory_minus_base}:
  figures/<scheme>/learning/096_ltsplit_shiftnull_<scheme>_<W>_*.png/.pdf (prefix shortened 2026-09-24: Windows MAX_PATH hit for area_acronym_custom)
  096_learning_trial_split_shiftnull_stats_<W>.csv

Usage (local, after pulling 095 parquets): python 096_learning_trial_split_shiftnull_figures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f091", OUT_DIR / "091_learning_trial_split_figures.py")
f091 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f091)

WINDOW_LABELS = {
    "sensory": "sensory 5-50ms",
    "baseline": "baseline -200 to -10ms",
    "sensory_minus_base": "sensory minus baseline (per unit)",
}


def main():
    f091.FILE_PREFIX = "096_ltsplit_shiftnull"
    f091.NULL_LABEL = "shift null"
    f091.SRC_TEMPLATE = "095_lt_split_shiftnull_{scheme}.parquet"
    f091.METRICS = [
        ("matched_abovenull", "matched: acc - shift null", lambda d, ep: d[f"acc_{ep}_matched"] - d[f"nullmean_{ep}_matched"], 0.0),
        ("matched_raw", "matched: balanced accuracy", lambda d, ep: d[f"acc_{ep}_matched"], 0.5),
        ("full_abovenull", "unmatched: acc - shift null", lambda d, ep: d[f"acc_{ep}_full"] - d[f"nullmean_{ep}_full"], 0.0),
    ]
    for window, label in WINDOW_LABELS.items():
        print(f"=== window {window}")
        f091.WINDOW_FILTER = window
        f091.WIN_TAG = window
        f091.WIN_LABEL = label
        f091.STATS_CSV = f"096_learning_trial_split_shiftnull_stats_{window}.csv"
        f091.main()


if __name__ == "__main__":
    main()
