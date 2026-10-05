"""150 -- Passive pre -> post change of the alignment of passive patterns with the active hit / miss axes, beyond the
LINEAR-SHIFT null (user 2026-10-05: "do the shifts for these analyses too"): 135 lick axis (whole brain) and 140 coding
direction of the second half (hit-median split). Same null and framing as 146 / 149 (shift_null.py): the axis is rebuilt from
labels shifted against the time-ordered active whisker trials, the passive patterns are the real ones.
Rows: 135 lick axis | 140 CD_2 (first column: passive whisker - auditory axis, raw cos). Columns (excess = real change - mean null change; negative = moves away from the axis more than
session time alone):
  a  whisker-evoked, raw cos      b  auditory-evoked, raw cos      c  whisker - auditory, raw cos
  d  whisker - auditory, projection on the unit axis / sqrt(n units) (linear: a shared additive drift cancels exactly)
  e  whisker-evoked, noise-corrected cos (sensitivity only: the shifted axes are mostly below the 0.05 reliability floor)
Tests: within cohort Wilcoxon | t (excess vs 0); R+ vs R- Mann-Whitney | Welch. Session = unit; uncorrected; scopes all /
learners. Outputs: figures/publication/150_axis_alignment_shift_null_<scope>.{png,pdf,svg}; 150_stats.csv
Run (haas, repo root): python .../150_axis_alignment_shift_null_figure.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
H = importlib.import_module("143_lt_split_windows_figures")
F149 = importlib.import_module("149_passive_readout_shift_null_figure")
FIGDIR = H.FIGDIR
COLS = [("shift_excess_dWR", "whisker-evoked\nraw cos"), ("shift_excess_dAR", "auditory-evoked\nraw cos"),
        ("shift_excess_dWAR", "whisker - auditory\nraw cos"), ("shift_excess_dWAP", "whisker - auditory\nprojection"),
        ("shift_excess_dWN", "whisker-evoked\nnoise-corr. cos (sens.)")]


EXTRA = {"140 coding direction (half 2)": [("shift_excess_daxisR", "passive whisker axis\n(W - A), raw cos")]}   # 140 only


def load():
    a = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    a = a[(a.area == "All units") & a.skipped_reason.isna() & a.shift_n.notna()]
    c = pd.read_parquet(EA / "140_coding_direction_noise.parquet")
    c = c[(c.split == "hitmedian") & c.skipped_reason.isna() & c.shift_n.notna()]
    return [("135 lick axis", a), ("140 coding direction (half 2)", c)]


def main():
    H.setup()
    rows = []
    for scope in ("all", "learners"):
        fig, axes = plt.subplots(2, len(COLS) + 1, figsize=(8.8, 4.4))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.08, hspace=1.0, wspace=0.7)
        for r, (name, d) in enumerate(load()):
            if scope == "learners":
                d = d[H.mouse_of(d).isin(H.learners())]
            cols = EXTRA.get(name, []) + COLS
            for j, (col, lab) in enumerate(cols):
                F149.excess_panel(axes[r, j], d, col, lab, rows, scope, name)
            for j in range(len(cols), axes.shape[1]):
                axes[r, j].set_visible(False)
            axes[r, 0].set_ylabel(f"{name}\nexcess post - pre change\n(real - shift null)", fontsize=5.5)
            fig.text(0.01, axes[r, 0].get_position().y1 + 0.06, f"{'ab'[r]}", fontsize=8, weight="bold")
        fig.suptitle("Passive pre -> post alignment with the active hit / miss axes beyond the linear-shift null\n"
                     f"(whole brain, shared tracked stable units, {scope}; negative = away from the axis beyond session time)", fontsize=7)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGDIR / f"150_axis_alignment_shift_null_{scope}.{ext}", dpi=300)
        plt.close(fig)
    R = pd.DataFrame(rows).rename(columns={"response": "analysis"})
    R.to_csv(EA / "150_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
