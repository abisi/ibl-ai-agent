"""Area-comparison view of the performance-state sweep (complements
`018_plot_perfstate_progress.py`, which compares cohort/state within one
area per panel): here each panel is one (cohort, state) group, with every
area's mean decode curve overlaid so areas can be compared directly against
each other. Chance level (0.5 -- exact for balanced accuracy regardless of
class imbalance) is always drawn as a horizontal reference line.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
PARTIAL_PATH = OUT_DIR / "015_perfstate_results_partial.parquet"
BIN_EDGES_PATH = OUT_DIR / "015_bin_edges.json"
CHANCE_LEVEL = 0.5  # exact for balanced accuracy, any class balance

AREA_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#aec7e8",
]


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[df["skipped_reason"].isna() & (df["area_col"] == "area_group")].copy()
    if len(df) == 0:
        print("no computed area_group rows yet")
        return

    bin_edges = json.loads(BIN_EDGES_PATH.read_text())
    bin_centers_ms = np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])

    areas = sorted(df["area_value"].unique(), key=lambda a: -df[df.area_value == a]["session_id"].nunique())
    area_color = {a: AREA_COLORS[i % len(AREA_COLORS)] for i, a in enumerate(areas)}

    n_sessions = df["session_id"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    panels = [("R+", "high"), ("R+", "low"), ("R-", "high"), ("R-", "low")]

    for ax, (cohort, state) in zip(axes.flat, panels):
        sub = df[(df.reward_group == cohort) & (df.perf_state == state)]
        for area in areas:
            curves = [np.array(c) for c in sub.loc[sub.area_value == area, "real_curve"]]
            if len(curves) < 2:
                continue
            mean_curve = np.nanmean(np.stack(curves), axis=0)
            ax.plot(bin_centers_ms, mean_curve, color=area_color[area], lw=2, label=area)

        ax.axhline(CHANCE_LEVEL, color="#888888", lw=1.2, linestyle=":", zorder=0, label="chance (0.5)")
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_title(f"{cohort}, {state}-perf state", fontsize=11)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("time from start_time (ms)", fontsize=9)
        ax.set_ylabel("balanced accuracy", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    by_label = {}
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            by_label.setdefault(ll, hh)
    ordered_labels = [a for a in areas if a in by_label] + [l for l in by_label if l not in areas]
    fig.legend([by_label[l] for l in ordered_labels], ordered_labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.suptitle(f"Area comparison within cohort x performance-state ({n_sessions} sessions so far)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out_path = OUT_DIR / "019_perfstate_area_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path} ({n_sessions} sessions)")


if __name__ == "__main__":
    main()
