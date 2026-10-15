"""Regenerable progress figure for `002_learning_stage_pilot_sweep.py`: real
decode-accuracy-vs-time curves for whatever has landed in
`002_pilot_results_partial.parquet` so far, faceted by `area_group` (the
coarse scheme -- comparable across sessions, unlike the many small
`area_acronym_custom` areas), colored by cohort. Rerun any time to refresh
against the latest partial results; safe while the sweep is still writing
(reads a snapshot each call).
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
PARTIAL_PATH = OUT_DIR / "002_pilot_results_partial.parquet"
BIN_EDGES_PATH = OUT_DIR / "002_bin_edges.json"

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}  # established SSL R+/R- convention


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[df["skipped_reason"].isna() & (df["area_col"] == "area_group")].copy()
    if len(df) == 0:
        print("no computed area_group rows yet")
        return

    bin_edges = json.loads(BIN_EDGES_PATH.read_text())
    bin_centers_ms = np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])

    areas = sorted(df["area_value"].unique())
    n_areas = len(areas)
    ncols = min(3, n_areas)
    nrows = int(np.ceil(n_areas / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False, sharey=True)

    n_sessions = df["session_id"].nunique()
    for i, area in enumerate(areas):
        ax = axes[i // ncols, i % ncols]
        sub = df[df["area_value"] == area]
        for half, ls in (("first", "-"), ("second", "--")):
            half_sub = sub[sub["half"] == half]
            for cohort in ("R+", "R-"):
                c_sub = half_sub[half_sub["reward_group"] == cohort]
                curves = [np.array(c) for c in c_sub["real_curve"]]
                for curve in curves:
                    ax.plot(bin_centers_ms, curve, color=COHORT_COLOR[cohort], alpha=0.35, lw=1.2, linestyle=ls)
                if len(curves) >= 2:
                    mean_curve = np.nanmean(np.stack(curves), axis=0)
                    ax.plot(bin_centers_ms, mean_curve, color=COHORT_COLOR[cohort], alpha=0.95, lw=2.2, linestyle=ls)

        ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
        ax.axvspan(-5, 5, color="#dddddd", alpha=0.6, zorder=0)  # dead-zone gap
        ax.set_title(f"{area}  (n={sub['session_id'].nunique()} sess)", fontsize=10)
        ax.set_ylim(0.35, 1.0)
        ax.set_xlabel("time from start_time (ms)", fontsize=9)
        ax.set_ylabel("balanced accuracy", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n_areas, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    handles = [
        plt.Line2D([0], [0], color=COHORT_COLOR["R+"], lw=2.2, label="R+ (mean)"),
        plt.Line2D([0], [0], color=COHORT_COLOR["R-"], lw=2.2, label="R- (mean)"),
        plt.Line2D([0], [0], color="black", lw=1.2, linestyle="-", label="first half"),
        plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="second half"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Whisker hit/miss decoding, learning stage, area_group -- pilot progress ({n_sessions} sessions so far)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path = OUT_DIR / "003_pilot_progress.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path} ({n_sessions} sessions, {len(df)} rows)")


if __name__ == "__main__":
    main()
