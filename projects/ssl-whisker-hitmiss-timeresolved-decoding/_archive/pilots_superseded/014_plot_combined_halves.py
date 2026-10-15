"""Re-plot of the report's two headline curve figures (start_time_learning,
lick_time_learning; entire scope, area_group scheme) with both session
halves overlaid on the same panel per area, instead of separate columns --
solid = first half, dashed = second half, so the two halves can be compared
directly at a glance. No significance shading here (that's already in the
existing two-column figures and the stats heatmaps); this is a pure
side-by-side comparison view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

OUT_DIR = Path(__file__).resolve().parent
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
DATASETS = {
    "start_time_learning": dict(parquet="002_pilot_results_partial.parquet", bin_edges="002_bin_edges.json",
                                 xlabel="time from start_time (ms)", title="Stimulus-locked, learning stage"),
    "lick_time_learning": dict(parquet="008_lick_aligned_results_partial.parquet", bin_edges="008_bin_edges.json",
                                xlabel="time from lick_time (ms)", title="Lick-locked, learning stage"),
}
N_AREAS = 7


def bin_centers_ms_for(bin_edges_name: str) -> np.ndarray:
    edges = json.loads((OUT_DIR / bin_edges_name).read_text())
    return np.array([(b[0] + b[1]) / 2 * 1000 for b in edges])


def main():
    for name, cfg in DATASETS.items():
        df = pd.read_parquet(OUT_DIR / cfg["parquet"])
        df = df[(df.area_col == "area_group") & df.skipped_reason.isna()]
        bin_centers_ms = bin_centers_ms_for(cfg["bin_edges"])

        areas = df.groupby("area_value")["session_id"].nunique().sort_values(ascending=False).head(N_AREAS).index.tolist()
        ncols = 3
        nrows = int(np.ceil(len(areas) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)

        for i, area in enumerate(areas):
            ax = axes[i // ncols, i % ncols]
            sub = df[df.area_value == area]
            for half, ls, alpha_mean in (("first", "-", 1.0), ("second", "--", 1.0)):
                half_sub = sub[sub.half == half]
                for cohort in ("R+", "R-"):
                    curves = [np.array(c) for c in half_sub.loc[half_sub.reward_group == cohort, "real_curve"]]
                    if len(curves) < 2:
                        continue
                    mean_curve = np.nanmean(np.stack(curves), axis=0)
                    sem_curve = np.nanstd(np.stack(curves), axis=0) / np.sqrt(len(curves))
                    ax.plot(bin_centers_ms, mean_curve, color=COHORT_COLOR[cohort], lw=2, linestyle=ls, alpha=alpha_mean)
                    ax.fill_between(bin_centers_ms, mean_curve - sem_curve, mean_curve + sem_curve,
                                     color=COHORT_COLOR[cohort], alpha=0.12, lw=0)
            ax.axhline(0.5, color="#999999", lw=1, linestyle=":")
            ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4)
            ax.set_title(area, fontsize=10)
            ax.set_xlabel(cfg["xlabel"], fontsize=8.5)
            ax.set_ylabel("balanced accuracy", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)

        for j in range(len(areas), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        handles = [
            plt.Line2D([0], [0], color=COHORT_COLOR["R+"], lw=2, label="R+"),
            plt.Line2D([0], [0], color=COHORT_COLOR["R-"], lw=2, label="R-"),
            plt.Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="first half"),
            plt.Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="second half"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{cfg['title']} -- both halves overlaid per area", fontsize=12)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        out_path = OUT_DIR / f"014_{name}_combined_halves.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
