"""Pairwise-line version of `058`'s Figure 1 (`058_perfquant_real_vs_null_
distributions.png`) -- user follow-up after asking which figure shows real
vs null: "Can you do the pairwise line figure". Pure post-hoc on `058`'s
already-computed CSV (`058_perfquant_pls_nulldist_fullpool.csv`, the
canonical learning-stage, full-pool 88-session PLS+1SE validation), no new
compute.

Same facet layout as `058`'s violin figure (rows=target, cols=metric), but
instead of violin distributions, one line per SESSION connecting that
session's own null value (mean across its 1000 shift-null shuffles) to its
own real (out-of-fold test) value for that metric -- makes the paired,
within-session real-vs-null comparison directly visible instead of only
the two marginal distributions. Solid/full-opacity line = real > null
("above null" for that session); dashed/faint = real <= null, same
win/lose convention as `084`/`085`'s paired-comparison figures elsewhere
in this project.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson", "spearman"]
METHOD = "PLS+1SE"  # the only method 058 still computes (plain PLS dropped 2026-09-20)
TARGET_COLOR = {"whisker_curve": "#1f77b4", "falsealarm_curve": "#ff7f0e", "performance_curve": "#2ca02c"}


def main():
    df = pd.read_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv")
    df = df[df.method == METHOD]

    fig, axes = plt.subplots(len(TARGETS), len(METRICS), figsize=(4.2 * len(METRICS), 3.6 * len(TARGETS)),
                              constrained_layout=True, squeeze=False)
    x = np.array([0, 1])
    for row_i, target in enumerate(TARGETS):
        sub = df[df.target == target]
        color = TARGET_COLOR[target]
        for col_i, metric in enumerate(METRICS):
            ax = axes[row_i][col_i]
            null_vals = sub[f"null_{metric}_mean"].to_numpy()
            real_vals = sub[f"test_{metric}"].to_numpy()
            n_wins = 0
            for null_v, real_v in zip(null_vals, real_vals):
                if np.isnan(null_v) or np.isnan(real_v):
                    continue
                wins = real_v > null_v
                n_wins += int(wins)
                ax.plot(x, [null_v, real_v], color=color, lw=1.1 if wins else 0.6,
                         alpha=0.55 if wins else 0.22, linestyle="-" if wins else "--",
                         marker="o", markersize=2.5, zorder=3 if wins else 1)
            n = int(np.sum(~np.isnan(null_vals) & ~np.isnan(real_vals)))
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(["null\n(mean)", "real\n(test)"], fontsize=8)
            ax.set_title(f"{target} -- {metric}\nreal > null: {n_wins}/{n} sessions", fontsize=9)
            ax.set_ylabel(metric, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{METHOD}, learning-stage full-pool (n={df['session_id'].nunique()} sessions) -- "
                 f"real vs null, one line per session", fontsize=12)
    fig_path = OUT_DIR / "086_perfquant_pls_nulldist_paired.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig_path.name} (+ .pdf)")
    print("DONE_086")


if __name__ == "__main__":
    main()
