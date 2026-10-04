"""Two follow-up figures for `083`'s matched PLS+1SE vs RRR comparison
(user: "Make a figure and annotate curve figures") -- pure post-hoc from
`083`'s already-computed cache/CSV, no new compute.

1. **Annotated curves**: same 9-panel true-vs-predicted grid as
   `083_pls_rrr_lgar1_curves.png`, but each panel's title now states the
   actual real-r / above-null numbers for PLS and RRR directly, instead
   of requiring a cross-reference to the separate bar chart or CSV.
2. **Paired comparison**: a single compact figure with one connected
   line per (mouse, target) cell (9 lines total), PLS above-null on the
   left, RRR above-null on the right -- the quantitative table from the
   PLS-vs-RRR comparison, as one figure, matching this project's
   established paired-dot convention (`046`'s `_paired_panel`, `076`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
TARGET_COLOR = {"whisker_curve": "#1f77b4", "falsealarm_curve": "#ff7f0e", "performance_curve": "#2ca02c"}


def main():
    with open(OUT_DIR / "083_pls_rrr_lgar1_cache.pkl", "rb") as f:
        results = pickle.load(f)
    df = pd.read_csv(OUT_DIR / "083_pls_rrr_lgar1_comparable_summary.csv")

    # --- Figure 1: annotated curves ---
    n_sessions = len(results)
    fig1, axes1 = plt.subplots(n_sessions, len(TARGETS), figsize=(5.2 * len(TARGETS), 3.9 * n_sessions),
                                constrained_layout=True, squeeze=False)
    for row_i, res in enumerate(results):
        for col_i, target in enumerate(TARGETS):
            ax = axes1[row_i][col_i]
            t = res["targets"][target]
            trial_idx = np.arange(len(t["y"]))
            ax.plot(trial_idx, t["y"], color="#000000", lw=1.4, label="true", zorder=5)
            ax.plot(trial_idx, t["d_pls_obs"], color="#1f77b4", lw=0.7, alpha=0.5, label="PLS+1SE")
            ax.plot(trial_idx, t["d_rrr_obs"], color="#ff7f0e", lw=0.7, alpha=0.5, label="RRR")
            ax.plot(trial_idx, t["pls_smoothed_curve_obs"], color="#1f77b4", lw=1.6, linestyle="--", label="PLS+LGAR1")
            ax.plot(trial_idx, t["rrr_smoothed_curve_obs"], color="#ff7f0e", lw=1.6, linestyle="--", label="RRR+LGAR1")
            row = df[(df.mouse == res["mouse"]) & (df.target == target)].iloc[0]
            winner = "RRR" if row["rrr_above_null"] > row["pls_above_null"] else "PLS"
            ax.set_title(
                f"{res['mouse']} -- {target}  [{winner} wins]\n"
                f"PLS above-null={row['pls_above_null']:+.3f} (r={row['pls_real']:.2f}, null={row['pls_null_mean']:.2f})\n"
                f"RRR above-null={row['rrr_above_null']:+.3f} (r={row['rrr_real']:.2f}, null={row['rrr_null_mean']:.2f})",
                fontsize=7.5, color="#006600" if winner == "RRR" else "#663300", loc="center")
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    fig1.suptitle("PLS+1SE vs RRR (+/- LG-AR1): true vs predicted, annotated with real-r and above-null scores",
                   fontsize=12, y=1.02)
    fig1_path = OUT_DIR / "084_pls_rrr_curves_annotated.png"
    fig1.savefig(fig1_path, dpi=140)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: paired comparison, one line per (mouse, target) cell ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    x = np.array([0, 1])
    for _, row in df.iterrows():
        color = TARGET_COLOR[row["target"]]
        y_vals = [row["pls_above_null"], row["rrr_above_null"]]
        wins = row["rrr_above_null"] > row["pls_above_null"]
        ax2.plot(x, y_vals, color=color, lw=1.6 if wins else 1.0, alpha=0.9 if wins else 0.5,
                 marker="o", markersize=6, linestyle="-" if wins else "--")
    ax2.axhline(0, color="#888888", lw=0.7, linestyle=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["PLS+1SE", "RRR"], fontsize=11)
    ax2.set_ylabel("above-null Pearson r", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], color=TARGET_COLOR[t], lw=2, label=t.replace("_curve", "")) for t in TARGETS]
    handles.append(plt.Line2D([0], [0], color="#888888", lw=1.6, linestyle="-", label="RRR wins"))
    handles.append(plt.Line2D([0], [0], color="#888888", lw=1.0, linestyle="--", alpha=0.5, label="PLS wins"))
    ax2.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    n_rrr_wins = int((df["rrr_above_null"] > df["pls_above_null"]).sum())
    mean_pls, mean_rrr = df["pls_above_null"].mean(), df["rrr_above_null"].mean()
    ax2.set_title(f"PLS vs RRR, above-null Pearson r, all 9 (mouse x target) cells\n"
                   f"RRR wins {n_rrr_wins}/9 -- mean PLS={mean_pls:.3f}, mean RRR={mean_rrr:.3f}", fontsize=10)
    fig2_path = OUT_DIR / "084_pls_vs_rrr_paired.png"
    fig2.savefig(fig2_path, dpi=140)
    print(f"saved {fig2_path.name}")
    print("DONE_084")


if __name__ == "__main__":
    main()
