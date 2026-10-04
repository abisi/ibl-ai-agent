"""PLS+1SE-only figures (user: "Forget about RRR for now, update figures
with the PLS+1SE results" -- RRR's above-null advantage is still an open
question, see the "is RRR's null unfairly bad" discussion; shelving RRR
for now rather than resolving it here). Pure post-hoc from `083`'s
already-computed cache/CSV (baseline/ITI window, same 3 pilot sessions),
no new compute -- just re-plotting with RRR dropped.

1. **Curves**: same 9-panel true-vs-predicted grid, PLS+1SE (real) and
   PLS+LGAR1 (smoothed) only, annotated with real-r/null/above-null for
   both.
2. **Summary**: paired PLS-vs-PLS+LGAR1 above-null, one line per
   (mouse, target) cell -- shows directly whether LG-AR1 smoothing helps
   PLS's own output (per §21/§23: mixed, mostly small effect either way).
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

    # --- Figure 1: PLS-only annotated curves ---
    n_sessions = len(results)
    fig1, axes1 = plt.subplots(n_sessions, len(TARGETS), figsize=(4.6 * len(TARGETS), 3.6 * n_sessions),
                                constrained_layout=True, squeeze=False)
    for row_i, res in enumerate(results):
        for col_i, target in enumerate(TARGETS):
            ax = axes1[row_i][col_i]
            t = res["targets"][target]
            trial_idx = np.arange(len(t["y"]))
            ax.plot(trial_idx, t["y"], color="#000000", lw=1.5, label="true", zorder=5)
            ax.plot(trial_idx, t["d_pls_obs"], color="#1f77b4", lw=0.8, alpha=0.55, label="PLS+1SE")
            ax.plot(trial_idx, t["pls_smoothed_curve_obs"], color="#1f77b4", lw=1.8, linestyle="--", label="PLS+LGAR1")
            row = df[(df.mouse == res["mouse"]) & (df.target == target)].iloc[0]
            ax.set_title(
                f"{res['mouse']} -- {target}\n"
                f"PLS above-null={row['pls_above_null']:+.3f} (r={row['pls_real']:.2f}, null={row['pls_null_mean']:.2f})\n"
                f"PLS+LGAR1 above-null={row['pls_lgar1_above_null']:+.3f}",
                fontsize=8)
            ax.tick_params(labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig1.suptitle("PLS+1SE: true vs predicted (baseline/ITI window), annotated", fontsize=12, y=1.02)
    fig1_path = OUT_DIR / "085_pls_only_curves.png"
    fig1.savefig(fig1_path, dpi=140)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: paired PLS vs PLS+LGAR1 ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    x = np.array([0, 1])
    for _, row in df.iterrows():
        color = TARGET_COLOR[row["target"]]
        y_vals = [row["pls_above_null"], row["pls_lgar1_above_null"]]
        helps = row["pls_lgar1_above_null"] > row["pls_above_null"]
        ax2.plot(x, y_vals, color=color, lw=1.6 if helps else 1.0, alpha=0.9 if helps else 0.5,
                 marker="o", markersize=6, linestyle="-" if helps else "--")
    ax2.axhline(0, color="#888888", lw=0.7, linestyle=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["PLS+1SE", "PLS+LGAR1"], fontsize=11)
    ax2.set_ylabel("above-null Pearson r", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], color=TARGET_COLOR[t], lw=2, label=t.replace("_curve", "")) for t in TARGETS]
    handles.append(plt.Line2D([0], [0], color="#888888", lw=1.6, linestyle="-", label="LGAR1 helps"))
    handles.append(plt.Line2D([0], [0], color="#888888", lw=1.0, linestyle="--", alpha=0.5, label="LGAR1 hurts"))
    ax2.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    n_helps = int((df["pls_lgar1_above_null"] > df["pls_above_null"]).sum())
    mean_pls, mean_pls_lgar1 = df["pls_above_null"].mean(), df["pls_lgar1_above_null"].mean()
    ax2.set_title(f"PLS+1SE vs PLS+LGAR1, above-null Pearson r, all 9 cells\n"
                   f"LGAR1 helps {n_helps}/9 -- mean PLS={mean_pls:.3f}, mean PLS+LGAR1={mean_pls_lgar1:.3f}", fontsize=10)
    fig2_path = OUT_DIR / "085_pls_vs_pls_lgar1_paired.png"
    fig2.savefig(fig2_path, dpi=140)
    print(f"saved {fig2_path.name}")
    print("DONE_085")


if __name__ == "__main__":
    main()
