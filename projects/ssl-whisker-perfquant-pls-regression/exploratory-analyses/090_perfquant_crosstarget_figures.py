"""Figures for `089`'s cross-target specificity + PLS weight-vector
comparison (user: "Make figures of these"). Pure post-hoc on `089`'s
already-computed CSVs, no new compute.

Three figures:
1. **Specificity matrix**: 3x3 heatmap of mean above-null Pearson
   (train_target predicting test_target), diagonal boxed. Per-cell
   one-sample Wilcoxon vs 0 across the 12 pilot sessions, BH-FDR across
   all 9 cells.
2. **Paired diagonal-vs-cross-target**: one line per (session, train_
   target) -- own-target above-null vs that row's mean cross-target
   above-null, 36 paired points (12 sessions x 3 targets). Group-level
   Wilcoxon + t-test on the paired difference.
3. **Weight-vector cosine similarity**: strip plot, one panel per target
   pair (3 panels), 12 sessions each, one-sample Wilcoxon + t-test vs 0,
   BH-FDR across the 3 pairs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
TARGET_COLOR = {"whisker_curve": "#1f77b4", "falsealarm_curve": "#ff7f0e", "performance_curve": "#2ca02c"}
TARGET_SHORT = {"whisker_curve": "whisker", "falsealarm_curve": "falsealarm", "performance_curve": "performance"}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def stars(q):
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else ""))


def main():
    df = pd.read_csv(OUT_DIR / "089_perfquant_crosstarget_specificity.csv")
    cos_df = pd.read_csv(OUT_DIR / "089_perfquant_crosstarget_weightcosine.csv")
    n_sessions = df["session_id"].nunique()

    # --- per-cell group test (above_null vs 0), 9 cells, BH-FDR ---
    cell_rows = []
    for train_target in TARGETS:
        for test_target in TARGETS:
            sub = df[(df.train_target == train_target) & (df.test_target == test_target)]
            vals = sub["above_null"].to_numpy()
            wp = wilcoxon(vals).pvalue if len(vals) >= 5 else np.nan
            tp = ttest_1samp(vals, 0.0).pvalue
            cell_rows.append(dict(train_target=train_target, test_target=test_target,
                                   mean_above_null=vals.mean(), wilcoxon_p=wp, ttest_p=tp))
    cell_df = pd.DataFrame(cell_rows)
    cell_df["wilcoxon_q"] = bh_fdr(cell_df["wilcoxon_p"].to_numpy())
    cell_df["ttest_q"] = bh_fdr(cell_df["ttest_p"].to_numpy())
    cell_df.to_csv(OUT_DIR / "090_perfquant_crosstarget_cellstats.csv", index=False)

    # ================= Figure 1: specificity matrix heatmap =================
    mat = cell_df.pivot(index="train_target", columns="test_target", values="mean_above_null").loc[TARGETS, TARGETS]
    qmat = cell_df.pivot(index="train_target", columns="test_target", values="wilcoxon_q").loc[TARGETS, TARGETS]
    fig1, ax1 = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    vmax = np.abs(mat.to_numpy()).max()
    im = ax1.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(3):
        for j in range(3):
            val, q = mat.iloc[i, j], qmat.iloc[i, j]
            txt = f"{val:+.3f}\n{stars(q)}" if stars(q) else f"{val:+.3f}"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=11,
                     color="white" if abs(val) > vmax * 0.55 else "black", fontweight="bold" if i == j else "normal")
            if i == j:
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
    ax1.set_xticks(range(3)); ax1.set_xticklabels([TARGET_SHORT[t] for t in TARGETS], fontsize=10)
    ax1.set_yticks(range(3)); ax1.set_yticklabels([TARGET_SHORT[t] for t in TARGETS], fontsize=10)
    ax1.set_xlabel("predicting target (true curve)", fontsize=10)
    ax1.set_ylabel("decoder trained on target", fontsize=10)
    plt.colorbar(im, ax=ax1, label="mean above-null Pearson r", shrink=0.85)
    ax1.set_title(f"Cross-target specificity matrix (n={n_sessions} pilot sessions)\n"
                  f"boxed = own target; '*' = BH-FDR q<0.05 vs 0 (Wilcoxon)", fontsize=11)
    fig1_path = OUT_DIR / "090_perfquant_crosstarget_matrix.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
    fig1.savefig(fig1_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig1_path.name} (+ .pdf)")

    # ================= Figure 2: paired diagonal vs cross-target =================
    pair_rows = []
    for (mouse, sid, train_target), g in df.groupby(["mouse", "session_id", "train_target"]):
        diag = g[g.is_diagonal]["above_null"].iloc[0]
        cross = g[~g.is_diagonal]["above_null"].mean()
        pair_rows.append(dict(mouse=mouse, session_id=sid, train_target=train_target, diag=diag, cross=cross))
    pair_df = pd.DataFrame(pair_rows)
    diff = pair_df["diag"] - pair_df["cross"]
    wp_pair = wilcoxon(diff).pvalue
    tp_pair = ttest_1samp(diff, 0.0).pvalue

    fig2, ax2 = plt.subplots(figsize=(5.6, 6), constrained_layout=True)
    x = np.array([0, 1])
    for _, row in pair_df.iterrows():
        color = TARGET_COLOR[row["train_target"]]
        wins = row["diag"] > row["cross"]
        ax2.plot(x, [row["cross"], row["diag"]], color=color, lw=1.6 if wins else 1.0,
                 alpha=0.85 if wins else 0.4, linestyle="-" if wins else "--", marker="o", markersize=5)
    ax2.axhline(0, color="#888888", lw=0.7, linestyle=":")
    ax2.set_xticks(x); ax2.set_xticklabels(["mean cross-target", "own target"], fontsize=11)
    ax2.set_ylabel("above-null Pearson r", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    n_wins = int((pair_df["diag"] > pair_df["cross"]).sum())
    handles = [plt.Line2D([0], [0], color=TARGET_COLOR[t], lw=2, label=TARGET_SHORT[t]) for t in TARGETS]
    ax2.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax2.set_title(f"Own target beats mean cross-target: {n_wins}/{len(pair_df)} (session x train_target) rows\n"
                  f"paired Wilcoxon p={wp_pair:.3g}, paired t-test p={tp_pair:.3g}", fontsize=10)
    fig2_path = OUT_DIR / "090_perfquant_crosstarget_paired.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    fig2.savefig(fig2_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig2_path.name} (+ .pdf)")

    # ================= Figure 3: weight-vector cosine similarity =================
    pairs = [("whisker_curve", "falsealarm_curve"), ("whisker_curve", "performance_curve"),
             ("falsealarm_curve", "performance_curve")]
    cos_stats = []
    for ta, tb in pairs:
        vals = cos_df[(cos_df.target_a == ta) & (cos_df.target_b == tb)]["cosine"].to_numpy()
        wp = wilcoxon(vals).pvalue if len(vals) >= 5 else np.nan
        tp = ttest_1samp(vals, 0.0).pvalue
        cos_stats.append(dict(target_a=ta, target_b=tb, mean=vals.mean(), wilcoxon_p=wp, ttest_p=tp))
    cos_stats_df = pd.DataFrame(cos_stats)
    cos_stats_df["wilcoxon_q"] = bh_fdr(cos_stats_df["wilcoxon_p"].to_numpy())
    cos_stats_df["ttest_q"] = bh_fdr(cos_stats_df["ttest_p"].to_numpy())
    cos_stats_df.to_csv(OUT_DIR / "090_perfquant_crosstarget_weightcosine_stats.csv", index=False)

    fig3, axes3 = plt.subplots(1, 3, figsize=(4.2 * 3, 4.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (ta, tb) in zip(axes3, pairs):
        vals = cos_df[(cos_df.target_a == ta) & (cos_df.target_b == tb)]["cosine"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(jitter, vals, s=30, color="#555555", alpha=0.6, edgecolors="none", zorder=2)
        mean, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(0, mean, yerr=sem, fmt="D", color="#d62728", markersize=8, capsize=4, zorder=3)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        row = cos_stats_df[(cos_stats_df.target_a == ta) & (cos_stats_df.target_b == tb)].iloc[0]
        ax.set_title(f"{TARGET_SHORT[ta]} vs {TARGET_SHORT[tb]}\n"
                     f"mean={mean:+.3f}, Wilcoxon q={row['wilcoxon_q']:.3g} {stars(row['wilcoxon_q'])}", fontsize=9.5)
        ax.set_xticks([]); ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-1, 1)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    axes3[0].set_ylabel("PLS weight-vector cosine similarity\n(per session, n=12)", fontsize=9)
    fig3.suptitle("PLS weight-vector overlap across target pairs -- shared (near +-1) vs distinct (near 0) neural axis", fontsize=11)
    fig3_path = OUT_DIR / "090_perfquant_crosstarget_weightcosine.png"
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    fig3.savefig(fig3_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig3_path.name} (+ .pdf)")

    print("\n=== cell-level stats (specificity matrix) ===")
    print(cell_df.to_string(index=False))
    print(f"\n=== paired diagonal-vs-cross-target ===\nn_wins={n_wins}/{len(pair_df)}, "
          f"Wilcoxon p={wp_pair:.3g}, t-test p={tp_pair:.3g}")
    print("\n=== weight-vector cosine stats ===")
    print(cos_stats_df.to_string(index=False))
    print("DONE_090")


if __name__ == "__main__":
    main()
