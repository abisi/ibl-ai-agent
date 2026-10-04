"""Fuller figures for `071`'s within-mouse condition x cohort test (user
request 2026-09-23: "Make figures" -- `071`'s own figure only showed
Pearson means +/- SEM as bars, no individual mice, no other metrics).
Recomputes the per-mouse aggregation directly from `058`/`070`'s CSVs
(self-contained, this project's established per-script convention --
`042`/`047`) rather than importing from `071`.

Two figures:
1. **Per-mouse strip plot**, all 3 metrics, 4 cells (day_stage x cohort) x
   3 targets -- every mouse as its own jittered point (paired-panel style,
   `046`'s `_paired_panel` convention), mean +/- SEM as a black diamond,
   BH-FDR significance star from `071`'s own test. Shows the actual
   per-mouse spread `071`'s bar chart hid, including the expert-R- n=6
   cells where individual points make the low-power caveat visible rather
   than just stated.
2. **Heatmap**, mean above-null value x (day_stage, cohort) row x target
   column, one panel per metric (3 panels, not just Pearson like `067`'s
   single-metric heatmap) -- BH-FDR stars annotated.
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
METRICS = ["r2", "pearson", "spearman"]
COHORTS = ["R+", "R-"]
STAGES = ["learning", "expert"]
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}  # standing project convention


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


def _stars(q):
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))


def main():
    d_learning = pd.read_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv")
    d_learning["day_stage"] = "learning"
    d_expert = pd.read_csv(OUT_DIR / "070_perfquant_expert_pls_nulldist_fullpool.csv")
    d_expert["day_stage"] = "expert"
    df = pd.concat([d_learning, d_expert], ignore_index=True)

    agg_cols = {f"above_null_{m}": "mean" for m in METRICS}
    per_mouse = (df.groupby(["mouse", "reward_group", "day_stage", "target"])
                 .agg(agg_cols).reset_index())

    # Per-cell stats (Wilcoxon, same as 071) for star annotation, per metric.
    cells = []
    for stage in STAGES:
        for cohort in COHORTS:
            for target in TARGETS:
                g = per_mouse[(per_mouse.day_stage == stage) & (per_mouse.reward_group == cohort)
                               & (per_mouse.target == target)]
                rec = dict(day_stage=stage, cohort=cohort, target=target, n_mice=len(g))
                for metric in METRICS:
                    vals = g[f"above_null_{metric}"].dropna().values
                    rec[f"{metric}_vals"] = vals
                    if len(vals) >= 5:
                        try:
                            rec[f"{metric}_p"] = wilcoxon(vals).pvalue
                        except ValueError:
                            rec[f"{metric}_p"] = np.nan
                    else:
                        rec[f"{metric}_p"] = np.nan
                cells.append(rec)
    cells_df = pd.DataFrame(cells)
    for metric in METRICS:
        mask = cells_df[f"{metric}_p"].notna()
        q = np.full(len(cells_df), np.nan)
        q[mask.values] = bh_fdr(cells_df.loc[mask, f"{metric}_p"].values)
        cells_df[f"{metric}_q"] = q

    # --- Figure 1: per-mouse strip plot, rows=metric, cols=(day_stage,cohort) pairs grouped by target on x ---
    cell_order = [(s, c) for s in STAGES for c in COHORTS]
    fig1, axes1 = plt.subplots(len(METRICS), len(cell_order), figsize=(3.6 * len(cell_order), 3.2 * len(METRICS)),
                                constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for row_i, metric in enumerate(METRICS):
        for col_i, (stage, cohort) in enumerate(cell_order):
            ax = axes1[row_i][col_i]
            color = COHORT_COLOR[cohort]
            all_vals = []
            for x, target in enumerate(TARGETS):
                row = cells_df[(cells_df.day_stage == stage) & (cells_df.cohort == cohort)
                                & (cells_df.target == target)].iloc[0]
                vals = row[f"{metric}_vals"]
                all_vals.append(vals)
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=16, color=color, alpha=0.6,
                           edgecolors="none", zorder=2)
                mean, sem = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.errorbar(x, mean, yerr=sem, fmt="D", color="black", markersize=5, capsize=3, zorder=3)
                star = _stars(row[f"{metric}_q"])
                ymax = max(vals) if len(vals) else mean
                ax.text(x, ymax + 0.02, star, ha="center", va="bottom", fontsize=8)
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.set_xticks(range(len(TARGETS)))
            ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=7.5, rotation=15, ha="right")
            n_mice = cells_df[(cells_df.day_stage == stage) & (cells_df.cohort == cohort)]["n_mice"].iloc[0]
            if row_i == 0:
                ax.set_title(f"{stage} / {cohort}\n(n={n_mice} mice)", fontsize=9.5, color=color)
            if col_i == 0:
                ax.set_ylabel(f"above-null {metric}\n(per mouse)", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig1.suptitle("Within-mouse above-null scores, individual mice (* = BH-FDR q<0.05, Wilcoxon vs 0)", fontsize=12)
    fig1_path = OUT_DIR / "072_perfquant_withinmouse_permouse_strip.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: heatmap, one panel per metric, rows=(day_stage,cohort), cols=target ---
    row_labels = [f"{s}\n{c}" for s, c in cell_order]
    fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 0.6 * len(cell_order) + 1.6),
                                constrained_layout=True)
    for ax, metric in zip(axes2, METRICS):
        mat = np.full((len(cell_order), len(TARGETS)), np.nan)
        stars = np.full((len(cell_order), len(TARGETS)), "", dtype=object)
        for i, (stage, cohort) in enumerate(cell_order):
            for j, target in enumerate(TARGETS):
                row = cells_df[(cells_df.day_stage == stage) & (cells_df.cohort == cohort)
                                & (cells_df.target == target)].iloc[0]
                vals = row[f"{metric}_vals"]
                mat[i, j] = np.mean(vals) if len(vals) else np.nan
                stars[i, j] = _stars(row[f"{metric}_q"])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.22)
        for i in range(len(cell_order)):
            for j in range(len(TARGETS)):
                if stars[i, j] and stars[i, j] != "n.s.":
                    ax.text(j, i, stars[i, j], ha="center", va="center", fontsize=9, color="black")
        ax.set_yticks(range(len(cell_order)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=8.5, rotation=20, ha="right")
        ax.set_title(f"{metric} (* = q<0.05)", fontsize=10)
        fig2.colorbar(im, ax=ax, label=f"mean above-null {metric}", shrink=0.7)
    fig2_path = OUT_DIR / "072_perfquant_withinmouse_heatmap.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")
    print("DONE_072")


if __name__ == "__main__":
    main()
