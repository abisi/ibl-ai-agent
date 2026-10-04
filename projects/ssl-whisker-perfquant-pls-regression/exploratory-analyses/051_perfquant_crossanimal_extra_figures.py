"""Additional illustrative figures for `050_perfquant_crossanimal_arealevel_
test.py`'s cross-animal generalization results (user request 2026-09-20:
"make many figures to illustrate"). Purely post-hoc -- reads `050`'s
already-saved per-subject CSV (`050_perfquant_crossanimal_arealevel_test.csv`),
does NOT re-run any neural decoding, so it's cheap to regenerate any time
`050`'s CSV is refreshed.

Three new figures, complementing `050`'s own grid/examples/metrics-comparison:
1. Real vs. null distribution histograms, overlaid, per target x metric --
   a population-level (not per-subject-dot) view of the same real-vs-null
   comparison `050`'s metrics-comparison figure shows per subject.
2. Cohort (R+ vs R-) box+strip comparison, per target x metric -- the
   distributional analog of the Welch/Mann-Whitney summary `050` already
   prints as text.
3. Fraction-of-subjects-above-null-plus-1sd bar chart, per target x
   metric x cohort -- an at-a-glance summary of the "how many sessions
   individually clear their own null" counts `050` prints as text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "050_perfquant_crossanimal_arealevel_test.csv"

TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson_r", "spearman_r"]
METRIC_LABELS = {"r2": "R2", "pearson_r": "Pearson r", "spearman_r": "Spearman rho"}
NULL_BASE = {"r2": "r2", "pearson_r": "pearson", "spearman_r": "spearman"}
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"loaded {len(df)} subject-target rows from {CSV_PATH.name} "
          f"({df['mouse'].nunique()} unique subjects, {df['target'].nunique()} targets)")

    # --- Figure 1: real vs null histograms, overlaid, per target x metric ---
    fig1, axes1 = plt.subplots(len(TARGETS), len(METRICS), figsize=(5 * len(METRICS), 3.4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        sub = df[df["target"] == target]
        for col_i, metric in enumerate(METRICS):
            ax = axes1[row_i][col_i]
            base = NULL_BASE[metric]
            real_vals = sub[metric].dropna()
            null_vals = sub[f"null_{base}_mean"].dropna()
            all_vals = np.concatenate([real_vals, null_vals])
            bins = np.linspace(all_vals.min(), all_vals.max(), 20) if len(all_vals) else 10
            ax.hist(null_vals, bins=bins, color="#888888", alpha=0.5, label="null (per-subject mean)", density=True)
            ax.hist(real_vals, bins=bins, color="#1f77b4", alpha=0.6, label="real", density=True)
            ax.axvline(real_vals.mean(), color="#1f77b4", lw=2, linestyle="-")
            ax.axvline(null_vals.mean(), color="#555555", lw=2, linestyle="--")
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.set_title(f"{target}: {METRIC_LABELS[metric]}", fontsize=9)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / "051_perfquant_real_vs_null_histograms.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: cohort box+strip comparison, per target x metric ---
    fig2, axes2 = plt.subplots(len(TARGETS), len(METRICS), figsize=(4 * len(METRICS), 3.4 * len(TARGETS)), constrained_layout=True)
    rng = np.random.default_rng(0)
    for row_i, target in enumerate(TARGETS):
        sub = df[df["target"] == target]
        for col_i, metric in enumerate(METRICS):
            ax = axes2[row_i][col_i]
            data = [sub[sub["reward_group"] == c][metric].dropna().to_numpy() for c in ("R+", "R-")]
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True)
            for patch, c in zip(bp["boxes"], ("R+", "R-")):
                patch.set_facecolor(COHORT_COLORS[c])
                patch.set_alpha(0.3)
            for i, (c, vals) in enumerate(zip(("R+", "R-"), data)):
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, color=COHORT_COLORS[c], s=20, zorder=3)
            ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["R+", "R-"])
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_title(f"{target}: {METRIC_LABELS[metric]}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig2_path = OUT_DIR / "051_perfquant_cohort_comparison_boxplots.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: fraction-above-null bar chart, per target x metric x cohort ---
    fig3, ax3 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bar_width = 0.12
    x_base = np.arange(len(TARGETS))
    offset = 0
    legend_handles = []
    for metric in METRICS:
        base = NULL_BASE[metric]
        for cohort in ("R+", "R-"):
            fracs = []
            for target in TARGETS:
                sub = df[(df["target"] == target) & (df["reward_group"] == cohort)]
                above = (sub[metric] > sub[f"null_{base}_mean"] + sub[f"null_{base}_std"]).sum()
                fracs.append(above / len(sub) if len(sub) else np.nan)
            color = COHORT_COLORS[cohort]
            hatch = None if metric == "r2" else ("//" if metric == "pearson_r" else "..")
            bars = ax3.bar(x_base + offset, fracs, width=bar_width, color=color, alpha=0.7, hatch=hatch,
                            edgecolor="black", linewidth=0.5, label=f"{METRIC_LABELS[metric]}, {cohort}")
            legend_handles.append(bars)
            offset += bar_width
    ax3.axhline(0.5, color="#888888", lw=0.8, linestyle=":")
    ax3.set_xticks(x_base + bar_width * 2.5)
    ax3.set_xticklabels(TARGETS)
    ax3.set_ylabel("fraction of subjects above null+1sd")
    ax3.set_title("Cross-animal generalization: fraction of subjects individually exceeding their own null")
    ax3.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "051_perfquant_fraction_above_null_bars.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")


if __name__ == "__main__":
    main()
