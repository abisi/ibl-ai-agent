"""Violin-plot comparison across windows x real/null control x cohort, for
`052`'s single-animal per-neuron multi-head regression (user request
2026-09-20: "Compare with mean of null distributions for both sensory and
baseline. Should the baseline be smaller? Then plot distributions e.g.
violinplots of windows, controls, cohorts and statistical annotations.").

Purely post-hoc -- reads `052`'s two saved per-session summary CSVs
(sensory-window and baseline-window runs), no new decoding.

Per target: one figure, 2 panels (sensory, baseline), each showing 4
violins (R+ real, R+ null, R- real, R- null) for Pearson r. Statistical
annotations: Wilcoxon signed-rank (paired, real vs null within a cohort --
same sessions) and, matching this project's established R+/R- convention,
both Mann-Whitney U (non-parametric) and Welch's t (parametric) for the
R+-real vs R--real cohort comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson", "spearman"]  # user 2026-09-20: "Make figures" -- extend from pearson-only to all three
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_bracket(ax, x1, x2, y, step, text, color="black"):
    """`step` is an ADDITIVE offset (a fraction of the panel's own data
    range), not multiplicative -- multiplicative offsets (y*1.03) break
    down for R^2, whose values can be very negative or near zero, unlike
    Pearson/Spearman's roughly 0-1 range this was first written for."""
    ax.plot([x1, x1, x2, x2], [y, y + step, y + step, y], lw=1, color=color)
    ax.text((x1 + x2) / 2, y + step * 1.15, text, ha="center", va="bottom", fontsize=8, color=color)


def violin_panel(ax, df, target, window_label, metric):
    real_col, null_col = f"{target}_real_{metric}", f"{target}_null_{metric}_mean"
    groups, positions, colors, alphas = [], [], [], []
    labels = []
    x = 0
    stats_lines = []
    pending_brackets = []  # (x1, x2, top_of_these_two_groups, text)
    for cohort in ("R+", "R-"):
        sub = df[df["reward_group"] == cohort]
        real_vals = sub[real_col].dropna().to_numpy()
        null_vals = sub[null_col].dropna().to_numpy()
        groups += [real_vals, null_vals]
        positions += [x, x + 0.8]
        colors += [COHORT_COLORS[cohort], COHORT_COLORS[cohort]]
        alphas += [0.75, 0.35]
        labels += [f"{cohort}\nreal", f"{cohort}\nnull"]
        # paired real-vs-null within this cohort (same sessions, matched order)
        paired = sub[[real_col, null_col]].dropna()
        if len(paired) >= 6:
            try:
                stat, p = wilcoxon(paired[real_col], paired[null_col])
                stats_lines.append(f"{cohort} real vs null (Wilcoxon): p={p:.4f} ({sig_stars(p)})")
                pending_brackets.append((x, x + 0.8, max(real_vals.max(), null_vals.max()), sig_stars(p), "black"))
            except ValueError:
                pass
        x += 2.2

    vp = ax.violinplot(groups, positions=positions, widths=0.7, showmeans=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(alphas[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    rng = np.random.default_rng(0)
    for pos, g, c in zip(positions, groups, colors):
        jitter = rng.uniform(-0.08, 0.08, size=len(g))
        ax.scatter(np.full(len(g), pos) + jitter, g, color="black", s=8, alpha=0.5, zorder=3)

    # cohort comparison on the REAL values
    rplus_real = df[df["reward_group"] == "R+"][real_col].dropna()
    rminus_real = df[df["reward_group"] == "R-"][real_col].dropna()
    if len(rplus_real) >= 2 and len(rminus_real) >= 2:
        t_stat, t_p = ttest_ind(rplus_real, rminus_real, equal_var=False)
        u_stat, u_p = mannwhitneyu(rplus_real, rminus_real, alternative="two-sided")
        stats_lines.append(f"R+ vs R- real (Welch t): p={t_p:.4f} ({sig_stars(t_p)}); (Mann-Whitney U): p={u_p:.4f} ({sig_stars(u_p)})")
        pending_brackets.append((positions[0], positions[2], max(np.concatenate(groups)), f"MWU {sig_stars(u_p)}", "#444444"))

    # Draw brackets using a step SCALED to this panel's own data range --
    # a fixed additive/multiplicative offset breaks down for R^2, whose
    # values can be very negative or near zero, unlike Pearson/Spearman's
    # roughly 0-1 range.
    all_vals = np.concatenate(groups)
    y_range = max(all_vals.max() - all_vals.min(), 1e-6)
    step = 0.06 * y_range
    # sort so the widest (cohort-comparison) bracket is drawn highest, tallest first
    for i, (x1, x2, top, text, color) in enumerate(sorted(pending_brackets, key=lambda b: b[1] - b[0])):
        annotate_bracket(ax, x1, x2, top + step * (0.3 + i * 1.3), step, text, color=color)

    ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{metric} r" if metric != "r2" else "R2")
    ax.set_title(f"{target}: {window_label}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(all_vals.min() - step, all_vals.max() + step * (0.3 + len(pending_brackets) * 1.3) + step * 2)
    return stats_lines


def main():
    sens = pd.read_csv(OUT_DIR / "052_perfquant_singleanimal_multihead_test.sensorywindow.csv")
    base = pd.read_csv(OUT_DIR / "052_perfquant_singleanimal_multihead_test.csv")
    print(f"sensory n={len(sens)} sessions, baseline n={len(base)} sessions")

    # User 2026-09-20: "make all stats figures fit on one grid" -- one
    # combined figure instead of 3 separate per-metric PNGs. Rows =
    # target x metric (9), columns = window (2).
    n_rows = len(TARGETS) * len(METRICS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.2 * n_rows), constrained_layout=True)
    row = 0
    for target in TARGETS:
        for metric in METRICS:
            print(f"\n=== {target} ({metric}): real vs null means, sensory vs baseline ===")
            rc, nc = f"{target}_real_{metric}", f"{target}_null_{metric}_mean"
            print(f"  sensory real={sens[rc].mean():.3f} null={sens[nc].mean():.3f} | "
                  f"baseline real={base[rc].mean():.3f} null={base[nc].mean():.3f} | "
                  f"{'baseline null > sensory null' if base[nc].mean() > sens[nc].mean() else 'baseline null <= sensory null'}")
            for col_i, (df, window_label) in enumerate([(sens, "sensory (5-50ms post-stim)"), (base, "baseline (-2000/-10ms pre-stim)")]):
                ax = axes[row][col_i]
                lines = violin_panel(ax, df, target, window_label, metric)
                for line in lines:
                    print(f"  [{window_label}] {line}")
            row += 1

    fig_path = OUT_DIR / "054_perfquant_singleanimal_violin_comparison_all.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")

    # --- companion figure: mouse-level (real - null) DIFFERENCE distribution ---
    # user 2026-09-20: "also plot the mouse-level difference between real
    # and null value, and distribution of that." This is the same paired
    # quantity the Wilcoxon test above already operates on, made visible
    # directly rather than only reported as a p-value.
    fig_diff, axes_diff = plt.subplots(n_rows, 2, figsize=(9, 3.0 * n_rows), constrained_layout=True)
    row = 0
    for target in TARGETS:
        for metric in METRICS:
            for col_i, (df, window_label) in enumerate([(sens, "sensory"), (base, "baseline")]):
                ax = axes_diff[row][col_i]
                real_col, null_col = f"{target}_real_{metric}", f"{target}_null_{metric}_mean"
                groups, positions, colors = [], [], []
                x = 0
                for cohort in ("R+", "R-"):
                    sub = df[df["reward_group"] == cohort]
                    diff = (sub[real_col] - sub[null_col]).dropna().to_numpy()
                    groups.append(diff)
                    positions.append(x)
                    colors.append(COHORT_COLORS[cohort])
                    x += 1.2
                vp = ax.violinplot(groups, positions=positions, widths=0.9, showmeans=True, showextrema=False)
                for i, body in enumerate(vp["bodies"]):
                    body.set_facecolor(colors[i])
                    body.set_alpha(0.55)
                    body.set_edgecolor("black")
                    body.set_linewidth(0.6)
                rng = np.random.default_rng(1)
                for pos, g, c in zip(positions, groups, colors):
                    if len(g) == 0:
                        continue
                    jitter = rng.uniform(-0.18, 0.18, size=len(g))
                    ax.scatter(np.full(len(g), pos) + jitter, g, color="black", s=8, alpha=0.4, zorder=3)
                    if len(g) >= 6:
                        try:
                            _, p = wilcoxon(g)
                            y_text = g.max() + 0.05 * max(abs(g.max()), abs(g.min()), 1e-6)
                            ax.text(pos, y_text, sig_stars(p), ha="center", fontsize=8)
                        except ValueError:
                            pass
                ax.axhline(0, color="#888888", lw=1, linestyle=":")
                ax.set_xticks(positions)
                ax.set_xticklabels(["R+", "R-"], fontsize=8)
                ax.set_ylabel(f"real - null\n({metric})", fontsize=8)
                ax.set_title(f"{target}: {window_label}", fontsize=9)
                ax.spines[["top", "right"]].set_visible(False)
            row += 1

    diff_fig_path = OUT_DIR / "054_perfquant_singleanimal_diff_distribution.png"
    fig_diff.savefig(diff_fig_path, dpi=150)
    print(f"saved {diff_fig_path.name}")


if __name__ == "__main__":
    main()
