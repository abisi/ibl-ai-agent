"""RHMI vs EHMI comparison, within each cohort separately (per user request
2026-08-22 -- not pooled across R+/R-), on the within-task (auditory-only)
day-0/learners data (012/013 output). Answers "does reward-history and
error-history modulation differ in magnitude/direction for the same units",
as a question distinct from the R+-vs-R- cohort tests in 013/014.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from importlib import import_module
plot_mod = import_module("006_plot_publication_figure")
INK_PRIMARY, INK_SECONDARY, AXIS_LINE = plot_mod.INK_PRIMARY, plot_mod.INK_SECONDARY, plot_mod.AXIS_LINE
style_ax, box_style, sig_label = plot_mod.style_ax, plot_mod.box_style, plot_mod.sig_label

from history_lib import run_paired_metric_tests

# dataviz skill categorical palette, slots 3/4 (aqua, yellow) -- distinct from the
# blue/orange already used for R+/R- cohort identity throughout this project
COLOR_RHMI = "#1baf7a"
COLOR_EHMI = "#eda100"

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/within_task_auditory_day0_learners.parquet")
RESULTS_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/015_paired_metric_results.json")
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")
TITLE_SUFFIX = "within task (auditory-only), day 0, learners only"


def plot_paired(df, results):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.0), dpi=200)
    for ax, cohort in zip(axes, ("R+", "R-")):
        res = results.get(cohort, {})
        if "error" in res:
            ax.text(0.5, 0.5, "insufficient paired data", transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=INK_SECONDARY)
            ax.set_title(cohort, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
            style_ax(ax)
            continue

        sub = df[df.reward_group == cohort]
        paired = sub[sub["rhmi_index"].notna() & sub["ehmi_index"].notna()]
        rhmi, ehmi = paired["rhmi_index"].to_numpy(), paired["ehmi_index"].to_numpy()

        bp = ax.boxplot([rhmi, ehmi], positions=[1, 2], widths=0.55, showfliers=True, patch_artist=False)
        box_style(bp, "#898781")
        for artist, color in zip(bp["boxes"], (COLOR_RHMI, COLOR_EHMI)):
            artist.set_color(color)
        for i, (data, color) in enumerate(zip((rhmi, ehmi), (COLOR_RHMI, COLOR_EHMI)), start=1):
            rng = np.random.default_rng(0)
            jitter = rng.normal(0, 0.06, size=len(data))
            ax.scatter(np.full(len(data), i) + jitter, data, s=5, color=color, alpha=0.18, linewidths=0, zorder=2)

        ax.axhline(0, color=AXIS_LINE, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"RHMI\n(n={len(rhmi)} units)", f"EHMI\n(n={len(ehmi)} units)"])

        subtitle = (f"mouse-level paired (n={res['n_mice']} mice): {sig_label(res['mouse_level_wilcoxon_p'])}\n"
                    f"unit-level paired: {sig_label(res['unit_level_wilcoxon_p'])}")
        ax.set_title(cohort, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
        ax.text(0.5, 1.085, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)
        style_ax(ax)

    axes[0].set_ylabel("Modulation index", color=INK_PRIMARY)
    fig.suptitle(f"RHMI vs EHMI, within cohort, {TITLE_SUFFIX}", fontsize=12.5, color=INK_PRIMARY, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = OUT_DIR / "015_rhmi_vs_ehmi_within_cohort.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    results = run_paired_metric_tests(IN_PATH, RESULTS_PATH)
    df = pd.read_parquet(IN_PATH)
    plot_paired(df, results)


if __name__ == "__main__":
    main()
