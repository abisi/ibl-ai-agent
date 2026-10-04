"""Plots for the proportion/distribution tests (010_distribution_proportion_tests.py),
day 0 / learners only. Reuses 006_plot_publication_figure.py's palette and
style helpers for visual consistency.

- ECDF plot: R+ vs R- empirical CDFs per metric, with the KS statistic (max
  vertical gap between curves) and both the naive neuron-level and
  mouse-block-permutation p-values annotated.
- Proportion plot: per-mouse fraction of index>0 units, R+ vs R-, box+jitter.
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
COLOR_RPLUS, COLOR_RMINUS = plot_mod.COLOR_RPLUS, plot_mod.COLOR_RMINUS
INK_PRIMARY, INK_SECONDARY, AXIS_LINE = plot_mod.INK_PRIMARY, plot_mod.INK_SECONDARY, plot_mod.AXIS_LINE
style_ax, box_style, sig_label = plot_mod.style_ax, plot_mod.box_style, plot_mod.sig_label

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/day0_learners_unit_indices.parquet")
RESULTS_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/010_distribution_proportion_results.json")
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")
TITLE_SUFFIX = "day 0, learners only"


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_ecdf(df, results, out_dir=OUT_DIR, prefix="011", title_suffix=TITLE_SUFFIX):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), dpi=200)
    for ax, metric, title in zip(axes, ("rhmi_index", "ehmi_index"), ("RHMI", "EHMI")):
        if "error" in results.get(metric, {}) or "neuron_level_ks" not in results.get(metric, {}):
            ax.text(0.5, 0.5, f"{title}: insufficient data", transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=INK_SECONDARY)
            ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
            style_ax(ax, horizontal_grid=False)
            continue
        valid = df[df[metric].notna()]
        rplus = valid.loc[valid.reward_group == "R+", metric].to_numpy()
        rminus = valid.loc[valid.reward_group == "R-", metric].to_numpy()

        xp, yp = ecdf(rplus)
        xm, ym = ecdf(rminus)
        ax.step(xp, yp, where="post", color=COLOR_RPLUS, linewidth=1.6, label=f"R+ (n={len(rplus)})")
        ax.step(xm, ym, where="post", color=COLOR_RMINUS, linewidth=1.6, label=f"R- (n={len(rminus)})")

        # locate and mark the KS max-gap point
        grid = np.union1d(xp, xm)
        cdf_p = np.searchsorted(xp, grid, side="right") / len(xp)
        cdf_m = np.searchsorted(xm, grid, side="right") / len(xm)
        gap = np.abs(cdf_p - cdf_m)
        i_max = np.argmax(gap)
        x_at_max = grid[i_max]
        ax.vlines(x_at_max, cdf_m[i_max], cdf_p[i_max], color=INK_PRIMARY, linewidth=1.5,
                  linestyle=":", zorder=3)

        ax.set_xlim(np.percentile(np.concatenate([rplus, rminus]), 0.5),
                    np.percentile(np.concatenate([rplus, rminus]), 99.5))
        ax.set_xlabel("Modulation index")
        ax.set_ylabel("Cumulative fraction of units")

        ks = results[metric]["neuron_level_ks"]
        perm = results[metric]["mouse_block_permutation_ks"]
        subtitle = (f"KS={ks['ks_stat']:.3f}   neuron-level {sig_label(ks['p'])}\n"
                    f"mouse-block permutation {sig_label(perm['p_perm'])} ({perm['n_perm']} perms)")
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
        ax.text(0.5, 1.16, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        style_ax(ax, horizontal_grid=False)
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(f"R+ vs R- empirical CDF, {title_suffix}", fontsize=12.5, color=INK_PRIMARY, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = out_dir / f"{prefix}_ecdf_by_cohort.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_proportion(df, results, out_dir=OUT_DIR, prefix="011", title_suffix=TITLE_SUFFIX):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.0), dpi=200)
    for ax, metric, title in zip(axes, ("rhmi_index", "ehmi_index"), ("RHMI", "EHMI")):
        if "error" in results.get(metric, {}) or "mouse_level_proportion" not in results.get(metric, {}):
            ax.text(0.5, 0.5, f"{title}: insufficient data", transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=INK_SECONDARY)
            ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
            style_ax(ax)
            continue
        valid = df[df[metric].notna()]
        per_mouse = valid.groupby(["mouse_id", "reward_group"])[metric].apply(
            lambda s: float((s > 0).mean())
        ).reset_index(name="frac_positive")
        rplus = per_mouse.loc[per_mouse.reward_group == "R+", "frac_positive"].to_numpy()
        rminus = per_mouse.loc[per_mouse.reward_group == "R-", "frac_positive"].to_numpy()

        bp = ax.boxplot([rplus, rminus], positions=[1, 2], widths=0.55, showfliers=True, patch_artist=False)
        box_style(bp, "#898781")
        for artist, color in zip(bp["boxes"], (COLOR_RPLUS, COLOR_RMINUS)):
            artist.set_color(color)
        for i, (data, color) in enumerate(zip((rplus, rminus), (COLOR_RPLUS, COLOR_RMINUS)), start=1):
            rng = np.random.default_rng(0)
            jitter = rng.normal(0, 0.06, size=len(data))
            ax.scatter(np.full(len(data), i) + jitter, data, s=14, color=color, alpha=0.55, linewidths=0, zorder=2)

        ax.axhline(0.5, color=AXIS_LINE, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"R+\n(n={len(rplus)} mice)", f"R-\n(n={len(rminus)} mice)"])

        prop = results[metric]["mouse_level_proportion"]
        subtitle = f"Mann-Whitney: {sig_label(prop['mannwhitney_p'])}   |   Welch's t: {sig_label(prop['welch_p'])}"
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
        ax.text(0.5, 1.085, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)
        style_ax(ax)

    axes[0].set_ylabel("Fraction of units with index > 0 (per mouse)", color=INK_PRIMARY)
    fig.suptitle(f"Fraction of positively-modulated units by cohort, {title_suffix}",
                 fontsize=12.5, color=INK_PRIMARY, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = out_dir / f"{prefix}_proportion_by_cohort.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def run(in_path=IN_PATH, results_path=RESULTS_PATH, out_dir=OUT_DIR, prefix="011", title_suffix=TITLE_SUFFIX):
    df = pd.read_parquet(in_path)
    with open(results_path) as f:
        results = json.load(f)
    plot_ecdf(df, results, out_dir=out_dir, prefix=prefix, title_suffix=title_suffix)
    plot_proportion(df, results, out_dir=out_dir, prefix=prefix, title_suffix=title_suffix)


if __name__ == "__main__":
    run()
