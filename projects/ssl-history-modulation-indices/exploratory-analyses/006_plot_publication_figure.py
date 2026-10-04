"""Publication-quality RHMI/EHMI distribution figures, full cohort
(004_compute_full_cohort.py output), R+ vs R- cohort comparison overall and
by area_group. Palette/marks follow the repo's dataviz skill defaults
(categorical slot 1 blue = R+, slot 2 orange = R-, validated CVD-safe pair).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/full_cohort_unit_indices.parquet")
STATS_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/005_stats_results.json")
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")
MIN_UNITS_PER_AREA_PER_COHORT = 15

# dataviz skill categorical palette, slots 1/2 (validated CVD-safe adjacent pair)
COLOR_RPLUS = "#2a78d6"
COLOR_RMINUS = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": AXIS_LINE,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax, horizontal_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_LINE)
    ax.spines["bottom"].set_color(AXIS_LINE)
    if horizontal_grid:
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def box_style(bp, color):
    for element in ("boxes", "whiskers", "caps"):
        for artist in bp[element]:
            artist.set_color(color)
            artist.set_linewidth(1.1)
    for artist in bp["medians"]:
        artist.set_color(INK_PRIMARY)
        artist.set_linewidth(1.4)
    for artist in bp.get("fliers", []):
        artist.set_markeredgecolor(color)
        artist.set_markerfacecolor("none")
        artist.set_markersize(3)
        artist.set_alpha(0.5)


def sig_label(p):
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def plot_overall(df, stats_res, out_dir=OUT_DIR, prefix="006", title_suffix="full whisker-training cohort"):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.0), dpi=200)
    for ax, metric, title in zip(axes, ("rhmi_index", "ehmi_index"), ("RHMI", "EHMI")):
        valid = df[df[metric].notna()]
        rplus = valid.loc[valid.reward_group == "R+", metric].to_numpy()
        rminus = valid.loc[valid.reward_group == "R-", metric].to_numpy()

        bp = ax.boxplot(
            [rplus, rminus], positions=[1, 2], widths=0.55, showfliers=True, patch_artist=False,
        )
        box_style(bp, INK_MUTED)
        for artist, color in zip(bp["boxes"], (COLOR_RPLUS, COLOR_RMINUS)):
            artist.set_color(color)
        for i, (data, color) in enumerate(zip((rplus, rminus), (COLOR_RPLUS, COLOR_RMINUS)), start=1):
            rng = np.random.default_rng(0)
            jitter = rng.normal(0, 0.06, size=len(data))
            ax.scatter(np.full(len(data), i) + jitter, data, s=5, color=color, alpha=0.18, linewidths=0, zorder=2)

        ax.axhline(0, color=AXIS_LINE, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"R+\n(n={len(rplus)})", f"R-\n(n={len(rminus)})"])

        if "error" in stats_res.get(metric, {}):
            subtitle = "insufficient data for stats"
        else:
            mouse_p = stats_res[metric]["mouse_level"]["mannwhitney_p"]
            lmm_p = stats_res[metric]["neuron_level_lmm"]["p"]
            n_mice = stats_res[metric]["mouse_level"]["n_mice_rplus"] + stats_res[metric]["mouse_level"]["n_mice_rminus"]
            subtitle = f"mouse-level (n={n_mice}): {sig_label(mouse_p)}   |   neuron-level LMM: {sig_label(lmm_p)}"
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=10)
        ax.text(0.5, 1.085, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)
        style_ax(ax)

    axes[0].set_ylabel("Modulation index", color=INK_PRIMARY)
    fig.suptitle(f"Reward-/error-history modulation index by cohort, {title_suffix}",
                 fontsize=12.5, color=INK_PRIMARY, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = out_dir / f"{prefix}_index_by_cohort_overall.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_by_area(df, area_col="area_group", out_dir=OUT_DIR, prefix="006", title_suffix=""):
    for metric, title, fname in [
        ("rhmi_index", "RHMI", f"{prefix}_rhmi_by_area.png"),
        ("ehmi_index", "EHMI", f"{prefix}_ehmi_by_area.png"),
    ]:
        valid = df[df[metric].notna()]
        counts = valid.groupby([area_col, "reward_group"])[metric].size().unstack(fill_value=0)
        keep = counts.index[(counts.get("R+", 0) >= MIN_UNITS_PER_AREA_PER_COHORT) &
                             (counts.get("R-", 0) >= MIN_UNITS_PER_AREA_PER_COHORT)]
        keep = [a for a in keep if pd.notna(a)]
        if len(keep) == 0:
            print(f"{metric}: no area clears the {MIN_UNITS_PER_AREA_PER_COHORT}-units/cohort floor, skipping")
            continue

        medians = valid[valid[area_col].isin(keep)].groupby(area_col)[metric].median().sort_values()
        order = medians.index.tolist()

        fig, ax = plt.subplots(figsize=(7.5, 0.82 * len(order) + 1.6), dpi=200)
        spacing = 1.3  # distance between area-group centers
        gap = 0.28     # distance from center to each cohort's box center
        width = 0.34   # box width -- must satisfy 2*gap + width < spacing (no cross-area overlap)
        centers = np.arange(len(order)) * spacing + 1
        positions_rplus = centers - gap
        positions_rminus = centers + gap

        data_rplus = [valid.loc[(valid[area_col] == a) & (valid.reward_group == "R+"), metric].to_numpy() for a in order]
        data_rminus = [valid.loc[(valid[area_col] == a) & (valid.reward_group == "R-"), metric].to_numpy() for a in order]

        bp1 = ax.boxplot(data_rplus, positions=positions_rplus, widths=width, orientation="horizontal", showfliers=False, patch_artist=False)
        bp2 = ax.boxplot(data_rminus, positions=positions_rminus, widths=width, orientation="horizontal", showfliers=False, patch_artist=False)
        box_style(bp1, COLOR_RPLUS)
        box_style(bp2, COLOR_RMINUS)

        for c in centers[:-1]:
            ax.axhline(c + spacing / 2, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.axvline(0, color=AXIS_LINE, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks(centers)
        ax.set_yticklabels([f"{a}  (n={len(rp)}/{len(rm)})" for a, rp, rm in zip(order, data_rplus, data_rminus)])
        ax.set_xlabel("Modulation index")
        title_full = f"{title} by area (area_group), R+ vs R-  --  min {MIN_UNITS_PER_AREA_PER_COHORT} units/cohort/area"
        if title_suffix:
            title_full += f"\n{title_suffix}"
        ax.set_title(title_full, fontsize=11.5, color=INK_PRIMARY, pad=12)
        style_ax(ax)

        handles = [plt.Line2D([0], [0], color=COLOR_RPLUS, lw=2, label="R+"),
                   plt.Line2D([0], [0], color=COLOR_RMINUS, lw=2, label="R-")]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def run(in_path=IN_PATH, stats_path=STATS_PATH, out_dir=OUT_DIR, prefix="006", title_suffix="full whisker-training cohort"):
    df = pd.read_parquet(in_path)
    areas = pd.read_parquet("reports/ssl_analysis/derived/unit_area_labels.parquet")[
        ["session_id", "cluster_id", "area_group", "area_acronym_custom"]
    ]
    df = df.merge(areas, on=["session_id", "cluster_id"], how="left")

    with open(stats_path) as f:
        stats_res = json.load(f)

    plot_overall(df, stats_res, out_dir=out_dir, prefix=prefix, title_suffix=title_suffix)
    plot_by_area(df, out_dir=out_dir, prefix=prefix, title_suffix=title_suffix)


if __name__ == "__main__":
    run()
