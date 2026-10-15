"""Figures for the lick_time-aligned modality decode's group-level results.
Mirrors `005_plot_group_level_results.py`'s structure: curves for the most
notable areas (with significant cluster shading) and a stats heatmap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from ssl_timeresolved_decoding import cluster_permutation_test, rplus_rminus_curve_difference

OUT_DIR = Path(__file__).resolve().parent
PARTIAL_PATH = OUT_DIR / "008_lick_aligned_results_partial.parquet"
STATS_PATH = OUT_DIR / "010_lick_aligned_group_level_test_results.csv"
BIN_EDGES_PATH = OUT_DIR / "008_bin_edges.json"

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
NOTABLE_AREAS = ["Midbrain", "Motor and frontal areas", "Somatosensory areas",
                  "Striatum and pallidum", "Thalamus", "Retrosplenial areas"]


def load_bin_centers():
    bin_edges = json.loads(BIN_EDGES_PATH.read_text())
    return np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])


def figure_curves(df, bin_centers_ms, rng):
    fig, axes = plt.subplots(len(NOTABLE_AREAS), 2, figsize=(9, 2.6 * len(NOTABLE_AREAS)), sharex=True)
    for i, area in enumerate(NOTABLE_AREAS):
        for j, half in enumerate(("first", "second")):
            ax = axes[i, j]
            sub = df[(df.area_value == area) & (df.half == half)]
            recs = [dict(subject_id=r.subject_id, reward_group=r.reward_group, real_curve=np.array(r.real_curve))
                    for r in sub.itertuples()]
            if sum(r["reward_group"] == "R+" for r in recs) < 5 or sum(r["reward_group"] == "R-" for r in recs) < 5:
                ax.axis("off")
                continue
            diff_obs, diff_null = rplus_rminus_curve_difference(recs, rng, n_perm=1000)
            res = cluster_permutation_test(diff_obs, diff_null, two_sided=True)
            sig_mask = np.abs(res["observed_z"]) > 1.96

            for cohort in ("R+", "R-"):
                curves = np.stack([r["real_curve"] for r in recs if r["reward_group"] == cohort])
                mean_curve = np.nanmean(curves, axis=0)
                sem_curve = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])
                ax.plot(bin_centers_ms, mean_curve, color=COHORT_COLOR[cohort], lw=2, label=f"{cohort} (n={curves.shape[0]})")
                ax.fill_between(bin_centers_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=COHORT_COLOR[cohort], alpha=0.2, lw=0)

            ymin, ymax = ax.get_ylim()
            ax.fill_between(bin_centers_ms, ymin, ymax, where=sig_mask, color="#666666", alpha=0.15, step="mid", zorder=0)
            ax.set_ylim(ymin, ymax)
            ax.axhline(0.5, color="#999999", lw=1, linestyle=":")
            ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{area} -- {half} half\ncluster p={res['p_value']:.4f}", fontsize=9)
            if i == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper left")
            if i == len(NOTABLE_AREAS) - 1:
                ax.set_xlabel("time from lick_time (ms)", fontsize=9)
            if j == 0:
                ax.set_ylabel("balanced accuracy", fontsize=9)

    fig.suptitle("Whisker-vs-auditory decoding around the lick: R+ vs R- mean curves (SEM band), grey = significant cluster", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "011_lick_aligned_curves.png", dpi=140)
    print("saved 011_lick_aligned_curves.png")


def figure_stats_heatmap(stats):
    stats = stats.copy()
    stats["area_half"] = stats["area"] + " (" + stats["half"] + ")"
    cols = ["rplus_above_chance_p", "rminus_above_chance_p", "curve_diff_cluster_p"]
    col_labels = ["R+ above chance", "R- above chance", "R+ vs R- (curve-diff)"]
    mat = -np.log10(stats[cols].to_numpy().astype(float).clip(min=1e-4))

    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(stats) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=-np.log10(0.0005))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats["area_half"], fontsize=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            p = stats[cols[c]].iloc[r]
            marker = "*" if p < 0.05 else ""
            ax.text(c, r, f"{p:.3f}{marker}", ha="center", va="center", fontsize=7,
                    color="white" if mat[r, c] > mat.max() * 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("-log10(p)", fontsize=9)
    ax.set_title("Lick-aligned modality decode: significance summary (* = p<0.05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "011_lick_aligned_stats_heatmap.png", dpi=140)
    print("saved 011_lick_aligned_stats_heatmap.png")


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[(df.area_col == "area_group") & df.skipped_reason.isna()].copy()
    stats = pd.read_csv(STATS_PATH)
    bin_centers_ms = load_bin_centers()
    rng = np.random.default_rng(13)

    figure_curves(df, bin_centers_ms, rng)
    figure_stats_heatmap(stats)


if __name__ == "__main__":
    main()
