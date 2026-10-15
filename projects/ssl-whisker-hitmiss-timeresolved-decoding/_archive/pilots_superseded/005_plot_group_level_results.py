"""Figures for the learning-stage pilot's group-level results (lick_flag
target, area_group scheme, 89-session sweep + `004_pilot_group_level_tests.py`
tests). Three figures:
  006_curves_significant_areas.png -- mean decode curves (R+ vs R-) for the
    6 notable areas, both halves, with the curve-diff cluster-test's
    significant time windows shaded.
  006_stats_heatmap.png -- p-value heatmap across all 19 area x half cells,
    above-chance (each cohort) and curve-diff comparison.
  006_peak_accuracy_comparison.png -- per-session peak accuracy, R+ vs R-,
    one panel per area, both halves, with the curve-diff cluster p annotated.
"""

from __future__ import annotations

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
PARTIAL_PATH = OUT_DIR / "002_pilot_results_partial.parquet"
STATS_PATH = OUT_DIR / "004_group_level_test_results.csv"

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
NOTABLE_AREAS = ["Motor and frontal areas", "Somatosensory areas", "Striatum and pallidum",
                  "Hippocampus", "Midbrain", "Thalamus"]


def load_bin_centers():
    import json
    bin_edges = json.loads((OUT_DIR / "002_bin_edges.json").read_text())
    return np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])


def figure_curves(df: pd.DataFrame, bin_centers_ms: np.ndarray, rng: np.random.Generator):
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
            ax.axvspan(-5, 5, color="#dddddd", alpha=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            title = f"{area} -- {half} half\ncluster p={res['p_value']:.4f}"
            ax.set_title(title, fontsize=9)
            if i == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper left")
            if i == len(NOTABLE_AREAS) - 1:
                ax.set_xlabel("time from start_time (ms)", fontsize=9)
            if j == 0:
                ax.set_ylabel("balanced accuracy", fontsize=9)

    fig.suptitle("lick_flag decoding: R+ vs R- mean curves (SEM band), grey = significant cluster (curve-diff test)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "006_curves_significant_areas.png", dpi=140)
    print("saved 006_curves_significant_areas.png")


def figure_stats_heatmap(stats: pd.DataFrame):
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
    ax.set_title("Significance summary (* = p<0.05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "006_stats_heatmap.png", dpi=140)
    print("saved 006_stats_heatmap.png")


def figure_peak_accuracy(df: pd.DataFrame, stats: pd.DataFrame):
    areas = sorted(df.area_value.unique())
    ncols = 3
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    rng_jitter = np.random.default_rng(0)

    for i, area in enumerate(areas):
        ax = axes[i // ncols, i % ncols]
        for half, x0 in (("first", 0), ("second", 1)):
            sub = df[(df.area_value == area) & (df.half == half)]
            for cohort, dx in (("R+", -0.15), ("R-", 0.15)):
                vals = sub.loc[sub.reward_group == cohort, "peak_acc"].to_numpy()
                if len(vals) == 0:
                    continue
                x = x0 + dx + rng_jitter.uniform(-0.05, 0.05, size=len(vals))
                ax.scatter(x, vals, color=COHORT_COLOR[cohort], alpha=0.5, s=14, edgecolor="none")
                ax.errorbar([x0 + dx], [vals.mean()], yerr=[vals.std() / np.sqrt(len(vals))],
                            color=COHORT_COLOR[cohort], fmt="D", markersize=6, capsize=3, lw=1.5)
            row = stats[(stats.area == area) & (stats.half == half)]
            if len(row):
                p = row.iloc[0]["curve_diff_cluster_p"]
                marker = "*" if p < 0.05 else "n.s."
                ax.text(x0, ax.get_ylim()[1] if ax.get_ylim()[1] < 1.0 else 0.95, marker, ha="center", fontsize=9)
        ax.axhline(0.5, color="#999999", lw=1, linestyle=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["first half", "second half"], fontsize=9)
        ax.set_ylabel("peak balanced accuracy", fontsize=9)
        ax.set_title(area, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(areas), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    handles = [plt.Line2D([0], [0], marker="D", color=COHORT_COLOR["R+"], label="R+", linestyle="none"),
               plt.Line2D([0], [0], marker="D", color=COHORT_COLOR["R-"], label="R-", linestyle="none")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-session peak accuracy, R+ vs R- (mean +/- SEM diamonds); * = curve-diff cluster p<0.05", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT_DIR / "006_peak_accuracy_comparison.png", dpi=140)
    print("saved 006_peak_accuracy_comparison.png")


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[(df.area_col == "area_group") & df.skipped_reason.isna()].copy()
    stats = pd.read_csv(STATS_PATH)
    bin_centers_ms = load_bin_centers()
    rng = np.random.default_rng(11)

    figure_curves(df, bin_centers_ms, rng)
    figure_stats_heatmap(stats)
    figure_peak_accuracy(df, stats)


if __name__ == "__main__":
    main()
