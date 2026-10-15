"""Master figure generator across all planned variants (mirrors
`012_master_group_level_tests.py`'s dataset list). For each dataset that
exists and each population scope, produces:
  013_<dataset>_<scope>_curves.png       -- R+ vs R- mean curves (SEM band)
                                             for the areas most often
                                             significant on the curve-diff
                                             test, with the significant
                                             cluster shaded.
  013_<dataset>_<scope>_stats_heatmap.png -- p-value heatmap across all
                                             tested area x half cells,
                                             including the 50ms-post-stim
                                             column where applicable.
Reads `012_master_test_results.csv` (must be regenerated first if data
changed) to pick which areas to plot and to build the heatmap.
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
STATS_PATH = OUT_DIR / "012_master_test_results.csv"

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
DATASET_PARQUET = {
    "start_time_learning": "002_pilot_results_partial.parquet",
    "start_time_expert": "002_pilot_results_partial_expert.parquet",
    "lick_time_learning": "008_lick_aligned_results_partial.parquet",
    "lick_time_expert": "008_lick_aligned_results_partial_expert.parquet",
}
DATASET_BIN_EDGES = {
    "start_time_learning": "002_bin_edges.json",
    "start_time_expert": "002_bin_edges_expert.json",
    "lick_time_learning": "008_bin_edges.json",
    "lick_time_expert": "008_bin_edges_expert.json",
}
DATASET_XLABEL = {
    "start_time_learning": "time from start_time (ms)",
    "start_time_expert": "time from start_time (ms)",
    "lick_time_learning": "time from lick_time (ms)",
    "lick_time_expert": "time from lick_time (ms)",
}
N_TOP_AREAS = 6


def bin_centers_ms_for(name: str) -> np.ndarray:
    edges = json.loads((OUT_DIR / DATASET_BIN_EDGES[name]).read_text())
    return np.array([(b[0] + b[1]) / 2 * 1000 for b in edges])


def figure_curves(dataset_name: str, scope: str, area_col: str, areas: list[str], rng: np.random.Generator):
    parquet_path = OUT_DIR / DATASET_PARQUET[dataset_name]
    df = pd.read_parquet(parquet_path)
    df = df[(df.area_col == area_col) & df.skipped_reason.isna()]
    if scope == "learners_only":
        df = df[df.learning_category.isin(["good", "moderate"])]
    bin_centers_ms = bin_centers_ms_for(dataset_name)
    xlabel = DATASET_XLABEL[dataset_name]

    fig, axes = plt.subplots(len(areas), 2, figsize=(9, 2.6 * len(areas)), sharex=True, squeeze=False)
    for i, area in enumerate(areas):
        for j, half in enumerate(("first", "second")):
            ax = axes[i][j]
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
            if i == len(areas) - 1:
                ax.set_xlabel(xlabel, fontsize=9)
            if j == 0:
                ax.set_ylabel("balanced accuracy", fontsize=9)

    fig.suptitle(f"{dataset_name} ({scope}, {area_col}): R+ vs R- mean curves, grey = significant cluster", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUT_DIR / f"013_{dataset_name}_{scope}_{area_col}_curves.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path.name}")


def figure_stats_heatmap(stats_sub: pd.DataFrame, dataset_name: str, scope: str, area_col: str, has_poststim: bool):
    stats_sub = stats_sub.copy()
    stats_sub["area_half"] = stats_sub["area"] + " (" + stats_sub["half"] + ")"
    cols = ["rplus_above_chance_p", "rminus_above_chance_p", "curve_diff_cluster_p"]
    col_labels = ["R+ above chance", "R- above chance", "R+ vs R- (curve-diff)"]
    if has_poststim:
        cols.append("poststim50_mouseblock_perm_p")
        col_labels.append("R+ vs R- (0-50ms post-stim)")
    mat = -np.log10(stats_sub[cols].to_numpy().astype(float).clip(min=1e-4))

    fig, ax = plt.subplots(figsize=(6.5 + (1.6 if has_poststim else 0), 0.35 * len(stats_sub) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=-np.log10(0.0005))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(stats_sub)))
    ax.set_yticklabels(stats_sub["area_half"], fontsize=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            p = stats_sub[cols[c]].iloc[r]
            marker = "*" if p < 0.05 else ""
            ax.text(c, r, f"{p:.3f}{marker}", ha="center", va="center", fontsize=7,
                    color="white" if mat[r, c] > np.nanmax(mat) * 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("-log10(p)", fontsize=9)
    ax.set_title(f"{dataset_name} ({scope}, {area_col}): significance summary (* = p<0.05)", fontsize=10)
    fig.tight_layout()
    out_path = OUT_DIR / f"013_{dataset_name}_{scope}_{area_col}_stats_heatmap.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path.name}")


def main():
    if not STATS_PATH.exists():
        print("012_master_test_results.csv not found -- run 012_master_group_level_tests.py first")
        return
    stats = pd.read_csv(STATS_PATH)
    rng = np.random.default_rng(31)

    for (dataset_name, scope, area_col), grp in stats.groupby(["dataset", "scope", "area_col"]):
        has_poststim = "poststim50_mouseblock_perm_p" in grp.columns and grp["poststim50_mouseblock_perm_p"].notna().any()
        figure_stats_heatmap(grp, dataset_name, scope, area_col, has_poststim)

        top_areas = grp.sort_values("curve_diff_cluster_p").drop_duplicates("area")["area"].head(N_TOP_AREAS).tolist()
        if top_areas:
            figure_curves(dataset_name, scope, area_col, top_areas, rng)


if __name__ == "__main__":
    main()
