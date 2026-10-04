"""Per-area, per-cohort significance testing on top of `066`'s area
ranking (user 2026-09-21, confirming the earlier offer: "Yes" [to adding
per-area significance testing]).

Pure post-hoc analysis on `066`'s already-computed CSV -- no new model
fits. For each (area, cohort, target) with >= MIN_SESSIONS_FOR_REPORT
sessions, Wilcoxon signed-rank test on the per-session
(real - mean(null)) differences, same convention as `058`'s group-level
test. With ~45 tests run per metric (8-9 areas x 3 targets x 2 cohorts),
raw p-values are also Benjamini-Hochberg FDR-corrected to control for
multiple comparisons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson", "spearman"]
MIN_SESSIONS_FOR_REPORT = 15


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction -> q-values (same order as input)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def main():
    df = pd.read_csv(OUT_DIR / "066_perfquant_arealevel_fullpool.csv")

    records = []
    for cohort in ("R+", "R-"):
        for target in TARGETS:
            sub_base = df[(df.reward_group == cohort) & (df.target == target)]
            for area, g in sub_base.groupby("area"):
                if g["session_id"].nunique() < MIN_SESSIONS_FOR_REPORT:
                    continue
                rec = dict(cohort=cohort, target=target, area=area, n_sessions=g["session_id"].nunique())
                for metric in METRICS:
                    vals = g[f"above_null_{metric}"].dropna().values
                    if len(vals) < 5:
                        rec[f"{metric}_p"] = np.nan
                        rec[f"{metric}_median"] = np.nan
                        continue
                    try:
                        p = wilcoxon(vals).pvalue
                    except ValueError:
                        p = np.nan
                    rec[f"{metric}_p"] = p
                    rec[f"{metric}_median"] = float(np.median(vals))
                    rec[f"{metric}_mean"] = float(np.mean(vals))
                records.append(rec)
    result_df = pd.DataFrame(records)

    # BH-FDR correction, per metric, across all (cohort, target, area) tests for that metric
    for metric in METRICS:
        mask = result_df[f"{metric}_p"].notna()
        q = np.full(len(result_df), np.nan)
        q[mask.values] = bh_fdr(result_df.loc[mask, f"{metric}_p"].values)
        result_df[f"{metric}_q"] = q

    result_df = result_df.sort_values(["cohort", "target", "pearson_median"], ascending=[True, True, False])
    result_df.to_csv(OUT_DIR / "067_perfquant_arealevel_significance.csv", index=False)

    print(f"=== per-area significance (Wilcoxon signed-rank on above-null Pearson, BH-FDR corrected "
          f"across {result_df.shape[0]} area x cohort x target tests) ===")
    for cohort in ("R+", "R-"):
        for target in TARGETS:
            sub = result_df[(result_df.cohort == cohort) & (result_df.target == target)]
            print(f"\n  -- {cohort} / {target} --")
            for _, row in sub.iterrows():
                sig = "***" if row["pearson_q"] < 0.001 else ("**" if row["pearson_q"] < 0.01 else
                      ("*" if row["pearson_q"] < 0.05 else "n.s."))
                print(f"    {row['area']:<28} n={int(row['n_sessions']):>3}  "
                      f"median_above_null_pearson={row['pearson_median']:+.3f}  "
                      f"p={row['pearson_p']:.4g}  q={row['pearson_q']:.4g}  {sig}")

    n_sig_raw = (result_df["pearson_p"] < 0.05).sum()
    n_sig_fdr = (result_df["pearson_q"] < 0.05).sum()
    print(f"\n=== TOTALS (Pearson): {n_sig_raw}/{len(result_df)} significant at raw p<0.05; "
          f"{n_sig_fdr}/{len(result_df)} survive BH-FDR q<0.05 ===")
    for metric in ("r2", "spearman"):
        n_sig_raw_m = (result_df[f"{metric}_p"] < 0.05).sum()
        n_sig_fdr_m = (result_df[f"{metric}_q"] < 0.05).sum()
        print(f"=== TOTALS ({metric}): {n_sig_raw_m}/{len(result_df)} significant at raw p<0.05; "
              f"{n_sig_fdr_m}/{len(result_df)} survive BH-FDR q<0.05 ===")

    # --- heatmap: mean above-null Pearson, with significance stars, one panel per cohort ---
    areas_common = sorted(set(result_df.area))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(areas_common) + 2), constrained_layout=True)
    for ax, cohort in zip(axes, ("R+", "R-")):
        mat = np.full((len(areas_common), len(TARGETS)), np.nan)
        stars = np.full((len(areas_common), len(TARGETS)), "", dtype=object)
        for i, area in enumerate(areas_common):
            for j, target in enumerate(TARGETS):
                row = result_df[(result_df.cohort == cohort) & (result_df.area == area) & (result_df.target == target)]
                if len(row):
                    mat[i, j] = row["pearson_mean"].iloc[0]
                    q = row["pearson_q"].iloc[0]
                    stars[i, j] = "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else ""))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.2)
        for i in range(len(areas_common)):
            for j in range(len(TARGETS)):
                if stars[i, j]:
                    ax.text(j, i, stars[i, j], ha="center", va="center", fontsize=9, color="black")
        ax.set_yticks(range(len(areas_common)))
        ax.set_yticklabels(areas_common, fontsize=7)
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels(TARGETS, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{cohort} (* = BH-FDR q<0.05)", fontsize=10)
        fig.colorbar(im, ax=ax, label="mean above-null Pearson", shrink=0.7)
    fig_path = OUT_DIR / "067_perfquant_arealevel_significance_heatmap.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")
    print("DONE_067")


if __name__ == "__main__":
    main()
