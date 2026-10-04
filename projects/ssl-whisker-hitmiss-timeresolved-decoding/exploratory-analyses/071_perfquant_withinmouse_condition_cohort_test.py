"""Within-mouse test of real-vs-null decoding, broken down by condition
(day_stage: learning vs expert) x cohort (R+/R-) (user request 2026-09-23:
"Test within-mouse difference to mean of null dist across conditions and
cohorts").

Pure post-hoc analysis on `058` (learning) + `070` (expert)'s already-
computed CSVs -- no new model fits, same convention as `067`.

**Unit of analysis: MOUSE, not session** -- a deliberate, explicit
override of this project's usual session-level default for expert-stage
analyses ([[ssl_stats_unit_of_analysis]]), because the user asked for
"within-mouse" specifically. Sessions are averaged within (mouse,
day_stage, target) before testing -- a no-op at learning stage (1 session/
mouse there by construction) but real averaging at expert stage, where
some mice contribute 2 sessions (23 expert sessions / 16 mice).

For each (day_stage, cohort, target) cell, tests whether the per-mouse
`above_null_<metric>` (test score minus mean(null), already computed by
`058`/`070`) differs from 0: **both** Wilcoxon signed-rank (non-parametric)
and one-sample t-test (parametric), reported side by side -- same
"always report both" convention this project uses for cohort comparisons
([[ssl_rplus_rminus_test_pair]]), applied here to a one-sample/paired
test instead of an unpaired one. BH-FDR correction applied separately to
each test type x metric family (4 cells x 3 targets = 12 tests per
(test_type, metric) combination).

4 cells (2 day_stage x 2 cohort) -- no per-cell minimum-count filter
applied (unlike `066`/`067`'s per-area filter): every cell already has
>=6 mice (expert R- is the smallest at 6; learning R+/R- are 50/38).
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
    d_learning = pd.read_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv")
    d_learning["day_stage"] = "learning"
    d_expert = pd.read_csv(OUT_DIR / "070_perfquant_expert_pls_nulldist_fullpool.csv")
    d_expert["day_stage"] = "expert"
    df = pd.concat([d_learning, d_expert], ignore_index=True)

    # Within-mouse averaging: mean above_null_<metric> across sessions
    # within (mouse, day_stage, target) -- no-op at learning stage, real
    # averaging at expert stage (mice with 2 sessions).
    agg_cols = {f"above_null_{m}": "mean" for m in METRICS}
    agg_cols["session_id"] = "nunique"
    per_mouse = (df.groupby(["mouse", "reward_group", "day_stage", "target"])
                 .agg(agg_cols).rename(columns={"session_id": "n_sessions"}).reset_index())

    records = []
    for stage in STAGES:
        for cohort in COHORTS:
            for target in TARGETS:
                g = per_mouse[(per_mouse.day_stage == stage) & (per_mouse.reward_group == cohort)
                               & (per_mouse.target == target)]
                rec = dict(day_stage=stage, cohort=cohort, target=target, n_mice=len(g))
                for metric in METRICS:
                    vals = g[f"above_null_{metric}"].dropna().values
                    if len(vals) < 5:
                        rec[f"{metric}_wilcoxon_p"] = np.nan
                        rec[f"{metric}_ttest_p"] = np.nan
                        rec[f"{metric}_median"] = float(np.median(vals)) if len(vals) else np.nan
                        rec[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                        continue
                    try:
                        wp = wilcoxon(vals).pvalue
                    except ValueError:
                        wp = np.nan
                    tp = ttest_1samp(vals, 0.0).pvalue
                    rec[f"{metric}_wilcoxon_p"] = wp
                    rec[f"{metric}_ttest_p"] = tp
                    rec[f"{metric}_median"] = float(np.median(vals))
                    rec[f"{metric}_mean"] = float(np.mean(vals))
                    rec[f"{metric}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                records.append(rec)
    result_df = pd.DataFrame(records)

    # BH-FDR, separately per (test_type, metric) family across the 4 cells x 3 targets = 12 tests
    for metric in METRICS:
        for test_type in ("wilcoxon", "ttest"):
            col = f"{metric}_{test_type}_p"
            mask = result_df[col].notna()
            q = np.full(len(result_df), np.nan)
            q[mask.values] = bh_fdr(result_df.loc[mask, col].values)
            result_df[f"{metric}_{test_type}_q"] = q

    result_df = result_df.sort_values(["day_stage", "cohort", "target"])
    result_df.to_csv(OUT_DIR / "071_perfquant_withinmouse_condition_cohort_test.csv", index=False)

    def _stars(q):
        if np.isnan(q):
            return "n/a"
        return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))

    print("=== within-mouse above-null test, by day_stage x cohort x target ===")
    print("(both Wilcoxon signed-rank and one-sample t-test reported; BH-FDR q across the 12 "
          "(day_stage x cohort x target) tests per metric x test-type)\n")
    for stage in STAGES:
        for cohort in COHORTS:
            sub = result_df[(result_df.day_stage == stage) & (result_df.cohort == cohort)]
            print(f"-- {stage} / {cohort} --")
            for _, row in sub.iterrows():
                print(f"    {row['target']:<20} n_mice={int(row['n_mice']):>3}  "
                      f"pearson: median={row['pearson_median']:+.3f}  "
                      f"wilcoxon p={row['pearson_wilcoxon_p']:.4g} q={row['pearson_wilcoxon_q']:.4g} "
                      f"{_stars(row['pearson_wilcoxon_q'])}  |  "
                      f"t-test p={row['pearson_ttest_p']:.4g} q={row['pearson_ttest_q']:.4g} "
                      f"{_stars(row['pearson_ttest_q'])}")
        print()

    for metric in METRICS:
        for test_type in ("wilcoxon", "ttest"):
            n_sig = (result_df[f"{metric}_{test_type}_q"] < 0.05).sum()
            print(f"TOTAL {metric}/{test_type}: {n_sig}/{len(result_df)} cells survive BH-FDR q<0.05")

    # --- Figure: 2 (day_stage) x 3 (target) grid, bars = mean above-null pearson +/- SEM per cohort ---
    fig, axes = plt.subplots(len(STAGES), len(TARGETS), figsize=(4.2 * len(TARGETS), 3.6 * len(STAGES)),
                              constrained_layout=True, squeeze=False)
    for row_i, stage in enumerate(STAGES):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            sub = result_df[(result_df.day_stage == stage) & (result_df.target == target)]
            x = np.arange(len(COHORTS))
            means = [sub[sub.cohort == c]["pearson_mean"].iloc[0] if len(sub[sub.cohort == c]) else np.nan for c in COHORTS]
            sems = [sub[sub.cohort == c]["pearson_sem"].iloc[0] if len(sub[sub.cohort == c]) else np.nan for c in COHORTS]
            colors = [COHORT_COLOR[c] for c in COHORTS]
            ax.bar(x, means, yerr=sems, color=colors, capsize=4, width=0.6)
            for i, c in enumerate(COHORTS):
                row = sub[sub.cohort == c]
                if len(row):
                    q = row["pearson_wilcoxon_q"].iloc[0]
                    n_mice = int(row["n_mice"].iloc[0])
                    star = _stars(q)
                    y = (means[i] or 0) + (sems[i] or 0) + 0.01
                    ax.text(i, y, f"{star}\nn={n_mice}", ha="center", va="bottom", fontsize=7.5)
            ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels(COHORTS, fontsize=9)
            ax.set_title(f"{stage} -- {target}", fontsize=9.5)
            if col_i == 0:
                ax.set_ylabel("above-null Pearson\n(within-mouse mean +/- SEM)", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Within-mouse real-vs-null (above-null Pearson), by condition x cohort\n"
                 "(* = BH-FDR q<0.05, Wilcoxon signed-rank vs 0; n = mice)", fontsize=11)
    fig_path = OUT_DIR / "071_perfquant_withinmouse_condition_cohort_bars.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")
    print("DONE_071")


if __name__ == "__main__":
    main()
