"""Post-hoc comparison of the three `062` full-pool runs (sensory /
baseline / corrected feature windows), across metrics (R2, Pearson,
Spearman), cohorts (R+/R-), and conditions (user 2026-09-20: "Compare
distributions of fits across metrics, cohorts and conditions. Run
tests.").

Pure post-hoc analysis on the three already-computed CSVs -- no model
refitting. Tests:
- Across conditions (same 88 sessions, paired): Wilcoxon signed-rank on
  each pairwise condition difference (sensory-baseline, sensory-corrected,
  baseline-corrected), per target x metric.
- Across cohorts (R+ vs R-, unpaired), within each condition: Mann-Whitney
  U AND Welch's t (project convention: always report both), per target x
  metric x condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson", "spearman"]
CONDITIONS = ["sensory", "baseline", "corrected"]
CONDITION_COLORS = {"sensory": "#1f77b4", "baseline": "#ff7f0e", "corrected": "#2ca02c"}
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def load_all() -> pd.DataFrame:
    frames = []
    for cond in CONDITIONS:
        df = pd.read_csv(OUT_DIR / f"062_perfquant_pls1se_{cond}_fullpool.csv")
        df["condition"] = cond
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    print(f"loaded {len(df)} rows ({df['session_id'].nunique()} sessions x {len(CONDITIONS)} conditions)")

    # =========================================================
    # Part 1: distributions of TEST scores across conditions
    # =========================================================
    fig1, axes1 = plt.subplots(len(TARGETS), len(METRICS), figsize=(4.6 * len(METRICS), 3.6 * len(TARGETS)), constrained_layout=True)
    condition_pair_records = []
    for row_i, target in enumerate(TARGETS):
        for col_i, metric in enumerate(METRICS):
            ax = axes1[row_i][col_i]
            sub = df[df.target == target]
            data = [sub[sub.condition == c][f"test_{metric}"].dropna().values for c in CONDITIONS]
            parts = ax.violinplot(data, positions=range(len(CONDITIONS)), showmeans=True, showextrema=False)
            for pc, c in zip(parts["bodies"], CONDITIONS):
                pc.set_facecolor(CONDITION_COLORS[c])
                pc.set_alpha(0.65)
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels(CONDITIONS, fontsize=8)
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.set_title(f"{target} -- {metric} (test)", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)

            # paired Wilcoxon signed-rank across conditions (same session_id set, merge on session_id)
            wide = sub.pivot(index="session_id", columns="condition", values=f"test_{metric}")
            y_top = max(np.nanmax(v) for v in data) if all(len(v) for v in data) else 1.0
            step = 0.08 * (y_top - min(np.nanmin(v) for v in data if len(v)))
            bracket_y = y_top + step
            for i, (ca, cb) in enumerate([("sensory", "baseline"), ("sensory", "corrected"), ("baseline", "corrected")]):
                paired = wide[[ca, cb]].dropna()
                if len(paired) > 5:
                    stat_p = wilcoxon(paired[ca], paired[cb]).pvalue
                else:
                    stat_p = float("nan")
                condition_pair_records.append(dict(target=target, metric=metric, cond_a=ca, cond_b=cb,
                                                     n=len(paired), wilcoxon_p=stat_p,
                                                     median_diff=float(np.median(paired[ca] - paired[cb])) if len(paired) else np.nan))
                ax.plot([CONDITIONS.index(ca), CONDITIONS.index(cb)], [bracket_y + i * step, bracket_y + i * step], color="#333333", lw=0.8)
                ax.text((CONDITIONS.index(ca) + CONDITIONS.index(cb)) / 2, bracket_y + i * step + step * 0.15,
                        f"p={stat_p:.3g}", ha="center", fontsize=6)
    fig1_path = OUT_DIR / "063_condition_comparison_test_distributions.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    pair_df = pd.DataFrame(condition_pair_records)
    pair_df.to_csv(OUT_DIR / "063_condition_pairwise_wilcoxon.csv", index=False)
    print("\n=== paired Wilcoxon signed-rank across conditions (test score) ===")
    for _, row in pair_df.iterrows():
        sig = "*" if row["wilcoxon_p"] < 0.05 else " "
        print(f"  {sig} {row['target']:<18} {row['metric']:<9} {row['cond_a']:>9} vs {row['cond_b']:<9} "
              f"median_diff={row['median_diff']:+.3f} p={row['wilcoxon_p']:.4g} (n={row['n']})")

    # =========================================================
    # Part 2: cohort (R+/R-) comparison within each condition, Pearson focus
    # =========================================================
    fig2, axes2 = plt.subplots(len(TARGETS), len(CONDITIONS), figsize=(4.4 * len(CONDITIONS), 3.6 * len(TARGETS)), constrained_layout=True)
    cohort_records = []
    for row_i, target in enumerate(TARGETS):
        for col_i, cond in enumerate(CONDITIONS):
            ax = axes2[row_i][col_i]
            sub = df[(df.target == target) & (df.condition == cond)]
            data, colors, labels = [], [], []
            for cohort in ("R+", "R-"):
                vals = sub[sub.reward_group == cohort]["test_pearson"].dropna().values
                data.append(vals)
                colors.append(COHORT_COLORS[cohort])
                labels.append(f"{cohort}\n(n={len(vals)})")
            parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.65)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels, fontsize=7)
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.set_title(f"{target} -- {cond}\n(test pearson)", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)

            for metric in METRICS:
                rplus = sub[sub.reward_group == "R+"][f"test_{metric}"].dropna().values
                rminus = sub[sub.reward_group == "R-"][f"test_{metric}"].dropna().values
                if len(rplus) > 2 and len(rminus) > 2:
                    mw_p = mannwhitneyu(rplus, rminus, alternative="two-sided").pvalue
                    welch_p = ttest_ind(rplus, rminus, equal_var=False).pvalue
                else:
                    mw_p, welch_p = float("nan"), float("nan")
                cohort_records.append(dict(target=target, condition=cond, metric=metric,
                                            n_rplus=len(rplus), n_rminus=len(rminus),
                                            mannwhitney_p=mw_p, welch_p=welch_p))
                if metric == "pearson" and len(rplus) > 2 and len(rminus) > 2:
                    y = max(np.nanmax(rplus), np.nanmax(rminus)) + 0.05
                    ax.plot([0, 1], [y, y], color="#333333", lw=0.8)
                    ax.text(0.5, y + 0.01, f"MW p={mw_p:.3f}\nWelch p={welch_p:.3f}", ha="center", fontsize=6)
    fig2_path = OUT_DIR / "063_cohort_comparison_by_condition.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    cohort_df = pd.DataFrame(cohort_records)
    cohort_df.to_csv(OUT_DIR / "063_cohort_comparison_stats.csv", index=False)
    print("\n=== R+ vs R- cohort comparison (Mann-Whitney + Welch), per target x condition x metric ===")
    for _, row in cohort_df.iterrows():
        sig = "*" if (row["mannwhitney_p"] < 0.05 or row["welch_p"] < 0.05) else " "
        print(f"  {sig} {row['target']:<18} {row['condition']:<10} {row['metric']:<9} "
              f"MW p={row['mannwhitney_p']:.4g}  Welch p={row['welch_p']:.4g}  "
              f"(n R+={row['n_rplus']}, n R-={row['n_rminus']})")

    n_sig_cohort = ((cohort_df["mannwhitney_p"] < 0.05) | (cohort_df["welch_p"] < 0.05)).sum()
    n_sig_cond = (pair_df["wilcoxon_p"] < 0.05).sum()
    print(f"\n=== TOTALS: {n_sig_cond}/{len(pair_df)} condition-pair comparisons significant (p<0.05); "
          f"{n_sig_cohort}/{len(cohort_df)} cohort comparisons significant (p<0.05) ===")

    print("DONE_063")


if __name__ == "__main__":
    main()
