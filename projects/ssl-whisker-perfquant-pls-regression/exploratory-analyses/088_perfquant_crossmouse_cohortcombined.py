"""Cohort-combined summary of `081`'s full cross-mouse sweep (user:
"Give a summary figure that combines cohorts for the cross-mouse
distributions of corrected scores, testing against zero").

`081b` tested each of the 4 null types separately, split by recipient
cohort (R+: rminus_null, withinrplus_null; R-: rplus_null,
withinrminus_null) -- 4 separate cells. But each null type has a mirror
image testing the SAME underlying question from the other cohort's side
(see the null-hypothesis table this project settled on):
- **reward-dependence**: rminus_null (R+ recipient x R- donor) AND
  rplus_null (R- recipient x R+ donor) -- both ask "does this recipient's
  OWN neural activity beat a donor from the OTHER reward group's activity,
  at predicting the recipient's own curve."
- **specificity**: withinrplus_null (R+ recipient x other-R+ donor) AND
  withinrminus_null (R- recipient x other-R- donor) -- both ask "does
  this recipient's own activity predict ITS OWN curve better than a
  same-cohort donor's DIFFERENT curve."

Pure post-hoc on `081b`'s already-computed per-recipient CSV
(`081_perfquant_crossmouse_fullsweep_persession.csv`, `corrected` =
real - median donor score, one row per recipient x null_type x target x
metric), no new compute. Pools recipients across BOTH cohorts within
each of the two roles above, then re-runs the group-level test (Wilcoxon
signed-rank + one-sample t-test, both reported, against 0) on the
COMBINED sample (75 recipients per role instead of 37-38 per cohort) --
more power than either cohort-specific test in `081b`, at the cost of no
longer being able to see a role-by-cohort interaction (081b's cohort-
split cells remain the reference for that).

BH-FDR applied across all 2 roles x 3 targets x 3 metrics = 18 tests.
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
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
ROLE_NULL_TYPES = {
    "reward-dependence": ["rminus_null", "rplus_null"],
    "specificity": ["withinrplus_null", "withinrminus_null"],
}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def stars(q):
    if np.isnan(q):
        return "n/a"
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))


def main():
    per_recipient_df = pd.read_csv(OUT_DIR / "081_perfquant_crossmouse_fullsweep_persession.csv")

    cells = []
    for role, null_types in ROLE_NULL_TYPES.items():
        role_df = per_recipient_df[per_recipient_df.null_type.isin(null_types)]
        for target in TARGETS:
            for metric in METRICS:
                g = role_df[(role_df.target == target) & (role_df.metric == metric)]
                vals = g["corrected"].dropna().values
                rec = dict(role=role, target=target, metric=metric, n_recipients=len(vals),
                           n_rplus=int((g.cohort == "R+").sum()), n_rminus=int((g.cohort == "R-").sum()),
                           median_corrected=float(np.median(vals)) if len(vals) else np.nan,
                           mean_corrected=float(np.mean(vals)) if len(vals) else np.nan)
                if len(vals) >= 5:
                    try:
                        wp = wilcoxon(vals).pvalue
                    except ValueError:
                        wp = np.nan
                    tp = ttest_1samp(vals, 0.0).pvalue
                    rec.update(wilcoxon_p=wp, ttest_p=tp)
                else:
                    rec.update(wilcoxon_p=np.nan, ttest_p=np.nan)
                cells.append(rec)
    group_df = pd.DataFrame(cells)
    for col in ("wilcoxon_p", "ttest_p"):
        mask = group_df[col].notna()
        q = np.full(len(group_df), np.nan)
        q[mask.values] = bh_fdr(group_df.loc[mask, col].values)
        group_df[col.replace("_p", "_q")] = q
    group_df.to_csv(OUT_DIR / "088_perfquant_crossmouse_cohortcombined_grouplevel.csv", index=False)

    print("=== cohort-combined cross-mouse null results ===")
    for _, row in group_df.iterrows():
        print(f"  {row['role']:>18} / {row['target']:<18} / {row['metric']:<8}: "
              f"n={row['n_recipients']} (R+={row['n_rplus']}, R-={row['n_rminus']}), "
              f"median_corrected={row['median_corrected']:+.3f}, "
              f"Wilcoxon q={row['wilcoxon_q']:.3g} ({stars(row['wilcoxon_q'])}), "
              f"t-test q={row['ttest_q']:.3g} ({stars(row['ttest_q'])})")

    # --- Figure: rows=role, cols=metric, x=target, dots colored by cohort (pooled), diamond=combined mean+/-SEM ---
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(ROLE_NULL_TYPES), len(METRICS), figsize=(4.2 * len(METRICS), 4.2 * len(ROLE_NULL_TYPES)),
                              constrained_layout=True, squeeze=False)
    for row_i, (role, null_types) in enumerate(ROLE_NULL_TYPES.items()):
        role_df = per_recipient_df[per_recipient_df.null_type.isin(null_types)]
        for col_i, metric in enumerate(METRICS):
            ax = axes[row_i][col_i]
            for x, target in enumerate(TARGETS):
                g = role_df[(role_df.target == target) & (role_df.metric == metric)]
                for cohort in ("R+", "R-"):
                    cg = g[g.cohort == cohort]
                    vals = cg["corrected"].values
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(np.full(len(vals), x) + jitter, vals, s=11, color=COHORT_COLOR[cohort],
                               alpha=0.5, edgecolors="none", zorder=2)
                vals_all = g["corrected"].values
                mean, sem = np.mean(vals_all), np.std(vals_all, ddof=1) / np.sqrt(len(vals_all))
                ax.errorbar(x, mean, yerr=sem, fmt="D", color="#222222", markersize=6, capsize=3, zorder=3)
                row = group_df[(group_df.role == role) & (group_df.target == target) & (group_df.metric == metric)].iloc[0]
                ax.text(x, vals_all.max() + 0.03, stars(row["wilcoxon_q"]), ha="center", va="bottom", fontsize=8)
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.set_xticks(range(len(TARGETS)))
            ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=8)
            n_note = group_df[(group_df.role == role) & (group_df.metric == metric)].iloc[0]
            ax.set_title(f"{role} -- {metric}\n(n={n_note['n_recipients']} recipients, "
                         f"R+={n_note['n_rplus']}/R-={n_note['n_rminus']})", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COHORT_COLOR[c],
                                       markersize=6, label=f"{c} recipient") for c in ("R+", "R-")]
                handles.append(plt.Line2D([0], [0], marker="D", color="#222222", linestyle="none",
                                           markersize=6, label="combined mean +/- SEM"))
                ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper right")
        axes[row_i][0].set_ylabel("corrected score (real - median donor)\npooled across cohorts, per recipient", fontsize=8)
    fig.suptitle("Cross-mouse sweep, cohorts combined -- corrected score vs 0, by role x metric\n"
                 "(dot color = recipient's own cohort; * = BH-FDR q<0.05, Wilcoxon signed-rank vs 0)", fontsize=11)
    fig_path = OUT_DIR / "088_perfquant_crossmouse_cohortcombined.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nsaved {fig_path.name} (+ .pdf)")
    print("DONE_088")


if __name__ == "__main__":
    main()
