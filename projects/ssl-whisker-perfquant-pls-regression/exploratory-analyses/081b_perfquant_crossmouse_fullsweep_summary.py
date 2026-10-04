"""Group-level summary + figure for `081`'s full cross-mouse sweep --
standalone from the raw CSV (same split-from-compute pattern as
`079`/`079b`, and the same `np.float64(...)`-wrapper parsing fix `079b`
found).

For each (recipient cohort, null_type, target, METRIC): per-recipient
corrected score (real - median donor) and rank-based p, THEN a group-
level test (Wilcoxon signed-rank + one-sample t-test, both reported,
this project's standing convention) on the corrected scores across all
recipients in that cohort (38 R+ or 37 R-) against 0. BH-FDR across the
36 (2 cohort x 2 null_type x 3 target x 3 metric) tests.

**2026-09-23 update ("Extend the within-R+ specificity test to R2 and
Spearman too")**: was Pearson-only; generalized to all 3 metrics
(r2/pearson/spearman -- this project's standing "always report all
three" convention) for every null type, not just within-R+ -- no new
compute needed, `081`'s raw CSV already carries all three, this script
just wasn't reporting them.
"""

from __future__ import annotations

import re
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
GROUPS = [
    ("R+", ["rminus_null", "withinrplus_null"]),
    ("R-", ["rplus_null", "withinrminus_null"]),
]
NULL_LABEL = {
    "rminus_null": "R+ recip x R- donor\n(reward-dependence)",
    "withinrplus_null": "R+ recip x other-R+ donor\n(specificity)",
    "rplus_null": "R- recip x R+ donor\n(reward-dependence, mirrored)",
    "withinrminus_null": "R- recip x other-R- donor\n(specificity, mirrored)",
}

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_list_col(s: str) -> list[float]:
    s = s.replace("np.float64(", "").replace(")", "")
    return [float(x) for x in _FLOAT_RE.findall(s)]


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


def _stars(q):
    if np.isnan(q):
        return "n/a"
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))


def main():
    df = pd.read_csv(OUT_DIR / "081_perfquant_crossmouse_fullsweep_raw.csv")
    for m in METRICS:
        df[m] = df[m].apply(parse_list_col)
    print(f"{len(df)} raw cells loaded", flush=True)

    # --- per-recipient corrected scores, all 3 metrics ---
    per_recipient = []
    for cohort, null_types in GROUPS:
        real_sub = df[df.null_type == "real"]
        for null_type in null_types:
            null_sub = df[df.null_type == null_type]
            recipients = sorted(set(null_sub.recipient_sid))
            for rsid in recipients:
                real_row = real_sub[real_sub.recipient_sid == rsid]
                if len(real_row) == 0:
                    continue
                donors = null_sub[null_sub.recipient_sid == rsid]
                for k, target in enumerate(TARGETS):
                    for metric in METRICS:
                        real_val = real_row.iloc[0][metric][k]
                        donor_vals = np.array([v[k] for v in donors[metric]])
                        if len(donor_vals) == 0:
                            continue
                        corrected = real_val - np.median(donor_vals)
                        p = (1 + np.sum(donor_vals >= real_val)) / (1 + len(donor_vals))
                        per_recipient.append(dict(
                            cohort=cohort, null_type=null_type, target=target, metric=metric,
                            recipient_mouse=real_row.iloc[0]["recipient_mouse"], recipient_sid=rsid,
                            real=real_val, donor_median=float(np.median(donor_vals)), n_donors=len(donor_vals),
                            corrected=corrected, p=p))
    per_recipient_df = pd.DataFrame(per_recipient)
    per_recipient_df.to_csv(OUT_DIR / "081_perfquant_crossmouse_fullsweep_persession.csv", index=False)

    # --- group-level test, all 3 metrics ---
    cells = []
    for cohort, null_types in GROUPS:
        for null_type in null_types:
            for target in TARGETS:
                for metric in METRICS:
                    g = per_recipient_df[(per_recipient_df.cohort == cohort) & (per_recipient_df.null_type == null_type)
                                          & (per_recipient_df.target == target) & (per_recipient_df.metric == metric)]
                    vals = g["corrected"].dropna().values
                    n_indiv_sig = int((g["p"] < 0.05).sum())
                    rec = dict(cohort=cohort, null_type=null_type, target=target, metric=metric, n_recipients=len(vals),
                               n_individually_sig=n_indiv_sig, median_corrected=float(np.median(vals)) if len(vals) else np.nan)
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
    group_df.to_csv(OUT_DIR / "081_perfquant_crossmouse_fullsweep_grouplevel.csv", index=False)

    print("\n=== GROUP-LEVEL cross-mouse null results, all 3 metrics ===")
    print("--- highlighted: within-R+ specificity test (all metrics) ---")
    sub = group_df[(group_df.cohort == "R+") & (group_df.null_type == "withinrplus_null")]
    for target in TARGETS:
        for metric in METRICS:
            row = sub[(sub.target == target) & (sub.metric == metric)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            print(f"    {target:<20} {metric:<9} n={int(row['n_recipients']):>3}  "
                  f"median_corrected={row['median_corrected']:+.3f}  "
                  f"indiv_sig={int(row['n_individually_sig'])}/{int(row['n_recipients'])}  "
                  f"wilcoxon p={row['wilcoxon_p']:.3g} q={row['wilcoxon_q']:.3g} {_stars(row['wilcoxon_q'])}  "
                  f"t-test p={row['ttest_p']:.3g} q={row['ttest_q']:.3g} {_stars(row['ttest_q'])}")

    print("\n--- all cells ---")
    for cohort, null_types in GROUPS:
        for null_type in null_types:
            for metric in METRICS:
                sub = group_df[(group_df.cohort == cohort) & (group_df.null_type == null_type) & (group_df.metric == metric)]
                print(f"-- {cohort} recipients / {null_type} / {metric} --")
                for _, row in sub.iterrows():
                    print(f"    {row['target']:<20} n={int(row['n_recipients']):>3}  "
                          f"median_corrected={row['median_corrected']:+.3f}  "
                          f"indiv_sig={int(row['n_individually_sig'])}/{int(row['n_recipients'])}  "
                          f"wilcoxon p={row['wilcoxon_p']:.3g} q={row['wilcoxon_q']:.3g} {_stars(row['wilcoxon_q'])}  "
                          f"t-test p={row['ttest_p']:.3g} q={row['ttest_q']:.3g} {_stars(row['ttest_q'])}")

    n_sig_w = (group_df["wilcoxon_q"] < 0.05).sum()
    n_sig_t = (group_df["ttest_q"] < 0.05).sum()
    print(f"\nTOTAL: {n_sig_w}/{len(group_df)} cells significant (Wilcoxon, BH-FDR q<0.05); "
          f"{n_sig_t}/{len(group_df)} (t-test)")

    # --- Figure 1: per-recipient strip + group median, one panel per (cohort, null_type), x=target -- Pearson only, unchanged from before ---
    panels = [(c, nt) for c, nts in GROUPS for nt in nts]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (cohort, null_type) in zip(axes, panels):
        for x, target in enumerate(TARGETS):
            g = per_recipient_df[(per_recipient_df.cohort == cohort) & (per_recipient_df.null_type == null_type)
                                  & (per_recipient_df.target == target) & (per_recipient_df.metric == "pearson")]
            vals = g["corrected"].values
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=12, color="#555555", alpha=0.5, edgecolors="none")
            mean, sem = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x, mean, yerr=sem, fmt="D", color="#d62728", markersize=6, capsize=3, zorder=3)
            row = group_df[(group_df.cohort == cohort) & (group_df.null_type == null_type)
                            & (group_df.target == target) & (group_df.metric == "pearson")].iloc[0]
            ax.text(x, vals.max() + 0.03, _stars(row["wilcoxon_q"]), ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=8)
        ax.set_title(NULL_LABEL[null_type], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("corrected score (real - median donor)\nPearson, per recipient", fontsize=8.5)
    fig.suptitle("Full cross-mouse sweep: corrected score by cohort x null type "
                 "(* = BH-FDR q<0.05, group-level Wilcoxon vs 0)", fontsize=12)
    fig_path = OUT_DIR / "081_perfquant_crossmouse_fullsweep.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\nsaved {fig_path.name}")

    # --- Figure 2: within-R+ specificity test, all 3 metrics (2026-09-23 addition) ---
    fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.4), constrained_layout=True)
    for ax, metric in zip(axes2, METRICS):
        for x, target in enumerate(TARGETS):
            g = per_recipient_df[(per_recipient_df.cohort == "R+") & (per_recipient_df.null_type == "withinrplus_null")
                                  & (per_recipient_df.target == target) & (per_recipient_df.metric == metric)]
            vals = g["corrected"].values
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=14, color="#555555", alpha=0.5, edgecolors="none")
            mean, sem = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x, mean, yerr=sem, fmt="D", color="#d62728", markersize=6, capsize=3, zorder=3)
            row = group_df[(group_df.cohort == "R+") & (group_df.null_type == "withinrplus_null")
                            & (group_df.target == target) & (group_df.metric == metric)].iloc[0]
            ax.text(x, vals.max() + 0.03, _stars(row["wilcoxon_q"]), ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes2[0].set_ylabel("corrected score (real - median donor)\nper recipient", fontsize=9)
    fig2.suptitle("R+ within-cohort specificity test, all 3 metrics "
                  "(* = BH-FDR q<0.05, group-level Wilcoxon vs 0)", fontsize=12)
    fig2_path = OUT_DIR / "081_perfquant_withinrplus_allmetrics.png"
    fig2.savefig(fig2_path, dpi=140)
    print(f"saved {fig2_path.name}")
    print("DONE_081b")


if __name__ == "__main__":
    main()
