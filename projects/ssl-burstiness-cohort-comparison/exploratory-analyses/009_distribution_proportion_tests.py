"""Location/proportion/distribution test battery (the "three questions" from
skills/ssl-analyze/references/ssl_cohort_comparison_stats.md) for R+ vs R-,
per (day_stage x scope x tier x metric) -- location already done in
004_statistics.py (global_test + PERMANOVA); this adds the proportion test
(does the balance of enhanced- vs suppressed- units differ) and the
distribution test (does the full shape differ, neuron-level KS +
mouse-block permutation KS). Reuses the already-computed full-population
parquets -- no NWB reload needed.

Thresholds: burst_index has a natural sign split (threshold=0, matching the
"enhanced vs suppressed" convention in ssl_cohort_comparison_stats.md).
continuous_burstiness has no natural zero -- threshold=0.5 ("majority
bursty"), matching this project's own earlier ad hoc reporting convention.

mouse_block_permutation_ks_test's per-permutation label expansion is
vectorized (precomputed mouse-index array + numpy indexing) from the start,
same fix already applied to permanova.py after it was found to be a
bottleneck at this project's scale (~130k units in some combinations).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
METRICS = ["continuous_burstiness", "burst_index_whisker", "burst_index_auditory"]
PROPORTION_THRESHOLD = {"continuous_burstiness": 0.5, "burst_index_whisker": 0.0, "burst_index_auditory": 0.0}
N_PERM = 999


def mouse_level_proportion_test(df: pd.DataFrame, metric: str, threshold: float) -> dict:
    per_mouse = df.groupby(["mouse_id", "reward_group"])[metric].apply(
        lambda s: float((s > threshold).mean())
    ).reset_index(name="frac_positive")
    rplus = per_mouse.loc[per_mouse.reward_group == "R+", "frac_positive"].to_numpy()
    rminus = per_mouse.loc[per_mouse.reward_group == "R-", "frac_positive"].to_numpy()
    if len(rplus) < 2 or len(rminus) < 2:
        return {"threshold": threshold, "prop_mannwhitney_p": np.nan, "prop_welch_p": np.nan}
    u_stat, u_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)
    return {
        "threshold": threshold,
        "mean_frac_rplus": float(np.mean(rplus)), "mean_frac_rminus": float(np.mean(rminus)),
        "prop_mannwhitney_p": float(u_p), "prop_welch_p": float(t_p),
    }


def neuron_level_ks_test(df: pd.DataFrame, metric: str) -> dict:
    rplus = df.loc[df.reward_group == "R+", metric].to_numpy()
    rminus = df.loc[df.reward_group == "R-", metric].to_numpy()
    res = scipy_stats.ks_2samp(rplus, rminus)
    return {"ks_stat": float(res.statistic), "ks_p_pseudoreplicated": float(res.pvalue)}


def mouse_block_permutation_ks_test(df: pd.DataFrame, metric: str, n_perm: int = N_PERM, seed: int = 0) -> dict:
    mouse_labels = df.drop_duplicates("mouse_id")[["mouse_id", "reward_group"]].reset_index(drop=True)
    mouse_ids = mouse_labels["mouse_id"].to_numpy()
    true_labels = mouse_labels["reward_group"].to_numpy()
    values = df[metric].to_numpy()
    mouse_codes = pd.Categorical(df["mouse_id"], categories=mouse_ids).codes

    def ks_for_labels(group_arr):
        a = values[group_arr == "R+"]
        b = values[group_arr == "R-"]
        return scipy_stats.ks_2samp(a, b).statistic

    observed = ks_for_labels(df["reward_group"].to_numpy())
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(true_labels)
        group_arr = perm_labels[mouse_codes]
        null_stats[i] = ks_for_labels(group_arr)
    p_perm = float((1 + (null_stats >= observed).sum()) / (n_perm + 1))
    return {"ks_perm_n_mice_rplus": int((true_labels == "R+").sum()), "ks_perm_n_mice_rminus": int((true_labels == "R-").sum()),
            "ks_stat_check": float(observed), "ks_p_mouseblock": p_perm}


def main() -> None:
    results = []
    for day_stage in ["learning", "expert"]:
        df_full = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics.parquet")
        for scope_name, scope_filter in [
            ("all_mice", df_full["reward_group"].isin(["R+", "R-"])),
            ("learners_only", df_full["learning_category"].isin(["good", "moderate"])),
        ]:
            scoped = df_full[scope_filter]
            for tier_name, tier_filter in [
                ("good", scoped["quality_label"] == "good"),
                ("good_mua", scoped["quality_label"].isin(["good", "mua"])),
            ]:
                tiered = scoped[tier_filter]
                for metric in METRICS:
                    label = f"{day_stage}_{scope_name}_{tier_name}_{metric}"
                    valid_col = f"{metric}_valid"
                    sub = tiered[tiered[valid_col] & tiered[metric].notna()]
                    n_rplus_mice = sub.loc[sub.reward_group == "R+", "mouse_id"].nunique()
                    n_rminus_mice = sub.loc[sub.reward_group == "R-", "mouse_id"].nunique()
                    row = {"label": label, "day_stage": day_stage, "scope": scope_name, "tier": tier_name,
                           "metric": metric, "n_units": len(sub), "n_rplus_mice": n_rplus_mice, "n_rminus_mice": n_rminus_mice}
                    if n_rplus_mice < 2 or n_rminus_mice < 2:
                        row["skipped_reason"] = "fewer than 2 mice in a cohort"
                        results.append(row)
                        continue
                    print(f"=== {label} (n={len(sub)}) ===")
                    row.update(mouse_level_proportion_test(sub, metric, PROPORTION_THRESHOLD[metric]))
                    row.update(neuron_level_ks_test(sub, metric))
                    row.update(mouse_block_permutation_ks_test(sub, metric))
                    results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(ARTIFACTS_DIR / "distribution_proportion_summary.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label", "n_units", "prop_mannwhitney_p", "prop_welch_p", "ks_stat", "ks_p_pseudoreplicated", "ks_p_mouseblock"]
    print("\n" + out[cols].to_string(index=False))
    print(f"\nWrote {ARTIFACTS_DIR / 'distribution_proportion_summary.csv'}")


if __name__ == "__main__":
    main()
