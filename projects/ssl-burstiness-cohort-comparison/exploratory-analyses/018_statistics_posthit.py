"""Statistics battery on post-first-hit data (see 016_posthit_lib.py /
017_full_scale_compute_posthit.py), mirroring 004_statistics.py +
009_distribution_proportion_tests.py combined -- global location test,
PERMANOVA (mouse-block permutation), proportion test, distribution test
(neuron-level + mouse-block-permutation KS).

Per user's explicit request this round: NO multiple-testing correction
applied to the per-area post-hoc -- raw p-values reported, BH-FDR not used
for interpretation (still computed internally by the reused
run_with_posthoc for reference/comparison, just not the deciding criterion
this time). The "only run per-area post-hoc if the main PERMANOVA is
significant" gate is KEPT -- that's a computational-cost/hierarchical-
testing-order choice (avoids re-triggering the multi-hour runtime this
project hit earlier), not a multiple-comparison correction, so it stays
regardless of the "no correction" instruction. Stated explicitly rather
than silently deciding either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from permanova import permanova_euclidean, run_with_posthoc

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
METRICS = ["continuous_burstiness", "burst_index_whisker", "burst_index_auditory"]
PROPORTION_THRESHOLD = {"continuous_burstiness": 0.5, "burst_index_whisker": 0.0, "burst_index_auditory": 0.0}
KS_N_PERM = 999


def global_test(df: pd.DataFrame, metric: str, unit_col: str) -> dict:
    per_unit = df.groupby([unit_col, "reward_group"])[metric].median().reset_index()
    rplus = per_unit.loc[per_unit.reward_group == "R+", metric].to_numpy()
    rminus = per_unit.loc[per_unit.reward_group == "R-", metric].to_numpy()
    if len(rplus) < 2 or len(rminus) < 2:
        return {"mannwhitney_p": np.nan, "welch_p": np.nan}
    u_stat, u_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)
    return {"median_rplus": float(np.median(rplus)), "median_rminus": float(np.median(rminus)),
            "mannwhitney_p": float(u_p), "welch_p": float(t_p)}


def proportion_test(df: pd.DataFrame, metric: str, threshold: float) -> dict:
    per_mouse = df.groupby(["mouse_id", "reward_group"])[metric].apply(lambda s: float((s > threshold).mean())).reset_index(name="frac")
    rplus = per_mouse.loc[per_mouse.reward_group == "R+", "frac"].to_numpy()
    rminus = per_mouse.loc[per_mouse.reward_group == "R-", "frac"].to_numpy()
    if len(rplus) < 2 or len(rminus) < 2:
        return {"prop_mannwhitney_p": np.nan, "prop_welch_p": np.nan}
    u, up = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t, tp = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)
    return {"prop_mannwhitney_p": float(up), "prop_welch_p": float(tp)}


def distribution_tests(df: pd.DataFrame, metric: str) -> dict:
    rplus = df.loc[df.reward_group == "R+", metric].to_numpy()
    rminus = df.loc[df.reward_group == "R-", metric].to_numpy()
    ks_neuron = scipy_stats.ks_2samp(rplus, rminus)

    mouse_labels = df.drop_duplicates("mouse_id")[["mouse_id", "reward_group"]].reset_index(drop=True)
    mouse_ids = mouse_labels["mouse_id"].to_numpy()
    true_labels = mouse_labels["reward_group"].to_numpy()
    values = df[metric].to_numpy()
    mouse_codes = pd.Categorical(df["mouse_id"], categories=mouse_ids).codes

    def ks_for(group_arr):
        return scipy_stats.ks_2samp(values[group_arr == "R+"], values[group_arr == "R-"]).statistic

    observed = ks_for(df["reward_group"].to_numpy())
    rng = np.random.default_rng(0)
    null_stats = np.empty(KS_N_PERM)
    for i in range(KS_N_PERM):
        perm_labels = rng.permutation(true_labels)
        null_stats[i] = ks_for(perm_labels[mouse_codes])
    p_mouseblock = float((1 + (null_stats >= observed).sum()) / (KS_N_PERM + 1))
    return {"ks_p_neuron_pseudoreplicated": float(ks_neuron.pvalue), "ks_p_mouseblock": p_mouseblock}


def run_combination(df: pd.DataFrame, metric: str, unit_col: str, label: str) -> dict:
    valid_col = f"{metric}_valid"
    sub = df[df[valid_col] & df[metric].notna()].copy()
    n_rplus_mice = sub.loc[sub.reward_group == "R+", "mouse_id"].nunique()
    n_rminus_mice = sub.loc[sub.reward_group == "R-", "mouse_id"].nunique()
    row = {"label": label, "n_units": len(sub), "n_mice": sub.mouse_id.nunique(),
           "n_rplus_mice": n_rplus_mice, "n_rminus_mice": n_rminus_mice}
    if n_rplus_mice < 2 or n_rminus_mice < 2:
        row["skipped_reason"] = "fewer than 2 mice in a cohort"
        return row

    row.update(global_test(sub, metric, unit_col))
    row.update(proportion_test(sub, metric, PROPORTION_THRESHOLD[metric]))
    row.update(distribution_tests(sub, metric))

    X = sub[[metric]].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    f_obs, p_val, n_main, n_mice_main = permanova_euclidean(X, sub["reward_group"].to_numpy(), sub["mouse_id"].to_numpy())
    row["permanova_pseudo_F"] = f_obs
    row["permanova_p"] = p_val
    row["permanova_significant"] = bool(p_val < 0.05)

    if row["permanova_significant"]:
        _, posthoc = run_with_posthoc(sub, [metric], area_col="area_acronym_custom", label_prefix=label)
        row["n_areas_tested"] = len(posthoc)
        row["n_areas_nominal_p_lt_05"] = int((posthoc["p_value"] < 0.05).sum()) if len(posthoc) else 0
        posthoc.to_csv(ARTIFACTS_DIR / f"posthit_posthoc_{label}.csv", index=False)
    else:
        row["n_areas_tested"] = 0
        row["n_areas_nominal_p_lt_05"] = 0
    return row


def main() -> None:
    results = []
    for day_stage, unit_col in [("learning", "mouse_id"), ("expert", "session_id")]:
        df = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics_posthit.parquet")
        for scope_name, scope_filter in [
            ("all_mice", df["reward_group"].isin(["R+", "R-"])),
            ("learners_only", df["learning_category"].isin(["good", "moderate"])),
        ]:
            scoped = df[scope_filter]
            for tier_name, tier_filter in [
                ("good", scoped["quality_label"] == "good"),
                ("good_mua", scoped["quality_label"].isin(["good", "mua"])),
            ]:
                tiered = scoped[tier_filter]
                for metric in METRICS:
                    label = f"{day_stage}_{scope_name}_{tier_name}_{metric}"
                    print(f"=== {label} ===")
                    try:
                        row = run_combination(tiered, metric, unit_col, label)
                    except Exception as exc:
                        row = {"label": label, "error": str(exc)}
                    row.update({"day_stage": day_stage, "scope": scope_name, "tier": tier_name, "metric": metric})
                    results.append(row)

    summary = pd.DataFrame(results)
    summary.to_csv(ARTIFACTS_DIR / "statistics_summary_posthit.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label", "n_units", "n_mice", "mannwhitney_p", "welch_p", "prop_mannwhitney_p", "prop_welch_p",
            "ks_p_mouseblock", "permanova_p", "n_areas_tested", "n_areas_nominal_p_lt_05"]
    cols = [c for c in cols if c in summary.columns]
    print("\n" + summary[cols].to_string(index=False))
    print(f"\nWrote {ARTIFACTS_DIR / 'statistics_summary_posthit.csv'}")


if __name__ == "__main__":
    main()
