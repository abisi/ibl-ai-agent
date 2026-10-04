"""Statistics matrix: day-stage x population-scope x quality-tier x metric
(24 combinations). Global R+/R- test first, then PERMANOVA (mouse-block
permutation) with per-area BH-FDR post-hoc, gated on main significance.
See ../question.md for the full design and ../permanova.py (copied verbatim
from ssl-passive-coselectivity, itself from ssl-reward-history-modulation's
validated mouse-block-permutation fix) for the reused PERMANOVA code.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from permanova import permanova_euclidean, run_with_posthoc

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
METRICS = ["continuous_burstiness", "burst_index_whisker", "burst_index_auditory"]


def global_test(df: pd.DataFrame, metric: str, unit_col: str) -> dict:
    """Non-parametric (Mann-Whitney) + parametric (Welch) R+ vs R- on one
    value per `unit_col` (median across its valid units) -- mouse_id for
    learning, session_id for expert, per the established statistical-unit
    rule (ssl_analysis_patterns.md)."""
    per_unit = df.groupby([unit_col, "reward_group"])[metric].median().reset_index()
    rplus = per_unit.loc[per_unit.reward_group == "R+", metric].to_numpy()
    rminus = per_unit.loc[per_unit.reward_group == "R-", metric].to_numpy()
    if len(rplus) < 2 or len(rminus) < 2:
        return {"n_rplus": len(rplus), "n_rminus": len(rminus), "mannwhitney_p": np.nan, "welch_p": np.nan,
                "median_rplus": np.nan, "median_rminus": np.nan}
    u_stat, u_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)
    return {
        "n_rplus": len(rplus), "n_rminus": len(rminus),
        "median_rplus": float(np.median(rplus)), "median_rminus": float(np.median(rminus)),
        "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
        "welch_t": float(t_stat), "welch_p": float(t_p),
    }


def run_combination(df: pd.DataFrame, metric: str, unit_col: str, label: str) -> dict:
    valid_col = f"{metric}_valid"
    sub = df[df[valid_col] & df[metric].notna()].copy()
    n_units, n_mice = len(sub), sub["mouse_id"].nunique()
    n_rplus_mice = sub.loc[sub.reward_group == "R+", "mouse_id"].nunique()
    n_rminus_mice = sub.loc[sub.reward_group == "R-", "mouse_id"].nunique()
    row = {"label": label, "n_units": n_units, "n_mice": n_mice,
           "n_rplus_mice": n_rplus_mice, "n_rminus_mice": n_rminus_mice}
    if n_rplus_mice < 2 or n_rminus_mice < 2:
        row["skipped_reason"] = "fewer than 2 mice in a cohort after filters"
        return row

    g = global_test(sub, metric, unit_col)
    row.update({f"global_{k}": v for k, v in g.items()})

    # Main PERMANOVA only (pooled across ALL areas) -- the expensive per-area
    # post-hoc (a full permutation test per area, ~40-70 areas) must NOT run
    # unless this main test is significant (explicit user correction:
    # previously called run_with_posthoc unconditionally, which always runs
    # every area's permutation test regardless of main significance -- the
    # actual cause of the multi-hour runtime, not just N_PERM).
    X = sub[[metric]].to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    f_obs, p_val, n_main, n_mice_main = permanova_euclidean(X, sub["reward_group"].to_numpy(), sub["mouse_id"].to_numpy())
    row["permanova_pseudo_F"] = f_obs
    row["permanova_p"] = p_val
    row["permanova_n_units"] = n_main
    row["permanova_n_mice"] = n_mice_main
    row["permanova_significant"] = bool(p_val < 0.05)

    if row["permanova_significant"]:
        _, posthoc = run_with_posthoc(sub, [metric], area_col="area_acronym_custom", label_prefix=label)
        row["n_areas_tested"] = len(posthoc)
        row["n_areas_sig_fdr05"] = int(posthoc["significant_fdr05"].sum()) if len(posthoc) else 0
        posthoc.to_parquet(ARTIFACTS_DIR / f"posthoc_{label}.parquet", index=False)
    else:
        row["n_areas_tested"] = 0
        row["n_areas_sig_fdr05"] = 0
    return row


def main() -> None:
    results = []
    for day_stage, unit_col in [("learning", "mouse_id"), ("expert", "session_id")]:
        df = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics.parquet")
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
                        print(f"  FAILED: {exc}")
                        traceback.print_exc()
                        row = {"label": label, "error": str(exc)}
                    row.update({"day_stage": day_stage, "scope": scope_name, "tier": tier_name, "metric": metric})
                    results.append(row)

    summary = pd.DataFrame(results)
    cols_front = ["label", "day_stage", "scope", "tier", "metric", "n_units", "n_mice",
                  "n_rplus_mice", "n_rminus_mice", "global_mannwhitney_p", "global_welch_p",
                  "permanova_pseudo_F", "permanova_p", "permanova_significant", "n_areas_tested", "n_areas_sig_fdr05"]
    cols = [c for c in cols_front if c in summary.columns] + [c for c in summary.columns if c not in cols_front]
    summary = summary[cols]
    summary.to_csv(ARTIFACTS_DIR / "statistics_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + summary[cols_front].to_string(index=False))
    print(f"\nWrote {ARTIFACTS_DIR / 'statistics_summary.csv'}")


if __name__ == "__main__":
    main()
