"""Group-level tests on the performance-state sweep
(`015_perfstate_sweep.py`). For each area (both schemes) x population scope
with enough sessions:

1. **Naive** comparison: R+(state) vs R-(state) for state in {high, low}
   separately -- same state label both cohorts, mirroring
   `012_master_group_level_tests.py`'s half-based comparison but with
   `perf_state` in place of `half`.
2. **Discriminability-matched** comparison (user request, 2026-09-11,
   given the perf/discriminability relationship is reversed between
   cohorts): R+(high) vs R-(low) as the "high-discriminability" pairing,
   R+(low) vs R-(high) as the "low-discriminability" pairing. Uses the
   same generic curve-difference/group-test machinery -- these functions
   only look at each record's own `reward_group` field, so building the
   matched comparison is just a different selection of which records go
   into the R+/R- lists, no new statistical code.

Also reports each cell's mean false-alarm rate (already computed per
block in the sweep) alongside hit rate, as a behavioral sanity check that
the intended perf/discriminability reversal is present in this data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from ssl_timeresolved_decoding import (
    cluster_permutation_test,
    group_mouseblock_permutation_null,
    rplus_rminus_curve_difference,
    rplus_rminus_group_test,
)

OUT_DIR = Path(__file__).resolve().parent
MIN_SESSIONS_PER_COHORT = 5
N_PERM = 2000

DATASETS = [
    dict(name="perfstate_learning", parquet="015_perfstate_results_partial.parquet"),
    dict(name="perfstate_expert", parquet="015_perfstate_results_partial_expert.parquet"),
]


def to_recs(sub: pd.DataFrame) -> list[dict]:
    return [
        dict(subject_id=r.subject_id, reward_group=r.reward_group,
             real_curve=np.array(r.real_curve), surrogate_curve=np.array(r.surrogate_curve))
        for r in sub.itertuples()
    ]


def above_chance(recs: list[dict], rng: np.random.Generator) -> dict:
    obs, null_curves = group_mouseblock_permutation_null(recs, rng, n_perm=N_PERM)
    res = cluster_permutation_test(obs, null_curves)
    return dict(n_sessions=len(recs), n_mice=len(set(r["subject_id"] for r in recs)),
                peak_acc=float(np.nanmax(obs)), cluster_p=res["p_value"])


def compare(recs: list[dict], rng: np.random.Generator) -> dict:
    diff_obs, diff_null = rplus_rminus_curve_difference(recs, rng, n_perm=N_PERM)
    diff_res = cluster_permutation_test(diff_obs, diff_null, two_sided=True)
    peak_df = pd.DataFrame([dict(subject_id=r["subject_id"], reward_group=r["reward_group"],
                                  metric=float(np.nanmax(r["real_curve"]))) for r in recs])
    peak_res = rplus_rminus_group_test(peak_df, rng, n_perm=N_PERM)
    return dict(
        curve_diff_cluster_p=diff_res["p_value"], curve_diff_sign=diff_res["observed_max_cluster_sign"],
        peak_mannwhitney_p=peak_res["mannwhitney_p"], peak_welch_p=peak_res["welch_p"],
        peak_mouseblock_p=peak_res["mouse_block_perm_p"], peak_observed_diff=peak_res["observed_diff"],
    )


def main():
    rng = np.random.default_rng(41)
    all_results = []

    for ds in DATASETS:
        parquet_path = OUT_DIR / ds["parquet"]
        if not parquet_path.exists():
            print(f"=== {ds['name']}: SKIPPED (not yet computed) ===")
            continue
        print(f"=== {ds['name']} ===")
        df_full = pd.read_parquet(parquet_path)

        for area_col in ("area_group", "area_acronym_custom"):
            for scope in ("entire", "learners_only"):
                df = df_full[(df_full.area_col == area_col) & df_full.skipped_reason.isna()].copy()
                if scope == "learners_only":
                    df = df[df.learning_category.isin(["good", "moderate"])]
                if len(df) == 0:
                    continue
                for area in sorted(df.area_value.unique()):
                    sub = df[df.area_value == area]
                    n = sub.groupby(["reward_group", "perf_state"])["session_id"].nunique()
                    if not all(n.get((cohort, state), 0) >= MIN_SESSIONS_PER_COHORT
                               for cohort in ("R+", "R-") for state in ("high", "low")):
                        continue

                    rplus_high = to_recs(sub[(sub.reward_group == "R+") & (sub.perf_state == "high")])
                    rplus_low = to_recs(sub[(sub.reward_group == "R+") & (sub.perf_state == "low")])
                    rminus_high = to_recs(sub[(sub.reward_group == "R-") & (sub.perf_state == "high")])
                    rminus_low = to_recs(sub[(sub.reward_group == "R-") & (sub.perf_state == "low")])

                    fa_by_group = sub.groupby(["reward_group", "perf_state"])["mean_fa_rate"].mean()

                    row = dict(dataset=ds["name"], area_col=area_col, scope=scope, area=area)
                    row["rplus_high_above_chance_p"] = above_chance(rplus_high, rng)["cluster_p"]
                    row["rplus_low_above_chance_p"] = above_chance(rplus_low, rng)["cluster_p"]
                    row["rminus_high_above_chance_p"] = above_chance(rminus_high, rng)["cluster_p"]
                    row["rminus_low_above_chance_p"] = above_chance(rminus_low, rng)["cluster_p"]

                    naive_high = compare(rplus_high + rminus_high, rng)
                    naive_low = compare(rplus_low + rminus_low, rng)
                    matched_highdisc = compare(rplus_high + rminus_low, rng)  # R+ high-perf vs R- low-perf
                    matched_lowdisc = compare(rplus_low + rminus_high, rng)  # R+ low-perf vs R- high-perf

                    for label, res in (("naive_high", naive_high), ("naive_low", naive_low),
                                       ("matched_highdisc", matched_highdisc), ("matched_lowdisc", matched_lowdisc)):
                        for k, v in res.items():
                            row[f"{label}_{k}"] = v

                    row["fa_rplus_high"] = fa_by_group.get(("R+", "high"), np.nan)
                    row["fa_rplus_low"] = fa_by_group.get(("R+", "low"), np.nan)
                    row["fa_rminus_high"] = fa_by_group.get(("R-", "high"), np.nan)
                    row["fa_rminus_low"] = fa_by_group.get(("R-", "low"), np.nan)
                    row["n_rplus_high"] = len(rplus_high)
                    row["n_rplus_low"] = len(rplus_low)
                    row["n_rminus_high"] = len(rminus_high)
                    row["n_rminus_low"] = len(rminus_low)

                    all_results.append(row)
                    print(
                        f"  [{area_col}/{scope}] {area}: n=(R+ hi={row['n_rplus_high']},lo={row['n_rplus_low']}; "
                        f"R- hi={row['n_rminus_high']},lo={row['n_rminus_low']}) "
                        f"naive_high_p={naive_high['curve_diff_cluster_p']:.4f} naive_low_p={naive_low['curve_diff_cluster_p']:.4f} "
                        f"matched_highdisc_p={matched_highdisc['curve_diff_cluster_p']:.4f} matched_lowdisc_p={matched_lowdisc['curve_diff_cluster_p']:.4f}"
                    )

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_DIR / "016_perfstate_test_results.csv", index=False)
    print(f"\nsaved {len(out)} rows to 016_perfstate_test_results.csv")


if __name__ == "__main__":
    main()
