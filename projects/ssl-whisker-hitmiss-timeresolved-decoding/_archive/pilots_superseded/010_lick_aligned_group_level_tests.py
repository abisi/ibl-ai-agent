"""Formal group-level significance tests on the completed lick_time-aligned
modality (whisker vs auditory) decode sweep (`area_group` scheme, entire
89-session learning-stage cohort). Mirrors `004_pilot_group_level_tests.py`
exactly (same generic, target-agnostic test machinery), applied here to the
lick-aligned curves instead of the start_time-aligned lick_flag ones.
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
PARTIAL_PATH = OUT_DIR / "008_lick_aligned_results_partial.parquet"
MIN_SESSIONS_PER_COHORT = 5
N_PERM = 2000


def main():
    df = pd.read_parquet(PARTIAL_PATH)
    df = df[(df.area_col == "area_group") & df.skipped_reason.isna()].copy()
    rng = np.random.default_rng(9)

    results = []
    for area in sorted(df.area_value.unique()):
        for half in ("first", "second"):
            sub = df[(df.area_value == area) & (df.half == half)]
            n_by_group = sub.groupby("reward_group")["session_id"].nunique()
            n_rplus = int(n_by_group.get("R+", 0))
            n_rminus = int(n_by_group.get("R-", 0))
            if n_rplus < MIN_SESSIONS_PER_COHORT or n_rminus < MIN_SESSIONS_PER_COHORT:
                print(f"{area} [{half}]: skipped (n_R+={n_rplus}, n_R-={n_rminus}, need >={MIN_SESSIONS_PER_COHORT} each)")
                continue

            recs = [
                dict(subject_id=r.subject_id, reward_group=r.reward_group,
                     real_curve=np.array(r.real_curve), surrogate_curve=np.array(r.surrogate_curve))
                for r in sub.itertuples()
            ]

            above_chance = {}
            for cohort in ("R+", "R-"):
                cohort_recs = [r for r in recs if r["reward_group"] == cohort]
                obs, null_curves = group_mouseblock_permutation_null(cohort_recs, rng, n_perm=N_PERM)
                res = cluster_permutation_test(obs, null_curves)
                above_chance[cohort] = dict(
                    n_sessions=len(cohort_recs), n_mice=len(set(r["subject_id"] for r in cohort_recs)),
                    peak_acc=float(np.nanmax(obs)), cluster_p=res["p_value"],
                )

            diff_obs, diff_null = rplus_rminus_curve_difference(recs, rng, n_perm=N_PERM)
            diff_res = cluster_permutation_test(diff_obs, diff_null, two_sided=True)

            summary_df = pd.DataFrame([dict(subject_id=r["subject_id"], reward_group=r["reward_group"],
                                             metric=float(np.nanmax(r["real_curve"]))) for r in recs])
            summary_res = rplus_rminus_group_test(summary_df, rng, n_perm=N_PERM)

            row = dict(
                area=area, half=half,
                n_rplus_sessions=above_chance["R+"]["n_sessions"], n_rplus_mice=above_chance["R+"]["n_mice"],
                n_rminus_sessions=above_chance["R-"]["n_sessions"], n_rminus_mice=above_chance["R-"]["n_mice"],
                rplus_peak_acc=above_chance["R+"]["peak_acc"], rplus_above_chance_p=above_chance["R+"]["cluster_p"],
                rminus_peak_acc=above_chance["R-"]["peak_acc"], rminus_above_chance_p=above_chance["R-"]["cluster_p"],
                curve_diff_cluster_p=diff_res["p_value"], curve_diff_max_cluster_sign=diff_res["observed_max_cluster_sign"],
                summary_mannwhitney_p=summary_res["mannwhitney_p"], summary_welch_p=summary_res["welch_p"],
                summary_mouseblock_perm_p=summary_res["mouse_block_perm_p"],
                summary_observed_diff=summary_res["observed_diff"],
            )
            results.append(row)
            print(
                f"{area} [{half}]: R+ n={row['n_rplus_sessions']}sess/{row['n_rplus_mice']}mice peak={row['rplus_peak_acc']:.3f} p={row['rplus_above_chance_p']:.4f} | "
                f"R- n={row['n_rminus_sessions']}sess/{row['n_rminus_mice']}mice peak={row['rminus_peak_acc']:.3f} p={row['rminus_above_chance_p']:.4f} | "
                f"curve-diff cluster p={row['curve_diff_cluster_p']:.4f} (sign={row['curve_diff_max_cluster_sign']:+d}) | "
                f"summary MW p={row['summary_mannwhitney_p']:.4f} Welch p={row['summary_welch_p']:.4f} mouse-block p={row['summary_mouseblock_perm_p']:.4f}"
            )

    out = pd.DataFrame(results)
    out.to_csv(OUT_DIR / "010_lick_aligned_group_level_test_results.csv", index=False)
    print(f"\nsaved {len(out)} rows to 010_lick_aligned_group_level_test_results.csv")


if __name__ == "__main__":
    main()
