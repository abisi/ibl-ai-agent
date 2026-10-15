"""Master group-level test runner across all planned variants:
target (start_time-aligned lick_flag / lick_time-aligned modality) x
day_stage (learning / expert) x area scheme (area_group / area_acronym_custom)
x population scope (entire-dataset / learners-only).

For each (variant, area scheme, scope, area, half) cell with >=5 sessions
per cohort: above-chance test (each cohort), R+/R- summary-metric test
(peak accuracy), R+/R- per-bin curve-difference cluster test, and --
**new, start_time-aligned variants only** -- a fixed-window test: mean
accuracy in bins whose center falls in [0, 50]ms post-stimulus, R+/R-
tested the same way as the peak-accuracy summary metric (Mann-Whitney +
Welch + mouse-block permutation). This window has no natural analog for
the lick_time-aligned variants (bins aren't anchored to the stimulus
there), so it's skipped for those, not approximated.

Only processes datasets whose parquet file actually exists yet (a variant
sweep that hasn't been run/finished is skipped with a printed note, not an
error) -- safe to re-run as more sweeps complete.
"""

from __future__ import annotations

import json
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
POST_STIM_WINDOW_S = (0.0, 0.05)

DATASETS = [
    dict(name="start_time_learning", parquet="002_pilot_results_partial.parquet", bin_edges="002_bin_edges.json", post_stim_test=True),
    dict(name="start_time_expert", parquet="002_pilot_results_partial_expert.parquet", bin_edges="002_bin_edges_expert.json", post_stim_test=True),
    dict(name="lick_time_learning", parquet="008_lick_aligned_results_partial.parquet", bin_edges="008_bin_edges.json", post_stim_test=False),
    dict(name="lick_time_expert", parquet="008_lick_aligned_results_partial_expert.parquet", bin_edges="008_bin_edges_expert.json", post_stim_test=False),
]


def bin_centers_for(bin_edges_path: Path) -> np.ndarray:
    edges = json.loads(bin_edges_path.read_text())
    return np.array([(b[0] + b[1]) / 2 for b in edges])


def run_one_cell(recs: list[dict], rng: np.random.Generator, bin_centers: np.ndarray, post_stim_test: bool) -> dict:
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

    peak_df = pd.DataFrame([dict(subject_id=r["subject_id"], reward_group=r["reward_group"],
                                  metric=float(np.nanmax(r["real_curve"]))) for r in recs])
    peak_res = rplus_rminus_group_test(peak_df, rng, n_perm=N_PERM)

    row = dict(
        n_rplus_sessions=above_chance["R+"]["n_sessions"], n_rplus_mice=above_chance["R+"]["n_mice"],
        n_rminus_sessions=above_chance["R-"]["n_sessions"], n_rminus_mice=above_chance["R-"]["n_mice"],
        rplus_peak_acc=above_chance["R+"]["peak_acc"], rplus_above_chance_p=above_chance["R+"]["cluster_p"],
        rminus_peak_acc=above_chance["R-"]["peak_acc"], rminus_above_chance_p=above_chance["R-"]["cluster_p"],
        curve_diff_cluster_p=diff_res["p_value"], curve_diff_max_cluster_sign=diff_res["observed_max_cluster_sign"],
        peak_summary_mannwhitney_p=peak_res["mannwhitney_p"], peak_summary_welch_p=peak_res["welch_p"],
        peak_summary_mouseblock_perm_p=peak_res["mouse_block_perm_p"], peak_summary_observed_diff=peak_res["observed_diff"],
    )

    if post_stim_test:
        mask = (bin_centers >= POST_STIM_WINDOW_S[0]) & (bin_centers < POST_STIM_WINDOW_S[1])
        if mask.sum() == 0:
            row.update(poststim50_mannwhitney_p=np.nan, poststim50_welch_p=np.nan, poststim50_mouseblock_perm_p=np.nan, poststim50_observed_diff=np.nan)
        else:
            poststim_df = pd.DataFrame([dict(subject_id=r["subject_id"], reward_group=r["reward_group"],
                                              metric=float(np.nanmean(r["real_curve"][mask]))) for r in recs])
            poststim_res = rplus_rminus_group_test(poststim_df, rng, n_perm=N_PERM)
            row.update(
                poststim50_mannwhitney_p=poststim_res["mannwhitney_p"], poststim50_welch_p=poststim_res["welch_p"],
                poststim50_mouseblock_perm_p=poststim_res["mouse_block_perm_p"], poststim50_observed_diff=poststim_res["observed_diff"],
            )
    return row


def main():
    rng = np.random.default_rng(21)
    all_results = []

    for ds in DATASETS:
        parquet_path = OUT_DIR / ds["parquet"]
        bin_edges_path = OUT_DIR / ds["bin_edges"]
        if not parquet_path.exists():
            print(f"=== {ds['name']}: SKIPPED (not yet computed: {ds['parquet']}) ===")
            continue
        print(f"=== {ds['name']} ===")
        df_full = pd.read_parquet(parquet_path)
        bin_centers = bin_centers_for(bin_edges_path)

        for area_col in ("area_group", "area_acronym_custom"):
            for scope in ("entire", "learners_only"):
                df = df_full[(df_full.area_col == area_col) & df_full.skipped_reason.isna()].copy()
                if scope == "learners_only":
                    df = df[df.learning_category.isin(["good", "moderate"])]
                if len(df) == 0:
                    continue
                for area in sorted(df.area_value.unique()):
                    for half in ("first", "second"):
                        sub = df[(df.area_value == area) & (df.half == half)]
                        n_by_group = sub.groupby("reward_group")["session_id"].nunique()
                        if int(n_by_group.get("R+", 0)) < MIN_SESSIONS_PER_COHORT or int(n_by_group.get("R-", 0)) < MIN_SESSIONS_PER_COHORT:
                            continue
                        recs = [
                            dict(subject_id=r.subject_id, reward_group=r.reward_group,
                                 real_curve=np.array(r.real_curve), surrogate_curve=np.array(r.surrogate_curve))
                            for r in sub.itertuples()
                        ]
                        row = run_one_cell(recs, rng, bin_centers, ds["post_stim_test"])
                        row.update(dataset=ds["name"], area_col=area_col, scope=scope, area=area, half=half)
                        all_results.append(row)
                        print(f"  [{area_col}/{scope}] {area} [{half}]: n_R+={row['n_rplus_sessions']} n_R-={row['n_rminus_sessions']} "
                              f"curve_diff_p={row['curve_diff_cluster_p']:.4f} "
                              + (f"poststim50_p={row.get('poststim50_mouseblock_perm_p', float('nan')):.4f}" if ds["post_stim_test"] else ""))

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_DIR / "012_master_test_results.csv", index=False)
    print(f"\nsaved {len(out)} rows to 012_master_test_results.csv")


if __name__ == "__main__":
    main()
