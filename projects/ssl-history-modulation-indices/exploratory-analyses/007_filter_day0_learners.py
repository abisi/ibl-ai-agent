"""Filter the full-cohort RHMI/EHMI unit table (004_compute_full_cohort.py
output) to day 0 (day_stage=='learning') sessions and "learner" mice only,
per user request 2026-08-22 and established SSL practice
(ssl_task_semantics.md's Cohort/reward_group section): drop
learning_category in {bad, NaN} and require exclude_ephys==0, both from
joint_mouse_reference_weight.xlsx (reports/ssl_analysis/derived/mouse_reference.parquet).
"""
from pathlib import Path

import pandas as pd

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/full_cohort_unit_indices.parquet")
REF_PATH = Path("reports/ssl_analysis/derived/mouse_reference.parquet")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/day0_learners_unit_indices.parquet")


def main():
    df = pd.read_parquet(IN_PATH)
    n0 = len(df)

    df = df[df["day_stage"] == "learning"]
    n_day0 = len(df)

    ref = pd.read_parquet(REF_PATH)[["subject_id", "learning_category", "exclude_ephys"]]
    df = df.merge(ref, left_on="mouse_id", right_on="subject_id", how="left")

    is_learner = df["learning_category"].isin(["good", "moderate"]) & (df["exclude_ephys"] == 0)
    dropped_mice = sorted(df.loc[~is_learner, "mouse_id"].unique())
    df = df[is_learner].drop(columns=["subject_id", "learning_category", "exclude_ephys"])

    print(f"{n0} unit rows (full cohort) -> {n_day0} after day_stage=='learning' -> {len(df)} after learner filter")
    print(f"{len(dropped_mice)} mice dropped by learner filter: {dropped_mice}")
    print(f"{df['session_id'].nunique()} sessions, {df['mouse_id'].nunique()} mice")
    print(df.groupby("reward_group")["mouse_id"].nunique())

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
