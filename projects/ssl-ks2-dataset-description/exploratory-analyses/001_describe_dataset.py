"""Thorough description of ssl_ks2_ephys, split by day-stage (learning=day 0,
expert=day>0) and cohort (reward_group / wh_reward).

Must be run with unit_spikes_analysis's own venv python (for allen_utils).
Requires ../artifacts/session_metadata_extra.parquet from
000_extract_session_metadata.py.

"Insertion" = one (session_id, probe_name) pair -- one physical Neuropixels
probe placement for that session. A given insertion is counted under every
brain-structure "large area group" (allen_utils.get_custom_area_groups) that
at least one of its units maps to (an insertion can span multiple areas).
"""
import os
import sys

sys.path.insert(0, r"M:\analysis\Axel_Bisi\unit_spikes_analysis")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allen_utils as allen

DATASET_DIR = r"C:\Users\bisi\Github\int-brain-lab\ibl-ai-agent\reports\datasets\ssl_ks2_ephys\1.0.0"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

units = pd.read_parquet(os.path.join(DATASET_DIR, "metadata", "units.parquet"))
meta = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "session_metadata_extra.parquet"))

units = units.merge(meta[["session_id", "day", "day_stage", "reward_group"]], on="session_id", how="inner")

# allen_utils prefers ccf_atlas_acronym/ccf_atlas_parent_acronym whenever those
# columns are present, regardless of null values, and crashes on NaN (~1% of
# units here have no CCF-atlas registration, though the non-atlas ccf_acronym
# is always populated). Drop the atlas columns for this call so it falls back
# to the complete ccf_acronym/ccf_parent_acronym columns instead.
units_for_area = units.drop(columns=["ccf_atlas_acronym", "ccf_atlas_parent_acronym"], errors="ignore")
units_for_area = allen.create_area_custom_column(units_for_area)
units["area_acronym_custom"] = units_for_area["area_acronym_custom"]
units["area_group"] = units["area_acronym_custom"].map(allen.get_custom_area_groups_from_name()).fillna("Unassigned")
units["insertion_id"] = units["session_id"] + "::" + units["probe_name"].astype(str)

print(f"Total units after merge: {len(units)} (dropped {meta.shape[0] - units['session_id'].nunique()} sessions with unparsed day/reward_group, if any)")
print()

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)

summary_rows = []
for (day_stage, cohort), g in units.groupby(["day_stage", "reward_group"], dropna=False):
    n_sessions = g["session_id"].nunique()
    n_subjects = g["subject_id"].nunique() if "subject_id" in g.columns else meta[meta["session_id"].isin(g["session_id"].unique())]["subject_id"].nunique()
    n_insertions = g["insertion_id"].nunique()
    n_units = len(g)
    units_per_insertion = g.groupby("insertion_id").size()
    bc_counts = g["bc_label"].value_counts().to_dict()

    print(f"=== day_stage={day_stage}, cohort(reward_group)={cohort} ===")
    print(f"  sessions={n_sessions}  subjects={n_subjects}  insertions={n_insertions}  units={n_units}")
    print(f"  units per insertion: mean={units_per_insertion.mean():.1f}  median={units_per_insertion.median():.0f}  "
          f"min={units_per_insertion.min()}  max={units_per_insertion.max()}")
    print(f"  bc_label: {bc_counts}")

    area_counts = g.groupby("area_group")["insertion_id"].nunique().sort_values(ascending=False)
    area_units = g.groupby("area_group").size().reindex(area_counts.index)
    print("  insertions and units per area group:")
    for area in area_counts.index:
        print(f"    {area:30s} insertions={area_counts[area]:3d}  units={area_units[area]:5d}")
    print()

    summary_rows.append({
        "day_stage": day_stage, "reward_group": cohort, "n_sessions": n_sessions, "n_subjects": n_subjects,
        "n_insertions": n_insertions, "n_units": n_units,
        "units_per_insertion_mean": units_per_insertion.mean(), "units_per_insertion_median": units_per_insertion.median(),
        "n_good": bc_counts.get("good", 0), "n_mua": bc_counts.get("mua", 0), "n_non_soma": bc_counts.get("non-soma", 0),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(ARTIFACTS_DIR, "day_cohort_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"saved {summary_path}")

area_table = (
    units.groupby(["day_stage", "reward_group", "area_group"])["insertion_id"].nunique()
    .reset_index(name="n_insertions")
)
area_table_path = os.path.join(ARTIFACTS_DIR, "area_group_insertion_counts.csv")
area_table.to_csv(area_table_path, index=False)
print(f"saved {area_table_path}")

# ---------------- Bar plot: insertions per area group, by day_stage x cohort ----------------
pivot = area_table.pivot_table(index="area_group", columns=["day_stage", "reward_group"], values="n_insertions", fill_value=0)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(11, 6))
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("n insertions")
ax.set_title("Insertions per brain-structure area group, by day-stage x cohort (ssl_ks2_ephys)")
ax.legend(title="(day_stage, reward_group)", fontsize=7, ncol=2)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "001_insertions_per_area_by_daystage_cohort.png"), dpi=140, bbox_inches="tight")
print("saved 001_insertions_per_area_by_daystage_cohort.png")

# ---------------- Bar plot: bc_label composition by day_stage x cohort ----------------
bc_table = units.groupby(["day_stage", "reward_group", "bc_label"]).size().reset_index(name="n_units")
bc_pivot = bc_table.pivot_table(index=["day_stage", "reward_group"], columns="bc_label", values="n_units", fill_value=0)
fig, ax = plt.subplots(figsize=(7, 4))
bc_pivot.plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("n units")
ax.set_title("Unit QC composition (bc_label) by day-stage x cohort")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "001_bc_label_by_daystage_cohort.png"), dpi=140, bbox_inches="tight")
print("saved 001_bc_label_by_daystage_cohort.png")

print("\nDONE")
