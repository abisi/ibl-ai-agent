"""Re-extract target_region from raw NWB files (dropped by the ssl_ks2_ephys
builder when it flattened electrode_group to a name string), merge onto
units.parquet, run allen_utils.process_allen_labels, and filter to areas
shared across cohorts with enough coverage.

Area-inclusion rule (as specified): a session "qualifies" for an area if it
has >=5 good/mua units there; an area is kept if, among qualifying sessions,
>=2 mice per cohort (R+ and R-) contribute, and total qualifying units
(pooled across cohorts) >= 30.
"""
import sys
import pandas as pd
from pynwb import NWBHDF5IO

sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")
import allen_utils as allen

SSL_KS2_DIR = "reports/datasets/ssl_ks2_ephys/1.0.0"
NWB_ROOT = r"M:\analysis\Axel_Bisi\NWB_combined"
MIN_MICE_PER_COHORT = 2
MIN_UNITS_PER_SESSION = 5
MIN_TOTAL_UNITS = 30

mouse_split = pd.read_csv("projects/ssl-reward-history-modulation/artifacts/mouse_split.csv")
sessions = pd.read_parquet(f"{SSL_KS2_DIR}/metadata/sessions.parquet")
day0 = sessions[
    (sessions["session_description"] == "whisker_0")
    & sessions["has_ephys"]
    & sessions["subject_id"].isin(mouse_split["mouse_id"])
][["session_id", "subject_id", "source_filename"]]

units = pd.read_parquet(f"{SSL_KS2_DIR}/metadata/units.parquet")
units = units[units["session_id"].isin(day0["session_id"])].copy()
print(f"{len(units)} units across {units['session_id'].nunique()} day-0 sessions")

# --- Re-extract target_region per (session_id, electrode_group) from raw NWB ---
target_region_rows = []
for session_id, source_filename in day0[["session_id", "source_filename"]].itertuples(index=False):
    path = f"{NWB_ROOT}\\{source_filename}"
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwb = io.read()
        for group_name, eg in nwb.electrode_groups.items():
            location = eval(eg.location.replace("nan", "None"))
            target_region_rows.append({
                "session_id": session_id, "electrode_group": group_name, "target_region": location.get("area"),
            })
target_region = pd.DataFrame(target_region_rows)
print(f"Extracted target_region for {len(target_region)} (session, electrode_group) pairs")

units = units.merge(target_region, on=["session_id", "electrode_group"], how="left")
print(f"target_region missing for {units['target_region'].isna().sum()}/{len(units)} units")

units = units.merge(day0[["session_id", "subject_id"]], on="session_id").rename(columns={"subject_id": "mouse_id"})
units = units.merge(mouse_split[["mouse_id", "reward_group"]], on="mouse_id")

# --- allen_utils custom area labeling ---
units = allen.process_allen_labels(units, subdivide_areas=False)

# --- Shared-area filtering ---
qualifying = units[units["bc_label"].isin(["good", "mua"])].copy()
units_per_session_area = qualifying.groupby(["area_acronym_custom", "session_id", "mouse_id", "reward_group"]).size().rename("n_units").reset_index()
units_per_session_area = units_per_session_area[units_per_session_area["n_units"] >= MIN_UNITS_PER_SESSION]

mice_per_cohort = units_per_session_area.groupby(["area_acronym_custom", "reward_group"])["mouse_id"].nunique().unstack(fill_value=0)
total_units = units_per_session_area.groupby("area_acronym_custom")["n_units"].sum()

kept_areas = mice_per_cohort.index[
    (mice_per_cohort.get("R+", 0) >= MIN_MICE_PER_COHORT)
    & (mice_per_cohort.get("R-", 0) >= MIN_MICE_PER_COHORT)
    & (total_units.reindex(mice_per_cohort.index).fillna(0) >= MIN_TOTAL_UNITS)
]
print(f"Kept {len(kept_areas)}/{len(mice_per_cohort)} areas: {sorted(kept_areas)}")

units_filtered = units[units["area_acronym_custom"].isin(kept_areas)]
print(f"{len(units_filtered)}/{len(units)} units retained after area filtering")

units_filtered.to_parquet("projects/ssl-reward-history-modulation/artifacts/units_with_area.parquet", index=False)
