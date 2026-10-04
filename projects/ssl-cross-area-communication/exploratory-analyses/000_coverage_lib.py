"""Shared helpers for the recording-pair coverage step: Path B loading (unit
metadata only downstream use), two-level area hierarchy, and the >=20-unit
"valid area recording" floor / valid-pair enumeration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")

import pandas as pd

from ephys_utilities.helpers import data_utils
from ephys_utilities.neural_utils import unit_metrics_utils
import allen_utils

NWB_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_ks4")
REF_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")

THRESHOLDS = unit_metrics_utils.DEFAULT_METRIC_THRESHOLDS
EXCLUDE = ["Lratio", "isolationDistance", "presenceRatio", "maxDriftEstimate"]
MIN_UNITS_PER_AREA = 20

COARSE_GROUP_OF = allen_utils.get_custom_area_groups_from_name()

# v3 coarse grouping (Axel, 2026-08-31 refinements): drops Olfactory areas
# and Amygdala-and-hypothalamus; "Visual areas" -> "PPC" (verified 100% PPC
# units, zero true visual cortex); "Somatosensory areas" split into
# "SSp-w" (SSp-bfd + SSs) and "SSp-orofacial" (SSp-m + SSp-n); residual
# fine areas (AI, SSp-ul, SSp-ll, GU, VISC, and everything not listed
# below) are excluded at the coarse level (NaN) -- present at the fine
# level only. Reverse-engineered from `full_unit_table_metadata_v3.parquet`
# (the `area_group_coarse_v3` column) since the original computation was
# never saved as a reusable script -- this is now that script, and matches
# the cached parquet exactly (verified by rebuilding the Motor-frontal vs
# Striatum session list from a fresh unit_table and diffing against
# `motor_striatum_session_list.csv`: 63/63 sessions, byte-for-byte match).
COARSE_GROUP_V3_OF = {
    "AUD": "Auditory areas", "TEa": "Auditory areas",
    "CA1": "Hippocampus", "CA2": "Hippocampus", "CA3": "Hippocampus", "DG": "Hippocampus", "HPF": "Hippocampus",
    "APN": "Midbrain", "MB": "Midbrain", "MRN": "Midbrain", "PAG": "Midbrain", "RN": "Midbrain",
    "SCm": "Midbrain", "SCs": "Midbrain", "SNr": "Midbrain", "VTA": "Midbrain",
    "FRP": "Motor and frontal areas", "MO-ALM": "Motor and frontal areas", "MO-tjM1": "Motor and frontal areas",
    "MO-wM1": "Motor and frontal areas", "MO-wM2": "Motor and frontal areas", "ORB": "Motor and frontal areas",
    "mPFC": "Motor and frontal areas",
    "PPC": "PPC",
    "Pons": "Pons and medulla",
    "RSP": "Retrosplenial areas",
    "SSp-m": "SSp-orofacial", "SSp-n": "SSp-orofacial",
    "SSp-bfd": "SSp-w", "SSs": "SSp-w",
    "DLS": "Striatum and pallidum", "DMS": "Striatum and pallidum", "GPe": "Striatum and pallidum",
    "GPi": "Striatum and pallidum", "LS": "Striatum and pallidum", "PAL": "Striatum and pallidum",
    "SF": "Striatum and pallidum", "TS": "Striatum and pallidum", "VS": "Striatum and pallidum",
    "ATN": "Thalamus", "HA": "Thalamus", "LAT": "Thalamus", "LGN": "Thalamus", "MED": "Thalamus",
    "MGN": "Thalamus", "MTN": "Thalamus", "RT": "Thalamus", "TH": "Thalamus", "VP": "Thalamus",
}


def load_units(filenames: list[str], day_to_analyze: str = "all", max_workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Path B load: unit_table (+ area labels, quality) and trial_table for
    the given NWB filenames. `day_to_analyze` is one of 'learning'/'expert'/'all'
    (a bare 0/None silently disables day filtering -- ssl_loading_policy.md)."""
    nwb_list = [str(NWB_ROOT / f) for f in filenames]
    trial_table, unit_table, ephys_nwb_list = data_utils.combine_ephys_nwb(
        nwb_list, day_to_analyze=day_to_analyze, max_workers=max_workers
    )
    print(f"[{day_to_analyze}] {len(ephys_nwb_list)}/{len(filenames)} files had ephys; "
          f"{len(unit_table)} units, {len(trial_table)} trials")
    if len(unit_table) == 0:
        return unit_table, trial_table

    unit_table = unit_metrics_utils.classify_units_quality(
        unit_table, thresholds=THRESHOLDS, exclude=EXCLUDE, label_col="quality_label"
    )
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)
    unit_table["area_group_coarse"] = unit_table["area_acronym_custom"].map(COARSE_GROUP_OF)
    unit_table["area_group_coarse_v3"] = unit_table["area_acronym_custom"].map(COARSE_GROUP_V3_OF)
    # unit_uid: cluster_id here is the RAW per-probe KiloSort id from this Path B
    # pipeline (not offset the way the compressed-dataset builder does) -- verified
    # not session-unique in a multi-probe session in the burstiness project.
    unit_table["unit_uid"] = (
        unit_table["session_id"] + "::" + unit_table["electrode_group"].astype(str)
        + "::" + unit_table["cluster_id"].astype(str)
    )
    assert unit_table["unit_uid"].is_unique, "unit_uid not unique"
    return unit_table, trial_table


def apply_mouse_filters(unit_table: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    unit_table = unit_table.rename(columns={"reward_group": "wh_reward_reward_group"})
    merged = unit_table.merge(
        ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group", "learning_category"]],
        on="mouse_id", how="left",
    )
    n_before = len(merged)
    merged = merged[(merged["exclude"] == 0) & (merged["exclude_ephys"] == 0)]
    merged = merged[merged["reward_group"].isin(["R+", "R-"])]
    print(f"  Mouse filters (exclude==0, exclude_ephys==0, drop R+proba): {n_before} -> {len(merged)} units")
    return merged


def area_unit_counts(unit_table: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Per (session_id, area) unit counts, quality-filtered (drop non-soma)."""
    ut = unit_table[unit_table["quality_label"] != "non-soma"]
    counts = ut.groupby(["session_id", area_col]).size().reset_index(name="n_units")
    return counts


def valid_area_pairs(counts: pd.DataFrame, area_col: str, min_units: int = MIN_UNITS_PER_AREA) -> pd.DataFrame:
    """For each session, enumerate unordered pairs of areas that both clear
    the unit floor and are simultaneously recorded in that session."""
    rows = []
    for session_id, sess in counts.groupby("session_id"):
        valid_areas = sess.loc[sess["n_units"] >= min_units, area_col].tolist()
        n_units_by_area = dict(zip(sess[area_col], sess["n_units"]))
        for i in range(len(valid_areas)):
            for j in range(i + 1, len(valid_areas)):
                a, b = sorted([valid_areas[i], valid_areas[j]])
                rows.append({
                    "session_id": session_id, "area_a": a, "area_b": b,
                    "n_units_a": n_units_by_area[a], "n_units_b": n_units_by_area[b],
                })
    return pd.DataFrame(rows)
