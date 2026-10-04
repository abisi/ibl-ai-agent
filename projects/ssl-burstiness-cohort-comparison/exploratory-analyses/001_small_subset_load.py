"""Load a small subset of sessions (learning + expert) via the mandatory
Path B pipeline, apply mouse filters, and compute burstiness metrics per
unit, for intermediate-report validation only (not the full-scale run).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ephys_utilities.helpers import data_utils
from ephys_utilities.neural_utils import unit_metrics_utils
import allen_utils

import importlib
burst_lib = importlib.import_module("000_burst_lib")

NWB_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_ks4")
REF_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

THRESHOLDS = unit_metrics_utils.DEFAULT_METRIC_THRESHOLDS
EXCLUDE = ["Lratio", "isolationDistance", "presenceRatio", "maxDriftEstimate"]

# Small subset: 3 learning-arm + 3 expert-arm known-ephys sessions.
LEARNING_FILES = ["AB080_20230622_152205.nwb", "AB082_20230630_101353.nwb", "AB085_20231005_152636.nwb"]
EXPERT_FILES = ["MH001_20241203_121357.nwb", "MH019_20250306_121844.nwb", "MH021_20250311_110321.nwb"]


def load_arm(day_to_analyze: str, filenames: list[str]) -> pd.DataFrame:
    nwb_list = [str(NWB_ROOT / f) for f in filenames]
    trial_table, unit_table, ephys_nwb_list = data_utils.combine_ephys_nwb(
        nwb_list, day_to_analyze=day_to_analyze, max_workers=4
    )
    print(f"[{day_to_analyze}] {len(ephys_nwb_list)}/{len(filenames)} files had ephys; "
          f"{len(unit_table)} units, {len(trial_table)} trials")
    if len(unit_table) == 0:
        return unit_table, trial_table

    unit_table = unit_metrics_utils.classify_units_quality(
        unit_table, thresholds=THRESHOLDS, exclude=EXCLUDE, label_col="quality_label"
    )
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)
    unit_table["day_stage"] = day_to_analyze
    # cluster_id here is the RAW per-probe KiloSort id (this Path B pipeline does
    # not offset it the way ibl_ai_agent's compressed-dataset builder does) --
    # NOT unique within a multi-probe session (verified: session
    # MH021_20250311_110321 has cluster_id 314 on 3 of its 4 probes). Build an
    # explicitly unique key for every downstream lookup.
    unit_table["unit_uid"] = (
        unit_table["session_id"] + "::" + unit_table["electrode_group"].astype(str) + "::" + unit_table["cluster_id"].astype(str)
    )
    assert unit_table["unit_uid"].is_unique, "unit_uid not unique -- electrode_group+cluster_id doesn't disambiguate probes"
    return unit_table, trial_table


def apply_mouse_filters(unit_table: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    # unit_table already carries a native 'reward_group' from process_single_nwb
    # (session-metadata wh_reward, int 0/1) -- this project's cohort factor is
    # the reference-sheet's mouse-level reward_group instead (see question.md's
    # "Key methodological decision"); rename the native one first so the merge
    # doesn't silently suffix both columns to reward_group_x/_y.
    unit_table = unit_table.rename(columns={"reward_group": "wh_reward_reward_group"})
    merged = unit_table.merge(
        ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group", "learning_category"]],
        on="mouse_id", how="left",
    )
    n_before = len(merged)
    merged = merged[(merged["exclude"] == 0) & (merged["exclude_ephys"] == 0)]
    merged = merged[merged["reward_group"].isin(["R+", "R-"])]  # drop R+proba / unmatched
    print(f"  Mouse filters (exclude==0, exclude_ephys==0, drop R+proba): {n_before} -> {len(merged)} units")
    return merged


def compute_metrics_for_unit(row, whisker_starts, auditory_starts, all_whisker_starts_for_deadzone) -> dict:
    spike_times = np.asarray(row["spike_times"], dtype=float)
    spike_times = np.sort(spike_times)
    burstiness, clean_spikes, is_burst = burst_lib.continuous_burstiness(spike_times, all_whisker_starts_for_deadzone)
    out = {"continuous_burstiness": burstiness, "n_spikes_clean": len(clean_spikes), "n_burst_spikes": int(is_burst.sum())}
    wh = burst_lib.burst_index(spike_times, all_whisker_starts_for_deadzone, whisker_starts, is_whisker=True)
    au = burst_lib.burst_index(spike_times, all_whisker_starts_for_deadzone, auditory_starts, is_whisker=False)
    out["burst_index_whisker"] = wh["burst_index"]
    out["burst_index_auditory"] = au["burst_index"]
    out["fb_response_whisker"] = wh["fb_response"]
    out["fb_baseline_whisker"] = wh["fb_baseline"]
    out["n_total_response_whisker"] = wh["n_total_response"]
    out["n_total_baseline_whisker"] = wh["n_total_baseline"]
    out["fb_response_auditory"] = au["fb_response"]
    out["fb_baseline_auditory"] = au["fb_baseline"]
    out["n_total_response_auditory"] = au["n_total_response"]
    out["n_total_baseline_auditory"] = au["n_total_baseline"]
    return out


def main() -> None:
    ref_df = pd.read_excel(REF_XLSX, sheet_name="Sheet1")

    all_units = []
    for day_stage, files in [("learning", LEARNING_FILES), ("expert", EXPERT_FILES)]:
        unit_table, trial_table = load_arm(day_stage, files)
        if len(unit_table) == 0:
            continue
        unit_table = apply_mouse_filters(unit_table, ref_df)
        unit_table = unit_table[unit_table["quality_label"] != "non-soma"].copy()

        metrics_rows = []
        for session_id, sess_units in unit_table.groupby("session_id"):
            sess_trials = trial_table[trial_table["session_id"] == session_id]
            whisker_starts = np.sort(sess_trials.loc[sess_trials["trial_type"] == "whisker_trial", "start_time"].to_numpy(dtype=float))
            auditory_starts = np.sort(sess_trials.loc[sess_trials["trial_type"] == "auditory_trial", "start_time"].to_numpy(dtype=float))
            print(f"  {session_id}: {len(sess_units)} units, {len(whisker_starts)} whisker trials, {len(auditory_starts)} auditory trials")
            for _, row in sess_units.iterrows():
                m = compute_metrics_for_unit(row, whisker_starts, auditory_starts, whisker_starts)
                m.update({
                    "mouse_id": row["mouse_id"], "session_id": session_id, "cluster_id": row["cluster_id"],
                    "electrode_group": row["electrode_group"], "unit_uid": row["unit_uid"],
                    "day_stage": day_stage, "quality_label": row["quality_label"],
                    "area_acronym_custom": row["area_acronym_custom"], "reward_group": row["reward_group"],
                    "learning_category": row["learning_category"], "firing_rate": row.get("firing_rate", np.nan),
                })
                metrics_rows.append(m)
        all_units.append(pd.DataFrame(metrics_rows))

    result = pd.concat(all_units, ignore_index=True) if all_units else pd.DataFrame()
    out_path = ARTIFACTS_DIR / "small_subset_unit_metrics.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nWrote {len(result)} units to {out_path}")
    print(result[["day_stage", "quality_label", "reward_group"]].value_counts())


if __name__ == "__main__":
    main()
