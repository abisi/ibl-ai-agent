"""Full-population burstiness metrics, restricted to data at/after each
session's first cohort-corrected whisker-trial hit (see 016_posthit_lib.py).
Mirrors 003_full_scale_compute.py exactly except for this restriction.
Sessions with no qualifying hit are excluded entirely (not silently given an
unrestricted window) -- counted and reported.
"""
from __future__ import annotations

import sys
import importlib
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

burst_lib = importlib.import_module("000_burst_lib")
loader = importlib.import_module("001_small_subset_load")
posthit = importlib.import_module("016_posthit_lib")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MIN_WINDOW_SPIKES = 15
MIN_SPIKES_CONTINUOUS = 50


def list_all_nwb_files() -> list[str]:
    return [str(f) for f in sorted(loader.NWB_ROOT.glob("*.nwb"))]


def compute_metrics_posthit(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    hit_times = posthit.session_first_hit_times(unit_table, trial_table)
    n_no_hit = sum(v is None for v in hit_times.values())
    print(f"  {n_no_hit}/{len(hit_times)} sessions have NO qualifying cohort-corrected hit -- excluded entirely")

    session_bounds = {}
    for session_id, sess_trials in trial_table.groupby("session_id"):
        t_restrict = hit_times.get(session_id)
        if t_restrict is None:
            continue
        wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
        au_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "auditory_trial", "start_time"].to_numpy(float))
        wh_starts = posthit.restrict_starts(wh_starts, t_restrict)
        au_starts = posthit.restrict_starts(au_starts, t_restrict)
        session_bounds[session_id] = (wh_starts, au_starts, t_restrict)

    rows = []
    unit_table = unit_table[unit_table["session_id"].isin(session_bounds.keys())]
    for t in tqdm(unit_table.itertuples(), total=len(unit_table), desc="Computing post-hit burstiness metrics"):
        wh_starts, au_starts, t_restrict = session_bounds[t.session_id]
        spikes = np.sort(np.asarray(t.spike_times, dtype=float))
        spikes = posthit.restrict_spikes(spikes, t_restrict)
        burstiness, clean, is_burst = burst_lib.continuous_burstiness(spikes, wh_starts)
        wh = burst_lib.burst_index(spikes, wh_starts, wh_starts, is_whisker=True)
        au = burst_lib.burst_index(spikes, wh_starts, au_starts, is_whisker=False)
        rows.append({
            "mouse_id": t.mouse_id, "session_id": t.session_id, "cluster_id": t.cluster_id,
            "electrode_group": t.electrode_group, "unit_uid": t.unit_uid,
            "day_stage": t.day_stage, "quality_label": t.quality_label,
            "area_acronym_custom": t.area_acronym_custom, "reward_group": t.reward_group,
            "learning_category": t.learning_category, "firing_rate": getattr(t, "firing_rate", np.nan),
            "continuous_burstiness": burstiness, "n_spikes_clean": len(clean),
            "burst_index_whisker": wh["burst_index"],
            "n_total_response_whisker": wh["n_total_response"], "n_total_baseline_whisker": wh["n_total_baseline"],
            "burst_index_auditory": au["burst_index"],
            "n_total_response_auditory": au["n_total_response"], "n_total_baseline_auditory": au["n_total_baseline"],
        })
    df = pd.DataFrame(rows)
    df["continuous_burstiness_valid"] = df["n_spikes_clean"] >= MIN_SPIKES_CONTINUOUS
    df["burst_index_whisker_valid"] = (df["n_total_response_whisker"] >= MIN_WINDOW_SPIKES) & (df["n_total_baseline_whisker"] >= MIN_WINDOW_SPIKES)
    df["burst_index_auditory_valid"] = (df["n_total_response_auditory"] >= MIN_WINDOW_SPIKES) & (df["n_total_baseline_auditory"] >= MIN_WINDOW_SPIKES)
    meta = {"n_sessions_total": len(hit_times), "n_sessions_no_hit": n_no_hit}
    return df, meta


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    all_files = list_all_nwb_files()
    print(f"{len(all_files)} total NWB files under {loader.NWB_ROOT}")

    all_meta = {}
    for day_stage in ["learning", "expert"]:
        print(f"\n=== {day_stage} arm (post-first-hit) ===")
        unit_table, trial_table = loader.load_arm(day_stage, all_files)
        unit_table = loader.apply_mouse_filters(unit_table, ref_df)
        unit_table = unit_table[unit_table["quality_label"] != "non-soma"].copy()
        print(f"  {len(unit_table)} units, {unit_table['session_id'].nunique()} sessions, "
              f"{unit_table['mouse_id'].nunique()} mice after filters (before hit-restriction)")

        metrics, meta = compute_metrics_posthit(unit_table, trial_table)
        all_meta[day_stage] = meta
        out_path = ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics_posthit.parquet"
        metrics.to_parquet(out_path, index=False)
        print(f"  Wrote {len(metrics)} units to {out_path}")
        print(metrics[["quality_label", "reward_group"]].value_counts())

    import json
    with open(ARTIFACTS_DIR / "posthit_run_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)


if __name__ == "__main__":
    main()
