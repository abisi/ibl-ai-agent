"""Full-scale burstiness metric computation: both day-stages, all mice
passing the mandatory filters. See ../question.md for the full spec and
../artifacts/intermediate_report.html for the validated small-subset check
this scales up from.

Low-spike-count handling (locked here, not explicitly re-confirmed by the
user beyond "go" -- stated clearly for correction if wrong): the raw
per-unit metric is always computed and stored as specified (fB=0 convention
when a window has 0 spikes), but a separate `*_valid` boolean flags whether
that metric is stable enough to use in the statistics matrix -- a unit with
only 1-2 spikes in a window can hit burst_index=+-1.0 by construction (found
during the intermediate checkpoint), so statistics use only the *_valid
subset per metric:
  - continuous_burstiness_valid: n_spikes_clean >= MIN_SPIKES_CONTINUOUS (50)
  - burst_index_whisker_valid: >=15 spikes in BOTH response and baseline windows
  - burst_index_auditory_valid: same, auditory windows
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

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MIN_WINDOW_SPIKES = 15
MIN_SPIKES_CONTINUOUS = 50


def list_all_nwb_files() -> list[str]:
    return [str(f) for f in sorted(loader.NWB_ROOT.glob("*.nwb"))]


def compute_metrics_full(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> pd.DataFrame:
    # Precompute each session's trial-start arrays once, then iterate all
    # units in a single tqdm-wrapped loop (per-session groupby only used to
    # look up the right arrays) -- gives one overall progress bar instead of
    # a print every 20 sessions.
    session_starts = {}
    for session_id, sess_trials in trial_table.groupby("session_id"):
        wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
        au_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "auditory_trial", "start_time"].to_numpy(float))
        session_starts[session_id] = (wh_starts, au_starts)

    rows = []
    for t in tqdm(unit_table.itertuples(), total=len(unit_table), desc="Computing burstiness metrics"):
        wh_starts, au_starts = session_starts[t.session_id]
        spikes = np.sort(np.asarray(t.spike_times, dtype=float))
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
    return df


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    all_files = list_all_nwb_files()
    print(f"{len(all_files)} total NWB files under {loader.NWB_ROOT}")

    for day_stage in ["learning", "expert"]:
        print(f"\n=== {day_stage} arm ===")
        unit_table, trial_table = loader.load_arm(day_stage, all_files)
        unit_table = loader.apply_mouse_filters(unit_table, ref_df)
        unit_table = unit_table[unit_table["quality_label"] != "non-soma"].copy()
        print(f"  {len(unit_table)} units, {unit_table['session_id'].nunique()} sessions, "
              f"{unit_table['mouse_id'].nunique()} mice after filters")

        metrics = compute_metrics_full(unit_table, trial_table)
        out_path = ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics.parquet"
        metrics.to_parquet(out_path, index=False)
        print(f"  Wrote {len(metrics)} units to {out_path}")
        print(metrics[["quality_label", "reward_group"]].value_counts())
        print(f"  continuous_burstiness_valid: {metrics['continuous_burstiness_valid'].mean():.1%}")
        print(f"  burst_index_whisker_valid: {metrics['burst_index_whisker_valid'].mean():.1%}")
        print(f"  burst_index_auditory_valid: {metrics['burst_index_auditory_valid'].mean():.1%}")


if __name__ == "__main__":
    main()
