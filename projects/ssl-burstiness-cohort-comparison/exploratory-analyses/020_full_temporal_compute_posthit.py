"""Full-population temporal-dynamics computation, restricted to post-first-
hit data -- mirrors 013_full_temporal_compute.py, except "session
progression" is now rebased to run from [first-hit time, session end]
instead of [session start, session end]. Sessions with no qualifying hit
are excluded (same as 017).
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
temporal = importlib.import_module("010_temporal_dynamics_preview")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_BINS = temporal.N_BINS
ABS_N_BINS = temporal.ABS_N_BINS


def list_all_nwb_files() -> list[str]:
    return [str(f) for f in sorted(loader.NWB_ROOT.glob("*.nwb"))]


def compute_temporal_posthit(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> pd.DataFrame:
    hit_times = posthit.session_first_hit_times(unit_table, trial_table)
    n_no_hit = sum(v is None for v in hit_times.values())
    print(f"  {n_no_hit}/{len(hit_times)} sessions excluded (no qualifying hit)")

    session_bounds = {}
    for session_id, sess_trials in trial_table.groupby("session_id"):
        t_restrict = hit_times.get(session_id)
        if t_restrict is None:
            continue
        wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
        wh_starts = posthit.restrict_starts(wh_starts, t_restrict)
        t_end = float(sess_trials["stop_time"].max())
        session_bounds[session_id] = (wh_starts, t_restrict, t_end)

    unit_table = unit_table[unit_table["session_id"].isin(session_bounds.keys())]
    rows = []
    for t in tqdm(unit_table.itertuples(), total=len(unit_table), desc="Binning post-hit burstiness"):
        wh_starts, t_start, t_end = session_bounds[t.session_id]
        spikes = np.sort(np.asarray(t.spike_times, dtype=float))
        spikes = posthit.restrict_spikes(spikes, t_start)
        _, clean, is_burst = burst_lib.continuous_burstiness(spikes, wh_starts)
        if len(clean) < 20:
            continue
        norm_bins = temporal.normalized_bin_burstiness(clean, is_burst, t_start, t_end, n_bins=N_BINS)
        abs_bins = temporal.absolute_bin_burstiness(clean, is_burst, t_start, t_end, n_bins=ABS_N_BINS)
        row = {
            "unit_uid": t.unit_uid, "mouse_id": t.mouse_id, "session_id": t.session_id,
            "day_stage": t.day_stage, "quality_label": t.quality_label,
            "reward_group": t.reward_group, "learning_category": t.learning_category,
        }
        row.update({f"norm_bin_{i}": norm_bins[i] for i in range(N_BINS)})
        row.update({f"abs_bin_{i}": abs_bins[i] for i in range(ABS_N_BINS)})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    all_files = list_all_nwb_files()
    print(f"{len(all_files)} total NWB files under {loader.NWB_ROOT}")

    for day_stage in ["learning", "expert"]:
        print(f"\n=== {day_stage} arm (post-first-hit) ===")
        unit_table, trial_table = loader.load_arm(day_stage, all_files)
        unit_table = loader.apply_mouse_filters(unit_table, ref_df)
        unit_table = unit_table[unit_table["quality_label"].isin(["good", "mua"])].copy()
        print(f"  {len(unit_table)} units, {unit_table['session_id'].nunique()} sessions, "
              f"{unit_table['mouse_id'].nunique()} mice after filters (before hit-restriction)")

        temporal_df = compute_temporal_posthit(unit_table, trial_table)
        out_path = ARTIFACTS_DIR / f"full_{day_stage}_temporal_metrics_posthit.parquet"
        temporal_df.to_parquet(out_path, index=False)
        print(f"  Wrote {len(temporal_df)} units to {out_path}")


if __name__ == "__main__":
    main()
