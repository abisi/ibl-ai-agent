"""Per-trial baseline / evoked / evoked-baseline-corrected firing rates for
every retained unit, on passive_pre and passive_post whisker_trial and
auditory_trial trials. Dead-zone-aware (whisker only) via ssl_wm_lib.

Writes two tables:
- trial_window_rates.parquet: per (unit, trial) rows (for QC / variance).
- unit_condition_rates.parquet: per (unit, trial_type, passive_epoch) means
  -- the primary table for the PSTH grid, LMM, and modulation index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssl_wm_lib import BASELINE_WINDOW, EVOKED_WINDOW, per_unit_trial_window_counts, tag_passive_subepoch  # noqa: E402
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
DERIVED_DIR = Path("reports/ssl_analysis/derived")


def main() -> None:
    units = pd.read_parquet(DERIVED_DIR / "analysis_units.parquet")
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    epochs = pd.read_parquet(DATASET_DIR / "metadata/epochs.parquet")

    trials = trials[trials["context"].astype(str) == "passive"]
    trials = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])]
    trials = tag_passive_subepoch(trials, epochs)
    trials = trials[trials["passive_epoch"].isin(["passive_pre", "passive_post"])].copy()
    print(f"Passive whisker/auditory trials, tagged pre/post: {len(trials)}")
    print(trials.groupby(["trial_type", "passive_epoch"]).size())

    session_ids = sorted(units["session_id"].unique())
    trial_rows: list[pd.DataFrame] = []

    for i, session_id in enumerate(session_ids, 1):
        sess_units = units[units["session_id"] == session_id]
        sess_trials = trials[trials["session_id"] == session_id]
        if sess_trials.empty or sess_units.empty:
            continue

        shard = load_spike_shard(DATASET_DIR / "spikes" / session_id)
        cluster_ids = shard["cluster_ids"]
        spike_clusters_local = shard["spike_clusters"]
        spike_times = shard["spike_times_seconds"]
        cluster_id_to_local = {int(cid): idx for idx, cid in enumerate(cluster_ids)}

        order = np.argsort(spike_clusters_local, kind="stable")
        sc_sorted = spike_clusters_local[order]
        st_sorted = spike_times[order]
        boundaries = np.searchsorted(sc_sorted, np.arange(len(cluster_ids) + 1))

        for trial_type, is_whisker in (("whisker_trial", True), ("auditory_trial", False)):
            tt_trials = sess_trials[sess_trials["trial_type"] == trial_type]
            if tt_trials.empty:
                continue
            trial_starts = tt_trials["start_time"].to_numpy()
            passive_epoch = tt_trials["passive_epoch"].to_numpy()
            trial_id = tt_trials["trial_id"].to_numpy() if "trial_id" in tt_trials.columns else np.arange(len(tt_trials))

            for _, unit_row in sess_units.iterrows():
                local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
                if local_idx is None:
                    continue
                lo, hi = boundaries[local_idx], boundaries[local_idx + 1]
                unit_spikes = st_sorted[lo:hi]

                base_n, base_dur = per_unit_trial_window_counts(unit_spikes, trial_starts, BASELINE_WINDOW, is_whisker)
                evok_n, evok_dur = per_unit_trial_window_counts(unit_spikes, trial_starts, EVOKED_WINDOW, is_whisker)

                with np.errstate(divide="ignore", invalid="ignore"):
                    base_rate = np.where(base_dur > 0, base_n / base_dur, np.nan)
                    evok_rate = np.where(evok_dur > 0, evok_n / evok_dur, np.nan)

                trial_rows.append(pd.DataFrame({
                    "session_id": session_id,
                    "cluster_id": int(unit_row["cluster_id"]),
                    "mouse_id": unit_row["mouse_id"],
                    "area_group": unit_row["area_group"],
                    "reward_group": unit_row["reward_group"],
                    "trial_type": trial_type,
                    "passive_epoch": passive_epoch,
                    "trial_id": trial_id,
                    "baseline_rate_hz": base_rate,
                    "evoked_rate_hz": evok_rate,
                    "evoked_corrected_hz": evok_rate - base_rate,
                }))

        if i % 10 == 0 or i == len(session_ids):
            print(f"[{i}/{len(session_ids)}] {session_id} done", flush=True)

    trial_df = pd.concat(trial_rows, ignore_index=True)
    trial_df.to_parquet(DERIVED_DIR / "trial_window_rates.parquet", index=False)
    print(f"\nWrote {len(trial_df)} trial-level rows.")

    unit_cond = (
        trial_df.groupby(["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "trial_type", "passive_epoch"])
        .agg(
            n_trials=("baseline_rate_hz", "size"),
            baseline_rate_hz=("baseline_rate_hz", "mean"),
            evoked_rate_hz=("evoked_rate_hz", "mean"),
            evoked_corrected_hz=("evoked_corrected_hz", "mean"),
        )
        .reset_index()
    )
    unit_cond.to_parquet(DERIVED_DIR / "unit_condition_rates.parquet", index=False)
    print(f"Wrote {len(unit_cond)} unit-condition summary rows.")


if __name__ == "__main__":
    main()
