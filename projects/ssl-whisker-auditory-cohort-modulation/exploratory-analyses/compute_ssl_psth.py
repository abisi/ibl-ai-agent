"""Per-unit PSTHs (10ms bins, 2ms stride) for whisker_trial / auditory_trial,
passive_pre / passive_post, dead-zone-aware for whisker trials.

Strategy for efficiency at ~103k units: for each unit x condition, build a
fine (1ms) resolution pooled histogram of spike counts across trials, plus a
parallel "valid trial-seconds per 1ms bin" array (reduced to 0 inside the
whisker dead zone, [-10ms, +5ms) around start_time). The 10ms/2ms PSTH curve
is then a moving sum over 10 consecutive 1ms bins, stride 2ms, of
(spike_counts / valid_trial_seconds). This keeps dead-zone bins correctly
excluded from any *real* rate estimate (duration goes to 0 -> NaN), while
still letting aggregation (mouse/area/cohort) be a simple weighted sum of the
per-unit fine histograms before the final moving-sum step.

Output: one .npz per (trial_type, passive_epoch) with, per retained unit,
the fine spike-count histogram and valid-seconds-per-bin array, plus a
units_index.parquet mapping row -> (session_id, cluster_id, mouse_id,
area_group, reward_group). Aggregation/plotting is a separate script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssl_wm_lib import DEAD_ZONE_POST_S, DEAD_ZONE_PRE_S, tag_passive_subepoch  # noqa: E402
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
DERIVED_DIR = Path("reports/ssl_analysis/derived")
PSTH_DIR = DERIVED_DIR / "psth"

WINDOW_START_S = -0.150
WINDOW_END_S = 0.200
FINE_BIN_S = 0.001
BIN_EDGES = np.arange(WINDOW_START_S, WINDOW_END_S + FINE_BIN_S / 2, FINE_BIN_S)
N_FINE_BINS = len(BIN_EDGES) - 1
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def fine_bin_valid_seconds(is_whisker: bool) -> np.ndarray:
    """Per-fine-bin usable seconds of a single trial (FINE_BIN_S normally,
    0 for any 1ms bin fully inside the whisker dead zone; partial overlap
    reduced proportionally)."""
    valid = np.full(N_FINE_BINS, FINE_BIN_S, dtype=np.float64)
    if not is_whisker:
        return valid
    dz_start, dz_end = -DEAD_ZONE_PRE_S, DEAD_ZONE_POST_S
    overlap_start = np.maximum(BIN_EDGES[:-1], dz_start)
    overlap_end = np.minimum(BIN_EDGES[1:], dz_end)
    overlap = np.clip(overlap_end - overlap_start, 0, None)
    valid = valid - overlap
    return np.clip(valid, 0, None)


def main() -> None:
    PSTH_DIR.mkdir(parents=True, exist_ok=True)

    units = pd.read_parquet(DERIVED_DIR / "analysis_units.parquet")
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    epochs = pd.read_parquet(DATASET_DIR / "metadata/epochs.parquet")

    trials = trials[trials["context"].astype(str) == "passive"]
    trials = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])]
    trials = tag_passive_subepoch(trials, epochs)
    trials = trials[trials["passive_epoch"].isin(["passive_pre", "passive_post"])].copy()

    conditions = [
        ("whisker_trial", "passive_pre", True),
        ("whisker_trial", "passive_post", True),
        ("auditory_trial", "passive_pre", False),
        ("auditory_trial", "passive_post", False),
    ]
    valid_seconds_by_iswhisker = {True: fine_bin_valid_seconds(True), False: fine_bin_valid_seconds(False)}

    session_ids = sorted(units["session_id"].unique())

    for trial_type, passive_epoch, is_whisker in conditions:
        cond_trials = trials[(trials["trial_type"] == trial_type) & (trials["passive_epoch"] == passive_epoch)]
        valid_seconds_per_bin = valid_seconds_by_iswhisker[is_whisker]

        index_rows = []
        counts_rows = []

        for i, session_id in enumerate(session_ids, 1):
            sess_units = units[units["session_id"] == session_id]
            sess_trials = cond_trials[cond_trials["session_id"] == session_id]
            if sess_trials.empty or sess_units.empty:
                continue
            trial_starts = sess_trials["start_time"].to_numpy()
            n_trials = len(trial_starts)

            shard = load_spike_shard(DATASET_DIR / "spikes" / session_id)
            cluster_ids = shard["cluster_ids"]
            spike_clusters_local = shard["spike_clusters"]
            spike_times = shard["spike_times_seconds"]
            cluster_id_to_local = {int(cid): idx for idx, cid in enumerate(cluster_ids)}

            order = np.argsort(spike_clusters_local, kind="stable")
            sc_sorted = spike_clusters_local[order]
            st_sorted = spike_times[order]
            boundaries = np.searchsorted(sc_sorted, np.arange(len(cluster_ids) + 1))

            window_lo = trial_starts + WINDOW_START_S
            window_hi = trial_starts + WINDOW_END_S

            for _, unit_row in sess_units.iterrows():
                local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
                if local_idx is None:
                    continue
                lo, hi = boundaries[local_idx], boundaries[local_idx + 1]
                unit_spikes = st_sorted[lo:hi]

                lo_idx = np.searchsorted(unit_spikes, window_lo, side="left")
                hi_idx = np.searchsorted(unit_spikes, window_hi, side="left")
                rel_chunks = [
                    unit_spikes[lo_idx[t_idx]:hi_idx[t_idx]] - trial_starts[t_idx]
                    for t_idx in range(n_trials)
                    if hi_idx[t_idx] > lo_idx[t_idx]
                ]
                if rel_chunks:
                    all_rel = np.concatenate(rel_chunks)
                    fine_counts, _ = np.histogram(all_rel, bins=BIN_EDGES)
                else:
                    fine_counts = np.zeros(N_FINE_BINS, dtype=np.int64)

                index_rows.append({
                    "session_id": session_id, "cluster_id": int(unit_row["cluster_id"]),
                    "mouse_id": unit_row["mouse_id"], "area_group": unit_row["area_group"],
                    "reward_group": unit_row["reward_group"], "n_trials": n_trials,
                })
                counts_rows.append(fine_counts)

            if i % 20 == 0 or i == len(session_ids):
                print(f"[{trial_type}/{passive_epoch}] [{i}/{len(session_ids)}] {session_id}", flush=True)

        index_df = pd.DataFrame(index_rows)
        counts_arr = np.stack(counts_rows) if counts_rows else np.zeros((0, N_FINE_BINS), dtype=np.int32)
        index_df["n_trials_valid_seconds_per_bin"] = [valid_seconds_per_bin.tolist()] * len(index_df)

        out_path = PSTH_DIR / f"{trial_type}__{passive_epoch}.npz"
        np.savez_compressed(
            out_path,
            fine_counts=counts_arr,
            n_trials=index_df["n_trials"].to_numpy() if len(index_df) else np.array([]),
            valid_seconds_per_bin=valid_seconds_per_bin,
            bin_centers=BIN_CENTERS,
        )
        index_df.drop(columns=["n_trials_valid_seconds_per_bin"]).to_parquet(
            PSTH_DIR / f"{trial_type}__{passive_epoch}_index.parquet", index=False
        )
        print(f"Wrote {out_path} ({counts_arr.shape})")


if __name__ == "__main__":
    main()
