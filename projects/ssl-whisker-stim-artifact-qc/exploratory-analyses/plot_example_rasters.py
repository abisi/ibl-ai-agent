"""Example multi-unit raster diagnostic: real spike times (no correction,
no Poisson fill) across many units and trials in one example session,
zoomed to the stimulus period, to check the dropout-then-burst signature is
visible at the single-trial level and not purely an averaging artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
ANALYSIS_UNITS = Path("reports/ssl_analysis/derived/analysis_units.parquet")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")

EXAMPLE_SESSION = "AB116_20240724_102941"
N_EXAMPLE_UNITS = 60
WINDOW = (-0.015, 0.015)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    units = pd.read_parquet(ANALYSIS_UNITS)

    sess_trials = trials[(trials["session_id"] == EXAMPLE_SESSION) & (trials["trial_type"] == "whisker_trial") & (trials["context"].astype(str) == "passive")]
    trial_starts = sess_trials["start_time"].to_numpy()
    print(f"Example session {EXAMPLE_SESSION}: {len(trial_starts)} whisker trials (all used)")

    sess_units = units[units["session_id"] == EXAMPLE_SESSION].head(N_EXAMPLE_UNITS)
    shard = load_spike_shard(DATASET_DIR / "spikes" / EXAMPLE_SESSION)
    cluster_ids = shard["cluster_ids"]
    spike_clusters_local = shard["spike_clusters"]
    spike_times = shard["spike_times_seconds"]
    cluster_id_to_local = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
    order = np.argsort(spike_clusters_local, kind="stable")
    sc_sorted = spike_clusters_local[order]
    st_sorted = spike_times[order]
    boundaries = np.searchsorted(sc_sorted, np.arange(len(cluster_ids) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # panel 1: all example units (rows), spikes from ALL trials overlaid per
    # row -- a single trial or single unit is too sparse to show the
    # population-level pattern found by pooling ~100k units x ~75 trials
    # each; overlaying all of one session's trials per unit row is a middle
    # ground that still shows real, un-pooled single-session data.
    ax = axes[0]
    for row, (_, unit_row) in enumerate(sess_units.iterrows()):
        local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
        if local_idx is None:
            continue
        u_spikes = st_sorted[boundaries[local_idx]:boundaries[local_idx + 1]]
        all_rel = []
        for t0 in trial_starts:
            seg = u_spikes[(u_spikes >= t0 + WINDOW[0]) & (u_spikes < t0 + WINDOW[1])] - t0
            if seg.size:
                all_rel.append(seg)
        if all_rel:
            rel = np.concatenate(all_rel)
            ax.scatter(rel * 1000, np.full(rel.shape, row), s=2, color="black", alpha=0.4)
    ax.axvspan(-10, 5, color="grey", alpha=0.15, label="ssl_artifact_dead_zone.md (-10/+5ms)")
    ax.axvspan(-1, 4, color="tab:red", alpha=0.15, label="actual correction window (-1/+4ms)")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("time from start_time (ms)")
    ax.set_ylabel(f"unit # (n={len(sess_units)})")
    ax.set_title(f"All example units, all {len(trial_starts)} whisker trials overlaid\n(session {EXAMPLE_SESSION})")

    # panel 2: single unit (highest-rate), many trials (rows = trials)
    ax = axes[1]
    rates = []
    for _, unit_row in sess_units.iterrows():
        local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
        if local_idx is None:
            continue
        rates.append((unit_row["cluster_id"], boundaries[local_idx + 1] - boundaries[local_idx]))
    best_cluster_id = max(rates, key=lambda t: t[1])[0]
    local_idx = cluster_id_to_local[int(best_cluster_id)]
    u_spikes = st_sorted[boundaries[local_idx]:boundaries[local_idx + 1]]
    for row, t0 in enumerate(trial_starts):
        rel = u_spikes[(u_spikes >= t0 + WINDOW[0]) & (u_spikes < t0 + WINDOW[1])] - t0
        ax.scatter(rel * 1000, np.full(rel.shape, row), s=4, color="black")
    ax.axvspan(-10, 5, color="grey", alpha=0.2)
    ax.axvline(0, color="tab:red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time from start_time (ms)")
    ax.set_ylabel(f"trial # (n={len(trial_starts)})")
    ax.set_title(f"Single unit (cluster {best_cluster_id}, highest spike count),\nall example trials")

    fig.suptitle("Example raster diagnostic (raw, uncorrected; grey band = documented dead zone)")
    fig.tight_layout()
    out_path = FIG_DIR / "example_rasters.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
