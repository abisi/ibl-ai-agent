"""Replot the raw-vs-corrected whisker-trial average neural activity figure
using the actual artifact-correction algorithm from
M:\\analysis\\Axel_Bisi\\unit_spikes_analysis\\neural_utils.py
(see peri_event_histogram.py), not the -10ms/+5ms population-Poisson-
interpolation procedure from ssl_artifact_dead_zone.md.

Per unit: build a (n_trials, n_bins) PETH at 1ms resolution for that unit's
whisker trials, average across trials -> one curve per unit; average across
units -> population curve. Done twice: artifact_correction=False (raw) and
artifact_correction=True (real per-trial Poisson correction, window
[-1ms, +4ms)), both output at 1ms bins for a direct comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "scripts")
from peri_event_histogram import ART_START_MS, ART_STOP_MS, compute_unit_peri_event_histogram  # noqa: E402
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
ANALYSIS_UNITS = Path("reports/ssl_analysis/derived/analysis_units.parquet")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")

TIME_START = -0.05
TIME_STOP = 0.10
BIN_SIZE = 0.001  # both curves at native 1ms so they're directly comparable
OLD_DEAD_ZONE = (-0.010, 0.005)
RNG_SEED = 20260815


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    units = pd.read_parquet(ANALYSIS_UNITS)
    rng = np.random.default_rng(RNG_SEED)

    wh_trials = trials[(trials["trial_type"] == "whisker_trial") & (trials["context"].astype(str) == "passive")]

    n_bins = int(round((TIME_STOP - TIME_START) / BIN_SIZE))
    bin_centers = TIME_START + (np.arange(n_bins) + 0.5) * BIN_SIZE

    sum_raw = np.zeros(n_bins)
    sum_corr = np.zeros(n_bins)
    n_units_used = 0

    sessions = sorted(units["session_id"].unique())
    for i, session_id in enumerate(sessions, 1):
        sess_units = units[units["session_id"] == session_id]
        sess_trials = wh_trials[wh_trials["session_id"] == session_id]
        if sess_trials.empty or sess_units.empty:
            continue
        event_times = sess_trials["start_time"].to_numpy()

        shard = load_spike_shard(DATASET_DIR / "spikes" / session_id)
        cluster_ids = shard["cluster_ids"]
        spike_clusters_local = shard["spike_clusters"]
        spike_times = shard["spike_times_seconds"]
        cluster_id_to_local = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
        order = np.argsort(spike_clusters_local, kind="stable")
        sc_sorted = spike_clusters_local[order]
        st_sorted = spike_times[order]
        boundaries = np.searchsorted(sc_sorted, np.arange(len(cluster_ids) + 1))

        for _, unit_row in sess_units.iterrows():
            local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
            if local_idx is None:
                continue
            u_spikes = st_sorted[boundaries[local_idx]:boundaries[local_idx + 1]]

            peth_raw = compute_unit_peri_event_histogram(u_spikes, event_times, BIN_SIZE, TIME_START, TIME_STOP, artifact_correction=False)
            peth_corr = compute_unit_peri_event_histogram(u_spikes, event_times, BIN_SIZE, TIME_START, TIME_STOP, artifact_correction=True, rng=rng)

            sum_raw += peth_raw.mean(axis=0)
            sum_corr += peth_corr.mean(axis=0)
            n_units_used += 1

        if i % 20 == 0 or i == len(sessions):
            print(f"[{i}/{len(sessions)}] {session_id}: {n_units_used} units so far", flush=True)

    mean_counts_raw = sum_raw / n_units_used
    mean_counts_corr = sum_corr / n_units_used
    rate_raw = mean_counts_raw / BIN_SIZE
    rate_corr = mean_counts_corr / BIN_SIZE

    print(f"\nTotal units used: {n_units_used}")

    art_start_s, art_stop_s = ART_START_MS / 1000, ART_STOP_MS / 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xlim, title in ((axes[0], (-50, 100), "Wide view"), (axes[1], (-15, 15), "Zoomed")):
        ax.axvspan(OLD_DEAD_ZONE[0] * 1000, OLD_DEAD_ZONE[1] * 1000, color="grey", alpha=0.15, label="ssl_artifact_dead_zone.md window (-10/+5ms)\n-- shown for comparison, not used here")
        ax.axvspan(art_start_s * 1000, art_stop_s * 1000, color="tab:red", alpha=0.15, label="actual correction window (-1/+4ms)")
        ax.plot(bin_centers * 1000, rate_raw, color="tab:red", linewidth=1.2, marker="o" if title == "Zoomed" else None, markersize=3, label="no artifact correction")
        ax.plot(bin_centers * 1000, rate_corr, color="tab:blue", linewidth=1.2, marker="o" if title == "Zoomed" else None, markersize=3, label="with artifact correction\n(real algorithm: per-trial, per-unit Poisson)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_xlim(*xlim)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("population average firing rate (Hz)")
        ax.set_title(title)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Whisker-trial average neural activity: raw vs. correctly artifact-corrected\n(n={n_units_used} units, mean-of-per-unit-mean-PETH, native 1ms bins)")
    fig.tight_layout()
    out_path = FIG_DIR / "artifact_correction_comparison_correct.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    np.savez(
        "projects/ssl-whisker-stim-artifact-qc/artifacts/correct_artifact_comparison.npz",
        bin_centers=bin_centers, rate_raw=rate_raw, rate_corr=rate_corr, n_units=n_units_used,
    )


if __name__ == "__main__":
    main()
