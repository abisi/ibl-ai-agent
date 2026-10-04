"""Replot raw-vs-corrected whisker-trial average neural activity using the
v2 visualization-only correction: spike-time-based (not binned), replacement
window [-10ms, +5ms), baseline excludes [-10ms, +5ms) entirely (ends at
-10ms), per trial per unit -- see peri_event_histogram_v2.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "scripts")
from peri_event_histogram_v2 import REPLACE_START_S, REPLACE_STOP_S, corrected_trial_spike_times  # noqa: E402
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
ANALYSIS_UNITS = Path("reports/ssl_analysis/derived/analysis_units.parquet")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")

TIME_START = -0.05
TIME_STOP = 0.10
BIN_SIZE = 0.001
RNG_SEED = 20260815


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    units = pd.read_parquet(ANALYSIS_UNITS)
    rng = np.random.default_rng(RNG_SEED)

    wh_trials = trials[(trials["trial_type"] == "whisker_trial") & (trials["context"].astype(str) == "passive")]

    n_bins = int(round((TIME_STOP - TIME_START) / BIN_SIZE))
    edges = TIME_START + np.arange(n_bins + 1) * BIN_SIZE
    bin_centers = (edges[:-1] + edges[1:]) / 2

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
        n_trials = len(event_times)

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

            # raw: fixed per-trial histogram (no correction)
            lo_idx = np.searchsorted(u_spikes, event_times + TIME_START, side="left")
            hi_idx = np.searchsorted(u_spikes, event_times + TIME_STOP, side="left")
            raw_chunks = [u_spikes[lo_idx[t]:hi_idx[t]] - event_times[t] for t in range(n_trials) if hi_idx[t] > lo_idx[t]]
            raw_counts = np.histogram(np.concatenate(raw_chunks), bins=edges)[0] if raw_chunks else np.zeros(n_bins)

            # corrected: v2 spike-time-based, per trial
            corrected_trials = corrected_trial_spike_times(u_spikes, event_times, TIME_START, TIME_STOP, rng)
            corr_chunks = [c for c in corrected_trials if c.size]
            corr_counts = np.histogram(np.concatenate(corr_chunks), bins=edges)[0] if corr_chunks else np.zeros(n_bins)

            sum_raw += raw_counts / n_trials
            sum_corr += corr_counts / n_trials
            n_units_used += 1

        if i % 20 == 0 or i == len(sessions):
            print(f"[{i}/{len(sessions)}] {session_id}: {n_units_used} units so far", flush=True)

    rate_raw = (sum_raw / n_units_used) / BIN_SIZE
    rate_corr = (sum_corr / n_units_used) / BIN_SIZE
    print(f"\nTotal units used: {n_units_used}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xlim, title in ((axes[0], (-50, 100), "Wide view"), (axes[1], (-15, 15), "Zoomed")):
        ax.axvspan(REPLACE_START_S * 1000, REPLACE_STOP_S * 1000, color="tab:red", alpha=0.15, label="replacement window (-10/+5ms)")
        ax.plot(bin_centers * 1000, rate_raw, color="tab:red", linewidth=1.2, marker="o" if title == "Zoomed" else None, markersize=3, label="no artifact correction")
        ax.plot(bin_centers * 1000, rate_corr, color="tab:blue", linewidth=1.2, marker="o" if title == "Zoomed" else None, markersize=3, label="with artifact correction\n(v2: per-trial spike-time Poisson,\nbaseline excludes -10/+5ms)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_xlim(*xlim)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("population average firing rate (Hz)")
        ax.set_title(title)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Whisker-trial average neural activity: raw vs. v2 artifact-corrected\n(n={n_units_used} units, native 1ms bins)")
    fig.tight_layout()
    out_path = FIG_DIR / "artifact_correction_comparison_v2.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
