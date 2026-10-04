"""v2 raw-vs-corrected whisker-trial artifact comparison (spike-time-based,
[-10ms,+5ms) window, baseline excludes that window -- see
peri_event_histogram_v2.py), broken down per area_group instead of pooled
globally.
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
MIN_UNITS_PER_AREA = 200


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    units = pd.read_parquet(ANALYSIS_UNITS)
    rng = np.random.default_rng(RNG_SEED)

    wh_trials = trials[(trials["trial_type"] == "whisker_trial") & (trials["context"].astype(str) == "passive")]

    area_counts = units[units["area_group"].notna()].groupby("area_group").size().sort_values(ascending=False)
    areas = [a for a in area_counts.index if area_counts[a] >= MIN_UNITS_PER_AREA]
    print(f"Areas included (>= {MIN_UNITS_PER_AREA} units): {areas}")

    n_bins = int(round((TIME_STOP - TIME_START) / BIN_SIZE))
    edges = TIME_START + np.arange(n_bins + 1) * BIN_SIZE
    bin_centers = (edges[:-1] + edges[1:]) / 2

    sum_raw = {a: np.zeros(n_bins) for a in areas}
    sum_corr = {a: np.zeros(n_bins) for a in areas}
    n_units = {a: 0 for a in areas}

    sessions = sorted(units["session_id"].unique())
    for i, session_id in enumerate(sessions, 1):
        sess_units = units[(units["session_id"] == session_id) & (units["area_group"].isin(areas))]
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
            area = unit_row["area_group"]
            u_spikes = st_sorted[boundaries[local_idx]:boundaries[local_idx + 1]]

            lo_idx = np.searchsorted(u_spikes, event_times + TIME_START, side="left")
            hi_idx = np.searchsorted(u_spikes, event_times + TIME_STOP, side="left")
            raw_chunks = [u_spikes[lo_idx[t]:hi_idx[t]] - event_times[t] for t in range(n_trials) if hi_idx[t] > lo_idx[t]]
            raw_counts = np.histogram(np.concatenate(raw_chunks), bins=edges)[0] if raw_chunks else np.zeros(n_bins)

            corrected_trials = corrected_trial_spike_times(u_spikes, event_times, TIME_START, TIME_STOP, rng)
            corr_chunks = [c for c in corrected_trials if c.size]
            corr_counts = np.histogram(np.concatenate(corr_chunks), bins=edges)[0] if corr_chunks else np.zeros(n_bins)

            sum_raw[area] += raw_counts / n_trials
            sum_corr[area] += corr_counts / n_trials
            n_units[area] += 1

        if i % 20 == 0 or i == len(sessions):
            print(f"[{i}/{len(sessions)}] {session_id}: {n_units} units so far", flush=True)

    fig, axes = plt.subplots(len(areas), 2, figsize=(12, 3 * len(areas)), sharex="col")
    for row, area in enumerate(areas):
        rate_raw = (sum_raw[area] / n_units[area]) / BIN_SIZE
        rate_corr = (sum_corr[area] / n_units[area]) / BIN_SIZE
        for col, (xlim, title) in enumerate((((-50, 100), "Wide view"), ((-15, 15), "Zoomed"))):
            ax = axes[row, col]
            ax.axvspan(REPLACE_START_S * 1000, REPLACE_STOP_S * 1000, color="tab:red", alpha=0.15)
            ax.plot(bin_centers * 1000, rate_raw, color="tab:red", linewidth=1.1, label="no correction")
            ax.plot(bin_centers * 1000, rate_corr, color="tab:blue", linewidth=1.1, label="v2 corrected")
            ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
            ax.set_xlim(*xlim)
            if col == 0:
                ax.set_ylabel(f"{area}\n(n={n_units[area]})\nHz")
            if row == 0:
                ax.set_title("Wide view" if col == 0 else "Zoomed")
            if row == len(areas) - 1:
                ax.set_xlabel("time from start_time (ms)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Whisker-trial average neural activity by area_group: raw vs. v2 artifact-corrected")
    fig.tight_layout()
    out_path = FIG_DIR / "artifact_correction_comparison_v2_by_area.png"
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
