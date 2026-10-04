"""PSTH figures for the most-modulated example units from
001_compute_example_units.py, split by t-1 rewarded vs t-1 non-rewarded, for
the RHMI (t=correct trials) and EHMI (t=incorrect trials) conditions
separately. See ../question.md for the metric definition.

Selection: per session, the unit with the largest |index| among units with
a defined index and denom_mean_fr > 1 Hz (excludes near-zero-firing-rate
units where a tiny denominator inflates the index without a real effect).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from ssl_bwm_trial_prep import load_reward_group
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from history_lib import DATASET_ROOT, EVOKED_WINDOW_S, DEAD_ZONE_S, QC_VALUES, prep_trials

PSTH_WINDOW_S = (-0.05, 0.15)
BIN_S = 0.005
DENOM_FLOOR_HZ = 1.0
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")


def trial_psth(spike_times_sorted, event_times, window, bin_s):
    """Per-trial-averaged PSTH (Hz) +/- SEM across `event_times`.
    Returns bin_centers, mean_rate, sem_rate."""
    edges = np.arange(window[0], window[1] + bin_s / 2, bin_s)
    n_trials = len(event_times)
    counts = np.zeros((n_trials, len(edges) - 1))
    for i, t0 in enumerate(event_times):
        rel = spike_times_sorted[(spike_times_sorted >= t0 + window[0]) & (spike_times_sorted < t0 + window[1])] - t0
        counts[i], _ = np.histogram(rel, bins=edges)
    rates = counts / bin_s
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rates.mean(axis=0), rates.std(axis=0, ddof=1) / np.sqrt(n_trials)


def plot_unit(session_id, reward_group, cluster_id, unit_spikes, start_time, correct, t1_rewarded, rhmi_index, ehmi_index, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    specs = [
        ("RHMI (t = correct trials)", correct, rhmi_index, axes[0]),
        ("EHMI (t = incorrect trials)", ~correct, ehmi_index, axes[1]),
    ]
    for title, t_mask, index_val, ax in specs:
        for label, bucket_mask, color in [
            ("t-1 rewarded", t_mask & t1_rewarded, "tab:blue"),
            ("t-1 non-rewarded", t_mask & ~t1_rewarded, "tab:orange"),
        ]:
            onsets = start_time[bucket_mask]
            centers, mean_rate, sem_rate = trial_psth(unit_spikes, onsets, PSTH_WINDOW_S, BIN_S)
            ax.plot(centers * 1000, mean_rate, color=color, label=f"{label} (n={bucket_mask.sum()})")
            ax.fill_between(centers * 1000, mean_rate - sem_rate, mean_rate + sem_rate, color=color, alpha=0.2)
        ax.axvspan(DEAD_ZONE_S[0] * 1000, DEAD_ZONE_S[1] * 1000, color="gray", alpha=0.25,
                   label="whisker dead zone" if title.startswith("RHMI") else None)
        ax.axvspan(EVOKED_WINDOW_S[0] * 1000, EVOKED_WINDOW_S[1] * 1000, color="tab:green", alpha=0.1,
                   label="evoked window" if title.startswith("RHMI") else None)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Time from start_time (ms)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f"{title}\nindex={index_val:.3f}")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{session_id} ({reward_group}), cluster {cluster_id}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    df = pd.read_parquet(OUT_DIR / "example_unit_indices.parquet")
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")

    for session_id, sub in df.groupby("session_id"):
        reward_group = sub["reward_group"].iloc[0]
        subject_id = sessions_tbl.loc[sessions_tbl.session_id == session_id, "subject_id"].iloc[0]
        assert load_reward_group(subject_id) == reward_group

        trials = prep_trials(session_id, sessions_tbl, trials_tbl, reward_group)
        start_time = trials["start_time"].to_numpy()
        correct = trials["correct"].to_numpy()
        t1_rewarded = trials["t1_reward_delivered"].to_numpy().astype(bool)

        shard = load_spike_shard(DATASET_ROOT / "spikes" / session_id)
        spike_times_all = shard["spike_times_seconds"]
        spike_clusters_dense = shard["spike_clusters"]
        cluster_ids = shard["cluster_ids"]

        print(f"\n=== {session_id} ({reward_group}) ===")
        for metric in ("rhmi", "ehmi"):
            candidates = sub[(sub[f"{metric}_index"].notna()) & (sub[f"{metric}_denom_mean_fr"] > DENOM_FLOOR_HZ)]
            if len(candidates) == 0:
                print(f"  {metric.upper()}: no unit above the {DENOM_FLOOR_HZ} Hz denom floor, skipping")
                continue
            top = candidates.loc[candidates[f"{metric}_index"].abs().idxmax()]
            cid = int(top["cluster_id"])
            print(f"  {metric.upper()} top unit: cluster {cid}, index={top[f'{metric}_index']:.3f}, "
                  f"denom={top[f'{metric}_denom_mean_fr']:.2f} Hz")

            dense_idx = np.where(cluster_ids == cid)[0][0]
            unit_spikes = np.sort(spike_times_all[spike_clusters_dense == dense_idx])

            rhmi_val = sub.loc[sub.cluster_id == cid, "rhmi_index"].iloc[0]
            ehmi_val = sub.loc[sub.cluster_id == cid, "ehmi_index"].iloc[0]
            out_path = OUT_DIR / f"002_psth_{session_id}_cluster{cid}_top{metric}.png"
            plot_unit(session_id, reward_group, cid, unit_spikes, start_time, correct, t1_rewarded, rhmi_val, ehmi_val, out_path)


if __name__ == "__main__":
    main()
