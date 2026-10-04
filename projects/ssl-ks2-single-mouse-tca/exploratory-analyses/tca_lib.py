"""Shared building blocks for single-mouse TCA on the ssl_ks2_ephys dataset.

Ported and adapted from Axel Bisi's megamouse pipeline
(M:\\analysis\\Axel_Bisi\\brain_wide_analysis\\tca\\tca_pipeline_bis.py), but
building one (trials, neurons, time_bins) tensor per session instead of
pooling neurons across mice, and reading from the compressed ssl_ks2_ephys
parquet/shard format instead of an in-memory unit_table/trial_table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.ssl_ks2_ephys import load_spike_shard

DATASET_DIR = resolve_dataset_dir("ssl_ks2_ephys")


def load_session_tables():
    """Load the three metadata tables this analysis needs.

    Returns
    -------
    sessions : DataFrame, one row per session
    units : DataFrame, one row per unit (all sessions)
    trials : DataFrame, one row per trial (all sessions)
    """
    meta = DATASET_DIR / "metadata"
    sessions = pd.read_parquet(meta / "sessions.parquet")
    units = pd.read_parquet(meta / "units.parquet")
    trials = pd.read_parquet(meta / "trials.parquet")
    return sessions, units, trials


def load_session_spike_times(session_id, units_session):
    """Attach a per-unit spike-time array to a session's unit rows.

    Parameters
    ----------
    session_id : str
    units_session : DataFrame, this session's rows from `units` (must have a
        `cluster_id` column, in any order)

    Returns
    -------
    spike_times_by_cluster : dict {cluster_id (int): spike times in seconds
        (1D array)}, one entry per unit in `units_session`
    """
    shard = load_spike_shard(DATASET_DIR / "spikes" / session_id)
    spike_times = np.asarray(shard["spike_times_seconds"], dtype=float)
    spike_clusters_dense = np.asarray(shard["spike_clusters"], dtype=int)
    cluster_ids = np.asarray(shard["cluster_ids"], dtype=int)
    # spike_clusters is stored as dense local indices; map through cluster_ids
    # to get the real cluster id of each spike (same pattern as bwm_ephys).
    spike_cluster_ids = cluster_ids[spike_clusters_dense]

    spike_times_by_cluster = {}
    for cluster_id in units_session["cluster_id"].to_numpy(dtype=int):
        spike_times_by_cluster[cluster_id] = spike_times[spike_cluster_ids == cluster_id]
    return spike_times_by_cluster


def align_whisker_trials_first_hit(trials_session, trial_window_pre=30, trial_window_post=50):
    """Select whisker trials around the first active-context hit, like the
    megamouse pipeline's 'first_hit' alignment mode, for one session.

    A "hit" is the first active-context whisker trial with lick_flag == 1.
    Passive-context whisker trials before it are kept (to see the
    passive-to-active transition); active trials are kept up to
    `trial_window_post` after the hit.

    Parameters
    ----------
    trials_session : DataFrame, this session's rows from `trials`
    trial_window_pre : int, trials to keep before the hit
    trial_window_post : int, trials to keep after the hit (inclusive of the
        hit trial itself)

    Returns
    -------
    aligned : DataFrame, trial_window_pre + trial_window_post + 1 rows (fewer
        if the session doesn't have enough trials on one side), with an added
        `whisker_trial_id` column (0 = the hit trial, negative = before,
        positive = after)
    """
    whisker = trials_session[trials_session["trial_type"] == "whisker_trial"].sort_values("start_time").reset_index(drop=True)
    active = whisker[whisker["context"] == "active"]
    hits = active[active["lick_flag"] == 1]
    # .index[0] on `hits` is a position in `whisker` (both share the same
    # reset index), so no separate position lookup is needed.
    hit_pos = int(hits.index[0]) if len(hits) else int(active.index[0])

    start = max(0, hit_pos - trial_window_pre)
    stop = min(len(whisker), hit_pos + trial_window_post + 1)
    aligned = whisker.iloc[start:stop].reset_index(drop=True)
    aligned["whisker_trial_id"] = np.arange(start, stop) - hit_pos
    return aligned


def build_spike_tensor(spike_times_by_cluster, units_session, trials_aligned,
                        time_window=(-0.1, 0.2), bin_size=0.01,
                        remove_artifact=True, artifact_window=(-0.002, 0.004),
                        baseline_window=(-0.1, 0.0), rng=None):
    """Bin spikes into a (trials, neurons, time_bins) tensor around
    `whisker_stim_time`, one row per trial in `trials_aligned` and one column
    per unit in `units_session` (same order).

    Spikes in `artifact_window` around stimulus onset are a known electrical
    artifact from whisker stimulation; they are replaced with Poisson spikes
    drawn from each neuron's own baseline-window rate (same approach as the
    megamouse pipeline), so the artifact doesn't masquerade as a real evoked
    response in the tensor.

    Parameters
    ----------
    spike_times_by_cluster : dict, from `load_session_spike_times`
    units_session : DataFrame, defines neuron (column) order via `cluster_id`
    trials_aligned : DataFrame, defines trial (row) order; must have
        `whisker_stim_time`
    time_window : (start, end) in seconds relative to stimulus onset
    bin_size : bin width in seconds
    rng : numpy.random.Generator, for reproducible artifact replacement

    Returns
    -------
    tensor : ndarray, shape (n_trials, n_neurons, n_time_bins), spike counts
    time_bins : ndarray, shape (n_time_bins + 1,), bin edges in seconds
    """
    if rng is None:
        rng = np.random.default_rng(0)

    time_bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    n_time_bins = len(time_bins) - 1
    # A handful of trials (seen in AB152_20250127_105124) have a NaN
    # whisker_stim_time but a valid start_time; the two are identical
    # whenever both are present (checked against AB116), so start_time is a
    # safe fallback rather than dropping the trial.
    align_times = trials_aligned["whisker_stim_time"].fillna(trials_aligned["start_time"]).to_numpy(dtype=float)
    n_trials = len(align_times)
    cluster_ids = units_session["cluster_id"].to_numpy(dtype=int)
    n_neurons = len(cluster_ids)

    tensor = np.zeros((n_trials, n_neurons, n_time_bins))
    for neuron_pos, cluster_id in enumerate(cluster_ids):
        spike_times = spike_times_by_cluster[cluster_id]

        if remove_artifact:
            spike_times = _replace_artifact_spikes(
                spike_times, align_times, artifact_window, baseline_window, rng
            )

        for trial_pos, t0 in enumerate(align_times):
            counts, _ = np.histogram(spike_times - t0, bins=time_bins)
            tensor[trial_pos, neuron_pos, :] = counts
    return tensor, time_bins


def _replace_artifact_spikes(spike_times, align_times, artifact_window, baseline_window, rng):
    """Remove spikes in `artifact_window` around each trial's alignment time
    and replace them with Poisson spikes at this neuron's own baseline rate
    (estimated once, pooling the baseline window across all trials).
    """
    baseline_duration = baseline_window[1] - baseline_window[0]
    baseline_counts = np.array([
        np.sum((spike_times >= t0 + baseline_window[0]) & (spike_times < t0 + baseline_window[1]))
        for t0 in align_times
    ])
    baseline_rate = baseline_counts.mean() / baseline_duration  # spikes/sec

    artifact_duration = artifact_window[1] - artifact_window[0]
    cleaned = spike_times.copy()
    for t0 in align_times:
        lo, hi = t0 + artifact_window[0], t0 + artifact_window[1]
        cleaned = cleaned[(cleaned < lo) | (cleaned >= hi)]
        n_new = rng.poisson(baseline_rate * artifact_duration)
        if n_new > 0:
            cleaned = np.concatenate([cleaned, rng.uniform(lo, hi, n_new)])
    return np.sort(cleaned)
