"""Shared helpers for the SSL whisker/auditory passive-epoch analysis:
trial-to-passive-subepoch tagging, and dead-zone-aware window firing rates.

Reference point for all windows is trial `start_time`, per
skills/ssl-analyze/references/ssl_artifact_dead_zone.md. Whisker-trial dead
zone: [-10ms, +5ms) around start_time -- spikes there are never used in any
quantitative computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEAD_ZONE_PRE_S = 0.010
DEAD_ZONE_POST_S = 0.005

BASELINE_WINDOW = (-1.0, 0.0)   # seconds relative to start_time
EVOKED_WINDOW = (0.005, 0.045)  # seconds relative to start_time


def tag_passive_subepoch(trials: pd.DataFrame, epochs: pd.DataFrame) -> pd.DataFrame:
    """Add a `passive_epoch` column (pre/post/NA) to a trials table, using
    epochs.parquet boundaries -- trials.context does not itself distinguish
    passive_pre from passive_post."""
    trials = trials.copy()
    trials["passive_epoch"] = pd.Series([None] * len(trials), dtype=object)

    for session_id, sess_epochs in epochs.groupby("session_id"):
        mask_session = trials["session_id"] == session_id
        for _, row in sess_epochs.iterrows():
            name = row["epoch_name"]
            if name not in ("passive_pre", "passive_post"):
                continue
            in_epoch = mask_session & (trials["start_time"] >= row["start_time"]) & (trials["start_time"] < row["stop_time"])
            trials.loc[in_epoch, "passive_epoch"] = name
    return trials


def clipped_window(window: tuple[float, float], is_whisker: bool) -> tuple[float, float] | None:
    """Clip a (start, end) window (seconds, relative to start_time) against
    the whisker artifact dead zone. Returns None if fully inside the dead
    zone (no real spikes usable for that trial in that window)."""
    start, end = window
    if not is_whisker:
        return (start, end)
    dz_start, dz_end = -DEAD_ZONE_PRE_S, DEAD_ZONE_POST_S
    if start >= dz_start and end <= dz_end:
        return None
    new_start = max(start, dz_end) if start < dz_end <= end else start
    new_end = min(end, dz_start) if start <= dz_start < end else end
    # General clip: remove the portion of [start,end) that overlaps [dz_start,dz_end)
    if start < dz_start and end > dz_end:
        # window straddles the whole dead zone on both sides -- ambiguous for a
        # single contiguous window; callers should not pass such windows for
        # whisker trials (our BASELINE/EVOKED windows never do).
        raise ValueError(f"window {window} straddles the entire dead zone; split it explicitly")
    if end > dz_start and start < dz_start:
        end = dz_start
    if start < dz_end and end > dz_end:
        start = dz_end
    if start >= end:
        return None
    return (start, end)


def per_unit_trial_window_counts(
    spike_times: np.ndarray,
    trial_starts: np.ndarray,
    window: tuple[float, float],
    is_whisker: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """For one unit's sorted spike_times (seconds) and an array of trial
    start_times, return (n_spikes, duration_s) per trial for `window`
    (seconds relative to start_time), dead-zone-clipped for whisker trials.
    duration_s is the *usable* (post-clip) window duration; n_spikes counts
    only real spikes in that usable window."""
    clipped = clipped_window(window, is_whisker)
    n_trials = trial_starts.shape[0]
    if clipped is None:
        return np.zeros(n_trials, dtype=np.int64), np.zeros(n_trials, dtype=np.float64)
    start, end = clipped
    lo = np.searchsorted(spike_times, trial_starts + start, side="left")
    hi = np.searchsorted(spike_times, trial_starts + end, side="left")
    counts = (hi - lo).astype(np.int64)
    durations = np.full(n_trials, end - start, dtype=np.float64)
    return counts, durations
