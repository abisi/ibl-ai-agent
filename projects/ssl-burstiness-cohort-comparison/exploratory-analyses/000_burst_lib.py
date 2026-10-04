"""Burst detection and burstiness/burst-index metrics for the SSL burstiness
x cohort comparison. See ../question.md for the full method spec.

Burst definition (user-specified): a maximal run of >=2 consecutive spikes
where every intra-run ISI <= 10ms, followed either by the end of the
recording or by a "tail" ISI > 15ms. A run whose following ISI falls in
(10ms, 15ms] is ambiguous under this strict definition and does not qualify
as a burst -- its spikes are left labeled as non-burst. This edge case is
rare (most ISIs are either well below 10ms within a burst or well above
15ms between bursts) but is checked explicitly in the intermediate report.
"""
from __future__ import annotations

import numpy as np

ISI_MAX_S = 0.010   # 10ms: max intra-burst ISI
TAIL_ISI_MIN_S = 0.015  # 15ms: min tail ISI to confirm burst termination
DEAD_ZONE_PRE_S = 0.001   # 1ms before start_time (ssl_artifact_dead_zone.md)
DEAD_ZONE_POST_S = 0.004  # 4ms after start_time
RESPONSE_WINDOW_S = (0.005, 0.050)   # 5-50ms post start_time
BASELINE_WINDOW_S = (-0.200, -0.010)  # -200ms to -10ms pre start_time


def label_bursts(spike_times: np.ndarray, isi_max: float = ISI_MAX_S, tail_isi_min: float = TAIL_ISI_MIN_S) -> np.ndarray:
    """Vectorized-by-run ISI burst labeling.

    :param spike_times: (n,) sorted spike times in seconds.
    :return: (n,) bool array, True where the spike belongs to a valid burst.
    """
    n = len(spike_times)
    is_burst = np.zeros(n, dtype=bool)
    if n < 2:
        return is_burst
    isi = np.diff(spike_times)
    short = isi <= isi_max  # True between spike i and i+1 if ISI qualifies as intra-burst

    # Contiguous True-runs in `short` via edge detection (few runs vs many spikes).
    padded = np.concatenate(([False], short, [False]))
    run_starts = np.flatnonzero(~padded[:-1] & padded[1:])       # index into `short`
    run_ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1     # inclusive index into `short`

    for a, b in zip(run_starts, run_ends):
        first_spike, last_spike = a, b + 1
        tail_ok = last_spike == n - 1 or isi[last_spike] > tail_isi_min
        if tail_ok:
            is_burst[first_spike:last_spike + 1] = True
    return is_burst


def burst_runs(spike_times: np.ndarray, isi_max: float = ISI_MAX_S, tail_isi_min: float = TAIL_ISI_MIN_S) -> list[dict]:
    """Same run-detection as `label_bursts`, but returns each valid burst's
    own (n_spikes, duration_s, start_time) instead of a flat boolean mask --
    for characterizing burst structure itself (spikes/burst, burst duration,
    inter-burst interval), not just the per-spike label."""
    n = len(spike_times)
    if n < 2:
        return []
    isi = np.diff(spike_times)
    short = isi <= isi_max
    padded = np.concatenate(([False], short, [False]))
    run_starts = np.flatnonzero(~padded[:-1] & padded[1:])
    run_ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1

    runs = []
    for a, b in zip(run_starts, run_ends):
        first_spike, last_spike = a, b + 1
        tail_ok = last_spike == n - 1 or isi[last_spike] > tail_isi_min
        if tail_ok:
            runs.append({
                "n_spikes": last_spike - first_spike + 1,
                "duration_s": float(spike_times[last_spike] - spike_times[first_spike]),
                "start_time": float(spike_times[first_spike]),
            })
    return runs


def excise_whisker_dead_zone(spike_times: np.ndarray, whisker_start_times: np.ndarray,
                              pre: float = DEAD_ZONE_PRE_S, post: float = DEAD_ZONE_POST_S) -> np.ndarray:
    """Remove spikes in [start_time-pre, start_time+post) for every whisker
    trial, per the mandatory ssl_artifact_dead_zone.md exclusion -- applied
    before burst labeling so a real spike next to the dead zone can't be
    spuriously chained to an artifact-corrupted spike inside it.

    Vectorized (diff-array + cumsum) rather than a per-trial Python loop --
    this runs once per unit across the whole population (~hundreds of
    thousands of calls), so the per-trial-loop version was a real bottleneck.
    np.add.at correctly handles overlapping dead-zone windows from
    closely-spaced trials (multiplicities cancel via the >0 threshold).
    """
    if len(whisker_start_times) == 0 or len(spike_times) == 0:
        return spike_times
    lo = np.searchsorted(spike_times, whisker_start_times - pre, side="left")
    hi = np.searchsorted(spike_times, whisker_start_times + post, side="left")
    diff = np.zeros(len(spike_times) + 1, dtype=np.int32)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi, -1)
    mask = np.cumsum(diff[:-1]) > 0
    return spike_times[~mask]


def continuous_burstiness(spike_times: np.ndarray, whisker_start_times: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fraction of burst spikes among all spikes over the whole session,
    whisker dead-zone spikes excised first. Returns (burstiness, clean_spike_times, is_burst).
    """
    clean = excise_whisker_dead_zone(spike_times, whisker_start_times)
    is_burst = label_bursts(clean)
    burstiness = is_burst.sum() / len(clean) if len(clean) else np.nan
    return burstiness, clean, is_burst


def _window_counts(spike_times: np.ndarray, is_burst: np.ndarray, start_times: np.ndarray,
                    offset: tuple[float, float]) -> tuple[int, int]:
    """Pool spike/burst counts across all trials into one window bucket.
    Vectorized: searchsorted accepts the whole `start_times` array at once,
    and a cumulative-sum of `is_burst` turns each trial's burst count into an
    O(1) range-sum lookup -- avoids a per-trial Python loop (same rationale
    as `excise_whisker_dead_zone`)."""
    if len(start_times) == 0:
        return 0, 0
    lo_off, hi_off = offset
    los = np.searchsorted(spike_times, start_times + lo_off, side="left")
    his = np.searchsorted(spike_times, start_times + hi_off, side="left")
    n_total = int((his - los).sum())
    cum_burst = np.concatenate(([0], np.cumsum(is_burst)))
    n_burst = int((cum_burst[his] - cum_burst[los]).sum())
    return n_burst, n_total


def burst_index(spike_times: np.ndarray, whisker_start_times: np.ndarray, trial_start_times: np.ndarray,
                 is_whisker: bool) -> dict:
    """Sensory-evoked burst index for one trial_type: fB(response 5-50ms) -
    fB(baseline -200/-10ms) around start_time, spikes pooled across all
    `trial_start_times` of that type. fB=0 if a window has no spikes (per
    spec), so response-only/baseline-only bursting units aren't dropped.
    Dead-zone excision only applies when `is_whisker` (auditory trials have
    no stimulus artifact, ssl_artifact_dead_zone.md).
    """
    clean = excise_whisker_dead_zone(spike_times, whisker_start_times) if is_whisker else spike_times
    is_burst = label_bursts(clean)
    n_burst_r, n_total_r = _window_counts(clean, is_burst, trial_start_times, RESPONSE_WINDOW_S)
    n_burst_b, n_total_b = _window_counts(clean, is_burst, trial_start_times, BASELINE_WINDOW_S)
    fb_r = n_burst_r / n_total_r if n_total_r > 0 else 0.0
    fb_b = n_burst_b / n_total_b if n_total_b > 0 else 0.0
    return {
        "n_burst_response": n_burst_r, "n_total_response": n_total_r, "fb_response": fb_r,
        "n_burst_baseline": n_burst_b, "n_total_baseline": n_total_b, "fb_baseline": fb_b,
        "burst_index": fb_r - fb_b,
    }
