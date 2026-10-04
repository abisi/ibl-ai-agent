"""Visualization-only artifact correction, v2 per user refinement
(2026-08-15): spike-time-based (not binned), replacement window
[-10ms, +5ms) (not the -1/+4ms quantitative minimum from neural_utils.py --
that remains the correctness floor for real computations; this is a wider,
cosmetic-only window for plots, same as ssl_artifact_dead_zone.md always
allowed "an analysis may use a wider dead zone").

Per trial, per unit, whisker trials only:
1. baseline_hz = that trial's own spike rate over [time_start, -10ms)
   (explicitly excludes the whole [-10ms, +5ms) replacement window from
   contributing to its own replacement rate).
2. Remove real spikes in [-10ms, +5ms) for that trial.
3. n_synthetic ~ Poisson(baseline_hz * 0.015s); synthetic offsets ~
   Uniform(-10ms, +5ms), independently, i.e. a homogeneous Poisson process
   realized via the standard "Poisson count + uniform placement" equivalence.
4. Corrected trial spike times = kept real spikes + synthetic spikes.
"""

from __future__ import annotations

import numpy as np

REPLACE_START_S = -0.010
REPLACE_STOP_S = 0.005
REPLACE_WINDOW_S = REPLACE_STOP_S - REPLACE_START_S


def corrected_trial_spike_times(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    time_start: float,
    time_stop: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Per trial (whisker trials only -- caller passes only whisker event
    times), corrected spike times relative to that trial's start_time,
    within [time_start, time_stop)."""
    out = []
    for t0 in event_times:
        lo = np.searchsorted(spike_times, t0 + time_start, side="left")
        hi = np.searchsorted(spike_times, t0 + time_stop, side="left")
        rel = spike_times[lo:hi] - t0

        baseline_mask = rel < REPLACE_START_S
        baseline_duration = REPLACE_START_S - time_start
        baseline_hz = baseline_mask.sum() / baseline_duration if baseline_duration > 0 else 0.0

        keep_mask = (rel < REPLACE_START_S) | (rel >= REPLACE_STOP_S)
        kept = rel[keep_mask]

        n_synthetic = rng.poisson(baseline_hz * REPLACE_WINDOW_S)
        synthetic = rng.uniform(REPLACE_START_S, REPLACE_STOP_S, size=n_synthetic)

        out.append(np.sort(np.concatenate([kept, synthetic])))
    return out
