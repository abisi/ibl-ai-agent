"""Event-window firing-rate helpers for the SSL BWM-style single-cell tests,
built fresh against the **current** dead-zone spec
(`skills/ssl-analyze/references/ssl_artifact_dead_zone.md`, -1ms/+4ms) --
deliberately not reusing `scripts/ssl_wm_lib.py`, which still hardcodes the
superseded -10ms/+5ms window from before the 2026-08-15 correction.
"""

from __future__ import annotations

import numpy as np

DEAD_ZONE_START_S = -0.001  # start_time - 1ms
DEAD_ZONE_STOP_S = 0.004    # start_time + 4ms


def clip_window_for_whisker(window: tuple[float, float], is_whisker: bool) -> tuple[float, float] | None:
    """Clip a (start, end) window, seconds relative to trial start_time,
    against the mandatory whisker dead zone. Returns None if the window is
    fully inside the dead zone. Raises if the window straddles the entire
    dead zone (ambiguous for a single contiguous window -- callers should
    not construct such windows)."""
    start, end = window
    if not is_whisker:
        return (start, end)
    dz0, dz1 = DEAD_ZONE_START_S, DEAD_ZONE_STOP_S
    if start >= dz0 and end <= dz1:
        return None
    if start < dz0 and end > dz1:
        raise ValueError(f"window {window} straddles the entire dead zone; split it explicitly")
    new_start, new_end = start, end
    if start < dz0 < end <= dz1:
        new_end = dz0
    if dz0 <= start < dz1 < end:
        new_start = dz1
    if new_start >= new_end:
        return None
    return (new_start, new_end)


def unit_rates_for_trials(
    spike_times_sorted: np.ndarray,
    trial_start_times: np.ndarray,
    trial_is_whisker: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Per-trial firing rate (Hz) for one unit's sorted spike times, in
    `window` (seconds relative to `start_time`), dead-zone-clipped for
    whisker trials. NaN for a trial whose clipped window is empty (fully
    inside the dead zone)."""
    n = len(trial_start_times)
    rates = np.full(n, np.nan)

    clipped_wh = clip_window_for_whisker(window, is_whisker=True)
    clipped_other = clip_window_for_whisker(window, is_whisker=False)

    for is_wh, clipped in ((True, clipped_wh), (False, clipped_other)):
        mask = trial_is_whisker == is_wh
        if not mask.any() or clipped is None:
            continue
        w0, w1 = clipped
        starts = trial_start_times[mask]
        lo = np.searchsorted(spike_times_sorted, starts + w0, side="left")
        hi = np.searchsorted(spike_times_sorted, starts + w1, side="left")
        counts = (hi - lo).astype(np.float64)
        rates[mask] = counts / (w1 - w0)

    return rates
