"""Faithful reimplementation of the artifact-correction algorithm in
M:\\analysis\\Axel_Bisi\\unit_spikes_analysis\\neural_utils.py
(compute_unit_peri_event_histogram, compute_trial_baseline_from_peth),
which is the actual method in use -- NOT the -10ms/+5ms population-level
Poisson-interpolation procedure documented in
ssl-analyze/references/ssl_artifact_dead_zone.md, which the user identified
as wrong.

Key differences from what this project (and ssl_artifact_dead_zone.md)
assumed:
- Artifact window is [-1ms, +4ms) (art_start=-1, art_stop=stim_dur+1=4 with
  stim_dur=3), a 5ms window -- not [-10ms, +5ms) (15ms).
- Correction is per-trial, per-unit: each trial's own artifact-window bins
  are replaced with independent Poisson draws using that SAME trial's own
  baseline rate (mean count/1ms-bin over bins [0:bas_stop], bas_stop = the
  bin 5ms before stimulus), computed from that trial's own PETH row -- not
  a population-pooled curve with an interpolated lambda.
- Correction happens at 1ms bins, before any trial-averaging; a coarser
  final bin_size is obtained by reshaping and summing 1ms bins afterward
  (non-overlapping -- no stride/sliding-window concept in the reference
  implementation).

Trial-histogram construction is vectorized with searchsorted instead of the
reference's per-trial np.histogram loop, for speed at ~103k units; the
statistics (baseline computation, Poisson replacement, rebinning) are
otherwise identical to the reference.
"""

from __future__ import annotations

import numpy as np

BIN_SIZE_HIST = 0.001
STIM_DUR_MS = 3
ART_START_MS = -1
ART_STOP_MS = STIM_DUR_MS + 1  # +4


def build_trial_histograms(spike_times: np.ndarray, event_times: np.ndarray, time_start: float, time_stop: float) -> np.ndarray:
    """(n_trials, n_bins) 1ms-bin spike counts, vectorized equivalent of the
    reference function's per-trial np.histogram loop."""
    n_bins = int(round((time_stop - time_start) / BIN_SIZE_HIST))
    edges = time_start + np.arange(n_bins + 1) * BIN_SIZE_HIST
    n_trials = len(event_times)
    hist = np.zeros((n_trials, n_bins), dtype=np.int64)
    for i, t0 in enumerate(event_times):
        lo = np.searchsorted(spike_times, t0 + time_start, side="left")
        hi = np.searchsorted(spike_times, t0 + time_stop, side="left")
        rel = spike_times[lo:hi] - t0
        bin_idx = np.clip(np.searchsorted(edges, rel, side="right") - 1, 0, n_bins - 1)
        np.add.at(hist[i], bin_idx, 1)
    return hist


def compute_unit_peri_event_histogram(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    bin_size: float,
    time_start: float,
    time_stop: float,
    artifact_correction: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if artifact_correction:
        bin_size_hist = BIN_SIZE_HIST
    else:
        bin_size_hist = bin_size

    n_bins = int(round((time_stop - time_start) / bin_size_hist))
    peri_stim_hist = build_trial_histograms(spike_times, event_times, time_start, time_stop).astype(np.float64)
    if not artifact_correction:
        return peri_stim_hist

    art_start_bin = int(abs(time_start) / bin_size_hist) + ART_START_MS
    art_stop_bin = int(abs(time_start) / bin_size_hist) + ART_STOP_MS

    bas_stop = int(abs(time_start) / bin_size_hist) - 5
    trial_baselines = np.mean(peri_stim_hist[:, 0:bas_stop], axis=1)

    if rng is None:
        rng = np.random.default_rng()
    art_width = art_stop_bin - art_start_bin
    poisson_noise = rng.poisson(lam=trial_baselines[:, None], size=(len(event_times), art_width))
    peri_stim_hist[:, art_start_bin:art_stop_bin] = poisson_noise

    if bin_size != bin_size_hist:
        current_bin_size_ms = round(bin_size_hist * 1000)
        new_bin_size_ms = round(bin_size * 1000)
        n_trials = peri_stim_hist.shape[0]
        peri_stim_hist = peri_stim_hist.reshape(n_trials, -1, new_bin_size_ms // current_bin_size_ms).sum(axis=2)

    return peri_stim_hist
