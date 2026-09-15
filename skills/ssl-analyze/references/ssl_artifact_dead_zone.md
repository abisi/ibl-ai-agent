## Purpose
Mandatory, general rule for excluding whisker-trial stimulus-artifact spikes
from any neural computation on the SSL dataset. Unlike the rest of
`ssl_analysis_patterns.md`, this is not a prior-project convention that a
new analysis may deviate from — it is a correctness requirement: spikes in
this window are not trustworthy data, regardless of which project or
analysis is using them.

**Correction (2026-08-15)**: the window and the visualization-correction
method below were rewritten after `projects/ssl-whisker-stim-artifact-qc/`
found the previous version of this file (window `-10ms/+5ms`,
population-pooled Poisson interpolation) did not match the actual
correction method in use, and was not empirically validated. The version
below is sourced directly from
`M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\neural_utils\neural_utils.py`
(path corrected 2026-08-22, re-verified against source; both functions
confirmed present at this path)
(`compute_unit_peri_event_histogram`, `compute_trial_baseline_from_peth` —
the same "Path B" external pipeline referenced in
`ssl-load/references/ssl_loading_policy.md`), and empirically checked
against raw spike data (see that project's report for the full analysis).
If this file and that source file ever disagree again, the source file
wins — re-derive from it rather than trusting this description.

## The rule
**Whisker trials only** (magnetic-particle stimulation produces a real
electrical/mechanical artifact around stimulus delivery; auditory and
no-stimulus trials do not have this artifact and are not subject to this
rule): spikes in the window **-1ms to +4ms relative to trial `start_time`**
(5ms total, asymmetric — an assumed 3ms stimulus duration, `art_start =
-1ms`, `art_stop = stim_dur(3ms) + 1ms = +4ms`) are not trustworthy and
must never be included in any computation — firing rates, PSTHs, tensors
for TCA, GLMM inputs, anything. Treat this as a **dead zone**, not a
small-bias-acceptable region.

Empirical support (`projects/ssl-whisker-stim-artifact-qc/`, 103,054 units,
81 sessions, native 1ms-resolution raw/uncorrected spike counts): a sharp
spike-detection dropout at +1.5ms (population rate 0.36 Hz vs. ~5 Hz
baseline) immediately followed by a spurious spike burst at +3.5ms (11 Hz)
— both fall inside `-1ms..+4ms`. The dropout, the cleaner and unambiguous
artifact marker (a below-baseline dip cannot be confused with a genuine
sensory response), is contained inside this exact window in 80/81 analyzed
sessions.

Concretely, for any whisker-trial window that would otherwise span across
`start_time`:
- A **baseline window** ending at `start_time` must instead end at
  `start_time - 1ms` (exclude the last 1ms before start).
- A **response/stimulus window** starting at `start_time` must instead start
  at `start_time + 4ms` (exclude the first 4ms after start).
- Any window that would otherwise fall entirely inside `-1ms..+4ms` cannot
  be computed from real spikes at all for that trial.

This is a **minimum** exclusion. A wider dead zone remains safe/conservative
— it only discards additional real (non-artifact) data, it does not admit
artifact-contaminated data — and existing analyses that used the previous,
wider `-10ms/+5ms` window (e.g.
`../../projects/ssl-whisker-auditory-cohort-modulation/`, whose baseline
window ended at `start_time - 10ms` and evoked window started at
`start_time + 5ms`) are **not biased** by having used the old window; they
were simply more conservative than necessary and do not need to be rerun on
that basis alone. `ssl-passive-sensory-selectivity`'s response windows
starting at +10ms are likewise still a valid (wider) choice — see
`ssl_analysis_patterns.md`'s passive section. A narrower exclusion than
`-1ms/+4ms` is not acceptable. That project also applied the same 10ms
exclusion to auditory trials, but that was a design-symmetry choice for
cross-modality comparability in that specific project, not because
auditory trials have this artifact — do not generalize any of these
numbers as an auditory requirement from that precedent.

This rule assumes `trials.parquet`'s `start_time` is the artifact-relevant
reference point. Directly checked in `ssl-whisker-stim-artifact-qc`:
`start_time`, `whisker_stim_time`, and `stim_onset` are identical for every
whisker trial (zero offset, zero jitter, 17,327 trials checked) — there is
no alternative timestamp in the data that differs from `start_time`, so
this assumption is confirmed, not just plausible.

## PSTH / visualization-only Poisson dead-zone correction
For plots where the artifact gap would look jarring (event-aligned PSTHs,
single-trial rasters used for illustration), real spikes in the dead zone
may be **replaced** with synthetic Poisson-process spikes, purely for visual
smoothness. This is **cosmetic only** — never feed these synthetic spikes
into a firing-rate statistic, response-magnitude metric, tensor, or any
quantitative computation; only into the rendered plot. This visualization
convention uses a **wider window than the `-1ms/+4ms` quantitative minimum
above** (`-10ms/+5ms`) — that is intentional (see "This is a minimum
exclusion" above: wider is always an acceptable, more conservative choice),
not a re-opening of where the real artifact is.

**Corrected 2026-08-15** (superseding an earlier, binned-per-unit version
of this section that used the narrow `-1ms/+4ms` window and per-bin Poisson
counts, adapted from `neural_utils.py`'s `compute_unit_peri_event_histogram`
— that binned approach is still the source for the `-1ms/+4ms` quantitative
minimum above, but per direct user specification this is a **different**,
spike-time-based procedure for visualization specifically):

This is a **per-trial, per-unit** correction on continuous spike **times**
(not binned counts), applied before any averaging across trials — not a
correction applied to an already-trial-averaged population curve. Each
trial's own replacement-window spikes are replaced using a Poisson rate
drawn from that same trial's own pre-window baseline, not a shared/global
or population-level rate.

Procedure, per (unit, trial), whisker trials only:
1. Baseline rate `lambda_hz` = that trial's own spike count in
   `[time_start, -10ms)` divided by that window's duration. This baseline
   window **excludes the entire `[-10ms, +5ms)` replacement window** — it
   does not extend up to stimulus onset the way an ordinary pre-stimulus
   baseline would.
2. Remove that trial's real spikes in `[-10ms, +5ms)`.
3. Draw `n_synthetic ~ Poisson(lambda_hz * 0.015s)`, then `n_synthetic`
   spike times independently and identically distributed
   `Uniform(-10ms, +5ms)` (the standard count-then-uniform-placement
   construction of a homogeneous Poisson process on an interval).
4. That trial's corrected spike times = its kept real spikes (outside the
   replacement window) + the synthetic spikes from step 3.
5. Bin/average the corrected per-trial spike times as usual for the plot
   (any bin size or sliding window — since correction is on continuous
   times, not fixed counting bins, there is no rebinning-order constraint
   the way the binned quantitative procedure has).
6. Label the affected window in the figure (e.g. shaded band, caption note)
   so a reader knows that segment is synthetic, not measured.

Reference implementation (`projects/ssl-whisker-stim-artifact-qc/exploratory-analyses/peri_event_histogram_v2.py`):
```python
import numpy as np

REPLACE_START_S, REPLACE_STOP_S = -0.010, 0.005
REPLACE_WINDOW_S = REPLACE_STOP_S - REPLACE_START_S

def corrected_trial_spike_times(spike_times, event_times, time_start, time_stop, rng):
    """Per whisker trial, corrected spike times relative to start_time."""
    out = []
    for t0 in event_times:
        rel = spike_times[(spike_times >= t0 + time_start) & (spike_times < t0 + time_stop)] - t0

        baseline_duration = REPLACE_START_S - time_start
        baseline_hz = (rel < REPLACE_START_S).sum() / baseline_duration

        kept = rel[(rel < REPLACE_START_S) | (rel >= REPLACE_STOP_S)]
        n_synthetic = rng.poisson(baseline_hz * REPLACE_WINDOW_S)
        synthetic = rng.uniform(REPLACE_START_S, REPLACE_STOP_S, size=n_synthetic)

        out.append(np.sort(np.concatenate([kept, synthetic])))
    return out
```

## Quality gates
- Reject any whisker-trial firing-rate, PSTH, or tensor-building computation
  that includes real spikes from `-1ms` to `+4ms` around `start_time`. A
  wider exclusion (e.g. the historical `-10ms/+5ms`) is also acceptable,
  just more conservative than required.
- Reject any analysis that treats Poisson dead-zone-correction spikes as
  real data in a statistic, model input, or tensor — correction is
  visualization-only.
- Reject a dead-zone Poisson correction computed from a population-pooled
  or already-trial-averaged curve — it must be per-trial, per-unit, using
  that trial's own baseline, applied before averaging.
- Reject applying this whisker-specific artifact exclusion to auditory or
  no-stimulus trials as if it were a general stimulus-artifact rule for all
  modalities.
