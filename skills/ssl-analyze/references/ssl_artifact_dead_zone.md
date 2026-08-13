## Purpose
Mandatory, general rule for excluding whisker-trial stimulus-artifact spikes
from any neural computation on the SSL dataset. Confirmed by Axel Bisi
2026-08-14. Unlike the rest of `ssl_analysis_patterns.md`, this is not a
prior-project convention that a new analysis may deviate from — it is a
correctness requirement: spikes in this window are not trustworthy data,
regardless of which project or analysis is using them.

## The rule
**Whisker trials only** (magnetic-particle stimulation produces a real
electrical/mechanical artifact around stimulus delivery; auditory and
no-stimulus trials do not have this artifact and are not subject to this
rule): spikes in the window **-10ms to +5ms relative to trial `start_time`**
(15ms total, asymmetric) are not trustworthy and must never be included in
any computation — firing rates, PSTHs, tensors for TCA, GLMM inputs,
anything. Treat this as a **dead zone**, not a small-bias-acceptable region.

Concretely, for any whisker-trial window that would otherwise span across
`start_time`:
- A **baseline window** ending at `start_time` must instead end at
  `start_time - 10ms` (exclude the last 10ms before start).
- A **response/stimulus window** starting at `start_time` must instead start
  at `start_time + 5ms` (exclude the first 5ms after start).
- Any window that would otherwise fall entirely inside `-10ms..+5ms` cannot
  be computed from real spikes at all for that trial.

This is a **minimum** exclusion. An analysis may use a wider dead zone (e.g.
`ssl-passive-sensory-selectivity`'s response windows start at +10ms, wider
than the +5ms minimum here — see `ssl_analysis_patterns.md`'s passive
section) without conflicting with this rule. A narrower exclusion is not
acceptable. That project also applied the same 10ms exclusion to auditory
trials, but that was a design-symmetry choice for cross-modality
comparability in that specific project, not because auditory trials have
this artifact — do not generalize "10ms" as an auditory requirement from
that precedent.

This rule assumes `trials.parquet`'s `start_time` is the artifact-relevant
reference point (consistent with how existing SSL projects anchor response/
baseline windows to it). If a future analysis has reason to believe the
actual magnetic-stimulus delivery time differs from `start_time` for some
sessions, flag that explicitly rather than assuming.

## PSTH / visualization-only Poisson dead-zone masking
For plots where the artifact gap would look jarring (event-aligned PSTHs,
single-trial rasters used for illustration), real spikes in the dead zone may
be **replaced** with synthetic homogeneous-Poisson-process spikes, purely for
visual smoothness. This is **cosmetic only** — never feed these synthetic
spikes into a firing-rate statistic, response-magnitude metric, tensor, or
any quantitative computation; only into the rendered plot.

Procedure, per (neuron, trial):
1. Estimate that neuron's rate on that trial, `lambda_hz`, as the mean firing
   rate over the **2 seconds immediately before `start_time`** (a pre-stimulus
   baseline estimate, wider than the -10ms/+5ms dead zone itself).
2. Generate spike times via a homogeneous Poisson process at `lambda_hz`,
   restricted to the 15ms dead-zone window (`start_time - 10ms` to
   `start_time + 5ms`).
3. Splice these synthetic spikes into the plotted spike train / PSTH bin
   counts in place of (not in addition to) the excluded real spikes for that
   window.
4. Label the affected window in the figure (e.g. shaded band, caption note)
   so a reader knows that segment is synthetic, not measured.

Minimal implementation, using a fixed seed per
`../../ibl-analyze/references/reproducibility_qc.md`'s "deterministic seeds
for stochastic steps" rule:
```python
import numpy as np

DEAD_ZONE_PRE_S = 0.010   # 10ms before start_time
DEAD_ZONE_POST_S = 0.005  # 5ms after start_time
BASELINE_RATE_WINDOW_S = 2.0

def dead_zone_poisson_spikes(lambda_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic spike offsets (seconds, relative to start_time) filling the
    dead zone for PSTH cosmetics only — never for quantitative analysis."""
    window_s = DEAD_ZONE_PRE_S + DEAD_ZONE_POST_S
    n = rng.poisson(lambda_hz * window_s)
    offsets = rng.uniform(-DEAD_ZONE_PRE_S, DEAD_ZONE_POST_S, size=n)
    return np.sort(offsets)

# per (neuron, trial):
#   lambda_hz = n_spikes_in(start_time - BASELINE_RATE_WINDOW_S, start_time) / BASELINE_RATE_WINDOW_S
#   synthetic_offsets = dead_zone_poisson_spikes(lambda_hz, rng)
#   synthetic_times = start_time + synthetic_offsets
```

## Quality gates
- Reject any whisker-trial firing-rate, PSTH, or tensor-building computation
  that includes real spikes from `-10ms` to `+5ms` around `start_time`.
- Reject any analysis that treats Poisson dead-zone-masking spikes as real
  data in a statistic, model input, or tensor — masking is visualization-only.
- Reject applying this whisker-specific artifact exclusion to auditory or
  no-stimulus trials as if it were a general stimulus-artifact rule for all
  modalities.
