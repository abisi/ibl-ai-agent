# SSL stimulus-arrival decoding

## Request (Axel Bisi, 2026-10-04)
Rebuild the pseudo-population decoding of whisker vs auditory stimulus, time-resolved and stimulus-aligned, pooling all
data (both cohorts, learning day and expert days), to map in detail the arrival of sensory information. Repeat with 20-ms
bins and 2-ms steps over -20..+100 ms; show both resolutions side by side; find the first significant bin at both
resolutions. Vary the pseudo-population size (good + mua, 20 to 500 neurons). Figures: time-resolved corrected balanced
accuracy per area and neuron count; 5-50 ms accuracy vs neuron count; onset vs early accuracy; matched-accuracy curves
("how many neurons would area B need to match area A").

## Method (confirmed with the user, 2026-10-04)
| Item | Decision |
|---|---|
| Decoder | ssl-pseudopopulation-area-decoding 002 (imported unchanged): L2 logistic per bin, 3-fold outer CV built on each session's real trials, inner 2-fold for C, 100 + 100 pseudo-trials per class, balanced trial reuse, z-scoring in the training fold, balanced accuracy |
| Data | all whisker-training sessions of the v2 unit table (both cohorts, day 0 and expert days); KS4 NWB |
| Spikes | whisker-artefact-corrected spike trains (roc_utils_new: -10..+5 ms around each whisker onset replaced by a Poisson train at the pre-onset rate, seeded per session); no dead-zone excision on top (user: "decoding on the corrected data") |
| Trials | active, perf != 6, warm-up cut (1 trial before the first whisker trial kept), A1 tail trim; all whisker vs all auditory trials |
| Units | good + mua (v2 quality_label); session eligible for an area with >= 5 units and >= 3 trials per class |
| Sampling | sessions are independent: 20 sessions with replacement -> units of the area within each session (with replacement; N / 20 per session, remainder spread at random) -> trials |
| N | 20, 50, 100, 200, 300, 500 (+ matched counts) |
| Null | trial shuffling: labels permuted within each session before pooling, same draw and same per-bin C; 10 shuffles averaged; reported: corrected balanced accuracy d = real - null |
| Significance | bin above chance when the 5th percentile of d over iterations > 0 |
| Onset | first post-stimulus bin, itself above chance, with >= 80 % of the bins in the next 25 ms above chance (wide: 4 of 5; zoom: 11 of 13); the literal "20 of 25 bins" (50 ms) is also stored |
| Resolutions | wide: causal 50-ms bins, 5-ms steps, -200..+600 ms; zoom: causal 20-ms bins, 2-ms steps, -20..+100 ms |
| Areas | area groups: Somatosensory-whisker, Auditory areas, Motor areas, Midbrain, Striatum, Thalamus; areas: SSp-bfd, SSs, SCm, MO-wM1, MO-wM2, DMS, DLS, MO-ALM |
| Early window | mean d over the zoom bins ending 5..50 ms after stimulus onset |
| Matched accuracy | references Somatosensory-whisker, Midbrain, Auditory areas at N = 100; each area's matched N from its accuracy-vs-N curve (log-N interpolation; extrapolated beyond 500, cap 2000), then decoded at that N |
| Iterations | 100 iterations x 10 shuffles: PILOT values |

## Notes
- The onset rule: an earlier message proposed "20 of the next 25 bins" as the same time span as the 4-of-5 rule; at 2-ms
  steps that spans 50 ms, not 25 ms. The primary rule keeps the 25-ms span (80 % of the bins); the 50-ms version is
  stored alongside (onset_zoom_20of25bins_ms).
- Whisker trials: spikes in -10..+5 ms are replaced by baseline-rate Poisson spikes, so information can only appear from
  +5 ms on whisker trials; a causal 20-ms bin ending at t covers t-20..t.
- MH062_20260113_125836: the NWB trials table has 377 trials with context "nan", 241 passive and 21 active; the standard
  rule keeps the 21 active trials (as every other SSL analysis). Flagged to the user.
