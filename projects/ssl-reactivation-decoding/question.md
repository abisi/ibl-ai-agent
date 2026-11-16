# SSL reactivation decoding

## Request (Axel Bisi, 2026-10-16; project started 2026-10-23)
Apply stimulus-presence decoders (from ssl-stimulus-detection-decoding) to continuous data, to detect whisker
"reactivation" events (whisker-stimulus-like population patterns outside stimulus times). Study their shape, content and
statistics across learning (day 0 vs expert), cohorts (R+ / R-) and relative to task events (stimuli, licks, rewards, ITI,
passive blocks). Same pipeline for auditory reactivations as a control.

## Decisions so far (user, 2026-10-16 / 10-18)
| Item | Decision |
|---|---|
| Stage | day 0 (learning) and expert sessions always kept separate, never pooled |
| Control | auditory reactivations, same pipeline |
| Training data | ALL whisker trials (task hits + misses, + passive where present) vs no stimulus; not hit-only |
| Main model ("clean template") | quiet stimulus trials vs correct rejections (both after a quiet window, state-matched); no state hard-negatives |
| Training window | 5-35 ms after stimulus onset |
| Fit objective | low false positives: choose C by sensitivity at a fixed low FPR / partial AUC, not accuracy; stim / no-stim balanced within context |
| Decoder | pseudo-population weights cannot be applied to one session as is. Options: (a) pooled code = per-neuron weights averaged over the pseudo-population iterations each neuron appeared in (dedicated run saving weights at the training window), applied per session; (b) per-session decoders with an identical recipe; compare both (agreement = shared code; stage disagreement = drift) |
| Comparable chance | fixed N neurons per area (subsampled), normalised readout |
| Session-own nulls | decoder-weight shuffle across neurons (keeps population-wide fluctuations) AND independent LINEAR (non-wrapping) time shifts of spike trains (keeps rates / autocorrelation, breaks co-activation) |
| Threshold | per session, at a fixed null event rate (e.g. 0.1 / min); an event must exceed both nulls; report excess over null (difference / ratio) |
| Exclusions / validation | real stimulus windows excluded from detection and used as validation; per-session hit rate on held-out stimuli at the threshold reported (fixing FPR does not equalise sensitivity) |
| States | FPR measured per state (quiet / whisking / licking / grooming, DLC + licks) on held-out continuous windows; first explore what is detected, then decide on changes (restrict to quiet, or a state hard-negative model) |
| Checks | DLC whisking inside training windows; task <-> passive cross-decoding where passive exists; later: project events on the hit / miss axis for decision content |

## Decisions 2026-10-23 (user)
| Item | Decision |
|---|---|
| Decoder | per-session decoders first (pooled code later); start with example sessions and visualisations, check before any full run |
| Populations | key area groups at fixed subsampled N: Ss-whisker, Thalamus, Midbrain, Striatum, Motor (+ Auditory where recorded) |
| Periods | continuous readout over the whole session, then split into chunks by epoch (passive pre, task, passive post) and trial type (time after hit / miss / CR / FA / auditory trials, ITIs) |
| Pilot | 4 sessions per stage (2 R+, 2 R-), picked for units in the key groups, passive blocks and dual-view video; learners (sheet learning_category not bad) |

Pilot sessions (cohort from joint_mouse_reference_weight.xlsx):
| Stage | R+ | R- |
|---|---|---|
| day 0 | MH070_20260121_140848, AB150_20241219_144646 | AB136_20241117_140311, AB157_20250412_154744 |
| expert | MH030_20250503_154256, MH031_20250509_102210 | MH036_20250517_140456, MH035_20250517_105351 |

## Open
- Fixed N per area group (proposed 50) and readout step (proposed 30-ms window = training window length, 5-ms step).
- Figure list (README) to be confirmed before coding.
