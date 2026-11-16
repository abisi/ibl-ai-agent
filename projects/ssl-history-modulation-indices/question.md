# Question

## Original question (from user, 2026-08-21)
Derive single-neuron-level history-effect quantities for the SSL task,
inspired by RHMI/EHMI (Reward-/Error-History Modulation Index) from prior
literature: two consecutive reward outcomes indicate a "hit-stay" cue
(RHMI deflection >0 = enhanced response, <0 = suppressed); a mirror
"FA-shift" cue drives EHMI. Wanted **cohort-specific** (R+/R-) metrics for
both correct-history and error-history, with whisker and auditory trials
**pooled together** (not split by modality), and `no_stim_trial` excluded
entirely.

## Locked design decisions (2026-08-21)

### Two non-interchangeable per-trial booleans
The existing `ssl_bwm_trial_prep.py::cohort_corrected_rewarded()` (used by
the `ssl-bwm-style-single-cell-decoding` project) actually computes task
**correctness**, not literal reward receipt, and calls it "rewarded" — this
never mattered for R+ (correct==rewarded always there) but is wrong for R-
whisker trials, where withholding is *correct* but delivers **no water**
(confirmed by the user: "withhold does not lead to a reward in R-"). This
project defines two separate booleans instead of reusing that function:

- **`correct`** (cohort-corrected task rule; defines which trials fall in
  the RHMI vs EHMI pool):
  - R+: correct = lick (whisker AND auditory).
  - R-: correct = withhold/no-lick (whisker), lick (auditory).
- **`reward_delivered`** (literal water; the history/conditioning factor):
  `lick_flag & (auditory_trial | (whisker_trial & reward_group=="R+"))`.
  True only for R+ whisker-lick and for auditory-lick in either cohort.
  **Never** true for any R- whisker trial, correct or not.

`no_stim_trial` rows are excluded from both `t` and `t-1` entirely — only
`whisker_trial`/`auditory_trial` rows count, mixed together (no modality
split), per user request.

### Metric definition
Per unit, per cohort, computed separately over the active-epoch
whisker+auditory trial pool of that cohort's sessions:

- **RHMI**: `t` = trials where `correct == True` (mixed modality). Split by
  whether `t-1` (the immediately preceding active trial, any modality)
  had `reward_delivered == True` vs `False`.
- **EHMI**: same, but `t` = trials where `correct == False`.

```
index = (mean_FR(t | t-1 rewarded) - mean_FR(t | t-1 non-rewarded)) / mean_FR(all t-trials, both t-1 buckets pooled)
```

`mean_FR` = per-trial spike count in the evoked window, divided by window
duration (Hz). >0 = response enhanced following a rewarded previous trial;
<0 = suppressed.

Consequence flagged and accepted by the user: for R-, `t-1`-rewarded is
essentially "was the immediately preceding trial an auditory hit" (since R-
whisker trials never deliver water regardless of outcome) — R-'s RHMI/EHMI
are therefore a recent-auditory-reward-history effect on the (mostly
whisker) response now, structurally different in kind from R+'s literal
"two consecutive rewards" case (where reward and correctness coincide on
both modalities).

### Response window and dead zone
- Evoked window: **`[5ms, 50ms]`** post `start_time` (reused from the
  `ssl-reward-history-modulation` project's precedent).
- Whisker-trial dead zone: user specified **`-10ms` to `+5ms`** relative to
  `start_time` for this project — the wider, more conservative of the two
  documented options in `ssl_artifact_dead_zone.md` (`-1ms/+4ms` is the
  correctness minimum; `-10ms/+5ms` is explicitly still valid, just more
  conservative). Since the evoked window starts exactly at `+5ms`, it does
  not overlap this dead zone regardless of which of the two documented
  widths is used — no window-clipping logic is needed.

### Minimum trial floor
A unit's index for a given (cohort, RHMI/EHMI) combination is NaN if either
`t-1` bucket (rewarded / non-rewarded) has fewer than **5** qualifying
trials — same floor as `ssl-reward-history-modulation` and
`ssl-bwm-style-single-cell-decoding`.

## Example computation (2026-08-21)
One R+ and one R- example session, chosen from the 95 whisker-training
ephys sessions (`ssl_bwm_trial_prep.list_whisker_training_ephys_sessions`),
learning stage, high unit count:
- R+: `MH030_20250501_151231` (subject MH030, 407 trials, 3800 units)
- R-: `MH023_20250316_110814` (subject MH023, 566 trials, 3773 units)

Dataset: `ssl_ephys` (KS4) 1.0.0, per `ssl_loading_policy.md`'s default.
QC: not yet locked for this project — example run used **`good`** units
only (narrower QC than the good+mua default used elsewhere in
`ssl-bwm-style-single-cell-decoding`); revisit before any full-cohort run.

## Full-cohort run (2026-08-21)
QC widened to **good+non-soma** `bc_label` (user request; note this is
*narrower* than the good+mua scope used elsewhere in
`ssl-bwm-style-single-cell-decoding` -- mua is not included here). Ran over
all 95 whisker-training ephys sessions (`004_compute_full_cohort.py`,
checkpointing + regenerating stats/figures every 10 newly-completed mice,
per user request). Final: **56,275 units, 95 sessions, 80 mice (46 R+, 34
R-)**.

**Process incident**: when the plotting bugs below were found and fixed
mid-run, the job was "restarted" without stopping the original background
process first -- two copies of `004_compute_full_cohort.py` ran concurrently
for ~50 minutes, both checkpointing to the same output files. No data was
lost or corrupted (both computed identical deterministic content), but this
wasted compute and could have left a stale partial file on disk if the
older process's final write had landed after a checkpoint was reported to
the user. Resolved by stopping both, verifying the on-disk parquet matched
the expected full-cohort row/session/mouse counts, then doing one clean
single-process rerun of `005_stats.py`/`006_plot_publication_figure.py`
before reporting final numbers. **Lesson: always stop the old background
task explicitly before launching a replacement, never rely on only
stopping the Monitor watching it.**

**Plotting bugs found and fixed** during the first checkpoint (10 mice),
before most of the run: (1) title and stats-subtitle text overlapping
(manual `ax.text` y-position too close to `ax.set_title`'s pad) -- fixed by
retuning the offsets; (2) adjacent areas' paired boxes overlapping into each
other's rows in the by-area figure (box width + cohort gap exceeded the
row-to-row spacing: `2*gap + width > spacing`) -- fixed by widening spacing
and shrinking gap/width to satisfy that inequality with margin.

### Results
BH-FDR not applied here (a single overall cohort comparison per metric, not
a multi-region test family).

| Metric | Mouse-level Mann-Whitney | Mouse-level Welch's t | Neuron-level LMM (R- vs R+ coef) |
|---|---|---|---|
| RHMI | p=0.0024 (n=46 R+/34 R-) | p=0.0041 | coef=+0.112, p=0.0031 (n=48,526 units/80 mice) |
| EHMI | p=0.0032 (n=46 R+/24 R-) | p=0.0162 | coef=+0.003, p=0.956 (n=35,755 units/70 mice) |

**Important caveat, not yet resolved**: for **RHMI**, the mouse-level test
(each mouse weighted equally) shows R+ > R- (mean_rplus=0.073,
mean_rminus=-0.072), while the neuron-level LMM's fixed-effect coefficient
for R- vs R+ is *positive* (+0.112) -- **opposite direction**. For **EHMI**,
mouse-level is significant (R- more suppressed than R+) but the neuron-level
LMM is null (p=0.96). Both point to the same likely cause: unit count per
mouse varies enormously (tens to >1000), so the unweighted neuron-level LMM
fixed effect is dominated by whichever mice happen to contribute the most
units, unlike the mouse-level test where every mouse counts once. **Do not
report either metric as a confirmed cohort effect without resolving this
disagreement** -- candidates: add a random slope, weight/subsample units per
mouse, or treat the mouse-level test as primary and the LMM as a sensitivity
check only. Not yet decided.

## Day 0 / learners-only subset (2026-08-22)
Filtered the full-cohort table to `day_stage=='learning'` and learner mice
only (`learning_category in {good,moderate}` & `exclude_ephys==0`, per
established SSL practice) -- `007_filter_day0_learners.py`. 11 mice dropped:
AB104, AB107, AB117, AB141, AB142, AB151, MH007, MH009, MH013, MH020,
AB026. Result: **43,179 units, 68 sessions/mice (35 R+, 33 R-)**.

| Metric | Mouse-level Mann-Whitney | Mouse-level Welch's t | Neuron-level LMM (R- vs R+ coef) |
|---|---|---|---|
| RHMI | p<0.001 | p<0.001 | coef=+0.097, p=0.017 (n=37,214 units/68 mice) |
| EHMI | p=0.007 | p=0.020 | coef=+0.001, p=0.992 (n=28,309 units/59 mice) |

Same pattern as the full cohort: RHMI's mouse-level tests and neuron-level
LMM are both nominally significant but **disagree in direction**
(mouse-level: R+ > R-; LMM coefficient: R- > R+); EHMI is mouse-level
significant but LMM-null. Restricting to day 0 + learners did not resolve
this -- it is not an artifact of pooling expert-stage sessions or bad
learners into the full-cohort run. The per-mouse-unit-count-imbalance
hypothesis (see Full-cohort run section above) remains the leading
candidate explanation, still not tested.

## Proportion and distribution tests (2026-08-22, day 0 / learners only)
User asked how to test for a cohort difference in **proportion** or
**distribution**, distinct from the location-shift tests above (mouse-level
median, neuron-level LMM mean). Three tests added to `history_lib.py`,
run via `010_distribution_proportion_tests.py` on the day0/learners subset:

1. **Mouse-level proportion**: per-mouse fraction of units with index>0
   ("enhanced" vs "suppressed"), Mann-Whitney + Welch between cohorts --
   answers "does the balance of enhanced- vs suppressed-modulation units
   differ by cohort" (mouse-respecting by construction, one number/mouse).
2. **Neuron-level KS test**: standard two-sample Kolmogorov-Smirnov on
   pooled units -- tests for ANY distributional difference (location,
   spread, shape), not just a mean/median shift. Pseudoreplication-
   caveated like the neuron-level LMM (pools units across mice with very
   unequal per-mouse counts).
3. **Mouse-block permutation KS test**: the rigorous, mouse-respecting
   version of #2 -- null built by permuting the R+/R- label across whole
   mice (not units) and recomputing the pooled-unit KS statistic each
   time, 2000 permutations. Same mouse-block-permutation logic used to fix
   the pseudoreplication bug in `ssl-reward-history-modulation`'s
   PERMANOVA; this is the test to cite as the primary distributional
   result, not #2.

### Results
| Metric | Mouse-level proportion (frac>0) | Neuron-level KS | Mouse-block permutation KS |
|---|---|---|---|
| RHMI | Mann-Whitney p<0.001, Welch p<0.001 (R+ mean frac 0.54 vs R- 0.45) | KS=0.088, p=2.1e-62 | KS=0.088, p<0.001 (0/2000 null draws >= observed) |
| EHMI | Mann-Whitney p=0.065, Welch p=0.092 (marginal, R+ 0.48 vs R- 0.43) | KS=0.115, p=9.2e-81 | KS=0.115, p<0.001 (0/2000 null draws >= observed) |

**Notable finding**: for EHMI, the location-shift tests disagreed
(mouse-level significant, neuron-level LMM null -- see Day 0/learners
section above) and now the proportion test is only marginal too, **but the
mouse-block permutation KS test is strongly significant for both metrics**,
robust to the mouse-non-independence correction. This suggests the R+/R-
difference, especially for EHMI, is more a difference in distribution
shape/spread than a simple location shift -- consistent with the wider
R- boxes visible in several areas of the by-area figures. Plots:
`011_ecdf_by_cohort.png` (ECDF overlay, KS gap marked),
`011_proportion_by_cohort.png` (per-mouse fraction-positive box+jitter).

## Not yet decided / open
- Whether to scale to the full 95-session cohort, and at what QC scope
  (`good` vs `good+mua`).
- Whether to add the supplementary shared no-stim-FA index or
  auditory-only index discussed with the user but not requested yet.
- Statistical testing on top of the raw index (e.g. cohort comparison,
  area enrichment) — not requested yet, this project so far only computes
  the per-unit point estimate.
