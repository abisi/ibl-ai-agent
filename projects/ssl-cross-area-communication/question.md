# Question

## Original request (Axel Bisi, 2026-08-31)
"New analysis: perform a preliminary analysis, comparing the two cohorts, of
the task-aligned cross-area communication for different trial types. Use
broader brain structures first, then area_custom_acronym for the levels of
population of neurons communicating. Set at least 20 units to consider an
area recording. First look at the interaction dataset in terms of number of
recording pairs available. Ignore passive trials, we want to look at active
data here. use pCCA, partial CCA, for the measure of interaction.
Cross-validate pCCA with PCA before. Check influence of number of units in
correlation obtained. Plot correlation rank across dimensions and between
cohorts. Project PSTHs onto canonical axes to see correlation across time.
Compute cross/lagged pCCA to see which area leads or lag at the ms level."

Follow-up clarification (Axel Bisi, 2026-08-31): "The CCA is computed on
noise correlation i.e population single-trial activity after removing the
mean PSTHs for each neuron. Do not baseline correct. Take into account the
usual dead zone, that must be excluded from the analysis. Then, three
alternatives to do A: run CCA without a third variable, B: partialize out
the activity of all other neurons present in the recordings and C.
partialize out the mean variance of the piezo licks binned at 50 ms
resolution (illustrate this). Show all variants. Show rasters of input data
and example correlated trials."

## Refined explication
Distinct new project from `ssl-burstiness-cohort-comparison`. Preliminary,
exploratory R+/R- cohort comparison of inter-areal "communication" during
active whisker/auditory trials, measured as canonical correlation (CCA)
between two simultaneously-recorded area populations, computed on **noise
correlations**: each neuron's trial-averaged PSTH subtracted from its
single-trial, time-binned response (no separate baseline correction on top
of that), with the mandatory whisker-trial dead zone (-1/+4ms around
`start_time`, `ssl_artifact_dead_zone.md`) excised first.

Three partial-CCA variants, all reported side by side:
- **A**: plain CCA, no partialling.
- **B**: partial out the pooled activity of all other simultaneously-recorded
  units in that session (not in area A or B), reduced to its leading PCs.
- **C**: partial out a piezo-lick-rate regressor, licks binned at 50ms
  (own illustrative figure).

Two brain-area hierarchy levels (never mixed): Allen custom coarse groups
(`allen_utils.get_custom_area_groups_from_name()`, e.g. "Motor and frontal
areas", "Somatosensory areas", ...) computed first from `area_acronym_custom`
(`allen_utils.process_allen_labels(split_merge_areas=True)`). An
"area recording" (either level) requires **>=20 units** in that session.

Active trials only (`context=='active'`); `whisker_trial` + `auditory_trial`
(passive and `no_stim_trial` excluded by default — state this).

## Definitions locked so far
- **Area-pair / valid recording**: within one session, two areas (same
  hierarchy level) each with >=20 qualifying units, recorded simultaneously.
- **Noise correlation residual**: per unit, per trial type, per session:
  `single_trial_rate(t) - mean_over_trials_of_that_type(rate(t))`, on
  peri-`start_time` time-binned rates (window -200/+500ms, 10ms bins —
  tunable, to be confirmed at the intermediate checkpoint), dead zone
  excised first.
- **CCA/pCCA sample**: one (trial, time-bin) residual population vector per
  area, pooled across trials and time bins within a
  session/trial-type/area-pair.
- **Validation**: k-fold (5-fold) cross-validated canonical correlation
  (train/test split across trials) + a PCA-alignment baseline on the same
  split; unit-count subsampling check; scree (correlation-vs-dimension)
  plot; PSTH-on-canonical-axes projection; lagged pCCA (propose +/-100ms,
  10ms steps) for area lead/lag.
- **Cohort factor**: reference-sheet `reward_group` (`joint_mouse_reference_weight.xlsx`),
  mandatory mouse-inclusion filters, both population scopes (entire dataset,
  learners-only) — per `ssl_task_semantics.md`.

## Exploration / confirmation
Treated as fully exploratory/hypothesis-generating (same framing as
`ssl-burstiness-cohort-comparison` and `ssl-ks2-single-mouse-tca`), given
the number of free methodological parameters (bin width, window, lag range,
PC cutoff for variant B). Report results as exploratory, not confirmed
effects, with the free parameters stated explicitly wherever used.

## Plan approval
Plan approved by Axel Bisi 2026-08-31 (see `C:\Users\bisi\.claude\plans\cheerful-bouncing-cloud.md`
for the full approved plan text). Two hard checkpoints before scaling: (1)
the recording-pair coverage report itself, (2) an intermediate report on a
small (2-3 session) subset showing input rasters, the noise-correlation and
lick-regressor construction, and variant-A CV canonical correlation +
PCA baseline, before the full multi-variant/multi-pair/lag run.
