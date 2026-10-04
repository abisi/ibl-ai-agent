# Question

## Original question (user, 2026-09-10)
Single-trial decoding of hit/miss for whisker trials using area populations, across time bins, comparing decoding
ability across cohorts; good and MUA units; learning and expert sessions; first and second half of the session.

## Current question (2026-10-05, after the 2026-10-04 cleanup)
How much information does whole-brain and area-level population activity carry about the upcoming choice after a
whisker stimulus, and about the stimulus modality before the lick, during single-session whisker learning; when does it
appear; does it differ between the reward cohorts (R+: licking after the whisker stimulus is rewarded; R-: not
rewarded); and does it change within the learning session, at the time the behaviour changes?

Three parts:
1. **Session-wide decoding** (time-resolved, whole session): hit vs miss, whisker vs auditory at stimulus onset,
   whisker vs auditory before the first lick, and performance state; whole brain and `area_group`; time courses,
   window values, onset latencies, R+ vs R-, learning vs expert, relation to behavioural d'.
2. **Within-session change**: the same decodings split within the learning session -- session halves (separate,
   count-matched or single decoders; pseudo-populations), behavioural learning-trial (LT) splits, placebo splits at
   every whisker trial, cohort-label permutation, single-trial decoder margins vs the learning curve.
3. **Stimulus-onset geometry across epochs**: whisker vs auditory responses 5-35 ms after onset in passive pre, active
   and passive post; the whisker axis and the whisker-/auditory-evoked patterns relative to the active lick axis.

Out of scope (archived 2026-10-04, `_archive/`): performance-state decoding as its own analysis family (the
`perfstate_stim` tag of the 024 sweep stays in part 1), the first-generation LT split and the coupling model,
behaviour-only analyses (in ssl-task-performance / ssl-learning-trial-identification), `area_acronym_custom` (dropped).
The performance-quantity regression (PLS) is its own project: `ssl-whisker-perfquant-pls-regression`.

## Hypotheses
- H1: choice information after the whisker stimulus grows when R+ mice learn.
- H2: in R-, where licking to the whisker is not reinforced, it does not grow or declines.
- H3: the early sensory representation of the whisker stimulus is re-mapped relative to the motor (lick)
  representation depending on the reward contingency.

## Definitions
- **Hit vs miss**: whisker trials, label = `lick_flag` (lick in the response window), the same definition in both
  cohorts (decision 2026-09-10; a cohort-corrected "rewarded" label would flip between cohorts). It is therefore a
  lick vs no-lick decoding on whisker trials; the overt action differs in meaning between cohorts.
- **Modality (stimulus)**: whisker vs auditory trials, stimulus-aligned, any outcome.
- **Modality (pre-lick)**: licked whisker vs licked auditory trials, aligned to the corrected first lick
  (`start_time + lick_time - response_window_start_time`, skill `ssl-lick-alignment`).
- **Performance state**: high vs low hit-rate state of 5-whisker-trial blocks (part 1 only).
- **Windows**: baseline -200 to -10 ms; sensory 5-50 ms (and 5-100 ms for the split analyses); pre-lick -100 to 0 ms
  (changed from -150 ms on 2026-10-02); post-lick 5-200 ms; geometry 5-35 ms. Whisker artefact dead zone -10 to +5 ms
  from stimulus onset, excised.
- **Learning trial (LT)**: behavioural change point of the session (definitions in ssl-learning-trial-identification:
  L0 stored, L5 whisker-only Bayesian change point, L6 joint whisker/FA, L6x joint with lapse -- relaxed: log10 BF > 0.3
  and P(w > FA) > 0.9 over a 20-trial window -- and others). Not locked; L6x relaxed is the candidate.
- **Placebo split**: any other whisker trial of the same session used as the split, >= 10 whisker trials from the real
  one; excess = real change minus the mean placebo change.
- **Accuracy above null**: balanced accuracy minus the session's linear-shift null (labels shifted 10-50 % of the
  trials, non-wrapping), or minus the trial-shuffle null where stated.

## Data and inclusion
- SSL dataset, Kilosort 4 (`ssl_ephys` 1.0.0). Units: quality labels good + mua (tracked good units, >= 0.5 Hz in every
  epoch, for part 3). Areas: whole brain and `area_group` (allen_utils custom groups).
- Mice: `exclude == 0`, `exclude_ephys == 0`; cohort per mouse from `joint_mouse_reference_weight.xlsx`; R+proba dropped.
  Population scopes: entire dataset and learners only (`learning_category` good / moderate), R+ / R- separately.
- Stages: learning (whisker day 0) and expert (day > 0), never pooled; parts 2-3 are learning stage only.
- Trials (skill `ssl-trial-exclusion`): active context (per-trial ITI rule since 2026-10-04), `perf == 6` excluded
  except passive trials (since 2026-10-01), auditory warm-up block cut keeping one trial before the first whisker trial,
  end-of-session disengagement trimmed with rule A1 for neural analyses.

## Decoding
- StandardScaler + L2 logistic regression; one C per session (and area) by pooled CV on a wide window; stratified
  K-fold, K = min(5, minority class); balanced accuracy.
- Part 1: causal 50-ms bins labelled at their end, 5-ms stride (024 sweep); cluster-mass test against a group
  mouse-block sign-flip null; R+ vs R- two-sided cluster test with mouse-level cohort-label permutation; onset latency =
  first bin at 50 % of the group-mean peak, 95 % CI from mouse bootstrap.
- Part 2: window decoders with size matching (each epoch subsampled to the smaller epoch's class counts).
- Pseudo-populations: 20 sessions x 10 units per iteration, pseudo-trials within session halves, trial-shuffle null.

## Statistics
- Unit of analysis: session at the learning stage (one per mouse); mouse-block permutation for any mouse-level factor
  when a mouse contributes several sessions (expert stage).
- Non-parametric and parametric tests always reported together (Wilcoxon and t; Mann-Whitney and Welch; Friedman and
  RM-ANOVA). No multiple-comparison correction, except BH-FDR across areas in area post-hoc tests.

## Exploration / confirmation
Everything so far is exploratory. Parameters still to lock before any confirmatory test: the LT definition, the
geometry window and unit set (135b), pseudo-population repetitions (pilot values: 100 iterations).

## Known stale results
The 024 sweep (part 1), 117 (count-matched halves) and 123b (margins vs LT) were computed before the `perf == 6` rule
(13 sessions affected); expert-stage 024 tags partly lack the 2026-09-28 first-whisker-trial redo; expert sessions lack
MH062_20260113's task block (context rule, 2026-10-04). Reruns deferred (user, 2026-10-04).
