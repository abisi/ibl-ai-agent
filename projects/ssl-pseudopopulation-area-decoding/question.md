# SSL pseudo-population area-level decoding

## Original request (Axel Bisi, 2026-09-29)
"I want to start a new related project with pseudo-population whole-session area-level decoding." Apply the
hierarchical-bootstrap pseudo-population decoding of the reference method (pasted by the user, below) to this task
and dataset, on the learning day only. Targets chosen by the user: hit vs miss (stimulus-aligned), whisker vs
auditory (stimulus-aligned), whisker vs auditory (first-lick-aligned).

Reference method (as pasted):
- Hierarchical bootstrapping per brain area: recordings with < 10 neurons in the area rejected; sample 5 mice with
  replacement, 2 sessions per mouse with replacement; within each session sample fixed numbers of trials per trial
  type without replacement (sessions with fewer discarded) and 20 neurons with replacement -> 200-neuron
  pseudo-population; 100 bootstrap repetitions; mean +- SEM across repetitions.
- Population decoder per area and time point: spike rates in causal sliding windows of 200 ms, 10-ms steps; logistic
  regression; nested CV: outer 5-fold (test fold keeps the trial-type ratios), inner 4-fold to choose the
  regularisation; accuracy = mean over outer folds.
- Decoding onset: earliest time t such that at least 9 of the 10 following time bins are above chance.

## Question
Across brain areas (area groups) and cohorts (R+ vs R-), when and where can hit/miss and whisker/auditory identity
be decoded from pseudo-populations of equal size, built with equal trials per class and equal neurons per session,
so that area and cohort comparisons are not driven by different neuron or trial counts or by single mice?

## Adaptation to the SSL dataset (draft, 2026-09-29)
- Data: ssl_ephys (KS4), learning stage only (day 0: one session per mouse). Trial prep: scripts/ssl_bwm_trial_prep.py
  prep_session (context == active; warm-up block removed except the 1 trial before the first whisker trial; first
  whisker trial always kept; skills/ssl-analyze/references/ssl_auditory_warmup_block.md). Units: QC good + mua
  (quality_label), area_group labels (scripts/build_ssl_area_labels.py); also whole brain.
- Cohorts built separately (R+, R-); pooling across cohorts would mix the conditions being compared.
- Hierarchy: learning stage has one session per mouse, so "mouse -> session" collapses to sampling mice (= sessions)
  with replacement. To match the reference pseudo-population size: N_MICE = 10 mice per bootstrap, 20 neurons per
  session with replacement -> 200 neurons.
- Eligible session for (area, target): >= 10 QC units in the area and >= N_TRIALS trials of each class. Eligible
  (area, cohort): >= 5 distinct eligible mice.
- Pseudo-trials: within each sampled session, N_TRIALS trials per class drawn without replacement; pseudo-trial k of
  class c concatenates the k-th drawn class-c trial of every sampled session (trials are matched by class only; no
  trial-order alignment across sessions; cross-session noise correlations absent by construction).
- Time: causal 200-ms windows, 10-ms steps, labelled at the window end; stimulus-aligned -200..+600 ms, lick-aligned
  -600..+200 ms (corrected first lick, skills/ssl-lick-alignment). Whisker artefact dead zone (-10..+5 ms from
  stimulus onset) excised per trial from any window overlapping it.
- Decoder: StandardScaler + L2 logistic regression, nested CV (outer 5-fold stratified by class, inner 4-fold choosing
  C from the pipeline grid), balanced accuracy; 100 bootstrap repetitions; mean +- SEM across repetitions.
- Chance / significance: label-shuffled pseudo-populations (class labels permuted within each session before pooling),
  same pipeline, to give a null distribution per time bin.
- Statistics: cohort difference per bin and window from the bootstrap distributions; onset latency per the
  reference rule; areas compared at equal N.

## Parameters decided by the user (2026-09-29 review; supersede the draft above where they differ)
| # | Parameter | Decision |
|---|---|---|
| 1 | Stage | Learning day only; one session per mouse (mouse = session, no separate session sampling) |
| 2 | Cohorts | R+ and R- decoded separately (two pseudo-populations) |
| 3 | Areas | area_group and whole_brain |
| 4 | Session eligibility per area | >= 5 QC units (good + mua) in the area; any session with a passing area contributes to that area |
| 5 | Mice per bootstrap | 10, sampled with replacement from the eligible mice of the cohort |
| 6 | Neurons per session | 5, with replacement (-> 50-neuron pseudo-population, equal for every area and cohort) |
| 7 | Trials | >= 3 per class per session (3 hits and 3 misses; 3 whisker and 3 auditory) -- see open issue A |
| 8 | Targets | hit vs miss (stim-aligned); whisker vs auditory stimulus (stim-aligned, all trials, no auditory-choice balancing); whisker vs auditory lick (first-lick-aligned, licked trials) |
| 9 | Repetitions | 100 to start |
| 10 | Windows | as the per-session pipeline: causal 50-ms bins, 5-ms steps, labelled at the bin end; stim-aligned -200..+600 ms, lick-aligned -600..+200 ms |
| 11 | Whisker artefact | mandatory dead-zone excision (-10..+5 ms from stimulus onset, per trial) before any computation |
| 12 | Lick time | corrected first lick: start_time + (lick_time - response_window_start_time) |
| 13 | Features | spike rates z-scored inside each training fold |
| 14 | Warm-up | auditory warm-up block excluded except the 1 trial before the first whisker trial (prep_session rule) |
| 15 | Chance | 0.5 for whisker vs auditory stimulus (stim-aligned); linear-shift permutation null otherwise (hit/miss, lick-aligned modality): labels linearly shifted within each session before pooling |
| 16 | Above chance in a bin | bootstrap accuracy above the 95th percentile of the linear-shift null (hit/miss, lick-aligned); for stim-aligned modality against 0.5 -- see open issue C |
| 17 | Onset latency | earliest t with the next 5 bins above chance (reference used 9 of 10) -- see open issue D |
| 18 | R+ vs R- | permutation of cohort labels at the mouse level (pseudo-populations rebuilt from permuted mouse sets) |
| 19 | Area comparability | every area sampled identically: 10 mice, 5 neurons per session, same trials per class |
| 20 | Neuron-dropping curve | areas with > 500 neurons in total (per cohort): pseudo-populations of 500, 450, ..., 50 neurons (10 mice x 50, 45, ..., 5 neurons per session) |

## Decisions on the open issues (user, 2026-09-29; supersede rows 7, 15-18 above where they differ)
- A. Option (i), 3-fold CV. Folds are made per session and per class on the REAL trials (each session's trials of
  each class split into 3 folds, so every fold holds out >= 1 trial of every class from every sampled session);
  a split in which any fold lacks a class is redrawn. Training pseudo-trials are built only from training-fold trials,
  test pseudo-trials only from held-out trials (sampled with replacement within the fold), T = 20 pseudo-trials per
  class for training and for testing. Regularisation chosen per bin by an inner 2-fold split of the training trials
  (same per-session, per-class construction; training sets then have >= 1 trial per class per session).
- B. Linear-shift null: 100 shifted repetitions to start (labels linearly shifted within each session by 10-50% of
  its trials, random direction, before pooling; shifts leaving < 3 trials of a class redrawn).
- C. Above chance in a bin = bootstrap mean accuracy above the 95th percentile of the linear-shift null, for ALL
  targets (including stim-aligned whisker vs auditory; 0.5 is not used as chance).
- D. Onset latency = earliest bin t such that at least 4 of the next 5 bins are above chance.
- E. R+ vs R- (mouse-level cohort-label permutation, 200 permutations x 20 repetitions) deferred; first debug on the
  R+ cohort only.

## Null and inference design (user, 2026-09-29, after pilot 1; supersedes B and C above)
Pilot 1 (hitmiss R+, 3 areas, 100 real + 100 independent shift repetitions) compared the real mean curve with the
95th percentile of single shift repetitions -- valid but not comparable (resampling noise averaged out in the real
mean, not in single null repetitions). New design (user):
- Per iteration i: one draw (10 mice with replacement, 5 units per session); decode the real labels -> real_i; decode
  the SAME draw N_SHIFTS = 10 times with a new random linear shift per session each time -> null_i = mean of the 10;
  d_i = real_i - null_i.
- A bin is above chance when the 5th percentile of d_i across iterations is > 0 (one-sided bootstrap); onset = first
  bin with >= 4 of the next 5 bins above chance; windows: mean d, 5th percentile, p = fraction of iterations with d <= 0.
- Uncertainty = spread across iterations (2.5-97.5 percentiles), not SEM.
- Test pseudo-trials raised to 100 per class (training stays 20) to reduce per-iteration counting noise.
- 100 iterations and N_SHIFTS = 10 are PILOT values; the user may increase them (remind before final runs).

## After pilot 2 (user, 2026-09-29)
- Neurons per session: 20 (was 5) -> 200-neuron pseudo-populations (10 mice x 20, with replacement). Reason: with 5 the
  real accuracy varied from ~0.5 to ~0.9 across iterations (which units were drawn), so the 5th percentile of d cleared
  0 only late (onsets 110-265 ms) although mean curves left the null within ~20-50 ms.
- Pseudo-trials are resampled with replacement from the real trials of the relevant fold, so their number is not
  limited by the number of real trials (100 pseudo-trials per class are possible with 3 real trials per class; each
  pseudo-trial combines independent picks from 10 sessions). More pseudo-trials stabilise the decoder/score but add no
  information beyond the real trials.
- Training pseudo-trials per class: 100 (user: "go directly to 20 neurons per session and 100 trials"; the 004
  sensitivity pilot was stopped). Defaults in 002: 20 neurons/session, T_train 100, T_test 100.

## Speed-ups (user, 2026-09-29; 002 v2)
- C fixed across fits strictly: per-bin, per-fold C chosen by the inner CV of the real decode; the 10 shifted decodes
  of the same iteration use exactly those C values (no inner CV of their own).
- Vectorised fitting: one batched L2 logistic regression (Newton) for all time bins, warm-started along the C grid;
  intercept unpenalised (sklearn lbfgs semantics; v1 used liblinear, which also penalises the intercept).
- Work shared across decoding targets: one job per cohort x area; spikes loaded once; stim-aligned rates computed once
  per unit on whisker + auditory trials and sliced for hit/miss and whisker-vs-auditory; lick-aligned rates cached
  separately.

## After pilot 3 (user, 2026-09-29)
- 20 mice per draw (was 10), 10 neurons per session (per area; user changed from 20 before the full run) -> 200-neuron pseudo-populations; mice and units drawn
  with replacement.
- Trials: option (c) "balanced reuse" -- per session, class and fold, trials drawn without replacement until exhausted,
  then through a fresh random permutation (each real trial used equally often); minimum stays 3 per class.
- Figures: mean +- SD across iterations (a percentile-band variant, 2.5-97.5% accuracy / 5-95% difference, is
  available with SSL_PSEUDO_BANDS=pct). Statistics unchanged (5th percentile of d > 0; onset 4 of 5 bins).
- Onset refinement (user, 2026-09-29): the onset bin must itself be above chance (and >= 4 of the 5 bins starting at it).
  For stimulus-aligned targets (hitmiss, modality_stim) the onset search starts after the stimulus (bin end > 0).
- Time resolution unchanged (50-ms bins, 5-ms steps).

## Open issues after the review (2026-09-29; resolved above)
A. Trials per class vs nested CV: pseudo-trial k of a class concatenates the k-th sampled trial of that class from each
   session, so 3 trials per class per session gives only 3 pseudo-trials per class -- too few for outer 5-fold / inner
   4-fold CV. Options: (i) fixed T pseudo-trials per class (e.g. 20) built by resampling trials WITH replacement, with
   the train/test split made first at the level of each session's real trials (held-out pseudo-trials built only from
   held-out trials, so no trial is both trained on and tested); folds limited by the 3-trial minimum (e.g. 3-fold outer,
   leave-one-trial-per-class-out per session); (ii) raise the minimum (10 per class keeps 48 R+ / 33 R- sessions for
   hit/miss).
B. Shift null for pseudo-populations: number of shifted pseudo-population repetitions (cost scales like the real run).
C. Above chance for the stim-aligned modality target (chance 0.5): e.g. 5th percentile of the bootstrap above 0.5.
D. Onset rule: "the next 5 bins" -- all 5 of the next 5 bins above chance (with 5-ms steps = 25 ms), or the reference's
   majority form (e.g. 4 of 5)?
E. Cohort permutation cost: each permutation rebuilds and decodes pseudo-populations; e.g. 200 permutations x fewer
   repetitions each, per area and target.

## Open design choices (draft, superseded by the table above)
1. N_TRIALS per class per session (hit/miss): 10 keeps 48 R+ / 33 R- sessions, 15 keeps 45 / 28, 20 keeps 38 / 25.
2. Whisker vs auditory, stimulus-aligned: the reference balances stimulus x choice; here auditory no-lick trials are
   essentially absent (median 2 per session), so lick cannot be balanced within auditory trials. Options: (a) all
   whisker vs all auditory (lick confound: auditory ~all licked, whisker partly); (b) licked whisker vs licked
   auditory trials only (removes the lick confound, stimulus-aligned counterpart of the lick-aligned target).
3. "Above chance" for the onset rule: e.g. bootstrap mean above the 95th percentile of the shuffle null in that bin.

## Definitions
- Hit / miss: lick / no lick on a whisker trial (cohort-independent; skills/ssl-analyze task semantics).
- Learning stage: day-0 whisker-training session (one per mouse).
- Area group: custom Allen area grouping (allen_utils), same exclusions as the per-session pipeline where relevant.

## Exploration / confirmation sets
Exploratory only for now (all learning-stage sessions).
