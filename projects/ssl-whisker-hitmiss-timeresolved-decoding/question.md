# Question

## Original question (from user, 2026-09-10)
Single-trial decoding of hit/miss for whisker trials using area populations,
across time bins (10ms), comparing decoding ability across cohorts. Use both
good and MUA units, both learning and expert sessions, and both first-half
and second-half of session trials.

## Relationship to `ssl-bwm-style-single-cell-decoding`
This is a **new, separate project**, not folded into the sibling
`ssl-bwm-style-single-cell-decoding` project, because the analysis shape is
different (time-resolved *population* decoding across a bin grid vs. their
fixed-window per-neuron tests / session-level single-window decoding).
It deliberately **reuses** that project's already-locked infrastructure and
conventions rather than re-deriving them:
- `scripts/ssl_bwm_trial_prep.py`: `list_whisker_training_ephys_sessions`,
  `cohort_corrected_rewarded`, `prep_session` (active-epoch trials with a
  cohort-corrected `rewarded` column).
- `scripts/ssl_bwm_windows.py`: `clip_window_for_whisker` (mandatory
  `-1ms/+4ms` whisker dead-zone exclusion), `unit_rates_for_trials`.
- `scripts/ssl_bwm_decoding.py`: `population_design_matrix` pattern
  (per-unit `unit_rates_for_trials` calls against a spike shard),
  L1-logistic-regression + `StratifiedKFold` machinery.
- `reports/ssl_analysis/derived/unit_area_labels.parquet` (built by
  `ssl-whisker-auditory-cohort-modulation`): `area_acronym_custom` (fine) and
  `area_group` (coarse) per unit.
- `reports/ssl_analysis/derived/mouse_reference.parquet`: `reward_group`,
  `exclude`, `exclude_ephys`, `learning_category`.
- QC scope: `bc_label in {'good','mua'}` — reusing the sibling project's
  2026-08-19 decision to widen from good-only.
- Their "next round" backlog (question.md) already flagged a hit/miss test
  with this framing note, carried forward here: cohort-corrected "hit" is a
  **different overt action** across cohorts (R+ hit = lick, R- hit =
  withhold-lick), so a cohort difference in decodability may reflect
  motor/strategy differences, not only a learning/coding difference — state
  this caveat in any report.

**Gap found and fixed here, not yet fixed upstream**: neither
`list_whisker_training_ephys_sessions` nor `load_reward_group` in the
sibling project's `ssl_bwm_trial_prep.py` applies the mandatory
`exclude==0`/`exclude_ephys==0` mouse-inclusion filters
(`ssl_task_semantics.md`). Checked directly 2026-09-10: applying those
filters on top of their existing 119 has-ephys whisker-training sessions
narrows the reward-group-labeled set to **106 usable mice / a subset of
those 119 sessions** (65 R+, 41 R- mice; by day-stage: learning 51 R+/38 R-,
expert 15 R+/14 R-). This project applies the filter itself
(`hitmiss_session_list` in the new library, see below); the sibling
project's own session counts were not corrected as part of this work — flag
if reproducing their exact session count matters.

## Target
Cohort-corrected `rewarded` (hit) vs. non-hit (miss), **whisker trials
only**, `context=='active'`. Computed via
`ssl_bwm_trial_prep.prep_session`, filtered to `trial_type=='whisker_trial'`.

## Locked design decisions (via AskUserQuestion, 2026-09-10)
- **Bins**: **10ms, disjoint, no smoothing** (user's explicit choice, overriding
  the originally-asked 5ms — reasoning given: avoid dead-zone contamination).
  Bin grid is offset by 5ms from `start_time` (edges at ..., -15, -5, 5, 15,
  ... ms) so the mandatory `-1ms/+4ms` whisker artifact dead zone
  (`ssl_artifact_dead_zone.md`) falls **entirely inside one bin**
  (`[-5ms, 5ms)`), which is dropped from the decoding curve entirely (a
  clean gap, consistent with existing PSTH precedent) rather than clipping
  partial-duration bins into the time series.
- **Per-bin significance**: label-shuffle null + **cluster-based permutation
  across time**, not a full BWM-style imposter-session null at every bin
  (recommended for tractability; user confirmed). The established
  imposter-session null (`ssl_bwm_decoding.py:build_imposter_target`) is
  reserved for a **single summary-window confirmatory check** per
  (session, area, day_stage, half) condition, not run per bin.
- **Rollout**: small-scale exploratory pass first (1-2 sessions, both
  parcellation schemes, entire-dataset population scope) before any full
  sweep — per `AGENTS.md` Start Small. Real per-bin/per-session timing from
  this pass determines whether/how the full sweep is scoped.
- **Area parcellation**: **both** `area_group` (coarse) and
  `area_acronym_custom` (fine) from the start (user's choice — matches the
  sibling project's "always report both schemes" habit, at higher cost than
  the coarse-only recommendation).

## Comparison axes (all required, per the original question)
Run separately (not pooled) across:
1. **Day-stage**: `learning` (day==0) vs `expert` (day>0) — per
   `ssl_task_semantics.md`, never pooled.
2. **Cohort**: R+ vs R- (`mouse_reference.parquet`), `R+proba` dropped.
3. **Session-half**: first vs second half of active-epoch whisker trials,
   split at the median trial index within each (session, day_stage) —
   computed after `prep_session`'s chronological sort. This is a new split,
   not present in the sibling project.
4. **QC**: good+mua combined (single scope, not split further).
5. **Population scope**: entire-dataset and learners-only
   (`learning_category in {'good','moderate'}`), both, R+/R- split within
   each — per `ssl_task_semantics.md`'s Two population scopes rule.

## Data-sufficiency gates (new for this project)
- Min **5 units/area** (reused precedent from the sibling project).
- Min **8 trials/class** (hit and miss separately) **after** the
  session-half split — new gate, because halving an already-imbalanced
  hit/miss split (expert-stage hit rates run near ceiling in this task,
  parallel to the `auditory_miss`-rarity caution in
  `ssl-analyze/SKILL.md`) can produce degenerate folds. A
  (session, area, day_stage, half, scope) cell failing this gate is skipped
  and reported as insufficient data, not silently dropped from the N.

## Per-bin decoding procedure
- Fixed-C L1-logistic regression: the sibling project's nested-CV `C`
  selection is run **once per (session, area, day_stage, half, scope)** on a
  wide summary window (not per bin), then that fixed `C` is reused across
  every bin's plain `StratifiedKFold` CV — this is what makes the ~150-bin
  sweep tractable; only the one-time `C` selection pays the nested-CV cost.
- Balanced accuracy per bin, averaged over CV repeats, forming an
  accuracy-vs-time curve per (session, area, day_stage, half, scope, cohort).
- Cross-cohort comparison: reduce each curve to summary metrics (peak
  balanced accuracy, mean accuracy in a defined post-stimulus window) at
  `session_id` grain, then run the mandatory R+/R- test pair (Mann-Whitney U
  + Welch's t, `ssl_rplus_rminus_test_pair`/`ssl_cohort_comparison_stats.md`)
  **and** mouse-block permutation on the same statistic — `reward_group` is
  mouse-level even though the row grain here is already sessions (a
  distinction `ssl_stats_unit_of_analysis.md` draws explicitly: session-grain
  does not exempt a mouse-level factor from block permutation when a mouse
  contributes multiple expert sessions).

## Group-level significance pivot (locked 2026-09-10, replaces the per-bin
per-session null originally planned)
Smoke test (`exploratory-analyses/000_smoke_test.py`) found the per-session
label-shuffle null too expensive at the real scope (~1s/shuffle x hundreds
of shuffles needed for a non-floor p-value x ~thousands of
(session,area,half) cells). User-approved pivot: compute only the real
decode curve **plus exactly one label-permutation surrogate curve** per
session (2x a real-only decode, independent of shuffle count), then do all
significance testing at the **group level**:
- **Above-chance** (`group_mouseblock_permutation_null` +
  `cluster_permutation_test`): mouse-block sign-flip permutation --
  each mouse contributing sessions to a group independently picks
  real-vs-surrogate for *all* its sessions in that group, per permutation
  draw; cluster-mass test across time bins.
- **R+ vs R- summary metric** (`rplus_rminus_group_test`): session-grain
  Mann-Whitney + Welch pair (mandatory per `ssl_rplus_rminus_test_pair`)
  plus a mouse-block permutation of the group-mean difference in peak
  accuracy.
- **R+ vs R- per-bin curve difference** (`rplus_rminus_curve_difference`,
  added 2026-09-10 per user request "quantify for each time bin the
  difference of decoding accuracy"): per-bin R+ minus R- accuracy-difference
  curve, mouse-block-permutation null, `cluster_permutation_test(...,
  two_sided=True)` for cluster-corrected significant time windows in
  *either* direction. Uses only real curves already computed for the
  above-chance test -- no added per-session cost.
- `MIN_TRIALS_PER_CLASS` lowered to **5** (user decision, down from the
  originally-proposed 8).
Validated on synthetic data (planted effects correctly significant,
no-effect data correctly null, for all three test functions) and on real
spike data (`exploratory-analyses/001_group_level_demo.py`, 6 learning-stage
sessions, fixed area) before scaling up.

**Real-data timing note (2026-09-10, from the 001 demo)**: per-session cost
is higher than the smoke test suggested and scales with population size --
16-348s/session (both halves) for `area_group=='Motor and frontal areas'`
(one of the largest coarse-area buckets, 866-1410 units/session), mean
161.6s. Cost across the full sweep (many smaller/finer areas too) is not
yet known precisely -- see the learning-stage pilot below.

## Sharpened framing note (2026-09-10, from live pilot discussion): this is
functionally a lick-vs-no-lick decode, not an independent "reward" signal
`cohort_corrected_rewarded` (`ssl_bwm_trial_prep.py`) sets whisker-trial
`rewarded = lick_flag` for R+ and `rewarded = NOT lick_flag` for R-.
Balanced accuracy is identical for a target and its exact logical
complement, so **per cohort, decoding "hit vs miss" is mathematically
identical to decoding "lick vs no-lick."** This sharpens (not just repeats)
the framing caveat already in this file's Target section: the pilot's
real-data pattern so far (R+ decodes well above chance in Motor/frontal,
Somatosensory, Striatum/pallidum; R- stays near chance in the same areas,
24-25 sessions/cohort as of session 22) is therefore **not** "R+ encodes
reward outcome better than R-" -- both cohorts are decoding the same kind
of behavior (a lick) in the same regions, so a cleanly symmetric result
would have been the naive expectation if motor cortex just mechanically
tracks licking. The asymmetry itself is the interesting finding, with (at
least) two live explanations neither yet distinguished: (a) R- licks on
whisker trials are the rarer, more erratic action (impulsive/false-alarm-like
rather than the trained response) and therefore harder to decode, consistent
with the near-ceiling withhold behavior already observed (`AB094`: 33-35
hits vs 0-2 misses per half); or (b) a genuine cohort difference in
motor/frontal engagement. Not yet resolved -- flag prominently in any
write-up of this pilot's results, and revisit once group-level significance
results (not just point-estimate curves) are available.

## Decode target changed to raw lick_flag (user decision, 2026-09-10, mid-pilot)
Replaces the cohort-corrected `rewarded` target for the pilot sweep's decode
label: now `lick_flag` (1=licked, 0=no lick), **the same definition for
both cohorts** -- directly resolves the "sharpened framing note" issue
above by construction (no more per-cohort label flip, so a cohort
difference in decodability is now interpretable as a difference in how
well licking is neurally encoded, not a labeling artifact). The first
~24-session pilot run (cohort-corrected `rewarded` target) was stopped and
its outputs preserved under a `_cohortcorrected_hit` filename suffix, not
overwritten or discarded -- `002_pilot_results_partial_cohortcorrected_hit.parquet`,
`002_pilot_log_parallel_cohortcorrected_hit.txt`,
`002_bin_edges_cohortcorrected_hit.json`. The sweep restarted from scratch
under the new target (not resumable from the old run -- different label).

## Learning-stage pilot sweep (launched 2026-09-10)
User-approved next step: a full checkpointed/resumable background run over
**all learning-stage sessions** (both cohorts, both halves, both area
schemes, entire-dataset population scope only for this pilot) --
`exploratory-analyses/002_learning_stage_pilot_sweep.py` -- to get a real
full-scope ETA and produce real per-session/per-area/per-half decode curves
along the way. Learners-only population scope, expert-stage sessions, and
the full imposter-session confirmatory-check-per-condition are deferred
until this pilot's real timing is known.

## New analysis: lick_time-aligned modality decode (user request, 2026-09-10)
Original ask: repeat the analysis relative to `lick_time` instead of
`start_time` (-500ms/+200ms window). Blocked initially because `lick_time`
only exists for `lick_flag==1` trials -- no anchor exists for miss trials
under the `lick_flag` target. Resolved via AskUserQuestion: user redirected
the target itself rather than picking a miss-trial fallback --
**restrict to licked trials (lick_flag==1, both modalities) and decode
`trial_type` (whisker vs auditory)** instead of hit/miss. `start_time`-aligned
results (this file's earlier sections) are unchanged, not superseded.

**Known confound, checked before running** (30-session sample, learning
stage): whisker RT (`lick_time - start_time`) is **~120ms longer than
auditory RT on average**, paired per session (median whisker RT 0.414s vs
auditory 0.352s). Since the window is anchored to the lick, this means
`start_time` sits at a different relative position within the window for
the two modalities on average -- a modality decode near the lick could
partly reflect time-since-stimulus rather than pure modality content. Flag
prominently in any report of this analysis's results; not fixed or
controlled for, just disclosed. `mean_rt_whisker`/`mean_rt_auditory` are
saved per (session, area, half) row for follow-up checks (e.g. does the
decode-vs-RT-difference relationship hold within sessions).

**Dead-zone handling, new mechanism**: the mandatory whisker dead zone
(`-1ms/+4ms` around `start_time`) is fixed relative to `start_time`, not
the lick, so its position within a lick-aligned window shifts per trial
with that trial's own RT -- can't drop one global bin like the
`start_time`-aligned analysis does. Instead,
`event_aligned_rates_for_trials` masks (NaNs) any individual (trial, bin)
cell whose absolute window overlaps that trial's own absolute dead zone;
`decode_bin`/`select_fixed_c` now drop NaN rows before fitting (new
`_drop_nan_rows` helper) rather than assuming a clean grid the way the
`start_time`-aligned functions could.

**Pipeline**: `prep_lick_aligned_trials` (whisker+auditory active-epoch
trials, `lick_flag==1` only, `rt` column, chronological half-split),
`lick_aligned_bin_edges`/`lick_aligned_bin_population_matrices`/
`event_aligned_rates_for_trials` (new library functions,
`scripts/ssl_timeresolved_decoding.py`). Same group-level significance
machinery reused unchanged (target-agnostic). Smoke-tested on 2 sessions
(`007_lick_aligned_smoke_test.py`) before the full sweep
(`008_lick_aligned_sweep.py`, same parallel/checkpoint structure as
`002_learning_stage_pilot_sweep.py`) -- launched 2026-09-10.

## Classifier check + standardization added (user request, 2026-09-10)
User asked whether SVM/RBF would improve accuracy, given it "seems low."
Checked empirically on one real condition (`AB080`, Motor and frontal
areas, first half, wide summary window): linear SVM tied the existing
unscaled L1-logistic (0.965 vs 0.965 balanced accuracy, 10x5-fold CV);
RBF SVM was clearly worse (0.832) -- consistent with the high-dimension
(hundreds of units) / low-trial (tens) regime favoring linear methods,
same reasoning as the BWM paper's own L1-logistic choice. **Not** adopting
RBF or SVM.

**Found a real, separate issue while checking**: the pipeline never applied
feature standardization. L1's penalty is scale-sensitive, applied in raw
firing-rate units, so units with different baseline rates got an
inconsistent effective penalty. User confirmed this should be fixed.
`scripts/ssl_timeresolved_decoding.py`'s `_make_classifier` now wraps
`LogisticRegression` in a `Pipeline(StandardScaler(), ...)`, fit per-CV-fold
(never on held-out data, to avoid leakage) -- used in both `select_fixed_c`
and `decode_bin`.

**Both full 89-session sweeps were already run without this fix** --
user chose to re-run both rather than leave the unscaled results as final.
Prior (unscaled) outputs preserved with an `_unscaled` suffix, not
discarded: `002_pilot_results_partial_unscaled.parquet`,
`002_pilot_log_lickflag_unscaled.txt`, `002_bin_edges_unscaled.json`,
`008_lick_aligned_results_partial_unscaled.parquet`,
`008_lick_aligned_log_unscaled.txt`, `008_bin_edges_unscaled.json`. Both
sweeps re-launched sequentially (2026-09-10) with the corrected pipeline;
group-level tests and figures to be regenerated once they complete.

## Behavioral-state split (user request, 2026-09-11): performance-state
instead of session-half
User has an independent behavioral-state characterization: high-performance
blocks correlate with high discriminability for R+, but **reversed** for
R- (high-perf ~ low-disc, low-perf ~ high-disc). Requested a decode split
by performance state (5-trial blocks) instead of chronological half, plus
a discriminability-matched cross-cohort comparison given the reversal.

**Locked design** (via AskUserQuestion, 2026-09-11):
- **Block**: consecutive blocks of 5 whisker trials (`PERF_BLOCK_SIZE`),
  cohort-corrected hit rate per block, median-split into high/low per
  session (`prep_perfstate_trials`, `scripts/ssl_timeresolved_decoding.py`).
  Trailing trials that don't fill a full block are dropped.
- **Also computed**: `block_fa_rate` -- false-alarm rate (lick rate on
  `no_stim_trial` trials) within each block's own time span -- reported
  for validation, not used to define the state. Checked directly on 2
  real sessions before the sweep: AB080 (R+) shows higher FA in the
  high-perf state (0.064 vs 0.045); MH022 (R-) shows the **opposite**
  (0.183 high-perf vs 0.445 low-perf) -- the reversal is present in real
  data before any decoding is even run, consistent with the user's
  characterization.
- **Comparisons**: naive (R+ state=X vs R- state=X, same label) **and**
  discriminability-matched (R+ high vs R- low = "high-discriminability"
  pairing; R+ low vs R- high = "low-discriminability" pairing), reusing
  the existing generic curve-diff/above-chance machinery unchanged --
  matching is just a different selection of which records feed the R+/R-
  groups, not new statistical code.
- **Known data-sufficiency risk**: splitting by performance concentrates
  class imbalance within each state (e.g. AB080 low-perf state: 50
  trials, only ~12% hits ≈ 6 hit trials -- close to the 5-per-class gate).
  Expect more skipped (session, area, state) cells than the half-split
  design had.
- Pipeline: `015_perfstate_sweep.py` (mirrors `002_learning_stage_pilot_sweep.py`'s
  structure exactly, swapping `half` for `perf_state`), tests in
  `016_perfstate_group_level_tests.py`.

## Other statistical tests for contrasting decoding across time bins
(Answered for the user 2026-09-10, not yet built unless requested):
per-bin BH-FDR (simpler, less powered for temporally-extended effects than
cluster-mass); TFCE (threshold-free cluster enhancement, avoids picking an
arbitrary z-threshold); AUC-of-curve comparison (more outlier-robust
summary metric than peak accuracy); onset/latency permutation test
(when does decoding become reliable, not just whether it differs);
GAMM/mixed-effects with a time x cohort smooth term (parametric
curve-shape test in one model instead of per-bin/cluster testing).

## Exploration / confirmation split
Standard split, not skipped — small-scale pass (this file's Rollout
decision) reviewed with the user before any full-cohort run.

## Learning-trial split (added 2026-09-24)
User request: revisit hit/miss decoding with sessions split **pre vs post
each session's `learning_trial`** instead of first/second half, in the
sensory window, with generous imbalance handling.

- `learning_trial`: stored in `combined_results_ks4/<mouse>/whisker_0/
  learning_curve/<mouse>_whisker_0_whisker_trial_learning_curve_interp.h5`.
  0-based index into curve-aligned active whisker trials (from the first
  whisker trial, `perf!=6`). Produced by `behaviour_analysis/
  learning_utils.py` `identify_learning_trial_rewarded` (R+: start of first
  5-run with p_low>p_chance; 'expert' if the first 20 are above chance with
  mean p>=0.8) / `identify_learning_trial_nonrewarded(interp_flag=True)`
  (R-: LAST trial of the first above-chance 5-run followed by 5
  not-above trials; 'expert' if 20-run below chance with mean p<=0.2);
  only trials >= first whisker hit; values <=10 clamped to 10; R- with no
  criterion met hardcoded to 10. Ported as
  `ssl_timeresolved_decoding.reconstruct_learning_trial`, matches stored
  value 89/89 sessions (`089_learning_trial_sanity.py`); curve outcomes ==
  lick_flag 88/89 (MH038 length mismatch -> skipped).
- Decision (user, 2026-09-24): keep all stored values incl. hardcoded 10s,
  but label `lt_source` (learner / expert / *_floor10 / learner_fallback10)
  in every figure. Non-learners (NaN) excluded.
- Split: decoded trials (`prep_hitmiss_trials`) split by learning trial's
  start_time; learning trial itself is 'post'.
- Decode: single window 5-50ms (SENSORY_WINDOW), L2-logistic, pooled CV.
  Min 2 trials/class/epoch (user choice). Imbalance: both epochs
  subsampled (100x) to n_hit*=min(pre_hit,post_hit), n_miss*=min(pre_miss,
  post_miss) -- i.e. post matched to pre in the usual case (user choice);
  same n_folds=min(5,n_hit*,n_miss*) for both. Unmatched per-epoch scores
  kept as secondary. Per-epoch label-shuffle null. One C per (session,
  area) from all session trials.
- Caveat raised to user: any split point yields pre/post differences
  (drift, engagement, hit rate); whether the learning trial is *special*
  would need a placebo-split null (offered; user chose to proceed with the
  decoding comparison first).

### Learning-trial split: redo (2026-09-24, user request)
"Redo with drift linear shift null. Do both baseline-corrected and
non-corrected and baseline as windows for decoding. Drop last run of misses
at the end if it is clearly a stated state." -> `095`/`096` (090/091 kept):
- Null: session-level linear (non-wrapping) shift of the label sequence
  (10-50% of trials, random direction), each pair keeps the neural trial's
  epoch, full matched/unmatched per-epoch decode re-run; first 50 VALID
  shifts (>=2 per class per epoch) kept. Replaces 090's per-epoch label
  shuffle, which did not control slow drift (the project's own hit/miss
  convention since 2026-09-13/15 -- 090 should have used it).
- Windows: sensory 5-50ms, baseline -200..-10ms, sensory minus baseline.
- Disengagement: drop trailing block after the last lick on ANY trial type
  when it holds >=5 whisker AND >=3 auditory trials (8/89 sessions; check
  `095a_disengagement_check.png`). Deliberately NOT a whisker-only
  terminal-miss-run rule: for R- that run is the learned withholding (29/38
  sessions have >=5), and many R+ runs occur with auditory hit rate ~1
  (engaged) -- so neither is "clearly a state".

### Placebo-split test of the learning_trial (added 2026-09-24, user request)
Is the learning_trial special, or would any split show a pre/post change?
`097` re-runs 095's matched pre/post decode (whole brain, 3 windows,
disengagement dropped) at every 3rd valid whisker-trial split (<=60) plus
the real LT; statistic delta = matched post - pre balanced accuracy (same
reduced estimator for real and placebo; no per-split shift null -- the
placebo distribution is the null and shares the session's drift). `098`:
per-session percentile of the real delta among placebos (|k-LT|>=5),
also size-matched (placebos with x0.5-x2 the real matched trial count) and
magnitude (|delta - median|) variants; group Wilcoxon + t of pct vs 0.5
per cohort, R+ vs R- MWU + Welch. Prediction if LT is special for R-:
pct < 0.5 (larger drop at LT than elsewhere).
