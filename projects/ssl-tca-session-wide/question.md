# ssl-tca-session-wide: question

## Original request (user, 2026-11-21)
"Check my TCA analysis in brain_wide_analysis. Adapt it to do it session-wide, relate trial factors to behavioral
performance (learning curves), try to match components between mice. Do within-cohort analyses. Furthermore, be more
open than me in my assumptions in a way to relate decomposition to learning or the task. Look into baseline changes,
sensory evoked activity (5-50 ms after stim), decision signals, be quite open."

## Starting point: the user's megamouse pipeline
`brain_wide_analysis/tca/tca_pipeline_bis.py` (haas `~/code/brain_wide_analysis`, not a git repo):
- neurons of all mice concatenated on one neuron axis ("megamouse"), trial axis shared by index;
- whisker (or auditory) trials only, 18 before + 62 after the first active whisker hit (or +-10 around the inflection);
- window -0.1..0.2 s (0.4 s in the per-group branch), 10 ms bins, min-max per neuron, non-negative CP (ncp_hals),
  ranks 1-6, 5 replicates.

Review findings (to discuss, not yet fixed in the user's code):
1. Artefact window: spikes in -2..+4 ms are replaced by unseeded Poisson spikes at the baseline rate and then used as
   data. ssl-analyze rule: Poisson fill is for visualisation only; the evoked 0-10 ms bin is partly synthetic.
2. first_hit alignment: when a mouse has fewer than 18 trials before its first hit, `start_idx` is clipped to 0, but
   `whisker_trial_id` is still `arange(-18, ...)`, so its trial indices are shifted relative to the other mice. Concatenation
   on the neuron axis also needs identical trial counts, so mice with short windows either break or misalign.
3. No unit-stability filter visible: slow drift / unit loss over a session produces a trial factor that looks like learning.
4. Trial selection does not apply perf == 6, the warm-up cut or the per-trial context rule; passive trials enter with
   lick_flag forced to 0.
5. A shared trial factor across mice assumes all mice change at the same trial index; the window (to 0.2 / 0.4 s) does
   not reach most of the lick / decision period, and there is no pre-stimulus baseline beyond 100 ms.
6. Cohort is taken from the unit / trial table `reward_group` (session-level), not the reference sheet.

## Refined question (draft, to be refined in exploration)
Within single learning sessions (whisker day 0), which low-dimensional neuron x time x trial structure does brain-wide
activity have, and do trial factors (a) follow the behavioural learning curve, (b) separate task variables (trial type,
outcome, context passive/active, reward history), and (c) do the same components recur across mice within a cohort?

Sub-questions (open, as asked):
- Baseline: does pre-stimulus activity (-1..0 s) change across the session, and does it track learning, engagement or
  just time / drift?
- Early sensory evoked (5-50 ms after whisker / auditory onset): does the evoked gain change with learning
  (passive pre vs post, early vs late active trials), per area?
- Decision / action (50 ms to ~1 s, lick-related): do decision components emerge, and when relative to the learning trial?

## Definitions (draft)
- Session: one whisker day-0 NWB_ks4 session; unit set = stable good + mua (unit_info_all_days_v2 + 137_stable_units).
- Trials: all trials of the session in time order (passive pre, active whisker / auditory / no-stim, passive post), with
  the ssl-trial-exclusion rules (perf == 6 dropped except passive, warm-up cut, per-trial context rule; A1 disengagement
  trim for neural analyses).
- Learning curve: exact curve h5 (`combined_results_ks4/<mouse>/whisker_0/learning_curve/*_learning_curve_exact.h5`).
- Cohort: reference sheet `reward_group`; populations: all mice and learners (sheet good / moderate).
- Null for trial-factor vs learning curve: non-wrapping linear shift (memory rule), not circular / trial shuffling.
  Note: ssl_analysis_patterns.md (TCA section) still says circular shift; to be corrected.

## Main decomposition (user 2026-11-21)
Baseline-subtracted, signed TCA (norm `bsub` in 003: per-trial, per-unit baseline rate in -1.0..-0.010 s subtracted,
scaled by max |trial-averaged change| + 5 Hz, cp_als) is the main variant for learning, so that pre-stimulus state
changes do not enter the evoked / decision components. Non-negative fits (soft, minmax) are kept as references; the
baseline epoch (non-negative, soft) describes the state changes themselves. Remaining confounds to control: gain
changes (correlate with baseline-epoch trial factors) and time in session (partial on trial index, linear-shift null).

## Exploration / confirmation split
User 2026-11-21: small pilot (5 learning sessions per cohort + 2 expert per cohort, tables/sessions.csv `pilot`) to
choose epochs / normalisation / rank and define tests (exploration set); the remaining learning sessions (46 R+, 34 R-)
are the confirmation set for the locked tests (option text: "explore on ~5 sessions per cohort, then confirm on the
remaining ~80"). Confirmation runs only after explicit approval of the locked tests.
Expert sessions (19 R+, 17 R-) are fitted as a reference.

## Jaw-onset alignment (user 2026-11-21)
Decision signals: TCA aligned to jaw onset (pkl `jaw_dlc_onset`), -0.30..+0.20 s, lick trials (`jaw`) and all trials
with non-movement trials pseudo-aligned to the type-matched median onset (`jaw_all`, hit vs miss). A candidate decision
component should rise before jaw onset and be sharper jaw-aligned than stimulus-aligned; reaction time is a covariate.

## Event-aligned design (user 2026-11-22, replaces the 003 epoch grid for the main analysis)
- One builder (005_build_aligned.py) from NWB spike times; firing rates in 50-ms bins, unsmoothed; per-trial baseline
  (start - 1.0 .. start - 0.010 s of the same trial) subtracted, also for jaw / lick alignment (baseline borrowed
  from before that trial's start).
- Alignments: stimulus long window (-0.2..1.5 s), stimulus sensory window (+5..105 ms; 50 ms = 2 bins, so a 10-ms
  check is fitted too), jaw onset (-0.3..0.3 s), corrected first lick (-0.3..0.3 s; start + lick_time -
  response_window_start_time). Jaw and lick use the same trials (active lick trials with a valid jaw onset).
- Normalisation compared: rate (Hz), soft (/ max |trial-averaged change| + 5 Hz), z-score (/ SD over trials x time,
  not centred; user 2026-11-22 instead of min-max, which would undo the baseline subtraction if shifted).
- Sensory window in 10-ms bins only (50 ms gave 2 bins); stimulus long window 50 ms.
- Rank selection (user 2026-11-22): only the Williams et al. 2018 trade-off between error decrease and restart
  similarity (10 restarts; largest rank with mean similarity >= 0.85 and error decrease >= 5 % of the total from
  rank 1 to 10). No held-out data. Model comparison in-sample vs the condition-mean model.
- Normalisation also matters for matching components between mice (unit-weight / area profiles comparable).
- Pilot: one example mouse (AB123, R+, good learner, day 0), then decide.
- Trial markers: trial-type colour; filled = lick, open = no lick; passive = lightened colour.

## Methods recap for the full run (2026-11-22, awaiting user approval)
Data
- Sessions: whisker day 0 (learning; 51 R+, 39 R-) + expert reference (19 R+, 17 R-); cohort and learner status from
  the reference sheet. Units: stable good + mua (unit_info_all_days_v2 + 137_stable_units). Trials: ssl-trial-exclusion
  rules (per-trial context, perf != 6 and early_lick == 0 for active, warm-up cut, A1 trim). Artefact: spikes in
  -10..+5 ms of every trial start removed; bins overlapping it use the rate over their valid part.
- Single-trial baseline subtraction (user): each trial's rate in start - 1.0 .. start - 0.010 s subtracted per unit,
  for every alignment (jaw / lick borrow the baseline before their own trial's start).
- Every window keeps pre-event bins (user): stimulus long -0.2..+1.5 s (50 ms); stimulus sensory -0.05..-0.01 s and
  +0.005..+0.105 s (10 ms, artefact gap -10..+5 ms left out); jaw onset and first lick -0.3..+0.3 s (50 ms).
- Trials: stimulus windows whisker + auditory (passive + active; no-stim dropped); jaw / lick: active lick trials
  with a valid jaw onset (same trials for both). Units with mean rate < 0.5 Hz dropped. Unsmoothed.
- Normalisation: z-score by scaling only (unit's baseline-subtracted values / their SD over trials x time). No capping,
  no outlier-trial masking (user).
Decompositions (Kuma GPU, tensorly, float32; 006g_tca_gpu.py)
- Normal TCA: unconstrained (signed) CP-ALS. Known issue: cancelling component pairs from rank ~4; reported with the
  component-overlap diagnostic (most negative Tucker congruence), no ridge.
- Split non-negative TCA: increase channel max(D, 0) and decrease channel max(-D, 0), non-negative CP (HALS);
  reported unit weight = increase - decrease (net); directionality (inc - dec) / (inc + dec) flags variability
  components.
- Ranks 1..12, 10 random restarts, best kept; restart similarity = tensortools kruskal_align rule (mean |cosine| over
  modes, Hungarian). Comparisons across mice at one fixed rank (to choose).

## Analysis plan: components related to performance and learning (draft, 2026-11-22, awaiting approval)
Unit of analysis: mouse (one day-0 session per mouse); populations: all mice and learners (sheet good / moderate);
R+ vs R- with Mann-Whitney + Welch; both decompositions; fixed rank. Exploration on the pilot sessions; locked tests on
the confirmation set only after approval.
1. Component descriptors (per session, per component)
   - time: peak latency, duration, sign (net), pre-event activity (bins before the event).
   - area: net unit weight per area group -> area profile; specificity = share of |net weight| in the top area group,
     against a null that permutes area labels across units of the session (labels, not time).
   - trial: mean trial weight per class (passive / active x whisker / auditory x lick).
2. Within-session relation to behaviour (trial weights)
   - performance: whisker hit vs miss (and FA vs CR) separation of trial weights (AUC);
   - learning: Spearman correlation of trial weights with the exact whisker learning curve (p_wh_i), partialled on
     trial index; and change at the learning trial (before vs after);
   - null: non-wrapping linear shifts of the trial weights against behaviour (10-50 % lags), never circular.
3. Across mice: which components recur
   - match components between sessions on time factor + area profile + trial-class profile (tensortools mean rule,
     Hungarian), consensus clusters of components across mice;
   - null for matching: same procedure on components from fits to data with trials shifted (linear) per unit.
4. Classification of recurring components
   - global: area profile spread over groups (specificity not above the label-permutation null);
   - area-specific: specificity above the null, same top area group in most mice of a cluster;
   - cohort-specific: cluster membership or behaviour relation differs between R+ and R- (mouse-level permutation of
     cohort labels).
5. Expert sessions as reference: do learning-related day-0 components persist / dominate at expert stage.
Outputs: per-component table (session, alignment, method, descriptors, behaviour statistics), cluster table, figures.

## User decisions on the recap (2026-11-22)
- Fixed rank: chosen after fitting the final datasets, cohort-specific (one rank for R+, one for R-), same criterion.
- Data use: all sessions, exploratory (supersedes the exploration / confirmation split); results labelled
  exploratory, uncorrected unless stated.
- "Normal TCA": user asked for advice (conflicting aims: single-trial baseline subtraction vs a standard method).
- Decompositions (user 2026-11-22, after advice): main = split non-negative TCA on single-trial baseline-subtracted,
  z-scored data, unit weights reported as increase - decrease (net); reference = non-negative TCA on rates without
  baseline subtraction (Williams et al. 2018), z-scored by scaling only. Signed CP dropped (cancelling pairs).
- Sensory window: one regular 10-ms grid shifted by -2.5 ms (-52.5..+107.5 ms, 16 bins); the two bins touching the
  artefact (-10..+5 ms) are kept with the valid-time rate correction (2.5 ms valid each).
