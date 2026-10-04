# Question

## Original question (from user, 2026-08-26)
Preliminary analysis comparing levels of continuous and sensory-evoked
burstiness between cohorts (R+/R-). Global comparison first, then PERMANOVA
across cohorts, post-hoc across area levels if significant. Thoroughly
document the process with figures, from burst detection, quantification and
visualization to statistics. Do for good and good+MUA separately. Control for
movement by correlating burstiness with overall orofacial motion and licking
(DLC if available, or piezo lick time).

User-supplied method sketch (paraphrased from a source paper the user is
adapting, then corrected to this project's actual windows):
- Burst = run of >=2 spikes with intra-run ISI <=10ms, tail ISI >15ms.
- Burstiness = fraction of burst spikes among all spikes in the analysis
  window (continuous: whole recording).
- Burst index = fB(response window) - fB(baseline window), fB = n_burst/n_all
  in that window (0 if no spikes, so response-only or baseline-only bursting
  units aren't dropped). User's own windows for this project (supersedes the
  paper's generic 0-200ms/-600 to -400ms example): response 5-50ms, baseline
  -200ms to -10ms, around `start_time`.
- Continuous: remove the (whisker) dead zone, then just the overall ratio.

## Clarifications resolved with user (2026-08-26, plan-mode Q&A)
- **Day-stage scope**: both learning (`day==0`) and expert (`day>0`), run
  separately.
- **Exploration/confirmation split**: explicitly skipped for this pass —
  "preliminary analysis" — treat as fully exploratory/hypothesis-generating,
  per the two documented precedents in `ssl_analysis_patterns.md` that did
  the same with recorded approval (`ssl-passive-sensory-selectivity`,
  `ssl-ks2-single-mouse-tca`).
- **Post-hoc area scheme**: fine `area_acronym_custom` with
  `split_merge_areas=True` (same scheme as this session's earlier QC-report
  artifact), not the coarser `area_group` scheme used in prior PERMANOVA
  precedents. Expect several fine areas to fail the post-hoc's per-area
  unit/mouse count floor (`MIN_UNITS_PER_COHORT_AREA=5`, >=2 mice/cohort) —
  expected, not a bug.
- **Intermediate checkpoint** (requested after plan approval): before the
  full-scale run, produce a small-subset report showing burst-detection
  validation figures and example "burst-coding" neurons (largest-magnitude
  burst index units), for sign-off before scaling up.

## Cohort factor source (own decision, stated explicitly — see change-log)
Uses `reward_group` from `joint_mouse_reference_weight.xlsx` (mouse-level,
`R+`/`R-`, `R+proba` dropped) as the cohort factor, NOT the native per-session
`wh_reward` column — `wh_reward` can change value session-to-session for the
same mouse (confirmed this session: `MH064` flips between its two expert
sessions), which is incompatible with mouse-block permutation's assumption
that the permuted factor is constant per mouse. Matches the established
PERMANOVA precedent in `ssl-reward-history-modulation`/`ssl-passive-coselectivity`.

## Mandatory filters applied (per this session's new SSL rules)
- `exclude == 0` (all analyses).
- `exclude_ephys == 0` (neural analysis, additionally).
- `reward_group == 'R+proba'` dropped.
- Both population scopes computed: entire dataset (above filters only) and
  learners-only (`learning_category in {'good','moderate'}`, additionally).

## Burst detection
See `ssl_artifact_dead_zone.md`'s mandatory whisker-trial dead zone
(-1ms/+4ms around `start_time`) — applied here as a new case: dead-zone
spikes are excised from a unit's spike train **before** ISI/burst labeling
for any whisker-trial-referenced computation, so a real spike adjacent to
the dead zone can't be spuriously chained to an artifact spike inside it.

Burst labeling is done once per unit on the **full continuous spike train**
(not per-window), so burst membership isn't an artifact of window edges; all
windowed quantities below are just filtered counts of these same pre-labeled
spikes.

## Metrics (per unit)
- **Continuous burstiness**: `n_burst / n_total` over the entire session
  recording (all epochs), whisker dead-zone windows excised, one overall
  ratio.
- **Burst index**, separately for `whisker_trial` and `auditory_trial`:
  `fB(response 5-50ms) - fB(baseline -200/-10ms)` around `start_time`, spikes
  pooled across all qualifying trials of that type before computing each
  window's `fB`. Dead-zone excision applies to whisker only.

## Statistics matrix (24 combinations)
day-stage (learning, expert) x population scope (all, learners-only) x
quality tier (good, good+MUA) x metric (continuous burstiness, burst index
whisker, burst index auditory):
1. Global: mouse-level (learning) / session-level (expert) Mann-Whitney +
   Welch, R+ vs R-, pooled across areas.
2. PERMANOVA (mouse-block permutation, `permanova_euclidean` reused from
   `projects/ssl-passive-coselectivity/exploratory-analyses/permanova.py`).
3. Post-hoc per `area_acronym_custom` with BH-FDR 5%, only if main PERMANOVA
   p is significant (`run_with_posthoc`, same module).

## Motion/licking control
Spearman correlation of session-level continuous burstiness vs. session-level
orofacial motion (DLC-style keypoint motion energy from `ssl_behavior`) and
licking (`spout` keypoint, or a piezo lick event from `events.parquet` if
found). Investigated during implementation, not assumed in advance.

## Status
Plan approved 2026-08-26. Implementation in progress — see `TODO.md`.
