## Purpose
Task design and trial/column semantics for the SSL dataset. Verified against
the `ssl_ephys`/`ssl_ks2_ephys` builder code in `ibl_ai_agent/datasets/`,
directly confirmed by Axel Bisi (the experimenter) on 2026-08-11, term
explications locked in the `ssl-passive-sensory-selectivity` and
`ssl-reward-history-modulation` projects' question.md files, and — as the
authoritative source for task design itself — the full paradigm description
in `ssl_behavioral_paradigm.md` (Axel's own paper/thesis text, transcribed
2026-08-13). Read that file first for task design; this file focuses on how
the design maps onto actual dataset columns.

## Task design
Mice perform a stimulus-detection task on whisker and auditory stimuli, with
**reward delivered on hits** for auditory trials (both cohorts) and for
whisker trials in the R+ cohort only. The task is a true **Go/No-Go** only
for R- whisker trials specifically — R- mice are rewarded for *withholding*
a lick on whisker trials, while auditory trials (both cohorts) and R+ whisker
trials are a standard Go task throughout. See `ssl_behavioral_paradigm.md`
for the full three-stage design (free-licking → auditory pretraining →
whisker learning) and the complete outcome/reward-contingency tables.

Each session has up to three epochs: `passive_pre` (stimulus exposure, no
behavioral contingency, spout retracted) → `active` (the reward contingency
is in effect) → `passive_post`. Passive blocks (35 auditory + 35 whisker
stimuli interleaved every 3 s) are recorded on every day with neural
activity, but **only for subjects AB116 and above** — subjects with numeric
index below 116 never have passive epochs, on any day — see
`ssl_behavioral_paradigm.md`'s Passive stimulation section. Not every session
has all three epochs recorded — see
`../../ssl-load/references/ssl_dataset_schema.md` for which epochs exist per source/subject.

The `spout` keypoint tracked in `ssl_behavior`/`ssl_ks2_behavior` corresponds
to licking for reward on hits.

## Day / training-stage semantics
`sessions.session_description` encodes `<behavior_type>_<day>`, e.g.
`"whisker_0"`, `"whisker_+2"`, `"auditory_0"`, `"free_licking_0"`,
`"whisker_on_1_opto"`. The `<behavior_type>` prefix names the paradigm stage
from `ssl_behavioral_paradigm.md` (`free_licking` → stage 1, `auditory` →
stage 2, `whisker`/`whisker_on_N_opto`/`whisker_off_N_opto` → stage 3). Two
day-numbering conventions recur across projects, both scoped to stage 3:
- **`day==0` = "learning"**: each subject's first whisker-training ephys
  session. For most subjects this is the *only* ephys session (verified:
  learning is exactly one session per subject in both cohorts in
  `ssl-ks2-dataset-description`, 36/36 and 53/53).
- **`day>0` = "expert"**: later sessions. Unevenly distributed — a minority
  of subjects contribute several expert-day sessions (e.g. 6 subjects → 9
  sessions in one cohort). **Use subject, not session, as the statistical
  unit for any expert-arm analysis.**
- `whisker_on_N_*`/`whisker_off_N_*` are opto variants; day 0 never appears
  in those. `auditory_*`/`free_licking_*` are pre-whisker-training sessions.
  Filter these out explicitly when a question means whisker-learning day 0.

## Trial columns (`trials.parquet`, both ephys sources)
- `trial_type`: `whisker_trial` / `auditory_trial` / `no_stim_trial`. **Before
  computing any firing rate, PSTH, or spike-based metric around `start_time`
  for `whisker_trial` rows, apply the mandatory dead-zone exclusion in
  `ssl_artifact_dead_zone.md` (-10ms/+5ms around `start_time`, magnetic
  stimulation artifact) — this is a correctness requirement, not optional.**
- `lick_flag`: 1 if the mouse licked during the response window, else 0.
- `context`: `active` / `passive` — use this, not `epochs.parquet` boundaries,
  when the two disagree (see `../../ssl-load/references/ssl_dataset_schema.md`).
- `perf` (a `TRIAL_MAP`-style outcome code, from the user's external
  `M:\analysis\Axel_Bisi\Github\ephys_utilities\neural_utils\neural_utils.py`,
  **not** reproduced as a stored column in this repo's compressed tables —
  recompute from `trial_type`/`lick_flag` if needed):
  `0=whisker_miss, 1=auditory_miss, 2=whisker_hit, 3=auditory_hit,
  4=correct_rejection, 5=false_alarm, 6=association`. This labels the raw
  stimulus-response pattern only — it is **not cohort-corrected** (see below).

### Naive (cohort-agnostic) outcome classification
Validated against ground-truth event categories in `events.parquet` in the
`ssl-task-performance` project:
- `whisker_trial` / `auditory_trial`: `lick_flag==1` → hit, `==0` → miss.
- `no_stim_trial`: `lick_flag==1` → false alarm, `==0` → correct rejection.
This is sufficient for plotting raw hit-rate/false-alarm curves, but **is not
the rewarded outcome** for whisker trials — see below.

### Cohort-dependent ("cohort-corrected") outcome — required for reward semantics
Only the **whisker** outcome flips meaning by cohort; auditory and no-stim do
not.
- Whisker trials, **R+** cohort: rewarded outcome ("hit") = `whisker_hit`
  (lick), non-hit = `whisker_miss`.
- Whisker trials, **R-** cohort: rewarded outcome ("hit") = `whisker_miss`
  — **withholding is the rewarded action** for this cohort — non-hit =
  `whisker_hit`.
- Auditory trials (both cohorts): hit = `auditory_hit`. `auditory_miss` is
  rare enough that it is routinely excluded as a target/outcome-of-interest
  trial, though it can still be used as a *conditioning* event (e.g. as a
  `t-1` trial in a history-effect analysis).
- `no_stim_trial` outcomes (`correct_rejection`/`false_alarm`) are not
  reward-cohort-dependent and are typically excluded from reward-history
  analyses entirely.
**Any analysis that compares "hit rate" or reward-related firing across
cohorts must use the cohort-corrected definition, not the naive
`lick_flag`-only one** — using the naive definition silently reverses the
whisker result for the R- cohort.

## Cohort / reward_group — two non-interchangeable sources
See `ssl-load/references/ssl_loading_policy.md` for the loading mechanics.
Semantically:
- **Session-metadata `wh_reward`** (from NWB `experiment_description`):
  per-session binary field describing which modality was rewarded for that
  session's subject.
- **`joint_mouse_reference_weight.xlsx` `reward_group`**: per-mouse reference
  sheet, three levels `R+`/`R-`/`R+proba`. `R+proba` mice are a distinct,
  smaller probabilistic-reward group and were dropped entirely in
  `ssl-reward-history-modulation` (unannotated for `learning_category`,
  outside the standard learning-cohort framework) — do not silently fold
  `R+proba` into `R+`/`R-` unless the question specifically wants it.
  This sheet also carries `learning_category` (`good`/`moderate`/`bad`/NaN)
  and `exclude_ephys` — established practice drops `learning_category in
  {bad, NaN}` ("bad-learner exclusion") and requires `exclude_ephys==0` for
  neural-population analyses.
State which source was used; do not assume `wh_reward` and the reference
sheet's `reward_group` always agree for a given mouse/session.

## Brain-area grouping
Two distinct schemes recur across SSL projects — see
`ssl_analysis_patterns.md`'s Area Grouping section for when to use which.
