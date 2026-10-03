## Purpose
Mandatory rule for excluding a within-session auditory-only warm-up block
that precedes whisker-trial introduction in many whisker-training sessions.
This is **not** the same thing as the auditory-*pretraining* stage (a
separate, earlier session type, `session_description` prefix `auditory_*`,
already excluded from any whisker-training session list by
`list_whisker_training_ephys_sessions`'s regex — see `ssl_task_semantics.md`'s
Day/training-stage semantics section). This rule is about trials **inside**
an already-correctly-included whisker-training session's own `active`
epoch — no existing filter (day-stage regex, `context=='active'`) removes
these trials, because they genuinely belong to a valid whisker-training-day
session's active epoch.

## The rule (source: Axel Bisi, the experimenter, 2026-09-14)
Many whisker-training sessions begin with a block of auditory-detection-only
trials (no whisker trials present at all) before whisker trials are
introduced. Purpose: wake the mouse up, engage it in the task, and bring it
to auditory-expert performance before whisker discrimination begins. Only
after this warm-up does the session introduce `whisker_trial`s and become
the mixed-modality/whisker-learning epoch an analysis is normally asking
about.

**Concretely** (rule updated 2026-09-28, Axel Bisi): for any analysis of a
whisker-training session's `active` trials (hit/miss decoding, modality
decoding, performance-state analysis, trial-order/behavioral analyses, or any
other per-trial statistic), find that session's **first active
`whisker_trial`** in chronological order and drop every active trial before
it **except the one immediately preceding it**:

- **Never remove the first whisker trial.**
- **Always keep exactly 1 trial before the first whisker trial** (the last
  warm-up trial). If the session's first active trial is already a whisker
  trial, there is nothing before it and nothing is removed.
- Compute any t-1 / trial-history column (`t1_rewarded`, run index, ...) on
  the full active sequence **before** the cut, so both kept boundary trials
  have a real t-1. Do not add any other "drop the first trial" step; an
  analysis that needs a t-1 value drops its own NaN-t1 rows (only possible
  for a session's very first active trial).

Apply this **in addition to**, not instead of, the existing
`context=='active'` filter.

## Why this matters (discovered 2026-09-14)
A whisker/auditory **modality** decoder trained on the raw (untrimmed)
active-trial sequence can show apparent above-chance decoding **before
stimulus onset** (during the nominal pre-stimulus baseline), which looks
like an impossible anticipatory signal. Root cause: the untrimmed sequence
starts with a long run of auditory-only trials (the warm-up block), which
inflates the trial-type sequence's early-session serial autocorrelation
(measured directly: mean lag-1 autocorrelation of trial type was **+0.18**
in the first half of the active-trial sequence vs. **-0.13** in the second
half, across a 20-session check). A classifier can then decode "upcoming
modality" during baseline by picking up any short-lived residual/carry-over
signal from the *previous* trial's modality in the population's ongoing
state — not real anticipatory coding, just exploiting the fact that nearby
trials in the untrimmed sequence are not independently drawn. Trimming the
warm-up block out removes this artificial run structure at its source.
This does not explain the large, fast, genuine modality decoding rise
immediately after stimulus onset (a real sensory-evoked signal), only the
spurious pre-stimulus baseline effect.

**Action required**: any modality-decoding result computed before this fix
was applied needs to be rerun with the fix in place before its baseline
period is trusted.

**Directly confirmed** (2026-09-14, `AB086_20231015_141742`): the raw
active-epoch trial sequence for this session starts with 26 consecutive
`auditory_trial`/`no_stim_trial` rows — zero `whisker_trial`s — before the
first whisker trial appears. That is the warm-up block itself, not an
inference from aggregate statistics; the trim removes exactly this kind of
run.

## Where this is implemented
`scripts/ssl_bwm_trial_prep.py`'s `prep_session` — the single shared
entry point every SSL decoding trial-prep function
(`prep_hitmiss_trials`/`prep_modality_trials`/`prep_perfstate_trials_generic`/
`prep_lick_aligned_trials` in `scripts/ssl_timeresolved_decoding.py`) calls
first, applied immediately after the existing `context=='active'` filter
and before any `rewarded`/`t1_rewarded`/`run_index` derived column is
computed (so those reflect the true post-warm-up sequence too, not reaching
back into the warm-up block for "trial -1" context at the boundary).

**Correction (2026-09-28)**: the 2026-09-14 implementation cut the sequence
AT the first whisker trial and then applied a blanket "drop the first trial
(no t-1)", which removed the first whisker trial itself (76/88 learning
sessions). Whisker-only analyses (`prep_hitmiss_trials`, hit/miss decoding)
WERE therefore affected, contrary to what this file previously stated; and
before 2026-09-14 the same blanket drop removed the first whisker trial in
every session without a warm-up block. Results computed with either version
lack a whisker trial in those sessions and must be rerun (see the project
change logs for which were).

Exception: `_active_trials_from_whisker_onset_for_curve` (learning-curve
based performance states) starts at the first whisker trial to stay
positionally aligned with the stored learning-curve files; it keeps the
first whisker trial but not the preceding trial.

## Quality gates
- Reject any analysis whose trial set lacks the session's first active
  `whisker_trial`, or that keeps more (or, when one exists, fewer) than 1
  trial before it.
- Reject any modality-decoding (or other mixed-trial-type) analysis of a
  whisker-training session's active trials that keeps the warm-up block
  (more than the 1 trial immediately preceding the first whisker trial).
- Reject treating a pre-stimulus/baseline-period above-chance modality
  decoding result as a real anticipatory signal without first checking
  whether the warm-up-block trim was applied — see "Why this matters" above
  for the exact confound mechanism.
- Do not trust whisker-only results computed with `prep_session` between
  2026-09-14 and 2026-09-28 (or before 2026-09-14 for sessions without a
  warm-up block) without checking the first whisker trial is present.
