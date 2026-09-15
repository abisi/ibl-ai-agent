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

**Concretely**: for any analysis of a whisker-training session's `active`
trials (hit/miss decoding, modality decoding, performance-state analysis,
trial-order/behavioral analyses, or any other per-trial statistic), find
that session's **first active `whisker_trial`** in chronological order and
drop every active trial before it. Apply this **in addition to**, not
instead of, the existing `context=='active'` filter — it is a further trim
on top of that, not a replacement for it.

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

Note this fix only changes behavior for analyses that keep **both**
`whisker_trial` and `auditory_trial` rows together (modality decoding,
lick-aligned modality decoding) — `prep_hitmiss_trials` and
`prep_perfstate_trials_generic` immediately filter to `whisker_trial` rows
only, and recompute their own chronological `half`/`block_id` **after**
that filter, so a preceding all-auditory block (which by definition
contains zero whisker trials) was never counted into their own
whisker-only sequence or its median/block split in the first place — those
two targets' existing results are unaffected by this fix and do not need
to be rerun on this basis alone.

## Quality gates
- Reject any modality-decoding (or other mixed-trial-type) analysis of a
  whisker-training session's active trials that does not first drop trials
  preceding that session's first active `whisker_trial`.
- Reject treating a pre-stimulus/baseline-period above-chance modality
  decoding result as a real anticipatory signal without first checking
  whether the warm-up-block trim was applied — see "Why this matters" above
  for the exact confound mechanism.
- Do not apply this trim's rationale to `prep_hitmiss_trials`/
  `prep_perfstate_trials_generic` results as if they needed rerunning too —
  they filter to whisker-only trials before any chronological split, so a
  preceding all-auditory block cannot have entered their computation.
