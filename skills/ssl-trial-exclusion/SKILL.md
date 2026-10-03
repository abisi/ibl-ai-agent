---
name: ssl-trial-exclusion
description: Use this skill whenever an SSL analysis selects trials (behaviour, learning curves, learning-trial methods, decoding, single-cell or population analyses). Lists the mandatory trial exclusions (perf == 6 everywhere except passive trials, auditory warm-up block, end-of-session disengagement for neural analyses) and where each is already implemented.
---

# SSL Trial Exclusion

## Use this skill when
- Building any trial set from `trials.parquet` (local `ssl_ephys` / `ssl_behavior`) or NWB `trials`.
- Writing a new trial-preparation helper, or reviewing one.

## Rules (in order of application)
1. **perf == 6 is always excluded, except for passive trials.** `perf == 6` marks excluded trials (e.g. AB105: an 86-trial block at
   the end of whisker day 0). Drop them from every active-context analysis: behaviour, learning curves, learning-trial
   identification, decoding, single-cell. Passive trials (`context == 'passive'`) keep their own definition (see
   `skills/ssl-analyze/references/ssl_task_semantics.md`) and are not filtered on `perf`. User rule, 2026-10-01.
2. **Context.** Active analyses use `context == 'active'` (sessions without recorded context store the string `'nan'`
   and are all active).
3. **Auditory warm-up block.** Drop the auditory-only block before the first whisker trial, but keep exactly one trial
   before the first whisker trial and never drop the first whisker trial
   (`skills/ssl-analyze/references/ssl_auditory_warmup_block.md`). The kept pre-whisker trial is usually the last warm-up
   (auditory) trial; it enters any analysis that uses auditory trials (e.g. modality decoding) unless that analysis
   removes trials before the first whisker trial explicitly.
4. **End-of-session disengagement (neural analyses only): rule A1.** Drop the tail after the session's last lick on any
   trial type when it holds >= 5 whisker and >= 1 auditory trials
   (`ssl_timeresolved_decoding.detect_terminal_disengagement`, default `min_auditory=1`). Never trim for learning-trial
   identification or learning curves.
5. Stimulus-artefact dead zone and lick alignment are not trial exclusions but apply to the same pipelines:
   `skills/ssl-analyze/references/ssl_artifact_dead_zone.md`, `skills/ssl-lick-alignment/SKILL.md`.

## Where it is implemented
- `scripts/ssl_bwm_trial_prep.py::prep_session` -- context, perf != 6 (added 2026-10-01), warm-up cut. Used by
  `prep_hitmiss_trials`, `prep_modality_trials`, `prep_lick_aligned_trials` and the decoding pipelines.
- `scripts/ssl_timeresolved_decoding.py::_active_trials_from_whisker_onset_for_curve` /
  `_active_trials_for_curve_untrimmed` -- the learning-curve trial sets (active, perf != 6).
- New helpers that read raw trial tables must apply rule 1 explicitly; check with
  `assert not (trials.loc[trials.context != 'passive', 'perf'] == 6).any()`.

## Provenance
- Results computed before 2026-10-01 with `prep_session` included active perf == 6 trials (313 trials in 26 ssl_ephys
  sessions); learning-curve sets always excluded them. State the exclusion set in every saved result's config.
