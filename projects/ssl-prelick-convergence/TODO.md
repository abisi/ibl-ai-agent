# TODO

Steps performed in `ssl-rastermap-psth-variants` (2026-09-29 → 2026-10-03) before the split; outputs in
`combined_results_ks4/_roc_prelick{,_sl}/` (see the README there).
- [X] 051–056 pre-lick ROC (WH/AH/ref), variants (all, short/long RT, RT-matched), min FR 0.1 Hz, stage × area
      summaries, PSTHs.
- [X] 057 λ and cross-validated distances; 058/059 convergence summary and single-neuron transfer; 060 λ_LDA.
- [X] 061 learners-only rule (bad mice lose only their day-0 session).
- [X] 063 decoder sweep; 068/069 single-session decoders with linear-shift null (both references).
- [X] 064 pseudo-population hierarchical bootstrap (mice → sessions → neurons → trials), shift null; 2026-10-03 rerun
      with transfer_bin and num_prob readouts (SL first; FA running).
- [X] 065–067 converging-unit locations, area attrition, CCF density maps; 070 recap; 071 SL timing control.
- [X] 062 publication figures (distance-based Fig 3, Fig 3S λ/λ_LDA, Fig 1h, Fig 2f/g, Fig 2S, Fig 5 pseudo-pop rows,
      COSYNE v2/v3), complete captions, article PDFs per variant (build_article.py + render.sh).
- [X] Split into this project; LOCKED.md (locked set, results, caveats, TODOs).

Planned
- [X] Within-session remapping (072, 2026-10-03; SL reference; moved to project ssl-within-day-remapping as 001): halves split at the median auditory hit (equal AH per
      half; session end trimmed by A1), count-matched subsamples (20), Δd / λ / components per half, whole-session CV
      decoder evaluated per half and cross-half decoder (train one half, read out the other), linear-shift chance,
      odd/even null split, controls (d(AH,SL), event rates, RT, raw rates). Output `_roc_prelick_sl/within_session/`
      (within_session_sessions.csv, <pop>/within_session_tests.csv, within_session_{time,oddeven,controls}.png).
- [ ] Increase pseudo-population repetitions for final runs.
- [ ] RT-matched single-session decoders and pseudo-populations (FA reference).
- [ ] Final COSYNE figure choice (v3) and abstract text.

## Next iteration (decided 2026-10-07)
Spontaneous-lick reference (072/073 example sessions, figures in combined_results_ks4/ssl-prelick-convergence/sl_definitions/):
- [ ] New SL definition "C8" in 051 (replaces the current one): lick train = piezo licks + corrected first lick of every
      trial (piezo detection misses many false-alarm licks), licks < 50 ms apart collapsed; SL = first lick after >= 1 s
      without licking (bout onset, as now); false-alarm licks (no-stim response windows) COUNT as SL (user: same
      behaviour, measured in a window); exclude SL in [stim - 0.5 s, stim + W] of every whisker / auditory trial, hit or
      miss (stimulus processing, reward collection).
- [ ] Set W from the data first: duration of post-hit licking (reward collection) across sessions; default W = 8 s.
- [ ] Rerun 051 (SL reference, new tag) and everything downstream (057/058/059/060/061/062, 063/064/068/069, 065-067,
      070, 071) and ssl-within-day-remapping; side-by-side comparison of the headline statistics old vs C8; then update
      LOCKED.md with the user's choice.
- [ ] 071 timing control becomes redundant for exclusions <= W; keep only longer windows.
- [ ] Upstream (NWB conversion, behaviour_analysis): false-alarm / weak licks missing from piezo_lick_times (TODO in
      spontaneous_licks_utils.py); spontaneous_licks_utils exclusion only around hits lets false-alarm-window and
      post-miss licks in, and its defaults lack trial_exclusion_window_s.
Harmonisation with ssl-within-day-remapping (see the method map in README):
- [ ] One minimum of events per class across analyses (now 3 in 051 ROC, 4 in 057 / 068 / within-day 001-002, 6 in 064).
- [ ] One implementation each of the decoder + linear-shift null and of the mouse-level cohort-permutation interaction
      (now 068 cv_bacc / within-day 001 ws_predict, and 062 group_stats / within-day 001 perm_interaction).
- [ ] Migrate results to combined_results_ks4/<slug>/ (now _roc_prelick{,_sl}/ and _within_day_sl/) and the report to
      the project-report standard (report_lib).


# Part II: within days (formerly ssl-within-day-remapping/TODO.md)


- [X] 001 halves analysis (moved from ssl-prelick-convergence 072, 2026-10-03): `within_day/sl/halves/`
      (within_session_sessions.csv; <pop>/within_session_tests.csv; within_session_{time,oddeven,controls}.png).
- [X] 002 trial-level slopes on a fixed session axis with expert comparison (2026-10-03): `within_day/sl/slopes/`
      (trial_slopes_sessions.csv, trial_slopes_trajectories.csv, <pop>/trial_slopes_tests.csv, <pop>/trial_slopes.png).
      First run had per-fold normalisation blowing up; fixed: unit axis per fold, one session normalisation, d′ ≥ 0.3.
- [X] 003 within-day vs across-day on a common footing (2026-10-03): phase-matched halves (split at median AH) in every
      session, fixed 6 events per class and 150 units, session-anchored measures (WH − SL position on the session axis,
      whole-session decoder with shift null read out per epoch, Δd / session d(AH, SL)); contrasts within-day,
      across-day (early, late), carry-over; hierarchical bootstrap (mice → sessions), mouse-level cohort permutation,
      MixedLM. Output `within_day/sl/epochs/` (epoch_sessions.csv, <pop>/epoch_contrasts.csv, <pop>/epoch_comparison.png).
      Only 5 R− expert sessions (3 mice) pass 6 events per class per half.
- [X] 003 with 4 events per class, both cohorts (`--n-fix 4`, output `within_day/sl/epochs_n4/`): R− experts 7 sessions /
      5 mice (vs 5 / 3 with 6 events).
- [X] 004 per-half first-lick PSTHs (spikes re-binned per session half, 051 events): `within_day/sl/psth_halves/`.
- [X] 005 COSYNE 5-panel summary "R− whisker-hit activity changes more within the learning day than within expert
      sessions" (PSTHs per half, whisker-hit rate, trial-level WH − SL trajectory, per-half normalised distance,
      |within-session change| day 0 vs expert): `within_day/sl/cosyne/COSYNE_within_day.{png,pdf,svg}` + stats csv.
- [X] 006 compact COSYNE timeline, 007 caption compiler, 008 expanded COSYNE figure (methods with examples, results,
      controls), coding-direction (CD) wording and equations in captions (2026-10-04).
- [X] 009 single-trial mixed model (CD projection ~ whisker hit × time × cohort; per-session random intercept, time slope
      and whisker-hit offset; cohort shuffles across mice): `within_day/sl/mixed_model/`.
- [X] 003 decoder schemes (`--n-units`, `--unit-draws`): 150 × 10, all units, 50 / 100 × 10, 400 × 5; 010 comparison figure.
- [ ] Learning-aligned version (trials since each mouse's learning trial).
- [ ] Robustness splits for 001: session midpoint (trial index) and median whisker hit.
- [ ] Drift control: all classes drift up the SL → AH axis within sessions (all groups); model the drift explicitly
      (e.g. regress WH on time with SL-trial scores as a time-varying reference).
- [ ] Per-mouse saturating fit on day 0 (when does R− WH reach SL?) against the behavioural drop in whisker licking.
- [ ] Report.

## Next iteration (decided 2026-10-07)
- [ ] Spontaneous-lick reference "C8" (false-alarm licks counted, exclusion [stim - 0.5 s, stim + W] after every
      whisker / auditory trial, trial licks merged into the lick train): inherited from ssl-prelick-convergence 051 --
      rerun 001-010 after the 051 rerun; see ssl-prelick-convergence/TODO.md "Next iteration".
- [ ] Harmonise with ssl-prelick-convergence: event minima, one decoder / shift-null / interaction implementation.
