# TODO

- [X] 001 halves analysis (moved from ssl-prelick-convergence 072, 2026-10-03): `_within_day_sl/halves/`
      (within_session_sessions.csv; <pop>/within_session_tests.csv; within_session_{time,oddeven,controls}.png).
- [X] 002 trial-level slopes on a fixed session axis with expert comparison (2026-10-03): `_within_day_sl/slopes/`
      (trial_slopes_sessions.csv, trial_slopes_trajectories.csv, <pop>/trial_slopes_tests.csv, <pop>/trial_slopes.png).
      First run had per-fold normalisation blowing up; fixed: unit axis per fold, one session normalisation, d′ ≥ 0.3.
- [X] 003 within-day vs across-day on a common footing (2026-10-03): phase-matched halves (split at median AH) in every
      session, fixed 6 events per class and 150 units, session-anchored measures (WH − SL position on the session axis,
      whole-session decoder with shift null read out per epoch, Δd / session d(AH, SL)); contrasts within-day,
      across-day (early, late), carry-over; hierarchical bootstrap (mice → sessions), mouse-level cohort permutation,
      MixedLM. Output `_within_day_sl/epochs/` (epoch_sessions.csv, <pop>/epoch_contrasts.csv, <pop>/epoch_comparison.png).
      Only 5 R− expert sessions (3 mice) pass 6 events per class per half.
- [X] 003 with 4 events per class, both cohorts (`--n-fix 4`, output `_within_day_sl/epochs_n4/`): R− experts 7 sessions /
      5 mice (vs 5 / 3 with 6 events).
- [X] 004 per-half first-lick PSTHs (spikes re-binned per session half, 051 events): `_within_day_sl/psth_halves/`.
- [X] 005 COSYNE 5-panel summary "R− whisker-hit activity changes more within the learning day than within expert
      sessions" (PSTHs per half, whisker-hit rate, trial-level WH − SL trajectory, per-half normalised distance,
      |within-session change| day 0 vs expert): `_within_day_sl/cosyne/COSYNE_within_day.{png,pdf,svg}` + stats csv.
- [X] 006 compact COSYNE timeline, 007 caption compiler, 008 expanded COSYNE figure (methods with examples, results,
      controls), coding-direction (CD) wording and equations in captions (2026-10-04).
- [X] 009 single-trial mixed model (CD projection ~ whisker hit × time × cohort; per-session random intercept, time slope
      and whisker-hit offset; cohort shuffles across mice): `_within_day_sl/mixed_model/`.
- [X] 003 decoder schemes (`--n-units`, `--unit-draws`): 150 × 10, all units, 50 / 100 × 10, 400 × 5; 010 comparison figure.
- [ ] Learning-aligned version (trials since each mouse's learning trial).
- [ ] Robustness splits for 001: session midpoint (trial index) and median whisker hit.
- [ ] Drift control: all classes drift up the SL → AH axis within sessions (all groups); model the drift explicitly
      (e.g. regress WH on time with SL-trial scores as a time-varying reference).
- [ ] Per-mouse saturating fit on day 0 (when does R− WH reach SL?) against the behavioural drop in whisker licking.
- [ ] Report.
