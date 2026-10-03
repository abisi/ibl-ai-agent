# TODO

- [X] 001 halves analysis (moved from ssl-prelick-convergence 072, 2026-10-03): `_within_day_sl/halves/`
      (within_session_sessions.csv; <pop>/within_session_tests.csv; within_session_{time,oddeven,controls}.png).
- [X] 002 trial-level slopes on a fixed session axis with expert comparison (2026-10-03): `_within_day_sl/slopes/`
      (trial_slopes_sessions.csv, trial_slopes_trajectories.csv, <pop>/trial_slopes_tests.csv, <pop>/trial_slopes.png).
      First run had per-fold normalisation blowing up; fixed: unit axis per fold, one session normalisation, d′ ≥ 0.3.
- [ ] Robustness splits for 001: session midpoint (trial index) and median whisker hit.
- [ ] Drift control: all classes drift up the SL → AH axis within sessions (all groups); model the drift explicitly
      (e.g. regress WH on time with SL-trial scores as a time-varying reference).
- [ ] Per-mouse saturating fit on day 0 (when does R− WH reach SL?) against the behavioural drop in whisker licking.
- [ ] Report.
