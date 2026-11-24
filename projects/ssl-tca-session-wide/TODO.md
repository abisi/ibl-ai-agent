# ssl-tca-session-wide: TODO

Code: `projects/ssl-tca-session-wide/`; results: `combined_results_ks4/ssl-tca-session-wide/` (haas, NAS).

## 0. Planning
- [X] Review the user's megamouse pipeline (findings in question.md)
- [X] Data inventory: whisker day 0, stable good + mua: 51 R+ sessions (26 good, 13 moderate, 12 bad), 39 R- (33 / 4 / 2);
      median ~1,100-1,300 units, ~11 areas per session; expert: 19 R+, 17 R- sessions
- [X] CONSULT: plan, split, compute location -> user 2026-11-21: explore on a small pilot (5 day-0 sessions per
      cohort), confirm locked tests on the remaining sessions; expert sessions included as reference; compute on haas (Kuma if needed)

## 1. Tensors (artifacts, reusable)
- [X] 001 session table (001_sessions.py -> tables/sessions.csv, units.parquet; pilot = 5 day-0 per cohort
      stratified by learning_category + 2 expert per cohort, seed 0)
- [X] 001b trial selection in tca_lib.select_trials (was: 001 session table + trial table per session (exclusion rules, cohort from sheet, learning curve joined per trial)
- [X] 002 (pilot, 14 sessions) 002_build_tensors.py -> artifacts/tensors/<sid>.npz|_units|_trials.parquet; artefact
      window -10..+5 ms (ssl-s1m2-communication 006), not -1..+4
- [X] 002 (all 126 sessions) done 2026-11-21 (tables/tensor_build_log.csv)
- [ ] (superseded) 002 (all sessions) per-session spike-count tensors at 5 ms, stimulus-aligned -1.0..+1.0 s, all trials (no-stim aligned to
      their nominal onset), artefact dead zone -1..+4 ms masked (NaN, not Poisson-filled); stored once, re-binned
      as needed

## 2. Pilot (exploration set, a few sessions per cohort)
- [ ] 003 TCA per session: three epoch tensors + one whole-trial tensor
      - baseline -1..0 s (50 ms bins), early evoked 5-50 ms (5 ms bins), late 50 ms-1 s (20-25 ms bins), whole -0.5..1 s
      - normalisation grid: min-max (user) vs soft-normalise (rate / (range + 5 Hz)) vs z-score + unconstrained CP
      - rank 1-12, 10-20 replicates; masked fit for the dead zone; choose rank by error + similarity + held-out trials
- [ ] 003d 50-ms bins (user 2026-11-22, after the pilot rank selection: bsub held-out error > 1 from rank 3-4, train
      error 0.96-0.99 at 10 ms): epochs trial_b50 / late_b50 / whole_b50 (bsub + soft) and jaw_b50, rate over the
      valid part of artefact bins; evoked stays at 5 ms (window is 45 ms). Pilot running (logs/003_pilot_b50.log,
      003b_jaw_b50.log)
- [ ] (on hold 2026-11-22, waiter stopped until the 50-ms rank choice) 003b full run (logs/run_full_bsub_v2.sh): bsub x {trial, evoked, late} + soft x baseline (whole dropped, user 2026-11-21),
      all sessions, starts when the pilot fits end; fits only, no tests on confirmation sessions before approval
- [ ] 003c jaw-onset-aligned TCA pilot (003b_tca_jaw.py, user 2026-11-21): bsub, -0.30..+0.20 s around jaw onset,
      10-ms bins, artefact bins masked (mcp_als); `jaw` = active lick trials, `jaw_all` = + pseudo-aligned trials
      (type-matched median onset); running on haas (logs/003b_jaw_pilot.log). Compare with the stimulus-aligned `trial`
      components (sensory response sits before jaw onset at ~0.1 s RT)
- [ ] 004 sanity: drift control (trial factor vs unit-presence / amplitude), time-in-session vs learning
- [ ] CONSULT: pilot figures, rank / normalisation choice

- [ ] 005-007 event-aligned TCA (user 2026-11-22): 005 builder (stim_long, stim_sensory(10), jaw, lick), 006 fits +
      model comparison (rate vs soft), 007 figures; pilot AB123_20240806_110231 running (logs/006_AB123.log)
- [X] 006g GPU port on Kuma (tensorly), validated on AB123 (006v); 6-mouse pilot fitted (009 summary)
- [X] 6-mouse pilot: z-score; split non-negative main + non-negative on rates as reference (user 2026-11-22)
- [X] full run: 005 rebuild (126 sessions), 006g on Kuma (977 fits), 011 rank per cohort, 012 component table,
      013 recurring clusters; interim summary report/interim_summary_2026-11-22.md
- [ ] CONSULT: rank rule saturates (split jaw / lick at 12); candidate learning / cohort components; next checks
      (reference method, rank stability, lick / reward separation, correction)

## 3. Relating trial factors to the task and learning (exploration set)
- [ ] 005 trial factors vs exact learning curve (whisker and auditory), linear-shift null; partial on trial index
- [ ] 006 trial factors vs task variables: trial type, outcome, passive pre vs post, previous reward, lick / jaw
- [ ] 007 baseline component(s) vs engagement / arousal (auditory hit rate, ITI, pupil / motion where available)
- [ ] 008 evoked 5-50 ms gain per area: passive pre vs post, early vs late active, relative to the learning trial
- [ ] 009 decision components: emergence relative to learning trial, hit vs miss, R+ vs R- shape
- [ ] 010 cross-mouse component matching within cohort: time factor + area-loading profile + trial-type profile,
      Hungarian consensus; area-pooled TCA (neurons -> areas) as a second, directly comparable route
- [ ] CONSULT: candidate effects, lock tests

## 4. Confirmation (only after explicit approval)
- [ ] 011 locked tests on the confirmation set, both populations (all / learners), within cohort and R+ vs R-

## 5. Report
- [ ] report in `combined_results_ks4/ssl-tca-session-wide/report/`

## ON HOLD (user 2026-11-23)
State: two candidate reports built (not published): report/candidate_sensory_cl10/ (sensory Ss-whisker cl. 10, 018)
and report/candidate_go_cl69/ (jaw-aligned go component, 022 / 023). Main report/report.pdf is outdated (valid-time
tensors, old learn_ctrl candidates). Nothing committed.
Resume points:
- [ ] go component: lick vigour covariates (lick count / bout length), paired day 0 vs expert, area breakdown,
      pre-movement-only TCA or dPCA, learning-trial alignment
- [ ] sensory cl. 10: whisking / pupil covariates, cross-validated trend, latency per area
- [ ] passive pre / post (020): R- whisker decrease; extinction link (unrewarded whisker licks), paired expert
- [ ] sliceTCA / full dPCA on Kuma (proposed, not started)
- [ ] rewrite main report (artefact fill, 017 trend measure, 019-023), then publish (ask)
- [ ] commit code (ask): 005, 006g, 011-013 (env options), 015-023, report/build_*.py
