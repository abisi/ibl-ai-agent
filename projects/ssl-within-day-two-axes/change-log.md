# change-log

- 2026-10-16: project created (user 2026-10-15: combined within-day report of the two coding directions, expert sessions as control; jaw-aligned
  versions not used). 001 reaction-time and jaw controls of the reward-lick whisker-hit drift (per-event join of 002 scores, 051 RT and DLC jaw);
  002 movement state around passive trials (pre-stimulus and evoked whisking / jaw, post - pre); 003 classification of the within-session changes
  (learning-specific / stronger with learning / both stages / day 0 only, stage n.s.) for both axes, whole brain and area groups; common.py (paths,
  statistics, panel helper). First report (report/build_report.py on report_lib), replacing the former ssl-within-day-remapping report.
- 2026-10-16: 004 reward-lick time course day 0 vs expert (Figure S1). 005 relative position lambda = (WH - SL) / (AH - SL) from per-class linear
  fits over the session (start, end, change, session mean; per-third display), class slopes and contrasts incl. auditory hits; 006 per-event mixed
  model score ~ tau x WH x expert + (1 + tau | session), references SL and AH, with cohort interaction; 007 WH - SL drift per trial, per whisker hit
  (also first 30), per minute, per reward. Result: the R+ within-session drift is the same at both stages; learning shifts the starting level
  (day-0 start far below the expert level, end close to it; mixed-model start offset expert - day 0 p < 0.001, slope n.s.). Report updated
  (Figures 3-4, S2; Eqs. 3-4). common.dot_panel short class labels.
- 2026-10-16: day-0 specificity tests (user: "try without baseline correction ... run all 6 tests"). 008 per-event table (scores with and
  without per-event baseline correction, same estimator as within_day 002; whisker-trial index; learning curves; learning trial; early / late
  CD stability). 009 baseline comparison (same conclusions; only the auditory-hit drift is baseline-borne; mixed-model expert - day 0 slope
  term fold-dependent). 010 step at the learning trial vs placebo splits (none). 011 coupling to the whisker - FA curve beyond time (none).
  012 mixture of SL-like / AH-like whisker hits (one shifting population; same within-session change at both stages). 013 CD stability early
  vs late (stable; modest R+ day-0 reorganisation n.s.). 014 whisker-trial units (day-0 convergence trend, n.s.). 015 hit - miss axis split at
  the learning trial (repo env; 137b day-0 and expert unit lists; R- change not concentrated at the learning trial; expert control 4 R- / 0 R+).
  016 correlation with the whisker P(lick) curve directly (none for whisker hits; R+ drift continues after the curve plateaus). Report:
  six-test section, Figures 8-14, S3-S8, Table 2, Eqs. 5-6; level wording corrected (day 0 stays below expert at the same point of the session).
  haas: 132 shared module restored (passive copy had become the wrapper), decoding 048 / 115 synced.
- 2026-10-16: 020-023 (separability, lick-aligned trajectories, history / post-lick licking, new axes) moved to the new project
  ssl-learning-axis-exploration (renumbered 001-004; results moved on the NAS). They were never part of the report.
