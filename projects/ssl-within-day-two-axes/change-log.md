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
