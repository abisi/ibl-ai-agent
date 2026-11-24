# Change log

- 2026-09-24: project created (question.md, TODO.md).
- 2026-09-25: 005 R- learner criterion relaxed ("within 0.1 of FA" -> "discrimination at least halves")
  after review showed gradual learners rejected; R+ high_from_start judged on first 20 trials (was the
  best-null first segment, mislabelled late learners). Added L6_lenient (log10 BF > 0).
- 2026-09-25: added L7 half-way rule for gradual learners; final table defines lt_recommended (L6) and
  lt_recommended_broad (L6 else L7).
- 2026-09-25: question.md -- follow-up section (Q1 FA, Q2 more learners, clean-separation note). 007 gate changed from whisker-only (W30, 0.1) to either-criterion (W20, 0.05, P>0.9).
- 2026-09-25: added joint whisker+FA models (lt_joint.py) and 6-session test (011, 011b).
- 2026-09-30: 021_learning_curve_figures.py -- square learning-curve figure per session (exact sigma = 1 re-fit, FA fitted at real no-stim times and interpolated to whisker trials, whisker + no-stim raster, LTs L0/L6/lenient cascade, P(W > FA)), saved as NEW files <mouse>_whisker_0_learning_curves_exact_sigma1.{pdf,png,svg} in combined_results_ks4/<mouse>/whisker_0/learning_curve/ (existing pipeline files untouched).
- 2026-09-30: 021 revised (user): no text header, learning trials hidden, larger labels (8/7 pt), auditory curve added (untrimmed curve trial set, real times -> whisker axis), lick trials as ticks above and no-lick below the curves, P(whisker > FA) as a separate lower panel with labelled axes, one-line legend; all 88 sessions regenerated (same new filenames).
- 2026-09-30: 021 revised again (user): P(whisker > FA) panel removed, single square 3 x 3 in panel, tighter margins, thicker curves/ticks/axes; all 88 regenerated.
- 2026-09-30: 022_behaviour_only_inputs.py -- day-0 NWB inputs for behaviour-only mice (searched NWB, NWBFull, NWB_combined, NWB_Myriam, NWB_ks4; checked against stored curve outcomes): 11 sessions (AB077, AB079, AB096, AB118, AB135, AB137, MH001, MH002, MH019, MH038, MH064); not found/unusable: AB091 (no whisker_0 NWB), AB105 (no NWB matches its stored curve), MH006 (NWB_Myriam file empty, NWB_combined session aborted). 021: legend removed, square axes box, behaviour-only mice included; 99 figures regenerated.
- 2026-09-30: 022 now NWB_ks4 only (user); stored curve = NWB set minus last trial accepted and flagged (AB105 whisker: 66 vs stored 65, no-stim/auditory exact; AB105 has 86 perf==6 trials at the session end, rows 142-152 and 197-271). 12 behaviour-only sessions: AB077, AB096, AB105, AB118, AB135, AB137, MH001, MH002, MH006, MH019, MH038, MH064; AB079/AB091 have no whisker_0 NWB in NWB_ks4. 021: full left spine (not bounded), whisker hit/miss ticks in cohort colour, FA black, CR grey; regenerated.
- 2026-09-30: 023_check_inputs_vs_nwb_ks4.py -- all 100 curve-input sessions (88 ephys from ssl_ephys + 12 behaviour-only from NWB_ks4) match their NWB_ks4 files exactly (counts, outcomes, start times < 1 ms) for whisker, no-stim and auditory trials (artifacts/023_inputs_vs_nwb_ks4.csv). 021: whisker hit ticks in the R+ (rewarded) colour, misses in the R- (non-rewarded) colour; 98 figures regenerated.
- 2026-09-30: 021 colours now defined as GROUP_COLORS (ephys_utilities plotting_utils: rplus #00B400 rewarded whisker, rminus #C800C8 non-rewarded); whisker hit ticks rplus, miss ticks rminus -- same values as the current figures, no regeneration needed.
- 2026-09-30: created combined_results_ks4/{AB096,AB135}/whisker_0/learning_curve (user); 100 learning-curve figures regenerated (whisker hits rplus green, misses rminus purple).
- 2026-11-23: 037_methods_overview.py -- all learning-trial definitions side by side on the same 100 day-0 sessions
  (exact rule of Figure 1, L0, L3, L5, L6, L6 lenient, L7, L8, cascade, LR 033, CP shuffle 034, CP delta 035, L6x relaxed
  036): coverage, LT position, pairwise |dLT| and defined/undefined agreement, implied learner label vs sheet
  (artifacts/037_methods_lt.csv, 037_methods_summary.csv, exploratory-analyses/037_methods_overview.png). R+: methods
  agree on WHERE (median |dLT| 0-4 trials, except LR) but not WHETHER (coverage 17-91 %); only L5 is FA-free. R-: rule
  vs change-point methods disagree by 30-85 trials. Context: user dislikes FA-relative metrics (R+ more rewards -> more FA).
- 2026-11-23: 038_chain_exact_curves.py -- LT chain (003-007, 013) rerun on the CURRENT exact curves (user): inputs from
  the stored exact h5 (combined_results_ks4/<mouse>/whisker_0/learning_curve/*_exact.h5; 100/100 sessions), curves at
  fixed sigma 1 in both 002 slots (refit = stored p_mean, max |diff| 0). Workspace artifacts/038_chain_exact/, table
  artifacts/038_learning_trials_all_methods_exact.csv, comparison artifacts/038_exact_vs_028.csv. Raw-trial methods
  (L0, L5, L6, L6 lenient) unchanged; curve-based change: R+ L3 39 -> 28 LTs, L8 29 -> 35, L7 27 -> 28 (shared LTs
  within ~1 trial); R- L7 16 -> 30, L8 15 -> 35, cascade 18 -> 30. 037 --exact -> 037_methods_overview_exact.png,
  037_methods_*_exact.csv. Must run with the repo venv (.venv; 003 imports ibl_ai_agent).
