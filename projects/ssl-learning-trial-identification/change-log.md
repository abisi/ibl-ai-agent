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
