# Change log

- 2026-10-16 Project created (user): stimulus vs no-stimulus decoding, whisker and auditory targets, task and passive epochs,
  same pipeline as ssl-stimulus-arrival-decoding. Checks: task CRs 32-203 per session (sampled sessions); passive blocks have
  0-1 no-stimulus trials, so passive no-stimulus events = sham onsets 0.8 s before the other modality's stimulus (user);
  linear-shift null only (user); artefact replacement at every event onset. 001 written; 002-009 adapted from arrival
  (labels, linear-shift wording, per-target folders, cross-epoch outputs in <target>_task_vs_passive/). `~/cDet.sh` launched:
  caches, then N = 200 at 100 x 10 for the 4 conditions, then the N sweep.
- 2026-10-16 010 anatomy link: Liu et al. 2024 (ephys_utilities allen_utils/data/Liu_et_al_Group_averages_ranked.xlsx), ipsilateral,
  wS1 and wS2 injections combined (avg_ipsi = mean of 2 sites x 6 Cre lines; asserted), per unit = deepest Liu target structure
  containing it (CCF annotation + structure tree), per area = mean over units; areas with < 50 % of units mapped excluded
  (hippocampus 4 %, olfactory 1 %). Allen densities at unit positions as before. Figures as results come (user): ~/cDetFig.sh
  watcher runs 005 + 010 after each N = 200 pilot, 006 / 008 / 009 when both epochs of a target exist, 002 + 004 after each sweep.
- 2026-10-16 10:05 Run order changed (user): N = 200 pilot of the 18 area groups for all 4 conditions first, then the 40 fine areas,
  then the N sweep (`~/cDet2.sh`, log ~/log/cDet.log; first run's log ~/log/cDet_first.log, its 5 finished areas kept). ~5.5 h
  per condition for all 58 areas. 006 / 009 / 010 skip a level that has not been decoded yet.
- 2026-10-16 010: source areas included (user: include Ss-whisker). Allen measure per modality = per unit mean of the modality's
  source densities leaving out the unit's own source (SSp-bfd units -> SSs projection, SSs units -> SSp-bfd; AUDp <-> AUD-sec);
  Liu already defined there. Whisker task, area groups (18): Allen early accuracy rho = +0.49 (p = 0.040), onset rho = -0.44
  (p = 0.080); Liu early rho = +0.41 (p = 0.12), onset rho = -0.39 (p = 0.14).
- 2026-10-16 13:40 Whisker passive (area groups) done. 006 (mkdir parents) and 009 (levels not decoded yet skipped) fixed, whisker
  task-vs-passive rerun by hand. Passive vs task: same onsets (median diff 0 ms, n = 12, Wilcoxon p = 0.91), early accuracy
  lower in passive (median -0.027, p = 0.003), late decoding (> 150 ms) much lower in passive (no lick / reward); pre-stimulus
  corrected accuracy above task in passive (median +0.011, p = 0.007; midbrain +0.04, pallidum +0.065): candidate cause =
  fixed 3-s passive ITI (stimulus windows end at a predictable stimulus, shams 0.8 s earlier; time since previous stimulus
  ~2.85 vs ~2.05 s).
- 2026-10-20 06:31 All runs done (`~/log/cDet.log`: N = 200 for both levels and 4 conditions, N sweep); the figure watcher finished
  (`~/log/cDetFig.log`; the 006 / 009 errors of 10-16 were fixed in ea05277, later runs clean).
- 2026-10-23 User: 100 iterations x 10 shifts are the final repetitions for this analysis (no 500 x 20 run; 005 / 007 not needed).
- 2026-10-23 010 figures (user): OLS line of each measure on log10 projection with 95 % CI (solid only if p < 0.05), Pearson r
  on log density + Spearman rho in each panel title, unreliable onsets drawn open (not tested); area-group labels in the
  area's colour; fine areas: legend of the group colours. Tests unchanged.
- 2026-10-23 010 (user): Liu et al. do not mask their injection sites, so at the fine-area level SSp-bfd units take the wS2
  value only and SSs units the wS1 value only (as the Allen leave-own-source-out); area groups keep the grouped wS1 + wS2 value
  (`liu_ipsi_grouped` saved too). Injection sites ringed in every panel (Ss-whisker / SSp-bfd / SSs; Auditory / AUD; Liu
  figures: whisker sites); grouped injection sizes under each figure (Allen: whisker 89 exp., 27 lines, median 0.092 mm^3;
  auditory 31 exp., 20 lines, median 0.085 mm^3; Liu: 37 x 25 nl, 6 lines). Liu vs Allen whisker input across areas:
  fine rho = 0.52 (p = 0.001), groups rho = 0.47 (p = 0.07); main disagreements DMS / VS / MO-ALM (Liu: one CP / frontal value)
  and RT / GPe / VT / SSp-oro (Allen high).
- 2026-10-23 First report (report/build_report.py + render.sh; skills/project-report): passive, task, comparison, with the
  whisker and auditory targets side by side; onset vs early information; presence vs identity onsets (identity project N = 200
  finals); anatomy (Allen + Liu, specificity control). 7 main + 9 supplementary figures, 4 + 4 tables, 3 equations, claims
  registry, frozen statistics (report/_snapshots/2026-10-23/), summary.md. Not included (in the result folders): summary
  figures, per-N curves, area rankings, the other conditions' N effect. Matched accuracy (003) pending (~/cDetMatched.sh);
  the report says so and adds Figure 6 / Table 5 when matched_accuracy_check.csv exists. Published to the private reports repo.
- 2026-10-24 Matched-accuracy runs (003) finished; report rebuilt with Figure 6 / Table 5, onsets at the matched N added to Table 5 and summarised in the text (computed: onset range narrower at matched accuracy in 5 of 6 target x reference comparisons, up to 12 ms left vs 26 ms at N = 100; Midbrain earliest in 5 of 6). Matched control removed from the open questions. Published (reports repo bea67d0).
