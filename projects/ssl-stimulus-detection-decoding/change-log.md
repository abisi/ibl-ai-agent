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
