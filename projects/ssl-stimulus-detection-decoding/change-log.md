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
