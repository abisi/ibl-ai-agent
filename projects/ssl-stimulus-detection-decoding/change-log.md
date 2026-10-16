# Change log

- 2026-10-16 Project created (user): stimulus vs no-stimulus decoding, whisker and auditory targets, task and passive epochs,
  same pipeline as ssl-stimulus-arrival-decoding. Checks: task CRs 32-203 per session (sampled sessions); passive blocks have
  0-1 no-stimulus trials, so passive no-stimulus events = sham onsets 0.8 s before the other modality's stimulus (user);
  linear-shift null only (user); artefact replacement at every event onset. 001 written; 002-009 adapted from arrival
  (labels, linear-shift wording, per-target folders, cross-epoch outputs in <target>_task_vs_passive/). `~/cDet.sh` launched:
  caches, then N = 200 at 100 x 10 for the 4 conditions, then the N sweep.
