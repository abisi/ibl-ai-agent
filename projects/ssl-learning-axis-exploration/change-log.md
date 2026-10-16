# change-log

- 2026-10-16: project created (user: "put all these tests in a separate project for cleanliness"): ssl-within-day-two-axes 020-023 moved
  here as 001 separability, 002 lick-aligned trajectories, 003 trial history and post-lick licking, 004 new axes (reward expectation,
  outcome, alignment, orthogonalised CD, motor axis); outputs renamed 001-004 and moved to combined_results_ks4/ssl-learning-axis-exploration;
  common.py re-exports the within-day helpers and reads the within-day 008 per-event table.
- 2026-10-16: 005-007 added (user: "Do 3 ... Do 4 as well"). 006 first run added an RT >= 250 ms hit subset (expert hits are faster, so
  the pre-lick window is closer to the lick). 007 first run (no detrending) gave a drift PC1 with r(drift, time) ~ 0.97, collinear with the
  time nuisance (logistic coefficients unstable); drift redefined as the detrended slow fluctuation, cubic trend reported separately; the
  auditory-lick test needs >= 5 misses (dropped from the figure). Runs need OMP_NUM_THREADS=1 (BLAS oversubscription on haas).
