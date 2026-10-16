# change-log

- 2026-10-16: project created (user: "For the RL model, make a separate project for that"). Model code vendored from
  github.com/abisi/rl_model_auditory_whisker (branch shared-pretraining, local commits b757cf9, 76cd3e0): rl_independent / rl_shared /
  rl_mixed (paths from common.py, no directory creation at import), rl_data_utils (free licking removed; trim with cumcount for recent
  pandas), rl_joint (joint fit). Changes relative to the original repo: one pretraining parameter set for R+ and R- (user request:
  "model parameters are the same for the auditory pretraining"), pretraining simulated once and both cohorts start the transfer from the
  same state; fix of shared_model._worker calling the undefined _compute_mse (every grid point scored inf); trial table rebuilt from the
  NWBs (the original csv was not in the repository). First joint mean fits (3 models x common / free tr) were started from the repo
  checkout on haas before the move; their outputs are copied to fits/ with the 003_ prefix.
