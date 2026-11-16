# TODO -- learning-trial identification (side project, autonomous overnight 2026-09-24/25)

- [X] 000 inventory (`000_inventory_and_placement.py` -> `artifacts/000_inventory.csv`, `000_all_curves.png`):
      98 curve files, all with 40 posterior samples; LT criterion holds <50% of next 20 trials in 36/91.
- [X] 001 inputs (`001a_prepare_inputs.py` -> `artifacts/001_inputs.pkl`, 88 sessions, outcomes verified
      identical to stored); PyMC validation (`001c_validate_grid_vs_pymc.py` -> `001c_validation.png/.csv`):
      stored 10/10 sampler R-hat 1.4-2.0, ESS 12-17; converged PyMC matches exact grid (MAE <= 0.005).
      (`001b_refit_pymc_convergence.py` written but superseded by the exact grid method -- not run.)
- [X] 002 exact curves (`lt_lib.py`, `002_exact_curves.py` -> `artifacts/002_curves.pkl`, `002_summary.csv`):
      data prefer sigma 0.49 (R+) / 0.14 (R-) vs prior ~1; log BF > 3 in 30/88; FA even-grid interpolation
      off by 0.06 (R+ mean), up to 0.29.
- [X] 003 rules L0-L4 (`003_learning_trial_rules.py` -> `artifacts/003_*.csv`, `003_rule_summary.png`).
- [X] 004 whisker-only Bayesian change point L5 (`004_bayesian_changepoint.py`) -- misses FA-drop learners.
- [X] 005 joint whisker+FA Bayesian change point L6 (`005_joint_changepoint.py` -> `artifacts/005_*`,
      `005_review_grid_*.png`); categories revised twice after review (R- relaxed, R+ high_from_start on
      first 20 trials) -- see change-log.
- [X] 006 half-way rule L7 + final table (`006_halfway_rule_and_final_table.py` ->
      `artifacts/006_learning_trials_final.csv`, `006_final_review_*.png`): L6/L7 agree (R+ median |diff| 1,
      14/14 inside L6 CI; R- 3, 9/13).
- [X] 007 hit/miss LT-split decoding with new LTs (hitmiss project 095 with SSL_LT_COLUMN; comparison
      `100_new_learning_trial_decoding.py`): L6 LT -> R+ sensory 0.065->0.177 (p=0.003, n=14); R- s-b
      0.148->0.060 (p=0.017, n=13); R+ vs R- p<=0.014. See SUMMARY.md.
- [X] Morning summary (`SUMMARY.md` + chat).
- [ ] CONSULT USER: adopt L6/L7 + exact curves? port into behaviour_analysis? placebo test with new LTs?
- [X] 2026-09-25: 007 lenient LTs + clean gate + FA question; 008 process figure; 009 population + aligned
      performance; 010 lenient review grids. Report: ../ssl-whisker-hitmiss-timeresolved-decoding/report_learning_trial_split.html
- [X] 2026-09-25: joint whisker+FA models (`lt_joint.py`: continuous g+d and switching; exact 2-D grid), synthetic
      recovery test (switch at 120 -> 120 [111,123], log10 BF 7.3; null data -0.7), 6-session test
      `011_joint_model_test.py` -> `011_joint_model_test.png/.pdf`, `011_joint_model_summary.csv`; PyMC check
      `011b_joint_pymc_check.py` -> `011b_joint_pymc_check.png`, `artifacts/011b_pymc_check.csv` (converged; AB107
      |diff| g 0.04 / d 0.02 logit; MH028 d 0.40, AB082 d 0.28 -- coarse 6-value sigma grid + weak d
      identifiability near ceiling). Finding: logit-additive d treats co-moving whisker+FA near 0/1 as general
      licking (MH028 FA-drop, AB085 R- decline not "whisker learning"); probability-scale dp from the joint
      posterior restores agreement with L6 (AB119 47 vs 49, MH028 29 vs 27). Switching model weak in 4/6.
- [ ] CONSULT USER: scale for "learning" (logit d vs probability dp); switching vs continuous; run on all sessions?
- [X] 2026-09-25: 012 smoothness (sigma per session vs shared; cohort comparison + lick-rate confound) and change point
      explained; 013 pipeline + rule schematics (synthetic R+/R-), 8 real examples, all-methods table incl. L8 fixed
      margin (`artifacts/013_learning_trials_all_methods.csv`, `013_methods_summary.csv`); 014 -> `report.html`
      (all methods, joint model as appendix, not adopted).
- [ ] CONSULT USER: single LT rule (L7 half-way vs L6 change point) + gate; then rerun decoding.
