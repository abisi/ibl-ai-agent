# TODO (from the 2026-10-16 review of the original model code)

- Fit on the per-trial likelihood of each mouse's own trial sequence (actual stimuli, licks and rewards) instead of MSE on smoothed,
  time-normalised curves: the causal 100-trial smoothing of the targets lags the curves while the simulated curves are not smoothed the
  same way (bias towards slow learning), and normalising to 1000 points discards the number of trials each mouse performed.
- b and v trade off (002 recovers b x v, not each); reparameterise or fix one.
- Per-mouse fits: also fit b (and a, v or tr) per mouse, otherwise inter-mouse differences in learning speed go into the initial weights.
- Reward coding: unrewarded R- whisker licks and false alarms are R = -1 although not punished; compare with R = 0 (matters through a dQ).
- Transfer trial proportions (50 / 40 / 10 %) are hard-coded; take them from the data.
- Nelder-Mead ignores the declared bounds; use bounded optimisation or a transform.
- Original filters behind trials_data_final_filtered.csv (108 mice) unknown; 001 keeps mice with an R+ / R- label in the joint reference
  sheet (29 early mice without a label dropped); confirm with the user.
- Then: per-trial model latents (whisker value, P(lick)) for the neural analyses of ssl-learning-axis-exploration.
