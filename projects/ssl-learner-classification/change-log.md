# Change log

- 2026-11-13: project created (user request: agnostic learner classification from exact curves, learning-trial
  definitions and behaviour clustering; supplementary figure and report). 001 classification (nine criteria, consensus,
  user constraint on the R+ bad mice), 002 supplementary figure (SuppFigure_learners), report v1. tslearn 0.7.0 installed
  for the DTW criterion in ~/pylibs/tslearn on haas (PYTHONPATH; not in the shared venv).
- 2026-11-13: report v1 built (render.sh) and published to abisi/ibl-ai-agent-reports docs/ssl-learner-classification
  (pdf, md, html). Supplementary figure snapshot: manuscript Figures/SuppFigure_learners_2026-11-13 (not in the planner yet).
- 2026-11-19: 003 per-mouse day-0 exact learning curves of every sheet 'bad' mouse and every sheet learner the agnostic label calls a non-learner (figures/003_mouse_curves.png/pdf).
- 2026-11-23: peak within-session d' added (user: candidate learner measure). 001: peak_dprime = max over whisker
  trials of z(whisker curve) - z(chance curve) (exact curves, clipped 0.01-0.99; from the cached fits), peak position,
  end d'; labelling peak d' >= 1 (R+ only, NOT in the consensus: agnostic labels unchanged, checked for all 101 mice),
  in agreement / validation / simple-rule / comparison tables; threshold sweep tables/001_peak_dprime_sweep.csv.
  002: panel A row (continuous peak d'), B / C rows, new row H (peak d' per mouse, lines 1 / best kappa 1.75 / R- 95th
  pct 2.62), I (R+ threshold sweep), J (peak d' vs day +1 d'). 003: peak d' in titles, dotted line at the peak trial.
  Results: peak d' >= 1 in 56/59 R+ (9/12 sheet-bad); R- (no whisker reward) median 1.8 (95th pct 2.62); best kappa
  with the agnostic label at 1.75 (0.65), with the sheet 0.52; R+ rho with day +1 d' 0.59.
  Supp figure snapshot Figures/SuppFigure_learners_2026-11-23.
- 2026-11-23: 004_rminus_null_rule.py (recommendation 1, user: FA-relative metrics disfavoured since R+ get more rewards
  and false-alarm more): R+ learner = day-0 statistic above the R- 95th (90th) percentile; FA-relative (frac_above,
  peak d') and whisker-only statistics (W P(lick) 2nd half, sustained 20-trial W P(lick), fraction of W trials above
  the pooled R- whisker-curve 95th pct = 0.73). Result: R- whisker licking on day 0 is high (generalisation), so the
  R- null is strict: 14-35 R+ learners of 59 (sheet 47); best whisker-only rules: fraction above R- level (34, kappa
  consensus 0.58, sheet 0.37) and sustained W P(lick) at 90th pct (30, kappa consensus 0.66, AUC day +1 d' 0.83).
  Tables 004_rminus_rule_*.csv, figure figures/004_rminus_rule.png. Learning-trial definitions to be revisited (user).
- 2026-11-23: 005_known_bad.py (user: how to isolate the known R+ non-learners?): AUC / best threshold of every day-0
  feature and LT-definition indicator, known bad (7 fixed) vs good / moderate (47), flexible mice held out; pairs of
  features. Sustained whisker P(lick) (best 20-trial window of the exact whisker curve) <= 0.37 separates them perfectly
  (FA-free); next: day-0 W P(lick) (2 confusers), fraction above chance (3). Flexible mice all above the threshold
  (MH007 0.42 closest). Tables 005_known_bad_*.csv, figures/005_known_bad.png.
- 2026-11-23: curve check: 001 whisker curves identical to the stored exact curves (combined_results_ks4/<mouse>/
  whisker_0/learning_curve/*_exact.h5, 2026-11-03) for all 101 mice; chance curve differs at the session start only
  (Figure 1 day-0 warm-up cut); LT 89/90 identical. (ssl-learning-trial-identification still uses its own 2026-10-01
  fits: same trials, different curves.)
- 2026-11-23: 006_sustained_metrics.py (user: develop the 005 metrics): best k-trial window (k 10 / 20 / 30) of the
  exact whisker curve (S_mean), its lower 80 % bound (S_low), raw licked fraction (S_raw), whisker / auditory ratio
  (S_rel), from the stored exact h5. Known bad vs good / moderate: S_raw_20 and S_mean_20 separate perfectly (raw:
  <= 8 of 20 consecutive whisker trials licked, margin +0.21 SD, LOO 1.00); k 10 / 30, lower bound and the auditory
  ratio leave 1-7 confusers (AB086, AB145 first). Flexible mice all above at k 20 (MH007 0.50 raw); 50 % of R- mice
  also above (generalisation). Tables 006_sustained_*.csv, figures/006_sustained_metrics.png.
- 2026-11-23: fairness check (user: the 005 / 006 sustained metric is not matched for R-): the same rule (>= 9 of 20
  consecutive whisker trials licked) removes 7/59 R+ (all sheet bad) but 21/42 R- (all sheet good: in R- not licking
  the whisker is the target behaviour). R+ "not engaged" mice reach whisker P(lick) 0.56 / d' 0.97 on day +1 (slow
  learners). Conclusion: cohort-asymmetric selection; not for R+ vs R- comparisons (see chat for the fair options).
- 2026-11-23: 007_rminus_rule.py (user: two rules per cohort; what rule for R-?): R- candidate metrics from the exact h5
  (late sustained whisker licking, end-of-session whisker lick fraction, late W - FA, credibly-above-chance fraction in
  the second half, specific suppression window with auditory still licked). No day-0 R- rule predicts day +1 whisker
  P(lick) (all MW p > 0.4): day +1 R- whisker licking is low for nearly every mouse (median ~0.12; MH037 0.37 the
  exception). MH064 is the only clear day-0 non-learner (late licking 17/20, last 20 W trials 55 %); MH037 borderline on
  day 0 (5/20 at the end) but still licking on day +1. Tables 007_rminus_metrics.csv / 007_rminus_rules.csv, figure
  figures/007_rminus_rule.png.
- 2026-11-23: session-end cut (user: whole session or last 10 % removed?): 001 / 004-007 used the WHOLE session.
  exact_cut.py: day-0 curves refitted on the session cut at first whisker trial + keep x (last - first whisker trial)
  (paper rule keep 0.85, Figure 1 session_end_time; keep 1.0 reproduces the stored p_mean / p_low exactly). 006 and 007
  take --keep (default 0.85), outputs suffixed _keep85 / _keep90; 007 rewritten day-0 only (no day +1 columns).
  R+ sustained metric unchanged by the cut (same separation, S_raw_20 threshold 0.40). R- with the 15 % cut: the end
  licking rises once disengagement is removed (MH064 last 20 W 70 %, MH062 45 %, MH022 35 %, MH037 30 %, AB133 30 %);
  end-of-session <= 25 % flags AB133, MH022, MH037, MH062, MH064 (both sheet-bad included); <= 30 % flags MH022, MH062,
  MH064. 10 % cut: fewer (MH062, MH064 at <= 25 %).

## 2026-11-23 -- 008 border review of the sustained-licking rule
- `exploratory-analyses/008_border_review.py`: one panel per R+ mouse with S_mean_20 (006, keep 0.85) in [0.25, 0.55]
  (12 mice): refitted whisker curve + 80 % band, auditory and no-stim curves, raw outcomes (dropped tail grey), best
  20-trial window, threshold 0.374. Outputs tables/008_border_review.csv, figures/008_border_review.png|pdf.
- Reading: the gap at the threshold is 0.02 (AB117 0.37 vs AB086 0.39). AB086 and AB145 (sheet good / moderate) look like
  the known-bad pattern (one transient burst, or whisker at the no-stim level, day +1 W 0.29 / 0.14); MH004, AB130, MH032
  hold the whisker curve above the no-stim curve over many trials.

## 2026-11-23 -- 009 neural sensitivity of Figure 4 K to the learner definition
- `exploratory-analyses/009_neural_sensitivity_fig4K.py` (reads ssl-hitmiss-axis-passive 135 / 169 tables, nothing recomputed;
  reproduces Figure 4 K sheet-learner means exactly): all 57 day-0 sessions with the K values (32 R+ / 25 R-). Figures 009_
  effect_vs_behaviour, correlation_matrix, populations (6 learner definitions x z-scored / raw-rate axes), influence (leave one
  mouse out, bootstrap), controls (n units, response size, trials, OLS). Tables 009_*.csv.
- AB086 has no day-0 session in the K set; AB117, AB142, AB151 (known bad), MH026, AB141 (flexible), MH037 (R- bad) do.
- Whisker projection cohort difference holds or strengthens under every definition that keeps n (all day-0 mice MW p 0.005);
  weakest for the smallest sets (clear learners 22/21: MW 0.08). Interaction p <= 0.03 everywhere. No single mouse flips it.
- Angle: cohort difference not robust to confounds (OLS with n units + rho_pre + n whisker: p 0.28 all mice).
- R-: q_W correlates with whisker licking (early_w rho -0.68, d' -0.70; partial on rho_pre still p 0.007 / 0.014); R+: none.
