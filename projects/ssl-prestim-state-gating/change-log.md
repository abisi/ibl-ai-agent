# change-log

- 2026-11-11: project created (user: "Yes go ahead, do that as a separate project"). 001_state_gating.py: pre-stimulus state
  axis, single-trial gating of the early whisker response, passive state shift and state-controlled passive change.
- 2026-11-11: 001 full runs (day 0 area groups 90 sessions, fine labels, expert); 002_summary.py (stats + figures).
  Day-0 learners, whole brain: state axis exists (bal. acc. − linear-shift null R+ +0.068, R− +0.078, p <= 1e-4), larger at
  expert in R+ (+0.156, day 0 vs expert p = 0.007); no consistent single-trial gating (gate_rho ~0; within misses +0.08,
  within hits R+ −0.10: opposite signs; passive R− +0.10 p = 0.019; auditory control null); no within-day half change.
  Passive: state moves to the miss side after the task for whisker AND auditory trials (R+ −0.45, R− −0.69), the session state
  shift correlates with the passive sensory readout shift (rho 0.48 / 0.47, p ~0.01); controlling for state shrinks the
  readout change R+ −0.17 -> −0.14 (n.s.), R− −0.38 -> −0.28 (paired p = 0.007, still p = 0.005); expert R+ −0.29 -> −0.02
  (fully state). Passive whisker gain on the passive-pre axis drops strongly in both cohorts (and at expert), not explained
  by state. Fine labels: state decodable in SCm, MRN, DMS, CA1 (R−, BH q < 0.05).
- 2026-11-12: 003_figures.py (publication figures: 003_main whole brain A-L, 003_areas / 003_areas_fine forest plots) and
  004_trial_tests.py (user: "explore this hypothesis more"). Day-0 learners, whole brain: P(lick) rises across state quintiles
  (per-session rho R+ 0.50, R- 0.33); the whisker-trained state also separates auditory hits / misses (AUC − 0.5 R+ +0.29,
  n = 6; R- +0.17, n = 9) and predicts the lick beyond pre-stimulus jaw / nose / whisker motion and pupil (Δ AUC +0.10 /
  +0.08): general engagement, not movement. State weakly anti-correlated with pre-stimulus whisking (R+ rho −0.17) and not
  with pupil. Gating: response vs time-residualised state quintile flat within hits and misses; gating controlled for
  movements ~0; cross-area (whole-brain state -> each area's response, area state -> Ss-whisker response) null. Positive
  control: pre-stimulus jaw / nose / whisker motion reduce the early whisker response (rho −0.10 to −0.19, both cohorts),
  pupil +0.07 (R+) / −0.06 (R-). Learning-trial alignment (006 lt_recommended_broad, R+ 17, R- 11 sessions): R- state and
  response within misses drop after the learning trial (state −0.72 p = 0.002; response −0.45 p = 0.006), R+ no significant
  change; not separable from session time.
- 2026-11-12: 005_engagement_movement.py + 006_engagement_figure.py (user: do the joint movement / state model and test the
  state as a slow engagement variable; response dynamics and the passive drop not pursued: the drop is known to be lasting).
  Whole brain, day-0 learners. Choice prediction (CV AUC − 0.5, R+ / R-): time 0.18 / 0.09, movements 0.09 / 0.04, state
  0.20 / 0.10, time + state 0.22 / 0.12: the state's choice information is largely shared with the within-session trend.
  Joint OLS of the early whisker response: jaw (and nose, whisker in R-) negative, pupil + in R+, state ~0. The early
  (5-35 ms) sensory hit / miss readout is anti-correlated with pre-stimulus jaw / nose / whisker motion (rho −0.25 to −0.30
  R+, −0.10 to −0.13 R-) and correlated with the state (0.25 / 0.17); beyond all pre-stimulus variables it adds +0.02 / +0.01
  AUC (n.s.); partial rho with the lick 0.15 (p = 4e-4) in R+, 0.02 (n.s.) in R-. Slow engagement: the mean state of the
  previous 1-10 trials predicts the whisker choice (AUC − 0.5 ~0.04-0.06 vs 0.06-0.10 for the current trial); 10-trial block
  state tracks the block whisker hit rate (partial rho − shift null 0.48 / 0.26, p <= 1e-4) but not the auditory hit rate
  (near ceiling) nor the block whisker response (no slow gating); single-trial autocorrelation after detrending small
  (+0.02 / +0.03 over permutation, tau ~1-3 trials; single-trial scores are noisy); previous lick raises the state (0.10 /
  0.09), previous reward adds nothing.
- 2026-11-12: 005 + 006 linear-shift nulls (user: "do the shift nulls for engagement figure"; whole brain; grey markers, stars
  real vs null). The state's choice AUC is trial-specific (null +0.02 / +0.00 vs real +0.20 / +0.10), session time's is mostly
  slow (null 0.14 / 0.05 of 0.18 / 0.09); past-1..10-trial state beats its null (+0.03 to +0.08); movement -> early readout
  and the previous-lick effect are all above null; nulls of the B / C coefficients ~0. Revises the earlier reading that the
  state's choice information is largely shared with the within-session trend.
- 2026-11-12: 007_orthogonalization.py + 009_orthogonalization_figure.py (user: "test that learning induces orthogonalization
  of the sensory response with that of the response predicting hit/miss using the coding directions"). Cross-validated cosine
  of the sensory CD (passive pre / post, active class-balanced) with the hit / miss CD, early (5-35 ms) and mid (35-75 ms)
  windows; responses scaled but NOT mean-centred (001 centres: its g template is pre-whisker minus the session grand mean,
  which biases the 001/003 passive gain drop g_dW; 007 g_u is the uncentred replacement). Day-0 learners, whole brain: hit /
  miss CD strongly aligned with the sensory CD (R+ ~0.8, R- ~0.4-0.6); no within-session change in R+ (halves, learning trial);
  R- mid window late half drops (-0.68, p = 0.0007; R+ vs R- p = 0.01). Passive post sensory CD less aligned with the hit /
  miss CD than passive pre (R+ -0.23 p = 0.01, R- -0.58 p < 1e-4; R+ vs R- p = 0.002). Passive change on the hit / miss CD
  (hit / miss distance units): R- -0.30 = gain -0.17 + rotation -0.13; R+ -0.02 (gain -0.05, rotation n.s.); relative gain
  change along the pre CD -15 % / -26 %. Expert vs day 0: n.s. (expert n R+ 9-10, R- 5-6). Fine-label run queued.
- 2026-11-12: 008_glm_state.py (user: link the state and the movement effect to the Figure 1 behaviour GLM): GLM v6 predictors
  + neural columns, glm_v5 block CV at the session's C*. Adding the state: dAUC +0.031 / +0.013 (vs shift null p = 1e-4 /
  0.011); adding the early readout +0.033 / +0.023; the state is explained by the GLM predictors (block-CV R^2 0.19 / 0.25;
  cumulative reward fraction -, time since reward -, lick history n-1..3 +, nose / whisker speed - in R+). Mediation: jaw,
  nose and whisker speed reduce the early whisker response (a -0.14 to -0.23) which raises P(lick) (b +0.35 / +0.24): indirect
  effects -0.04 to -0.09, all p <= 0.003, movement GLM weights shrink when the response is added. Fixed a race:
  glm_source_dir() rewrites the patched GLM file on every call, so it is built once and passed to the workers.
- 2026-11-12: project closed (user). Conclusions and open items in question.md (Status); candidate figure added to the
  figure planner (tab Candidate_prestim_state). Fine-label 007 run finished (1,420 / 1,453 rows ok), no fine 009 figure.
