# Question (2026-09-24, user request)

"Look at all learning curves. Can you assess whether the learning trial
seems well positioned? If not, how to improve this learning trial
identification in the context of this HMM learning curve? ... Provide
alternative learning curves if necessary and maybe learning trial
identified, if possible, a bit later in the session."

Side project of `ssl-whisker-hitmiss-timeresolved-decoding` (whose
pre/post-learning-trial hit/miss decoding depends on this trial).

## Current method (read 2026-09-24)
`M:\analysis\Axel_Bisi\behaviour_analysis\beh_plotting_functions.py::
plot_single_mouse_session_learning_curve` (+ `learning_utils.py::
identify_learning_trial_rewarded/_nonrewarded`):
- active trials, early_lick==0 removed; per trial type (whisker, no-stim).
- PyMC state-space model: x_t Gaussian random walk (sigma = 1/sqrt(tau),
  tau ~ Gamma(10,10)), p_t = invlogit(x_t), outcome_t ~ Bernoulli(p_t).
  Sampled with `pm.sample(10, tune=10, chains=4)` = 40 posterior samples
  after 10 tuning steps (!). p_mean, 80% CI from those 40 samples.
- chance = no-stim (FA) curve, same model, interpolated at whisker times.
- learning_trial: trials >= first whisker hit; R+: first 5-run with
  p_low > p_chance ('expert' if 20-run in first 20 trials with mean p>=0.8);
  R-: last trial of the first 5-run above chance followed by 5 not-above
  (interp_fa branch); values <= 10 clamped to 10; R- with no criterion
  -> hardcoded 10.

## Candidate problems to assess
1. MCMC: 10 tune / 10 draws x 4 chains -> unconverged, noisy mean/CI;
   crossings of p_low vs p_chance partly sampling noise.
2. Prior sigma ~ 1 logit/trial -> very wiggly curves; a 5-trial run can be
   a transient excursion.
3. Criterion: only 5 consecutive trials, no sustain requirement; floor at
   10; R- defined as the END of a first above-chance run (not the onset of
   stable withholding) and a hardcoded fallback.

## Definitions
- d(t) = p_whisker(t) - p_FA(t) ("discrimination"); R+ learned = d reliably
  > 0 and stays so; R- learned = whisker licking no longer above FA
  (d not > 0) and stays so, after having been above.
- Exploration set = all ephys learning-stage (day 0) sessions with curve
  files (89); no confirmatory split planned (method-development project).

## 2026-09-25 follow-up (user): fewer conservative LTs, the FA question, clean separation
Q1 "Should FA be used, given R+ increase FA compared to R-?" (`007_fa_behavior.csv`, `009_population_comparison.png` f):
- FA (no-stim lick rate), first 20% vs middle 20-80% of whisker trials: R+ 0.30 -> 0.27 (stays high),
  R- 0.21 -> 0.11 (falls). Whisker: R+ 0.49 -> 0.50, R- 0.32 -> 0.18.
- Consequence: for R+, requiring the whisker-FA DISCRIMINATION to rise under-detects learners whose FA
  stays high or rises with engagement; for R-, FA is essential -- without it a general drop in licking
  (whisker AND FA falling, e.g. MH018, MH034) would be called whisker learning.
- Decision implemented (007): R+ may use a whisker-only rise with FA only as a FLOOR (learned whisker
  rate > FA) -> `L5w_len`; R- keeps the joint whisker+FA criterion. Both kept FA as a floor/check.
Q2 "How to get more mice considered learners?" -- less conservative candidates, in order of preference:
  L6 (strict joint, log10 BF > 0.5) -> L6_len (log10 BF > 0, relaxed checks) -> L5w_len (R+ whisker-only
  rise, FA floor) -> L7 (half-way on smooth curve) = `lt_lenient`: 34/50 R+, 17/38 R- (strict: 17 / 14).
CLEAN-LEARNING NOTE (user: "keeping a cleanliness in the behavioural separation pre vs post learning"):
- Gate applied to every LT candidate: 20 whisker trials before vs after the LT must differ in the learned
  direction by > 0.05 in EITHER whisker lick rate OR whisker-FA discrimination, with posterior P > 0.9
  (Beta posteriors). `lt_lenient_clean` = lt_lenient passing it: 25/50 R+, 14/38 R-.
- The same gate passes only 20/47 R+ and 13/34 R- STORED LTs.
- Trade-off (sessions passing, R+/R-, of 34/17 lenient candidates; "either" criterion):
  W=20: margin 0 -> 31/17 (P>0.9), 33/17 (P>0.8); margin 0.05 -> 25/14 (P>0.9), 32/17 (P>0.8);
        margin 0.10 -> 22/12 (P>0.9), 28/15 (P>0.8).
  Whisker-only criterion (first version, W=30, margin 0.1, P>0.9): 14/13 -- rejects R+ FA-drop learners.
- Caveat: aligning to a change point makes the aligned behavioral step sharp partly by construction;
  the gate checks that the step is real in raw outcomes, not that learning is abrupt.
