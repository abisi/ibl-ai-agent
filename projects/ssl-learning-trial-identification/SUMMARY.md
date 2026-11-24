# Learning-trial identification — overnight review (2026-09-24/25)

Scope: all 98 stored whisker day-0 learning curves (`combined_results_ks4/<mouse>/whisker_0/learning_curve/`), and the 88 ephys learning sessions (exact trial data available) in depth. Method code reviewed: `M:\analysis\Axel_Bisi\behaviour_analysis\beh_plotting_functions.py::plot_single_mouse_session_learning_curve` + `learning_utils.py::identify_learning_trial_*`.

## Is the stored learning trial well positioned? Often not.
1. **Transient placements.** In 36/91 sessions the LT criterion (R+: p_low > p_chance; R-: whisker ≤ FA) holds on fewer than half of the 20 trials after the LT (R+ mean 0.71, R- 0.43) — the 5-consecutive-trial run is frequently a short excursion (`000_all_curves.png`, `artifacts/000_inventory.csv`).
2. **Hard-coded values.** In the 88 ephys sessions, 18/47 R+ and 6/34 R- stored LTs are exactly 10 (the `<=10 -> 10` clamp, plus 4 R- "no criterion -> 10" fallbacks). Several R- LTs sit on isolated late blips (AB093 118, AB139 117, AB140 173, MH014 144) in mice that never licked the whisker above FA.
3. **Root causes in the curve estimate:**
   - **Unconverged sampler**: `pm.sample(10, tune=10, chains=4)` -> 40 samples; R-hat 1.4–2.0, ESS 12–17 (`001c_validation.png`). The posterior MEAN is close to the exact answer (MAE 0.015), but the 80% band (which defines "above chance") is noisy.
   - **Prior forces wiggly curves**: tau ~ Gamma(10,10) -> sigma ≈ 1 logit/trial. With sigma chosen by the data, R+ sessions want 0.49 and R- 0.14 (median); Bayes factor > 20:1 (log BF > 3) for the smoother model in 30/88 sessions, never the reverse (`artifacts/002_summary.csv`).
   - **FA interpolation bug**: `learning_utils.interp_fa_curve` places the no-stim curve on an evenly spaced time grid instead of the actual no-stim trial times -> the "chance" line is time-shifted by 0.06 on average in R+ (up to 0.29).
   - **Whisker-only logic misses FA-drop learners**: several R+ mice lick the whisker from trial 1 and "learn" by FA dropping (MH028, MH030, MH070, MH011).
   - **Many sessions have no learning event** but still get an LT (flat curves, never-licked R-, general licking decline where whisker and FA fall together, e.g. MH018/MH034).

## Alternative curves (recommended)
Exact grid (HMM forward–backward) inference for the SAME state-space model (`exploratory-analyses/lt_lib.py`): deterministic, seconds per session, validated against converged PyMC (MAE ≤ 0.005). With the random-walk step size chosen by the data ('eb' prior), curves are much smoother and interpretable (`006_final_review_Rplus.png`, `006_final_review_Rminus.png`; thin line = stored curve). FA curve placed at actual no-stim times.

## Alternative learning trial (recommended)
- **L6 — joint whisker + FA Bayesian change point** (`005_joint_changepoint.py`): left-to-right segment model (R+: naive -> learned -> optional end decline; R-: generalising -> learned), whisker AND no-stim rates per segment, integrated analytically. Gives a posterior over the LT (median + 90% CI), a Bayes factor for "is there a transition", and a category (learner / high_from_start / never_licked / no_clear_transition).
- **L7 — half-way rule on the smooth discrimination curve** (whisker − FA), for gradual learners.
- L6 and L7 agree closely where both apply (R+ median |diff| 1 trial, 14/14 L7 inside L6's CI; R- 3 trials, 9/13).
- **Later in the session, as asked**: where both old and new exist, new LT is later in 12/17 R+ (median +4 trials) and 8/12 R- (median +11.5); no floor at 10.
- Transitions are sharper at the new LT (model-free whisker−FA rate, 20 trials after − before): R+ +0.40 vs +0.21 stored; R- −0.27 vs −0.14 stored (`005_rule_summary.csv`, `006_rule_summary.csv`).
- Coverage: an LT is defined for 17 (L6) / 26 (L6 else L7) of 50 R+ and 14 / 17 of 38 R- sessions; the rest have no identifiable learning event (categories in the table). **This is the main cost**: fewer sessions for pre/post analyses, but those dropped had LTs that were arbitrary.
- Table: `artifacts/006_learning_trials_final.csv` (per session: stored LT + source, L6 median/CI/BF/category, L6_lenient, L7, `lt_recommended`, `lt_recommended_broad`, evaluation metrics).

## Suggested changes to the pipeline code
1. Replace `pm.sample(10, tune=10)` by the exact grid fit (`lt_lib.fit_curve`, drop-in: same model) — or at least `pm.sample(1000, tune=1000, target_accept=0.95)`.
2. Let the data set the smoothness (broad prior on sigma, e.g. log-uniform 0.02–1.5 or HalfNormal(0.3)) instead of Gamma(10,10) on tau.
3. Interpolate the FA curve at the actual no-stim trial times.
4. Define learning on discrimination (whisker − FA), with a posterior/CI and an explicit "no transition" outcome, instead of a 5-trial run + floor + fallback.

## Impact on the hit/miss pre/post decoding
Same pipeline as the redo (`ssl-whisker-hitmiss-timeresolved-decoding/095`: size-matched, session-level
shift null, disengagement dropped, whole brain), only the learning trial changed
(`100_new_learning_trial_decoding.py` -> `figures/whole_brain/learning/100_hitmiss_newlt_whole_brain_{all,learners}.png`,
`100_new_learning_trial_stats.csv`). Matched accuracy minus shift null, pre -> post:

| LT | cohort | window | n | pre -> post | W p / t p |
|---|---|---|---|---|---|
| stored | R+ | sensory 5-50ms | 41 | 0.080 -> 0.087 | 0.72 / 0.79 |
| **L6 (recommended)** | **R+** | sensory 5-50ms | 14 | **0.065 -> 0.177** | **0.003 / 0.002** |
| stored | R- | sensory - baseline | 27 | 0.089 -> 0.026 | 0.012 / 0.009 |
| **L6** | **R-** | sensory - baseline | 13 | **0.148 -> 0.060** | **0.017 / 0.012** |
| L6 else L7 (broad) | R+ | sensory | 21 | 0.109 -> 0.145 | 0.29 / 0.33 |
| L6 else L7 (broad) | R- | sensory - baseline | 16 | 0.143 -> 0.058 | 0.013 / 0.010 |

R+ vs R- on the change with L6: sensory MW p = 0.014 / Welch 0.009; sensory - baseline 0.006 / 0.008.
Baseline window: no significant change for any LT version. With a cleanly identified learning trial,
R+ hit/miss decodability in the sensory window RISES after learning (masked with the stored LT), and the
R- loss of evoked hit/miss information is preserved. Caveats: small n (14/13), sessions selected as
clear step learners (selection could correlate with decodability), exploratory, uncorrected across
windows/versions; the broad set (adds L7-only, lower-confidence LTs) dilutes the R+ effect.
