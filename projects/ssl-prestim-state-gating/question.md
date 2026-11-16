# ssl-prestim-state-gating

## Question (user, 2026-11-11)
An expert's hypothesis, paraphrased by the user: an attentional decision variable, a pre-stimulus brain state that predicts
whether the mouse will detect (lick to) the whisker stimulus, changes with learning and sets the strength of the sensory
response. How to test it, and does it explain the passive post − pre change of the whisker response on the hit − miss axis
(ssl-hitmiss-axis-passive 135 / 146 / 179: after the task, passive whisker responses move to the miss side in both cohorts)?

## Starting evidence (ssl-hitmiss-axis-passive 178 / 179, 2026-11-11)
- Hit / miss is decodable from the pre-stimulus window (-200 to -10 ms) beyond the linear-shift null: whole brain +0.05
  balanced accuracy in R+ and R- day-0 learners (about half the 0-50 ms value), strongest in midbrain and motor areas.
- Across area groups, neither early (0-50 ms) nor pre-stimulus decoding predicts the passive post − pre change.

## Tests (predictions of the hypothesis)
1. State axis: a pre-stimulus hit / miss axis exists (cross-validated decodability beyond the linear-shift null), per area.
2. Gating: within session, the trial's pre-stimulus state score predicts the size of its early whisker response (5-35 ms,
   projection on the session's passive-pre whisker response), beyond choice and session time; also on passive trials (no
   choice) and within hits / within misses; linear-shift null.
3. Learning: state decodability and the gating slope grow with learning (active halves, learning-trial splits, day 0 vs
   expert), possibly more in R+.
4. Passive change: passive post sits on the miss-state side (disengagement / satiety after the task) and the state shift
   accounts for the passive sensory readout shift; the post − pre change controlled for state shows what is left as
   sensory plasticity.

## Conventions
Tracked stable units (137b, as 133-146), trials as 132 / 146 (active from the first whisker trial, perf != 6, A1 trimmed;
early licks before 35 ms dropped), session = unit, R+ vs R- Mann-Whitney + Welch, Wilcoxon vs 0 within cohort, linear
(non-wrapping) shift nulls only. Areas: whole brain, area_group, fine custom acronyms (area_acronym_custom).

## Results
combined_results_ks4/ssl-prestim-state-gating/{tables,figures}.

## Status: CLOSED 2026-11-12 (user: "Ok let's close the gating project")
Answer to the hypothesis, day-0 learners, whole brain:
1. State axis: yes. A pre-stimulus state predicts the whisker choice trial by trial (beyond the linear-shift null and beyond
   pre-stimulus movements and pupil); it is general engagement (also separates auditory hits / misses), drifts slowly within
   the session, adds to the Figure 1 behaviour GLM (dAUC +0.031 / +0.013) and is partly explained by its predictors.
2. Gating: no. The state does not set the early whisker response size; pre-stimulus movements do (jaw, nose, whisker speed
   reduce it), and that reduction mediates part of their effect on P(lick) (008).
3. Learning: state decodability higher at expert in R+; within day 0 no consistent change (R- learning-trial effects not
   separable from session time).
4. Passive change: the state moves to the miss side after the task and explains part of the readout shift (all of it at
   expert R+); the passive whisker gain drop is not explained by the state. In R- the passive whisker response stops pointing
   along the hit / miss direction (cos 0.54 -> -0.04; gain -0.17 + rotation -0.13): re-mapping / decoupling, not an
   orthogonalisation of the choice signal (the within-session R- hit / miss separation shrinks rather than rotates).
Candidate figure: planner tab Candidate_prestim_state (19 panels). Not done (pick up only if reopened): rebuild 003 panel L
with the uncentred gain (007 g_u; 001 centring biases it negative); fine-area 009 figure (007_orthogonalization_fine exists);
late R- whisker licks vs spontaneous licks; projection on the early-half hit / miss CD; more expert sessions.
