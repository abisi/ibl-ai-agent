# Model v6 — design (2026-10-23)

Follows v5 (model_v5_design.md, 021/022). User 2026-10-23: model predictions are much flatter within session than the data (022_2); what
in the poster model (Matteucci et al., "Mousemind-Q"; model_v3_design.md 2.2) could fix that? Then "yes, but before running the full
thing, let's check predictions".

## Diagnosis (raw data, days >= -4)
Sessions end after 40 (10th pct) to 147 (90th pct) rewards. FA rate by trials to session end: 0.28 until ~80 trials before the end, then
0.18, 0.10, 0.05 (last 10); auditory 0.85 -> 0.46 in the last 10 trials. By rewards received so far the decline is smeared (FA 0.27 ->
0.08); by fraction of that session's rewards it is a plateau (FA ~0.33, auditory ~0.9) then a collapse in the last ~30%. FA in the last
20 trials is 0.05-0.07 whatever the session's total rewards. So every session ends in the same sated state after a different number of
rewards: a single cohort rho (v5) averages these collapses into a flat decline, and E3 exploration (recent reward rate) stays high to
the end because auditory rewards continue.

## Model (rl7_model)
Learning core as v5 best (overlap transfer g, asymmetric alpha, delta rule on licked stimulus trials; FA licks do not change weights).
Choice is logistic with a stimulus-independent thirst-driven lick bias (poster):

  z_t = beta * T_t * V_t - c + B(T_t)          (V = 0 on no-stimulus trials)
  P(lick) = 0.01 + 0.98 * e_t * sigmoid(z_t)
  thirst T_t = max(0, 1 - rho_s * S_t), S_t = satiety (water drunk this session, see absorption), reset each session
  bias   lin: B = s_B * T          thr: B = s_B * softplus(20 (T - theta)) / 20   (plateau, then collapse once T < theta)
  absorption  inst: S_t = rewards so far      abs: S_t = sum_j (1 - exp(-(t - t_j) / tau_a)) (satiation lags drinking)
  engagement  noeng: e = 1                    eng: v5 engagement e = e_min + (1 - e_min)(1 - exp(-Rbar / r_e))
  capacity    pop: one rho per cohort         sess: rho_s per session, log rho_s ~ N(mu_rho, sd_rho^2), integrated out

Per-session rho is a random effect integrated by 11-point Gauss-Hermite quadrature. This is exact: weights, Rbar and satiety depend only
on observed licks and rewards, never on T, so sessions are conditionally independent given the shared parameters. The one-step-ahead
predictive P(lick_t | trials before t in the session) mixes the quadrature nodes by their posterior given the session's earlier trials
(causal, no look-ahead); its log product is the marginal likelihood.

Models <capacity>-<bias>-<absorption>-<engagement> (16). Prediction check (023) on 6 of them, full data, no CV:
pop-lin-inst-noeng (v5-like motivation, reference), sess-lin-inst-noeng (per-session capacity only), sess-thr-inst-noeng (+ poster
threshold bias), sess-thr-abs-noeng (+ absorption), sess-lin-inst-eng, sess-thr-inst-eng (+ warm-up / engagement), plus v5 best.
Predictions shown two ways: causal predictive (data-conditioned through the rho_s posterior) and simulation (rho_s drawn from its
session posterior, learning on simulated licks). Pure prior simulation cannot reproduce the end-aligned collapse because session length
is set by the mouse disengaging (endogenous); a stopping rule would be needed for that.

Not modelled yet: FA-specific learning across days (pretraining FA rise, day-0 FA drop); novelty on W.
