# Model v4 — design (2026-10-23)

Goal (user, 2026-10-23): a simple network RL model in the style of the Bathellier-lab models, covering the whole auditory -> whisker
training; the readout is the synaptic weights (node -> lick) over training, compared across a small set of models. Changes vs v2/v3,
following 014-017:
- per-mouse fitting and agent labels per mouse are not identifiable (015-017) -> parameters are fitted at the cohort level (pooled
  likelihood of all mice of a group), and weight trajectories (not model labels) are what recovery has to return;
- motivation dominates within-session dynamics (014) -> a single thirst term is always in;
- false alarms (FA, = spontaneous licks on no-stimulus trials) come from an exploration term combined with the stimulus drive as an OR, so
  that the FA rate can fall while auditory licking stays high (user: "framing an exploration term that only affects false alarm rates");
  bias learning (v2 kappa) is kept only as the comparison model;
- hypothesis (user): R- learns fast, R+ slowly over days -> asymmetric learning rates vs between-session consolidation.

## 1. Network and choice

Nodes: S (any stimulus, active on auditory and whisker trials), A, W -> lick. Weights start at 0 (no stimulus knowledge).

  V_t(s) = gamma * w_S + (1 - gamma) * w_s          s in {A, W};  no value-driven licking on no-stimulus trials
  drive  = sigmoid(beta * (T_t * V_t(s) - c))       (0 on no-stimulus trials)
  P(lick) = (1 - lapse) * [1 - (1 - eps_t) * (1 - drive)]      lapse fixed 0.02

gamma: auditory -> whisker generalisation (share of stimulus learning credited to S). c: lick threshold (free). T_t: thirst.

## 2. Learning (on licked stimulus trials only)

  delta = o - V,   alpha = alpha+ if delta > 0 else alpha-
  w_S += alpha * gamma * delta,   w_s += alpha * (1 - gamma) * delta

Outcomes as v3: auditory lick +1 (water), whisker lick +1 (R+, water) / -1 (R-, no water), FA lick -0.5 (used only by E2 and B).

Learning-rate variants: sym (alpha+ = alpha-) and asym.

Timescale variants:
- 1ts: weights carry over unchanged between sessions.
- 2ts (consolidation): w = w_fast + w_slow; within a session updates go to w_fast; at every session boundary
  w_slow += k_c * w_fast, w_fast = 0. k_c < 1: overnight forgetting; k_c = 1: same as 1ts; k_c > 1: offline gain (bounds 0-2).

## 3. Motivation

T_t = max(0, 1 - rho * rewards so far in the session), reset to 1 at each session start (v3 decision). Scales the learned value
(sec. 1) and, in E0-E2, the exploration term.

## 4. False alarms: exploration variants

- E0 constant:      eps_t = eps0 * T_t
- E1 decay:         eps_t = [eps_inf + (eps0 - eps_inf) * exp(-n_t / tau)] * T_t       n_t = training trials so far (all sessions)
- E2 FA-learning:   eps decays only after FA licks: eps <- eps_inf + (eps - eps_inf) * (1 - alpha_eps); eps_t = eps * T_t
- E3 reward rate:   eps_t = eps_inf + eta * Rbar_t,  Rbar <- Rbar + alpha_R * (water_t - Rbar) every trial (reset to 0 at session start)
- B  bias learning (v2-like, comparison): eps_t = eps0 * T_t fixed, plus a learned bias w_b entering every trial:
     drive = sigmoid(beta * (T_t * (V_t(s) + w_b) - c)) for all trial types (no-stimulus: V = 0);
     w_b += alpha * kappa * delta after every lick (FA outcome -0.5)

## 5. Fitting and comparison

- Pooled maximum likelihood over all mice of a group (L-BFGS-B, multi-start). Two groupings: shared (one parameter set for R+ and R-;
  cohorts differ only through the outcome) and cohort (separate R+ and R- parameter sets).
- Models: 5 FA variants x {sym, asym} x {1ts, 2ts} = 20, each shared and cohort -> 40 fits.
- Comparison: 5-fold cross-validation over mice (held-out mice' log-likelihood per trial), plus BIC on the full fit.
- Readout: weight trajectories (w_S, w_A, w_W, effective V(A), V(W)) and eps over training, per cohort.

## 6. Recovery (before interpreting real fits)

Simulate the 111 mice (their actual trial sequences, licks and outcomes simulated) from the cohort-level fits of selected generators;
refit all candidate models. Check: (1) model recovery (BIC on pooled fits), (2) parameter recovery, (3) weight-trajectory recovery
(correlation and error of the cohort-mean V(W) / V(A) / eps trajectories true vs refit), (4) recovery of the R+ vs R- difference.

## 7. Data check: session boundaries (018)

Per mouse and boundary (day d -> d+1, d >= 0): lick rate in the last vs first 60 trials of consecutive sessions, per trial type.
Offline gain of whisker licking = start(d+1) - end(d), compared with the same quantity for auditory (thirst control) and FA. Predicts
2ts (k_c > 1, R+) vs asymmetric-only learning (no step at boundaries beyond the thirst reset).
