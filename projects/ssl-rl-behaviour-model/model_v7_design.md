# Model v7 — hierarchical mouse-level fits (2026-10-23)

User 2026-10-23: mouse-specific parameters; "yes go" to: (1) fix the within-session shape (v6, 023), (2) fit mouse-specific parameters
hierarchically on the v6 variant that fixes it, (3) recovery, (4) only then test the R- fast-extinction subgroup suggested by 024.

## Levels
- Mouse: alpha+, alpha-, beta, c, s_B, mu_rho (+ theta for thr models). Population prior per cohort on the unbounded scale (scaled logit
  of the bounded parameter, log first where rl7 uses log): y_m ~ N(mu, diag sd^2).
- Session: rho_s, log rho_s ~ N(mu_rho of the mouse, sd_rho^2), integrated out exactly (rl7 quadrature).
- Cohort: g, sd_rho, tau_a, engagement (e_min, r_e, alpha_R, R0), fixed at the 023 cohort fit (weakly identifiable per mouse; g = 0 in
  every cohort fit).

## Fitting (rl8_hier, 025)
EM with Laplace approximation (Huys et al. 2011): per mouse MAP and Hessian, then population mu, sd from MAPs + posterior variances;
iterate to a change in summed log evidence < 0.5. Mice independent given the population (weights do not interact), parallel per mouse.
Per-mouse Laplace evidence -> per-mouse model comparison between v6 variants; iBIC for the cohort-level comparison.

## Read-outs
Per-mouse weight trajectories (w_A, w_W, V), per-mouse parameters; R+ vs R- per parameter with mouse as the unit (Mann-Whitney + Welch on
the unbounded scale), including the scale-free log(beta * alpha+).

## Recovery
(a) Simulate every mouse from its MAP (rho_s from its prior), refit: per-parameter Spearman true vs refit, bias, R+ vs R- difference
preserved. (b) Mixed R- cohort, 11 random R- mice alpha- = 0.5, rest 0.001: AUC of refit alpha- for fast vs slow mice. If (b) fails,
the subgroup question cannot be answered with this model.

## Then
Subgroup: is per-mouse alpha- bimodal in R-, and does it track the day-0 whisker excess over FA (024: ~10 R- mice)? Novelty on W as a
per-mouse factor competes with alpha- for the same excess, added after this.
