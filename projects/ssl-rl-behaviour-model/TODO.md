# TODO

Converged model (2026-10-30): core = simple-gen + absorbing HMM, g_W0 shared between cohorts. Dropped for good: absorption lag, kappa,
theta, ctx step, -1 coding. Open:
- Select single-mouse add-ons (rec, fast, pers, hab, habr, gw) on held-out ΔLL and binned R2_dev / dispersion, esp. R-; combine winners.
- Shared g_W0 profile for the chosen model; per-mouse refit (025 --lean, 028, 031), recovery, alpha_W vs model-free learning.
- Model-recovery matrix (036), more starts for the final cohort fits, bootstrap over mice for R+ vs R- CIs.
- Report v9 around the converged model.

Older open points (v1 to v8) are superseded by the converged model; see change-log.md for their resolution.
