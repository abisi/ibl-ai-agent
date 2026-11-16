# Model v5 — design (2026-10-23)

Follows v4 (model_v4_design.md, 019/020). User 2026-10-23: "yes go" to (1) one interpretable transfer parameter and (2) separating
within-session engagement from false-alarm exploration.

Why: in v4, gamma entered both the value read-out and the update split (effective transfer ~ gamma^2) and was fitted ~0 everywhere, so
day-0 whisker licking was carried by eps and the R- day-0 extinction curve was missed; and the best FA variant (E3, eps = recent reward
rate) did two jobs: within-session engagement (warm-up, satiation decline) and false alarms.

## Changes vs v4
- No shared S node. Stimulus nodes A, W -> lick. Transfer g, two variants:
  - init: at the mouse's first whisker trial w_W <- g * w_A (copy once), then independent learning. R- negative outcomes no longer
    lower auditory value.
  - overlap: overlapping input representations, auditory x = (1, g), whisker x = (g, 1); V = w . x; update w += alpha * delta * x
    (ongoing two-way generalisation, a linear network with shared inputs).
- Engagement factor on all licking (variant eng): P(lick) = (1 - lapse) * e_t * [1 - (1 - eps_t)(1 - drive)],
  e_t = e_min + (1 - e_min)(1 - exp(-Rbar_t / r_e)), Rbar recent reward rate (alpha_R), reset to R0 at session start.
  Engagement scales all trial types; eps keeps the FA-specific role.
- One timescale only (2ts added little in v4); cohort grouping only (cohort > shared in every v4 variant); FA variants E0-E3 (B dropped:
  worst in v4).
- Models: transfer {init, overlap} x FA {E0, E1, E2, E3} x {noeng, eng} x {sym, asym} = 32.

## Recovery
Generators from the real fits: best model; best model with g = 0.5 (recovery at g > 0, untested in v4); the other transfer variant,
the engagement toggle and the sym/asym toggle of the best model. Refit all 32. Same checks as v4.
