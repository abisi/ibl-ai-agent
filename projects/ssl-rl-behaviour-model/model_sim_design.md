# Simulation-based model of the whole training process (2026-10-18)

User: "I want a theory model, simulated with different parameter values. Then I compare simulation to each mouse to see which fits best.
I do not wish to model within-session dynamics beyond maybe end-of-session disengagement (motivation); rather, I want to fit the whole
training process, from auditory to whisker, and find which model overall best predicts mouse data (learners, non-learners, expert start,
gradual, etc)." Day-0 within-session learning is included (decided); phenotypes: the reference sheet's labels as they are
(learning_category; also gradual, expert_start), best model reported per category. This follows the original repository's intent
(simulations compared with learning trajectories) with per-mouse trial sequences, competing strategies and held-out comparison.

## Agents (theory models)

Each agent is run on the mouse's own trial sequence (same stimuli, sessions, days; licks are the agent's own, drawn from its P(lick)):
- core: bias (every trial) + auditory A + whisker W weights; after a lick δ = o − Q, α+ / α−, credit κ on the bias; outcomes fixed by
  the design (auditory +1 water, whisker +1 water R+ / −1 R−, no stimulus −0.5); P(lick) = λ/2 + (1 − λ)·σ(β·T·Q − c)
- disengagement (in every agent): thirst T reset to 1 at session start, − ρ per water reward (end-of-session drop); the only
  within-session process besides learning
- G generalisation: stimulus-present feature with credit share γ (whisker inherits part of the auditory value)
- N novelty: bonus ν·exp(−presentations/λ_ν) for the tone (pretraining start) and the whisker stimulus (day 0)
- Z context switch: bias × (1 − ζ) at the first trial of day 0
Models: core + any of G, N, Z → 8 agents (core alone = stimulus-specific learning).

## Summaries compared (mouse vs simulation)

Per session: lick rate of auditory, whisker and no-stimulus trials in the first and second half of the session; day 0: whisker trials in
blocks of 10 (first 100 whisker trials) instead of halves. Each bin is a count of licks over trials.

## Matching and comparison

- Score of a parameter set: binomial log-likelihood of the mouse's bin counts given the simulated mean lick rate per bin (mean P(lick)
  over R runs; clipped to [0.001, 0.999]).
- Bank: N parameter sets per agent (Latin hypercube; log scale for rates), then Nelder-Mead refinement from the best bank sets with fixed
  random seeds (common random numbers).
- Held-out comparison: leave-one-session-out on the bins (parameters chosen on the other sessions' bins, the left-out session's bins
  scored), summed over sessions; best agent per mouse = highest held-out score. Day-0 fold reported separately.
- Recovery: pseudo-mice simulated from each agent's best parameters for real mice, matched with all agents → confusion matrix and
  parameter recovery.
- Reported: best agent per mouse overall, by cohort, by learning_category (and gradual, expert_start); data vs best-agent trajectories.
