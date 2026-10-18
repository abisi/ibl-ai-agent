# Model v3 — design for review (2026-10-18; decisions of 2026-10-18 in §9)

Goal (user): fit behavioural data to identify the learning strategy mice use, explaining auditory-to-whisker learning and the R+ / R-
difference, the latter being only a design choice (task contingency). v3 = v2 learning core + motivation (thirst/satiation, after
Matteucci, Guyoton, Favero Montero, Grabau & El-Boustani, "Mousemind-Q" poster) + candidate strategy components, each switchable, compared
by held-out likelihood. Body weight is not used.

## 1. Trial structure and observables

Per mouse, one continuous sequence of trials over pretraining and whisker days (sessions in order). Trial t: stimulus s_t ∈ {A, W, N}
(auditory, whisker, no stimulus), lick l_t ∈ {0, 1}, trial start time, session index. Reward time of a rewarded lick:
t_rew = start_time + (lick_time − response_window_start_time) (corrected reaction time; constant reward volume = 1 unit).

Outcome of a lick (the only cohort-dependent quantity):

| | auditory | whisker | no stimulus |
|---|---|---|---|
| R+ | +1, water | +1, water | −0.5, no water |
| R− | +1, water | −1, no water | −0.5, no water |

Fixed values (decided 2026-10-18): an R− whisker lick is an active mistake (a stimulus-triggered lick left unrewarded), −1; a
spontaneous lick without stimulus (false alarm) −0.5. No free outcome parameter: how the cost of the R− whisker lick shifts over time is
carried by the learned weights.

o_t drives learning; water drives satiation. R− mice therefore also drink less per session — a consequence of the design, not a
parameter.

## 2. Components

### 2.1 Learned value of licking (core, from v2)

Features: bias b (every trial), stimulus-present S (A or W), auditory A, whisker W.

  Q_t(s) = w_b,c + γ·w_S + (1 − γ)·w_s        for s ∈ {A, W}
  Q_t(N) = w_b,c

After a lick (no update without a lick, as no outcome is observed):

  δ_t = o_t − Q_t(s_t),   α = α+ if δ_t > 0 else α−
  w_b,c += α·κ·δ_t,   w_S += α·γ·δ_t,   w_s += α·(1 − γ)·δ_t

All weights start at 0 (the fixed lick cost c below is the intercept; v2's free initial bias w0 is replaced by c).
c indexes the context (2.4); without context inference there is a single bias w_b.

### 2.2 Motivation: thirst and satiation (poster)

Thirst T_t ∈ [0, 1], reset to 1 at every session start (decided; with β free a per-mouse starting level is not identifiable; no body
weight). Default (instantaneous absorption): thirst falls by ρ per reward received:

  T_t = max(0, 1 − ρ · (number of rewards before t))

Optional, only if recovery shows it identifiable: absorption with time constant τ (poster), so satiation lags the rewards:
T_t = max(0, 1 − ρ · Σ_j (1 − exp(−(t − t_rew,j) / τ))).

Thirst scales the expected value and drives a stimulus-independent Pavlovian lick bias:

  B(T) = s_B · log(1 + exp(k·(T − θ))) / k      (softplus ≈ s_B · [T − θ]+; k fixed, e.g. 20)

### 2.3 Novelty / exploration

A stimulus gets an exploration bonus that decays with exposure (number of presentations n_t(s), licked or not), not with outcome:

  Nov_t(s) = ν · exp(−n_t(s) / λ_ν)        for s ∈ {A, W};  0 for N

Applies to the tone at the start of pretraining and to the whisker stimulus at day 0 (same ν, λ_ν).

### 2.4 Context inference at whisker onset

At the first whisker session the mouse may infer a new context. The bias of the new context starts as a mix of the old one and naive:

  w_b,whisker-context ← (1 − ζ)·w_b,auditory-context      (ζ = 0: same context; ζ = 1: new context, bias relearnt from 0)

Learned stimulus weights are kept (they are features of the stimuli, not of the context). Predicts a drop of no-stimulus licking at day 0
in both cohorts.

### 2.5 Average reward / opportunity cost (Niv et al. 2007, poster)

The mouse tracks its recent reward rate R̄ and evaluates outcomes relative to it (per trial, Δt = 1 trial):

  R̄ ← R̄ + α_R·(r_t − R̄)  on every trial (r_t = 1 if water, else 0),   o'_t = o_t − η·R̄

so a lick outcome is worth less when rewards have been plentiful. η scales the opportunity cost. R̄ reset to 0 at session start.

### 2.6 Perseveration

Choice kernel, reset to 0.5 at session start:

  C_{t+1} = C_t + φ·(l_t − C_t),   contribution π·(2C_t − 1)

## 3. Decision

  DV_t = β · T_t · Q_t(s_t)  +  B(T_t)  +  Nov_t(s_t)  +  π·(2C_t − 1)  −  c

  P(lick) = (1 − λ) · [1 − (1 − p_sp) · (1 − σ(DV_t))]

λ: disengagement lapse (no lick whatever the value); p_sp: spontaneous licks (lick whatever the value); σ: logistic. (Poster form.)

## 4. Parameters

| Component | Parameters | Bounds | Switch-off value |
|---|---|---|---|
| Core learning | α+, α−, κ | (0, 1), (0, 1), [0, 3] | — |
| Generalisation | γ | [0, 1] | γ = 0 |
| Decision | β, c, λ, p_sp | (0.1, 50), (−5, 5), [0, 0.5], [0, 0.5] | — |
| Motivation | ρ, τ, s_B, θ | [0, 0.05] per reward, (10 s, 30 min), [0, 10], [0, 1] | ρ = 0, s_B = 0 (T ≡ 1) |
| Novelty | ν, λ_ν | [0, 5], (1, 500) presentations | ν = 0 |
| Context | ζ | [0, 1] | ζ = 0 |
| Average reward | α_R, η | (0, 1), [0, 3] | η = 0 |
| Perseveration | π, φ | [−3, 3], (0, 1) | π = 0 |

Core = 7 parameters; full model = 18 (+ τ if added).

## 5. Model space

Core always on; six switchable components: generalisation (G), motivation (M), novelty (N), context (Z), average reward (R),
perseveration (P) → 2^6 = 64 models. Every model fitted per mouse. Cohorts share everything; R+ / R− enter only through §1.

## 6. Fitting

- Maximum likelihood of the mouse's licks, trial by trial; bounded L-BFGS-B, multiple starts; nested consistency (a model is also started
  from the optimum of every model it contains, so a superset cannot end worse than a subset). Numba kernels.
- Trials: all active trials except perf == 6 (decided); no end-of-session trimming (disengagement is what motivation and λ capture).

## 7. Model selection

- Leave-one-session-out held-out log-likelihood per mouse: for each session k, parameters fitted on the likelihood of all other sessions;
  the model runs through the whole sequence (states updated with the actual licks and outcomes, so the state at every session start is
  correct) and session k is scored by one-step-ahead log-likelihood. Summed over folds; also reported for the day-0 session alone (where
  strategies differ most).
- Group level: paired comparison of held-out log-likelihood across mice; random-effects Bayesian model selection (protected exceedance
  probabilities; held-out log-likelihood as log evidence); component-wise (family) inference: probability that each component is used,
  summed over the 16 models with vs without it. Cohort comparison of model frequencies as a check of "design only".

## 8. Validation before real data

- Model recovery: simulate each of the 32 models on real trial sequences (parameters from realistic ranges), fit all 32, confusion
  matrix by held-out likelihood. Components that cannot be told apart are merged or redefined.
- Parameter recovery for the full model.
- Predictive checks as v2: one-step and closed-loop curves per cohort and trial type; swapped-contingency simulations; plus within-session
  curves (licking vs trial within session) for the motivation component, and the day-0 transition (first whisker trials vs FA) for
  generalisation / novelty / context.

## 9. Decisions (2026-10-18)

1. Thirst reset to 1 at every session start.
2. Absorption: instantaneous (thirst = 1 − ρ × rewards so far) by default; τ only if recoverable.
3. Novelty applies to the tone at pretraining start and to the whisker stimulus (same ν, λ_ν).
4. Context switch: once (first whisker session), bias only — explained: models the step drop of no-stimulus licking at whisker onset in
   both cohorts, which gradual learning cannot produce.
5. Average-reward term: included as a switchable component (R).
6. One constant lick cost c.
7. Trials: all active trials except perf == 6; no trimming.
8. Outcomes fixed: auditory +1, R+ whisker +1, R− whisker −1 (active mistake), false alarm −0.5; no decaying cost (the weights carry
   the change over time).
