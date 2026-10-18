# ssl-rl-behaviour-model

## Question
Can a normative reinforcement-learning model of the auditory-to-whisker transfer account for R+ and R- learning curves with one learning
rule, when the auditory pretraining (identical for both cohorts until day 0) is modelled with one shared parameter set, and the cohorts
differ only through the whisker reward contingency? Which architecture (independent stimulus weights, a node shared by auditory and
whisker inputs, or both) is needed, and does any parameter have to differ between cohorts in the transfer phase? The per-trial model
quantities are intended as behavioural latents for the neural analyses (ssl-learning-axis-exploration).

## Model
P(lick | x) = Phi(dQ(x)), x = [auditory, whisker, no stimulus, context]; excitatory / inhibitory weights with multiplicative updates
scaled by the prediction error R - a dQ, a learning rate b, an asymmetric gain v (positive errors on auditory trials, negative errors on
no-stimulus trials) and tr (positive whisker errors). Architectures: independent, shared (auditory and whisker through one node), mixed.
Original model and fits: github.com/abisi/rl_model_auditory_whisker (Lila Meflah); vendored here as rl_*.py (branch shared-pretraining).

## Model v2 (2026-10-18, rl2_model.py; supersedes the curve-fitting design of 002-005)
User: "the model is to fit behavioural data to see what type of learning strategy mice use, explaining both auditory-to-whisker learning
and the R+ / R- difference; the latter should be just a design choice ... keep R- whisker lick as -1".
One learning rule for every mouse: features bias (every trial), stimulus-present S, auditory A, whisker W (a no-stimulus trial is the bias
alone; the separate no-stimulus node of v1 is redundant with the bias and is kept only as a test variant). Delta rule on the lick value
with credit split between the generic S and the specific stimulus (gamma) and onto the bias (kappa), asymmetric rates, sigmoid choice with
lapses. Cohorts differ only in the outcome of a whisker lick (+1 R+, -1 R-). Fitted by maximum likelihood of each mouse's own trial
sequence; per-trial latents saved for neural alignment.

## Analyses
- 001 trial table from the behaviour NWBs (auditory, whisker, free-licking sessions; active trials, perf != 6; cohort from the joint mouse
  reference sheet) and interpolated targets (individual and group mean, free licking removed).
- 002 parameter recovery of the joint fit.
- 003 joint fit of the group-mean targets: common pretraining, common or group-specific transfer parameters (nested models).
- 004 per-mouse fits with common learning parameters.
- 005 figures of v1.
- 006 v2: per-mouse maximum-likelihood fits of six variants (full, specific gamma=0, general gamma=1, no bias learning, symmetric
  rates, + no-stimulus weight), pooled common vs cohort-specific fits, per-trial latents.
- 007 v2 evaluation: parameter and model recovery on each mouse's own sequence, one-step and closed-loop predictive curves, simulations
  under the swapped cohort contingency.
- 008 v2 figures.

## Data
Behaviour NWBs in NWB_combined (all mice with an R+ / R- label, at least one auditory day and a whisker day 0). Outputs:
combined_results_ks4/ssl-rl-behaviour-model/{tables,targets,fits,figures}.
