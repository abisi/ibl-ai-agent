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

## Analyses
- 001 trial table from the behaviour NWBs (auditory, whisker, free-licking sessions; active trials, perf != 6; cohort from the joint mouse
  reference sheet) and interpolated targets (individual and group mean, free licking removed).
- 002 parameter recovery of the joint fit.
- 003 joint fit of the group-mean targets: common pretraining, common or group-specific transfer parameters (nested models).
- 004 per-mouse fits with common learning parameters.

## Data
Behaviour NWBs in NWB_combined (all mice with an R+ / R- label, at least one auditory day and a whisker day 0). Outputs:
combined_results_ks4/ssl-rl-behaviour-model/{tables,targets,fits,figures}.
