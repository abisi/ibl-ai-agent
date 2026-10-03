# ssl-within-day-remapping

Does pre-lick activity on whisker hits (WH) change within the learning day, relative to auditory hits (AH, rewarded) and
spontaneous licks (SL, unrewarded), in opposite directions in R+ and R− mice, and how does the end of day 0 compare with
expert sessions? Builds on `ssl-prelick-convergence` (pre-lick events and loaders from its scripts 051/057/061/062,
imported from `../ssl-prelick-convergence/exploratory-analyses`).

## Scripts (`exploratory-analyses/`, run on haas with `PRELICK_REF=sl`)
| script | analysis | output |
|---|---|---|
| 001_within_session_halves.py | halves split at the median auditory hit; count-matched subsamples; Δd, λ, distance components per half; whole-session and cross-half decoders with linear-shift chance; odd/even null split; behavioural controls | `combined_results_ks4/_within_day_sl/halves/` |
| 002_trial_slopes.py | trial-level scores on a fixed session axis (cross-validated mean-difference axis, d′ ≥ 0.3, and decoder P(AH)); slopes vs normalised time per class; WH − SL and WH − AH relative slopes; binned trajectories; day-0 fitted start / end vs expert level | `combined_results_ks4/_within_day_sl/slopes/` |

| 003_epoch_comparison.py | within-day vs across-day on a common footing: phase-matched halves in every session, fixed events per class and units, session-anchored position / decoder / normalised Δd; within-day, across-day and carry-over contrasts; hierarchical bootstrap, mouse permutation, MixedLM | `combined_results_ks4/_within_day_sl/epochs/` |

Statistics: session as unit; Wilcoxon / one-sample t vs 0 per group; day 0 R+ vs R− (MWU, Welch, mouse-level cohort
permutation); cohort × stage by mouse-level permutation; populations all mice and learners.
