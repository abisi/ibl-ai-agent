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
| 004_halves_psth.py | first-lick-aligned PSTHs per session half (WH, AH, SL; baseline-subtracted, unit-averaged per session) | `_within_day_sl/psth_halves/` |
| 005_cosyne_within_day.py | 5-panel COSYNE summary of the within-day vs expert comparison | `_within_day_sl/cosyne/` |
| 006 / 008 | COSYNE convergence timeline (compact / expanded with methods examples and controls) + captions | `_within_day_sl/cosyne/` |
| 007 | compiles the captions of all COSYNE figures | `_within_day_sl/cosyne/COSYNE_captions.md` |
| 009_mixed_model.py | single-trial mixed model of the CD projection (whisker hit × time × cohort) | `_within_day_sl/mixed_model/` |
| 010_decoder_schemes.py | decoder results by number of neurons per single-session decoder | `_within_day_sl/cosyne/decoder_schemes.*` |

Statistics: session as unit; Wilcoxon / one-sample t vs 0 per group; day 0 R+ vs R− (MWU, Welch, mouse-level cohort
permutation); cohort × stage by mouse-level permutation; populations all mice and learners.

## Shared with ssl-prelick-convergence (method map, 2026-10-07)
| What | Defined in (ssl-prelick-convergence) | Used here |
|---|---|---|
| Trials, classes (WH / AH / reference), corrected first lick, warm-up cut, A1 trim, SL reference, baselines, pre-lick window (100 ms), per-event rates (`*_trials.npz`), min FR 0.1 Hz | 051 (`select_trials`, `spontaneous_licks`, `unit_rates`, `MIN_FR`, `SL_BASE`, `PSTH_*`, `OUTROOT` / `TAG`) | all scripts (004 recomputes PSTHs per half with 051 functions) |
| λ and cross-validated distances (d(WH,SL), d(WH,AH), d(AH,SL)) | 057 `lam` | 001, 003 |
| Session list, learners rule | 026 `all_sessions`, 061 `learner_filter` | 001, 002, 003, 008 |
| Figure style, group colours, `dots_panel`, `fmt_p` | 062 | 001-010 |
| Decoder (logistic, C = 0.05, balanced, 5-fold, z-scored on training folds) + linear-shift null (40 shifts, \|k\| 5..n/3) | 068 (`cv_bacc`, `decode_shift`) | re-implemented in 001 (`ws_predict`, `xh_predict`, `shifts`, `shifted`); same parameters |
| Mouse-level cohort-permutation interaction (10 000 permutations) | 062 `group_stats` | re-implemented in 001 (`perm_interaction`, + day-0 R+ vs R−); same statistic |
| Units | good + mua, min FR 0.1 Hz, >= 5 units | same |
| Minimum events per class | 3 (051 ROC), 4 (057, 068), 6 (064 pseudo-populations) | 4 (001), 4 whisker hits (002) |
Results of this project feed `ssl-whisker-hitmiss-timeresolved-decoding` 154 (`_within_day_sl/slopes`).
