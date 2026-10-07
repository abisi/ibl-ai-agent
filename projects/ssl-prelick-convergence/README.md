# ssl-prelick-convergence

Two parts (merged 2026-10-07 with ssl-within-day-remapping): **Part I, across days** (learning day vs expert sessions)
and **Part II, within days** (time course within the learning day and within expert sessions). Code: Part I in
`exploratory-analyses/`, Part II in `exploratory-analyses/within_day/` (imports Part I modules). Results:
`combined_results_ks4/ssl-prelick-convergence/{across_days,within_day}/<ref>/` (ref = sl | fa); the old folders
`_roc_prelick{,_sl}` and `_within_day_sl` are symlinks (manifest `results_home_manifest_20261007.tsv`).

Do whisker-triggered licks come to resemble auditory-triggered licks before the lick, when whisker licks are rewarded
(R+) but not otherwise (R−)? Pre-lick (100 ms before the corrected first lick) single-neuron ROC, population geometry
(cross-validated distances, λ, λ_LDA), chance-corrected decoders (single sessions, hierarchical pseudo-populations) and
brain-area analyses, with spontaneous licks (SL, headline) or false alarms (FA) as the unrewarded-lick reference.

Split out of `ssl-rastermap-psth-variants` on 2026-10-03 (scripts keep their original numbers). The locked analysis set,
headline results, caveats and open TODOs are in `LOCKED.md`.

## Layout
- `exploratory-analyses/` — analysis scripts (numbered; run on haas from `~/code/unit_spikes_analysis` with
  `PYTHONPATH=~/code/NWB_reader:.`). The reference is selected with `PRELICK_REF=sl|fa`; outputs go to
  `combined_results_ks4/ssl-prelick-convergence/across_days/sl/` or `.../across_days/fa/`.
- `run/` — shell drivers used on haas (rebuild all figures; rerun pseudo-populations; figures + article sources).
- Results (not in git): `/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence/across_days/{fa,sl}/` (README.md there
  documents every version, null and figure).

## Pipeline
| step | script | output |
|---|---|---|
| shared loaders / ROC infrastructure (copied from ssl-rastermap-psth-variants) | 026, 045, 047, 048, 049 | — |
| pre-lick rates + ROC per unit (WH/AH/ref), trial npz | 051_roc_prelick.py | `prelick_units.parquet`, `*_trials.npz` |
| examples, tables, convergence, stage summaries, PSTHs | 052–056 | figures, `psth/` |
| λ and cross-validated distances (Δd) | 057_roc_prelick_lambda.py | `lambda/` |
| convergence summary, single-neuron transfer | 058, 059 | `transfer/` |
| λ_LDA (shrinkage LDA) | 060 | `lambda_lda/` |
| learners-only population rule | 061 | `learners/` |
| publication figures, captions, stats, COSYNE v2/v3 | 062_pub_convergence_figures.py | `publication/<pop>/` |
| decoder sweep | 063 | `decoder_sweep/` |
| pseudo-population hierarchical bootstrap (shift null) | 064 | `pseudopop/<pop>/` |
| converging-unit location, area attrition, density maps | 065–067 | `generalizing_units/`, `area_attrition/` |
| single-session decoders with linear-shift null | 068, 069 | `decoder_shift/`, `decoder_transfer_shift/` |
| recap across iterations | 070 | `recap/` |
| SL timing control | 071 | `sl_timing/` |
| article (per reference × population) | build_article.py (haas) + render.sh (local Quarto/typst) | `publication/<pop>/prelick_convergence_<ref>_<pop>.pdf` |

## Part II: within days (formerly ssl-within-day-remapping)

Does pre-lick activity on whisker hits (WH) change within the learning day, relative to auditory hits (AH, rewarded) and
spontaneous licks (SL, unrewarded), in opposite directions in R+ and R− mice, and how does the end of day 0 compare with
expert sessions? Builds on Part I (pre-lick events and loaders from scripts 051/057/061/062, imported from the parent folder).

### Scripts (`exploratory-analyses/within_day/`, run on haas with `PRELICK_REF=sl`)
| script | analysis | output |
|---|---|---|
| 001_within_session_halves.py | halves split at the median auditory hit; count-matched subsamples; Δd, λ, distance components per half; whole-session and cross-half decoders with linear-shift chance; odd/even null split; behavioural controls | `combined_results_ks4/ssl-prelick-convergence/within_day/sl/halves/` |
| 002_trial_slopes.py | trial-level scores on a fixed session axis (cross-validated mean-difference axis, d′ ≥ 0.3, and decoder P(AH)); slopes vs normalised time per class; WH − SL and WH − AH relative slopes; binned trajectories; day-0 fitted start / end vs expert level | `combined_results_ks4/ssl-prelick-convergence/within_day/sl/slopes/` |

| 003_epoch_comparison.py | within-day vs across-day on a common footing: phase-matched halves in every session, fixed events per class and units, session-anchored position / decoder / normalised Δd; within-day, across-day and carry-over contrasts; hierarchical bootstrap, mouse permutation, MixedLM | `combined_results_ks4/ssl-prelick-convergence/within_day/sl/epochs/` |
| 004_halves_psth.py | first-lick-aligned PSTHs per session half (WH, AH, SL; baseline-subtracted, unit-averaged per session) | `within_day/sl/psth_halves/` |
| 005_cosyne_within_day.py | 5-panel COSYNE summary of the within-day vs expert comparison | `within_day/sl/cosyne/` |
| 006 / 008 | COSYNE convergence timeline (compact / expanded with methods examples and controls) + captions | `within_day/sl/cosyne/` |
| 007 | compiles the captions of all COSYNE figures | `within_day/sl/cosyne/COSYNE_captions.md` |
| 009_mixed_model.py | single-trial mixed model of the CD projection (whisker hit × time × cohort) | `within_day/sl/mixed_model/` |
| 010_decoder_schemes.py | decoder results by number of neurons per single-session decoder | `within_day/sl/cosyne/decoder_schemes.*` |

Statistics: session as unit; Wilcoxon / one-sample t vs 0 per group; day 0 R+ vs R− (MWU, Welch, mouse-level cohort
permutation); cohort × stage by mouse-level permutation; populations all mice and learners.

### Shared with Part I (method map, 2026-10-07)
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
Results of this project feed `ssl-whisker-hitmiss-timeresolved-decoding` 154 (`within_day/sl/slopes`).
