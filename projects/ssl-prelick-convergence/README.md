# ssl-prelick-convergence

Do whisker-triggered licks come to resemble auditory-triggered licks before the lick, when whisker licks are rewarded
(R+) but not otherwise (R−)? Pre-lick (100 ms before the corrected first lick) single-neuron ROC, population geometry
(cross-validated distances, λ, λ_LDA), chance-corrected decoders (single sessions, hierarchical pseudo-populations) and
brain-area analyses, with spontaneous licks (SL, headline) or false alarms (FA) as the unrewarded-lick reference.

Split out of `ssl-rastermap-psth-variants` on 2026-10-03 (scripts keep their original numbers). The locked analysis set,
headline results, caveats and open TODOs are in `LOCKED.md`.

## Layout
- `exploratory-analyses/` — analysis scripts (numbered; run on haas from `~/code/unit_spikes_analysis` with
  `PYTHONPATH=~/code/NWB_reader:.`). The reference is selected with `PRELICK_REF=sl|fa`; outputs go to
  `combined_results_ks4/_roc_prelick_sl/` or `combined_results_ks4/_roc_prelick/`.
- `run/` — shell drivers used on haas (rebuild all figures; rerun pseudo-populations; figures + article sources).
- Results (not in git): `/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/_roc_prelick{,_sl}/` (README.md there
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
