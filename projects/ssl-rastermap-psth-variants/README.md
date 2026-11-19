# ssl-rastermap-psth-variants

Rastermap clustering of single-neuron PSTHs (Axel Bisi's `unit_spikes_analysis/rastermap_psth` pipeline) run as labelled
analysis variants, plus the analyses built on it. Question and history: `question.md`, `TODO.md`, `change-log.md`.

- Code: this folder (synced to haas `~/code/ibl-ai-agent/projects/ssl-rastermap-psth-variants`).
- Results: `combined_results_ks4/rastermap_variants/<variant>/` (one folder per variant, `variant_provenance.json`).
- Pipeline functions live in `unit_spikes_analysis/rastermap_psth` (`run_clustering_new`, `population_matrix_summary`,
  `rastermap_utils`, `gmm_utils_new`); the scripts here only call them (no local copies).

## Main (publication) variant

`qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin`: good + MUA units of learners (reference-sheet
learning category good / moderate), whisker learning day (day 0, one session per mouse), corrected lick times, long
windows, locality 0.25, grid upsampling 10; cross-validated fit on odd trials, 100 clusters, contiguous metaclusters
(1 - r cut 0.40, 24 metaclusters).

Used by: Figure 2 (metacluster description, `045`), Supplementary Figures 3 and 4
(`projects/ssl-paper-figures/figures/SuppFigure3_rastermap_clustering`, `SuppFigure4_rastermap_validation`).

## Scripts

`exploratory-analyses/` (numbering kept; scripts import siblings of their own folder only)

| Script | What |
|---|---|
| 000 | driver: build feature matrix, run rastermap, figures, provenance (all variants) |
| 001 | ROC types per cluster / metacluster (EDA, Fisher + BH, Cramér's V) |
| 002, 008 | population-matrix summaries (`population_matrix_summary`), regeneration with current defaults |
| 003, 004 | R+ / R- composition (F_mk) and run_stats_only per metacluster |
| 005 | locality test (0 / 0.25 / 0.75): agreement tables in `_locality_test_summary/` |
| 006 | grid-upsampled rastermap, clusters = k-means nodes (imported by 009, 015, 042, 044) |
| 007 | contiguous metaclustering (segments), threshold sweep |
| 009 | control: whisker-hit conditions trimmed (`trimStim / trimLick / trimBoth` variants) |
| 015 | lick-fixed rebuild (creates the `__lickfix_longwin` variants) |
| 023, 043 | significant-cluster figures; regeneration after the ROC rate conversion |
| 042 | variant without the whisker-reward bins (`__noWhiskerReward`) |
| 044 | learning + expert sessions variant (`__learning_expert`) |
| 045 | metacluster description figure (Figure 2; Supp. Fig. 3 with `--cols 1 --fs 6 --width-mm 183`) |
| 046, 047 | spectral (Bayesian) GMM on the main variant: Dirichlet-process fit, fixed K = 100 / 24 (Supp. Fig. 4) |

Topic subfolders (separate questions that grew here; moved 2026-11-19, otherwise unchanged):

| Folder | Scripts | What |
|---|---|---|
| `lick_timing/` | 011, 013, 014 | trials.lick_time offset: sync check, explanation figure, corrected alignment (skill `ssl-lick-alignment`) |
| `mmd_rewardfree/` | 016-022 | reward-free R+ vs R- MMD^2 (CV, method figures, kernel sensitivity, time-resolved, per area, publication overview) |
| `roc_rates/` | 024-026 | single-neuron ROC on firing rates instead of counts (one-off conversion of all ROC results) |
| `rpe_novelty/` | 010, 012, 027-041 | RPE ROC (pilot, v1, v2), novelty / exposure models and figures |

The pre-lick convergence analyses (051-071) moved to `projects/ssl-prelick-convergence` on 2026-10-03.

## Results folders that are not variants

`_cache/` (feature tables shared with other projects, ~89 GB), `_locality_test_summary/`, `_trim_test_summary/`,
`_publication_layout_preview/` (045 prototype), `_test/` (8-mouse test run), `_stopped_qc_good_learning_expert_with_duplicates/`
(aborted run).
