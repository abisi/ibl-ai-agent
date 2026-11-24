# ssl-rastermap-psth-variants

Rastermap clustering of single-neuron PSTHs (Axel Bisi's `unit_spikes_analysis/rastermap_psth` pipeline) run as labelled
analysis variants, plus the analyses built on it. Question and history: `question.md`, `TODO.md`, `change-log.md`.

- Code: this folder (synced to haas `~/code/ibl-ai-agent/projects/ssl-rastermap-psth-variants`).
- Results: `combined_results_ks4/rastermap_variants/<variant>/` (one folder per variant, `variant_provenance.json`).
- Pipeline functions live in `unit_spikes_analysis/rastermap_psth` (`run_clustering_new`, `population_matrix_summary`,
  `rastermap_utils`, `gmm_utils_new`); the scripts here only call them (no local copies).

## v2 variants (2026-11-23, `050_rastermap_variants_v2.py`)

Results: `combined_results_ks4/rastermap_variants/v2/<variant>/` (`variant_provenance.json` with the full config,
pipeline tree `rastermap_clustering/`, organised copies in `figures/<topic>/` + `figures/index.md`); comparison with main
in `v2/_comparison/`; fresh day-0 tables in `v2/_tables/`.

**Main** (all settings set in 050, not taken from config.yaml defaults): stable units (good + MUA passing coverage >= 0.9,
presence >= 0.5, drift independence; `build_feature_matrix.stable_unit_mask`, `stable_only: true` in config.yaml) of
learners (reference-sheet learning category good / moderate), whisker day 0, one session per mouse; corrected lick times
(code default, no longer in names); windows passive -0.1..0.2, active -0.1..0.4, lick -0.3..0.2 s; 10-ms bins, 5-ms
stride; baseline-normalised; FR filters 0.01 Hz (session) / 0.1 Hz (condition); odd/even r >= 0.1; CV fit on odd trials,
100 clusters = rastermap k-means nodes, locality 0.25, grid upsampling 10; contiguous metaclusters, 1 - r cut 0.40.

Variants (one change each): `units_all` (stable filter off), `units_good`, `loc0p00`, `loc0p75`, `noFR`, `mice_all`
(learners + non-learners), `noWhiskerReward` (reward-containing whisker-hit bins removed).

Phases: `tables` -> `features` -> `fit` -> `figures` (003, 004, 001, 023, 045) -> `organize` -> `compare`; `--test` runs
8 mice into `rastermap_variants/_v2_test/`. The figure scripts read `RASTERMAP_VARIANTS_ROOT` / `RASTERMAP_TABLES_CACHE`.

Until the paper figures move to v2, Figure 2 / Supp. Figs 3-4 and 045-048 read the archived v1 main run
`rastermap_variants/_archive_v1/qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin` (good + MUA
including unstable units). All v1 variants were moved to `_archive_v1/` on 2026-11-23.

## Scripts

`exploratory-analyses/` (numbering kept; scripts import siblings of their own folder only)

| Script | What |
|---|---|
| 050 | **v2 driver**: main + robustness variants, all figures, organised folders, comparison (supersedes 000/006/015/042 for new runs) |
| 000 | v1 driver: build feature matrix, run rastermap, figures, provenance (`load_tables` reused by 050) |
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

`_cache/` (feature tables shared with other projects, ~89 GB; never touch), `_archive_v1/` (all v1 variants and their
test summaries, archived 2026-11-23), `_v2_test/` (8-mouse test of 050).
