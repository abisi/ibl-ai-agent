# TODO

- [X] Map unit_spikes_analysis rastermap_psth logic (build_feature_matrix → run_rastermap → run_stats_only),
      folder naming, learners filter (hardcoded), figure names (fig8_flow_umap = fig_flow_umap)
- [X] Driver `exploratory-analyses/000_run_rastermap_variants.py` (4 variants, one labelled root each,
      provenance json, new cluster-mean matrix | F_mk figure with MW + Welch)
- [X] Small test: 8 mice, 1 variant → `combined_results_ks4/rastermap_variants/_test/` (7 min; found fig8_flow_umap never saves → driver re-calls + saves `fig_flow_umap_cv`)
- [~] Full run, 4 variants (haas, started 2026-09-25, log ~/log/rastermap_variants_full.log) → `combined_results_ks4/rastermap_variants/<variant>/`
- [ ] Pull back key figures (fig_flow_umap_cv, cluster-mean matrix | F_mk) and summarize to user
- [X] 2026-09-26 01:00: first full run died writing a 10.5 GB cache to haas ~/tmp (root fs 100% full); cache deleted,
      moved to NAS `rastermap_variants/_cache/` (atomic write), relaunched with `python -u` + EXIT_CODE logging
- [X] EDA script `exploratory-analyses/001_cluster_roc_eda.py`: metaclustering (user's run_meta_clustering),
      ROC join on (mouse_id, session_id, electrode_group, cluster_id), NaN-AUC -> not significant, choice direction
      inverted, decision/gated labels (roc_utils_new.compute_neuron_labels formulas), Fisher enrichment BH-FDR,
      heterogeneity (Cramér's V), rastermap matrix + metacluster strip + ROC-fraction columns; tested on _test
- [ ] EDA on the 4 variants (chained on haas after variant run: ~/tmp/chain_eda_20260926.sh, log ~/log/rastermap_roc_eda_full.log)
- [ ] CONSULT USER: EDA results, column choice, whether to test cluster enrichment at mouse level (confirmatory)
- [X] 2026-09-26: `rastermap_psth/population_matrix_summary.py` added to user's unit_spikes_analysis repo (~/code and NAS copy):
      fig5_population_matrix_ext (selectable columns: anatomy | waveform | area_group | area_acronym_custom | metacluster |
      7 ROC categories (fraction / mean selectivity) | reward F_mk share or figS4 panels), row_mode neuron / cluster_mean /
      cluster_mean_uniform, order rastermap / metacluster (+dendrogram, leaf order oriented to rastermap direction),
      reordering assessment, ROC-category PSTH sanity checks. Driver `exploratory-analyses/002_matrix_summary.py`.
      ROC definitions confirmed by user (gated = choice & hit_vs_spont & !whisker_sensory & !spont_licks_vs_cr;
      sensory = passive_pre|passive_post|active; modality = any of 3 wh_vs_aud epochs; either direction).
- [~] matrix_summary for qc_good__* variants (running); qc_good_mua__* chained after EDA (~/tmp/chain_matrix_summary_20260926.sh)
