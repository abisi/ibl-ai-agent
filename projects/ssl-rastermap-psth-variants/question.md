# ssl-rastermap-psth-variants

## Original request (2026-09-25)
Run the rastermap_psth analysis following the logic of `unit_spikes_analysis.py`, saving data and
clustering in separate, correctly labelled folders, for 4 variants: unit quality (good vs good+mua)
× learning_category (all vs good+moderate). Plot `fig_umap_flow`, and a version of the CV population
matrix averaged per cluster, side by side with the per-cluster fractional-representation (F_mk) stats.

## Definitions
- **Data**: SSL KS4 NWB (`NWB_ks4`), day 0 (learning stage), mice with exclude==0, exclude_ephys==0,
  reward_group in {R+, R-}, recording==1; AB068/AB077 excluded (as in unit_spikes_analysis.py).
- **Unit quality**: `quality_label` from `classify_units_quality` after drift-shift QC merge.
- **Learning category**: `joint_mouse_reference_weight.xlsx` `learning_category`; "all" = every
  category incl. bad/NaN; "good_moderate" = learners (pipeline default, hardcoded in build_feature_matrix).
- **fig_umap_flow**: `run_clustering_new.fig8_flow_umap` on even-trial (CV) data → `fig_flow_umap_cv`.
- **CV population matrix**: `fig5_population_matrix_cv` — even-trial PSTH matrix in the rastermap
  order fitted on odd trials.
- **F_mk**: fraction of mouse m's neurons in cluster k (`rastermap_utils.run_reward_group_stats`).
  R+ vs R- per cluster: Mann-Whitney U and Welch t, each BH-FDR across clusters.
