# SSL ROC across learning stages

Single-neuron ROC selectivity (rate-based, `roc_utils_new`) from the learning day (day 0) to expert days, per cohort
(R+ / R-), area group and analysis type. Results (not in git): `combined_results_ks4/ssl-roc-learning-stages/`
(old path `_roc_stage_analysis` is a symlink). Moved here from `ssl-rastermap-psth-variants` on 2026-10-05.

| Script | What it does |
|---|---|
| 045_roc_stage_table.py | master tables: `units.parquet`, `roc_long.parquet` (one row per unit x ROC type) |
| 046_roc_stage_psth.py | PSTHs per session for the stage figures |
| 047_roc_stage_stats.py | statistics per ROC type and region (pooled over units; 2026-10-02 version) |
| 048_roc_stage_type_figures.py | one page per ROC type |
| 049_roc_stage_ccf.py | CCF density maps |
| 050_roc_stage_summary.py | summary figures |
| 051_stage_change.py | COSYNE figures: change of the fraction of responsive units and of mean |selectivity| from learning to expert, whisker vs auditory (`whisker_active`, `auditory_active`), back-to-back dot plots per cohort, ANOVA stage x area, focality (concentration index sum(x^2)/(sum x)^2) of whisker responsiveness across area groups; `--population all|learners` |

Dependents: the master tables are read by ssl-sensory-spatial-maps (003), ssl-stimulus-arrival-decoding (001) and
ssl-prelick-convergence (053); ssl-prelick-convergence keeps its own copies of 045 / 047-049 (imported by its scripts).

## Decisions (user, 2026-10-05)

- Units: good + mua; session x area group with >= 5 tested units; an area group is kept within a cohort if it has >= 3
  sessions at both stages; values averaged over sessions (unit of analysis: session); cohort per mouse from the
  reference sheet; populations all mice and learners.
- Responsiveness: `whisker_active` / `auditory_active` significant (p < 0.05). (A passive pre AND post, Bonferroni
  version was tried first: `passive_stage_change/`.)
- Focality: concentration index sum(x^2) / (sum x)^2 across included area groups (replaces Gini / entropy).

## TODO

- [ ] Re-run 047-049 on the current tables (the 2026-10-02 `stats/` and `figures/types` predate the v2 unit table,
      the per-mouse cohort labels and the 2026-10-04 context-fix ROC re-run).
- [ ] Choose the COSYNE panels (metric, population) and write captions.
