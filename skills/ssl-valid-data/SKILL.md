---
name: ssl-valid-data
description: Use this skill for Axel Bisi's canonical unit-quality (quality_label, not raw bc_label), Allen area-labeling, and shared-area-filtering pipeline for the SSL dataset (combine_ephys_nwb, drift-shift QC merge, presence/coverage metrics, classify_units_quality, process_allen_labels, keep_shared_areas) — applies across all SSL projects, not just one.
---

# SSL Valid Data

## Use this skill when
- Any SSL project is deciding which units count as "good" for an analysis — the canonical answer is `quality_label` (from `unit_metrics_utils.classify_units_quality`), **not** the raw bombcell `bc_label` alone.
- Building or refreshing a unit-area-label table (the SSL equivalent of `unit_area_labels.parquet`) for any SSL project.
- Restricting an R+/R- (or other cross-cohort) area-level comparison to a "shared areas" set — `data_utils.keep_shared_areas` is the canonical criterion for which areas are analysis-worthy across both cohorts.

## Do not use this skill when
- The question only needs whether a unit is `non-soma` per bombcell — that part of `bc_label` is never overridden by `quality_label` (see Workflow step 4).

## References
- `references/ssl_valid_data_pipeline.md`: exact function signatures, argument quirks (dtype/encoding traps), and what each pipeline step changes about the data, sourced directly from `M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\{helpers,neural_utils}\*.py`.

## Workflow
1. Load neural + trial data: `data_utils.combine_ephys_nwb(nwb_list, day_to_analyze=..., max_workers=...)` → `(trial_table, unit_table, nwb_neural_files)`. `unit_table` gets a fresh global `unit_id` (index-based, not `cluster_id` — every downstream join on this pipeline's tables should use `unit_id`, or the (mouse_id, session_id, cluster_id, electrode_group) key from `ephys_unit_level_join_keys` memory when working across pipelines).
2. Presence/coverage metrics: `unit_metrics_utils.compute_presence_coverage_metrics(unit_table)` adds `presence_ratio`/`coverage_ratio`, computed from each unit's own spike train against its session's recording span (see reference for the exact per-session-grouping semantics) — **this project's own already-loaded spike shards are a valid input** (no NWB re-read required) if `unit_table` doesn't carry a `spike_times` column; adapt by building that column from `load_session_unit_spikes`/`load_spike_shard` first.
3. Drift/motion QC (optional but recommended — see quality gate below): `load_helpers_new.load_motion_dredge_shift_test_results(nwb_neural_files, day_to_analyze=..., max_workers=...)` → merge onto `unit_table` on `(mouse_id, session_id, cluster_id, electrode_group)`, `validate='one_to_one'`, after casting `cluster_id` to `str` on both sides (dtype mismatch silently drops matches otherwise). Renames `p_conservative`→`drift_shift_test_pval`, adds `drift_abs_r = r.abs()`.
4. Quality classification: `unit_metrics_utils.classify_units_quality(unit_table, label_col='quality_label')` re-derives `quality_label` ('good'/'mua'/'non-soma') from a battery of bombcell + custom per-metric thresholds (`nSpikes`, `percentageSpikesMissing_gaussian`, `fractionRPVs_estimatedTauR`, `maxDriftEstimate`, `presenceRatio`, `isolationDistance`, `Lratio`, plus `presence_ratio`/`coverage_ratio` from step 2 and the joint `drift_abs_r`/`drift_shift_test_pval` check from step 3 if present). A unit's `bc_label=='non-soma'` always carries through unchanged; otherwise `quality_label` can disagree with `bc_label` in both directions (recovers some `mua`→`good`, can also demote). **Use `quality_label`, not `bc_label`, as the QC filter downstream** — this is the whole point of running this pipeline instead of reading `bc_label` off the compressed dataset directly. Expect this to pass *more* units through than `bc_label in {good, mua}` alone (richer per-unit feature count for population analyses), at the cost of extra compute to derive it.
5. Area labeling: `allen_utils.process_allen_labels(unit_table, split_merge_areas=True)` (current API name as of 2026-09 — was `subdivide_areas` in older code; if a script errors on the kwarg, re-check the M:-drive source directly, this API has already changed once). Produces `area_acronym_custom`; map to `area_group` via `allen_utils.get_custom_area_groups_from_name()`.
6. Shared-area filtering (cross-cohort comparisons only): `data_utils.keep_shared_areas(unit_table, nomenclature='area_acronym_custom' | 'area_group', n_min_units=5, n_min_mice=3)` — keeps only areas present in **both** reward groups with ≥`n_min_units` QC-passing (`quality_label` in {good, mua}) units **and** ≥`n_min_mice` distinct mice, checked **separately per reward group** (not pooled). Requires `reward_group` encoded as `1`/`0`, not the `"R+"`/`"R-"` strings used elsewhere in this dataset.
7. Before adopting any new area_group/area_acronym_custom value set project-wide, list the actual distinct values present (dataset-wide and per-session) and confirm with the user rather than assuming coverage — allen_utils' custom-group scheme has changed at least once already (Somatosensory split into whisker/orofacial/body, Cortical subplate activated, etc.), so a value list from months ago may be stale.

## Quality gates
- Reject using raw `bc_label` as the sole quality filter in any analysis that could instead run this pipeline and use `quality_label` — they disagree for a nontrivial number of units.
- Reject calling `keep_shared_areas` with `reward_group` as `"R+"`/`"R-"` strings — expects `1`/`0`, fails silently (wrong or empty shared-area set) otherwise.
- Reject skipping the drift-shift QC merge (step 3) silently — `classify_units_quality`'s joint drift check no-ops with only a warning when `drift_abs_r`/`drift_shift_test_pval` are absent, quietly loosening the classification; state explicitly whether the drift check ran.
- Reject an R+/R- (or other cross-cohort) area-level comparison that skips `keep_shared_areas` (or an equivalent per-cohort mouse-count check) — an area well-sampled in one cohort but sparse/absent in the other is not a valid basis for a cross-cohort comparison.
- Reject assuming last-known area_group/area_acronym_custom value lists are current without checking the live data first (step 7).
