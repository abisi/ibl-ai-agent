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
- `references/ssl_dataset_inclusion.md`: canonical SSL dataset record — sources (raw NWB_ks4, mouse sheet), inclusion rules (exclude flags, `recording` for day 0 only, mouse-level cohort), known metadata errors and decisions, current unit-table version and paths. Read before any session-inclusion decision and update it whenever a dataset decision is made.
- `references/ssl_valid_data_pipeline.md`: exact function signatures, argument quirks (dtype/encoding traps), and what each pipeline step changes about the data, sourced directly from `M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\{helpers,neural_utils}\*.py`.

## Workflow
1. Load neural + trial data: `data_utils.combine_ephys_nwb(nwb_list, day_to_analyze=..., max_workers=...)` → `(trial_table, unit_table, nwb_neural_files)`. `unit_table` gets a fresh global `unit_id` (index-based, not `cluster_id` — every downstream join on this pipeline's tables should use `unit_id`, or the (mouse_id, session_id, cluster_id, electrode_group) key from `ephys_unit_level_join_keys` memory when working across pipelines).
2. Presence/coverage metrics: `unit_metrics_utils.compute_presence_coverage_metrics(unit_table)` adds `presence_ratio`/`coverage_ratio`, computed from each unit's own spike train against its session's recording span (see reference for the exact per-session-grouping semantics) — **this project's own already-loaded spike shards are a valid input** (no NWB re-read required) if `unit_table` doesn't carry a `spike_times` column; adapt by building that column from `load_session_unit_spikes`/`load_spike_shard` first.
3. Drift/motion QC (optional but recommended — see quality gate below): `load_helpers_new.load_motion_dredge_shift_test_results(nwb_neural_files, day_to_analyze=..., max_workers=...)` → merge onto `unit_table` on `(mouse_id, session_id, cluster_id, electrode_group)`, `validate='one_to_one'`, after casting `cluster_id` to `str` on both sides (dtype mismatch silently drops matches otherwise). Renames `p_conservative`→`drift_shift_test_pval`, adds `drift_abs_r = r.abs()`.
4. Quality classification: `unit_metrics_utils.classify_units_quality(unit_table, label_col='quality_label')` re-derives `quality_label` ('good'/'mua'/'non-soma') from a battery of bombcell + custom per-metric thresholds (`nSpikes`, `percentageSpikesMissing_gaussian`, `fractionRPVs_estimatedTauR`, `maxDriftEstimate`, `presenceRatio`, `isolationDistance`, `Lratio`, plus `presence_ratio`/`coverage_ratio` from step 2 and the joint `drift_abs_r`/`drift_shift_test_pval` check from step 3 if present). A unit's `bc_label=='non-soma'` always carries through unchanged; otherwise `quality_label` can disagree with `bc_label` in both directions (recovers some `mua`→`good`, can also demote). **Use `quality_label`, not `bc_label`, as the QC filter downstream** — this is the whole point of running this pipeline instead of reading `bc_label` off the compressed dataset directly. Expect this to pass *more* units through than `bc_label in {good, mua}` alone (richer per-unit feature count for population analyses), at the cost of extra compute to derive it.
5. Area labeling: `allen_utils.process_allen_labels(unit_table, split_merge_areas=True)` (current API name as of 2026-09 — was `subdivide_areas` in older code; if a script errors on the kwarg, re-check the M:-drive source directly, this API has already changed once). Produces `area_acronym_custom`; map to `area_group` via `allen_utils.get_custom_area_groups_from_name()`.
6. Shared-area filtering (cross-cohort comparisons only): `data_utils.keep_shared_areas(unit_table, nomenclature='area_acronym_custom' | 'area_group', n_min_units=5, n_min_mice=3)` — keeps only areas present in **both** reward groups with ≥`n_min_units` QC-passing (`quality_label` in {good, mua}) units **and** ≥`n_min_mice` distinct mice, checked **separately per reward group** (not pooled). Requires `reward_group` encoded as `1`/`0`, not the `"R+"`/`"R-"` strings used elsewhere in this dataset.
7. Before adopting any new area_group/area_acronym_custom value set project-wide, list the actual distinct values present (dataset-wide and per-session) and confirm with the user rather than assuming coverage — allen_utils' custom-group scheme has changed at least once already (Somatosensory split into whisker/orofacial/body, Cortical subplate activated, etc.), so a value list from months ago may be stale.

## Unit sets (user definition, 2026-10-05)
Three nested sets; every analysis states which one it uses.
1. **All units**: `quality_label` in {good, mua} (bombcell `non-soma` excluded).
2. **Stable units** (good or mua): pass exactly the three stability criteria of `unit_metrics_utils.classify_units_quality`
   (`DEFAULT_METRIC_THRESHOLDS`): `coverage_ratio` >= 0.9, `presence_ratio` >= 0.5, and independence from probe drift --
   the DREDge drift-shift joint test (Harris 2021 shift test, `unit_fr_motion_shift_test_harris.py`) fails only when the
   rate-motion correlation is both large and significant (`drift_abs_r` > 0.5 AND `drift_shift_test_pval` < 0.01). No other
   criterion enters stability (spike count, amplitude cut-off, refractory violations and isolation are not stability).
   A unit without a drift result is NOT stable (independence cannot be shown); the test is undefined for units with no spikes
   in its central segment (all but the first / last 500 s), i.e. units that appear or disappear during the session.
3. **Good units**: `quality_label == 'good'` AND stable. Always intersect with the stable set: `quality_label` tables built
   without the drift merge (e.g. `reports/ssl_analysis/derived/unit_area_labels.parquet`, from `compute_ssl_quality_label.py`)
   carry no drift check.
   Why the intersection, checked 2026-10-05: `classify_units_quality` (DEFAULT_METRIC_THRESHOLDS; exclude = Lratio,
   isolationDistance, presenceRatio, maxDriftEstimate) labels a unit good only if nSpikes >= 300, spikes missing <= 20 %,
   fractionRPVs <= 0.1, presence_ratio >= 0.5, coverage_ratio >= 0.9 and the drift joint check pass -- so good implies
   stable when the drift results were merged (v2 table: 43114 / 43114 good units stable). But (a) a NaN metric counts as a
   pass, so a unit without a drift result can be labelled good, and (b) label tables built without the drift merge skip the
   drift check silently (`unit_area_labels.parquet`: 745 good units fail the drift test).
Tracked analyses add a per-analysis rate criterion (e.g. >= 0.5 Hz in every analysed epoch) on top of the set.
Implementation: `projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/137_stable_units.py` ->
`combined_results_ks4/ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units.parquet` (columns `unit_set_all`,
`stable`, `good`; join keys mouse_id, session_id, electrode_group, cluster_id; dataset cluster_id = probe index * 1e6 + NWB
cluster id). Drift results: per-session CSVs in `combined_results_ks4/<mouse>/<behaviour>_<day>/single_neuron_motion_shift_test/`
plus units tested later in `combined_results_ks4/_drift_rerun_20261005/`; the test needs an env with spikeinterface
(haas: `~/code/unit_spikes_analysis/.venv`).

## Quality gates
- Reject a "good" or "stable" unit set that does not apply all three stability criteria (coverage, presence, drift joint test),
  that adds other criteria under the name "stable", or that admits units without a drift result.
- Reject using raw `bc_label` as the sole quality filter in any analysis that could instead run this pipeline and use `quality_label` — they disagree for a nontrivial number of units.
- Reject calling `keep_shared_areas` with `reward_group` as `"R+"`/`"R-"` strings — expects `1`/`0`, fails silently (wrong or empty shared-area set) otherwise.
- Reject skipping the drift-shift QC merge (step 3) silently — `classify_units_quality`'s joint drift check no-ops with only a warning when `drift_abs_r`/`drift_shift_test_pval` are absent, quietly loosening the classification; state explicitly whether the drift check ran.
- Reject an R+/R- (or other cross-cohort) area-level comparison that skips `keep_shared_areas` (or an equivalent per-cohort mouse-count check) — an area well-sampled in one cohort but sparse/absent in the other is not a valid basis for a cross-cohort comparison.
- Reject assuming last-known area_group/area_acronym_custom value lists are current without checking the live data first (step 7).
- Reject taking the R+/R- cohort from a session-level field (unit-table `reward_group`, NWB `wh_reward`) without checking it against the mouse reference sheet. The cohort is a property of the mouse: take it from `M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx` (`reward_group`; haas `/mnt/share_internal/...`; only `R+`/`R-`, other groups such as `R+proba` excluded; mice flagged `exclude` or `exclude_ephys` excluded), and report every session whose own label disagrees or is missing. Known session-label errors (2026-10-03): MH035 day 0 and MH064 day +1 carry `wh_reward = 1` but both mice are R-. Also check that no mouse ends up in two cohorts and that sessions without a unit-table label (no quality / area metadata) are reported rather than silently dropped.
