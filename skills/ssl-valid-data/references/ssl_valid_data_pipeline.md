## Purpose
Exact function signatures and data-shape notes for the pipeline in `SKILL.md`'s
Workflow, sourced directly from
`M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\{helpers,neural_utils}\*.py`
(read 2026-09-11). Re-check against that source if a call errors — this
external codebase's API has already changed at least once (see
`process_allen_labels`'s `subdivide_areas`→`split_merge_areas` rename,
`ssl_artifact_dead_zone.md`'s own note about the same codebase).

## `data_utils.combine_ephys_nwb`
```python
def combine_ephys_nwb(nwb_list, day_to_analyze='learning', max_workers=24):
    """Combine neural and behavioural data from multiple NWB files.
    :param nwb_list: list of NWB file paths.
    :return: (trial_table, unit_table, ephys_nwb_list)
    """
```
`unit_table` gets a fresh `unit_id = unit_table.index` (global, not `cluster_id`)
after concatenation across sessions — every downstream join in a script using
this pipeline should key on `unit_id` for within-pipeline joins.

## `unit_metrics_utils.compute_presence_coverage_metrics`
```python
def compute_presence_coverage_metrics(unit_table):
    """Computes presence and coverage ratio to units."""
    unit_table['coverage_ratio'] = compute_coverage_ratio(unit_table)
    unit_table['presence_ratio'] = compute_presence_ratio(unit_table)
    return unit_table
```
Both sub-functions need a `spike_times` column (list/array of spike times per
unit) and a `session_id` column, and compute their per-unit ratio **against
that unit's own session's recording span** (grouped by `session_id`, not
pooled across sessions):
- `compute_presence_ratio`: fraction of 60s bins (recording start=earliest
  spike across the session's units, end=latest) containing >=1 spike, per
  unit.
- `compute_coverage_ratio`: fraction of the session's recording duration
  spanned by a unit's own first-to-last spike; 0.0 for <2 spikes.

Neither needs the NWB object itself, only spike times — so a script that
already has spike shards loaded (e.g. via `load_session_unit_spikes`) can
build the required `spike_times` column directly instead of re-reading NWB.

## `load_helpers_new.load_motion_dredge_shift_test_results`
```python
def load_motion_dredge_shift_test_results(nwb_files, day_to_analyze, experimenter='AB', max_workers=12):
    """Load per-session time-binned drift (motion) shift-test results
    (single_neuron_shift_test_figs.py output) from per-session CSVs, in parallel.
    :return: pandas.DataFrame, concatenated motion-drift shift-test results.
    """
```
Needs actual NWB file objects (uses `nwb_reader.get_mouse_id(nwb_file)`
internally) to resolve which per-session CSV to read, and an
`experimenter` flag (`'AB'` or `'MH'`) selecting the results root
(`ROOT_PATH_AXEL`/`ROOT_PATH_MYRIAM`) — this is the one step of the pipeline
that cannot currently be reproduced from the compressed `ssl_ephys` dataset
alone; it requires resolving each session's NWB path first.

Returned columns used downstream: `mouse_id, session_id, cluster_id,
electrode_group, p_conservative, r`. Rename `p_conservative` →
`drift_shift_test_pval`, add `drift_abs_r = r.abs()`, then merge onto
`unit_table` on `(mouse_id, session_id, cluster_id, electrode_group)` with
`validate='one_to_one'` — **cast `cluster_id` to `str` on both sides first**,
a dtype mismatch (e.g. int vs str) silently drops every match instead of
raising.

## `unit_metrics_utils.classify_units_quality`
```python
def classify_units_quality(
    unit_table: pd.DataFrame,
    thresholds: dict = None,               # defaults to DEFAULT_METRIC_THRESHOLDS below
    exclude: list = ['Lratio', 'isolationDistance', 'presenceRatio', 'maxDriftEstimate'],
    label_col: str = "quality_label",
) -> pd.DataFrame:
```
```python
DEFAULT_METRIC_THRESHOLDS = {  # (min, max); None = unbounded
    "nSpikes":                          (300,  None),
    "percentageSpikesMissing_gaussian": (None, 20),
    "fractionRPVs_estimatedTauR":       (None, 0.1),
    "maxDriftEstimate":                 (100,  1000),
    "presenceRatio":                    (0.7,  None),  # bombcell's own column
    "isolationDistance":                (20,   None),
    "Lratio":                           (None, 0.3),
    "presence_ratio":                   (0.5,  None),  # this pipeline's own (step above)
    "coverage_ratio":                   (0.9,  None),  # this pipeline's own (step above)
    "drift_shift_test_pval":            (0.01, None),
    "drift_abs_r":                      (0.5,  None),
}
```
Notes:
- A **NaN** value for any metric is **ignored** for that unit (does not count
  as a fail) — a unit missing a metric column entirely is not penalized for it.
- `drift_abs_r`/`drift_shift_test_pval` are evaluated as **one joint check**:
  a unit only fails the drift criterion if **both** are simultaneously out of
  range (passing either alone is enough) — protects against either metric
  being individually noisy. If either name is in `exclude`, or both columns
  are missing from `unit_table`, the whole joint check is skipped (with a
  `warnings.warn` if only one of the two is present — a config bug, not a
  silent skip in that specific case).
- The default `exclude` list already drops `Lratio`, `isolationDistance`,
  `presenceRatio`, `maxDriftEstimate` from the per-metric checks — only
  override this deliberately.
- Final label: `"good"` if `pass_mask`, else `"mua"`, **except** units where
  raw `bc_label == "non-soma"` always get `quality_label = "non-soma"`
  regardless of the other metrics (bombcell's non-soma call is never
  overridden — it is a different kind of judgment than the good/mua quality
  split).

## `allen_utils.process_allen_labels`
Current signature (2026-09): `process_allen_labels(df, split_merge_areas=False)`.
Requires columns `target_region, ccf_atlas_acronym, ccf_atlas_parent_acronym,
ccf_ap, ccf_ml, ccf_dv, mouse_id` (raises `ValueError` listing any missing
ones). `split_merge_areas=True` is what applies the custom subdivisions
(e.g. Somatosensory → whisker/orofacial/body) — pass it explicitly, the
default is `False`. Emits (harmless) pandas `SettingWithCopyWarning`s and a
`ccf label is nan` warning for units the atlas alignment failed on
(dataset-inherent, not a bug in this call) — check the warning's mouse list
against real ephys-QC exclusions before treating it as expected.

Area-group color palette: `allen_utils.get_custom_area_groups_colors()`
returns a `dict[group_name -> hex]`, keyed by the **current** (split) group
names. Use this, not a generic matplotlib color cycle, for any figure that
compares/overlays areas — see the cross-area-communication project's own
locked convention (`ssl_cross_area_rplus_rminus_colors` memory) for the
separate R+/R- cohort-color convention, which is unrelated to and layered on
top of this area-color palette.

## `data_utils.keep_shared_areas`
```python
def keep_shared_areas(data_df, nomenclature, n_min_units=5, n_min_mice=3):
```
- `reward_group` must be `1`/`0` (R+/R-), **not** the `"R+"`/`"R-"` strings
  used elsewhere in this dataset's own trial/session tables — convert before
  calling, or the reward_group==1/0 masks silently match nothing.
- QC mask used internally: `data_df['quality_label'].isin(['good', 'mua'])` —
  requires `quality_label` to already exist (run `classify_units_quality`
  first).
- Computes, per candidate area (intersection of areas seen in both reward
  groups): `n_units_rplus`/`n_units_rminus` (unique QC-passing `unit_id` per
  area) and `n_mice_rplus`/`n_mice_rminus` (unique `mouse_id` per area) —
  **all four must clear their respective thresholds** (`n_min_units` for
  both unit counts, `n_min_mice` for both mouse counts) for an area to survive.
- Prints the full accepted/removed-area breakdown with per-area unit/mouse
  counts on both sides — capture this if building a methods note, don't
  silently discard stdout.
- Returns `(filtered_df, ...)` — the exact second return value's shape/meaning
  was not read from source as part of this pass; check directly before relying
  on it (list of kept area names is a reasonable guess but unverified here).
