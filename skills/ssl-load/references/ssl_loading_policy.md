## Purpose
Canonical loading policy for SSL questions, mirroring `ibl-load/references/bwm_runtime_policy.md`'s local-first pattern.

## Scope
Use this as the shared policy reference for `ssl-load` and `ssl-analyze`.

## Core policy
1. Resolve the local compressed dataset root first, same mechanism as BWM:
   ```python
   from ibl_ai_agent.data_locations import resolve_dataset_dir
   d = resolve_dataset_dir("ssl_ephys")       # or ssl_ks2_ephys / ssl_behavior / ssl_ks2_behavior
   ```
2. Prefer the compressed local dataset (`metadata/*.parquet` + shard loader)
   over raw NWB reads or the user's external M:-drive pipeline whenever the
   compressed schema covers the question — see `ssl-load/references/ssl_dataset_schema.md`
   for exactly which fields exist.
3. **KS4 (`ssl_ephys`/`ssl_behavior`) is the default SSL spike-sorting source**
   (Axel Bisi, 2026-08-17). Use KS2 (`ssl_ks2_ephys`/`ssl_ks2_behavior`) only
   when the question explicitly needs it — e.g. reproducing a specific prior
   KS2-based project (`ssl-ks2-*`), or a question about spike-sorting-pipeline
   differences themselves. State the choice either way.

## The compressed dataset build is independent of `ephys_utilities` — do not assume otherwise
**Verified directly against source 2026-08-26.** `ibl_ai_agent/datasets/ssl_ephys.py`
and `ssl_behavior.py` — the code that actually builds/compresses `ssl_ephys`/
`ssl_behavior` from raw NWB (invoked via `scripts/run_ssl_ks4_build.ps1`) —
read raw NWB directly with their own `_read_one_session`/`_list_session_files`
logic and **never import or call `ephys_utilities`**. Consequences:
- `units.parquet`'s `bc_label`/`ks_label` columns are whatever bombcell/KiloSort
  already wrote into the raw NWB at spike-sorting time, carried through
  unchanged. They are **not** produced by `classify_units_quality`, and
  rebuilding the compressed dataset does not run `classify_units_quality` or
  `process_single_nwb` — re-running `run_ssl_ks4_build.ps1` alone will never
  make the compressed dataset start using them.
- The Mandatory Path B rule below therefore governs **analysis-time / raw-NWB
  loading only** (Path A and Path B, described next) — not the compressed
  dataset builders themselves. If a future task needs the builders to also
  apply `classify_units_quality` and produce a `quality_label` column in
  `units.parquet` directly, that is a separate, larger code change to
  `ssl_ephys.py`/`ssl_behavior.py` (new column, provenance/schema updates) and
  must be scoped and confirmed with the user explicitly before being made —
  do not fold it into a "rebuild the dataset" request silently.
- `bc_label` (compressed dataset) and `quality_label` (Path B, see below)
  remain two independent, non-interchangeable QC labels that can disagree
  per-unit — always state which one an analysis used.

## When the compressed dataset is insufficient — fallback rule
Several established SSL projects needed fields outside `ssl_ephys`/`ssl_ks2_ephys`
and used one of two fallback paths. Do not invent a third; pick the narrowest one:

**A. One-off metadata re-extraction from raw NWB** (cheap — metadata only, no
spike data), used when only a small field is missing:
- `target_region`: re-open the raw NWB file(s) for just the sessions/probes
  needed and read `electrode_group.location['area']` per
  `(session_id, probe_name)`, then merge onto `units.parquet`. Do not
  guess or backfill this field. **Only needed when staying compressed-dataset-only.**
  If Path B (`process_single_nwb`/`combine_ephys_nwb`) is already in use for
  the same analysis, `target_region` comes for free — `process_single_nwb`
  calls `convert_electrode_group_object_to_columns` internally, which already
  adds `unit_table['target_region']` from `electrode_group.location['area']`
  (verified directly against source 2026-08-26, `data_utils.py`). Do not
  re-derive it a second time via a separate raw-NWB reopen in that case.
- `reward_group`/cohort from session metadata: re-open the raw NWB and parse
  `experiment_description` (a **stringified Python dict**, not JSON — use
  `ast.literal_eval`, not `json.loads`) for the `wh_reward` field. This is a
  small metadata read, safe to do per-session even at scale (verified: 111/111
  sessions parsed with no failures in `ssl-ks2-dataset-description`).

**B. The user's own external validated pipeline** — `ephys_utilities`, now at
`M:\analysis\Axel_Bisi\Github\ephys_utilities` (a `uv`/pip-installable package,
top-level import name `ephys_utilities`; **the older bare clone at
`C:\Users\bisi\Github\ephys_utilities` is a stub — README/`.gitignore` only,
no code — do not use it**), plus `allen_utils` from
`M:\analysis\Axel_Bisi\Github\allen_utils`. Used when the analysis needs
multiple derived fields this repo's dataset builders intentionally don't
produce (e.g. `area_acronym_custom`, `d_prime_w`, cohort-corrected outcome
labels via the `TRIAL_MAP`/`perf` lookup, unit drift/quality QC, cortical
hierarchy scores). This requires that external repo's own Python environment
and `sys.path`/import setup for `ephys_utilities` and `allen_utils` — it is
not part of this repo and not always available. Ask the user before assuming
this path is available in the current environment; prefer path A or the
compressed dataset when they suffice.

### Module layout inside `ephys_utilities` (verified directly against source 2026-08-26)
The package was refactored during August 2026; two functions this policy
depends on moved out of `neural_utils/neural_utils.py` into their own modules.
Re-verify against source before trusting any of this again (see the
re-verify rule below) rather than assuming it's stable:
- `ephys_utilities.helpers.data_utils` — **`process_single_nwb`** (per-session
  NWB loader/worker) and **`combine_ephys_nwb`** (the multiprocessing wrapper
  that calls `process_single_nwb` over a file list) — moved here from
  `neural_utils.py`. Also `keep_active_trials`, `keep_active_from_whisker_onset`,
  `keep_shared_areas`, `keep_units_params`, etc.
- `ephys_utilities.neural_utils.unit_metrics_utils` — **`classify_units_quality`**
  (also moved from `neural_utils.py`), plus `compute_presence_coverage_metrics`,
  `compute_presence_ratio`, `compute_coverage_ratio`, `merge_unit_quantifications`.
- `ephys_utilities.neural_utils.neural_utils` — everything else this skill
  already cites still lives here unchanged: `TRIAL_MAP`, `compute_unit_peri_event_histogram`,
  `compute_trial_baseline_from_peth` (see `ssl-analyze/references/ssl_artifact_dead_zone.md`),
  and the PETH-building functions. Do not move these citations to `data_utils`/
  `unit_metrics_utils` — only the loading and quality-classification functions moved.
- `ephys_utilities.allen_utils.allen_utils` — unchanged (`process_allen_labels`,
  `get_custom_area_groups_from_name`, `merge_liu_avg_ipsi`,
  `merge_hierarchy_columns_from_gao`, `merge_hierarchy_from_harris`).

**Priority rule** (Axel Bisi, 2026-08-22, module paths corrected 2026-08-26):
once a question needs Path B at all (i.e. path A and the compressed dataset
don't cover it), these `ephys_utilities` functions are **mandatory — the
sole canonical source for that processing, at all times**:
- **Loading**: `data_utils.process_single_nwb` (directly, or via
  `data_utils.combine_ephys_nwb` for multi-session runs) — never a hand-rolled
  NWB read, a different/older module (e.g. `neural_utils_old.py`), or a
  reimplementation that skips its day-filtering/column-derivation logic.
- **Quality classification**: `unit_metrics_utils.classify_units_quality` —
  never a hand-rolled reimplementation, a different/older module, or the
  plain `bc_label`/`area_acronym_custom` compressed-dataset columns as a
  substitute for the fuller `quality_label`/hierarchy fields this pipeline
  produces.
Do not swap in an equivalent-looking function from elsewhere without
checking with the user first.

### `process_single_nwb`'s `day_to_analyze` default
`process_single_nwb`/`combine_ephys_nwb` only filter by day when
`day_to_analyze` is exactly the string `'learning'`, `'expert'`, or `'all'`
(each checked with `==` against a specific string) — any other value (e.g.
`0`, `None`) matches none of the three branches and silently disables day
filtering entirely (every session included regardless of day). **Fixed
2026-08-26** (Axel Bisi): both functions' default was `0` — a silent
all-days no-op, easy to misread as "day 0 only" — and has been changed in
source to default to `'learning'`, this project's intended default, in
`M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\helpers\data_utils.py`.
Re-verify this against source if it's ever cited again (see the re-verify
rule below) — it's a live, actively-developed clone. Regardless of the
default, **always state in the analysis report which of
`'learning'`/`'expert'`/`'all'` was actually used** for a given result;
passing anything other than one of those three strings (including `0`) is
still a silent no-op even now that the default itself is fixed.

### Mandatory filter trace for `classify_units_quality`
Every time `classify_units_quality` is called, the analysis report must
record, verbatim, the filters actually in effect — not just that the
function was called:
- **`thresholds`** used — if the default (`unit_metrics_utils.DEFAULT_METRIC_THRESHOLDS`),
  say so explicitly by name; if a custom dict was passed, log the full dict.
  As of 2026-08-26 the default is: `nSpikes>=300`,
  `percentageSpikesMissing_gaussian<=20`, `fractionRPVs_estimatedTauR<=0.1`,
  `maxDriftEstimate` in `[100, 1000]`, `presenceRatio>=0.7`,
  `isolationDistance>=20`, `Lratio<=0.3`, this pipeline's own
  `presence_ratio>=0.5` and `coverage_ratio>=0.9`, and the joint drift pair
  `drift_shift_test_pval>=0.01` / `drift_abs_r>=0.5` (see below). A NaN metric
  value for a unit is *ignored* for that unit (not an automatic fail) —
  state this if it materially affects which units pass.
- **`exclude`** list actually used — the function's own default is
  `['Lratio', 'isolationDistance', 'presenceRatio', 'maxDriftEstimate']`; if a
  different list was passed, log it. Metrics in `exclude` are skipped
  entirely for *all* units.
- **`label_col`** — the output column name (commonly `quality_label`); state
  it so it isn't confused with `bc_label`.
- **The joint drift criterion** — `drift_abs_r` and `drift_shift_test_pval`
  are evaluated jointly: a unit only fails if *both* are simultaneously out
  of range (passing either alone is enough to pass). State whether this
  joint check ran (both columns present and neither name excluded) or was
  skipped (a column missing, or either name in `exclude`) — this changes how
  aggressively drift alone can exclude units.
Do not report a `classify_units_quality`-derived result ("N good units",
"units after QC") without this trace; a bare unit count is not reproducible
without knowing which of these varied.

**Re-verify against source before relying on it, every session** (Axel
Bisi, 2026-08-22): `ephys_utilities` and `allen_utils` are both live git
clones the user actively develops
(`M:\analysis\Axel_Bisi\Github\ephys_utilities`, remote
`LSENS-BMI-EPFL/ephys_utilities`; `M:\analysis\Axel_Bisi\Github\allen_utils`)
— not static vendored copies. This is exactly why the module-layout
correction above was needed: prior notes (including a now-stale `combine_ephys_nwb`/
`classify_units_quality` location in `neural_utils.py` and a since-abandoned
bare clone path) went stale as the repo evolved. Before trusting a function
signature/path recorded in this skill (including everything in the recipe
below), re-check it against the live file rather than assuming last time's
answer still holds:
```bash
git -C "M:/analysis/Axel_Bisi/Github/ephys_utilities" status
git -C "M:/analysis/Axel_Bisi/Github/allen_utils" status
```
If a repo is behind `origin/main`, ask the user before running `git pull`
rather than doing it automatically — do **not** pull a repo with
uncommitted local changes without the user's go-ahead, since a pull can
conflict with their in-progress edits.

**Full metadata recipe** (module paths corrected 2026-08-26 against live
source; supersedes the `neural_utils.combine_ephys_nwb`/
`neural_utils.classify_units_quality` recipe used before this correction —
the canonical way to get a `trial_table`/`unit_table` pair with every derived
field this repo's own dataset builders omit):
```python
from ephys_utilities.helpers import data_utils
from ephys_utilities.helpers import load_helpers
from ephys_utilities.neural_utils import unit_metrics_utils
from allen_utils import allen_utils

# 1. Combine raw NWB files into one trial_table/unit_table pair.
#    day_to_analyze MUST be 'learning' / 'expert' / 'all' -- see the
#    day_to_analyze default section above; a bare 0/None silently disables day filtering.
trial_table, unit_table, ephys_nwb_list = data_utils.combine_ephys_nwb(
    nwb_list, day_to_analyze='all', max_workers=24)

# 2. Motion/drift QC (DREDge shift-test results, per-session CSVs).
dredge_df = load_helpers.load_motion_dredge_shift_test_results(nwb_neural_files)
dredge_df = dredge_df[['mouse_id', 'session_id', 'cluster_id', 'electrode_group',
                        'p_conservative', 'r']].rename(columns={'p_conservative': 'drift_shift_test_pval'})
dredge_df['drift_abs_r'] = dredge_df['r'].abs()
unit_table['cluster_id'] = unit_table['cluster_id'].astype(int)
dredge_df['cluster_id'] = dredge_df['cluster_id'].astype(int)
unit_table = unit_table.merge(dredge_df, on=['mouse_id', 'session_id', 'cluster_id', 'electrode_group'],
                               how='left', validate='one_to_one')

# 3. Presence/coverage metrics -- unit_metrics_utils (moved from neural_utils.py).
unit_table = unit_metrics_utils.compute_presence_coverage_metrics(unit_table)

# 4. MANDATORY quality classification -- always unit_metrics_utils.classify_units_quality,
#    never a hand-rolled equivalent. Record thresholds/exclude/label_col/joint-drift
#    status in the analysis report every time -- see the filter-trace rule above.
thresholds_used = unit_metrics_utils.DEFAULT_METRIC_THRESHOLDS  # or log the literal override dict
exclude_used = ['Lratio', 'isolationDistance', 'presenceRatio', 'maxDriftEstimate']  # default; log if overridden
unit_table = unit_metrics_utils.classify_units_quality(
    unit_table, thresholds=thresholds_used, exclude=exclude_used, label_col='quality_label')

# 5. Allen custom area labels and anatomical hierarchy scores.
#    process_allen_labels's real parameter is split_merge_areas (default
#    False) -- NOT subdivide_areas, which does not exist on this function
#    and was miscited in at least two prior SSL projects' question.md files
#    (verified directly against source 2026-08-22, corrects those citations).
unit_table = allen_utils.process_allen_labels(unit_table)
unit_table = allen_utils.merge_liu_avg_ipsi(unit_table)                    # adds avg_ipsi
unit_table = allen_utils.merge_hierarchy_columns_from_gao(unit_table)      # nearest-column hierarchy (cortex only)
unit_table = allen_utils.merge_hierarchy_from_harris(unit_table)           # Harris et al. hierarchy scores
```
Two non-interchangeable QC labels result: `bc_label` (this repo's default
compressed-dataset column, baked in from bombcell/KiloSort at spike-sorting
time -- see "The compressed dataset build is independent of `ephys_utilities`"
above) vs `quality_label` (the custom reclassification from step 4) -- state
which one an analysis uses, same rule as the cohort/reward_group source
ambiguity below.

Do not silently fall back to raw per-session NWB reads for large-scale spike
data — that defeats the purpose of the compressed dataset and can mean tens
of GB if loaded across all sessions at once. Reserve raw NWB reads for small
metadata fields (path A above).

## Cohort/reward_group source ambiguity — state which one was used
Two different sources have been used for "cohort" across prior SSL projects
and are **not** interchangeable:
- Session-metadata `wh_reward` field (path A above) — binary, per-session,
  extracted directly from the NWB the session came from.
- `joint_mouse_reference_weight.xlsx` (`M:\share_internal\Axel_Bisi_Share\dataset_info\`,
  sheet `Sheet1`) — per-mouse reference sheet with `reward_group` levels
  `R+`/`R-`/`R+proba`, plus `learning_category` (`good`/`moderate`/`bad`/NaN)
  and `exclude_ephys` flags used for cohort/quality filtering in the
  `ssl-reward-history-modulation` project.
Always name which source was used; do not assume they agree on every mouse.

## Reproducibility rule
- Record which spike-sorting source (KS4/KS2), dataset version, and cohort
  source were used — KS4 unless stated otherwise (see Core policy above).
- Record the day filter explicitly: `day==0` ("learning", each subject's
  first ephys session) vs `day>0` ("expert", later sessions) — parsed from
  `sessions.session_description` (e.g. `"whisker_0"`). Note that `day==0` is
  exactly one session per subject for most subjects, while `day>0` sessions
  are unevenly distributed (some subjects contribute several) — use
  `session_id`, not `mouse_id`, as the statistical unit whenever the expert
  arm is included; see `ssl-analyze/references/ssl_task_semantics.md`'s
  Day / training-stage semantics section for the full rule.

## Quality gates
- Reject plans that read raw NWB spike data across many sessions when the
  compressed dataset already covers the question.
- Reject plans that mix KS4 and KS2 units for the same session without
  flagging it.
- Reject plans that report a cohort/reward_group result without stating
  which of the two sources it came from.
- Reject plans that use the Path B `quality_label` column (or `bc_label`)
  for unit QC without stating which of the two was used — they are
  independent classifications and can disagree per-unit.
- Reject any Path B path (`data_utils.py`, `unit_metrics_utils.py`,
  `neural_utils.py`, `allen_utils.py`, etc.) copied from an older project's
  notes without re-verifying it exists — this policy's own function
  locations were wrong (pointing at `neural_utils.py` for functions that had
  already moved) until checked directly 2026-08-26.
- Reject substituting a hand-rolled equivalent, a different module, or a
  bare compressed-dataset column (`bc_label`, `area_acronym_custom`) for the
  `ephys_utilities`/`allen_utils` functions once Path B is already in use —
  those functions are the mandatory, priority source for that processing,
  per the Priority rule above.
- Reject any `classify_units_quality`-derived result reported without its
  filter trace (`thresholds`, `exclude`, `label_col`, joint-drift status) —
  see the Mandatory filter trace section above.
- Reject a Path B load that uses `process_single_nwb`'s bare/default
  `day_to_analyze` (i.e. not one of `'learning'`/`'expert'`/`'all'`) while
  believing a day filter was applied — see the `day_to_analyze` default section above.
- Reject trusting a Path B function signature/path from this file (or from
  memory of a prior session) without a fresh `git status` check on the
  relevant repo first — they are live, actively-developed clones, not
  vendored snapshots; do not force `git pull` on a repo with uncommitted
  local changes without asking the user first.
- Reject any claim that rebuilding `ssl_ephys`/`ssl_behavior` (running
  `run_ssl_ks4_build.ps1`) applies `classify_units_quality` or
  `process_single_nwb` to the compressed dataset — it does not; see
  "The compressed dataset build is independent of `ephys_utilities`" above.
