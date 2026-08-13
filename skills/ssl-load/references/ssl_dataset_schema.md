## Purpose
Schema and on-disk contract for the four SSL datasets built by the
`ssl_ephys`, `ssl_behavior`, `ssl_ks2_ephys`, `ssl_ks2_behavior` modules under
`ibl_ai_agent/datasets/`. Verified directly against those builder modules and
the `ssl-ks2-dataset-description` project report (2026-08-06).

## Source and naming convention
- Source is a local directory of per-session NWB files (not ONE/Alyx), named
  `<subject>_<YYYYMMDD>_<HHMMSS>.nwb` (e.g. `AB116_20240724_102941.nwb`).
- **Two distinct spike-sorting sources**, both built from largely-overlapping
  session sets but genuinely different unit tables for the same physical
  session:
  - `ssl_ephys` / `ssl_behavior`: source `M:\analysis\Axel_Bisi\NWB_ks4` (Kilosort4).
  - `ssl_ks2_ephys` / `ssl_ks2_behavior`: source `M:\analysis\Axel_Bisi\NWB_combined` (Kilosort2/"combined").
  - Verified example: session `AB080_20230622_152205` has 892 units in KS2 vs
    939 in KS4. KS2's units table is missing `Lratio`/`isolationDistance`/
    `maxDriftEstimate`/`cumDriftEstimate` (present in KS4) and adds
    `bc_cluster_id` (not in KS4). Do not assume either is a superset of the other.
- Roughly a quarter of session files carry ephys (Neuropixels units +
  electrodes); the rest are behavior/training sessions with trials only.

## Tables (ssl_ephys / ssl_ks2_ephys — identical schema)
All under `metadata/*.parquet`:
- `sessions` (pk `session_id`): `subject_id`, `date`, `session_start_time`,
  `session_description` (encodes day, e.g. `"whisker_0"`, `"whisker_+2"`),
  `has_ephys`, `has_passive_epochs`, `n_units`, `n_trials`, `n_probes`, `duration_s`.
- `subjects` (pk `subject_id`): `sex`, `genotype`, `species`, `strain`, `date_of_birth`, `description`.
- `probes` (pk `session_id`, `probe_name`): `n_electrodes`, `device_name`.
- `electrodes` (pk `session_id`, `electrode_id`): per-electrode metadata, `probe_name`.
- `units` (pk `session_id`, `cluster_id`): per-unit metadata plus QC columns
  (`bc_label`, `ks_label`, ...) and CCF columns (`ccf_acronym`, `ccf_name`,
  `ccf_atlas_acronym`, `ccf_atlas_name`, `ccf_parent_acronym`, `ccf_parent_name`, ...).
  `cluster_id` is unique **within a session** by construction (see Cluster-id
  offsetting below); `raw_cluster_id` preserves KiloSort's original
  per-probe-only-unique id.
- `trials` (pk `session_id`, `trial_id`): raw NWB trials columns, including
  `trial_type` (`whisker_trial`/`auditory_trial`/`no_stim_trial`), `lick_flag`,
  `perf`, `context` (`active`/`passive`), `start_time`, `stop_time`. See
  `ssl-analyze/references/ssl_task_semantics.md` for outcome semantics.
- `events` (no single-column pk): `event_type`, `time` — sparse behavioral events.
- `epochs` (pk `session_id`, `epoch_name`): `start_time`, `stop_time` for
  `passive_pre`/`active`/`passive_post` (or just `active` for earlier subjects
  without recorded passive blocks). **Caveat:** derived differently per
  source — see "Epoch derivation differs by source" below — and has been
  observed unreliable (overlapping active/passive_post windows) for at least
  two sessions (AB119, AB120). Prefer the trial-level `context` column when
  the two disagree.

## Cluster-id offsetting (both ephys sources)
KiloSort `cluster_id` is only unique **per probe**, not per session — a
multi-probe session has real duplicate raw ids across probes (verified: raw
`cluster_id` 4 exists on both `imec0` and `imec1`). The builder offsets by
`probe_order_index * 1_000_000` to produce a session-unique `cluster_id`; the
untouched value is kept as `raw_cluster_id`. Always join spike shards and
`units.parquet` on the offset `cluster_id`, never `raw_cluster_id`, when more
than one probe is present.

## Spike shards
`stores.spikes` (path `spikes/<session_id>/meta.json + *.blosc`), one shard
per ephys session, delta-encoded like `bwm_ephys`. Decode with the dataset
module's own loader — do not hand-roll:
```python
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard      # KS4
from ibl_ai_agent.datasets.ssl_ks2_ephys import load_spike_shard  # KS2
shard = load_spike_shard(spikes_dir / session_id)
# shard["spike_times_seconds"], shard["spike_clusters"] (dense local index),
# shard["cluster_ids"] (offset cluster_id per dense index), shard["cluster_spike_counts"]
```
`spike_clusters` holds **dense local indices** into `cluster_ids`, not the
`cluster_id` values themselves — map through `cluster_ids[spike_clusters]` to
recover per-spike `cluster_id`.

## Epoch derivation differs by source
- `ssl_ephys`/`ssl_behavior` (KS4): epochs read directly from the NWB
  `processing/behavior/BehavioralEpochs/interval_series` interface when
  present. Confirmed rule (2026-08-13, see
  `ssl-analyze/references/ssl_behavioral_paradigm.md`'s Passive stimulation
  section): subjects numbered **AB116 and above** have `passive_pre`/
  `passive_post` recorded on every day with neural activity; subjects
  numbered **below AB116** never have passive epochs, on any day — this is a
  per-subject design fact, not a per-session artifact. A missing passive
  epoch for an AB116+ subject's ephys session is a data-quality flag worth
  checking, not expected. Still check per session before assuming, since the
  compressed table only reflects what NWB actually recorded.
- `ssl_ks2_ephys`/`ssl_ks2_behavior` (KS2): **no `BehavioralEpochs` interface
  exists in this source at all**, even for post-AB116 sessions where the KS4
  source has one. Epochs are instead derived from the trial-level `context`
  column (`active`/`passive`) plus trial order: `active` span = min/max
  active-trial times; `passive_pre` = passive trials before the first active
  trial; `passive_post` = passive trials after the last active trial. This is
  the same method validated in the `ssl-passive-sensory-selectivity` project.

## Behavior (keypoint tracking) tables — ssl_behavior / ssl_ks2_behavior
- `sessions` (pk `session_id`): `has_tracking`, `n_keypoints_available`, `n_keypoints_unavailable`.
- `keypoint_availability` (pk `session_id`, `keypoint_name`): `available` (bool), `n_frames`, `n_timestamps`, `timing_group`.
- `stores.tracking` (path `tracking/<session_id>/...`): DLC-style face/whisker/
  top-camera tracking — jaw, nose, tongue, pupil, **spout** (licking for
  reward on hits — see `ssl_task_semantics.md`), particle, whisker keypoints,
  each with a `_likelihood` companion.
- **Timestamp integrity is session-specific, never assumed.** Some sessions
  have no `BehavioralTimeSeries` at all (no video); some have valid
  equal-length `data`/`timestamps`; some have empty or mismatched-length
  `timestamps` (observed in real files). Only keypoints with
  `data_len == timestamps_len > 0` are compressed; the rest are `available=False`
  in `keypoint_availability.parquet`, not reconstructed from an assumed frame rate.
- Keypoints are grouped into "timing groups" sharing one delta-encoded
  timestamps array per group — do not assume a name-prefix (e.g. `top_`)
  predicts which keypoints share timing; check `timing_group` per keypoint.
- Decode with `load_tracking_shard` from the matching dataset module
  (`ssl_behavior` or `ssl_ks2_behavior`), never by reading shard arrays directly.

## Columns intentionally absent from the compressed datasets
These are common asks that require going outside `ssl_ephys`/`ssl_ks2_ephys`
— see `references/ssl_loading_policy.md` for the fallback path:
- `target_region` (raw electrode-group location string) — dropped when
  `electrode_group` was flattened to a name string during the build.
- `reward_group` / cohort — not derivable from the compressed tables alone;
  see `ssl-analyze/references/ssl_task_semantics.md` for both known sources.
- Any behavior-analysis-pipeline-specific derived column (e.g. `d_prime_w`,
  `area_acronym_custom`) — these live in the user's external M:-drive
  analysis code, not in this repo's dataset builders.
