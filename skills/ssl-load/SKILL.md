---
name: ssl-load
description: Use this skill when loading Axel Bisi's single-session-learning (SSL) whisker/auditory NWB-derived datasets (ssl_ephys, ssl_behavior, ssl_ks2_ephys, ssl_ks2_behavior), preferring the local compressed dataset over raw NWB or external M:-drive pipelines.
---

# SSL Load

## Use this skill when
- A question needs data from the SSL whisker/auditory detection-task cohort (subjects named like `AB116`, session ids like `AB116_20240724_102941`).
- You need to choose between `ssl_ephys`/`ssl_ks2_ephys` (two different spike-sorting sources), `ssl_behavior`/`ssl_ks2_behavior` (keypoint tracking), and an external NWB/M:-drive pipeline.

## Do not use this skill when
- The question is about IBL/BWM data — use `ibl-load`. SSL is a separate, private, non-IBL dataset; do not mix SSL and BWM loading paths or area-grouping schemes in one analysis without flagging it.

## References
- `references/ssl_dataset_schema.md`: tables, spike/tracking shard contracts, and source differences (KS4 vs KS2, epochs vs `context` column).
- `references/ssl_loading_policy.md`: canonical loading policy — local dataset first, when to fall back to external tooling.
- `../ibl-load/references/brain_regions_qc.md`: still applicable for spike-sorting QC concepts, but SSL area labels use a different grouping scheme (see `ssl-analyze/references/ssl_analysis_patterns.md`).

## Workflow
1. Resolve the dataset root with `ibl_ai_agent.data_locations.resolve_dataset_dir("ssl_ephys" | "ssl_ks2_ephys" | "ssl_behavior" | "ssl_ks2_behavior")`, same mechanism as BWM.
2. Pick KS4 (`ssl_ephys`/`ssl_behavior`) vs KS2 (`ssl_ks2_ephys`/`ssl_ks2_behavior`) explicitly per `references/ssl_dataset_schema.md` — they are different spike-sorting outputs over largely-overlapping sessions, not interchangeable versions of the same table.
3. Read `metadata/*.parquet` tables directly with pandas; decode spike/tracking shards with the dataset module's own `load_spike_shard`/`load_tracking_shard` function (never hand-roll shard decoding).
4. Check whether the question needs a field the compressed dataset does not carry (`target_region`, `reward_group`/cohort, precise epoch boundaries) before reaching for raw NWB or the user's external M:-drive pipeline — see the fallback rule in `references/ssl_loading_policy.md`.
5. State which source (KS4/KS2), which dataset version, and which cohort/day filter was used in Methods.

## Quality gates
- Never silently pool `ssl_ephys` and `ssl_ks2_ephys` units for the same session — they come from different spike-sorting runs with different unit counts/QC columns for the same physical recording.
- Do not treat `epochs.parquet` boundaries as authoritative for passive_pre/active/passive_post splits without cross-checking the trial-level `context` column; some sessions have overlapping epoch windows.
- State explicitly whether `reward_group`/cohort came from session metadata (`wh_reward`) or from the external `joint_mouse_reference_weight.xlsx` reference sheet — they are different sources with different semantics (see `ssl-analyze/references/ssl_task_semantics.md`).
