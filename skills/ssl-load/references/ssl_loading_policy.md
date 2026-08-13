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
3. Pick KS4 (`ssl_ephys`) vs KS2 (`ssl_ks2_ephys`) deliberately and state the
   choice; do not default to one without checking whether the question cares
   about spike-sorting-pipeline-dependent counts.

## When the compressed dataset is insufficient — fallback rule
Several established SSL projects needed fields outside `ssl_ephys`/`ssl_ks2_ephys`
and used one of two fallback paths. Do not invent a third; pick the narrowest one:

**A. One-off metadata re-extraction from raw NWB** (cheap — metadata only, no
spike data), used when only a small field is missing:
- `target_region`: re-open the raw NWB file(s) for just the sessions/probes
  needed and read `electrode_group.location['area']` per
  `(session_id, probe_name)`, then merge onto `units.parquet`. Do not
  guess or backfill this field.
- `reward_group`/cohort from session metadata: re-open the raw NWB and parse
  `experiment_description` (a **stringified Python dict**, not JSON — use
  `ast.literal_eval`, not `json.loads`) for the `wh_reward` field. This is a
  small metadata read, safe to do per-session even at scale (verified: 111/111
  sessions parsed with no failures in `ssl-ks2-dataset-description`).

**B. The user's own external validated pipeline** (`combine_ephys_nwb` from
`M:\analysis\Axel_Bisi\unit_spikes_analysis\neural_utils.py`, plus
`allen_utils` from `M:\analysis\Axel_Bisi\Github\allen_utils`), used when the
analysis needs multiple derived fields this repo's dataset builders
intentionally don't produce (e.g. `area_acronym_custom`, `d_prime_w`,
cohort-corrected outcome labels via the `TRIAL_MAP`/`perf` lookup). This
requires that external repo's own Python environment (verified against
Python 3.14.3) and `sys.path` additions for both `unit_spikes_analysis` and
`allen_utils` — it is not part of this repo and not always available. Ask the
user before assuming this path is available in the current environment;
prefer path A or the compressed dataset when they suffice.

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
  source were used.
- Record the day filter explicitly: `day==0` ("learning", each subject's
  first ephys session) vs `day>0` ("expert", later sessions) — parsed from
  `sessions.session_description` (e.g. `"whisker_0"`). Note that `day==0` is
  exactly one session per subject for most subjects, while `day>0` sessions
  are unevenly distributed (some subjects contribute several) — use subject,
  not session, as the statistical unit whenever the expert arm is included.

## Quality gates
- Reject plans that read raw NWB spike data across many sessions when the
  compressed dataset already covers the question.
- Reject plans that mix KS4 and KS2 units for the same session without
  flagging it.
- Reject plans that report a cohort/reward_group result without stating
  which of the two sources it came from.
