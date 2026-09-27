---
name: ssl-lick-alignment
description: Use this skill whenever an SSL analysis aligns to, windows around, or computes reaction times from a trial's first lick (`trials.lick_time`, local `ssl_ephys`/`ssl_behavior` or raw NWB). NWB `lick_time` is late by the per-session artifact window; this gives the correction, why, checks, and clock caveats for other lick sources.
---

# SSL Lick Alignment

## Use this skill when
- Aligning spikes, PSTHs, decoders, video or behaviour to the first lick of a trial, or computing reaction times, from the SSL `trials` table (`reports/datasets/ssl_*/1.0.0/metadata/trials.parquet` or NWB `trials`).
- Comparing trial licks with other lick sources (piezo `events`, spontaneous-lick CSVs, DLC licks).

## Rule
Never align to `lick_time` as stored. Compute, per trial:

```python
trials["reaction_time"] = trials["lick_time"] - trials["response_window_start_time"]  # s, from stimulus onset
trials["first_lick_time"] = trials["start_time"] + trials["reaction_time"]            # physical first lick
```

- Compute it per trial: the artifact window (`response_window_start_time - start_time`) is a session parameter and can differ between sessions. Never hardcode 100 ms (the value in all 62 day-0 learner sessions checked).
- Add new columns; keep `lick_time` as stored, and record the correction in outputs/provenance.
- Reaction time = `reaction_time` above, not `lick_time - start_time` (~100 ms too long).

## Why
- `behavior_control/main_control.m`: `reaction_time = hit_time_adjusted - baseline_window/1000`, i.e. measured from stimulus onset (`hit_time_adjusted` = first piezo threshold crossing).
- `behavior_control/update_parameters.m`: `response_window_start = artifact_window + baseline_window`.
- `NWB_converter/utils/behavior_converter_misc.py:527`: `lick_time = response_window_start_time + reaction_time` -> late by one artifact window. `NWB_analysis` (`nwb_utils/utils_behavior.py`) uses the same convention.
- Verified 2026-09-27 (AB130, AB154, MH065): lick-locked units peak ~-100 ms relative to stored `lick_time` and ~0 relative to `first_lick_time`; `start_time` is on the neural clock (whisker artefact at 0), so the correction stays on the trial clock.

## Workflow
1. Add `first_lick_time` / `reaction_time` as above before any lick-aligned window.
2. Hit trials only: `lick_flag == 1` and `lick_time` not NaN (early/aborted licks are not in `lick_time`).
3. For new cohorts or sessions, spot-check: piezo licks (`events`, `event_type == "piezo_lick_times"`) fall within ~±30 ms of `first_lick_time` on hit trials.
4. Other lick sources are on the piezo clock, which differs from the corrected trial lick by a session-specific ~±25 ms:
   - spontaneous-lick CSVs (`combined_results_ks4/<mouse>/<beh>_<day>/spontaneous_licks/<session>_spontaneous_licks.csv`; `single` / `short_cluster` / `bout` onsets);
   - `piezo_lick_times` events.
   State this whenever trial licks are compared with spontaneous licks. Do not shift one onto the other without a per-session estimate.
5. Spontaneous-lick CSV caveats:
   - its `rewarded_lick_time` rows are the uncorrected `lick_time`;
   - its `reward_time` rows mix both clocks;
   - some `single` / `short_cluster` / `bout` rows are trial licks. Drop events equal to a trial `lick_time` or inside `[start_time - 0.2 s, response_window_stop_time + 0.5 s]`.
6. `jaw_dlc_licks` / `tongue_dlc_licks` use a different time base (offsets of hundreds of seconds). Do not align to them without converting.

## Known affected code
- Rastermap feature building (`unit_spikes_analysis/rastermap_psth/rastermap_utils.precompute_event_map`):
  - config flag `correct_lick_time` (default `True` since 2026-09-27; `False` reproduces old runs);
  - every `rastermap_variants/*` run built before that except `*__lickfix` has its "Whisker/Auditory hit (lick)" conditions shifted by +1 artifact window.
- Uses the stored `lick_time` and needs updating before reuse: `scripts/ssl_timeresolved_decoding.py`, plus lick-aligned scripts in `projects/ssl-whisker-hitmiss-timeresolved-decoding`, `ssl-cross-area-communication`, `ssl-burstiness-cohort-comparison` and `ssl-task-performance` (`rg lick_time projects scripts`).

## Quality gate
- Every lick-aligned result or RT states the lick definition used (`first_lick_time` corrected vs stored `lick_time`). Any mixing of trial and piezo clocks must be stated.
