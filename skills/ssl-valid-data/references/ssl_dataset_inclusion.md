# SSL dataset: sources, inclusion rules and known metadata errors

Canonical record of dataset-level decisions for all SSL analyses (created 2026-10-03; update in place, never silently).

## Sources
- NWB files (KS4 sorting): haas `/mnt/lsens-analysis/Axel_Bisi/NWB_ks4/` (= `M:\analysis\Axel_Bisi\NWB_ks4`). Analyses read
  spikes, trials and licks from these raw NWB files (not from the compressed `ssl_ephys` / `ssl_behavior` dataset).
- **Load each session once, from its session NWB only** (user 2026-11-05).
  - Read only files named `<mouse>_<YYYYMMDD>_<HHMMSS>.nwb`, i.e. regex `^[A-Za-z]+\d+_\d{8}_\d{6}\.nwb$`.
  - Never read `*_backup*.nwb`. NWB_ks4 holds such copies, e.g. 20 `*_backup_20261104[_HHMM].nwb` left by the
    2026-11-04 DLC reconversions.
  - A bare `*.nwb` or `<mouse>_*.nwb` glob picks them up and silently duplicates sessions. On 2026-11-05 this
    happened to the Figure 1 behaviour table and the GLM run.
  - More generally, no session may be loaded twice. After building any table, assert that each NWB file is read
    once, that `session_id` is unique per session-level row, and that (`session_id`, `start_time`) is unique per
    trial-level row. Fail loudly; never deduplicate silently.
  - `trial_id` is NOT unique within a session for the day +2 sessions of AB077-AB107. Those days were recorded as
    two NWBs (`whisker_on_1`/`_2`), which `reindex_whisker_days` merges under one `session_id`, and `trial_id`
    restarts in the second half (checked 2026-11-05: 14 sessions, 866 rows). They are not duplicates.
  - The compressed `ssl_*` builders already apply this regex (`SESSION_FILENAME_RE`).
- Mouse reference sheet: `M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx`
  (haas `/mnt/share_internal/Axel_Bisi_Share/dataset_info/...`). Columns used: `reward_group`, `exclude`, `exclude_ephys`,
  `recording`, `learning_category`.
- Raw preprocessing (DREDge motion for the drift test): `Axel_Bisi/data/<mouse>/<session>/Ephys/catgt_*/..._imec*/dredge(_fast)/motion/motion`.
- Per-session results: `combined_results_ks4/<mouse>/<behaviour>_<day>/...`.

## Inclusion rules
1. Mouse included if `exclude == 0`, `exclude_ephys == 0`, `reward_group` in {R+, R-} (R+proba excluded), not in
   `EXCLUDED_MICE = [AB068, AB077]`.
2. `recording` applies to **day 0 only**: day-0 (learning) sessions of mice with `recording != 1` are dropped; expert
   (day >= 1) sessions are kept whatever `recording` says (user 2026-10-03).
3. Cohort (R+ / R-) is a **mouse** property taken from the sheet's `reward_group`, never from the session-level NWB
   `wh_reward` / unit-table `reward_group`; every disagreement is reported.
4. Unit quality: `quality_label` in {good, mua} from `classify_units_quality`, with the joint drift check (requires the
   drift-test result of the session; sessions without it are listed in the unit-table provenance).
5. Stage: learning = day 0, expert = day >= 1. Learners population: drop only the day-0 session of mice whose
   `learning_category` is not good / moderate; keep their expert sessions.

## Known metadata errors and decisions (keep this list current)
Per-session NWB conversion exceptions (camera runs, DLC timestamps, json overrides, skipped probes, epochs) are listed
in `nwb_session_exceptions.md`.

| item | issue | decision |
|---|---|---|
| MH035 day 0 (MH035_20250514_142713) | NWB `wh_reward = 1`, session label R+ | mouse is R- (sheet; user 2026-10-03) |
| MH064 day +1 (MH064_20260115_104148) | NWB `wh_reward = 1`, session label R+; days +1/+2 `no_stim_weight = 0` but 121 / 114 no-stim trials exist (7 / 8 with licks) | mouse is R- (sheet; user) |
| `wh_reward` vs `reward_available` (scan 2026-11-03, all 376 whisker sessions) | trials follow the sheet cohort everywhere; metadata `wh_reward` wrong in MH035 0 (1, no whisker trial reward_available), MH064 +1 (1, idem), MH038_20250525_130335 +1 (0, all 107 whisker trials reward_available; mouse R+); AB124 +1, MH008 +1, MH018 +1 (R-) have one reward_available whisker trial at session start (trial 0-2) | `Training/<session>/session_config_corrected.json` written 2026-11-03 (user) with `wh_reward` 0 / 0 / 1 (MH038: added to the existing corrected file, which also sets `camera_flag` 1); NWBs need reconversion (converter reads the corrected json from `Training/`) |
| AB105 day 0 | `recording = 0` ("recorded but noise chunks"); not converted for analysis | excluded |
| MH001 +3 / +4, MH067 +2 / +3 | `recording = 0` in the sheet (day-0 flag) | expert sessions included (rule 2) |
| MH019 | `exclude_ephys = 1` | excluded |
| MH065 day +1 (MH065_20260115_163926) | 3 early-half auditory hits | pre-lick 051 keeps the last warm-up auditory hit (`WARMUP_KEEP_AH`), documented exception |
| Drift-test files (before 2026-10-03) | `run_motion_shift_test_analysis` used one `session_day` for a whole multi-day table: expert sessions of a mouse were written to one folder (mostly `whisker_1`) and overwritten; most expert sessions had no or mislabelled drift results | function patched to a per-session label (backup `.bak_20261003`); mislabelled files renamed `*.mislabeled_20261003`; all expert sessions recomputed (`ssl-prelick-convergence/data_prep/run_missing_drift_tests.py`) |
| Context labels (per-trial rule, user 2026-10-04) | MH062_20260113_125836 (R-, day +4): 377 training trials unlabelled ("nan"), so the old relabelling put them into passive_pre / passive_post (no whisker_active ROC); MH064_20260114 (day 0): 21 trials labelled active at the end run at a fixed 3.07 s ITI right before the passive block; AB128_20240829: 2 trials | passive = fixed ~3 s ITI, else the context column (unlabelled -> active; perf 6 inside active blocks stays excluded-active). `roc_utils_new.process_nwb_tables` patched (backup `.bak_20261004`); main ROC of the 3 sessions recomputed (old files `*.context_bug_20261004`); ROC table 045 rebuilt. NOT yet applied in 051 `select_trials` / `prep_session` (ssl-prelick-convergence, ssl-within-day-remapping, older decoding projects) |
| Active perf == 6 trials (ssl_ephys 1.0.0) | 313 trials: 155 pretraining, 128 in six whisker day-0 sessions (AB107 52 mid-session, AB142 43, AB149 25 at the end; AB128 / AB162 / AB164 2-3 each), 30 in seven expert sessions; ~10 s ITI, not passive | invalid trials, always excluded (user 2026-10-04: not an NWB-version artefact); `prep_session` has applied this since 2026-10-01, and it also applies the per-trial context rule since 2026-10-04 (decoding sees MH062_20260113 with 398 instead of 21 active trials) |
| Unit sets (user definition 2026-10-05) | stability previously mixed with other criteria (spike count, amplitude cut-off) and quality_label tables without the drift merge | three sets: all (good + mua), stable (coverage >= 0.9, presence >= 0.5, drift joint test passed; no drift result = not stable), good = quality_label good AND stable; skill ssl-valid-data 'Unit sets', table 137_stable_units.parquet (174140 / 142242 / 43679 units) |
| AB155 day 0 joined session (AB155_20250208_175103; runs g0 + g1 sorted together, run 4819418) | passive post recorded ~904 s after the active block; on imec4, 13 of 33 good units drop to < 20 % of their passive-pre rate at the junction and neighbours gain spikes (tissue / waveform change during the gap; DREDge shows no displacement, KS4 adds a spurious +23.5 um step); presence / coverage / drift-shift test (central segment excludes both passive blocks) and KS amplitude / position do not catch it | user 2026-11-05: imec4 excluded from passive pre vs post analyses via `ephys_utilities.helpers.data_utils.PASSIVE_EXCLUDED_PROBES` (applied by `keep_passive_mice` = `keep_passive_data`, the function to always use for passive selection, also by Myriam Hamon); AB155 stays in the mouse-level passive exclusion until the joined NWB replaces the pre-only one. Joined NWB: `recording_gap` BehavioralEpochs, presence / coverage computed with `exclude_intervals`. Project ssl-ab155-session-merge (014-017). User 2026-11-06: the final NWB keeps the recorded gap (part 2 shifted by 903.668 s, as `012_insert_gap_nwb.py`), not the joined clock; `exclude_intervals` stays in `unit_metrics_utils`; only imec4 is excluded (not the 6 imec1 / 2 imec2 collapsing units); merged data copied (not moved) into `data/AB155/AB155_20250208_175103` by `009_move_to_data.sh`; joined NWB in NWB_ks4 since 2026-11-06 (019), AB155 removed from the mouse-level passive exclusion (only imec4 excluded) |
| all-days unit cache v1 (`tables_all_days.pkl`) | duplicate unit rows; session-level cohort; `recording` applied to all days; expert sessions without drift check | superseded by v2 |

## Current tables (v2)
- Builder: `projects/ssl-prelick-convergence/data_prep/build_unit_table_all_days.py`.
- Outputs: `combined_results_ks4/rastermap_variants/_cache/tables_all_days_v2.pkl` (+ `_provenance.json`: subjects,
  sessions per day, day-0 sessions dropped by `recording`, sessions without drift results),
  `combined_results_ks4/_roc_stage_analysis/unit_info_all_days_v2.parquet` (metadata read by 045),
  `_roc_stage_analysis/sessions_cohort_check.csv` (per-session cohort label vs sheet).
- Pre-lick tables (053) re-apply the sheet cohort and write `ssl-prelick-convergence/across_days/{fa,sl}/cohort_check.csv`.
- Any new analysis must read units through these tables (or apply the same rules) and report session inclusion
  (sessions and mice per cohort x stage) in its provenance.
