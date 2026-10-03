# SSL dataset: sources, inclusion rules and known metadata errors

Canonical record of dataset-level decisions for all SSL analyses (created 2026-10-03; update in place, never silently).

## Sources
- NWB files (KS4 sorting): haas `/mnt/lsens-analysis/Axel_Bisi/NWB_ks4/` (= `M:\analysis\Axel_Bisi\NWB_ks4`). Analyses read
  spikes, trials and licks from these raw NWB files (not from the compressed `ssl_ephys` / `ssl_behavior` dataset).
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
| item | issue | decision |
|---|---|---|
| MH035 day 0 (MH035_20250514_142713) | NWB `wh_reward = 1`, session label R+ | mouse is R- (sheet; user 2026-10-03) |
| MH064 day +1 (MH064_20260115_104148) | NWB `wh_reward = 1`, session label R+; days +1/+2 `no_stim_weight = 0` but 121 / 114 no-stim trials exist (7 / 8 with licks) | mouse is R- (sheet; user) |
| AB105 day 0 | `recording = 0` ("recorded but noise chunks"); not converted for analysis | excluded |
| MH001 +3 / +4, MH067 +2 / +3 | `recording = 0` in the sheet (day-0 flag) | expert sessions included (rule 2) |
| MH019 | `exclude_ephys = 1` | excluded |
| MH065 day +1 (MH065_20260115_163926) | 3 early-half auditory hits | pre-lick 051 keeps the last warm-up auditory hit (`WARMUP_KEEP_AH`), documented exception |
| Drift-test files (before 2026-10-03) | `run_motion_shift_test_analysis` used one `session_day` for a whole multi-day table: expert sessions of a mouse were written to one folder (mostly `whisker_1`) and overwritten; most expert sessions had no or mislabelled drift results | function patched to a per-session label (backup `.bak_20261003`); mislabelled files renamed `*.mislabeled_20261003`; all expert sessions recomputed (`ssl-prelick-convergence/data_prep/run_missing_drift_tests.py`) |
| all-days unit cache v1 (`tables_all_days.pkl`) | duplicate unit rows; session-level cohort; `recording` applied to all days; expert sessions without drift check | superseded by v2 |

## Current tables (v2)
- Builder: `projects/ssl-prelick-convergence/data_prep/build_unit_table_all_days.py`.
- Outputs: `combined_results_ks4/rastermap_variants/_cache/tables_all_days_v2.pkl` (+ `_provenance.json`: subjects,
  sessions per day, day-0 sessions dropped by `recording`, sessions without drift results),
  `combined_results_ks4/_roc_stage_analysis/unit_info_all_days_v2.parquet` (metadata read by 045),
  `_roc_stage_analysis/sessions_cohort_check.csv` (per-session cohort label vs sheet).
- Pre-lick tables (053) re-apply the sheet cohort and write `_roc_prelick{,_sl}/cohort_check.csv`.
- Any new analysis must read units through these tables (or apply the same rules) and report session inclusion
  (sessions and mice per cohort x stage) in its provenance.
