# ssl-behaviour-motifs

Behavioural motifs from side / top DLC keypoints of day-0 whisker sessions (pilot, 8 sessions).
Results: `combined_results_ks4/ssl-behaviour-motifs/` (tables/, cache/, figures/, reports/, provenance/).

| Script | Where | What |
|---|---|---|
| bm_lib.py | haas | paths, constants, trial / lick / jaw-onset loaders |
| 001_select_sessions.py | haas | candidate table + 8 selected sessions (tables/001_*) |
| 002_prepare_inputs.py | haas | aligned 50 Hz keypoints, features, heuristic labels, epochs (cache/inputs/) |
| kuma/003_kpms.py | Kuma GPU | keypoint-MoSeq: PCA, AR-HMM kappa sweep, full model x seeds per view |
| kuma/004_lightning_action.py | Kuma GPU | lightning-action dtcn on heuristic labels, 4 folds (train 6 / test 2) |
| kuma/005_arhmm_features.py | Kuma GPU | sticky AR-HMM (jax_moseq) on interpretable features per view |
| kuma/006_example_frames.py | Kuma CPU | raw-video frames of the top states |
| kuma/011_video_pass.py (+ run_video_pass.sbatch) | Kuma CPU (l40s / mig24gb slot) | region motion energy 50 Hz, per-state motion-energy maps, mean frame, clips |
| kuma/012_grid_movies.py | Kuma CPU | grid movies of the global top states (2 x 4 sessions) |
| 009_fingerprints_metastates.py | haas | state fingerprints, contingency + Hungarian, meta-states (Ward on session-state fingerprints) |
| bm_atlas.py | haas | state-atlas pages (007) and meta-state / mapping pages (008) |
| 007_session_reports.py | haas | one PDF per session (both views, three methods) |
| 008_summary.py | haas | cross-session summary (pilot), incl. bm_checks pages (dropout, geometry, held-out, drift, power) |
| 013_full_sessions.py | haas | full-run session list + exclusions (tables/full_sessions.csv, full_exclusions.csv) |
| 014_full_inputs.py | haas | full-run inputs, same definitions as 002, thresholds recomputed on all sessions |
| 015_trial_measures.py | haas | per-trial action / meta-state usage, transitions, latencies (full run) |
| 016_learning_stats.py | haas | learning analyses A-E (full run) |
| 017_report_figures.py | haas | report figures |
| 018_report_examples.py | haas | `compute` (per-session atlas / peri-stim cache, cache/full/018) + `figures` (atlases, example sessions / trials, epoch time courses, worked learning examples) |
| report/build_report.py | local / haas | article report (report_lib) -> combined_results_ks4/ssl-behaviour-motifs/report/ |
| kuma/run_full.sh | Kuma | full-run job chain (BM_SET=full) |

Kuma: workspace `/work/lsens/bisi/behaviour_motifs` (venv_kpms, lightning-action clone), partition l40s only (never h100);
data moved NAS <-> Kuma with rclone (`nas-sv:Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4/ssl-behaviour-motifs`).

## Pilot settings (2026-11-19)
- 50 Hz (block mean of 4 frames of the 200 Hz DLC); keypoint times = each camera's NWB timestamps (dlc_helpers).
- Alignment: similarity transform of median static points (side spout / jaw_ref / nose_base; top nose_base / particle).
- kpms 0.8.7: fix_heading=True; side without tongue (see change-log), latent_dim side 4 / top 3 (>= 90 % variance);
  AR-stage sweep kappa 1e3-1e7 (side medians 80 / 120 / 200 / 300 / 400 ms); full model: 50 AR iterations at kappa 1e5
  then 300 full iterations at kappa 1e4, seeds 0-2.
- AR-HMM: jax_moseq arhmm, 4 z-scored features per view, nlags 3, 30 states max, kappa chosen for a ~250 ms median.
- lightning-action: dtcn (dilated TCN), 64 hidden units, 2 layers, 4 lags, 100 epochs, inputs aligned xy + velocity.
- Kuma sync: `bash scripts/sync_to_nas.sh` (Kuma) -> combined_results_ks4/ssl-behaviour-motifs/kuma/.

## v3 (current, 2026-11-19 / 20)
NWB_ks4 DLC series at 200 Hz; px -> mm + spout-tip translation; pooled thresholds / scaling (tables/label_thresholds.json,
full_label_thresholds.json); kpms on deviations from each session's resting pose (side nose_tip, jaw, jaw_ref; top
nose_tip, whisker_base, whisker_tip, particle), float64; AR-HMM on physical features with pooled scaling; lightning-action
(still / whisking / jaw / lick; nose only in the pilot). Pilot = 8 sessions (BM_SET=pilot, default); full run = 79
sessions (BM_SET=full; Kuma results/full, NAS kuma/full, cache/full). v1 (50 Hz) results in v1_50hz/.

## Full run (2026-11-20)
79 eligible sessions fitted / applied on Kuma; 76 analysed (3 sessions with 100 Hz video left out). Run on haas with
`~/anaconda3/envs/brain_wide_analysis/bin/python`, `BM_SET=full`: 009 -> 015 -> 016 -> 017 -> report/build_report.py, then
`bash build.sh` in combined_results_ks4/ssl-behaviour-motifs/report/ (local TinyTeX). Model tags: tables/full_selected_models.json.
