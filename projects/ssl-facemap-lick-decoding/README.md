# ssl-facemap-lick-decoding

Does the mouse's face before stimulus onset (side video, Facemap motion SVD) predict the upcoming lick on whisker
trials, beyond trial position and lick / reward history, on day 0 and in expert sessions, and does it do better than
DLC keypoints? Do pre-stimulus face measures follow the within-session learning curve?

Results, figures and the report live on the NAS: `combined_results_ks4/ssl-facemap-lick-decoding/`
(`/mnt/lsens-analysis/Axel_Bisi/...` on haas, `M:\analysis\Axel_Bisi\...` on Windows). No results are stored here.

## Run order

| step | script | where | inputs | outputs (results folder) |
|---|---|---|---|---|
| 1 | `001_extract_clips.py` | haas | side videos, NWB_ks4 trials, ssl-pupil-qc video matcher, `sessions.txt` | `clips/<session>_clips.{mp4,npz}`, `clips/summary_*.csv` |
| 2 | `002_kuma_submit.sh` -> `002_kuma_job.sh` -> `002_kuma_facemap.py` | Kuma (GPU) | `clips/*_clips.mp4` (rclone) | `facemap/<session>_facemap_small.npz` |
| 3 | `003_analysis.py` | haas | clips index, Facemap SVDs, NWB_ks4 trials + DLC, exact learning curves | `analysis/decoding_per_session.csv`, `decoding_summary.csv`, `learning_curve_correlations.csv`, `per_trial_measures.csv`, summary figure |
| 4 | `004_roi_auto.py` | haas | clips videos, DLC h5 (side view) | `roi/rois.csv`, `roi/previews/`, `roi/videos/<session>_<roi>.mp4` |
| 5 | `005_kuma_roi_submit.sh` -> `005_kuma_roi_job.sh` -> `005_kuma_roi_facemap.py` | Kuma (GPU) | `roi/videos/*.mp4` | `roi/facemap/<session>_<roi>_facemap_small.npz` |
| 6 | `006_compare_roi_dlc.py` | haas | outputs of 1, 2, 4, 5; NWB DLC; learning curves | `roi/compare/per_session*.csv`, `summary_vs_null.csv`, `roi_vs_dlc_paired.csv`, `roi_vs_dlc.png` |
| 7 | `007_feature_by_feature.py` | haas | as 6 (+ 006's per-session table, beh_correlates outputs) | `roi/compare/features_*.csv`, `*_real_vs_null.png` |
| 8 | `008_report_figures.py` | haas | tables of 3, 4, 6, 7 (read only) | `figures/fig1-6` (png + pdf) |
| 9 | `report/build_report.py`, then `report/render.sh` | haas, then PC | tables + figures | `report/` (report.pdf / .md / .html) |

Shared module: `dlc_kpca.py` (NWB trials + DLC loading, exact-curve trial table with history predictors, DLC keypoint
PCA and four-body-part features, blocked-CV AUC). Moved from the retired `ssl-face-lick-decoding` pilot; also imported
by `projects/ssl-iti-signal-screen/001_features.py`.

Run commands are in each script's docstring (haas: `nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python <script>`;
Kuma scripts expect the code under `/work/lsens/bisi/facemap_full/code/` and `/work/lsens/bisi/facemap_roi_full/code/`).

## Design in brief

- Trials: active whisker trials with `perf != 6` and `early_lick == 0` (the exact-learning-curve trials); target
  `lick_flag`. Day 0 and expert (day > 0) analysed separately, per cohort (R+ / R- from the reference sheet).
- Video: only peri-trial clips (-2.5..+0.5 s around each whisker trial) are read from the raw side videos; Facemap
  motion / movie SVD per session (full frame), and per automatic face ROI (eye, snout, whisker pad, jaw).
- Decoding: logistic regression, blocked 5-fold CV; feature sets time, history, base = time + history, Facemap, DLC
  keypoint PCA, and base + each face set; metrics AUC, balanced accuracy, held-out r; linear-shift null.
- Learning curve: per-session correlation of face measures with the exact learning curve, linear-shift null.
- ROI vs DLC: feature sets and single features scored against their own shift null.

## History

- `projects/ssl-facemap-pilot/`: 6-session pilot (full frame and hand-drawn ROIs) that set the Facemap parameters;
  superseded by this project.
- `projects/ssl-face-lick-decoding/`: DLC keypoint-PCA pilot (day 0); its helpers now live in `dlc_kpca.py`.
