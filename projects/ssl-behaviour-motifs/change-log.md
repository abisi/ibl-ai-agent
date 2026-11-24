# change-log
- 2026-11-19: project created (question.md, README.md, TODO.md).
- 2026-11-19: user (via coordinator): session selection kept quick. 001 strict gates (geometry <= 40 px) left 0
  non-learners; final 8 picked by hand from the 001 candidate table (setup 1, no check_dlc flags): only one usable R-
  non-learner exists among dual-view mice (MH037; MH064 has frame/timestamp flags), so R- = 3 learners + 1
  non-learner. Rationale per session in tables/001_selected_sessions.csv.
- 2026-11-19: top static points = nose_base + particle (top spout not visible in MH037 / MH039).
- 2026-11-19: MH026 has no dlc_jaw_onset_times.pkl -> jaw onsets from dlc_helpers.jaw_onset_times (same function).
- 2026-11-19: jaw heuristic label threshold raised to max(pkl threshold, half the median opening at licks): the pkl
  threshold alone labelled 30-60 % of frames as jaw movement (slow drift).
- 2026-11-19: kpms side view without the tongue (tongue excursions carried 99.7 % of the pose variance, latent_dim 1);
  side latent_dim 4 (98 %).
- 2026-11-19: first kpms fit (uncentred keypoints; side 320-340 ms, top 400-420 ms median syllable, 21-30 syllables
  > 1 %) gave session-specific syllables: 69 % (side) / 82 % (top) of syllables had > 80 % of their usage in one session
  (median 1 session per syllable). Decision: centre every keypoint per session (minus session median, plus mean of
  medians) for kpms and lightning-action, z-score AR-HMM features per session; refit everything. First fits kept as
  kuma/kpms_uncentred, kuma/lightning_action_uncentred.
- 2026-11-19: kpms 0.8.7 reindex_syllables_in_checkpoint returns the index, not the model: extraction now reloads the
  checkpoint (stage `extract` for already fitted models).
- 2026-11-19: AR-HMM sweep split into one job per kappa (Gibbs ~35 s / iteration on 1.5 M frames).
- 2026-11-19: final pilot run: kpms centred (side 260 ms / 18-23 syllables, top 360-380 ms / 28-32), AR-HMM per-session
  z-scored (side kappa 1e8 -> 120 ms, below the 150 ms target even at the largest kappa; top kappa 1e6 -> 320 ms),
  lightning-action centred inputs (held-out F1 essentially unchanged). Reports 007 + summary 008 built.
- 2026-11-19: user feedback: states hard to interpret, map them between methods / views / sessions. Added state
  fingerprints (14 within-session z-scored features incl. video motion energy per face region), contingency + Hungarian
  matching, meta-states (Ward clustering of session-state fingerprints, K by silhouette), atlas pages (fingerprint
  heatmap, ME maps, onset-aligned keypoint displacement), grid movies. Atlas pages replace the snapshot pages (006).
- 2026-11-19: 009 run: K = 4 meta-states (M0 still 56 %, M1 whisker + snout motion 28 %, M2 whisker-pad + snout motion 12 %,
  M3 mouth motion + jaw open / licking 4 %); reports and summary rebuilt (PDFs open in a viewer were written as *_new.pdf).
- 2026-11-19 (v2, user decisions): rerun of the pilot on the same 8 sessions. (1) keypoints, per-keypoint likelihoods and
  timestamps from the NWB_ks4 DLC series (dlc_helpers.load_dlc), no h5; check vs the v1 h5: lag 0 frames and r >= 0.98
  in all 16 session x views (MH039 h5 one frame longer). (2) native 200 Hz, no downsampling; whisking phase (Hilbert)
  added; kpms / AR-HMM nlags 10, fitted on 60 s chunks every 240 s (25 % of frames) and applied to all frames; sweeps
  over kappa x number of states; normalisation compared (centring / whitening / per-mouse scale). (3) lightning-action
  classes + nose (snout movement). (4) pilot checks (dropout, normalisation, held-out stability, drift, power) as
  summary pages (bm_checks.py). (5) movies: STATE ON bar. v1 (50 Hz) results moved to combined_results_ks4/
  ssl-behaviour-motifs/v1_50hz/ (Kuma results_v1_50hz/, data_v1_50hz/).
- 2026-11-19 (v3, user: states must mean the same in every animal): geometry = px -> mm (setup calibration) + translation
  to a common spout-tip position (user correction: spout tip as reference; top spout from the h5, not tracked in the
  top view of MH037 / MH039 -> nose_base + mean (spout - nose_base) vector); particle and jaw_ref are kinematic points
  (modelled, used for the whisker / jaw angles), never references. Labels and AR-HMM features with POOLED thresholds /
  scaling in physical units (tables/label_thresholds.json); kpms on deviations from each session's resting pose
  (pooled-threshold rest frames). Validation: syllables used by >= 6 / 8 mice, session-identity decoding (pose and
  matched-behaviour usage), leave-mouse-out recurrence, physical fingerprints across mice, states used by < 3 mice
  flagged and excluded from cross-animal comparisons; lightning-action leave-one-mouse-out.
- 2026-11-19 (v3 fixes): kpms side without tongue again (with it: ~100 % of the variance, degenerate whitened PCs, NaN
  AR parameters); kpms latent_dim = all pose dimensions (whitened); float64 (jax_enable_x64) for kpms and the AR-HMM:
  at 200 Hz the 10 AR lags are nearly collinear and float32 gave NaN Ab / Q (kpms) or one state (AR-HMM); stale
  checkpoints removed before refits. Nose class: leave-one-mouse-out F1 0.24 (side) / 0.42 (top) -> not separable,
  dropped (merged into background) in the full run.
- 2026-11-19 (v3 model choice, pilot sweeps): kpms side rest, 25 states, kappa AR 1e9 / full 1e8 (AR stage 300 ms,
  3 % of frames in session-specific syllables); kpms top rest, 25 states, kappa 1e11 / 1e10 (AR stage 135 ms, 18 %;
  without whisker_tip 225 ms / 21 %: the tip was kept, as requested); AR-HMM 15 states, kappa 1e10 both views (side
  75 ms, top 50 ms: AR-HMM states stay short at every kappa up to 1e10; 0 % session-specific frames with pooled
  scaling). Geometry check (AR stage, kappa 1e8, 50 states): session-specific frames side raw 10 % / centre 8 % /
  rest 9 %, top 30 / 32 / 31 %: on top the session specificity is not geometric.
- 2026-11-19: user: be autonomous; after the pilot run the full dataset (013 session list, 014 inputs with thresholds
  recomputed on the full data, kpms / AR-HMM fitted on 16 balanced mice and applied to all, lightning-action 5 folds
  grouped by mouse + final all-data model), learning analyses (015 per-trial measures, 016 statistics), article report.
  Full run without the video pass (fingerprints from DLC features only).
- 2026-11-20 (full run, compute): 79 sessions included (R+ 45, R- 34; tables/full_sessions.csv, exclusions in
  full_exclusions.csv); training mice for kpms / AR-HMM: 16 (R+ 4 learners + 4 non-learners, R- 7 + 1, random seed 0;
  full_train_sessions.txt). kpms full fits are slow at 200 Hz in float64 (side ~47 s / iteration, top ~100-150 s):
  top full model 80 iterations (side 150) and kpms.apply_model 20 iterations per session in the full run (50 in the
  pilot). lightning-action: side 5 folds + final model, top final model only (job count; analyses use the side view).
- 2026-11-20: kpms.apply_model needs the config (noise prior / error estimator) -> passed; pilot fits resumed from their
  checkpoints (BM_RESUME=1) without refitting.
- 2026-11-20 (resume after PC reboot): all Kuma jobs had completed (pilot v3 kpms resumed fits, AR-HMM, video pass; full
  run kpms / AR-HMM 79/79 sessions per view, lightning-action side 79 out-of-mouse predictions, top final model only);
  synced to NAS (kuma/, kuma/full/). tables/full_selected_models.json written (ho0 fits = 16 training mice, seed 0).
- 2026-11-20: 3 sessions with 100 Hz video (AB102, AB104, AB107; R+, not training mice) left out of the full-run
  analyses (009 full, 015 onwards): kpms / AR-HMM dynamics (10 lags) and lightning-action velocities were fitted at
  200 Hz and 015 transition rates assume 200 Hz -> 76 sessions analysed.
- 2026-11-20: 009 full run without lightning-action top (no per-session predictions); provenance 009_full.json.
- 2026-11-20 (haas runs, brain_wide_analysis env): pilot 009 (K = 5 meta-states) -> 007 (8 session PDFs) -> 008 summary
  (incl. bm_checks, 010 tables); full 009 (K = 4: still 62 %, nose moving + whisk 30 %, jaw open 6 %, strong whisk +
  set-point high 1 %) -> 015 (67282 session x trial x epoch rows, 76 sessions) -> 016 -> 017 (fig1 TypeError on the
  sweep json `norm` key fixed; fig2 margins fixed, lightning-action top row dropped).
- 2026-11-20: report built (report/build_report.py -> combined_results_ks4/ssl-behaviour-motifs/report/report.pdf,
  .md, .html; 3853 words, 6 figures, 6 tables). Learning results paragraphs + abstract sentence added (all numbers
  from 016 tables). Not published to the reports repo (no push in this session).
- 2026-11-20 findings: beyond licking, few learning-related action changes. R+ ITI (no-lick frames) less still / more
  whisking as whisker P(lick) rises (excess r -0.24 / +0.34, p < 1e-4), absent in R- -> likely reward consumption.
  Pre-stim: no cohort x trial interaction (both tests), only R+ stillness weakly follows the curve (+0.18, p = 0.003,
  uncorrected). Pre-stim stillness predicts whisker hits in both cohorts (AUC ~0.58-0.62, whisking ~0.39-0.44), early
  and late halves alike (no learning change). R+ slightly more still pre-stim than R- (0.61 vs 0.54, MW / Welch p ~0.03).
- 2026-11-20 (user: report needs many more examples to interpret states, example sessions, example correlations, the
  pre / resp / ITI windows): 018_report_examples.py (compute on haas, cache/full/018/<sid>.pkl, 76 sessions; figures
  018_fig_*): state atlases (main = lightning-action classes + kpms side meta-states; supplementary = shared kpms side / top
  syllables and AR-HMM side / top states), example sessions (rule-based, one per cohort x learner group:
  tables/018_example_sessions.csv), example trials, peri-stimulus time courses by outcome / cohort / session third
  (tables/018_timecourse_epoch_means.csv), worked learning-correlation examples. Report rebuilt: 10 main + 6
  supplementary figures. New finding: ITI after auditory hits (rewarded in both cohorts) is as non-still in R- as R+
  (P(still) R+ 0.23 / R- 0.11; after whisker hits 0.20 / 0.48) -> the R+ ITI learning correlation follows reward.
  Caveat seen in example trials: lightning-action jaw class can span seconds of a slightly open, immobile jaw; whisking
  class covers < 5 deg movements.
- 2026-11-20 (user: ITI cohort differences?): 019_iti_cohort.py (haas; tables/019_iti_cohort.csv, 019_iti_per_session.csv,
  figures/019_iti_cohort.*): R+ vs R- per outcome, MW + Welch, uncorrected. WH: R+ lick / non-still for 5 s (P(still)
  early 0.05 vs 0.42, no-lick 0.33 vs 0.53). AH (rewarded in both): R- licks MORE early (0.51 vs 0.37) and is less still
  late (0.17 vs 0.35, no-lick 0.16 vs 0.35), more whisking / protraction. Unrewarded outcomes (WM, CR, FA): no difference
  in still / whisking away from licks; R+ slightly more licking (0.04 vs 0.02) and jaw opening (+0.07-0.10 mm), partly a
  pre-stim offset (jaw +0.04-0.05 mm). WH-AH within session ~0 in R+, strongly negative in R-.
- 2026-11-22: segmentation comparison done (kuma/032-034, 035 on haas; tables 035_seg_*, 035_repr_*, figures/035_segment_compare).
  Keypoints alone best (lick AUC 0.99, whisk 0.96, still 0.98); adding Facemap / BEAST hurts; daart pred loss ~+0.005.
  Cross-mouse linear read-outs: Facemap / BEAST R2 < 0 for whisking / nose speed (per-session SVD, appearance), AR-HMM
  side best (0.38 / 0.37). 035 fix: AR-HMM one-hot fixed width (65). Then (user) no new compute: learning / task synthesis
  from existing 016-026 tables (kpms_side / meta_side rows ignored): learning = licking; pre-stim stillness predicts hits
  without learning change; engagement = stillness; evoked whisker movement habituates, larger after a whisker hit in R+
  only (026 C, cohort p 0.02, exploratory); no state reinforcement.
- 2026-11-22: 036_evoked_reward_figures.py (haas, 024 cache only): figure suite 036_1-4 + tables 036_* for the two 026
  reward effects. Both are explained by hits carrying larger evoked (pre-jaw-onset) whisker movement: after-hit effect
  gone when the next trial is a miss or with next-hit + position in an OLS (cohort p 0.3-0.5; only R+ whisker speed
  residual, p 0.01-0.05 within cohort, cohort p 0.05-0.1); LT effect (R- drop) tracks the R- hit-rate drop at the LT
  and is gone on misses only (cohort p 0.2-0.9). Conclusion: evoked whisker movement is a pre-lick correlate of the
  upcoming lick, not reward-driven reinforcement.
