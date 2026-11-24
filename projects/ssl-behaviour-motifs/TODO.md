# TODO
- [X] 001 session selection (tables/001_candidates.csv, tables/001_selected_sessions.csv)
- [X] 002 inputs: keypoints, features, heuristic labels, epochs (cache/inputs/, tables/002_likelihood_per_bodypart.csv)
- [X] Kuma env: venv_kpms (keypoint-moseq 0.8.7, jax 0.10.2 cuda12), venv_la (torch 2.14 cu126, lightning-action -e)
- [X] 003 kpms: kappa sweep (AR), full model, 3 seeds, per view (first fit uncentred -> session-specific syllables; refit with per-session centring; kuma/kpms, kuma/kpms_uncentred)
- [X] 005 AR-HMM on features, per view (per-session z-score; side kappa 1e8 / 120 ms, top 1e6 / 320 ms; kuma/arhmm)
- [X] 004 lightning-action, 4 folds, per view (kuma/lightning_action, f1_heldout.csv)
- [X] 006 example frames from raw video (kuma/frames)
- [X] 007 per-session reports (reports/<sid>_behaviour_motifs.pdf); 008 summary (reports/summary_behaviour_motifs.pdf, tables/008_*)
- [ ] consult user: method choices, kappa, label heuristics, before scaling beyond the pilot
- [ ] consult user: session specificity of kpms / AR-HMM states (16-53 % of frames in one-session states even after
      per-session centring) -> options: more sessions, per-mouse whitening, shared-syllable constraint, or features-only
- [ ] consult user: side AR-HMM states stay short (120 ms at kappa 1e8): drop tongue / binary features or lower AR order
- [X] (user 2026-11-19) state interpretation: 011 video pass (ME per region, ME maps, clips), 012 grid movies,
      009 fingerprints / contingency / Hungarian / meta-states, atlas pages in 007, mapping pages in 008
      (tables/009_*, kuma/video, reports/movies/ 48 mp4; K = 4 meta-states by silhouette, flat 0.39-0.43 for K 4-10)
- [ ] consult user: K = 4 is coarse (still / whisking / strong whisk + nose / mouth-jaw-lick); finer K or features?
- [ ] v2 (user 2026-11-19): NWB keypoints, 200 Hz, sweeps (kappa x states, normalisation), held-out fit, LA with nose
      class, pilot checks, STATE ON movies, regenerate reports + summary
- [X] (2026-11-20) full run: Kuma fits (79 sessions) synced; pilot v3 009 / 007 / 008 regenerated; full 009 / 015 / 016 / 017; report built
- [ ] user review of the report (combined_results_ks4/ssl-behaviour-motifs/report/report.pdf); publish to the private reports repo when approved
- [ ] optional: lightning-action top out-of-mouse folds (only the final model exists); kpms seed stability in the full run (1 seed)
- [ ] optional: 100 Hz sessions (AB102 / AB104 / AB107) need models fitted at 100 Hz (or 200 Hz upsampling) to be included
- [X] (2026-11-20) report examples: 018 atlases, example sessions / trials, epoch time courses, worked correlation examples
- [ ] optional: ITI effect in R+ vs reward (partly answered by 018 time courses: auditory-hit ITI is non-still in both cohorts): split ITI after rewarded hits vs misses / auditory hits to separate reward consumption
