# TODO -- ssl-pseudopopulation-area-decoding

- [X] 000 Inventory of learning-stage sessions: trials per class per target, QC units per area_group / whole brain,
      eligible mice per area and cohort -> `artifacts/000_learning_session_inventory.csv` (88 sessions, 50 R+ / 38 R-;
      MH038 has no usable trials).
- [X] Method review with the user (2026-09-29): parameters decided (question.md table); open issues A-E to settle.
- [X] Open issues A-E settled with the user (2026-09-29; question.md "Decisions on the open issues").
- [ ] 001 Feature cache (dropped: haas has 8.5 GB free; rates are computed per sampled unit inside 002 and cached
      per job instead): per session x area (area_group, whole brain), causal 50-ms / 5-ms bins (stim-aligned
      -200..+600 ms, corrected-first-lick-aligned -600..+200 ms), artefact excised, trial labels; appendable artifacts.
- [ ] 002 `exploratory-analyses/002_pseudopop_decoding.py` (2026-09-29). Timing test (hitmiss R+ Somatosensory-whisker,
      33 eligible sessions): 17 s per real or shift repetition; library spike loader too slow (10-90 s/session) ->
      project-local fast loader (identical spikes, 2-7 s/session); repetitions split into chunks of 20 run in parallel.
      R+ pilot RUNNING (haas, log /tmp/pseudo_pilot.log): hitmiss, whole brain + Somatosensory-whisker + Motor areas,
      100 real + 100 shift -> artifacts/002_pseudo_hitmiss_R+.parquet. Then all areas / targets for R+, then R-.
- [X] Pilot 1 (unpaired null; hitmiss R+, whole brain + Somatosensory-whisker + Motor areas, 100 + 100):
      `artifacts/002_pseudo_hitmiss_R+_pilot1_unpaired.parquet`, `003_summary_hitmiss_R+_pilot1_unpaired.csv`,
      `figures/003_pseudo_hitmiss_R+_pilot1_unpaired.*`. Onsets 130-320 ms (comparison not like-for-like) -> paired design.
- [X] Pilot 2 (paired design, 5 neurons/session, T_train 20, T_test 100, 100 iterations x 10 shifts):
      `artifacts/002_pseudo_hitmiss_R+.parquet`, `003_summary_hitmiss_R+.csv`, `figures/003_pseudo_hitmiss_R+.*`. Null
      0.52-0.53; real spread across iterations 0.45-0.92 -> onsets 110-265 ms. User: 20 neurons per session.
- [X] 004 sensitivity pilot STOPPED by the user (2026-09-29: "go directly to 20 neurons per session and 100 trials");
      `004_sensitivity_pilot.py` kept but not run to completion. Pilot 2 files renamed *_pilot2_n5_t20.*.
- [X] Pilot 3 (v1 sklearn, n20/T100) STOPPED by the user before completion (no output) for the speed-up rewrite.
- [X] 002 v2 (2026-09-29): C chosen per bin/fold on the real decode and reused for the 10 shifted decodes; one job
      per cohort x area running all three targets with shared loading and stim-aligned rate cache; batched Newton solver
      (--selftest vs sklearn lbfgs: rel. coef diff <= 5e-6, identical predictions) turned out SLOWER than per-bin
      liblinear at this size (56 s vs 33 s per real decode, same accuracy, curve corr 1.00) -> default ENGINE liblinear,
      batched kept as an option. Timing (R+ Somatosensory-whisker, 33 sessions): load 180 s for all targets; one
      iteration (1 real + 10 shifts) ~160 s with the batched engine; liblinear expected ~60 s.
      First real-data values: hitmiss real post-stim 0.68 vs shift null 0.60 (drift decodable with 20 units/session);
      modality_stim 0.72 vs 0.49; modality_lick 0.59 vs 0.50.
- [X] Pilot 3 (v2, liblinear; 10 mice x 20 neurons, T 100/100, 100 x 10 shifts): R+, 3 areas, 3 targets, 31 min on 30
      workers (~56 s per iteration x target). Onsets: hitmiss 50/50/130 ms, modality_stim 10/20/5 ms, modality_lick
      -220/-155/-320 ms (whole brain / Motor / S1-whisker). Files renamed *_pilot3_10mice before the full run.
- [X] Balanced-reuse trial sampling + 20 mice per draw implemented; check: pool of 3 -> 33-34 uses each; one hitmiss
      iteration with 20 mice x 20 neurons = 66 s (load 173 s per task).
- [X] FULL RUN DONE 2026-09-29 21:19 (R+ 16:51-19:01, R- 19:01-21:18), 20 mice x 10 neurons per session (a 20-neuron start at 16:45 was aborted, partial files *_aborted_n20); 003 summaries/figures and 005 overlays for both cohorts (exploratory-analyses/run_full.sh, log artifacts/002_full_log.txt; pilot 3 files renamed *_pilot3_10mice): R+ and R-, whole brain + all area groups, 3 targets.
- [ ] 003 `exploratory-analyses/003_summarize_pseudopop.py` written (summary csv + figure per target x cohort). Statistics: above chance vs the shift-null 95th percentile (0.5 for stim-aligned modality), onset latency
      (next-5-bins rule), R+ vs R- by mouse-level cohort-label permutation, area comparisons at equal N.
- [ ] 003b Neuron-dropping curves (areas with > 500 neurons per cohort; 500 -> 50 in steps of 50).
- [ ] 004 Figures (learning stage only), then report.html.
- [X] R+ vs R- comparison from the bootstraps (006)
- [ ] Window permutation with shifts (009) RUNNING since 2026-09-29 ~23:00; then rerun 010 (abstract figure) and add p to 006
- [ ] 1000 iterations (002, run_1000.sh) RUNNING; then rerun 003/005/006/008/010 with n_iter 1000
- [ ] Decide a display cutoff for areas with few eligible mice (SS-body R+ 2 mice, R- 3; several areas at 4-5).
- [ ] Increase iterations / shifts beyond the pilot 100 x 10 before final figures (002 resumes and only adds iterations).
