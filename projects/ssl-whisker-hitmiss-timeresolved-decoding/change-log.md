# Change log

- 2026-09-24: question.md -- added "Learning-trial split" section (new
  pre/post learning_trial hit/miss decoding question and locked design).
  TODO.md -- added matching step list.
- 2026-09-24: question.md -- added "Learning-trial split: redo" (shift null,
  3 windows, terminal-disengagement rule). TODO.md -- redo steps.
- 2026-09-24: question.md -- added "Placebo-split test of the learning_trial". TODO.md -- placebo steps.
- 2026-09-25: TODO.md -- follow-ups (disengagement detail, halves comparison, placebo results, new-LT runs).
- 2026-09-26: Decoder choice locked -- keep the current L2 logistic (per-fold StandardScaler,
  one C per session/window via select_fixed_c_pooled). 109 compared it on 84 learning-stage
  sessions against nested-C logistic, shrinkage LDA (Ledoit-Wolf), PCA(20)+logistic and RBF-SVM:
  RBF-SVM (-0.036 sensory) and shrinkage LDA (-0.028) were worse, PCA+logistic equal; the
  current model's C-on-all-trials gives ~+0.005 optimism that cancels in within-pipeline
  comparisons. User decision: keep the current model.
- 2026-09-27: Lick-aligned modality decode switched to the CORRECTED first lick
  (skills/ssl-lick-alignment): first_lick_time = start_time + lick_time - response_window_start_time,
  reaction_time = lick_time - response_window_start_time, per trial (artifact window 100 ms in 873/894
  sessions, 50/150/300 ms in the rest). The stored lick_time is late by that window, so the old
  "pre-lick" window (-150, 0) ms actually covered -50..+100 ms around the physical lick, and the old
  full window (-600, +200) covered -500..+300 ms. Windows are unchanged in definition, now relative
  to the physical lick. Median RT 336 -> 236 ms; 15% of licked trials have RT < 150 ms (stimulus
  inside the pre-lick window; artifact dead zone still masked per trial).
  Code: scripts/ssl_timeresolved_decoding.py (add_first_lick_time, CORRECT_LICK_TIME /
  SSL_CORRECT_LICK_TIME=0 reproduces old runs, LICK_TIME_DEFINITION; prep_lick_aligned_trials and
  licked_only perf-state preps), 024 (event_time = first_lick_time, lick_time_definition column),
  RT / stim-time histograms in 028, 032, 035, 042, axis labels in 025/027-030/032/042.
  Hit/miss and stim-aligned results unaffected. All 5 modality_lick tags rerun (learning whole_brain,
  area_group, area_acronym_custom; expert whole_brain, area_group) via 024_lickfix_queue.sh, followed
  by 025-030, 032, 034, 034b, 041, 042. Old results kept as *.bak-prelickfix-20260927.
  Superseded pilot scripts 007-014 still use the stored lick_time (not rerun).
- 2026-09-27: 077 perfquant face-validity RT (here and in ssl-whisker-perfquant-pls-regression) now
  lick_time - response_window_start_time. Not rerun: its per-session Pearson/Spearman r are unchanged
  by a session-constant offset (artifact window constant within 893/894 sessions). 032 now also saves
  a vector .pdf next to each .png.
- 2026-09-28: 110_publication_figures.py -- one publication-style multi-panel figure per decoding analysis
  (hitmiss, modality_stim, modality_lick, perfstate) in figures/publication/ (vector PDF + PNG), all tests in
  110_publication_stats.csv, plotted session values in 110_publication_values.parquet, panel summary in
  figures/publication/110_captions.md.
- 2026-09-28: 034 and 042 resolved allen_utils through a hardcoded M:\ path, so on haas every area figure fell back
  to grey colours and alphabetical order (incl. the 2026-09-27 lick-fix queue outputs). Now resolved via
  scripts/axel_bisi_paths.py; all area-level 034/042/034b figures regenerated.
- 2026-09-28: 111_coupling_model_v2.py (autonomous run) -- coupling v2 on all learning sessions: fold-wise class-aware
  polynomial detrending (running-mean detrend rejected: it anti-correlates z with the local hit rate), LOO learning-curve
  lick tendency as GLM offset, >= 5 hits/misses per epoch, common-window linear shift null for the trial-position version
  (primary); the before/after-LT version uses the full session + truncating shift null (secondary; the common window cuts
  the pre-LT epoch of early learners). Result: coupling DECREASES across the session in R+ learners (sensory b3_z -0.61,
  Wilcoxon p=0.022, t p=0.018, mouse sign-flip p=0.017; sensory-minus-baseline p=0.031/0.050); no group effect elsewhere.
  Caveat: the within-class decline of the neural score over trials is NOT removed by the detrending (median |r| 0.29 ->
  0.31), and 111b shows it is not a decline of the population whisker response in R+ (R- learners: response declines).
  Unresolved -> not interpretable yet.
- 2026-09-28: 112_publication_sets.py -- Set 1 (session-wide: whole-brain + per-area figures incl. per-area time
  courses, heatmaps, latency and R+ - R- latency, pooled-cohort area ranking) and Set 2 (within-session: halves and
  performance states over time and per area, paired sign-flip cluster tests, cross-condition generalisation, expert)
  for hitmiss, modality_stim, modality_lick (+ compact Set 1 for perfstate); all tests in 112_set_stats.csv.
  110.load_tag now also keeps perfstate-condition rows and cross-generalisation curves.
- 2026-09-28: 112 sets are learning-stage only (user: "Do not show expert results in those"): expert time course,
  learning-vs-expert, expert per-area and expert window-change panels removed; Set 1 whole-brain gains a real-vs-null
  panel. 112_set_stats.csv regenerated (1748 tests, all learning stage).
- 2026-09-28: 113_publication_merged.py supersedes the 112 figure sets (112 figure folders removed; script kept):
  whole brain and area group merged on one figure per set and analysis (learning stage only). Set 1 adds example
  sessions, decoding vs behavioural d' correlation (113_behaviour_dprime.csv), area overlays without error bars,
  latency point estimates without spread, latency-ordered heatmaps, cohort-coloured area comparison with two-way
  ANOVA (cohort x area) + FDR post-hoc. Set 2 is halves only (no performance states): mean +- SEM 1st vs 2nd half,
  change over time, cross-half generalisation, per-cohort half x area ANOVA with paired post-hoc, and change per
  area R+ vs R- with cohort x area ANOVA. All tests in 113_stats.csv. 110.load_tag also keeps the null SD.
- 2026-09-28: TRIAL-PREP RULE (Axel Bisi): never remove the first whisker trial; remove the auditory warm-up block
  except the 1 trial immediately before the first whisker trial; t-1 columns computed before the cut; no other
  first-trial drop (scripts/ssl_bwm_trial_prep.py prep_session; rule in
  skills/ssl-analyze/references/ssl_auditory_warmup_block.md + ssl-analyze SKILL.md). Previous versions removed the
  first whisker trial: 2026-09-14..28 in 76/88 learning sessions (warm-up sessions), before 2026-09-14 in sessions
  without a warm-up block. ssl_bwm_single_cell drops NaN-t1 rows itself (session-initial whisker trial).
  MIN_TRIALS_PER_CLASS 5 -> 3 with adaptive folds (min(5, minority)); 024 gained a fill-in mode (SSL_FILLIN_TARGETS).
  Recompute (024c targets, 024 fill-in; *.bak-prefirstwhisker-20260928): hitmiss_stim / modality_stim /
  modality_lick x learning whole_brain, area_group, area_acronym_custom, expert whole_brain, area_group; then 114
  and all figure/stats scripts (025-030, 032, 034, 034b, 041, 042, 113). Sessions changed: hitmiss wb 13 (+8 min-3),
  hitmiss ag/acr 88, modality_stim 88 (expert 29), modality_lick 51 (expert 22).
- 2026-09-29: TODO.md -- expert-stage recompute paused by the user; hitmiss_stim_expert_area_group interrupted (5/28, mixed parquet).
- 2026-09-30: 115_cosyne_halves_figure.py -- COSYNE 6-panel split-half figure from 114 (single whole-session decoder scored per half), R+ vs R-, areas excluding olfactory, cortical subplate, pons/medulla, amygdala/hypothalamus, visual, SS-body; three-way cohort x half x area as OLS type II ANOVA and as a mixed model with a session random intercept; outputs figures/publication/115_*, 115_stats.csv, 115_values.csv.
- 2026-09-30: 115 extended (user): source 114 (single decoder, area level only 5-50 ms stored) or 024 (separate decoder per half, full area curves) and any window; rendered 114_5-50, 024_5-50, 024_5-100 (figures/publication/115_cosyne_halves_figure_<source>_<window>, 115_stats_/115_values_<source>_<window>.csv).
- 2026-09-30: 116_halves_single_decoder_v2.py written (single decoder, hitmiss 5-100 ms + lick-aligned modality); run stopped by the user after ~15/89 sessions (no output written) -- the COSYNE figure uses the 024 separate-decoder halves instead.
- 2026-09-30: 024d_stale_null_fillin_targets.py + 024d_fillin_run.sh -- rows whose stored shift null was entirely NaN (computed before the 2026-09-28 adaptive-fold change; exactly 5 minority trials so every shift left <= 4 under fixed 5-fold CV) recomputed in 024 fill-in mode: modality_lick whole_brain 2 / area_group 9 / area_acronym_custom 47 groups, hitmiss_stim whole_brain 2 (perfstate) / area_acronym_custom 32 groups; backups *.bak-prestalenull-20260930. First launch failed (cwd must be the repo root; nothing written), relaunched from the root.
- 2026-09-30: 117_halves_matched_pilot.py -- PILOT count-matched halves (hit vs miss, whole brain, 4 R+ + 4 R- sessions, 10 subsamples): change 2nd-1st (5-100 ms) unmatched vs matched r = 0.92, mean |change| 0.087 vs 0.098; subsample SD 0.03-0.05; matching does not remove the half differences (no evidence that the halves effect is a sample-size artefact). Outputs 117_halves_matched_pilot{.parquet,_summary.csv}, figures/117_halves_matched_pilot.png.
- 2026-09-30: 118_lt_definitions_split_decoding.py (all 12 LT definitions of the LT project + half split; whole brain; hit vs miss 5-50 & 5-100 ms, lick-aligned whisker vs auditory -150..0 ms; 095 scheme: disengagement drop, size-matched + unmatched, session-level shift null; overview settings N_SUBSAMPLE 50, N_SHIFT 25) launched on haas (40 workers, log 118_log.txt); 119_lt_definitions_comparison_figure.py renders the one-figure comparison (per-definition sessions and shared-sessions versions).
- 2026-09-30: 120_lt_defined_vs_undefined.py -- per LT definition: sessions with a definable LT (LT split) vs sessions without (half split), plus learners' own half split as control; from 118; runs after 118.
- 2026-09-30: 118 done (178 tasks, ~50 min, 0 errors); 119 + 120 figures rendered. AB157 (R-, 6 hits) has no valid shift under L1/L2 -> excluded from those two rows (no null).
- 2026-10-04 23:02 TODO: added 2026-09-30..10-04 split-decoding section (completed items, status report, open items).
- 2026-10-04 23:14 Report built (first article-style build, skills/project-report): within-session split decoding -- halves (117b, 016), LT splits (119/120), placebo (126, 122b), cohort permutation (131), mid/best split (129), behaviour (124/125), single-trial margins (127, 130), stimulus-onset geometry (133-135b); 17 figures, 111 numbers in numbers.json. Interim HTML 136 removed (superseded).
- 2026-10-04 23:44 Project cleanup (user, interactive, per block). Manifest: _archive/MANIFEST_20261004.tsv (local and haas).
  A session-wide decoding: pilots 000-023 kept as frozen; area_acronym_custom DROPPED (outputs kept in place, no longer updated); 024 perf==6 rerun NOT NOW (stale, 13 sessions); publication scripts 110/112/113/115 kept.
  B perf-state decoding (015-019, 036-037, 043-047, 069) -> _archive/perfstate; the 024 perfstate_stim tag stays with block A.
  C perfquant (049-072, 074-088): identical copies removed, files only here moved to ssl-whisker-perfquant-pls-regression (haas: new folder there); 073 kept (block A control).
  D first-generation LT split (089-108) + coupling model (108, 111, 111b) + 102 report -> _archive/lt_split_v1; old reports (report.html, template, assets, latex, build_report.py, report_learning_trial_split.html) -> _archive/reports.
  E halves: 116 + 117 kept, 114 kept (read by 113/115); 117 and 123b perf==6 reruns NOT NOW; superseded tagged runs (118-120 untagged/_A1, 122 first/partial_with_perf6, 126 untagged/_step1, 123 v1 pkl) -> _archive/superseded_runs.
  F 132 outputs -> _archive/superseded_runs (132 script kept: imported by 133-135); 133 all-units kept; 135 v1 -> superseded_runs.
  G behaviour (behavior/, 031) -> _archive/behaviour.
  .bak files kept. Results home: mapping to combined_results_ks4/<slug>/ to be proposed before any move.
- 2026-10-04 23:54 Results home (user approved): results moved from the code folder to combined_results_ks4/ssl-whisker-hitmiss-timeresolved-decoding/{tables,cache,cache/logs,figures,_archive}. haas: 407 files + figures/ moved, old paths are symlinks (0 broken); _archive results moved; manifest results_home_manifest_20261004.tsv there. The old project copy that sat in that NAS folder -> _snapshots/2026-10-04_old_project_copy/. Local (Windows) copy: 582 local-only files copied to their NAS home, 423 differing local versions to _snapshots/2026-10-04_local_copy/ (all byte-verified), then all result files removed from the local code folder (code folder now 2.3 MB). New: sync_results.sh -- run on haas after each job to move new / atomically rewritten outputs to the NAS and link them back.
- 2026-10-05 00:08 question.md rewritten to the current scope (three parts: session-wide decoding, within-session change, stimulus-onset geometry; definitions, inclusion, decoding, statistics, open parameters, known stale results). Report rebuilt (skills/project-report): Part I session-wide decoding added (113 for hit/miss, stimulus modality, performance state; 110 for pre-lick modality with the -100 ms window), Parts II-III renumbered, Part I caveats, supplementary S11-S15 (113 set 1 modality / perfstate, 113 set 2 hit/miss halves, 041 learning vs expert, 115); 33 pages, 24 figures, 160 numbers in numbers.json.
- 2026-10-05 00:38 TODO: added the hit-median split / coding-direction / noise-correlation plan and the stable-unit definition (user decisions 2026-10-05); cross-modality generalisation dropped (too few auditory misses).
- 2026-10-05 04:36 Overnight run (user: "be autonomous and run all these things tonight"):
  142 drift-shift test for the 6836 somatic units missing results (first attempt failed: no spikeinterface in the project venv, outputs in _drift_rerun_20261005_failed_no_spikeinterface; rerun with unit_spikes_analysis/.venv: 5943 units now have r / p); 137 stable-unit table rebuilt (893 somatic units still without a drift result -> quality_label good fallback).
  118 / 122 / 126 rerun with a 5-35 ms hit/miss window (tags _A1v2_hm35, _step1_hm35); 139 hit-median vs midpoint split (separate count-matched + single decoders, within-half shift null, cross-half generalisation); 140 coding direction rotation / gain, passive projection, noise ratio and Fisher information (tracked stable units).
  Figures: 143 (L5 split x 3 windows; step vs gradual learners), 144 (hit-median split), 145 (coding direction), each in all / learners scopes. Report rebuilt (II.6-II.8, III.2, methods, key results 7-9, discussion, caveats): report_20261005.pdf (report.pdf was locked by an open viewer), report.html, report.md.
- 2026-10-05 09:58 Correction: the 893 somatic units without a drift result are not short of data -- they have no spikes in the central segment the Harris shift test uses (first / last 500 s excluded), so the test is undefined; they appear / disappear during the session (median coverage 0.08) and none is in the stable set. Report methods now describe the drift test (script, settings, result paths) and the caveat is corrected; report.pdf rebuilt.
- 2026-10-05 10:08 Unit sets fixed to the user's definition (skill ssl-valid-data): stable = coverage, presence and drift joint test only (no drift result = not stable); good = quality_label good AND stable. 137 rebuilt: all 174140, stable 142242, good 43679 units; the first stable set used by 140 (stable_v1, 128785 units, also nSpikes >= 300 and spikes missing <= 20 %) is a strict subset and kept as a column. 140 not rerun yet.
- 2026-10-05 12:19 146 full run (choice decoder from active whisker trials read out on passive pre / active halves / passive post; decomposition; first-half-decoder and shuffle controls; stable and good units, whole brain and area groups); 135 / 135b and 140 / 145 rerun on the corrected unit sets (outputs _good / _stable; 140 previous set kept as _stable_v1); 147 projection now per unit (= size x cosine). Report: Methods (146), III.3, key result 10, Table 17, Figures 15, S16-S20; report.pdf rebuilt (54 pages).
- 2026-10-05 13:07 146 rerun with >= 3 hits / misses (was 6; documented) and three response variants (epoch baseline, per-trial baseline, baseline window alone) plus timing / reward / baseline-rate controls; 140 rerun with whisker- and auditory-evoked alignment; 147 learners scope and controls figure; 145 evoked-alignment figure. Report: methods (selection, controls), III.3 text revised, III.4 state and engagement controls (Figure 16, Table 18), Figures S21-S23, discussion and caveats; key result 10 reworded (first-half decoder weaker, part state).
- 2026-10-06 Drift-test status rechecked on haas (137_stable_units.parquet): 4226 units without a drift result = 3333 non-soma (never needed) + 893 somatic, all with presence <= 0.38 (fail presence >= 0.5 regardless); test undefined for them, not rerun. TODO item closed.
- 2026-10-08: 158 COSYNE figure v2 (user): within-day change on the hit / miss axis only, no expert, learners; state space (pre) -> projections per epoch (whisker, auditory) -> pre -> post excess over the linear-shift null (dots, Wilcoxon / Mann-Whitney) -> state space per cohort; one line.
- 2026-10-08: 159 within-day deck (user): methods derived step by step, whole-brain active (decoding within the session) and passive pre vs post (hit / miss axis, shift null, decoder, state space, exposure), per area, COSYNE figure 158, caveats, backup; no expert data. report/within_day_deck.pptx.
- 2026-10-08: anchored hit / miss projection (user): 135 SSL_135_ANCHOR=1 stores held-out active hit / miss projections and their d' on each split's lick axis (no extra random draws; other columns identical); 158 SSL_158_ANCHOR=1 variant: c = (pi - pi_miss) / (pi_hit - pi_miss), excess divided by the same separation, sessions with held-out d' >= 0.3; outputs _anchor.
