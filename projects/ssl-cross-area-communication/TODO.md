# TODO

- [X] Plan approved (2026-08-31); project scaffolding created.
- [X] Verified `ephys_utilities`/`allen_utils` function locations against live
  source (`combine_ephys_nwb`/`process_single_nwb` in `helpers/data_utils.py`,
  `classify_units_quality` in `neural_utils/unit_metrics_utils.py`,
  `process_allen_labels`/`get_custom_area_groups`/`get_custom_area_groups_from_name`
  in `allen_utils.py` — all as documented). Confirmed the `bwa` conda env
  (via `sys.path.insert` for `ephys_utilities`/`allen_utils`, not pip-installed)
  has `pynwb`, `sklearn` (`cross_decomposition.CCA`), `numpy`, `pandas`, `scipy`.
- [X] **Step 1 — recording-pair coverage report**: full Path B load, 119/119
  ephys sessions (`day_to_analyze='all'`, ~17.5min, `bwa` env, `max_workers=12`),
  mouse filters applied, both area hierarchy levels (coarse Allen groups /
  fine `area_acronym_custom`), >=20-unit floor, valid area-pairs enumerated
  per session x cohort x day-stage x population scope. Code:
  `000_coverage_lib.py`, `002_full_coverage_report.py`. Outputs:
  `artifacts/coverage_summary.csv`, `artifacts/valid_area_pairs.parquet`,
  `artifacts/top_fine_area_pairs.csv`, `artifacts/full_unit_table_metadata.parquet`.
  Result: coverage is ample at both levels, all scopes/cohorts/day-stages
  (see change-log for the full table) -- proceeding as planned.
- [X] Small-subset pipeline build: one session (`MH062_20260113_125836`,
  R-, expert, coarse-level "Motor and frontal areas" vs "Somatosensory
  areas", 40 units/area subsampled from ~1000/~970 available), windowing
  (`003_cca_lib.py`, reuses `scripts/ssl_bwm_windows.py`'s dead-zone
  clipping), noise-correlation residualization (PSTH-subtracted, no
  baseline correction), variant-A CV-CCA + PCA baseline, piezo-lick
  regressor construction (variant C mechanics). Code: `003_cca_lib.py`,
  `004_small_subset_pipeline.py`.
- [X] **Intermediate checkpoint figures produced** (`checkpoint_01..05.png`
  in `artifacts/`): input-data rasters, noise-correlation illustration,
  piezo-lick regressor illustration, held-out example correlated trials
  (fixed from an initial in-sample version -- see change-log), CV-CCA vs.
  PCA baseline scree. Presented to Axel 2026-08-31.
- [ ] **Extended checkpoint** (Axel follow-up, 2026-08-31): multi-panel
  pipeline figure (PSTH->residual->PCA->CCA, variants A/B/C together),
  per-canonical-dimension statistical test results per cohort, a final
  cross-cohort/cross-analysis correlation-magnitude summary, good-only vs.
  good+MUA quality tiers run separately, restricted to the **learning**
  day-stage only across **3 R+ + 3 R- example sessions/mice** (not yet the
  whole dataset) -- per Axel's explicit "show more examples for a few mice
  before doing the whole dataset" instruction. Code:
  `005_multi_session_checkpoint.py`. Ran cleanly (720 result rows, no
  exceptions). Outputs: `checkpoint2_01_pipeline_illustration.png`,
  `checkpoint2_02_scree_good.png`, `checkpoint2_02_scree_good_mua.png`,
  `checkpoint2_03_final_summary.png`, `multi_session_checkpoint_results.csv`,
  `multi_session_checkpoint_dim_stats.csv`, `multi_session_checkpoint_summary.csv`.
  Headline finding: variant A (plain) and C (partial out lick rate) give
  near-identical dim-1 correlations (~0.11-0.23 across tier/trial-type/cohort
  cells); variant B (partial out all other simultaneously-recorded neurons)
  drops sharply toward the PCA baseline in most cells -- suggests much of
  the apparent A-B "communication" is shared with the rest of the recorded
  population (brain-wide state), not licking-specific. Per-dimension
  R+-vs-R- and CCA-vs-PCA stats are descriptive only at n=3 sessions/cohort
  (Wilcoxon can't reach p<0.05 at this n) -- some paired-t nominal hits
  (e.g. good_mua/whisker/R-/dim1 p=0.017) worth re-checking at full scale.
  **Presented to Axel 2026-08-31 -- still a hard-gate checkpoint, do not
  scale to the full dataset or add expert-arm/other area-pairs without
  explicit approval.**
- [X] **PSTH-on-canonical-axes projection + lagged pCCA** (Axel follow-up,
  2026-08-31): 2 example sessions (1 R+, 1 R-, good+MUA, whisker trials),
  CCA fit on a train half of trials, everything computed on the held-out
  test half. Code: `006_temporal_lag_analysis.py`. Outputs:
  `checkpoint3_01_psth_projection_and_time_corr.png` (mean-PSTH canonical
  projection + trial-by-trial across-time correlation at lag=0),
  `checkpoint3_02_lagged_correlation_matrix.png` (time x lag matrix, +/-100ms
  in 10ms steps), `checkpoint3_03_best_lag_line.png` (lag=0 vs. best-mean-lag
  line). Ran cleanly, no exceptions. Result: best-lag improvement over
  lag=0 is small (R+: r=0.094->0.115 at +10ms; R-: r=0.133->0.135 at +80ms)
  -- likely mostly noise at n=1 session/cohort with 21 lags scanned (a
  "pick the best of 21" selection effect); not a confident lead/lag claim
  yet. Flagged to Axel as needing the full session count + a permutation-
  based null (e.g. circular-shift or trial-shuffle) before trusting a
  specific lag value. Presented 2026-08-31.
- [X] **Shuffle control, neuron-level matrix, R+/R- stats, 20 dims** (Axel
  follow-up, 2026-08-31): square/labeled plots; trial-identity shuffle
  control (500 shuffles, lag=0 only) for variants A/B/C; mandatory dead-zone
  exclusion applied throughout (never just NaN-masked -- found and fixed a
  real crash risk: sklearn's fit/predict/transform reject NaN outright, so
  every fit input now uses bin-axis-restricted tensors, not just the final
  analysis); neuron-level (raw mean-pairwise, no CCA) lagged matrix added
  alongside the canonical one, both annotating the excluded dead-zone bin;
  Mann-Whitney+Welch R+-vs-R- test per (tier, trial_type, variant,
  dimension); n_dims up to 20; variant-B nuisance PCA now retains PCs to
  99% variance (was 90%/15). Code: `007_shuffle_significance_and_stats.py`,
  new lib helpers in `003_cca_lib.py` (`dead_zone_bin_mask`,
  `project_tensor_multi`, `fit_nuisance_pca`). Ran cleanly (1440+240 result
  rows, no exceptions). Outputs: `checkpoint4_01_dims_significant_vs_shuffle.png`,
  `checkpoint4_02_lagged_matrix_canonical.png`,
  `checkpoint4_03_lagged_matrix_neuron_level.png`,
  `shuffle_stats_results.csv`, `shuffle_stats_Rplus_vs_Rminus.csv`. Result:
  dim 1 significant vs. shuffle in 68.6% of 72 iterations (p<0.05), decaying
  to near chance level by dim ~6-7 -- confirms communication is concentrated
  in 1-2 canonical dimensions. Variant B (partial out other neurons) again
  lower than A/C at dim 1 (0.072 vs 0.110/0.110), replicating the earlier
  finding. Neuron-level (no-CCA) pairwise correlation is near-zero
  everywhere the canonical matrix shows strong structure -- CCA finds a
  concentrated communication direction invisible to simple averaged pairwise
  correlation. R+-vs-R- stats mostly null at n=3/cohort (20/240 nominal
  hits at p<0.1, near chance-level count), with a directionally-consistent
  nominal hit (good_mua/whisker/dim1, Welch p=0.026, R+ 0.194 > R- 0.139)
  worth re-checking at scale. **Presented 2026-08-31 -- still checkpoint
  stage on the same 6-session subset; awaiting go-ahead to scale to the
  full dataset.**
- [X] **Switch to the `partial_CCA` library** (Axel, 2026-08-31): replaced
  sklearn's `CCA` (an iterative NIPALS/PLS approximation) with
  `partial_CCA.PartialCCA` (Gonzalez, Buzsaki/Chen labs, MIT license, from
  the paper doi:10.1038/s41586-026-10481-z) -- closed-form eigendecomposition
  CCA with explicit ridge regularization; partialling (Z) now handled
  internally by `.fit(X,Y,Z)` instead of a hand-rolled regress-out step.
  Installed into the `bwa` env from a locally downloaded TestPyPI wheel.
  Shuffle test corrected/re-explained: each of 500 draws permutes area B's
  TRIAL identity and REFITS PartialCCA from scratch (not just a reprojection
  through the real fitted weights, which was 007's method and tests a
  narrower question) -- gives a genuine per-dimension null distribution.
  Code: `008_pcca_library_shuffle_and_gallery.py`, new lib wrapper usage in
  `003_cca_lib.py`. Ran cleanly. Outputs: `pcca_01_scree_shuffle_good.png`,
  `pcca_01_scree_shuffle_good_mua.png` (shuffle mean+/-SEM band, significance
  stars), `pcca_02_psth_canonical_gallery.png` (all 6 sessions, labeled),
  `pcca_lib_results.csv`. **Important finding: with the corrected refit-based
  null, dim-1's observed correlation (0.267) is much closer to its own null
  (0.174) than the earlier (weaker) shuffle method suggested -- pooled
  p~=0.11, a materially more conservative picture.** The raw magnitudes are
  also higher across the board than the sklearn-CCA numbers (expected: no
  more PCA-truncation-driven underestimation, but also no more CV-based
  correction either -- the shuffle null is now the sole safeguard against
  the known dimensionality-driven inflation of naive CCA).
- [X] **New rules** (Axel, 2026-08-31): 5ms bins (was 10ms, dead zone still
  excluded), 2ms-resolution lagged pCCA (was 10ms), tier-specific unit
  floors (>=10 good OR >=30 good+MUA per area, replacing the flat >=20),
  and a canonical-correlation-across-areas-and-dimensions comparison. 3
  area pairs run (of the 3 areas eligible in all 6 sessions under the new
  floors: Motor and frontal, Somatosensory, Striatum and pallidum) x 2
  tiers x 6 sessions x 2 trial types, still refit-based shuffle null.
  2ms lagged pCCA kept to the original Motor-Somatosensory pair x 2 example
  sessions (compute-scope decision, stated explicitly) -- weights from the
  5ms fit projected onto a separately-built 2ms-binned tensor (canonical
  weights are neuron-indexed, not bin-indexed, so this is valid). Found and
  fixed a real bug while writing this: the lag-analysis section originally
  reselected a FRESH random unit subsample instead of reusing the exact
  population the cached model was fit on -- would have silently produced
  meaningless projections (`model.transform` indexes features by fit-time
  identity/order). Code: `009_new_rules_multi_area.py`. Outputs:
  `newrules_01_areas_x_dims_good.png`, `newrules_01_areas_x_dims_good_mua.png`,
  `newrules_02_lag_2ms_matrix.png`, `new_rules_results.csv`.
- [X] **Consolidated final figures** (Axel, 2026-08-31): pure replotting from
  `new_rules_results.csv` (no new compute -- that run already covers all 3
  pairs x 2 tiers x 6 sessions with the corrected shuffle null). Every
  correlation-vs-dimension plot now also shows correlation ABOVE the
  shuffle null (observed - null) as a second panel/line; significance now
  shown via marker size/saturation (large+dark = >=2/3 of that line's
  sessions individually beat their own 500-shuffle null at p<0.05; small+
  faded = not significant) instead of stars. Code: `010_final_figures.py`.
  Outputs: `final_01_corr_and_excess_by_cohort_{good,good_mua}.png` (3
  pairs x R+/R- x raw+excess, 6-panel grid per tier), `final_02_areas_
  pooled_corr_and_excess_{good,good_mua}.png` (3 pairs pooled across
  cohort, raw+excess). **Notable finding, verified not a bug (checked
  raw per-session numbers directly): observed correlation stays
  consistently, if weakly, above the shuffle null across MANY dimensions
  (not just dim 1-2) for Motor-frontal vs Striatum -- e.g. dim 10:
  observed 0.055 vs null 0.043, p=0.002, consistent in 5/6 sessions.**
  This is a materially different picture from the earlier "concentrated in
  1-2 dims" impression (checkpoint4) and needs scrutiny before trusting it
  as real broadband communication rather than a shared low-level confound
  (e.g. arousal/motion) affecting many canonical dimensions weakly and
  pervasively -- flagged to Axel, not yet resolved.
- [ ] **In progress**: variants A/B/C subplot figures, lick_flag stratification,
  per-variant output folders, pooled (n=6) distributional significance test
  (Axel, 2026-08-31). `auditory_trial x lick_flag==0` dropped -- checked
  trials.parquet first, only 0-6 trials/session across all 6 sessions
  (near-ceiling auditory performance), not analyzable at this scale.
  Code: `011_variant_subplots_by_outcome.py`. First run crashed: variant C's
  `lick_v` was never computed (initialized to None, no code ever set it) --
  every C iteration silently skipped, giving 0 rows, which then crashed the
  plotting section on an empty DataFrame. Variants A and B (2160 rows each,
  verified well-formed) had already completed and saved to disk before the
  crash, so the fix (added the missing piezo-lick loading/binning) reruns
  ONLY variant C and loads A/B from their saved CSVs rather than redoing
  ~2/3 of a very expensive run. Rerun completed cleanly (2160 rows each for
  A/B/C, matching exactly). A cosmetic layout bug (suptitle overlapping the
  3x3 grid's top-row subplot titles) found on visual review, fixed via a
  pure-replot script (`012_fix_figure_layout.py`, no compute, reads the
  saved CSVs) -- all 12 figures regenerated cleanly.
  Outputs: `variant_{A,B,C}/results.csv` (2160 rows each), `variants_ABC_
  {raw,excess}_{tier}_{trial_type}_lick{0,1}.png` (12 total), `variant_
  significance_tests.csv` (1080 rows, pooled n=6 paired Wilcoxon+t-test).
  151/1080 (dimension, pair, variant, condition, tier) combinations
  significant. Key patterns: variant B (partial out other neurons) has
  visibly fewer significant hits than A/C across all 3 pairs (31 vs 57/63)
  -- replicates the earlier finding that most apparent inter-areal
  correlation is shared with the rest of the recorded population, not
  specific to the pair. Somatosensory-vs-Striatum is the only pair whose
  signal partially survives variant B's partialling (a couple of
  significant dimensions remain), unlike Motor-Somatosensory and
  Motor-Striatum (variant B knocks both down to zero significant
  dimensions) -- Somatosensory-Striatum looks like the most genuinely
  pairwise-specific communication of the three. whisker_trial lick_flag=0
  (miss/withhold trials) shows MORE significant hits than lick_flag=1
  across all 3 variants (29/21/31 vs 12/2/12) -- an unexpected, notable
  lead worth flagging rather than assuming hit trials would show the
  strongest signal.
- [X] **Confirm-stage additions before full scaling** (Axel, 2026-08-31):
  removed significance-dot styling (uniform dots now), y-axis shared per
  area-pair row across the 3 variant columns (tick labels kept visible),
  lag-0 summary heatmaps (pairs x conditions, one panel per variant), and
  a peak-lag distribution across all 54 (session x pair x condition)
  interactions (variant A, good+MUA). Code: `013_uniform_dots_and_heatmaps.py`
  (pure replot, no compute), `014_peak_lag_distribution.py` (new but cheap
  compute -- no shuffling needed for a descriptive peak-lag summary).
  Peak-lag result: median 0.0ms, 69% of interactions within +/-2ms of
  lag-0, no directional bias in the scattered outliers -- consistent with
  genuine near-zero-lag communication, no confident lead/lag claim
  supported at this scale (as already flagged). All confirm-stage checks
  passed -- pipeline validated end to end.
- [X] **Full learning-dataset run LAUNCHED** (Axel said "confirm then run",
  2026-08-31): 50 sessions (26 R+, 24 R-) with all 3 area pairs at >=30
  good+MUA units (`artifacts/full_learning_session_list.csv`, checked
  directly against `full_unit_table_metadata.parquet`). Scope: good_mua
  tier only (dropped 'good' -- consistently same qualitative pattern
  throughout the checkpoint, a compute-budget call not a finding), 300
  shuffles (down from 500, ~0.003 p-resolution, also a compute-budget
  call given the ~8x larger session count). All 3 variants, all 3
  conditions, all 3 pairs unchanged. ~405,000 PartialCCA fits, estimated
  ~2.5-3h. Code: `015_full_learning_run.py` -- smoke-tested first on a
  2-session/5-shuffle slice (caught and fixed a real crash: the plotting
  section's y-limit computation raised `ValueError: min() iterable
  argument is empty` when a pair/condition combo had zero valid sessions
  in any variant/cohort; now falls back to auto-scale instead of crashing).
  Output goes to `artifacts/full_learning/` (separate from the 6-session
  checkpoint's `artifacts/variant_A|B|C/`, never overwrites it). Started
  2026-08-31, running in background with periodic check-ins.
- [X] **Full run completed cleanly** (~68min actual, faster than the
  2.5-3h estimate -- all 50 sessions done, no exceptions). 8400 rows/variant
  (420/450 possible session x pair x condition combos had enough
  trials/units), `variant_significance_tests.csv` (540 rows). At n=50,
  519/540 (96%) combinations are nominally significant (paired Wilcoxon +
  t-test both p<0.05) -- flagged as a large-N statistical-power effect, not
  a "everything is real" claim: report EFFECT SIZE (excess correlation),
  not the now-near-saturated significance count. Effect sizes replicate the
  6-session checkpoint cleanly: variant B excess=0.049 vs A=0.086/C=0.085
  (43% reduction, matches "partial out other neurons" finding); Somatosensory-
  Striatum has the largest excess of the 3 pairs under variant A (0.096 vs
  0.081/0.086) AND is the only pair whose signal partially survives variant
  B (excess 0.062 vs 0.030-0.035 for the other two pairs) -- both patterns
  replicate exactly. R+ vs R- comparison (Mann-Whitney + Welch, dim-1,
  variant A, whisker/lick1) found NO significant cohort difference for any
  of the 3 pairs (all p>0.17) -- a clean, well-powered null result, not
  an underpowered one. Outputs: `full_variants_ABC_{raw,excess}_
  {trial_type}_lick{flag}.png` (6 total), `full_heatmap_lag0_summary.png`.
  **Presented to Axel as the culminating report for this project phase.**
- [ ] Unit-count confound check, PSTH-on-canonical-axes projection at full
  scale (currently checkpoint-only).
- [X] **`report.html` written** (Axel, 2026-08-31, "write the detailed
  report, including lots of methodological details"): 10 sections --
  motivation, data/loading, full methodology (area hierarchy/unit floors,
  trial conditions, binning/dead-zone, noise-correlation construction, the
  pCCA engine and why it was switched, the three variants, the shuffle-test
  protocol re-explained, PSTH-on-canonical-axes, lagged pCCA), a dedicated
  "methodological corrections made along the way" section (7 bugs/design
  fixes documented as methodology, not just debugging notes), coverage
  results, 6 embedded figures, full-dataset effect-size results (3 findings
  stated explicitly), lag analysis, caveats/limitations, and suggested next
  steps. Code: `016_final_report.py` (all numbers computed from the saved
  CSVs at report-build time, not hand-typed). Private local file only
  (`projects/ssl-cross-area-communication/report.html`), not published --
  SSL data is private/unpublished by default per AGENTS.md.
- [X] **Extension list started** (Axel, 2026-08-31): within-area correlation
  control, all-pairs (coarse+fine) run, no unit subsampling, unit-count-vs-
  correlation sweep, new area x area heatmap design (shared colorbar,
  row=target/col=source), global PSTH replot, R+/R- colors fixed to
  #00B400/#C800C8 (saved to memory as a standing convention). Scoped via
  AskUserQuestion given the raw "all pairs" request would take
  days-to-weeks: Axel specified "pairs with >=3 mice per cohort, >=20
  good+MUA units/area" -> 40 coarse / 226 fine pairs. Timed real fits with
  synthetic data at actual dimensions before committing (median coarse
  0.23s/fit, max 1.3s/fit, fine ~0.01s/fit) -> chose N_SHUFFLES=100 (down
  from 300), estimated ~2 days total. A first compute script
  (`017_all_pairs_run.py`) was written and smoke-tested, but **Axel asked
  to stop the run and inspect the dataset first** -- stopped
  (`TaskStop`), smoke-test artifacts cleaned up, full run NOT yet
  launched.
- [X] **Coarse-area regrouping** (Axel, 2026-08-31, after inspecting the
  data): dropped Olfactory areas and Amygdala-and-hypothalamus entirely;
  renamed "Visual areas" -> "PPC" (verified: that coarse bucket is 100%
  PPC units in this dataset, 3466/3466, zero true visual-cortex units);
  split "Somatosensory areas" into "Whisker somatosensory" (SSp-bfd only)
  and "Orofacial somatosensory" (SSp-m + SSp-n) -- residual fine areas
  (SSs, AI, SSp-ul, SSp-ll, GU, VISC; 10,024/129,127 learning-arm units)
  dropped from the coarse pairing scheme, kept as-is at the fine level.
  Recomputed pairs under the new grouping: **38 coarse pairs** (was 40).
  Outputs: `full_unit_table_metadata_v2.parquet`, `pairs_final_coarse_v2.csv`,
  `pairs_heatmap_coarse_v2.png`. Also produced a relaxed, cohort-unsplit
  fine-level exploratory view (>=3 total co-recorded sessions, not
  requiring 3 mice/cohort) per Axel's "just to see" request: **554 fine
  pairs**, `pairs_heatmap_fine_loose.png`.
  Also produced (before the regrouping request): pairs-count summary
  table/figure (`pairs_per_level_summary.{csv,png}`) and per-cohort
  coverage heatmaps at the original grouping (`pairs_heatmap_{coarse,fine}.png`).
  **Awaiting Axel's review of the dataset/grouping before the all-pairs
  compute run is relaunched.**
- [X] **Didactic pipeline figure + fixes** (Axel, 2026-09-01): dead-zone
  reverted to -10ms/+5ms for this project specifically (project-local
  override in `003_cca_lib.py`, shared `scripts/ssl_bwm_windows.py` left
  untouched at -1ms/+4ms for other SSL projects -- sanctioned as safe per
  `ssl_artifact_dead_zone.md`'s own "wider dead zone is always
  conservative" clause). Found and fixed a real math bug: averaging PC1/
  CCA-variate over the SAME trials used to fit them gives exactly zero by
  construction (residual = raw - trial_mean) -- fixed with held-out
  train/test splits. Binning changed to a 5ms-width/2ms-stride SLIDING
  window (was non-overlapping 5ms tiles) -- new `sliding_window_*`/
  `*_sliding` functions added to `003_cca_lib.py` alongside (not replacing)
  the old tiling functions, for backward compat with scripts 004-017.
  Found a real instability: auditory-trial CCA (51 trials, ~800-1000
  features, no CV) gave a degenerate r=1.00 driven by one outlier held-out
  trial. Fixed with `select_regularization_trial_cv` (new, in
  `003_cca_lib.py`) -- trial-level k-fold CV (not naive row-level, which
  would leak across highly-correlated overlapping-window bins) to pick the
  ridge lambda; gated to only run for conditions with <100 trials (a first
  attempt with a full 6-lambda x 3-fold grid on every condition took >20min
  for 2/63 sessions, ~10h+ extrapolated -- cut to a 3-lambda x 2-fold grid,
  trial-count-gated). Also fixed a real oversight: the pipeline scripts
  (018/019) had silently dropped the lick_flag split, using only 3 trial
  TYPES instead of the correct 5 conditions (whisker x{0,1}, auditory x1,
  no_stim x{0,1}). Added a genuine correlation-ACROSS-TIME row (trial-by-
  trial Pearson r per time bin, `canonical_correlation_across_time`) --
  distinct from the canonical-VARIATE-amplitude row that was the only CCA
  view before Axel asked "are you plotting the variate or correlation?".
  Code: `018_pipeline_all_trial_types.py` (single example session),
  `019_pipeline_full_dataset_average.py` (63-session average, sign-aligned
  PC1/CCA-variate across sessions). Both smoke-tested before full runs.
  **Result: auditory-trial instability fully resolved** (no more r=1.00
  spike) -- shows a clean, physiologically sensible RISING correlation
  after auditory onset (~0.1 baseline -> ~0.3 post-stimulus, mean
  whole-window |r|=0.71). All 5 conditions show smooth, positive,
  0.1-0.35-range trial-by-trial correlation across time.
  Outputs: `example_pair_01_raster.png`, `example_pair_02_pipeline_across_time.png`,
  `example_pair_03_pipeline_full_dataset_average.png`.
  **Critical scoping finding: this ONE pair's 63-session, 5-condition
  average took ~4h50min wall time**, even with the reduced/gated CV. Naive
  extrapolation to all 39 coarse pairs is ~8 days -- NOT feasible as
  currently designed. Flagged to Axel; awaiting direction on how to cut
  cost further (e.g. a fixed lambda informed by this run's CV results
  instead of per-session CV, or CV once per pair instead of per
  session/condition) before building/launching the full multi-pair run
  Axel requested ("run the full pipeline across variants, coarse pairs,
  conditions... summarize findings").
- [X] **Coarse regrouping refined again** (Axel, 2026-08-31): SSp-w now =
  SSp-bfd + SSs (was SSp-bfd alone; SSs moved from "dropped residual" into
  the whisker group), SSp-orofacial unchanged (SSp-m + SSp-n). Confirmed:
  strict >=3 mice/cohort AND >=20 good+MUA units/area criteria kept for
  BOTH levels (the earlier fine-level "loose" view was exploratory only,
  not adopted as the real criterion -- `pairs_final_fine.csv`, 226 pairs,
  already used the strict criteria and is unchanged). New coarse count:
  **39 pairs** (was 38). Outputs: `full_unit_table_metadata_v3.parquet`,
  `pairs_final_coarse_v3.csv`, `pairs_heatmap_coarse_v3.png`. Still
  awaiting go-ahead before relaunching the all-pairs compute run.
- [~] **R+/R- split + correlation-vs-dimension for the example pair** (Axel,
  2026-09-01): 019's 63-session average pooled R+/R- and only ever captured
  dimension-1 correlation -- and never saved per-session intermediates, so
  neither could be produced retroactively without a full rerun. New script
  `020_pipeline_rplus_rminus_and_dims.py`: same fitting logic as 019
  (unchanged compute cost), but now (1) tags every session with
  `reward_group`/`mouse_id`/unit&trial counts, (2) reads out ALL canonical
  dimensions (up to `MAX_DIMS=10`) from the existing `model.transform()`
  call -- free, no extra CCA fits, 018/019 only ever read dim 0 -- and (3)
  pickles everything (curves + full pipeline config) to
  `pipeline_full_dataset_results.pkl` so future re-slicing never needs a
  recompute again. New standing convention saved to memory
  (`ssl_cross_area_always_save_metadata`, "All metadata must be always
  saved," Axel 2026-09-01). Smoke-tested on 4 sessions (2 R+, 2 R-) --
  clean run, dead-zone gaps correct, scree plot shows expected decreasing
  correlation with dimension (0.2-0.9 range, no degenerate spikes). Full
  63-session run launched (background task `bh5cclir7`), same ~5h order of
  magnitude as 019 expected. Outputs pending:
  `example_pair_04_rplus_rminus_comparison.png`,
  `example_pair_05_correlation_vs_dimension.png`.
  **Also found the correct conda env for this pipeline is `bwa`** (has
  `partial_CCA` installed from a local wheel per `003_cca_lib.py`'s
  docstring) -- `iblenv2`/`nwb` lack it, worth remembering for future runs.
- [~] **Binning changed to 10ms width / 5ms stride** (Axel, 2026-09-01:
  "Let's start again with 10ms resolution and 5 ms stride/step size"; was
  5ms/2ms). Stopped the in-progress 5ms/2ms full run (task `bh5cclir7`,
  no session had finished yet) before it wasted more compute. Updated
  `003_cca_lib.py`'s `BIN_WIDTH`/`BIN_STRIDE` module constants (all
  sliding-window/dead-zone-mask helpers are parameterized off these, no
  other code changes needed). Re-smoke-tested on 4 sessions (2 R+, 2 R-) --
  clean, dead-zone gap still renders correctly at the wider bin width. Full
  63-session run relaunched with the new binning (background task
  `bq68n4hiq`).
- [~] **Speed pass + methodology follow-ups** (Axel, 2026-09-01): fixed
  lambda per condition (no per-session CV), truncated `PartialCCA`
  weights to `MAX_DIMS` before `.transform()` (the vendored library loops
  in Python over ALL `min(n_x,n_y)` dims otherwise -- was likely the
  single biggest hidden cost), vectorized `003_cca_lib.py`'s spike-binning
  (dead-zone clip computed once/shared across units, batched
  `searchsorted` instead of a per-bin Python loop), and parallelized the
  session loop (`ProcessPoolExecutor`, 10 workers -- machine has 24
  cores/32 threads, was using 1). New script
  `021_pipeline_parallel_fixed_lambda.py`, smoke-tested clean on 8
  sessions (both cohorts, correct out-of-order completion confirming real
  parallelism). **Variant C (partial out piezo-lick rate) dropped from
  scope** -- only variants A (plain CCA, built) and B (partial out other
  neurons, still to build) remain planned. **Open, awaiting Axel's
  confirmation before wiring in**: (1) PCA-pre-reduce each area to
  K=min(K_90%-variance, n_trials/5) PCs before CCA, to keep CCA
  statistically well-posed given p (units) vs effective N (trials, not
  trials x bins -- overlapping sliding-window bins aren't independent);
  (2) reinstate trial-identity-shuffle null-distribution baseline
  alongside the (also not-yet-rebuilt) PCA-scores baseline, default 20
  shuffles/condition (refit per shuffle) pending confirmation.
  Housekeeping: the old serial 020 full run (background task `bq68n4hiq`,
  10ms/5ms binning, per-session CV) was still running ~62min after 021
  superseded it, and writes to the SAME output files as 021 -- stopped it
  to avoid it silently clobbering 021's (better-methodology) results and
  to free cores.
  **Confirmed by Axel (2026-09-01): 10 shuffles (not 20), and PCA reduces
  the neuron/feature axis (confirmed understanding, not a change).** Built
  `022_pipeline_pca_reduced_shuffle_baseline.py`: PCA-pre-reduces each
  area to K=min(K_90%-variance, n_trials//5) components before CCA (single
  small-n_components PCA fit, not a full-rank one -- stays cheap), adds a
  PCA-alignment baseline (naive PC1-vs-PC1 correlation, no CCA
  optimization) and a 10-shuffle trial-identity-null baseline (full refit
  per shuffle, area B's trial axis globally permuted before the whole
  PCA-reduce+fit+transform pipeline reruns). New comparison figure
  `example_pair_06_baselines_comparison.png` (CCA vs PCA-baseline vs
  shuffle-null, mean+/-SEM per cohort per condition).
  Smoke-tested on 8 sessions -- clean, sensible: median K ranges 4
  (no_stim x1, ~50-100 trials) to 42 (no_stim x0, ~150-300 trials); true
  CCA mean|r| (0.26-0.31) consistently exceeds shuffle-null (0.11-0.12)
  across all 5 conditions -- real cross-area signal beyond chance: PCA
  baseline is lower than CCA in most conditions (CCA finds structure
  beyond naive variance alignment) except auditory x1 and no_stim x1 where
  they're comparable. All 3 figures (04/05/06) reviewed, dead-zone gaps
  and K-capped scree-curve truncation both correct.
  Full 63-session run launched (background task `b9cu1k99r`).
- [~] **Full 39-coarse-pair batch run** (Axel, 2026-09-01: "run for the full
  dataset, all pairs"). New `026_all_pairs_driver.py`: derives per-pair
  session lists directly from `full_unit_table_metadata_v3.parquet` (no
  NWB reload needed for this step; verified byte-for-byte against
  `motor_striatum_session_list.csv`, 63/63 match), runs `024`'s full
  figure set (01/02/03/05/06/10) for each pair into
  `artifacts/runs/<area_a>_vs_<area_b>/`, cheapest pairs first, one row
  appended to `artifacts/runs/all_pairs_summary.csv` per pair as it
  completes (survives an interrupted run), each pair wrapped in
  try/except so one failure doesn't kill the batch.
  **Real bug caught and fixed during 2-pair smoke test**: `024`
  (via `022`) only ever produced the OLD `area_group_coarse` column when
  loading units fresh from NWB -- the 39-pair definitions use the v3
  regrouping (`area_group_coarse_v3`: PPC/SSp-w/SSp-orofacial splits),
  whose computation was never saved as a reusable script (only existed
  baked into the cached parquet from an earlier session). Both smoke-test
  pairs (Hippocampus-vs-SSp-orofacial, PPC-vs-SSp-w -- both touching v3-
  renamed/split areas) were silently skipped as "missing a cohort" before
  the fix, while Motor-frontal-vs-Striatum had worked all along only
  because neither area was touched by the v3 relabeling. Reverse-
  engineered the exact mapping from the parquet and added it as
  `COARSE_GROUP_V3_OF` in `000_coverage_lib.py` (`load_units()` now always
  produces `area_group_coarse_v3` too); `024`'s `build_sessions_for_pair`
  updated to filter on it. **Also cleared `artifacts/cache/` after this
  fix** -- the fingerprint cache covers input parameters, not code-logic
  changes, so a stale cache from before this fix could have been silently
  reused; worth remembering to clear the cache manually after any future
  pipeline logic change, not just parameter changes.
  Re-smoke-tested clean (both pairs `ok`, ~150-185s each cold, figures
  verified pair-specific and sensible). Full batch launched (background
  task `bb8kecb0g`), cheapest-pairs-first; ~956 total session-instances
  across 39 pairs extrapolates to roughly 4-6h wall time.
  **COMPLETE: 39/39 pairs `ok`, zero failures, total wall time ~4.45h**
  (sum of per-pair elapsed times, `artifacts/runs/all_pairs_summary.csv`).
  Each pair's 01/02/03/05/06/10 figures + `10_window_quantification_stats.csv`
  live in `artifacts/runs/<area_a>_vs_<area_b>/`. **Not yet done**: the
  cross-pair aggregation figures/analyses described earlier (summary
  matrices, R+/R- difference matrix with FDR correction, peak-lag matrix,
  scree summary, global canonical-axis PSTH) -- these read FROM the 39
  per-pair outputs and haven't been built yet. **Superseded for the
  cohort-comparison question by `031_windowed_summary.py`** (raw
  p-values not FDR-corrected per Axel's later instruction, R+/R- shown
  as separate triangle/bars not averaged, baseline/sensory windows always
  separate, filtered to 35/39 pairs with >=5 sessions/cohort) -- see that
  script's docstring and the 2026-09-01 report for the current findings
  (robust R+>R- effect on no_stim_trial(lick=1), replicates across both
  windows and both CCA variants).
- [~] **Fine-area-level batch** (Axel, 2026-09-01: "do the same but for
  the fine area level... only pairs with at least 5 recordings of each").
  Generalized `024_updated_figures.py` (`AREA_COLUMN`, defaults to
  `area_group_coarse_v3`, settable to `area_acronym_custom`) and built
  `032_all_fine_pairs_driver.py` deriving pairs directly from the
  >=5-sessions/cohort criterion (69 fine areas -> 82 qualifying pairs,
  1291 total session-instances, ~35% more than coarse's 956 -> expect
  ~5-6h). Smoke-tested clean on 2 pairs (AUD-vs-SCm, DLS-vs-MO-wM2),
  confirmed correct fine-level labeling. Full 82-pair batch launched
  (background task `btpqfpfpf`). Once done: rerun a fine-level analog of
  `031_windowed_summary.py` for the actual cross-pair report.
- [ ] Privacy checklist before anything leaves the local environment.
