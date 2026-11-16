# Change log

- 2026-08-26: Project created. Plan-mode Q&A resolved: both day-stages run
  separately; exploration/confirmation split explicitly skipped ("preliminary
  analysis"); post-hoc area scheme = fine `area_acronym_custom`
  (`split_merge_areas=True`). User then requested an intermediate
  burst-detection validation report (small subset) before any full-scale run
  — added as a hard checkpoint in `question.md`/`TODO.md`.
- 2026-08-26: Intermediate checkpoint built and validated on a 6-session
  subset (4366 units). Found and fixed two bugs during validation: (1)
  `cluster_id` is not session-unique in this Path B unit table (raw
  per-probe KiloSort id) — fixed with a `unit_uid` composite key used for
  every unit lookup from here on; (2) `burst_index` can hit exactly +-1.0 at
  very low spike counts (1-2 spikes total in a window) purely from the fB=0
  convention — added a >=15-spikes-per-window floor for example selection;
  **not yet resolved for the full statistics population** — flagged as an
  open item to decide with the user before the full-scale run (drop, floor,
  or weight low-count units). Report at `artifacts/intermediate_report.html`,
  awaiting user sign-off before scaling up.
- 2026-08-26: User approved ("Ok yes. Go.") without answering the low-count
  question explicitly — locked a default: raw metric always computed as
  specified (fB=0 convention preserved), but a `*_valid` flag per metric
  (continuous_burstiness: >=50 total spikes; burst_index: >=15 spikes in
  BOTH windows) gates inclusion in all statistics. Stated explicitly for
  correction if wrong.
- 2026-08-26: Full-scale compute (003) restarted once to add a tqdm progress
  bar across the per-unit loop (per user request), after confirming it was
  still in the NWB-loading phase (cheap to restart). Learning arm took
  ~81min (129,127 units, slower than the smoke-test estimate — likely
  larger average spike counts in the full population); expert arm ~12.5min
  (44,678 units).
- 2026-08-26: Statistics (004) found and fixed a real performance bug in the
  reused `permanova.py` (this project's own copy, not the original files in
  ssl-reward-history-modulation/ssl-passive-coselectivity): (1) the
  per-permutation label expansion used `pandas.Series.map()` over the full
  unit array every one of 4999 iterations — replaced with a precomputed
  integer mouse-index array + numpy fancy indexing, verified
  byte-for-byte-equivalent on synthetic data before deploying (~1.7x
  speedup alone); (2) far bigger issue, caught by explicit user correction:
  `run_with_posthoc` unconditionally runs a full permutation test for
  *every* area (~40-70 areas) regardless of main-PERMANOVA significance —
  this alone was the multi-hour cost, not N_PERM. Fixed by gating: compute
  only the main pooled PERMANOVA first, and only call the per-area post-hoc
  when p<0.05. Also reduced N_PERM 4999->999 for this preliminary pass
  (documented in permanova.py; rerun at 4999 if this becomes a locked
  confirmatory test). Combined, runtime dropped from an estimated ~4h to
  a few minutes. Result: 6/24 combinations PERMANOVA-significant
  (4x expert burst_index_whisker across both scopes/tiers, agreeing with
  the simple global test each time; 2x learning burst_index_auditory,
  good-tier-only, *disagreeing* with the simple global test — flagged per
  the project's own "report disagreements, don't cherry-pick" rule). 0/6
  areas survive BH-FDR post-hoc in any significant combination.
  continuous_burstiness was never significant anywhere.
- 2026-08-26: Motion/licking control (005) found via `piezo_lick_times`
  events and DLC jaw/whisker/nose velocity keypoints (both already in the
  compressed `ssl_ephys`/`ssl_behavior` datasets — used for this control
  only, not the mandatory Path B burstiness pipeline). No significant
  correlation with continuous burstiness in any arm/tier (all p>0.05);
  DLC tracking coverage was low for the expert arm (9/29 sessions).
- 2026-08-26: Follow-up burst-structure characterization (007, small
  validated subset): added `burst_lib.burst_runs()` to expose each burst's
  own (n_spikes, duration, start_time) rather than just the per-spike
  boolean label. 7,130,058 bursts across 4366 units: median 2 spikes/burst,
  median duration ~7.2ms, median inter-burst interval ~0.26s. Found a real,
  expected extreme tail (427/4366 units contribute bursts >20 spikes, up to
  11,393 spikes/26s in one case) driven entirely by high-sustained-rate
  **MUA** units (25-44Hz) -- a known limitation of pure-ISI burst detection
  on noisy/pooled multi-unit clusters, not a bug (checked directly:
  quality_label=='mua' for the top offenders). Plots capped at p99.9 to show
  the real bulk shape; the tail is reported in text rather than hidden.
- 2026-08-26: Extended burst-raster gallery (008): 6 continuous examples
  spanning the burstiness range (0.02-0.59, quantile-matched), 5 whisker +
  5 auditory sensory-evoked examples each including both signs of
  burst_index (not just the largest-positive ones shown in the intermediate
  checkpoint). All 6 continuous examples happened to land on `mua` units
  (quantile-nearest selection on this small 4366-unit subset) -- noted as a
  subset-size limitation, not a selection bug.
- 2026-08-27: Full location/proportion/distribution battery (009, per
  ssl_cohort_comparison_stats.md's "three questions" framework) run on the
  full population (no rerun needed, reused existing parquets). Consolidated
  with the earlier location/PERMANOVA results (004):
  - **Expert arm, burst_index_whisker**: significant on all four fronts
    (global location, PERMANOVA, proportion, mouse-block KS) across all
    4 scope/tier combinations -- the most robust finding in the project.
  - **Learning arm, burst_index_auditory**: significant on PERMANOVA (2/4,
    good-tier only) and mouse-block KS (4/4, both tiers) but NOT on the
    simple global location test or proportion test in any combination --
    a genuine distribution-*shape* difference (not a location shift),
    per the "these are different questions" principle in
    ssl_cohort_comparison_stats.md.
  - continuous_burstiness: null on every test, every combination.
  - Neuron-level (pseudoreplicated) KS was significant almost everywhere
    (tiny p-values) while mouse-block permutation KS was significant only
    in the pattern above -- textbook illustration of why the pseudoreplicated
    version is not trustworthy alone (matches the documented precedent from
    ssl-history-modulation-indices).
- 2026-08-27: Per-area distribution plots (011, descriptive, no new stats)
  for the 6 PERMANOVA-significant combinations, top-6 areas by post-hoc
  p-value each. Effect looks broad/diffuse (small, consistent shift across
  many areas) rather than concentrated in one region -- consistent with why
  no single area survives BH-FDR.
- 2026-08-27: Temporal dynamics preview (010, small subset only, 5 sessions
  after mouse filters -- 2 R+/1 R- learning, 2 R+/0 R- expert). Confirms the
  binned-burstiness method works cleanly (smooth per-session trajectories,
  no artifacts), both session-progression-normalized and absolute-time.
  Suggestive but NOT interpretable pattern: the one R- learning session
  shows rising burstiness across the session vs flat R+ sessions -- n=1 per
  cohort, cannot support any conclusion. Flagged full-population reprocessing
  cost (~2-4h estimated, per-bin computation is heavier than the scalar
  pass) and the circular/block-shift permutation null (per the TCA
  precedent in ssl_analysis_patterns.md) as the correct way to formally
  test a session-trajectory cohort difference, pending user go-ahead.
- 2026-08-27: Full-population temporal dynamics (013/014). Both arms
  computed cleanly (128,306 + 44,598 units; much faster than the earlier
  scalar-metrics baseline would suggest per-unit -- ~4.3ms/unit smoke-tested,
  full run still took ~76min learning + ~15min expert due to the same
  larger-average-spike-count effect seen in 003). Aggregated unit -> session
  -> mouse, plotted R+ vs R- mean+/-SEM trajectories (normalized progression
  + absolute time), both quality tiers, both scopes (16 panels).
  - **Omnibus test** (mouse-block PERMANOVA, 20 bins as joint multivariate
    features, reusing the already-fixed permanova.py): null everywhere
    (p=0.09-0.92 across all 16 arm/scope/tier/binning combinations). This is
    the primary, pre-registered-style conclusion: no cohort difference in
    trajectory shape.
  - **Visual read (corrected by follow-up test)**: all 4 learning-arm panels
    showed a visually striking step-like rise in R+ burstiness starting
    ~60% through the session while R- stayed flat. A targeted post-hoc test
    (per-mouse late-half-minus-early-half delta, R+ vs R-) did NOT confirm
    this in the learning arm (p>0.2 everywhere, tiny median deltas) -- not
    a robust per-mouse effect, correcting the visual impression.
  - **Unexpected finding from the same post-hoc test**: expert arm,
    good_mua tier, both scopes -- both Mann-Whitney and Welch agree
    (p=0.041/0.018 all-mice; p=0.016/0.007 learners-only): R+ burstiness
    stays flat/rises slightly across the session while R- declines. Explicitly
    exploratory: motivated by inspecting the plot after the fact (not
    pre-registered), small n (8-19 mice), and not corrected for the 8
    post-hoc tests run. Needs a held-out confirmatory test before trusting.
- 2026-08-27: New request -- repeat all analyses restricted to data at/after
  each session's first cohort-corrected whisker-trial hit; no
  multiple-testing correction; direct movement-correlation test. Added
  `016_posthit_lib.py` (first-hit detection, cohort-corrected per
  ssl_task_semantics.md) + `017_full_scale_compute_posthit.py` (scalar
  metrics) + `018_statistics_posthit.py` (full battery, raw p-values, BH-FDR
  not used for interpretation -- but the post-hoc-only-if-main-significant
  gate is KEPT, since that's a compute-cost/hierarchical-order choice, not a
  correction; stated explicitly rather than silently deciding) +
  `019_motion_control_posthit.py` + `020`/`021` (temporal, queued).
  Validated on the small subset first (all 5 sessions had qualifying hits,
  4-20% of session excluded) before the full run. Full run: 127,020/129,127
  learning units (1/89 sessions excluded, no qualifying hit), all 44,678
  expert units (0/29 excluded).
- 2026-08-27: **Bug found and fixed** in `019_motion_control_posthit.py`:
  the compressed `ssl_ephys` dataset's `context` trial column is the
  literal string `"nan"` (not a real null) for every row of subjects below
  AB116 -- these subjects never had passive epochs recorded in the raw NWB,
  so `context` was never populated there. Path B's own `process_single_nwb`
  normalizes this (all-"nan" session -> 'active'); the compressed dataset
  does not replicate that normalization. This silently zeroed out every
  `context=='active'` filter for those sessions, producing 12/88 "no
  qualifying hit" learning sessions via the compressed dataset vs only
  1/89 via Path B -- traced directly (AB080: 347 rows, all `context=="nan"`,
  113 real whisker trials). Fixed by replicating Path B's normalization
  logic before filtering; reconciled cleanly afterward (0/88 discrepant).
  Also fixed a `pandas.Series.map()`-with-None-values bug (produces an
  object-dtype column that crashes newer scipy's `spearmanr` -- a parquet
  round-trip was silently masking it on reload, which is why a manual
  replay of the saved data didn't reproduce the crash) -- now cast
  explicitly via `pd.to_numeric(..., errors="coerce")`.
- 2026-08-27: Motion control (post-hit) result: **no significant
  correlation between burstiness and movement/licking** in any of the 6
  testable arm/tier/variable combinations (p=0.09-0.99) -- same conclusion
  as the original (pre-restriction) motion control analysis. Expert-arm
  orofacial motion energy has zero usable sessions (DLC tracking coverage
  gap, same limitation as before).
- 2026-08-27: Statistics battery (post-hit, no BH-FDR) completed. Both
  previously-found effects REPLICATE cleanly under the post-first-hit
  restriction, same direction/pattern as the unrestricted analysis:
  - Expert arm burst_index_whisker: significant on all 4 fronts (global,
    proportion, mouse-block KS, PERMANOVA), all 4 scope/tier combos. Raw
    post-hoc: `MO-wM1` nominally significant (p=0.011, 0.026) in BOTH
    all-mice and learners-only scopes -- the most area-consistent lead in
    the whole project.
  - Learning arm burst_index_auditory: significant on distribution/PERMANOVA
    (not location/proportion) in 3-4/4 combos, matching the original
    "shape not shift" characterization. Raw post-hoc areas: DMS and TH
    recur across multiple combos (not perfectly consistent across all 4).
  - continuous_burstiness: null everywhere, still.
  - Explicit caveat given no correction was applied: at alpha=0.05 across
    ~22-45 areas per combo, 1-2 nominally "significant" areas are expected
    by chance alone under a true null -- these are leads, not confirmed
    localizations.
