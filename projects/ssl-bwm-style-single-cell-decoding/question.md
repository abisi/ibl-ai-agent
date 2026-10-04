# Question

## Original question (from user, 2026-08-18)
Perform the same analyses as the BWM paper on the SSL KS4 dataset, adapting
for SSL's different trial types, using the same statistical framework for
significance.

## Refined scope (clarified via AskUserQuestion, 2026-08-18)
"BWM paper analyses" narrowed to **both**:
1. The BWM paper's flagship **single-cell statistics** pipeline
   (`brainwidemap/single_cell_stats/` in `int-brain-lab/paper-brain-wide-map`):
   per-neuron, per-region significance tests of firing-rate modulation across
   task-aligned windows.
2. The BWM paper's **population decoding** pipeline
   (`brainwidemap/decoding/`): per-region cross-validated decoding of task
   variables with a pseudo-session null.

Verified directly from the upstream repo source (fetched 2026-08-18), not
paraphrased — see the plan file
`C:\Users\bisi\.claude\plans\happy-stargazing-hummingbird.md` for the full
BWM-framework writeup and the SSL adaptation table.

## Locked design decisions
- **Block analog**: SSL has no `probabilityLeft` block. Cohort-corrected
  **prior-trial outcome (t-1 hit vs non-hit)** stands in for it, both as the
  4th test's own factor and as the stratification/drift-control layer
  (`bx`/`by` block index) in the other 3 tests.
- **Multiple comparisons**: add **Benjamini-Hochberg FDR per test**, report
  both raw and FDR-corrected significance (upstream BWM example scripts
  report raw p-values only).
- **Decoding null**: **imposter-session resampling**, not a simpler
  circular-shift null.
- **Outcome-test window** (resolved 2026-08-18 by inspecting `trials.parquet`
  on session `AB080_20230622_152205`, see `exploratory-analyses/000_*.py`):
  `trials.parquet` carries far more columns than
  `ssl-load/references/ssl_dataset_schema.md` documents --
  `response_window_start_time`/`response_window_stop_time` (fixed
  `start_time + [0.15s, 1.15s]` across all trial types in the inspected
  session), `lick_time` (median ~0.37s post `start_time` for `lick_flag==1`
  trials; absent/NaN for no-lick trials), `stim_onset`,
  `whisker_stim_time`/`auditory_stim_time` (both equal to `start_time`,
  confirming `start_time` is the true modality-specific stimulus onset).
  Because a miss trial has no lick event, there is no single per-trial event
  usable as a common outcome/"feedback" anchor for both hit and miss trials
  (unlike BWM's `feedback_times`, which exists regardless of trial outcome).
  **Resolution**: anchor the outcome test at a **fixed window relative to
  `start_time`, `[0.15s, 0.45s]`** -- inside the response window, wide enough
  to span the median lick latency, applied identically to hit and miss
  trials, and deliberately *not* lick-locked so it stays distinct from the
  Response test (lick vs no-lick). This is a flagged deviation from BWM's
  literal `feedback_times` anchor, forced by the fact that SSL miss trials
  have no reward-delivery event at all.
  **Flag for skill maintenance** (not actioned in this project): the
  `ssl_dataset_schema.md` trials-column list is incomplete relative to the
  real `ssl_ephys` `trials.parquet` -- worth a follow-up skill update outside
  this project's scope.

## Locked confirmatory-run parameters (2026-08-19)
- **Evoked window** (Modality test): `[5ms, 35ms]` post `start_time`.
- **Baseline window** (Response, Prior-outcome tests): `[-200ms, -10ms]` pre `start_time`.
- **Outcome/feedback test**: dropped (see below) -- only Modality, Response,
  Prior-outcome remain.
- **`nShuf`**: 1000 (down from BWM's 3000; full-cohort timing estimate at
  this setting -- see `TODO.md` -- is ~3.5h for all 3 tests, so no
  vectorization/parallelization needed at this scale).
- **Area grouping**: `area_acronym_custom` from
  `allen_utils.process_allen_labels(subdivide_areas=True)`, **reused
  directly from the sibling `ssl-whisker-auditory-cohort-modulation`
  project's already-built** `reports/ssl_analysis/derived/unit_area_labels.parquet`
  (`session_id`, `cluster_id`, `mouse_id`, `probe_name`, `target_region`,
  `area_acronym_custom`, `area_group`, QC labels) rather than re-running the
  external `allen_utils` pipeline -- verified populated (169,944 unit rows,
  e.g. `DMS`, `SSp-bfd`, `MO-wM1`, ... top categories).

## Adapted single-cell test definitions (final, pending exploratory sanity check)
All four use the cohort-corrected hit/non-hit outcome
(`ssl_task_semantics.md`), run separately per day-stage (`day==0` vs
`day>0`, `session_id` as the unit for `day>0`), and apply the mandatory
`-1ms/+4ms` whisker-trial dead-zone exclusion before any window touches
whisker-trial spikes.

| Test | Align / window | Tested factor | Stratified by (4 strata) | Shuffle |
|---|---|---|---|---|
| Modality | `start_time` + `[5ms, 35ms]` | whisker vs auditory | response (lick/no-lick) x t-1 rewarded | t-1-rewarded-run-aware |
| Response | `start_time` + `[-200ms, -10ms]` | lick vs no-lick | modality x t-1 rewarded | t-1-rewarded-run-aware |
| Prior-outcome (block analog) | `start_time` + `[-200ms, -10ms]` | t-1 rewarded vs not | modality x response | plain (t-1 rewarded is the tested factor) |

**Outcome/feedback test dropped** (user decision, 2026-08-19): the
cohort-corrected reward status of a trial is a deterministic function of
`(trial_type, lick_flag)` in this task (whisker: flips by cohort; auditory:
always `lick_flag`) -- so "rewarded vs non-rewarded, stratified by response"
has zero within-stratum variance to test, unlike BWM's feedback test where
reward also depends on stimulus side matching choice. All 3 remaining tests
restrict candidate rows to `whisker_trial`/`auditory_trial` only (no
`no_stim_trial` rows, which have no modality label).

All windows/`nShuf` above are now locked for the confirmatory run (see
"Locked confirmatory-run parameters" above); validated non-degenerate on 2
real sessions in the exploratory small-scale checks
(`exploratory-analyses/002-004_*_small_scale.py`).

## Decoding
Two binary targets -- modality, response (`outcome` dropped, same
determinism reason as the single-cell test) -- **session-level only for
now** (per-region decoding deferred as a follow-up), L1-regularized logistic
regression, BWM's alpha grid inverted to sklearn's `C`, nested CV,
imposter-session null built by resampling other real SSL sessions'
target-factor sequences (same day-stage bucket).

**Locked decoding parameters (2026-08-19, user decision on cost)**:
`n_runs=5`, `n_pseudo=50` (down from BWM's 10/200) -- verified pipeline on
`AB080_20230622_152205`/`modality`: observed balanced accuracy 0.897 vs.
imposter-null mean 0.504 (std 0.026, `n_pseudo=20`, `n_runs=3` in that
smaller check) -- see `TODO.md` for the full small-scale write-up.
**Full-cohort confirmatory decoding run deferred at user request** pending
review of the small-scale results (both single-cell and decoding) -- not
yet launched.

## Exploration / confirmation split
Standard split per `exploration-confirmation/SKILL.md` -- not skipped. Small-
scale sanity checks and diagnostics first (1-2 sessions), reviewed with the
user, before any full-cohort confirmatory run.

## QC scope widened to good+mua (2026-08-19, user decision)
All single-cell and decoding runs now use `bc_label in {'good', 'mua'}`
(~139,275 units total across the ephys cohort) instead of `bc_label=='good'`
only (~16,426 units, ~8.5x fewer) -- `'non-soma'` units remain excluded.
`MIN_UNITS_PER_AREA` stays at **5**, unchanged. The prior good-only
confirmatory single-cell run and its area-summary figures/table were
preserved with a `_good_only` suffix rather than overwritten (see
`confirmatory-analyses/*_good_only.*`) for reference/comparison. All three
confirmatory scripts (`001`, `002`, `004`) updated to take a `qc_values`
tuple; `scripts/ssl_bwm_single_cell.py`'s `run_full_test`/`run_small_scale_test`
now accept `qc_values` (plural, `.isin()`) instead of a single `qc_value`.
Cost impact: single-session smoke test at good+mua took >180s (vs 11-36s at
good-only) -- full-cohort reruns will take noticeably longer than the
good-only runs; real estimate pending the smoke-test result before the full
jobs are relaunched.

## Next round (requested 2026-08-19, deferred until current decoding jobs finish)
User wants to replace/extend the current 3-test single-cell battery, add a
trajectory/manifold analysis pillar (a third BWM pillar alongside single-cell
stats and decoding), and change how ALL THREE pillars' results are reported.
Explicitly told to hold implementation until the currently-running decoding
finishes -- this section is a faithful record of the request plus my own
annotations of design questions to resolve before building, not a locked
design yet.

### Requested single-cell tests (replaces the current Modality/Response/Prior-outcome battery)
1. **Whisker responsiveness** (evoked vs baseline, whisker trials only).
2. **Auditory responsiveness** (evoked vs baseline, auditory trials only).
3. **Choice encoding**, window before **jaw movement onset** (not before
   `start_time` as the current Response test does).
4. **Hit/miss encoding for whisker trials**, window after `start_time`.
5. **Lick vs no-lick encoding**, window around `lick_time`.
6. **Reward-history effects in baseline** -- this is the existing
   Prior-outcome test; keep as-is.

Design questions to resolve before building (not yet answered):
- **1-2 (responsiveness)**: these are **paired** (same trials, pre vs post),
  unlike the current unpaired condition-combined shuffle test built for
  Modality/Response/Prior-outcome -- needs a paired-shuffle (sign-flip) test,
  a different primitive than `ssl_bwm_stats_util`'s current unpaired
  `TwoNmannWhitneyUshuf`/`Time_TwoNmannWhitneyUshuf` port.
- **3 (choice/jaw onset)**: jaw movement onset is not in `ssl_ephys`/
  `trials.parquet` -- would need `ssl_behavior`'s jaw keypoint tracking
  (`ssl-load/references/ssl_dataset_schema.md`'s Behavior tables section) to
  detect onset per trial, a data source this project hasn't touched yet.
  Need to define an onset-detection rule (e.g. velocity/displacement
  threshold on the jaw keypoint) before this test can run.
- **4 (hit/miss for whisker)**: this reintroduces the dropped Outcome test,
  but **must not stratify by response** this time -- cohort-corrected
  hit/miss is only degenerate-with-response when response is also a
  stratifying dimension (that was the actual bug in the original design,
  not hit/miss itself). Tested directly (not conditioned on response), hit
  vs miss is a legitimate within-session contrast even though it mixes
  physically different actions across R+ (hit=lick) and R- (hit=no-lick)
  cohorts under one label -- worth being explicit about that framing in the
  writeup. Stratify by something other than response (e.g. t-1 rewarded
  only, or day-stage only).
- **5 (lick/no-lick around `lick_time`)**: `lick_time` only exists for
  `lick_flag==1` trials (see the Locked design decisions section above) --
  unclear yet how a "no-lick" trial gets a comparable anchor for a
  `lick_time`-centered window. Needs clarification before building; a fixed
  window relative to `start_time` (like the dropped Outcome test used) is
  the fallback if no natural anchor exists for no-lick trials.

### Trajectory / manifold analysis (new pillar)
User wants "trajectory analysis like in the BWM paper" added. Source
identified (fetched 2026-08-19): `brainwidemap/manifold/` in
`paper-brain-wide-map` -- `state_space_bwm.py` computes PETHs per
task-variable split across insertions, `get_all_d_vars(split)` bins/saves
PETHs plus a **grouped distance metric** between condition-averaged
population trajectories, `d_var_stacked(split)` aggregates across insertions
and computes p-values, `plot_all()` produces the figures. Upstream cost is
~3h/split for PETH computation alone (per-split, on their cluster) -- SSL
adaptation, exact distance metric, and null/significance procedure not yet
designed; needs its own research pass (read `state_space_bwm.py` itself, not
just the README) before scoping, same way single-cell/decoding were grounded
from actual source rather than paraphrase.

### Results presentation (applies to single-cell, decoding, AND trajectory)
BWM-paper-style summary table/heatmap: **rows = analysis level** (i.e. one
row per task-variable test/pillar), **columns = area labels** (both
parcellations, per earlier work), **cell values = effect/significance**,
with a **different colormap per row/variable**. This replaces the current
per-test scatter-style area-summary figures (`003_*`, `005_*`) as the
primary reporting format once the new test battery + trajectory pillar are
built -- the existing scatter figures aren't necessarily thrown away, but
this grid/heatmap becomes the headline figure, matching the actual BWM paper
figure style.

## Relationship to other in-progress work
The uncommitted `scripts/ssl_*.py` files and `reports/ssl_analysis/` outputs
in this repo belong to the separate, already-scoped
`ssl-whisker-auditory-cohort-modulation` project (KS4 rerun of the passive
whisker/auditory cohort-modulation LMM analysis) -- unrelated to this
project, not modified or depended on here except possibly reusing generic
prep utilities (area labels, unit coverage) after inspection.
