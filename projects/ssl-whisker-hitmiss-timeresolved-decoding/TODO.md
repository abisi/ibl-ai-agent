# TODO

- [X] Design summarized and reviewed with user via AskUserQuestion
      (2026-09-10): bins=10ms disjoint no smoothing, significance=shuffle+
      cluster-permutation, rollout=exploratory pass first, area
      scheme=both. See `question.md`.
- [X] Checked mandatory mouse-inclusion filters against the sibling
      project's session list: 119 has-ephys whisker-training sessions ->
      106 usable mice after `exclude==0`/`exclude_ephys==0`/`R+proba`-drop
      (65 R+, 41 R-; by stage: learning 51 R+/38 R-, expert 15 R+/14 R-).
      Recorded as a gap vs. the sibling project's own (unfiltered) 119.
- [ ] Write `scripts/ssl_timeresolved_decoding.py`: new shared library --
      dead-zone-offset 10ms bin grid, session-half split, per-area
      population matrix (both parcellation schemes) from
      `unit_area_labels.parquet`, fixed-C-per-bin decoding loop, label-
      shuffle null, cluster-based permutation across time, data-sufficiency
      gates.
- [ ] `exploratory-analyses/000_smoke_test.py`: run the full pipeline on
      2 sessions (1 learning R+, 1 expert R-, picked 2026-09-10:
      `AB080_20230622_152205` learning/R+, `MH022_20250311_140628`
      expert/R-), 1-2 areas each with sufficient units, entire-dataset
      scope only, both parcellation schemes. Goals: verify no bugs, check
      class balance after the half-split, get real per-bin/per-session
      timing, sanity-check the accuracy-vs-time curve shape and the
      dead-zone gap.
- [ ] Review smoke-test results and timing with user before scoping/
      launching any full sweep (all sessions x areas x cohort x stage x
      half x scope).
- [X] Smoke test run (2026-09-10), `exploratory-analyses/000_smoke_test.py`,
      outputs `000_smoke_timings.csv`/`000_smoke_results.json`:
      - **Pipeline verified end-to-end**, dead-zone bin correctly dropped
        (grid gap at `[-5ms,+5ms)`), data-sufficiency gate correctly caught
        a degenerate cell (`MH022` expert, second half: hit=43/miss=5,
        skipped).
      - **Substantive observation** (not yet a finding, single session):
        `AB080` (learning, R+) whisker hit rate drops from 0.66 (first
        half, n=56) to 0.21 (second half, n=57) within one session --
        worth a dedicated look later, unrelated to this pipeline's own
        correctness.
      - Decoding worked: `AB080` first/second half both cluster-significant
        at the smoke-test's `n_shuf=30` (`cluster_p=0.032`, the floor at
        this n_shuf -- **not a real p-value yet**, just confirms the
        pipeline detects a real effect). `MH022` (expert, R-) first half
        was borderline/non-significant at both parcellations
        (`area_group` p=0.065, `area_acronym_custom` p=1.0).
      - **Found and fixed**: this environment's scikit-learn spams a
        FutureWarning+UserWarning pair on every `penalty="l1"` fit --
        300k of ~300k smoke-test log lines were this single pair, no
        informational content. Suppressed in `ssl_timeresolved_decoding.py`
        (cosmetic only, does not change fitted models).
      - **Real per-condition timing** (session, area, half):
        `t_matrices` 4.2-12.6s, `t_c` (one-time fixed-C selection) 0.06-0.9s,
        `t_curve` per repeat 0.56-1.3s, `t_null` per shuffle draw 0.56-1.3s
        across 78 bins.
      - **Cost problem found**: the per-session label-shuffle null (the
        dominant cost, ~1s/shuffle x n_shuf) needs `n_shuf` in the
        hundreds for a real (non-floor) p-value, and this project's scope
        is ~118 sessions x ~2 halves x (4 `area_group` + more
        `area_acronym_custom` areas/session) -- extrapolated full sweep at
        a meaningful `n_shuf` is many hundreds of hours, not tractable as
        designed. **Proposed pivot** (not yet approved): drop the
        per-session null entirely; compute only the real decode curve per
        (session, area, half, day_stage) cheaply, and move all
        significance testing (above-chance decoding, R+/R- comparison) to
        the **group level** via mouse-block permutation across sessions --
        matching this dataset's established cohort-comparison convention
        rather than thousands of independent per-session tests needing
        their own FDR correction. Cuts estimated full-sweep cost roughly
        3-5x plus removes the resolution-ceiling risk the sibling
        project's `nShuf=1000` single-cell run already hit once. Awaiting
        user decision before proceeding.
- [X] User decision (2026-09-10): adopt the group-level significance pivot,
      `MIN_TRIALS_PER_CLASS` lowered 8 -> 5. Implemented in
      `scripts/ssl_timeresolved_decoding.py`: `session_real_and_shuffled_curves`
      (real curve + exactly one label-permutation surrogate curve per
      session -- 2x cost of a real-only decode, not n_shuf x),
      `group_mouseblock_permutation_null` (mouse-block sign-flip
      permutation: for each of `n_perm` draws, each *mouse* contributing
      sessions to the group independently picks real-vs-surrogate for
      *all* its sessions in that group, then the group mean curve is
      recomputed -- reuses the already-generic `cluster_permutation_test`
      unchanged), `rplus_rminus_group_test` (Mann-Whitney + Welch pair at
      session grain, plus a mouse-block permutation of the group-mean
      difference, since `reward_group` is mouse-level even at session
      grain -- `ssl_stats_unit_of_analysis.md`).
- [X] Applied the shard-caching optimization: `load_session_unit_spikes`
      loads a session's shard once; `bin_population_matrices` now takes
      that pre-loaded dict instead of reloading per (area, half) call.
      `000_smoke_test.py` patched to the new signature (not rerun --
      already-logged results/timings predate this change, noted inline).
- [X] Synthetic self-test of the group-level machinery (not saved as a
      script, run inline 2026-09-10): confirmed `cluster_permutation_test`
      + `group_mouseblock_permutation_null` detects a planted above-chance
      bump (p=0.001) and returns null on no-signal data (p=0.32); confirmed
      `rplus_rminus_group_test` detects a planted R+/R- difference
      (mouse-block perm p=0.001) and returns null with none (p=0.53).
- [X] `exploratory-analyses/001_group_level_demo.py`: real-data run, fixed
      area (`area_group=='Motor and frontal areas'`, the most
      session-common coarse area), learning-stage only, 3 R+ + 3 R-
      sessions, both halves, `n_repeats=5`, `n_perm=2000`. Outputs
      `001_group_demo_timings.csv`, `001_group_demo_curves.npz`.
      **Results** (demo-scale, not a claim): all 4 (cohort, half)
      above-chance cluster tests returned p=1.0, and the R+/R- comparison
      was non-significant both halves -- **expected at n=2-3
      sessions/mice per cell** (mouse-block permutation with 3 mice has
      only 8 possible sign-flip patterns, many collapsing by symmetry; not
      informative about whether real decoding exists). One session-half
      (`AB086` second half) correctly skipped by the data-sufficiency gate
      (0 hits / 76 misses -- this mouse's hit rate apparently collapsed to
      zero mid-session, a substantive observation worth a separate look
      later, not pursued now).
      **Real, higher-than-expected timing**: 16-348s per session (both
      halves combined), mean 161.6s -- driven by this area's large unit
      counts (866-1410 units/session; "Motor and frontal areas" is one of
      the largest coarse-area buckets). This revises the smoke test's
      earlier (much smaller-population) extrapolation upward -- cost now
      clearly scales with population size, not just bin count, so a
      full-sweep estimate needs to account for the wide unit-count range
      across areas/sessions, not a single average. **Not yet
      extrapolated to a full-sweep ETA** -- next step is a small
      checkpointed background-job pilot (e.g. one full day-stage) to get
      that number honestly rather than hand-extrapolating further from 6
      sessions of one large area.
- [ ] Decide scope/launch a checkpointed background pilot run (proposed:
      one day_stage, both cohorts, both parcellation schemes, all
      sessions, both halves) to get a real full-scope ETA before deciding
      on the full confirmatory sweep. Not started -- awaiting user
      direction on scope.
- [X] All 4 planned variants swept (2026-09-11), sliding-window (50ms/10ms
      stride) binning + StandardScaler-wrapped classifier, entire-dataset
      population scope (learners-only computed post-hoc as a session
      filter, no separate sweep needed):
      - `start_time_learning`: 89 sessions, 686.8min (one very slow final
        session dominated the tail).
      - `start_time_expert`: 29 sessions, 31.2min. No area/half cell
        reached the >=5-sessions-per-cohort gate (too few expert sessions
        split two ways) -- a real sample-size limit, not a bug.
      - `lick_time_learning`: 89 sessions, 160.8min.
      - `lick_time_expert`: 29 sessions, 42.1min.
- [X] `012_master_group_level_tests.py` run on all 4 datasets x 2 area
      schemes x 2 population scopes (323 rows,
      `012_master_test_results.csv`), including the new 0-50ms-post-stim
      window test (start_time-aligned variants only). Key finding: with
      the corrected (sliding-window + standardized) pipeline, the
      `start_time_learning` R+/R- curve-diff effect is mostly
      **second-half-of-session only** (not first half), and the
      **0-50ms post-stim window test is not significant in any area** --
      the effect, where present, emerges later than 50ms post-stimulus.
- [X] `013_master_figures.py` run on final complete data -- per-dataset
      curve figures (top-6-by-curve-diff-p areas) and stats heatmaps,
      both area schemes, both population scopes.
- [X] Wrote `report.html`, published as a private Artifact
      (https://claude.ai/code/artifact/6b52eaf2-230d-4602-bef3-dace85a98e12):
      design/pipeline history, both decoders' results with figures,
      expert-stage limitation, robustness checks (fine parcellation,
      learners-only scope, classifier comparison table), caveats
      (framing, RT confound, sample size, multiple comparisons),
      reproducibility pointers. Not shared publicly (SSL data is
      private/unpublished per policy).

## Rebuilt sweep (causal bins, wider dead zone, `whole_brain` scheme) -- 2026-09-11 onward, not yet reflected above
`024_master_sweep.py` supersedes the `002`/`008`/`015`/`021` sliding-bin sweeps
(see its own docstring for the full design-decision log: causal 50ms/5ms-stride
bins, `-10ms/+5ms` dead zone, `whole_brain` area scheme added, cross-gen for
half/perfstate, second null control (`linear_shift_null_curves`) added
2026-09-13 for hitmiss/perfstate). `report.html` above predates this rebuild
and needs a refresh once it's complete -- not done yet.

**Interruption (2026-09-13)**: the three in-progress `024_master_sweep.py`
runs at `area_group` scale (`modality_stim_area_group`, `hitmiss_stim_area_group`,
`perfstate_stim_area_group`) and the from-scratch, non-checkpointed
`033_pseudopopulation_hitmiss_wholebrain.py` all died with no Python process
left running and no log progress for 1.5-3h -- consistent with the machine
sleeping/losing power, not a code bug. **Resumed** same evening via the
checkpointed per-session resume path (`load_done_sessions`/`append_rows`):
modality was at 84/89 sessions, hitmiss at 45/89, perfstate at 78/87 --
all three correctly picked up from those exact checkpoints. `033` has no
checkpointing and was restarted from scratch (it had died before finishing
even the first cohort's session-pooling step, so nothing was lost).

**Found (2026-09-13, while checking on the resume)**: the shift-null control
(`linear_shift_null_curves`, added to `024_master_sweep.py` 2026-09-13) is
missing from data computed before that addition -- confirmed
`shift_null_curves` isn't even a column in the completed `whole_brain`
`hitmiss_stim`/`perfstate_stim` results (both finished 2026-09-12, predate
the control), and the `area_group` `hitmiss_stim`/`perfstate_stim` partial
checkpoints (all sessions completed before today's crash) have the same
gap -- a plain resume only appends new sessions and can't retroactively add
a column to already-checkpointed ones, so it would have produced a file
with the control on some sessions and not others. `modality` never computes
this control (by design, no drift confound), so `modality_*_whole_brain`/
`modality_lick_area_group` are unaffected.

**Fix, two stages (user-approved 2026-09-13)**:
1. Old `whole_brain` `hitmiss_stim`/`perfstate_stim` results backed up to
   `*.bak-preshiftnull-20260913` and both re-launched from scratch (fresh,
   not resumed) so every session gets the control -- running now.
2. Stopped the in-progress `area_group` `hitmiss_stim`/`perfstate_stim`
   resumes (bx60f0q2n's `modality_stim_area_group` sibling was left running
   -- unaffected, no control to miss) since their output would need the
   same from-scratch redo regardless of whether the resume finished, so
   letting them run to completion first would have wasted the compute.
   User will check results before midnight; **if no further instruction by
   2026-09-14 00:00**, a backgrounded scheduler (already armed) backs up
   the current `area_group` `hitmiss_stim`/`perfstate_stim` partials the
   same way and relaunches both from scratch with the control.

**Figures reorganized per area level (2026-09-13, user request)**: `025`,
`027`, `028`, `029`, `030` (whole-brain-only scripts) now write to
`figures/whole_brain/`; the tag-driven `032`/`034` now route to
`figures/{whole_brain,area_group,area_acronym_custom}/` based on the tag
itself (`fig_dir()`/`fig_dir(tag)` helper, same convention in both). All
existing current-iteration (`024`+) PNGs moved into the new folders;
`031` (pure behavioral, no area level) stays in `exploratory-analyses/`
root. Also fixed while doing this: `get_area_color_map`'s `allen_utils`
import (from the `M:` network drive) was an unguarded crash when that
drive isn't mounted -- it killed `032`/`034` partway through a tag,
silently dropping every figure after the first. Now falls back to grey
areas / alphabetical order with a warning instead of crashing (`032` and
`034` both patched). Regenerated `032` figures for all 8 currently-live
tags against latest data; `034` and the whole-brain-only `025`-`030`
scripts still need a pass once the in-progress reruns actually finish
(most of tonight's whole_brain data is only 1-2/89 sessions in right now).

**Area colors blocked on `M:` drive (2026-09-13, user request "must be the
color of allen_utils")**: `get_area_color_map` (`032`, `034`) needs
`allen_utils` from `M:\analysis\Axel_Bisi\Github\ephys_utilities\...` --
confirmed via both Bash and PowerShell that `M:` isn't mounted anywhere in
this session (`Test-Path` false, `Get-PSDrive`/`net use` show no mapping),
and no local copy exists in the repo, venv, or common local dirs. This is
an environment/credentials issue only the user can resolve (map the drive),
not something fixable from here. Current `area_group` area-overlay (`032`)
and quantification-grid (`034`) figures use the grey fallback in the
meantime -- **re-run both scripts for every live tag once `M:` is mounted**
to get the real colors (code already picks them up automatically, no
further code change needed).

**`034` stats added (2026-09-13, user request)**: comparison-figure titles
now report a two-way ANOVA (cohort x area, interaction) main effect of
cohort (`cohort_anova_and_posthoc`), paired with a per-area Mann-Whitney U
post-hoc (no correction across areas, matching `026`'s convention) shown as
an asterisk (p<0.05) above the area's point -- the usual parametric +
non-parametric pairing (`ssl_rplus_rminus_test_pair.md`), applied here at
the exploratory area-grid level (per-session-per-area independence
assumption, not mouse-block permutation like the confirmatory group tests).
Legends removed from both `plot_cohort_grid` and `plot_comparison_grid`
(area identity already carried by color/x-tick-label; cohort by marker
shape only, no in-figure key). Re-ran `034` for all 4 live `area_group`
tags with this update.

**Post-hoc FDR (2026-09-13, user request)**: the per-area Mann-Whitney U
post-hoc in `cohort_anova_and_posthoc` is now Benjamini-Hochberg-corrected
across the areas tested in that one subplot (`statsmodels.stats.multitest.
multipletests`) -- FDR applies only to the post-ANOVA post-hoc, not to the
ANOVA itself (one omnibus test per subplot, nothing to correct across).
Asterisk threshold is now q<0.05. Re-ran all 4 live `area_group` tags.

**Figures further organized by day-stage (2026-09-13, user request: run
the same whole_brain computations for expert sessions, keep results
separated clearly)**: `fig_dir()` in `032`/`034` now nests
`figures/<area_level>/<learning|expert>/`, read off the tag's `_expert`
suffix (`024_master_sweep.py`'s own `_suffix` convention) -- e.g.
`hitmiss_stim_expert_whole_brain` -> `figures/whole_brain/expert/`.
Existing (all learning-stage) figures migrated into the new `learning/`
subfolders. **Expert-stage whole_brain hitmiss/perfstate reruns queued**
(with the shift-null control, matching the learning-stage reruns) into the
same scheduler as the `area_group` redo, positioned right after the
current jobs finish and before the (still midnight-or-later-gated)
`area_group` reruns -- queue order: current jobs -> expert whole_brain
hitmiss -> expert whole_brain perfstate -> (wait for midnight if not yet
passed) -> area_group hitmiss rerun -> area_group perfstate rerun. All
strictly sequential, nothing added concurrently to what's already running.
Modality expert-stage was NOT included (out of scope of this request --
ask if wanted). `025`-`030` (whole-brain-only scripts) still need their own
per-tag day-stage split before they can be pointed at expert tags -- not
done yet, deferred until expert data exists to plot.

**Combined-cohort grid variant (2026-09-13, user request), performance-
state decoding only**: `034`'s `plot_cohort_grid` generalized to
`plot_group_grid(sub_group, ..., label, ...)` (takes a pre-filtered
dataframe + label instead of a cohort name) so the same code serves the
per-cohort grids AND a new pooled-cohort `034_{tag}_grid_combined.png`
(both R+ and R- sessions treated as one group) -- produced only for
`decode_target=='perfstate'` tags (decoding perf-state itself, not
perfstate as a condition split of another target).

**RT-histogram overlay, exploratory preview only (2026-09-13, user
request, "just to see")**: new `exploratory-analyses/035_test_rt_
histogram_shared_axis.py` -- one-off, not wired into any pipeline. Overlays
a hit-trial reaction-time histogram (`lick_time - start_time`) on the same
axes as a stim-aligned decode curve (`hitmiss_stim_whole_brain`, whole
session, R+ & R- aggregated, using the pre-shift-null backup since it's
already complete -- this is a layout preview, not a result), via
`ax.twinx()` scaled so the tallest bar sits just under the 0.5 chance line
on the curve's own axis rather than sharing its scale; bars styled
semi-transparent grey with a thicker, darker grey edge. Visually shows the
curve inflecting (~100-150ms) and plateauing (~350-400ms) well before the
RT distribution's mass (concentrated ~250-450ms) -- consistent with the
report's existing RT-confound caveat. Not yet decided whether/how to
generalize this to other tags/conditions -- user said "just to see."

**RT histogram promoted from preview to the real pipeline (2026-09-13,
user request: "add the histogram for all start_time aligned figures")**:
`032_plot_decode_results.py` now overlays it on every panel of every
stim-aligned figure (`build_rt_cache`/`add_rt_histogram`, same twinx/
grey-transparent/thick-dark-edge styling as `035`) -- always whisker-trial
hit RT regardless of decode_target/condition_type (a stable timing
reference, not tied to what's being decoded). Not applied to lick-aligned
figures or the area-overlay comparison figures (scoped out, see `032`'s
docstring). Regenerated all 8 live tags.

**Ideas 3/4/5 implemented (2026-09-13)**, all exploratory, none wired into
the main sweep:
- **3 -- block-transition profile** (`036_perfstate_block_transition_
  profile.py`, behavioral-only, no new neural compute): pulls trials around
  every within-session low<->high perf_state transition, hit-rate vs. trial
  offset from the boundary. **Result: looks like a step at the boundary in
  both directions, not a gradual ramp** (both cohorts). A full neural
  per-trial-position decode (to check if this holds at the population level
  too) would need a new compute-heavy sweep -- not started, scope/launch on
  request.
- **4 -- behavioral-gap correlation** + **5 -- session-time confound**,
  combined in `037_perfstate_confound_and_behavior_correlations.py`
  (`perfstate_stim_whole_brain` backup, sensory-window decode metric,
  `regress_and_plot` copied verbatim from `030` per
  `ssl_correlation_figure_style.md`): x=`block_fa_rate(high)-block_fa_rate
  (low)` for idea 4 (FA-rate, not hit-rate, specifically to avoid
  circularity with the label-defining metric); x=ROC AUC of normalized
  trial-index alone predicting perf_state for idea 5. **Pooled-cohort
  result**: neither correlation significant (fa_gap r=0.18 p=0.12;
  trivial_auc r=0.06 p=0.57). **Per-cohort result (real finding, worth
  flagging)**: R+ decoding correlates with the FA-rate behavioral gap
  (r=0.48, p=0.0014) but NOT with trial-index-alone AUC (r=-0.33, p=0.031,
  i.e. if anything the opposite direction) -- consistent with genuine
  state-related signal. **R- is the opposite pattern**: no significant
  behavioral-gap correlation, but a strong positive correlation with
  trial-index-alone AUC (r=0.75, p=7.4e-08) -- i.e. in R- sessions, more of
  the "perf-state decoding" tracks trivial session-time separability than
  in R+. Flagged, not yet acted on -- worth deciding whether R- perf-state
  decoding results need this confound controlled for more directly (e.g.
  matching or residualizing trial-index) before being reported as a clean
  finding.

**Behavioral-gap correlation extended to 3 measures (2026-09-13, user
request)**: `037`'s idea-4 panel split into `hit_rate_gap` (raw
P(lick|whisker), high-low), `fa_rate_gap` (unchanged from before),
`delta_lick_prob_gap` (P(lick|whisker)-block_fa_rate gap, matching `030`'s
`perfstate_behavior` exactly) -- idea-5's `trivial_auc` panel kept
alongside, 4 panels total now. **Results (percohort)**:
- `hit_rate_gap`: strongly significant both cohorts but **opposite sign**
  (R+ r=0.58 p=5.1e-05, R- r=-0.65 p=8.2e-06) -- exactly the flagged
  circularity, and it flips sign because `cohort_corrected_rewarded`
  inverts lick_flag for R- (so "high" state = *low* raw lick-rate blocks
  for R-). Not an independent finding, expected by construction.
  **Pooled**: r=0.02, p=0.86 -- the two cohorts' opposite-signed clouds
  cancel out entirely when pooled, underscoring why the percohort split
  matters here.
- `fa_rate_gap`: unchanged from the single-measure version (R+ significant,
  R- not).
- `delta_lick_prob_gap`: **neither cohort significant** (R+ r=-0.16 p=0.30,
  R- r=-0.32 p=0.054 borderline) -- combining hit-rate and FA-rate into one
  discriminability index washes out the FA-only R+ signal rather than
  sharpening it. Worth knowing which behavioral measure is used before
  citing "decoding tracks behavioral distinctiveness" as a finding.
- `trivial_auc` confound result unchanged.

**Pseudopop vs. single-session comparison (2026-09-13/14, user request)**:
`038_pseudopop_vs_single_session_comparison.py` overlays `033`'s pooled
pseudo-population decode curve against the mean of per-session
`hitmiss_stim_whole_brain` curves (backup, complete 89 sessions), same
target/alignment/scheme. **Result: not usable as a real comparison.** Both
cohorts' pseudopop curves are pinned at 1.000 balanced accuracy across the
*entire* time course, including the -200-0ms pre-stimulus baseline where
there is no stimulus information to decode -- confirms the overfitting
signature already visible in `033`'s own log (`peak_acc=1.000` at `C=1e5`,
essentially unregularized, every single draw, both cohorts: total pooled
units in the tens of thousands vs. only 60 pseudo-trials/draw, a severe
n<<p regime, with no label-shuffle null in `033`'s first pass to catch it).
The single-session mean curves look normal (R+ baseline ~0.57 rising to
~0.82; R- baseline ~0.55 rising to ~0.70) and are the only trustworthy
curves in this figure. **Not acted on further** -- `033` would need
stronger regularization (smaller C, or drop it from the grid) and a
label-shuffle null before its numbers mean anything; flagged for the user
to decide whether that's worth doing.

**Modality expert-stage whole_brain queued (2026-09-13, user request,
"trial label shuffle like before")**: both `modality_stim_expert_
whole_brain` and `modality_lick_expert_whole_brain` added to the scheduler
queue, right after the hitmiss/perfstate expert-stage runs, still strictly
sequential and still gated behind the currently-running jobs. No
shift-null needed for modality (never computed for this target by design),
so these are plain fresh launches, no backup-and-redo dance required.

**2026-09-14 00:46 update**: `modality_stim_area_group`'s bottleneck broke
-- the first of its 5 remaining sessions alone consumed the entire 4+ hour
stall (259.5min), the other 4 are proceeding at a normal pace (3/5 done
shortly after). `033` pseudopop finished both cohorts (R- confirmed same
peak_acc=1.000 overfitting pattern as R+). Scheduler still gated on
`modality_stim_area_group`'s last 2 sessions, then `hitmiss_stim_whole_brain`
(18/89) and `perfstate_stim_whole_brain` (51/89).

**2026-09-14 report update + scheduler reorder (user request)**: expert-stage
whole_brain runs (hitmiss/perfstate/modality) moved to the END of the
scheduler queue, after the `area_group` redo rather than before it (was:
current jobs -> expert -> midnight -> area_group; now: current jobs ->
midnight (already passed, no-op) -> area_group redo -> expert). Stopped
and relaunched the scheduler mid-wait with no work lost (still gated on
`hitmiss_stim_whole_brain` at the time).

`report.html`/`report_template.html` rewritten to reflect this rebuild
(024-038) instead of the old sliding-window pipeline it still described --
republished to the existing artifact
(https://claude.ai/code/artifact/6b52eaf2-230d-4602-bef3-dace85a98e12,
v3). Explicitly framed as a mid-rebuild snapshot with a "what's finished,
what's running" status table, citing only complete/final sweeps
(`perfstate_stim_whole_brain` 89/89 with shift-null, `modality_*` all
variants) -- `hitmiss_stim_whole_brain` (mid-rerun) and `area_group`
hitmiss/perfstate (queued full redo) are explicitly marked pending, not
cited. Content covers: the causal-bin/dead-zone/shift-null design pivots,
whole-brain modality+perfstate curves (RT-histogram included), the
area_group modality ANOVA (no cohort effect, contrasted with perf-state),
the full perf-state deep dive (block-transition step-not-ramp, the
FA-rate-gap-vs-trivial-AUC cohort divergence table), and the pseudopop
overfitting caveat. Kept the existing bespoke design system (Fraunces/
Public Sans/IBM Plex Mono, teal accent, fixed R+/R- cohort colors,
warn/callout/timeline/findings-grid components, light+dark tokens) rather
than redesigning -- it already fit the subject well. Added a `.pending`
component (distinct from `.warn`) for "not cited yet, will change" notes,
kept separate from actual caveats. Images live in new `report_assets/`
(7 curated PNGs, not the full 100+-file sweep) and are referenced two ways:
the published Artifact uses the tool's multi-file `files` map directly;
the local standalone `report.html` is rebuilt from `report_template.html`
by a new `build_report.py` (base64-embeds `report_assets/` images) --
run it again after any future template edit before treating `report.html`
as current.

**Pseudopop dropped from the report (2026-09-14, user request: "forget
about the pseudo-population for now, do not include in reports")**:
removed the whole pseudopop section (was section 08) from
`report_template.html`, renumbered the TOC and the two following sections
(caveats 09->08, reproducibility 10->09), dropped its row from the
reproducibility table, removed its image from both the published artifact
(v4, via a `files` null-removal) and the local `report_assets/`, and
rebuilt `report.html`. `038_pseudopop_vs_single_session_comparison.py` and
its output figure still exist on disk (exploratory-analyses/figures/
behavior/) -- only the report stopped citing it, the underlying analysis
and its overfitting-caveat finding are unchanged and still logged above.

**CV estimator discussion + BWM comparison (2026-09-14)**: user proposed
fixing `decode_bin`/`select_fixed_c`'s CV leakage (plain `StratifiedKFold`
on trials, no respect for chronological/block structure -- a real concern
given this project's OWN findings: perf-state's step-wise 5-trial-block
autocorrelation, and trial-index-alone predicting perf_state well in R-
sessions). Grouped-CV (20-trial chronological `StratifiedGroupKFold`)
implementation was scoped (whole_brain, "whole" condition, hitmiss +
perfstate) then **cancelled by the user before any code was written** --
nothing to roll back, no files touched. Separately, user asked for a
comparison against the BWM paper's nested-CV decoding methodology
(pasted verbatim from the paper's methods text) -- found real deltas:
(1) BWM does per-outer-fold nested C selection (not adopted here, user
confirmed keeping single-C-across-all-bins for cost reasons), (2) BWM
scores via pooled/concatenated held-out predictions across one exhaustive
partition, not a mean of separately-scored folds, (3) BWM resamples a
fold assignment missing a class rather than skipping it, (4) BWM excludes
low-count sessions upfront rather than reactively. Flagged that BWM's
"interleaved" trial assignment is just as vulnerable to the
temporal-autocorrelation leakage as our own `StratifiedKFold(shuffle=True)`
-- adopting BWM's procedure verbatim would NOT fix that; it's an
orthogonal axis (estimator rigor vs. temporal leakage).

**Pooled-scoring + resample-on-missing-class implemented (2026-09-14,
user request, "same scoring estimator and class-balance handling" as
BWM)**: new functions added ALONGSIDE the originals in
`scripts/ssl_timeresolved_decoding.py` (nothing overwritten, matching the
same "keep current code as backup" instruction as the cancelled grouped-CV
work) -- `_stratified_kfold_with_resample` (redraws a whole partition,
fresh seed, up to 50 times, until every fold's training portion has both
classes), `select_fixed_c_pooled`, `decode_bin_pooled`, `decode_curve_pooled`,
`label_shuffle_null_curves_pooled`. Single-C-across-bins cost tradeoff
kept unchanged (per user confirmation) -- only the scoring aggregation
(pool-then-score-once vs. mean-of-fold-scores) and the fold-validity
handling (resample vs. skip) changed, same total number of model fits as
before. Motivation specific to this project: `MIN_TRIALS_PER_CLASS` as low
as 5 means individual folds can have 1-2 minority-class held-out trials,
making a single fold's balanced accuracy high-variance; pooling computes
each class's recall from its full held-out count across a whole partition
at once. Verified via synthetic sanity checks (planted signal ~0.85 both
old/new estimators, pure noise ~0.47-0.5 both, 5-vs-55 imbalanced case
runs cleanly under pooled scoring with no crash, resample correctly finds
a valid class-balanced partition without looping indefinitely) -- not yet
run on real session data or wired into any sweep script. Next step, not
yet started: decide a pilot scope (same whole_brain/whole-session/
hitmiss+perfstate scope discussed for the now-cancelled grouped-CV work
would be a natural first target) and compare real-data curves against the
existing `decode_bin`-based results side by side.

**Auditory warm-up block discovered and fixed (2026-09-14, Axel Bisi)**:
many whisker-training sessions begin with a block of auditory-only trials
(no whisker trials) to wake/engage the mouse before whisker trials are
introduced -- distinct from the auditory-*pretraining* stage (a separate
session type already excluded). This block was NOT excluded by any
existing filter (day-stage regex, `context=='active'`) since it's genuinely
part of a valid whisker-training session's active epoch. Directly confirmed
on `AB086_20231015_141742`: 26 consecutive auditory/no_stim trials before
the first whisker trial. This explains the modality-decoding
above-chance-in-pre-stimulus-baseline finding from earlier tonight (the
block inflates early-session trial-type autocorrelation, letting a
classifier exploit trial-to-trial carry-over rather than real anticipatory
coding). New mandatory skill rule + reference file:
`skills/ssl-analyze/references/ssl_auditory_warmup_block.md` (mirrors
`ssl_artifact_dead_zone.md`'s format), cross-referenced from
`skills/ssl-analyze/SKILL.md`'s References/Default analysis policy/Quality
gates. **Fix implemented** in `scripts/ssl_bwm_trial_prep.py`'s
`prep_session` (the shared entry point for every SSL decoding trial-prep
function): drop all active trials before the session's first active
`whisker_trial`, applied right after the existing context filter, before
`rewarded`/`t1_rewarded`/`run_index` are computed. Verified: only affects
targets that keep both trial types together (modality, lick-aligned
modality) -- `prep_hitmiss_trials`/`prep_perfstate_trials_generic` already
filter to whisker-only trials before computing their own chronological
half/block split, so a preceding all-auditory run (zero whisker trials by
definition) could never have entered their computation; **their existing
results do NOT need to be rerun for this reason**.

**Modality redo queued (2026-09-14)**: all 4 learning-stage modality tags
(`modality_stim_whole_brain`, `modality_lick_whole_brain`,
`modality_stim_area_group`, `modality_lick_area_group`) backed up as
`.bak-prewarmupfix-20260914` (parquet + log + bin-edges) and fresh reruns
appended to the END of the scheduler queue (after the already-queued
expert-stage whole_brain runs, which include `modality stim/lick expert
whole_brain` -- those haven't run yet so they'll pick up the fix
automatically, no separate backup/redo needed for them). Full queue order
now: current jobs -> pooled-CV pilot -> `area_group` hitmiss/perfstate
redo -> expert-stage whole_brain (hitmiss/perfstate/modality x2) ->
learning-stage modality redo (4 tags). Scheduler restarted mid-wait with
nothing lost (still gated on `hitmiss_stim_whole_brain` at time of restart).
**All downstream modality figures/stats (032, 034, 025/027/028/029/030's
modality rows, the report's modality-ANOVA section) will need refreshing
once this redo lands** -- current modality figures/numbers throughout this
session predate the fix.

**Null bands added to the pooled-CV comparison (2026-09-14, user request:
"add the mean+-sem of the appropriate shuffle distributions" -- clarified
to mean the already-finished smoke-test results, not the not-yet-run full
pilot)**: added `linear_shift_null_curves_pooled` to
`scripts/ssl_timeresolved_decoding.py` (the pooled-scoring analog of
`linear_shift_null_curves`, mirroring the existing
`label_shuffle_null_curves_pooled`) so both null types this pipeline uses
for hitmiss/perfstate elsewhere are available under the new estimator too.
`039_pooled_cv_pilot_wholebrain_whole.py` now computes all 4 null-curve
sets per session (label-shuffle + linear-shift, under both old and new
estimators) at a reduced `N_SHUF=10`/`NULL_N_REPEATS=2` (diagnostic
reference band, not a significance test) and plots them on the R+/R-/
aggregated panels: thin lines = old estimator, thick = new, dotted =
label-shuffle, dash-dot = linear-shift. Comparing null bands between
estimators is itself informative -- a well-behaved label-shuffle null
should sit at ~0.5 regardless of which CV estimator computed it, so any
systematic old-vs-new divergence there would flag a CV problem independent
of whatever the real curves show. `040_pooled_cv_smoke_test.py` rerun with
this change (same 2 sessions as before, AB080 R+/AB085 R-) -- takes
noticeably longer than the real-curve-only version given 4 additional
null-curve computations per session.

**Pooled-CV pilot built and queued (2026-09-14, user request)**: new
`039_pooled_cv_pilot_wholebrain_whole.py` -- for hitmiss and perfstate,
whole_brain scheme, "whole" condition only, learning stage, computes BOTH
the existing (`decode_bin`/`select_fixed_c`) and new
(`decode_bin_pooled`/`select_fixed_c_pooled`) curves per session from the
same population matrices, checkpointed/resumable per session
(`024_master_sweep.py`'s own pattern), then plots a 4-panel comparison
(R+, R-, aggregated, paired new-minus-old difference) per target. No null
in this pass -- point-estimate comparison only. Smoke-tested on one real
session (AB080, hitmiss): ran cleanly, ~79s single-threaded, notably
picked a very different C between methods for this session (old C=10.0,
new C=0.1) -- worth watching whether that's typical once the full sweep
lands. Not run at full scale yet -- queued in the scheduler (still gated
on `hitmiss_stim_whole_brain`, 72/89 as of 2026-09-14 10:43), positioned
right after the current jobs finish and before the `area_group` redo,
since it's cheap (~15min/target at 8 workers, uncontended) and the
"no piling on active jobs" rule stays in force.

**2026-09-14 03:01/03:48 updates**: `modality_stim_area_group` finished
completely (89/89) -- figures refreshed with final 88-session data.
`perfstate_stim_whole_brain` finished completely (89/89) WITH the
shift-null control -- figures refreshed, and `037`'s confound/behavior
correlation script switched from the pre-shift-null backup to this live
file (results unchanged in sign/significance pattern; R- session-time
confound if anything slightly stronger: r=0.81, p=9.4e-10, was r=0.75,
p=7.4e-08). Scheduler now gated solely on `hitmiss_stim_whole_brain`
(27/89 as of 03:48, ~500min elapsed, still the long pole -- naive ETA
remains unreliable given the early outlier session).

## Curve-based perf-state definition -- ADOPTED AS PIPELINE DEFAULT 2026-09-16

**Status update, 2026-09-16 (supersedes "NOT YET IMPLEMENTED" below, kept
for the investigation detail)**: implemented, extensively tested, and
adopted as the production default. User: "Use new curve-based state
definition for next runs and adopt in pipeline" (2026-09-16), after the
investigation below and three rounds of validation in
`exploratory-analyses/`:
- `043_perfstate_curve_labels_test.py`: 5-session then 20-session
  (decodable-in-all-4-cells subset) old-vs-new hitmiss x perfstate decode
  comparison, later split by R+/R- per user follow-up -- found the pooled
  aggregate was masking a cohort-specific story (old def's signal lives in
  R+ only; new def's lives in R- only, opposite sign).
- `044_perfstate_curve_diagnostic.py`: per-session `p_mean`/`p_chance`/
  state-allocation figure, visual sanity check.
- `045_perfstate_curve_label_sweep.py`: full-scale (all candidate
  sessions), decode-free label-agreement + class-balance sweep. Coverage
  climbed from 71/89 -> 75/89 (found+fixed the `perf!=6` "association"-
  outcome filter gap, see below) -> **88/89** (the Axel_Bisi share's
  curve-file coverage itself grew between sessions -- MH-prefix mice that
  had no file earlier in this session gained one; only `MH038` still fails,
  with an unexplained "0 whisker trials found" mismatch, logged not
  debugged).
- `behavior/001_perfstate_behavior_comparison.py` (new `behavior/` folder,
  2026-09-16): learning curves by cohort + P(lick) by trial-type x state x
  cohort with paired Wilcoxon+t stats -- confirms the new definition
  tracks real behavior (whisker P(lick) strongly state-dependent in both
  cohorts; no-stim false-alarm rate isn't, as expected).

**Two bugs found and fixed during this validation** (both in the
now-production `_active_trials_from_whisker_onset_for_curve` /
`prep_perfstate_trials_curve` in `scripts/ssl_timeresolved_decoding.py`):
1. Off-by-one: reusing `ssl_bwm_trial_prep.prep_session` drops one extra
   trial (its own unrelated t-1-outcome feature) beyond the warmup trim --
   broke positional alignment with the curve file's trial indexing for
   every session. Fixed by NOT reusing `prep_session` for this; a local
   helper replicates only the context+warmup trim.
2. Missing `perf != 6` ("association" outcome) filter for the ~4 subjects
   whose `context` field is the literal string `"nan"` (context filtering
   is a no-op for those) -- `perf != 6` was the only thing actually
   excluding those trials, silently missing. Fixed by adding it, matching
   `cd_analysis/utils/performance.py`'s `keep_active_from_whisker_onset`.

**Production wiring** (`024_master_sweep.py`): all 3
`prep_perfstate_trials_generic(...)` call sites replaced with
`prep_perfstate_trials_curve(...)` (same signature minus `block_size`).
The old function is untouched in `ssl_timeresolved_decoding.py` for
backward comparison, just no longer called by the sweep.
`prep_perfstate_trials_curve` returns `None` (clean skip, same as any
other "no usable trials" case) for any session without a curve file or a
trial-count mismatch -- MH038, and expert-stage sessions (no curve file
exists for those, learning-stage day-0 only), degrade gracefully rather
than erroring or silently falling back to the old definition.

**"Ignore auditory blocks for modality decoding"** (user decision
2026-09-16, resolving the "keep vs drop aud_block" open question below):
moot by construction -- this pipeline's warmup-block trim already drops
pre-whisker-onset trials before `prep_perfstate_trials_curve` ever sees
them, so there is no `aud_block` category here at all (unlike
`cd_analysis`'s own scheme, which labels that prefix rather than dropping
it) -- applies to every decode target, modality included.

**Redo scheduled** (2026-09-16): `hitmiss_stim_area_group`/
`perfstate_stim_area_group` (predated the 2026-09-15 estimator switch AND
used the old perf-state def) -- old outputs backed up
`*.bak-preestimatorswitch-20260916` (both haas and local), appended to
`~/haas_remote_scheduler.sh` after the in-progress `area_acronym_custom`
block. `tables`/pytables added as a project dependency (needed for
`pd.read_hdf`) and synced to haas via `uv sync`.

---

**Original investigation (2026-09-15, before implementation)**: In the
ephys_utilities, there is a loader functions to load performance states
defined differently than in the current pipeline. Investigated (read-only,
on haas' `/mnt/lsens-analysis` mount) to have this ready when the time
came:

**Raw loader** (confirmed in `ephys_utilities`, as the user specified):
`ephys_utilities.helpers.load_helpers` (and the parallelized
`load_helpers_new`) both define `load_learning_curves_data(path_to_data,
subject_ids)`, reading `<mouse>_whisker_0_whisker_trial_learning_curve_interp.h5`
from `<path_to_data>/<mouse>/whisker_0/learning_curve/` -- on haas that
root is `/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4` (KS4-specific;
non-ks4 variant is `combined_results`, per `cd_analysis`'s older copy of
the same loader). Returns per-**whisker-trial** `p_mean`/`p_low`/`p_high`/
`p_chance` columns -- a pre-computed psychometric/Bayesian trial-by-trial
performance-probability trajectory with confidence bounds, not derived
on-the-fly from raw lick/outcome columns the way this project's own
`perf_state` is.

**State-labeling + propagation logic** (found in `cd_analysis/utils/performance.py`
-- built on top of the above loader, not itself in `ephys_utilities`):
- `assign_expertise_blocks(trial_table, n_consecutive=5)`: per **whisker
  trial** (not a fixed block), criterion is `p_low > p_chance` for
  `reward_group==1` (R+) or `p_low < p_chance` for `reward_group==0` (R-)
  -- cohort-direction-aware, consistent with this project's established
  cohort-dependent hit/miss semantics. A trial is labeled `'high'` only if
  it's part of a run of >=5 *consecutive* trials meeting the criterion
  (contiguous-run detection, not a fixed-size block partition); default
  `'low'` otherwise.
- `keep_active_from_whisker_onset`: the same auditory-warm-up-block trim
  we already implemented ourselves this session (drop trials before the
  first whisker trial) -- this version also exists verbatim in
  `ephys_utilities.helpers.data_utils` and `neural_utils_old`, so our fix
  and this pipeline agree on that part already.
- `propagate_expertise_inplace`: assigns `block_perf_type` to non-whisker
  trials (auditory_trial, no_stim_trial) via `pd.merge_asof(...,
  direction='nearest')` on `start_time` -- nearest-in-time whisker trial's
  label, always assigns something (never drops for being outside a span).
  This is exactly the "infer by looking at the time" mechanism the user
  described.
- A third category, `'aud_block'`, is assigned to the pre-whisker-onset
  auditory warm-up trials themselves (which our pipeline currently just
  *drops* instead of labeling) -- `load_and_merge_learning_curves_to_trial_table`
  is the full orchestrating function (load curves -> assign_expertise_blocks
  on whisker trials -> trim+separate the warm-up prefix -> merge onto the
  active trial table by `mouse_id`+`whisker_trial_id` -> propagate to
  non-whisker trials -> concat the `aud_block`-labeled prefix back in).

**How this differs from `prep_perfstate_trials_generic`
(`scripts/ssl_timeresolved_decoding.py:173`)**: our current version uses
fixed-size non-overlapping `PERF_BLOCK_SIZE`-trial blocks of whisker
trials, per-block mean cohort-corrected `rewarded` rate, and a **per-session
median split** (high = at/above that session's own median block hit rate)
-- a purely relative, session-normalized definition with no external model
or chance-comparison. Non-whisker trials get assigned by interval
membership in a block's `[min,max]` `start_time` span, and trials outside
every span are **dropped** (no perf_state), unlike the merge_asof
nearest-neighbor version above which always assigns one.

**(Original "to implement later" plan -- superseded, all points resolved
2026-09-16, see the status update at the top of this section)**: the
`mouse_id`+`whisker_trial_id` join, the port-not-import decision, and the
`aud_block` keep-vs-drop question are all done -- see
`prep_perfstate_trials_curve` in `scripts/ssl_timeresolved_decoding.py`.

## HOW TO RESUME (written 2026-09-15, for a fresh session/after a PC restart)

Everything below is real, currently-in-effect state that isn't yet
reflected in the entries above this line. Read this section first if
picking this project back up cold.

**All compute for this pipeline runs on haas056.rcp.epfl.ch (SSH alias
`haas`), not locally** -- this machine (32 cores) is idle by design; haas
(112 cores) is the default (see memory `ssl_prefer_haas_compute`). To
check live status: `ssh haas "cat ~/haas_scheduler_top.log"` for the
current stage, and `ssh haas "tail -N ~/code/ibl-ai-agent/projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/<log>.txt"`
for per-tag progress. The scheduler script itself,
`~/haas_remote_scheduler.sh` on haas, is the authoritative record of the
queue order -- read it directly if this section ever goes stale.

**Remote scheduler queue** (sequential, no concurrency), as of 2026-09-16
(third update -- see `~/haas_remote_scheduler.sh` on haas for ground
truth if this goes stale):
1. DONE: `area_group` redo (hitmiss, perfstate), original OLD estimator --
   see step 5, now superseded/backed up.
2. DONE: learning-stage modality redo (auditory-warmup-block fix) --
   `modality_stim_whole_brain`, `modality_lick_whole_brain`,
   `modality_stim_area_group`, `modality_lick_area_group`, all 89/89 (or
   85-88/89 depending on tag). All pulled to local and plotted
   (`032`/`034`/`028` for whole_brain; `032`/`034` for area_group).
3. **EXPERT-STAGE ON HOLD** (user request 2026-09-15) -- unchanged, still
   commented-out-not-deleted in `~/haas_remote_scheduler.sh`, right after
   step 2's block. To resume: uncomment those 8 lines in place.
4. RUNNING: `hitmiss_stim_learning_area_acronym_custom` -- first-ever run
   of this scheme, ~86/89 as of this writing, ETA a few minutes.
   `perfstate stim learning area_acronym_custom` (which was queued right
   after it) is **CANCELLED** -- see step 4b.
4b. QUEUED, PRIORITY (added 2026-09-16, "cancel the remaining perfstate
   decoding and replaced with the added version I mentioned. Run this in
   priority after the hitmiss area level decoding"): **`046_perfstate_
   baseline_decoding.py 999`** -- the non-time-resolved quiet_window/ITI
   perf-state decode (all 3 trial types, `whole_brain` scheme), inserted
   immediately after step 4 finishes, before modality. Log:
   `046_perfstate_baseline_log.txt`. This REPLACES both cancelled
   time-resolved perfstate runs (this step and step 5's perfstate half) --
   the time-resolved `decode_target='perfstate'` target is no longer being
   computed for any area scheme going forward.
4c. QUEUED: `modality stim/lick learning area_acronym_custom` (unchanged,
   runs after 4b).
5. QUEUED: **`hitmiss_stim_area_group` REDO only** (perfstate half
   CANCELLED, see 4b) -- user request ("Schedule hitmiss_stim_area_group/
   perfstate_stim_area_group with current estimator and with linear shift
   nulls"), since step 1's `hitmiss_stim_area_group` results predate the
   2026-09-15 pooled-CV estimator switch AND used the old block-median
   perf-state definition (now superseded, see the curve-based-perf-state
   section above). Old outputs backed up
   `*.bak-preestimatorswitch-20260916` (parquet + bin-edges json + log,
   both on haas and local) before appending the redo commands, so
   `024_master_sweep.py`'s resume-from-existing-parquet logic starts
   genuinely fresh rather than seeing the old 89/89 and no-op'ing. The
   `perfstate_stim_area_group` backup from step 1 is now permanently
   stale/orphaned -- not queued for any further redo.

**Methodology changes now in effect for every tag still to be computed**
(steps 4-5 above; NOT steps 1-2, already complete under the old rules):
- Pooled-scoring + resample-on-missing-class CV estimator (`decode_bin_pooled`/
  `select_fixed_c_pooled`/etc., since 2026-09-15) -- originals kept
  untouched in `ssl_timeresolved_decoding.py` for backward comparison.
- One null-control type per decode target: shift-null for hitmiss/perfstate,
  label-shuffle for modality (since 2026-09-15).
- **Curve-based perf-state definition** (`prep_perfstate_trials_curve`,
  since 2026-09-16) -- see the dedicated section above. Learning-stage
  only; a session without a matching curve file is cleanly skipped for
  perfstate, not silently downgraded to the old definition.

**Figure/display changes, 2026-09-16** (applies when replotting any
already-computed tag, not just new sweep output):
- `032_plot_decode_results.py`: which null curve is DISPLAYED now also
  follows decode-target (shift-null for hitmiss/perfstate in every panel
  including the pooled aggregate, not just within-cohort; label-shuffle
  for modality) -- was label-shuffle everywhere + shift-null only in
  within-cohort panels before. Area-overlay figures also gained an RT/
  stim-time histogram and a per-area null-distribution overlay (added
  incrementally this session -- see the script's own docstring history).
- `034_area_window_quant_grid.py`: the "vs null"/"above control" grid
  figures (`plot_group_grid`/`plot_group_grid_abovecontrol`) now annotate
  each area with `*` (Mann-Whitney U, real vs null, q<0.05 FDR-corrected
  per panel) and `^` (one-sample t-test of real-minus-chance against 0,
  ANOVA-equivalent for a single group, same FDR) -- see
  `area_above_chance_stats`. Tested on `hitmiss_stim_whole_brain` (shift-
  null path) and `modality_stim_area_group` (theoretical-0.5-fallback
  path); NOT yet regenerated for every tag (only the ones already local
  when this was implemented) -- worth a full sweep once the haas queue's
  current batch lands.
- `042_area_decoding_latency.py`: deliberately NOT touched by any of the
  above (user 2026-09-16: "Latency method decision not yet taken" --
  still on hold from the earlier "ignore the latency plots" instruction).

**New `behavior/` folder** (sibling to `exploratory-analyses/`, added
2026-09-16): `001_perfstate_behavior_comparison.py` -- learning curves by
cohort + P(lick) by trial-type x state x cohort, validating the curve-based
perf-state definition against real behavior. See the curve-based-perf-state
section above for what it found.

**LaTeX report**: `report_latex/main.tex` still reflects the OLD-estimator,
OLD-perfstate-definition `area_group`/`whole_brain` hitmiss/perfstate
results -- now doubly stale (estimator AND perfstate-definition both
superseded). **Needs a refresh once step 5's redo lands** -- not yet done
as of this writing (flagged to the user, not silently skipped). Compiling
needs `C:\Users\bisi\AppData\Roaming\TinyTeX\bin\windows` on PATH
(installed 2026-09-15 via `quarto install tinytex`, user-scoped). `main.pdf`
may be locked by an open viewer; compile to a different `-jobname` if so.

**Other durable state**: `scripts/axel_bisi_paths.py` (resolves
`M:\analysis\Axel_Bisi\...` locally vs `/mnt/lsens-analysis/Axel_Bisi/...`
on haas). `SSL_DECODE_N_WORKERS` env var controls worker-pool size
(defaults to 110, sized for haas; override to something small if ever run
locally). `tables` (pytables, needed for `pd.read_hdf` -- the learning-
curve loader) added as a project dependency 2026-09-16, synced to haas via
`uv sync` (`uv` is at `~/.local/bin/uv` on haas, not on the non-interactive
SSH `PATH`, in case that trips up a fresh session).

**`decode_target=='perfstate'` trial-type widening, 2026-09-16** (user
decision, motivated by baseline-window (-200,-10ms) quantification being
the primary planned use of these results): `decode_trial_types` for the
perfstate *target itself* (not the perfstate *condition-split* under
hitmiss/modality, which is unaffected) widened from `["whisker_trial"]` to
`["whisker_trial", "auditory_trial", "no_stim_trial"]` in
`024_master_sweep.py`. Baseline precedes any stimulus, so trial identity
shouldn't matter there, and more trials per state helps sessions previously
skipped for class imbalance. Trade-off: auditory/no_stim trials' perf_state
comes from nearest-whisker-trial `merge_asof` propagation, not the direct
p_low/p_chance criterion -- somewhat noisier near a state-transition
boundary. Also fixed a latent bug this surfaced: `matrices_and_y`'s
`is_whisker` mask for this target was hardcoded `True` for every row
(correct only because every row used to genuinely be a whisker trial) --
now `(trial_type == "whisker_trial")`, so the whisker-stimulus artifact
dead zone no longer incorrectly applies to auditory/no_stim rows. Synced to
haas; picked up automatically by both the queued `hitmiss_stim_area_group`/
`perfstate_stim_area_group` redo AND the still-to-come `perfstate_stim_
area_acronym_custom`/`perfstate` sub-tags of the in-progress
`area_acronym_custom` sweep (same shared script, no separate scheduling
needed) -- **not** yet applied to `perfstate_stim_whole_brain`, computed
earlier this session before this change and now stale on this point too
(not separately queued for a redo).

**Open items, 2026-09-16**:
- Latency method for `042` -- still undecided (three candidates compared:
  cluster-permutation, sigmoid-fit, 50%-of-peak).
- MH038 -- unexplained 0-whisker-trial mismatch in the curve-based
  perf-state alignment check, logged not debugged.
- LaTeX report refresh (see above).
- `034`'s new stat-annotation grids -- not yet regenerated for every tag.
- `hitmiss_stim_whole_brain` / `perfstate_stim_whole_brain` -- both predate
  the curve-based perf-state definition adoption (2026-09-16, same
  staleness reason as `hitmiss_stim_area_group`/`perfstate_stim_area_group`,
  which ARE queued for a redo below -- the whole_brain tags were missed).
  Corrected 2026-09-16: the previous version of this note tied the
  staleness to the trial-type-widening change, which was implemented then
  reverted same-day ("Agreed, forget the redo I asked") -- that reasoning
  no longer applies; this is the more fundamental and still-true reason.
  Not queued for a redo yet.

**Standing rule, 2026-09-16 ("For all figures, add as rule that the
contours and borders should not be cut or clipped")**: every accuracy/
score axis with a hard-coded literal bound touching the data's real range
(most commonly `ax.set_ylim(0.4, 1.0)`-style calls, where 1.0 is an
achievable ceiling for balanced accuracy) must carry a small headroom
margin (+0.02 typically) so marker edges, error-bar caps, and curve peaks
sitting at the boundary aren't visually clipped by the axes border; same
for x-axis bounds set to literal bin-edge min/max (add ~2% padding). This
was NOT previously the case -- `038` had already been hand-padded to 1.05
as a one-off fix, but every other hard-bound accuracy panel had zero
margin. Fixed 2026-09-16 in the actively-regenerated production scripts:
`025`, `027`, `028`, `032`, `033`, `036`, `039`, `041`, `046` (categorical
x-limits like `ax.set_xlim(-0.4, 1.4)` for 2-category bar/box plots were
left alone -- those already have generous margin by construction, not a
literal bound touching real data). `034`'s stat-annotation placement was
checked and found already compliant (`min(top + pad, ylim[1] - pad)`
clamping, written in from the start). `042` deliberately left untouched
per the standing "latency method undecided, don't touch 042" rule --
revisit once the latency method is settled. Apply this margin convention
to any new hard-bound axis limit added to a figure script going forward.

**Pons and medulla excluded from area_group figures everywhere, 2026-09-16
("Replot the area_group figures but excluding pons and medulla
everywhere")**: `"Pons and medulla"` (the actual `area_group` category
name, confirmed from `024_master_results_modality_stim_area_group.parquet`
-- there's no separate "Pons"/"Medulla" split) added to both `032`'s and
`034`'s `EXCLUDED_OVERLAY_AREAS` set (previously only dropped
somatosensory-body/amygdala/cortical-subplate/insular/visual/olfactory).
`032`'s per-area grid figure (`make_figure`, the non-overlay top-8-by-
session-count figure) was ALSO newly filtered by this set -- it previously
only applied to the area-overlay figure, so "everywhere" required widening
the filter's scope, not just adding one area name. Regenerated locally:
`032`/`034` for `modality_stim_area_group` and `modality_lick_area_group`
(the only area_group tags with current, non-`.bak` data right now).
`hitmiss_stim_area_group`/`perfstate_stim_area_group` still need this
replot once their queued haas redo (see above) lands -- both scripts are
already fixed and synced to haas, so that replot is just a rerun once the
new parquet exists. Scripts synced to haas (all in this and the previous
clipping-margin pass).

**Real left-panel ylabel/ytick clipping bug found and fixed, 2026-09-16
(user report: "the figure 032_hitmiss_stim_area_group_whole_area_overlay.png
still has the clipped y label on the left")**: this was NOT the same issue
as the axis-headroom margin pass above -- root-caused by reproducing it
directly with real data (with/without the RT-histogram overlay) to
`add_rt_histogram`'s `ax.twinx()` call. A twin axes sharing a subplot's
position confuses matplotlib's `bbox_inches="tight"` bbox union in a
multi-column figure and silently crops the LEFTMOST panel's own ylabel and
part of its y-tick labels (right panel unaffected) -- reproducible even
with a fully synthetic RT cache, so it's a rendering bug, not a data
artifact. Fixed by drawing the histogram bars directly on the primary
`ax` in its own data coordinates (rescaled to sit below the chance line)
instead of via `ax.twinx()`, in both `032_plot_decode_results.py` and
`028_wholebrain_crossgen_timeresolved.py` (identical duplicated helper,
also multi-column, same bug). `042_area_decoding_latency.py` has the same
twinx pattern but was deliberately left untouched per the standing
"latency method undecided" rule -- fix it too once that's revisited.
Verified fixed by direct before/after render with real data, then
regenerated every `032` tag with current data locally (`hitmiss_stim_
area_acronym_custom`, `hitmiss_stim_whole_brain`, `modality_lick_area_
group`, `modality_lick_whole_brain`, `modality_lick_whole_brain_
20mswindow`, `modality_stim_area_group`, `modality_stim_whole_brain`,
`perfstate_stim_whole_brain`) and confirmed on the actual saved PNG, not
just the test render. Both scripts synced to haas. `hitmiss_stim_area_
group`/`perfstate_stim_area_group` -- the tag the user's report was
actually about -- will pick this fix up automatically on its next replot
once the queued haas redo lands (see above), since no fresh data exists
locally for it yet.

**`042` latency figures updated, 2026-09-16 (user request: "Update the
latency figures. Order areas from earliest to latest. Annotate.")** --
the "latency method undecided, don't touch 042" standing caution was
about NOT changing the underlying latency methodology while three
candidate methods were still being compared; this request is a display-
only update explicitly asked for now, so it applies (methodology itself
is untouched -- same 3 methods, same null, same thresholds):
- `plot_method_panel` (the 5-panel `_method_comparison.png` grid, the
  "primary ask" figure) now sorts each panel independently by that
  panel's OWN latency values, ascending -- previously a FIXED shared area
  order (atlas order) so the same area sat at the same x position across
  all 5 panels; areas with no finite value for that panel's method are
  appended after the sorted ones as before (grey x-mark at y=0, not
  hidden). This trades cross-panel positional alignment for each panel
  being independently readable, which is what "order earliest to latest"
  asked for. `plot_latency_bars` (the cohort-split cluster-only figure)
  was already sorted ascending -- unchanged on that point.
- `plot_fit_diagnostics` (small-multiples grid) now orders panels
  earliest-to-latest by cluster-permutation latency, falling back to
  sigmoid-fit then 50%-of-peak group-mean latency when cluster is NaN;
  areas with no usable latency from any method are pushed to the end.
  Caveat: an area whose position is driven by the halfpeak fallback (weak
  fit, e.g. `r2~0` cases) can land "earliest" without the visual evidence
  really supporting it -- the fallback exists so every area still gets
  *some* ordering, not because the fallback estimate is as trustworthy as
  cluster-permutation.
- Annotate: every bar panel across all 3 figure types now appends each
  area's contributing-mice count to its x-tick label (`"Area (n=12)"`,
  method-specific `n` for the two per-mouse SEM panels -- how many mice's
  individual fits succeeded, not the group total); the cluster-permutation
  panels additionally print that cluster's own p-value next to its
  ms-value label, and `plot_fit_diagnostics`'s per-panel title gains a
  `p=` note alongside the existing `r2=`. New `latency_cluster_p` column
  added to the saved `_area_latency.csv`.
- Also added `"Pons and medulla"` to `042`'s own (previously separate,
  now-consistent) `EXCLUDED_OVERLAY_AREAS` set -- this script was off-
  limits during the "exclude pons and medulla everywhere" request, so it
  had been missed; now caught up.
- Fixed a real, unrelated overlap bug surfaced while regenerating
  `plot_latency_bars`: the "no significant cluster: <full area list>"
  subtitle could run to 30-48 area names and visually collide with the
  panel above under `constrained_layout` -- collapsed to a count
  (`"31 areas: no significant cluster"`) once the list exceeds 8 names;
  full names remain in the CSV/log as before.
- Regenerated and visually verified for `hitmiss_stim_area_acronym_custom`,
  `modality_stim_area_group`, `modality_lick_area_group` (the tags with
  current, non-`.bak` area-scheme data). Script synced to haas.

**Time-resolved `perfstate` decoding CANCELLED from the haas queue, replaced
by `046_perfstate_baseline_decoding.py`, 2026-09-16 (user request: "cancel
the remaining perfstate decoding and replaced with the added version I
mentioned. Run this in priority after the hitmiss area level decoding.")**:
- Cancelled (removed from `~/haas_remote_scheduler.sh`, never started):
  `perfstate stim learning area_acronym_custom` (was queued right after the
  in-progress `hitmiss stim learning area_acronym_custom`) and
  `perfstate stim learning area_group` (the redo queued at the very end,
  alongside `hitmiss stim learning area_group`). Both were the
  time-resolved, whisker-only-trial-type `decode_target='perfstate'` runs.
- Replacing them: `046_perfstate_baseline_decoding.py` (the non-time-
  resolved quiet_window/ITI decode designed earlier this session --
  all 3 trial types, `whole_brain` scheme, same pooled-CV estimator +
  linear-shift null as everything else) inserted into the scheduler
  **immediately after** `hitmiss stim learning area_acronym_custom`
  finishes -- i.e. in priority, before `modality stim/lick learning
  area_acronym_custom` and before the `hitmiss stim learning area_group`
  redo. Invoked as `python 046_perfstate_baseline_decoding.py 999` (was
  previously only validated on a 10-session subset -- `999` covers all
  ~89+ candidate sessions, growing as more curve files land) ->
  `046_perfstate_baseline_log.txt`.
- `046_perfstate_baseline_decoding.py` changed from a hardcoded
  `N_WORKERS = 10` to `int(os.environ.get("SSL_DECODE_N_WORKERS", "10"))`,
  matching `024_master_sweep.py`'s own env-var convention -- the scheduler
  already exports `SSL_DECODE_N_WORKERS=110` at the top, so this run uses
  haas's ~110 cores instead of the local-machine default of 10.
- Writes to its own separate output files (`046_perfstate_baseline_
  decoding_results.csv`/`.png`) -- does NOT touch `024_master_results_
  perfstate_*` at all, so the old (now permanently stale, not queued for
  any further redo) time-resolved perfstate parquet files are simply
  left as-is/orphaned, not overwritten.
- Not yet done: extending `046` beyond `whole_brain` to `area_group`/
  `area_acronym_custom` -- still explicitly deferred as a "follow-up once
  validated" per the script's own docstring; this pass only promotes it
  from a 10-session validation to a full production run, doesn't widen
  its area scope.
- Script synced to haas; scheduler script edited live (line ahead of the
  in-progress `hitmiss stim learning area_acronym_custom` command, same
  safe-edit pattern verified earlier this session -- PID/elapsed time on
  the running job confirmed unchanged immediately after the edit).

**`046`'s comparison figure revised to above-null diff + cohort split,
2026-09-16 (user follow-ups: "Ploot change as well", then "Break it down by
R+/R- cohort", then "Improve layout, use correct colors for cohorts")**:
raw accuracy alone (real~0.86 both windows) turned out to be dominated by
the shift-null (~0.85-0.86) -- barely any of it is above-null. Figure
rebuilt in stages: (1) added a second panel plotting `real - null_mean`
alongside raw accuracy, plus one-sample-vs-0 tests per window; (2) split
into `combined`/`R+`/`R-` columns, joining `reward_group` post-hoc from
the `REF_PATH` mouse-reference table (not in the per-session decode
output); (3) final layout is a landscape 2 (raw / above-null) x 3
(combined/R+/R-) grid, points colored by this project's standing
`COHORT_COLOR`/`AGGREGATE_COLOR` convention (green/magenta/muted-teal, not
the old teal/magenta-by-window scheme) with quiet_window=filled,
ITI=hollow markers carrying window identity instead. Pooled finding
(n=83): above-null diff is tiny (~+0.006 to +0.008) and only
quiet_window's is nominally >0 (uncorrected Wilcoxon p=0.011); neither
survives splitting by cohort (R+ p=0.072, R- p=0.10). Synced to haas after
each revision.

**New script: `047_perfstate_baseline_behavior_correlation.py`, 2026-09-16
(user request: "Correlate performance state decoding with difference in
performance (whisker hit rate, false alarm and difference of the two) per
mouse across states")**: correlates `046`'s per-mouse above-null decode
diff (both windows) against three (high-low) behavioral gaps computed from
the SAME curve-based perf-state labels `046` uses (`prep_perfstate_trials_
curve`, all 3 trial types) -- `hit_rate_gap` (P(lick|whisker), raw not
cohort-corrected -- same near-tautological-for-R+ caveat `037_perfstate_
confound_and_behavior_correlations.py` already documents for its own,
older-definition version of this measure), `fa_rate_gap`
(P(lick|no_stim), computed directly here since the curve-based prep
function doesn't produce a `block_fa_rate` column the way the older
generic-split one does), and `delta_lick_prob_gap` (hit_rate_gap -
fa_rate_gap). Unit of analysis is the MOUSE (sessions averaged within
`subject_id` first) per the user's explicit "per mouse" -- this is
learning-stage/day-0 only, so it doesn't conflict with [[ssl_stats_unit_of_
analysis]]'s session-level default (that's for expert-stage analyses).
`regress_and_plot` (scatter+OLS+95% CI, solid line only if p<0.05) copied
verbatim from `030`/`037`'s reference implementation per
`ssl_correlation_figure_style.md`. Produces pooled and per-cohort figures
(`047_perfstate_baseline_behavior_corr_{pooled,percohort}.png`) plus a CSV,
in `figures/behavior/`.

**Finding worth flagging**: pooled correlations are all weak/non-
significant (r=0.06-0.19, p>0.08), but the per-cohort split reveals a
clean, sensible dissociation -- **R+ decode diff correlates with
hit_rate_gap** (quiet_window r=0.43, p=0.00237*; hit-minus-FA gap
r=0.33-0.34, p<0.024 both windows) while **R- decode diff correlates with
fa_rate_gap instead** (quiet_window r=-0.51, p=0.00184*; ITI r=-0.44,
p=0.00762*) -- i.e. each cohort's neural state-decodability best tracks
the behavioral measure that actually IS their own rewarded contingency
(R+ rewarded for licking on whisker trials -> hit rate; R- rewarded for
withholding on catch trials -> [inverse] false-alarm rate). None of these
are FDR-corrected across the 12 cells in the percohort grid -- worth
correcting before treating any single p as confirmatory.

**Expert-stage sweep resumed, 2026-09-17 (user: "Yes do expert sessions
now", after the learning-stage queue finished overnight)**: this was the
block put on hold 2026-09-15 ("Put the expert run on hold for now").
Launched as a SEPARATE script, `~/haas_remote_scheduler_expert.sh` (not
appended to `~/haas_remote_scheduler.sh`, which had already run to full
completion -- relaunching that whole file from the top would have
re-executed everything already done, most importantly re-running
`046_perfstate_baseline_decoding.py` from scratch since it has no
resume/checkpoint logic at all and its one-time full run just took ~2.5h).
Backgrounded via `nohup ... > ~/haas_scheduler_top_expert.log 2>&1 &
disown`, confirmed actually running (`hitmiss stim expert whole_brain`
picked up worker processes within seconds of launch).

**One change from the original held block: `perfstate stim expert
whole_brain`/`perfstate stim expert area_group` are DROPPED, not
restored** -- found while preparing this relaunch, not previously
flagged: `024_master_sweep.py` calls `prep_perfstate_trials_curve`
unconditionally for the `perfstate` decode target regardless of
`DAY_STAGE`, and that function is learning-stage-only by construction (the
per-mouse learning-curve H5 file it needs covers only the single
day-0/whisker_0 session -- same root constraint that makes `046` itself
learning-stage-only, per its own docstring). Running these two jobs as
originally written would have silently produced an all-skipped, empty
result file for every expert-stage session -- not a real decode. Flagged
here rather than launched blind. Open question for the user: (a) leave
expert-stage perfstate uncomputed, (b) fall back to the old block-median
`prep_perfstate_trials_generic` just for expert stage (definition would
then differ by stage), or (c) something else -- not decided, not acted on
further without an answer. `hitmiss`/`modality` expert-stage jobs are
unaffected (neither target touches perf-state) and ran as originally
planned: `hitmiss stim expert whole_brain`, `modality stim/lick expert
whole_brain`, `hitmiss stim expert area_group`, `modality stim/lick expert
area_group` (6 jobs, sequential).

**Answered, 2026-09-17: option (b)** ("For the perf state, use the
perf-state definition with old-median block and use the consensus single
bin decoders") -- expert-stage perf-state decoding uses the old
within-session block-median split (`prep_perfstate_trials_generic`,
day-stage-agnostic by construction) AND the non-time-resolved
quiet_window/ITI single-bin decoder (`046_perfstate_baseline_decoding.py`,
the "consensus" replacement for the time-resolved sweep adopted
2026-09-16), NOT the old time-resolved per-bin sweep with the block-median
definition.

`046_perfstate_baseline_decoding.py` extended (not duplicated -- kept the
single-script self-containment but parametrized rather than forking a
sibling file) to take an optional `day_stage` CLI arg (`python
046_perfstate_baseline_decoding.py [n_test] [day_stage]`, default
`learning`): for `expert`, trial labeling switches to
`prep_perfstate_trials_generic` and the session list switches from the
learning-only `session_description=='whisker_0'` filter to
`hitmiss_session_list(sessions_tbl)[day_stage=='expert']` (the same
canonical expert-stage session list `024_master_sweep.py` itself uses).
Output files get an `_expert` suffix (`046_perfstate_baseline_decoding_
results_expert.csv`/`_expert.png`) so they never collide with the
learning-stage results. Everything else (windows, pooled-CV estimator,
shift-null, plotting) is identical between stages.

Validated with a 5-session run on haas before committing to the full one
(ran cleanly, block-median labels applied, correct `_expert`-suffixed
filenames). Appended to the still-running `~/haas_remote_scheduler_
expert.sh` (`python 046_perfstate_baseline_decoding.py 999 expert`, after
the 6 hitmiss/modality expert jobs -- live-edited ahead of the
then-currently-running `modality stim expert whole_brain`, same safe-edit
pattern verified earlier this session, PID/elapsed confirmed unaffected).
Script synced to haas.

Not done (out of scope of this request, flagged for later): `047_perfstate_
baseline_behavior_correlation.py` still hardcodes the curve-based labeler
and the learning-stage results filename -- it would need the same
`day_stage` treatment to work for the new expert-stage 046 output.

**Expert-stage figures generated for current results, 2026-09-17 (user:
"Make expert figures for current expert results")**:

**CORRECTED 2026-09-17 (was wrong, see below): "`hitmiss_stim_expert_
whole_brain` has ZERO computed rows"** -- at the time this was written,
every one of the 10 attempted sessions had failed the
`MIN_TRIALS_PER_CLASS=5` data-sufficiency check (e.g. "hit=4/miss=79",
"hit=0/miss=54"), and this was mistakenly generalized to "hit/miss
decoding is likely not viable at all at expert stage." **That was wrong**
-- those 10 rows turned out to be only 4 of the 29 candidate sessions (the
sweep's checkpoint was incomplete, not a full run), and by chance those 4
were exactly the most severely imbalanced ones. `048_expert_hitmiss_
trial_counts.py` (new script, user request "Plot the hit/miss trial counts
for each expert sessions") checked all 29 candidates directly and found
**25/29 clear the 5-per-class floor** at the whole-session level -- only
4 don't (`MH035_20250515_161921`, `MH036_20250516_162405`,
`MH036_20250517_140456`, all hit=4; `MH062_20260113_125836`, 0 usable
whisker trials at all). See `048_expert_hitmiss_trial_counts.png`/`.csv`.
Hit/miss decoding at expert stage IS viable for most sessions -- the
currently-running haas queue's `hitmiss stim expert whole_brain` step
resumes from that same incomplete 4-session checkpoint and will pick up
the other 25 properly.

`modality_stim_expert_whole_brain` DID produce real figures (partial data
pulled mid-sweep: 26-27/29 sessions, will be re-pulled once the full sweep
finishes) -- full `025`/`027`/`028`/`029`/`030`/`032`/`034`/`041` suite
regenerated. `modality_lick_expert_whole_brain` has no results file yet
(still queued on haas).

**Five real bugs found and fixed while doing this pass** (all previously
latent -- expert-stage's "results file exists, every row skipped" shape
never occurred for any tag before now): `027`, `028`, `029`, `030`, `034`
all crashed (`KeyError`/`IndexError` on `.iloc[0]`/column access) when a
results file existed but every row had a non-null `skipped_reason`. Fixed
by adding the same `len(df) == 0` -> graceful-skip check `025` already
had, to each script's own load function/guard (`030`'s `load_decode_
metrics`, `034`'s `load` -- which had NO existence/emptiness guard at all
before this, now returns `(None, None, None)` and `main` checks for it).
Also added `030` a `savefig_retry` wrapper (it had none, and one of these
runs hit the recurring transient Windows OSError with nothing to catch
it -- crashed the whole script rather than retrying, unlike every other
figure script in this project). All five scripts synced to haas.

042 (area latency) not run for either expert whole_brain tag -- it
explicitly requires >1 area (`"whole_brain has no areas to rank, nothing
to plot"`), same as for the learning-stage whole_brain tags.

**`modality_lick` switched to the linear-shift null, 2026-09-17 (user
finding + "Ok let's implement that for modality_lick only, add on TODO the
redo of all that")**: `modality_lick` (lick-time-aligned) is restricted to
`lick_flag==1` trials (`prep_lick_aligned_trials`, needs a real `lick_time`
to align to) -- unlike `modality_stim`, which uses the full,
experimenter-randomized whisker+auditory trial set regardless of outcome
(`prep_modality_trials`, "any lick_flag"). Auditory trials are rewarded on
lick in both cohorts at a high, session-stable rate, so the auditory-hit
subset stays close to the original randomized delivery; whisker hit rate
is behaviorally variable across the session (learning/engagement, and for
R- it isn't even the rewarded response), so the whisker-hit subset is NOT
uniform in time. That gives the two decoded classes different temporal
densities purely as a hit-filter artifact -- a label-shuffle null destroys
exactly the label-autocorrelation structure needed to catch a decoder
exploiting that artifact (same failure mode hitmiss/perfstate already
moved off label-shuffle for in the 2026-09-13 shift-null addition).
`modality_stim` is unaffected (no hit-filter) and stays on label-shuffle.

**Code changes**:
- `024_master_sweep.py`: `compute_shift_null` now also true for
  `decode_target=='modality' and alignment=='lick'` (was `decode_target in
  ('hitmiss','perfstate')` only) -- `compute_label_shuffle` is its
  complement as before, so this is a swap (shift-null replaces
  label-shuffle for `modality_lick`), not an addition -- keeps the
  established "one null type per (decode_target, alignment)" invariant and
  avoids doubling null-control cost for this target.
- `032_plot_decode_results.py`: `null_col`/`null_label` selection in
  `main()` extended with the same `alignment=='lick'` condition, so
  `modality_lick` figures now plot the shift-null (`modality_stim`
  unchanged, still label-shuffle).
- `034_area_window_quant_grid.py` and `042_area_decoding_latency.py`
  needed NO code change -- both already select the null opportunistically
  by column presence (`"shift_null_curves" in df.columns and ...notna().any()`
  in 034; `getattr(row, "shift_null_curves", None)` in 042's
  `session_surrogate_curve`), not by hardcoding `decode_target`, so they
  pick up the new `modality_lick` shift-null automatically once it exists
  in the data -- verified by reading both, not assumed.

**Redo launched, all `modality_lick` tags** (old outputs backed up
`*.bak-prelickshiftnull-20260917`, parquet+bin-edges+log, on both haas and
local, before any of this ran):
- Learning-stage: `modality_lick_whole_brain`, `modality_lick_area_group`,
  `modality_lick_area_acronym_custom` (all previously complete under the
  old label-shuffle-only logic -- full redo, not resumed, same
  can't-retroactively-add-a-column reasoning as every prior null-control
  addition this project has done).
- Expert-stage: `modality_lick_expert_whole_brain` was actively RUNNING
  (10/29 sessions in, under the old logic) when this was implemented --
  killed (`pkill`, confirmed both the worker pool and the parent
  `haas_remote_scheduler_expert.sh` exited cleanly via `set -e`) and its
  partial output backed up the same way, rather than let it finish and
  redo later. `modality_lick_expert_area_group` had not started yet, no
  backup needed.
- NOT touched: `modality_lick_whole_brain_20mswindow`/
  `_precqualitylabel` -- old one-off variant runs, not part of the
  maintained tag set.

Launched as a new consolidated script, `~/haas_remote_scheduler_
lickshiftnull.sh` (replaces the now-exited `~/haas_remote_scheduler_
expert.sh`): redoes the 3 learning-stage `modality_lick` tags first, then
continues the expert-stage sequence from the top -- `hitmiss`/`modality_
stim` expert (whole_brain + area_group) will resume-noop near-instantly
(already 29/29, unaffected by this change), `modality_lick` expert
(whole_brain + area_group) runs fresh with the fix, `046` perf-state
baseline (expert, block-median def) runs last, unchanged by any of this.
Confirmed actually running post-launch. Scripts synced to haas.

**Once this lands**: every `032`/`034`/`042`/`028` figure already
generated for `modality_lick_*` tags this session (learning whole_brain,
learning area_group, learning area_acronym_custom, and whatever expert
figures existed) is now STALE (label-shuffle null, not shift-null) and
needs regenerating once the redo finishes -- not done yet, tracked here.

---

## STATUS RECAP (2026-09-17, for a fresh session or after a break)

**Running now on haas**: `~/haas_remote_scheduler_lickshiftnull.sh`
(nohup'd, `~/haas_scheduler_top_lickshiftnull.log`), 10-job sequential
queue, currently on job 1:
1. **RUNNING**: `modality_lick_whole_brain` (learning, shift-null redo) --
   ~14/89 sessions last checked, zero errors (bug fixed, see above).
2. QUEUED: `modality_lick_area_group` (learning, shift-null redo)
3. QUEUED: `modality_lick_area_acronym_custom` (learning, shift-null redo)
4. QUEUED: `hitmiss_stim_expert_whole_brain` (resume -- was stuck at an
   incomplete 4/29-session checkpoint, see the corrected finding above;
   will process the remaining 25 sessions properly this time)
5. QUEUED: `modality_stim_expert_whole_brain` (resume-noop, already
   ~27-29/29)
6. QUEUED: `modality_lick_expert_whole_brain` (fresh, shift-null)
7. QUEUED: `hitmiss_stim_expert_area_group` (fresh)
8. QUEUED: `modality_stim_expert_area_group` (fresh)
9. QUEUED: `modality_lick_expert_area_group` (fresh, shift-null)
10. QUEUED (last): `046_perfstate_baseline_decoding.py 999 expert`
    (block-median perf-state def, quiet_window/ITI single-bin decoder)

**Three things resolved this session, in order**:
1. `modality_lick` switched from label-shuffle to linear-shift null (user
   finding: its `lick_flag==1` hit-filter makes whisker-hit trials
   non-uniform in time, unlike auditory-hit trials -- label-shuffle can't
   catch a decoder exploiting that). Found and fixed a real crash bug
   along the way (`res["null_curves"][0]` unguarded against `None`).
2. Expert-stage `perfstate` cannot use the curve-based definition at all
   (the per-mouse learning-curve file is day-0-only by construction, no
   expert-stage equivalent exists or ever will under the current data) --
   replaced with the old block-median definition + `046`'s non-time-
   resolved quiet_window/ITI decoder, `day_stage='expert'`.
3. **Corrected a premature conclusion**: hit/miss decoding at expert stage
   IS viable for most sessions (25/29 clear the trial-count floor) --
   the earlier "ZERO computed rows, not viable" claim was based on an
   incomplete 4/29-session sweep checkpoint, not a real negative result.

**Figures needing regeneration once the queue above lands** (none of this
is done yet):
- Every `modality_lick_*` figure (`032`/`034`/`042`/`028`, learning
  whole_brain/area_group/area_acronym_custom) -- stale null type.
- `hitmiss_stim_expert_whole_brain` figures -- no real data existed until
  job 4 above runs.
- `modality_stim_expert_whole_brain` figures -- currently built on partial
  (26-27/29) data, re-pull once 29/29.
- `hitmiss_stim_expert_area_group`/`modality_*_expert_area_group` figures
  -- don't exist yet, first-ever run for these tags.
- `046` expert-stage perf-state comparison figure (raw + above-null +
  cohort-split, matching the learning-stage one) -- doesn't exist yet.

**Not done, explicitly deferred** (unchanged from earlier in this file,
still true): LaTeX report refresh; `034` stat-annotation grids not
regenerated for every tag; `042` latency method still undecided (three
candidates); `047_perfstate_baseline_behavior_correlation.py` not
extended for `day_stage='expert'`; `046`/`047` not extended beyond
`whole_brain` to `area_group`/`area_acronym_custom`; MH038's unexplained
0-whisker-trial mismatch, logged not debugged.

**SECOND instance of the `null_curves[0]`-on-`None` crash found + fixed,
2026-09-17 (surfaced while doing "For the learning case, make equivalent
plots with the ANOVA and post hoc window quantifications but looking at
the decoding accuracy above chance")**: `hitmiss_stim_area_group`'s
earlier "DONE" redo (the one from the `hitmiss_stim_area_group`/
`perfstate_stim_area_group` REDO section above) was NOT actually complete
-- checked its log and found 85/89 sessions had ERRORed with the exact
same `TypeError("'NoneType' object is not subscriptable")` the
`modality_lick` fix addressed, leaving only 4/89 sessions (all failing
the class-count floor, hit=0-3) in the output that got marked DONE. This
predates today's `modality_lick` work entirely -- `compute_shift_null`
has been `True` for `hitmiss`/`perfstate` unconditionally since well
before this session, so this crash has apparently existed since whatever
point `024_master_sweep.py` was last edited to introduce the "one null
type only" cost optimization, and would hit ANY hitmiss/perfstate tag
computed fresh after that edit.

**Checked every hitmiss/perfstate log for `ERROR` count to find the true
blast radius** (`grep -c ERROR`, not assumed): `hitmiss_stim_whole_brain`,
`hitmiss_stim_area_acronym_custom`, `perfstate_stim_whole_brain` are all
**genuinely clean (0 errors)** -- confirmed by checking that their
computed rows have BOTH `null_curves` and `shift_null_curves` populated,
proving they were computed under the OLDER "compute both nulls for
hitmiss/perfstate" policy, before whatever edit introduced this bug --
they predate it entirely, not immune to it. (They remain stale on the
separate, already-documented point: predating the pooled-CV estimator
and/or curve-based perf-state definition -- `hitmiss_stim_area_acronym_
custom` IS current on that front per the project's own "steps 4-5" note
above.) `hitmiss_stim_expert_whole_brain` has 25 errors -- this IS the
same bug, already covered by the currently-running queue's resume step.

**Fix**: `hitmiss_stim_area_group`'s broken output backed up
(`*.bak-nullcrash-20260917`, haas + local) and `hitmiss stim learning
area_group` re-queued -- inserted (live-edit of the still-running
`~/haas_remote_scheduler_lickshiftnull.sh`, same safe-ahead-of-current-line
pattern, PID/elapsed confirmed unaffected) right after the modality_lick
learning-stage redo and before the expert-stage sequence, so it runs with
some priority rather than at the very end.

**Delivered now, not blocked on the redo**: `hitmiss_stim_area_acronym_
custom` is valid/current data (confirmed clean above) -- regenerated all
9 `034` figures for it (`grid_plus/minus/comparison` x
`default/raw/abovecontrol`), giving the user the requested "ANOVA +
post-hoc window quantification, accuracy above chance" plots for the
learning case immediately, without waiting on the queue.
`034_hitmiss_stim_area_group_*` (same request, once its redo lands) and
any `perfstate` equivalent (none exists at the area level any more --
time-resolved `perfstate` was cancelled project-wide, see above) remain
outstanding.

**Also fixed while here**: `034_area_window_quant_grid.py` had no
`savefig_retry` wrapper at all (the last figure script in the project
still using a bare `fig.savefig`) -- hit the familiar transient Windows
`OSError` mid-run; added the same wrapper every other figure script uses.

**Three new comparison-axis figures added, 2026-09-18 (user: "Now make
within-cohort across-condition anova/posthoc across area figures, and
within-cohort across-learning-stage anova/posthoc across area figures, as
two other ways to compare the data. Do the same with paired_grid plots,
across-learning-stage with two cohorts overlaid.")**:

1. **`034`, within-cohort, across-CONDITION** (`034_{tag}_grid_{plus,
   minus}_condition.png`): generalized `cohort_anova_and_posthoc` into
   `factor_anova_and_posthoc` (any two-level factor, not just
   `reward_group` -- `cohort_anova_and_posthoc` is now a thin wrapper
   calling it with `reward_group`/R+/R-, every existing call site
   unchanged). New `plot_within_cohort_condition_grid`: within one fixed
   cohort, compares `half:first` vs `half:second` per area (ANOVA +
   per-area FDR-corrected MWU post-hoc). Scoped to `half` only, NOT also
   `perfstate` -- matches this script's own already-established exclusion
   of perfstate as a condition dimension (2026-09-13: "ignore the perf
   state decoding per state, as it is redundant"). Diff-from-chance only
   (not also raw/abovecontrol) -- scoped down deliberately, reuses the
   tag's existing `ylim`.
2. **`034`, within-cohort, across-LEARNING-STAGE** (`034_{tag}_grid_
   {plus,minus}_stage.png`): new `plot_factor_comparison_grid` (the
   generalized form of `plot_comparison_grid`, which is now a thin
   wrapper around it) called with `factor_col='day_stage'`. New
   `expert_tag_for`/`load_stage_merged` auto-derive and load the sibling
   expert-stage tag (e.g. `hitmiss_stim_area_group` ->
   `hitmiss_stim_expert_area_group`) and concatenate with a `day_stage`
   column -- silently skipped (printed, not an error) when the expert
   sibling isn't ready yet. Verified end-to-end on `hitmiss_stim_
   area_group` (now that both its learning and expert redos are done):
   real signal, e.g. baseline|whole ANOVA day_stage F=31.21, p=4.6e-08.
3. **`029`, across-LEARNING-STAGE, cohorts overlaid**
   (`029_{tag}_{window}_paired_grid_overlaid_stage{,_abovechance}.png`):
   new `STAGE_PAIRS` (learning tag, expert tag) list, `per_subject_whole_
   values` (aggregates the 'whole' condition to one value per (subject_id,
   reward_group), NOT per session_id -- a learning session and an expert
   session are different recordings of the same mouse, so pairing must be
   by mouse) and `render_stage_pair_figure` (reuses `overlaid_panel`
   directly, one square panel instead of the usual 2x2 grid since there's
   only one comparison here, not four). Both raw and above-chance variants
   produced, same convention as the 2026-09-17 addition. NOT added to
   `027` (the spaghetti-line, non-overlaid sibling) -- the user's request
   specifically said "with two cohorts overlaid", matching `029`'s style
   only.

All three verified working end-to-end (real figures generated and
visually checked, not just "runs without crashing") using the newly-fixed
`hitmiss_stim_area_group`/`hitmiss_stim_expert_area_group` data.
`modality_lick`'s stage-pair figure not yet produced -- its expert-stage
redo is still queued on haas. Both scripts synced to haas.

Explicitly out of scope for this pass (not done): raw/abovecontrol
variants of the two new `034` comparisons (diff-from-chance only);
extending either new comparison to `area_acronym_custom` tags (should work
unchanged since both reuse the tag-generic `common_areas`/`present_
columns` machinery, just not actually run/verified for that scheme yet);
a `perfstate`-condition version of the across-condition comparison.
Synced to haas.

**Sensory window widened 5-35ms -> 5-50ms post-whisker-stimulus, 2026-09-18
(user: "The sensory window is defined as 5ms to 35ms post whisker
stimulus. Update that to 50ms. update statistics figures")**: `SENSORY_
WINDOW = (0.005, 0.035)` -> `(0.005, 0.050)` in all 7 scripts that define
it independently (per-script self-containment convention, no shared
constants module): `026_wholebrain_window_stats.py`, `027_wholebrain_
window_stats_figures.py`, `029_wholebrain_paired_grid_overlaid.py`,
`030_wholebrain_behavior_correlation.py`, `034_area_window_quant_grid.py`,
`037_perfstate_confound_and_behavior_correlations.py`,
`041_learning_vs_expert_wholebrain_comparison.py`. Confirmed `032_plot_
decode_results.py` and `042_area_decoding_latency.py` do NOT independently
define this constant (grep, no match) -- `042` untouched per the standing
rule (latency method still undecided), no action needed there. `024_
master_sweep.py` (the decoding sweep itself) also doesn't define it --
this is a post-hoc window-quantification constant only, not a decoding
input, so the change does not require re-running any decoding, only the
downstream stats/figure scripts. All 7 edited files compiled clean
(`py_compile`) and synced to haas. Regeneration launched on haas
(026,027,029,030,037,041 no-arg, then 032+034 looped over the 14
currently-stable tags: hitmiss_stim_{whole_brain,area_group,area_acronym_
custom}, modality_stim_{whole_brain,area_group,area_acronym_custom},
modality_lick_{whole_brain,area_group,area_acronym_custom}, perfstate_
stim_whole_brain, hitmiss_stim_expert_{whole_brain,area_group}, modality_
stim_expert_whole_brain, modality_lick_expert_whole_brain) -- deliberately
excludes `modality_stim_expert_area_group`/`modality_lick_expert_
area_group`/expert `046` baseline, which were still running on haas at
the time this regen was launched (queued behind the lickshiftnull
scheduler); those three need a follow-up figure regen (032+034, plus
whichever no-arg scripts pick up new expert area_group tags) once their
decoding finishes.

Follow-up (same day): the lickshiftnull scheduler's decoding queue
finished in full -- `modality_stim_expert_area_group`, `modality_lick_
expert_area_group`, and `046` perf-state baseline (expert, block-median
def, same linear-shift-null convention as `024`) all landed clean (`046`
expert: 56/56 session x window cells decodable, quiet_window real=0.844
vs null=0.826, ITI real=0.839 vs null=0.820). Ran a follow-up batch:
`032`+`034` for the two newly-landed expert area_group tags, then reran
all 6 no-arg whole-brain scripts (`026,027,029,030,037,041`) once more so
they'd pick up the new expert area_group data too. Also caught and fixed
a launch bug in the first regen pass: `030_wholebrain_behavior_
correlation.py` reads `reports/ssl_analysis/derived/mouse_reference.
parquet` relative to the repo root (via `scripts/ssl_timeresolved_
decoding.py`'s `MOUSE_REF_PATH`), not relative to `exploratory-analyses/`
-- running the batch with CWD=`exploratory-analyses/` crashed `030` and,
under `set -e`, silently killed the rest of that batch (`037`/`041`/all
`032`/`034` tag loops never ran in the first pass). Relaunched from repo
root with per-script failure tolerance instead of `set -e`; this and the
follow-up batch both completed with exit 0, no `FAILED` lines.

**Net result**: every decode_target x alignment x area_level x stage
combination that has landed data now has its statistics figures
regenerated with the 50ms sensory window. Remaining known gap (unrelated
to this window change): expert-stage `area_acronym_custom` was never
queued for hitmiss/modality_stim/modality_lick -- only whole_brain and
area_group got expert redos scheduled. `047_perfstate_baseline_behavior_
correlation.py`'s expert extension also remains deferred, not requested.

Two more bugs found the same day, both post-hoc after the above (user:
"Yes but also update the single learning stage figures!" then "Why do the
figures still show (5-35)ms in the title then?"):

1. **Stale cross-stage data race**: `034`'s per-tag "_stage" comparison
   figure for the LEARNING tag `modality_lick_area_group` was generated
   at 12:19 -- but its expert sibling (`modality_lick_expert_area_group`)
   didn't finish decoding until 13:27, so that figure had been built
   against a still-partial/mid-write expert parquet, not the completed
   data (`modality_stim_area_group`'s equivalent, generated at 12:14,
   was fine -- its sibling had already landed at 11:42). Root cause: the
   follow-up regen batch only reran `034` on the two EXPERT tags once
   they landed, never on the corresponding LEARNING tags, so the
   learning tag's own `_stage` figure (which is what actually reads and
   merges the sibling) never got a chance to re-observe the now-complete
   expert data.
2. **Hardcoded window text, independent of `SENSORY_WINDOW`**: even after
   fixing (1), figures still read "5-35ms" in the title/axis label. Found
   two literal strings never wired to the `SENSORY_WINDOW` constant:
   `034_area_window_quant_grid.py`'s `secondary_label = "sensory
   (5,35ms)"` (line ~231) and `037_perfstate_confound_and_behavior_
   correlations.py`'s y-axis label `"...(sensory window, 5-35ms)"` (line
   ~232) -- both hardcoded at the time those features were written,
   completely separate from the `SENSORY_WINDOW` tuple that drives the
   actual computation (which HAD been using 50ms correctly the whole
   time -- only the on-figure text was stale). Fixed both to format the
   label from `SENSORY_WINDOW`/`PRE_LICK_WINDOW` directly so this can't
   drift again. A batch of `034` reruns for the 3 learning area_group
   tags had already launched using the pre-label-fix script by the time
   this was found -- killed it (process + parent bash), confirmed no
   leftover processes, and relaunched a full clean batch: `037` (no-arg)
   + `034` for all 16 currently-landed tags (14 stable + the 2 expert
   area_group). Completed exit 0, no `FAILED` lines; spot-checked every
   tag's log line and all now read `rows: [... 'sensory (5,50ms)' ...]`
   or `rows: ['pre-lick (-150,0ms)']` as appropriate -- no more `5,35ms`
   anywhere. `032` and `026/027/029/030/041` were grep-checked and do NOT
   contain this hardcoded-label pattern, so they needed no equivalent fix.
   Both fixed scripts synced to haas.

## 2026-09-19: curve-based expert-stage loader generalized + new "perfquant" regression target (validation only, mixed result)

User: "Did you rerun the state decoding with the curve-based states? Some
of it is now available for expert too, but in whisker_1, whisker_2,
whisker_3 etc folders instead: adjust data loading accordingly. Also, for
both learning and expert stages, instead of decoding state label, regress
the learning curves instead: whisker learning curves, false alarm and
difference of the two (call it performance), but quantize it in 5 classes
(20% splits), regress the value and check whether it lands in the correct
classes (so, a classification)."

**Data-availability check (before writing any code)**: checked haas
directly -- `combined_results_ks4/<mouse>/whisker_N/learning_curve/` now
has real curve data for `N>0` (expert-stage) for AB-cohort mice, but NOT
yet for any MH-cohort mouse (the actual ephys decoding cohort) -- checked
~10 MH mice's `whisker_1`/`whisker_2`/`whisker_3` folders directly, all
had `roc_analysis`/`spontaneous_licks`/etc. but no `learning_curve`. Asked
the user 3 clarifying questions given this mismatch + the new target's
design space; answers: (1) generalize the loader now for whenever MH data
lands, don't force expert-stage neural results today: (2) add the new
quantile-regression target(s) ALONGSIDE `perfstate`, don't replace it; (3)
quantile edges pooled across ALL sessions/trials globally, not per-mouse.

**Part 1 -- day-aware curve loader (`scripts/ssl_timeresolved_decoding.py`,
`scripts/ssl_bwm_trial_prep.py`)**: added public `session_training_day`
(`ssl_bwm_trial_prep.py`, refactored out of `list_whisker_training_ephys_
sessions`'s existing day-parsing regex -- this `day` value IS the same
`N` Axel Bisi's `whisker_N` folders use, no separate mapping needed).
Added `load_whisker_curve_row(curve_root, subject_id, day, trial_type,
interp)` -- direct H5 read, day-parameterized, replacing the dependency
on `ephys_utilities.helpers.load_helpers.load_learning_curves_data`
(hardcoded to `whisker_0` in its own filename template). Rewrote
`prep_perfstate_trials_curve` to use it. Validated against real data: all
88 learning-stage (day 0) sessions behave identically to before (same
file, same content); all 29 expert-stage MH sessions still correctly
return None (no file yet) -- fully backward-compatible, and will pick up
MH expert curves automatically the moment they land, no further code
change needed.

**Part 2 -- new `perfquant` regression->quantile-class target, added
alongside `perfstate`**: new `prep_perfquant_curve_targets` (per-whisker-
trial `whisker_curve` [p_mean, positionally aligned like `perfstate`'s
`p_low`], `falsealarm_curve` [no-stim/catch-trial curve's `p_mean`,
linearly interpolated onto each whisker trial's `start_time` -- no
positional correspondence between the two trial types], `performance_
curve` [their difference]); `quantile_edges_from_pooled_values` (5-class,
20%-split edges, `-inf`/`+inf`-capped); `_make_regressor`/
`select_fixed_alpha_pooled`/`decode_regression_quantile_pooled`/
`linear_shift_null_regression_quantile_pooled` (Ridge-regression analogs
of the existing `_make_classifier`/`select_fixed_c_pooled`/
`decode_bin_pooled`/`linear_shift_null_curves_pooled`, scored by
quantile-match accuracy against the pre-computed global edges rather than
balanced accuracy).

**Bug found + fixed while validating targets**: the no-stim/false-alarm
curve file indexes *every* no-stim trial in the session, including the
~11-16 pre-whisker "auditory warm-up" trials this project's standard
whisker-trial trimming excludes -- caused a systematic trial-count
mismatch (verified: e.g. AB080 had 173 curve rows vs. 161 in the trimmed
set, and the 12-trial gap exactly matched the count of no-stim trials
before that session's first whisker trial) that silently left
`falsealarm_curve`/`performance_curve` all-NaN for 76/88 sessions before
the fix. Fixed via a new `_active_trials_for_curve_untrimmed` helper
(context/perf filter only, no warmup-trim) used specifically for the
no-stim interpolation basis -- this doesn't reintroduce warmup trials into
any decoded trial set, since the curve is only ever an interpolation
basis, never decoded itself. Revalidated: 88/88 sessions now have
complete, non-NaN targets for all three curve types.

**Validation script**: new `049_perfquant_curve_regression_test.py`
(mirrors `043`'s role for the original curve-based perfstate rollout --
small-scale plausibility check before any production-sweep candidacy, NOT
a final result). Whole_brain only, single sensory window (5-50ms, current
`SENSORY_WINDOW`), same 20-mouse `TEST_MICE` subset `043` used, global
quantile edges fit on all 88 curve-available learning-stage sessions
first. Hit and fixed a second bug during this run: `data_sufficiency_ok`
(a binary hit/miss-style check) was wrongly called with a dummy all-True
`y` as a stand-in for "does this session have enough trials" -- that dummy
array has zero negative-class trials by construction, so it skipped every
single session. Replaced with a direct unit-count + trial-count check
appropriate for a continuous target; also added an empty-dataframe guard
in the summary step (same class of bug as the five `027`/`028`/`029`/
`030`/`034` fixes earlier this project).

**Result (real, not yet a positive finding)**: mechanism runs end-to-end
and produces per-trial regression -> quantile-class scoring correctly,
but at whole_brain/single-sensory-window scope the real quantile-match
accuracy is only marginally different from its own shift-null for all
three targets -- `whisker_curve`: mean real=0.548 vs null=0.539 (4/20
sessions above null+1sd); `falsealarm_curve`: mean real=0.473 vs
null=0.473 (1/20 above); `performance_curve`: mean real=0.436 vs
null=0.427 (2/20 above). All far above the naive 1/5=0.2 chance level,
but that gap is almost entirely explained by the curve's own slow,
autocorrelated drift over a session (the same phenomenon the shift-null
was designed to catch for `hitmiss`/`perfstate`/`modality_lick`) rather
than a real above-and-beyond neural-decoding effect at this scope. Not
run at any other area level or as a time-resolved curve yet -- open
question for the user on whether to try a narrower/different window,
per-area decoding, or a time-resolved curve (like `perfstate`'s own
sweep) before deciding whether this is worth a full production sweep.
049's own CSV/log kept on haas for reference, not pulled/regenerated as
figures (this was a validation run, not a sweep).

**2026-09-20 follow-up ("Ok try 1) and 2)"): null R^2 added, residualized
targets added -- result is nuanced, R^2 is a much more sensitive metric
than quantile accuracy here.** Extended `linear_shift_null_regression_
quantile_pooled` to return null R^2 alongside null quantile-accuracy; new
`residualize_curve` (each target minus its own session-local centered
15-trial rolling mean) decoded alongside every raw target (6 variants
total). Same 20-session set, whole_brain, sensory window.

Raw targets, on R^2 (NOT on quantile accuracy): real clearly and
consistently beats its own null -- `whisker_curve` real=0.411 vs
null=0.215 (13/20 sessions individually above null+1sd), `falsealarm_
curve` real=0.318 vs null=0.223 (6/20), `performance_curve` real=0.256 vs
null=0.136 (12/20). This is a MUCH cleaner separation than the original
quantile-accuracy metric showed (3/20, 1/20, 3/20 respectively, from the
2026-09-19 run) -- quantile-match accuracy turns out to be a much
noisier/lossier readout of the same regression fit than R^2 is at this
per-session, per-class trial count (5 classes over ~100-250 trials/session
means ~20-50 trials/class). Worth remembering going forward: R^2 looks
like the more sensitive diagnostic here, even though the user's original
design intentionally wanted a classification framing for interpretability
and comparability with the rest of the pipeline's "accuracy vs chance"
convention.

Residualized targets: absolute R^2 collapses to near/below zero for all
three (expected -- residualizing removes the large, easily-predictable
trend component, leaving much smaller/noisier residual variance to
predict) -- but the REAL-vs-NULL gap does NOT fully disappear for two of
the three: `whisker_curve_resid` real R^2=-0.070 vs null=-0.156 (13/20
sessions still above null+1sd -- real is reliably less-bad than null even
though neither is a "good" fit in absolute terms) and `performance_curve_
resid` real=-0.116 vs null=-0.168 (7/20 above). `falsealarm_curve_resid`
essentially vanishes (real=-0.165 vs null=-0.145, only 1/20 above --
real is actually slightly WORSE than null here). Read together: some
genuine trial-to-trial coupling beyond the shared slow-drift confound
plausibly exists for the whisker-curve and performance targets
specifically, but not clearly for false-alarm.

Not yet decided/done: whether to report R^2 (not quantile accuracy) as
the primary metric for this target going forward; whether to scale the
test set, try area-level decoding, a time-resolved curve, or the
pseudopopulation route (`033_pseudopopulation_hitmiss_wholebrain.py`'s
cross-session unit-pooling, discussed as an option for "more data" but
not yet adapted to quantile-class-matched pseudo-trials) before any
production-sweep candidacy. Still whole_brain/single-window/20-session

## 2026-09-20 (continued): cross-animal (050) and single-animal per-neuron
(052-054) perfquant follow-ups, plus a real numerical bug found and fixed

Large follow-up arc on the perfquant regression target, spanning several
new scripts: `050_perfquant_crossanimal_arealevel_test.py` (cross-SUBJECT
generalization, area-averaged FR features, leave-one-subject-out within
cohort), `051_perfquant_crossanimal_extra_figures.py` (post-hoc real-vs-
null histograms/boxplots/fraction-above-null bars from 050's CSV),
`052_perfquant_singleanimal_multihead_test.py` (single-SESSION, per-NEURON
features -- legitimate here since there's no cross-animal step -- multi-
output Ridge jointly predicting all 3 curves as "three heads", full
retrain-under-shift null), `053_perfquant_singleanimal_examples.py`
(single-trial-sequence example figures: true vs predicted, null mean+-
95%CI bands, train-vs-test overfitting comparison), `054_perfquant_
singleanimal_violin_comparison.py` (real-vs-null violin/diff-distribution
figures across window x metric x cohort, with statistical annotations).

**Cross-animal (050) headline, at full scale (88 subjects, 50 R+/38 R-,
6 common areas x 2 windows = 12 features, per-subject z-scored, raw
target)**: real Pearson beats null clearly for `whisker_curve` (0.168 vs
0.020, 47/88 subjects above null+1sd) and `performance_curve` (0.129 vs
0.005, 45/88), not for `falsealarm_curve` (0.066 vs 0.034, 19/88). R^2 is
a poor metric here specifically because of cross-SUBJECT miscalibration
(different animals, different absolute scales) -- Pearson/Spearman are
the trustworthy metrics for this cross-animal design.

**Single-animal (052-054) headline**: per-neuron features (~700-3000
neurons) massively outnumber trials (~100-250) -- train R^2/Pearson is
trivially 1.00 in every single case checked (perfect in-sample
interpolation, not a real fit), so TEST (pooled-CV out-of-fold) is the
only number that means anything. At full scale (88 sessions), sensory
window (5-50ms post-stim) shows a robust, consistent real>null gap across
both cohorts and all 3 metrics (Wilcoxon p<0.001 in 5/6 cohort x target
cells); the wide baseline window (-2000/-10ms pre-stim) shows real signal
too but less consistently (fails for R- on `falsealarm_curve` and, before
the bug fix below, `performance_curve`). A width-matched immediate
baseline-correction test (`FEATURE_MODE="corrected"`: sensory minus a
45ms immediate pre-stim window, same width as sensory) was attempted to
test whether subtracting a shared slow-drift component would sharpen the
signal -- not yet re-validated after the bug fix below (was the run that
surfaced it).

**Real bug found and fixed (2026-09-20), NOT the same as `048`-era
crashes -- a silent NUMERICAL corruption, not a crash**: `FEATURE_MODE=
"corrected"` produced R^2 around -1e28 for every session. Root cause,
confirmed by hand (varying alpha across the whole grid changed nothing,
ruling out regularization): a neuron's activity can have fine variance
across a WHOLE session yet still be exactly/near-constant within one
particular CV fold's training subset, purely by chance. Vanilla
`StandardScaler`'s fitted std for that fold is then a tiny-but-nonzero
float (not exactly 0, so sklearn's own zero-variance guard never
engages), so `1/std` explodes -- verified directly: scaled held-out test
features reached ~1e17 while the Ridge coefficients themselves stayed
perfectly normal (~0.03-0.05). A whole-session variance pre-filter
(dropping columns with session-wide std <=1e-6) does NOT catch this,
since it's a per-fold coincidence, not a whole-column property -- tried
first, confirmed insufficient by rerunning and seeing the identical
~1e28 blowup. Fix: floor the fitted `scale_` to `SCALE_FLOOR=1e-3` Hz
(added to `_make_multihead_regressor` in `052` and `_make_regressor` in
`053`) -- directly prevents the division blowup regardless of which
fold's training subset happens to be near-degenerate. Verified against
the exact reproduction case (MH022, `FEATURE_MODE="corrected"`): R^2 went
from -1e28 to 0.29-0.55, alpha selection went from always-maxed-out
(1e+1, desperately fighting the blowup) to 1e-5 (the actual best fit).

**This bug turned out to ALSO silently affect `FEATURE_MODE="baseline"`
(the wide 2000ms window)** -- discovered via `054`'s new diff-distribution
figure, whose R^2 panels showed a `1e28`-scale y-axis for baseline (not
sensory). At least one of the 88 full-pool baseline-window sessions hit
the same per-fold near-constant-column pathology (a raw, non-subtracted
window CAN still be exactly constant across an entire session for a
genuinely silent neuron, just much less often than after subtraction).
Both full-pool sensory and baseline windows were rerun with the fix
(archived: `*.sensorywindow.PRE_SCALEFLOORFIX.csv` / the pre-fix
baseline default CSV, kept for reference, NOT for reuse). Reassuring
finding: Pearson/Spearman were essentially UNCHANGED by the fix in both
reruns (e.g. falsealarm_curve pearson 0.710 vs pre-fix 0.717) --
confirming those metrics were never actually corrupted, only R^2 was
(consistent with R^2 being scale-sensitive and Pearson/Spearman being
scale-invariant). `054`'s figures regenerated with the corrected full-
pool data; baseline R^2 still shows some genuine (not-bugged) extreme
outlier sessions post-fix (e.g. one session's falsealarm_curve R^2 ~+45,
another's performance_curve ~-30) -- real instability, not the numerical
artifact, consistent with R^2 being the wrong metric for this window
regardless.

**Lesson for future perfquant work (or any per-fold-CV regression on
high-dimensional neural features)**: always use a floored/robust scaler,
not vanilla `StandardScaler`, whenever `p` (features) is large relative
to `n` (trials) within a fold -- near-constant-within-a-fold columns are
a real, silent failure mode, and R^2 will not visibly error out, it will
just report an absurd number that's easy to mistake for "the model is
just bad" rather than "the pipeline has a bug." Pearson/Spearman are much
more robust to this specific failure and should be the first cross-check
when R^2 looks unstable.

**Not yet done**: PLS (partial least squares) or univariate pre-selection
as overfitting mitigations, discussed as better-motivated than plain PCA
given the concern that real signal may be concentrated in low-variance
neurons PCA would discard; per-area (rather than per-neuron or area-
averaged-whole-brain) breakdown of the single-animal result; extending
`FEATURE_MODE="corrected"` to the full 88-session pool now that the fix
is verified; `047`-style behavior correlations for any of this.
only -- nothing here should be read as a production result.

## STATUS RECAP (2026-09-23, written on a status check; covers the 2026-09-21/22 gap)

Everything between the `2026-09-20` sections above and today was missing
from this file. Verified live today, not copied from the entries above.

**Nothing is running.** The haas queue for this project is fully drained:
`~/haas_scheduler_top_lickshiftnull.log` (last write 2026-09-18 13:33)
ends `DONE -- modality_lick shift-null redo (learning + expert) complete,
expert-stage hitmiss/modality (whole_brain+area_group) complete, 046
perf-state baseline (expert) complete`, and `pgrep -af
'haas_remote_scheduler|024_master_sweep|046_perfstate'` on haas returns
nothing (checked 2026-09-23). The only python on haas is an unrelated
`unit_spikes_analysis.py` job. So the whole expert-stage sequence that the
2026-09-15 "HOW TO RESUME" section lists as ON HOLD and the 2026-09-17
recap lists as QUEUED **has since run to completion** -- both of those
sections are stale on that point; this one is current.

**GAP: three finished remote outputs were never pulled to local**, so no
figure/stat script has ever seen them (they exist only on haas, computed
2026-09-18):
- `024_master_results_modality_stim_expert_area_group.parquet` (6.3 MB)
- `024_master_results_modality_lick_expert_area_group.parquet` (5.1 MB)
- `046_perfstate_baseline_decoding_results_expert.csv`
Local `exploratory-analyses/` has every other tag (newest local parquets
2026-09-18 10:26). Pulling these three is the first concrete to-do, and
per memory `ssl_wholebrain_all_figure_types_together` each one then needs
EVERY applicable figure/stats script regenerated in one pass, not just one.

**Perfquant branch (`049`-`068`) is now documented in its own two files,
not here**: `METHOD_perfquant_decoding.md` (method) and
`RESULTS_RECAP_perfquant.md` (results record, `049`-`068`). Read those
rather than the perfquant fragments scattered above, which predate the
method being locked. Consequences for anyone reading this file cold:
- **PLS+1SE is the adopted estimator** for perfquant regression. The
  "Not yet done: PLS ..." line at the end of the 2026-09-20 section above
  is therefore OBSOLETE -- PLS was done (`055`/`056`), won, and is the
  working method. dPCA and CEBRA were evaluated (`064`) and NOT adopted.
- Canonical full-pool validation is `058` (1000-shuffle null, R2/Pearson/
  Spearman, Wilcoxon): all 9 target x metric combinations reject the null
  at group level. Sensory window is canonical (`062`); no R+/R- cohort
  effect anywhere (`063`, 1/27 borderline).
- Area-level perfquant (`066`-`068`, 2026-09-21/22) is the newest work:
  per-`area_group` ranking + per-area Wilcoxon/BH-FDR, 39/45 tests
  significant, Midbrain/Striatum/Thalamus/Motor top for `whisker_curve`.
  Recorded in `RESULTS_RECAP_perfquant.md` §9 (added 2026-09-23).

**Last actual activity on the time-resolved (non-perfquant) side**:
2026-09-21 21:43-22:17 and 2026-09-22 09:23-09:26 -- `034` regenerated
with the first-half desaturation retrofit, and new
`034b_area_condition_megagrid.py` (one combined grid over all regular
`area_group` tags x half-condition x cohort). `perfstate_stim_area_group`
is excluded there by design (cancelled project-wide, superseded by `046`).

**Open items (superseding the older "not done" lists where they conflict)**:
1. Pull the three expert `area_group`/`046` outputs above from haas, then
   regenerate all applicable figures for those tags.
2. `046` expert-stage perf-state figure (raw + above-null + cohort-split,
   matching the learning-stage one) -- still doesn't exist.
3. Perfquant: pseudopopulation/multi-session pooling (repeatedly flagged
   as the single biggest untried lever); `FEATURE_MODE="corrected"` is
   now full-pool done (`062`), so that older to-do is closed.
4. Still deferred, unchanged: LaTeX report refresh; `042` latency method
   (three candidates, undecided); `047` not extended to expert stage or
   beyond `whole_brain`; MH038's 0-whisker-trial mismatch logged not
   debugged.

## 2026-09-23 (continued): pulled the 3 stranded expert outputs, regenerated all applicable figures

Closed item 1 from the recap above. All three haas outputs (computed
2026-09-18, never pulled) scp'd to local `exploratory-analyses/`
(rsync isn't installed in this Windows git-bash, used `scp` instead --
byte-size-verified against the remote copies): `024_master_results_
modality_stim_expert_area_group.parquet`, `024_master_results_modality_
lick_expert_area_group.parquet`, `046_perfstate_baseline_decoding_
results_expert.csv`, plus their two `024_bin_edges_*.json` companions
(needed by `032`/`034`/`042`, not explicitly listed in the earlier recap
but required to plot).

**Regenerated, one pass, all applicable scripts for the two new
`modality_*_expert_area_group` tags** (per
`ssl_wholebrain_all_figure_types_together`): `032` (12 pngs/tag, RT/stim-
time cache built fresh -- must run from repo root, `REF_PATH` is relative),
`034` (12 pngs/tag, both cohorts + comparison + condition grids; modality
tags have no shift-null so fall back to real-minus-0.5 as documented),
`042` (per-area latency, 3 methods x mean/per-mouse, 4 outputs/tag). All
under `figures/area_group/expert/`, confirmed by mtime, none skipped/
errored (the statsmodels "covariance of constraints... rank" warnings in
`034`'s ANOVA post-hoc are cosmetic, same as always). Then re-ran `034b`
(combined megagrid across all `area_group` tags) so it now includes these
two instead of skipping them -- 8 rows across 6 tags. **Note for next
time**: `034b`'s real output is `figures/area_group/034b_area_condition_
megagrid.png`, NOT the top-level `034b_area_condition_megagrid.png` from
2026-09-22 that's still sitting in `exploratory-analyses/` -- that stale
top-level copy predates the script's move to the `figures/` subdir
convention and should not be read as current; did not delete it (out of
scope for a status-check task) but flagging so it isn't mistaken for live.

**046 expert-stage figure -- built, not just regenerated** (closes recap
item 2): `046_perfstate_baseline_decoding.py` has no plot-only path (`main()`
always recomputes the full per-session decode before plotting), and
recomputing would both duplicate haas's already-complete 2026-09-18 run and
violate `ssl_prefer_haas_compute`. Added `069_perfstate_baseline_expert_
plot.py` -- lifts `046`'s plotting block verbatim, reads the pulled
`046_..._results_expert.csv` directly, day_stage hardcoded to "expert".
Output: `046_perfstate_baseline_decoding_expert.png` (top-level, matches
`046`'s own convention). Result: 56/56 (session, window) cells decodable,
quiet_window real=0.844/null=0.826, ITI real=0.839/null=0.820 -- same
"raw accuracy dominated by slow drift, above-null diff is the real signal"
pattern as the learning-stage figure; per-cohort panels included.

**Not touched this pass** (out of scope for "regenerate applicable
figures" -- no code/methodology change, just newly-arrived data): report
HTML, LaTeX report, `RESULTS_RECAP_perfquant.md` (unrelated branch).

## 2026-09-23 (continued): expert-stage PLS+1SE perfquant launched (500 shuffles)

User question ("Also on expert sessions? Run the 500 null shuffles.")
surfaced a real scope gap: `058` (the canonical full-pool PLS+1SE
validation) and its area-level follow-ups `066`-`068` are ALL filtered to
`hm[hm["day_stage"] == "learning"]` -- expert-stage perfquant regression
had never been attempted. Per [[ssl_scheduler_expert_stage_last]] this
would normally queue after everything else, but the haas queue is fully
drained (verified before launching), so it's the only job running -- no
interleaving violation.

**Coverage checked on haas before launching** (the `load_whisker_curve_row`
day>0 generalization landed 2026-09-19 but its own docstring said "not yet
for the MH-cohort ephys mice this project's neural decoding actually runs
on" -- needed to verify that's still true): of 29 has-ephys expert-stage
sessions, **24/29 have a real day>0 curve file** -- MH-cohort coverage
caught up since that docstring was written, so this is a genuinely viable
run, not a near-empty one. 5 missing (MH020, MH037, MH064, +2).

**Added `070_perfquant_expert_pls_nulldist_fullpool.py`** -- `058` verbatim
(same PLS+1SE recipe, sensory window, floored scaler, full retrain-under-
linear-shift null, R2/Pearson/Spearman) with two changes: `day_stage==
'expert'` session filter, and **`N_SHUF_NULL=500`** (user's explicit number
for this run, not 058's 1000). Outputs get a `070_perfquant_expert_*`
prefix so they never collide with 058's learning-stage files.

**Running now on haas**, launched 2026-09-23 ~09:07 (PID 2700935 + 48
workers, `nohup`'d after fixing a first-attempt relative-path crash --
must run from `~/code/ibl-ai-agent` repo root, not from inside
`exploratory-analyses/`, same gotcha as every other script here). Log:
`070_perfquant_expert_log.txt`. First 3 sessions already showing sane
real-vs-null pearson separation (e.g. MH001: real 0.41-0.57 vs null
0.21-0.44). A live Monitor is armed on this session watching for
`DONE_058`/errors in that log -- check `070_perfquant_expert_log.txt`
directly if picking this up cold instead.

**Once it lands**: pull `070_perfquant_expert_pls_nulldist_fullpool.csv`
+ 3 pngs to local, and note in `RESULTS_RECAP_perfquant.md` whether the
learning-stage headline (all 9 target x metric combos reject the null,
§2) replicates at expert stage -- worth flagging if not, since a
learning-stage-only effect would itself be a finding (task already
mastered, no more within-session performance drift to decode).

## 2026-09-23 (continued): expert-stage PLS+1SE run DONE, results pulled

`070_perfquant_expert_pls_nulldist_fullpool.py` finished on haas (~5 min
runtime, 23/29 sessions usable), results + 3 figures + log pulled to
local. **Headline: replicates at expert stage** -- all 9 target x metric
combinations reject the null (Wilcoxon p 2.4e-7 to 0.023), above-null
margins comparable to (slightly larger than) learning-stage. Full writeup
in `RESULTS_RECAP_perfquant.md` §10. Practical reading: within-session
behavioral-state decodability isn't learning-specific -- it's present
whether or not the mouse is still learning, which argues the effect is
more "slow within-session performance drift" than "a learning-process
signal." Closes the gap this section's earlier entry flagged.

## 2026-09-23 (continued): within-mouse condition x cohort test (`071`)

User request: "Test within-mouse difference to mean of null dist across
conditions and cohorts." Added `071_perfquant_withinmouse_condition_
cohort_test.py` -- combines `058` (learning) + `070` (expert), averages
sessions within mouse (real averaging only at expert stage, some mice
have 2 sessions), Wilcoxon + t-test per (day_stage, cohort, target),
BH-FDR across the 12 cells. Explicit mouse-level unit of analysis here
per the user's own wording, overriding [[ssl_stats_unit_of_analysis]]'s
session-level default for this one request.

Result: 11/12 (t-test) / 10/12 (Wilcoxon) cells significant at q<0.05.
Learning stage: 6/6, all q<1e-5. Expert stage: R+ significant on all 3
targets; **R- is underpowered (n=6 mice)**, only whisker_curve reliably
clears both tests -- effect sizes are comparable to other cells though,
so this reads as low n, not a real cohort difference. Full writeup
`RESULTS_RECAP_perfquant.md` §11.

## 2026-09-23 (continued): caveats/CV write-up + fuller figures for `071` (`072`)

Answered "any caveats / how was CV stratification done" and added
`072_perfquant_withinmouse_condition_cohort_figures.py` (per-mouse strip
plot + 3-metric heatmap, `071`'s own bar chart only showed Pearson means).
Full write-up in `RESULTS_RECAP_perfquant.md` §12: CV is plain unstratified
KFold throughout (continuous target, per-session model, no mouse-group
CV needed); headline caveat is expert-R- (n=6) being underpowered, not
null -- effect sizes there are as large as significant cells, visible in
the new heatmap.

## 2026-09-23 (continued): neuron-count control -- design decided, part 1 launched

User request: control for N units per recording/area, applied to both
the time-resolved decoding pipeline and the PLS/perfquant pipeline.
Clarified via AskUserQuestion (both recommended options chosen):
(1) PLS stays single-window (no time axis added -- just accuracy vs N at
the existing sensory window), (2) start with a 5-example-session pilot
(reusing `068`'s `EXAMPLE_MICE` picks), not the full pool.

**`073_hitmiss_timeresolved_nsubsample_control.py`** -- hitmiss_stim_
whole_brain, exact `024_master_sweep.py` recipe (causal 50ms bins, 5ms
stride, `WIDE_DEAD_ZONE`, `select_fixed_c_pooled`/`decode_curve_pooled`),
subsampled to N=100,200,...,1000 (+ true full count as an endpoint;
example sessions range 687-2704 whole_brain units), 3 draws/N averaged.
**Real accuracy only, no null** -- deliberate cost-saving choice, this
control asks how the score scales with N, not significance at every N.
Launched on haas (PID 2739510), monitor armed.

**`074_perfquant_nsubsample_control.py`** -- same 5 sessions, `058`'s
exact PLS+1SE recipe (sensory window, floored scaler, `select_pls_
components`, pooled-CV), same N grid, same no-null design. Written,
pushed to haas, **not yet launched** -- queued to run after `073`
finishes (this project's "run sequentially, not concurrently" convention,
`024`'s own docstring lesson about mutual slowdown).

## 2026-09-23 (continued): both N-subsample controls DONE, results pulled -- asymmetric finding

`073` (hitmiss, time-resolved) and `074` (PLS/perfquant) both finished on
haas, results pulled to local. **The two pipelines answer differently --
important, not a wash:**
- `073` (hitmiss): accuracy SATURATES by N~300-500 for every session, and
  peak accuracy does NOT track total unit count (MH011, smallest pool of
  the 5, has the 2nd-highest peak). Evidence AGAINST a unit-count
  confound for hitmiss decoding.
- `074` (PLS/perfquant): test Pearson does NOT clearly saturate for most
  sessions/targets within the tested range (several still rising at
  N=1200-2700). Evidence THAT REINFORCES the area-level unit-count
  caveat from §9 rather than resolving it.

Full write-up `RESULTS_RECAP_perfquant.md` §13/§14. Practical
implication: the covariate-adjustment / unit-matched-subsampling fix
discussed with the user before building this control is now MORE
warranted for the PLS area-level ranking specifically, not just
theoretically possible -- not yet built, next natural step if the user
wants the area ranking corrected.

## 2026-09-23 (continued): PLS N-control redone WITH null (`075`)

User follow-up on `074`: "Do the same perf quantification on the PLS
test data with R2 over null R2" -- `074` deliberately had no null;
`075_perfquant_nsubsample_r2_null_control.py` adds it back. Same 5
sessions/N-grid, but restructured to parallelize at (session, N)
granularity (51 cells) instead of per-session, since null shuffles
multiply cost a lot -- up to 50-way concurrent on haas instead of 5-way.
`N_SHUF_NULL=30` (not 058's 1000 -- this sweep touches ~150 (session, N,
draw) cells, scaled down accordingly, see script docstring). Launched,
monitor armed.

**Early signal (MH011, finished first)**: raw test R2 climbs with N as
`074` already showed, but NULL R2 climbs right along with it -- above-null
R2 for whisker_curve stays roughly flat (~0.22-0.24) across N=100 to 687.
If this holds across sessions, it meaningfully refines `074`'s reading:
the unit-count sensitivity may be mostly in the null's own capacity to
fit noise (more features = more overfitting room for a null with no real
signal too), not necessarily in the REAL above-null effect. Not yet
confirmed across all 5 sessions -- full write-up once `075` completes.

## 2026-09-23 (continued): `075` done, corrects the `074` reading

`075` finished (51/51 cells), pulled locally. **Corrects the earlier
`074`-based conclusion**: test R2 and null R2 climb together with N, so
above-null R2 is roughly flat vs N for most sessions/targets -- the
area-level unit-count worry (§9) is less supported than the `074`-only
reading suggested, since PLS's apparent N-sensitivity was substantially
a null-inflation effect. Full correction in `RESULTS_RECAP_perfquant.md`
§16. `073` (hitmiss)'s "not unit-count-limited" conclusion stands
unchanged -- it already accounted for the shape of the real curve alone
being saturating, no null needed to make that read.

## 2026-09-23 (continued): 065 updated to cover expert stage too

User: "Update session figures" -- clarified via AskUserQuestion to mean
`065_perfquant_allmice_predictions_with_null.py` (all-mice true-vs-
predicted + qualitative null band, one figure per target). Extended to
`day_stage='expert'` (didn't exist when 065 was first written, predates
`070`) -- `learning_category` confirmed populated for expert-stage rows
too (18 good/6 moderate/5 bad of 29), so the same good/moderate filter
applies unchanged. 76 learning + 24 expert = 100 sessions total, running
on haas (PID 2784347), monitor armed. Outputs get the standard `_expert`
suffix; existing learning-stage filenames unchanged so this is a
refresh + addition, not a breaking rename.

## 2026-09-23 (continued): 065 -- square panels, added a results cache

User: "Make figures, make them squares" (re: `065`). Added
`ax.set_box_aspect(1)` + square-ish figsize per panel. Also added a
pickle cache (`065_perfquant_allmice_cache.pkl`) of the full per-session
arrays (Y_true/Y_pred/null_mean/null_std/trial_index) -- discovered this
purely cosmetic change would otherwise force a full ~100-session PLS+null
recompute, since only the pearson summary was ever persisted before. Any
future cosmetic-only change to this figure now reads the cache instead.
Let the original (non-square, no-cache) run finish first rather than
kill it mid-flight -- its 6 figures (3 targets x learning/expert) are the
"content-correct" versions; relaunched immediately after with the squared
+cached script, running now on haas, monitor armed.

## 2026-09-23 (continued): 065 -- null-performance annotation added, cache paid off immediately

User: "On this figures, also show the mean performance of the null."
Added each panel's null Pearson to its title (`r=X.XX (null=X.XX)`),
sourced from `058`/`070`'s CSV (`null_pearson_mean`, the VALID properly-
scored null) via a (session_id, target) join -- deliberately NOT derived
from this script's own real-axis-remapped null band, which the docstring
already flags as qualitative/inflated. Relaunched from the
`065_perfquant_allmice_cache.pkl` built two steps ago (for the square-
panel change) -- finished in ~10s instead of another ~15+ min recompute,
exactly the payoff the cache was built for. All 6 figures (3 targets x
learning/expert) regenerated, squared panels + null annotation both
present. n=98/100 sessions total (76 learning, 23/24 expert -- one
expert session dropped, consistent with the known small attrition from
missing curve files/trial-count mismatches already documented for `070`).

## 2026-09-23 (continued): single-trial face-validity check -- strong result

Built `077_perfquant_singletrial_facevalidity.py` per the agreed plan:
reused `065`'s cache (no PLS recompute), correlated single-trial `Y_pred`
against RT and previous-trial outcome, per-session r then group-level
Wilcoxon+t-test (avoids the trial-autocorrelation p-value trap). Ran
locally (cheap, no haas needed -- confirmed local dataset access works
for `trials_tbl`).

**Result: 17/24 cells significant, and EVERY cell has the theoretically
expected sign** -- RT negative everywhere (faster response = higher
predicted state), prev-outcome positive everywhere (recent hit = higher
current predicted state). Non-significant cells are all in the same
underpowered expert-R- (n=6) territory already flagged in §11, still
correctly signed. Full write-up `RESULTS_RECAP_perfquant.md` §17 --
directly answers "does the model make sense": yes, its single-trial
predictions track independent behavior (not derived from spikes or the
target curve itself) in exactly the direction you'd expect.

## 2026-09-23 (continued): cross-mouse session-permutation nulls (`079`) -- pilot running; `078` killed

User proposed two new Harris-style nulls for perfquant: **R- null**
(recipient R+ mouse's own target Y, donor R- mouse's neural activity X
-- tests reward-dependence) and **within-R+ null** (recipient's own X,
donor OTHER R+ mouse's target Y -- tests individual specificity).
Confirmed mechanics and alignment scheme with the user before building:
"whisker-stim-count" alignment collapses to plain whisker-trial-index
truncation in this pipeline (X/Y already built on the whisker-trial-only
subsequence, no interpolation needed).

Built `079_perfquant_crossmouse_nulls.py` -- pilot scope: 5 R+ learning-
stage recipient sessions (top by `065`'s whole_brain test_pearson: MH070,
MH069, MH031, AB162, AB087) x ALL learning-stage donors (37 R-, 33 other
R+) x all 3 targets (free, joint PLS fit) = 355 cells. Per recipient x
null_type x target: corrected R2 (real - median(donor)) + rank-based p
(fraction of donors >= real).

**`078` (area-overlay single figure) killed mid-run** (user: "Do 79 now,
forget about 78 for now") -- no partial output was ever saved (078 only
writes its cache after the full sweep), so nothing lost, just needs a
fresh relaunch whenever it's picked back up.

`079` launched on haas (102 workers), monitor armed. Known limitation
carried into the writeup (user's own caveat): controls for generic drift
(R- null) and generic "reward introduced" dynamics (within-R+ null), NOT
for other R+/R- group differences (implant, cohort/batch) that could
inflate the R- null itself -- §4/§11's "no accuracy difference between
cohorts" is mild reassurance, not a guarantee.

## 2026-09-23 (continued): 079 crashed, fixed, relaunched

First launch died after 2/355 cells: `UnboundLocalError: cannot access
local variable 'np'` in `process_pair` -- a redundant `import numpy as np`
placed AFTER two earlier `np.where(...)` uses in the same function made
Python treat `np` as local to the whole function (classic scoping trap),
shadowing the working module-level import. `main()` had no try/except
around `fut.result()`, so the first failure killed the entire job
silently (workers stayed listed briefly during teardown, which is what
made it initially look like slow-but-alive rather than crashed).

**Fixed**: removed the misplaced inner import; added a try/except around
`fut.result()` so one bad cell can never again kill the whole run
(matches `066`'s own "one bad combination" resilience convention) --
failures now get counted and the first 5 logged, not silently fatal.
Verified the fix locally on 3 hand-picked cells (real, R- null,
within-R+ null) before relaunching -- all returned sane values
(MH070 real pearson ~0.94/0.95/0.84).

Also found (separately, not a bug in my job): haas load average was
154-185 on 112 cores from an unrelated `unit_spikes_analysis.py
--analyses-single roc_analysis` job the user has running -- confirmed
with the user, not touched, told to "keep going" alongside it.

Relaunched (`079_log_v2.txt`, PID 2827344, 102 workers), monitor armed.

## 2026-09-23 (continued): 079 DONE -- results in, second bug found+fixed

Compute finished cleanly (355/355 cells, 0 exceptions -- the earlier fix
held). But `main()`'s own summary/plotting code then crashed
(`row[metric][k]` on a pandas itertuples namedtuple -- doesn't support
string-key indexing) AFTER the expensive part was already safely on disk
(`079_perfquant_crossmouse_nulls_raw.csv`). Built `079b_perfquant_
crossmouse_nulls_summary.py` to rerun just the cheap reporting step
locally from that CSV -- found and fixed a THIRD bug in the process: the
CSV's stringified numpy-scalar list columns ("[np.float64(0.93), ...]")
broke a naive float-regex parser, which matched the literal "64" inside
"float64" as a spurious extra number, corrupting 2 of every 3 parsed
values and producing impossible |r|>1 "correlations" in the first pass.
Fixed by stripping the `np.float64(...)` wrapper before parsing.

**Result** (pilot, 5 recipients, not yet a group-level claim): R- null
(reward-dependence) beaten in 15/15 cells, several at the resolution
floor (p~0.026, real beat every one of 37 donors). Within-R+ null
(specificity) beaten in 12/15 cells but with visibly weaker separation
(donor violins reach 0.8-0.9) -- **AB162 is a genuine exception**, all 3
targets non-significant there, its own activity predicts OTHER R+ mice's
curves nearly as well as its own. Full writeup `RESULTS_RECAP_perfquant.md`
§18. Three real bugs found and fixed across this one build -- all now
documented in the scripts' own code comments so they don't recur.

## 2026-09-23 (continued): full cross-mouse sweep -- 2x2 design, precompute+sweep split for speed

User: "No [to frozen-weight transfer], do the full sweep of cross-animal
tests, optimize for speed" + confirmed adding the within-R- null (user's
own follow-up question "Should there be within R- null too?"). Full
design is now a proper 2x2: R+ recipient x {R- donor (`rminus_null`),
other-R+ donor (`withinrplus_null`)} and R- recipient x {R+ donor
(`rplus_null`, NEW), other-R- donor (`withinrminus_null`, NEW)}. All 75
learning-stage sessions (38 R+, 37 R-) as recipients this time, not just
5 -- ~5600 (recipient, donor, null_type) cells.

**Speed optimization**: split into `080` (precompute each session's
(X, Y_raw) ONCE, cached) + `081` (pure array-slicing + PLS fit against
the cache, no raw-data reload per pair -- would have been the dominant
cost otherwise, since a session can be a donor 70+ times). Caught a
SECOND speed trap before launching: passing the cached arrays as
`ProcessPoolExecutor` task arguments would force re-pickling/piping them
on every task -- fixed by loading the cache into a module-level global
BEFORE creating the worker pool, so forked workers inherit it
copy-on-write for free; tasks now carry only session-ID strings.

Smoke-tested the global-cache pattern locally (3 cell types, all OK)
before touching haas, per the lesson from `079`'s two bugs. `080`
launched on haas, running fast (13/75 in ~10s), monitor armed. `081`
queued to launch once `080`'s cache is ready.

## 2026-09-23 (continued): full cross-mouse sweep DONE -- weaker/more mixed than the pilot

`081` finished (5625/5625 cells, 0 errors, <1 min compute thanks to the
precompute+cache redesign). Group-level summary (`081b`) built. **Full
sweep result is notably weaker than the 5-recipient pilot** -- only 5/12
cells survive Wilcoxon+BH-FDR, 1/12 by t-test. R+ reward-dependence
holds for all 3 targets (Wilcoxon only, modest/skewed); R+ specificity
holds ONLY for whisker_curve; R- mirrored reward-dependence only
marginal; **R- specificity is completely null** (the new within-R- test)
-- a clean, informative negative control showing individuality isn't
just an any-animal property. Full writeup `RESULTS_RECAP_perfquant.md`
§19, explicitly flagged as revising/superseding §18's pilot-based
optimism (selection bias from testing only top-decoding sessions).

## 2026-09-23 (continued): RRR + LG-AR1 pilot -- new decoding approach from user spec

User provided a full spec for an alternative decoder: single-session
reduced-rank regression (RRR, per-trial spike TIME COURSE -> single-trial
estimate, not one aggregated window) + LG-AR1 (Kalman smoother across the
trial sequence, exploiting trial-to-trial persistence). Explicit caveat
from the user: the smoother can't distinguish real drift from nuisance
drift, so nulls must go through the FULL pipeline, not just an unsmoothed
baseline comparison.

Built `082_rrr_lgar1_pilot.py`, 3 sessions (MH070/MH069/MH031, same top
R+ recipients as `079`/`081`), all 3 targets, whole_brain. **Two
deviations from the user's sketch, both necessary**: (1) dual-ridge
(Woodbury) reformulation of the alternating-ridge RRR fit -- the sketch's
primal U-step solves an N*R-dim system (~1200-7500 at whole_brain scale),
infeasible at the scale needed here; the dual form solves a K-dim system
(K=n_trials~100-250) instead, identical solution, tractable. (2) LG-AR1
parameters fit by direct marginal-likelihood maximization
(`scipy.optimize`) rather than EM -- same objective, simpler to implement
correctly under this pilot's time budget.

Null: linear-shift null (this project's standing convention) applied
BEFORE every step of the pipeline (RRR out-of-fold CV fit AND LG-AR1
fit AND smoothing) -- exactly the user's caveat, implemented literally,
not compared against an unsmoothed baseline.

Smoke-tested locally first (1 session, shrunk grid, N_SHUF=2) before
touching haas -- worked, and already showed the expected nuance: LG-AR1
smoothing raised BOTH the real score (0.95->0.97) AND the null
(0.70->0.82), so above-null gain from smoothing was actually SMALLER
than RRR-only's above-null gain in that one example -- a genuine
illustration of the user's own caveat, not a bug.

Full run (all 3 sessions x 3 targets, N_SHUF_NULL=20, full hyperparameter
grid) launched on haas, parallelized across the 3 sessions. Monitor
armed.

## 2026-09-23 (continued): RRR+LG-AR1 pilot DONE -- informative negative result on smoothing

`082` finished cleanly, 9/9 cells. **RRR alone**: real above-null
decoding in all 9 (session x target) cells, same order of magnitude as
this project's own PLS+1SE numbers despite a totally different
architecture -- independent cross-method confirmation of decodability.
**LG-AR1 smoothing, null-controlled through the full pipeline as the
user specified**: DECREASES the above-null margin in all 9/9 cells, not
just fails to help -- smoothing raises both real AND null scores
together (fitted rho extremely high, 0.97-0.99, implying an effective
smoothing window comparable to or longer than the whole session). Full
writeup `RESULTS_RECAP_perfquant.md` §21, incl. a known cosmetic issue
in the figure (smoothed trace plotted on the raw latent scale, not
affine-corrected -- doesn't affect any reported correlation, which is
scale-invariant).

## 2026-09-23 (continued): matched PLS+1SE vs RRR (+/- LG-AR1) comparison

User: "Yes use the same data, it should be the baseline ITI data actually,
not the sensory window, Make it comparable" -- following up on the
"are these comparable / same input data?" question (answer was no).
Built `083_pls_rrr_lgar1_comparable.py`: SAME window for both
(`WIDE_BASELINE_WINDOW=(-2.0,-0.01)`, this project's own canonical
baseline window from `062`/§3 -- the harder, more autocorrelated test,
chosen deliberately over sensory), SAME feature scaling (added the
floored StandardScaler to RRR, matching PLS's existing `052`-bug-fix
convention -- `082`'s RRR only mean-centered), and LG-AR1 now applied to
BOTH decoders' outputs (previously only paired with RRR). RRR still gets
T=20 time-resolved bins across the same window (that IS its structural
point); PLS still aggregates to one flat mean (that IS its structural
point) -- deliberately NOT forced to match, since collapsing that
difference would defeat the comparison's purpose.

Smoke-tested locally (1 session, shrunk grid/null) before touching haas
-- worked; early numbers already interesting (PLS beat RRR on this
window in the smoke test, LG-AR1 reduced above-null for both again,
consistent with `082`'s finding). Full run (3 sessions x 3 targets,
N_SHUF_NULL=20) launched on haas, monitor armed.

## 2026-09-23 (continued): matched comparison DONE -- PLS beats RRR on baseline window

`083` finished, 9/9 cells. Once fairly matched (same window, same
scaling): **PLS+1SE above-null positive in 8/9 cells (mean +0.055); RRR
positive in only 5/9 (mean -0.014)** -- RRR does not show an advantage
here, contrary to what an unmatched comparison might have suggested.
RRR+LG-AR1 is highly volatile (one huge win +0.227, several losses down
to -0.187) vs PLS+LG-AR1 staying close to PLS alone. One confound NOT
resolved: RRR's plain CV-best hyperparameter selection is more liberal
than PLS's 1-SE rule -- can't yet separate "RRR architecture underperforms
here" from "RRR's regularization is less conservative." Full writeup
`RESULTS_RECAP_perfquant.md` §22.

## 2026-09-23 (continued): IMPORTANT CORRECTION -- §22's "PLS beats RRR" was an under-regularization artifact

User: "What are the models. Plot the curves" -- extended `083` to save
and plot the actual per-trial prediction curves (not just summary bars),
and affine-corrected the LG-AR1 smoothed curves back to observation
scale (fixing §21's known cosmetic issue) while doing so.

**The curves figure revealed a real problem, not a display issue**: RRR's
raw held-out predictions reached ~250-300x the standardized target's own
scale for at least one session (MH070) -- PLS and true stayed near 0-1,
RRR spiked to 300. Checked the CSV: the selected lambda for that cell was
100.0 -- the OLD `LAM_GRID`'s maximum. **The hyperparameter search had
been hitting its own ceiling**, meaning it never found a genuine interior
optimum -- with R up to 3 and N up to ~2000+ units, the U-step fits up to
~R*N~6000 parameters from ~80 training trials/fold, and under-
regularization there doesn't just overfit, it destabilizes the
alternating-ridge iteration on held-out folds.

**Fixed**: widened `LAM_GRID` from `(3,10,30,100)` to 9 values up to
30000; added an automatic warning when the selected lambda hits the
grid's max (so this doesn't silently recur). Smoke-tested on the known-
bad cell: lambda moved to 1000 (a genuine interior optimum, not the new
30000 ceiling), outlier range shrank substantially, and **RRR's above-
null score flipped from -0.067 to +0.420** for that one cell alone.

**This means §22's "PLS beats RRR on the baseline window" conclusion was
likely substantially an artifact of RRR being under-regularized, not a
real finding about the methods** -- §22 needs a real correction, not
just an addendum, once the full rerun (in progress, monitor armed)
completes. Do not cite §22 as-is until this is resolved.

## 2026-09-23 (continued): correction confirmed -- RRR actually wins, once properly regularized

Widened-grid rerun finished, ZERO grid-ceiling warnings this time (every
cell found a genuine interior lambda). **Corrected result: RRR beats PLS
in 8/9 cells, mean above-null RRR=+0.291 vs PLS=+0.080** -- the exact
opposite of §22's original conclusion, which is now formally superseded
by §23 (not just caveated). §22 stays in the doc as a record of what was
initially (incorrectly) concluded, per this project's append-don't-
rewrite convention, but should not be cited on its own.

Separately, the curves also revealed a real (non-bug) property: RRR's
raw predictions are well-correlated with truth but poorly scale-
calibrated (magnitude much larger than the target's own range) --
correlation is scale-invariant so this doesn't affect the validity of
the above-null numbers, but would need a simple post-hoc rescaling step
to fix for visualization purposes. Not yet done. Full writeup
`RESULTS_RECAP_perfquant.md` §23.

## 2026-09-23 (continued): scaling problem fully fixed

Two-layer correlation-preserving OLS calibration (`calibrate_to_target`)
applied to raw AND smoothed curves in `083`. Figure now fully
interpretable, all curves on comparable scale to true, no change to any
above-null number (mathematically guaranteed, verified numerically).
Added a results cache so future cosmetic changes to this figure are
free. Full writeup `RESULTS_RECAP_perfquant.md` §24.

## 2026-09-23 (continued): annotated curves + paired comparison figure (`084`)

User: "Make a figure and annotate curve figures." Built `084_pls_rrr_
annotated_and_paired.py`, pure post-hoc from `083`'s cache/CSV (no
compute). (1) Re-plotted the 9-panel curves grid with each panel's title
now stating PLS/RRR real-r, null, and above-null directly, plus a
[wins] tag -- fixed a first-pass title-overflow bug (text bled into
neighboring panels) by switching to 3 short lines instead of one long
one. (2) New compact paired-line figure: one connected PLS->RRR line per
(mouse, target) cell, color by target, solid=RRR wins/dashed=PLS wins --
the whole 9-cell comparison table as one glance-able figure (8/9 lines
rise sharply, 1 nearly flat).

## 2026-09-23 (continued): PLS-only figures (RRR shelved for now)

User: "Forget about RRR for now, update figures with the PLS+1SE
results" -- RRR's above-null advantage remains an open question (is it
real specificity or a hyperparameter-reuse artifact in the null? not yet
resolved, see prior discussion). Built `085_pls_only_curves_and_summary.py`,
pure post-hoc from `083`'s cache/CSV (no compute): (1) 9-panel annotated
curves, PLS+1SE + PLS+LGAR1 only, RRR dropped entirely; (2) paired
PLS-vs-PLS+LGAR1 line plot. Confirms §21/§23's earlier finding directly:
PLS+1SE tracks true closely in every panel; LG-AR1 helps only 2/9 cells
(mean above-null 0.065 -> 0.056, small net negative).

## 2026-09-23 (continued): results/figures copied to shared drive

User: "Copy all these results and figures in analysis/combined_results_ks4"
-- scope clarified as ALL decoding analyses + perf-state decoding +
PLS regression work (broader than just today's session). Copied to
`M:\analysis\Axel_Bisi\combined_results_ks4\ssl-whisker-hitmiss-
timeresolved-decoding\` (name confirmed with user -- matches this
project's own folder name for easy traceback; `combined_results_ks4` had
no prior precedent for cross-cutting/non-per-mouse content, previously
100% per-mouse subfolders).

Copied: all `.py`/`.png`/`.csv` under `exploratory-analyses/` (759 files,
~246MB, via robocopy /S preserving subdirectory structure incl.
`figures/`), plus the 4 top-level docs (TODO.md, RESULTS_RECAP_perfquant.md,
METHOD_perfquant_decoding.md, question.md). Deliberately EXCLUDED: large
raw sweep parquets and pickle caches (~283MB combined) -- intermediate
compute artifacts, not "results and figures"; not what was asked for.
Verified file counts/sizes match between source and destination.

## 2026-09-24: higher-res + PDF figures for area_group grid comparisons

User: "For the ANOVA plots, are post-hoc tests done for area_group?" ->
confirmed yes (034's `factor_anova_and_posthoc`/`cohort_anova_and_posthoc`,
BH-FDR-corrected Mann-Whitney U per area, asterisk at q<0.05) -> user:
"Regenerate the figures in higher resolution, also in pdf."

Modified `034_area_window_quant_grid.py`'s shared `savefig_retry` helper
to save both a `.png` (dpi bumped 140->300 at all 8 call sites) and a
`.pdf` (vector) per figure, each with independent retry-on-OSError.
Re-ran for all 6 area_group tags with a computed `024_master_results_*`
parquet (`hitmiss_stim_area_group`, `hitmiss_stim_expert_area_group`,
`modality_stim_area_group`, `modality_stim_expert_area_group`,
`modality_lick_area_group`, `modality_lick_expert_area_group`) from the
repo root. `perfstate_stim_area_group` skipped -- no master-results
parquet exists for it yet (never computed at the area_group level).

Verified: 42 new PNG+PDF pairs in `figures/area_group/learning/`, 36 in
`figures/area_group/expert/`, all with matching timestamps; PNG pixel
size confirmed at the new 300 dpi (e.g. grid_comparison.png now
3083x2540, up from the old 140 dpi size).

## 2026-09-24 (continued): full regeneration, all area schemes and stages

User: "Regenerate absolutely all figures, whole_brain, area_group,
acronym, learning, expert" -- extended the prior area_group-only pass to
every maintained `034_area_window_quant_grid.py` tag. Ran all 15 tags
with a computed `024_master_results_*` parquet, across all 3 area
schemes and both stages:
- area_group: hitmiss_stim(_expert), modality_stim(_expert),
  modality_lick(_expert)
- whole_brain: hitmiss_stim(_expert), modality_stim(_expert),
  modality_lick(_expert), perfstate_stim (learning-only, no expert
  sibling yet)
- area_acronym_custom: hitmiss_stim, modality_lick (learning-only, no
  expert sibling yet)

NOT touched (per established convention, [[034 34b regen]]): the two
old one-off variant tags `modality_lick_whole_brain_20mswindow` and
`_precqualitylabel` -- not part of the maintained tag set.

Verified: 192 fresh PNG+PDF pairs across `figures/{area_group,
whole_brain,area_acronym_custom}/{learning,expert}/`, all matched, all
at the new 300 dpi.

## 2026-09-24 (continued): R+/R- marker restyled to edge-color shading

User: "Redo 034 figures but keep area colors but change circles and
triangles by shades of colors: R+ are the rplus color and R- are rminus
color. Make dots smaller." In the cohort-comparison grids (`plot_
comparison_grid`, `plot_comparison_grid_raw`, `plot_comparison_grid_
abovecontrol`), R+/R- was previously shown as circle-vs-triangle marker
shape (`COHORT_MARKER`); now both cohorts use the same circle marker,
with the marker FACE still the area color and the marker EDGE colored by
`COHORT_COLOR` (#00B400=R+, #C800C8=R-, the established SSL R+/R-
convention already used in every other project script). Added an
`edge_a`/`edge_b`/`edge_width` param to the generic `plot_factor_
comparison_grid` so the unrelated day_stage (learning-vs-expert)
comparison, which reuses the same function, keeps its original
black-edge-for-expert convention unaffected. Also shrank every marker in
this script (`markersize` 6->4, 4->3) per "make dots smaller".

Re-ran all 15 maintained tags (same set as the prior 300dpi/PDF pass):
192 fresh PNG+PDF pairs, all matched, verified visually on
`hitmiss_stim_area_group`'s comparison figure (green edges on R+ dots,
magenta edges on R- dots, area-colored fill, smaller markers).

## 2026-09-24 (continued): label-shuffle-null fallback for modality_stim's abovecontrol figures

User: "Some of them are missing showing no data like 034_modality_stim_
area_group_grid_comparison_abovecontrol.png" -> confirmed expected/
pre-existing (`modality_stim` has no `shift_null_curves`, only
`hitmiss`/`perfstate`/`modality_lick` do -> `compute_shift_null` in
`024_master_sweep.py`), 12 files across 6 modality_stim(_expert) tags x
2 area schemes always empty -> user: "you can do the equivalent which is
accuracy minus chance (they have trial shuffles)".

`modality_stim` rows DO carry a `null_curves` column (label-shuffle
null, decode_target=='modality'/alignment=='stim') in the same
per-shuffle array-of-arrays shape `shift_null_window_mean` already
expects (its own docstring already noted this). Changed `load()` to
fall back to `null_curves` for the `{metric_prefix}_shift` control
column whenever `shift_null_curves` is absent, and to return a 4th
value, `null_kind` ("shift" or "label-shuffle"), so every downstream
figure/stat that consumes that column (abovecontrol grids, cohort
ANOVA, the grey null-reference dot) now gets real per-session-paired
label-shuffle-null data for modality_stim instead of nothing -- and
every user-facing "shift null" text label (chance_note, suptitle,
ylabel, "n/a (no X-null)") is now null_kind-aware so it honestly says
"label-shuffle null" rather than misleadingly implying a shift-null.
The two null types are NOT conflated statistically (shift-null = within-
session temporal-structure control, label-shuffle = trial-identity
control) -- only the plotting/labeling machinery is now generic across
both.

Verified on `modality_stim_area_group`: comparison-abovecontrol figure
now shows real per-area dots and a real cohort ANOVA (previously
"n/a (no shift-null)", fully empty axes), title/ylabel correctly read
"...above its own label-shuffle-null control".

Re-ran all 15 maintained tags: 192 fresh PNG+PDF pairs, all matched.

## 2026-09-24 (continued): synced updated figures to shared drive

User: "Copy the updated figures to combined_results_ks4." Re-ran the
same robocopy sync as before, this time also including `*.pdf` (not
copied in the original pass, since PDF output didn't exist yet) --
`robocopy ... *.py *.png *.csv *.pdf /S /XD __pycache__` from
`exploratory-analyses/` to `M:\analysis\Axel_Bisi\combined_results_ks4\
ssl-whisker-hitmiss-timeresolved-decoding\exploratory-analyses\`.
Robocopy only touches changed/new files, so this picked up every 300dpi
PNG re-render, every new PDF, and the R+/R-edge-color + label-shuffle-
null-fallback changes to `034_area_window_quant_grid.py` itself.

Verified: 192/192 PDFs match between source and destination. PNG count
differs by 6 (642 dest vs 636 local-enumerated source) -- confirmed
this is the pre-existing `_stale_pre_figdir_unification_20260914`
long-path PowerShell enumeration quirk noted earlier this session
(those 6 files exist on both sides; local `Get-ChildItem -Include` just
can't see them due to Windows MAX_PATH), not a real gap.

## 2026-09-24 (continued): non-square figures for area_acronym_custom

User: "For area _acronym_custom figures, make the figures non-square,
slightly rectangular to account for large number of areas on the
x-axis." Root cause: `style_ax`'s `ax.set_box_aspect(1)` forced every
subplot to a fixed square regardless of `figsize`, and every subplot-
creation call used a hardcoded `3.6 * len(columns)` width -- fine for
area_group's 13 areas, illegible for area_acronym_custom's 46-51 (labels
overlapping/unreadable, confirmed visually).

Added `area_fig_width_mult(areas)` (scales the per-column width
multiplier above 3.6 once `len(areas) > AREA_COUNT_SQUARE_THRESHOLD=20`,
unchanged below it) and made `style_ax`'s box_aspect `min(1.0, 20 /
len(areas))` instead of a hard `1` -- the two scale by the same ratio,
so the extra width is actually used by the axes box instead of leaving
whitespace around a still-square plot. area_group (13 areas) and
whole_brain (1 area) are exactly unchanged (width_mult=3.6, aspect=1.0);
area_acronym_custom (46-51 areas) gets ~2.3-2.6x wider, aspect ~0.4-0.43
(moderately rectangular, not extreme). Applied at all 8 figsize call
sites in the file (every `plot_*_grid*` function).

Regenerated `hitmiss_stim_area_acronym_custom` (46 areas) and
`modality_lick_area_acronym_custom` (51 areas) -- verified visually,
x-tick labels now fully legible instead of overlapping. Synced to
`M:\...\combined_results_ks4\...` via the same robocopy as before.

## 2026-09-24 (continued): fixed row-title/x-tick-label overlap

User: "Check the row-title overlap on the acronym_custom figures" --
confirmed visually: the 2nd row's 3-line subplot title (row_label |
col_label, n=, ANOVA) overlapped the 1st row's rotated x-tick labels
(e.g. "R+ n=36" text colliding with area-acronym labels). Root cause:
`fig.tight_layout(rect=(0, 0, 1, 0.95))` computes spacing before
`style_ax`'s `ax.set_box_aspect(...)` is applied at draw time, so its
margin estimate didn't account for the (now non-1) aspect -- this
existed in the original square figures too (same overlap, smaller
magnitude, easy to miss), just became obvious once the acronym_custom
fix made the mismatch bigger.

Replaced every `fig.tight_layout(rect=(0, 0, 1, 0.95))` (8 call sites)
with `constrained_layout=True` on the corresponding `plt.subplots(...)`
call instead -- matplotlib's constrained-layout engine re-solves
spacing after aspect-ratio constraints are applied, so it doesn't
have the tight_layout-vs-box_aspect ordering problem. Verified: no
more overlap on `hitmiss_stim_area_acronym_custom`'s comparison figure,
and `hitmiss_stim_area_group` (unaffected square case) renders
identically to before.

Re-ran all 15 maintained tags: 192 fresh PNG+PDF pairs, all matched.
Synced to `M:\...\combined_results_ks4\...`.

## 2026-09-24 (continued): pairwise-line figure for 058's real-vs-null

User asked which figure shows real-vs-null-median for PLS learning
predictions -> answered `058_perfquant_real_vs_null_distributions.png`
(058's Figure 1, violin-based, from the canonical 88-session full-pool
PLS+1SE validation) -> user: "Can you do the pairwise line figure".

New script `086_perfquant_pls_nulldist_paired.py`, pure post-hoc on
`058`'s already-computed CSV (no new compute): same rows=target x
cols=metric facet layout as `058`'s Figure 1, but one line per session
connecting its own null (mean across 1000 shift-null shuffles) to its
own real (out-of-fold test) score, instead of two marginal violin
distributions -- makes the within-session pairing visible directly.
Solid/darker = real > null for that session, faint dashed = not.
Session counts (real > null): whisker_curve 80/78/75 (r2/pearson/
spearman out of 88), falsealarm_curve 73/73/69, performance_curve
69/70/71 -- matches 058's console above-null summary.

Saved `086_perfquant_pls_nulldist_paired.png` + `.pdf`.

## 2026-09-24 (continued): cohort-combined cross-mouse summary + cohort-pooled trial-index PLS curves

Two follow-ups from a Q&A exchange about the cross-mouse/within-mouse
null designs ("What are the null hypotheses of these cross mouse and
within mouse runs" -- answered from `079`/`081`/`071`'s docstrings, no
new artifact):

**`087_perfquant_cohort_trialindex_curves.py`** (user: "I want the
figure that replace session by cohorts to predict trial index wise the
learning curves"). Pure post-hoc on `065`'s cache
(`065_perfquant_allmice_cache.pkl`), no new compute. Replaces `065`'s
one-panel-per-session grid with one panel per (day_stage, target),
pooling sessions WITHIN each cohort (R+/R-) onto a common normalized
trial-index axis (each session's trial_index rescaled to 0-1 fraction of
its own length, binned into 20 bins) -- plots cohort-mean true vs
predicted curves (+/- SEM across sessions) instead of per-session curves.
Saved `087_perfquant_cohort_trialindex_curves.png` + `.pdf`.

**`088_perfquant_crossmouse_cohortcombined.py`** (user: "Give a summary
figure that combines cohorts for the cross-mouse distributions of
corrected scores, testing against zero"). Pure post-hoc on `081b`'s
per-recipient CSV (`081_perfquant_crossmouse_fullsweep_persession.csv`),
no new compute. `081b` tested each of the 4 null types separately by
recipient cohort (4 cells); this pools the two mirror-image null types
per underlying question -- reward-dependence (`rminus_null` + `rplus_
null`) and specificity (`withinrplus_null` + `withinrminus_null`) --
across BOTH cohorts (n=75 recipients per test instead of 37-38), re-runs
Wilcoxon + one-sample t-test vs 0 on the pooled sample, BH-FDR across all
2 roles x 3 targets x 3 metrics = 18 tests. Dots stay colored by the
recipient's own cohort so the pooling is visible, not hidden.

Result: reward-dependence robust across nearly every target x metric
(q<0.05, mostly q<0.01) except falsealarm_curve/spearman (n.s.);
specificity weaker after pooling -- only whisker_curve significant
across all 3 metrics, falsealarm_curve n.s. everywhere, performance_curve
mixed (only r2 clears q<0.05). Saved
`088_perfquant_crossmouse_cohortcombined.png` + `.pdf` +
`088_perfquant_crossmouse_cohortcombined_grouplevel.csv`.

## 2026-09-24 (continued): hit/miss decoding split at learning_trial (pre vs post)

- [X] Sanity-check `learning_trial` vs data: `089_learning_trial_sanity.py`
      -> `089_learning_trial_sanity.csv`, `089_learning_trial_curves_grid.png`,
      `089_learning_trial_summary.png`. Reviewed with user; design locked
      via AskUserQuestion (see question.md "Learning-trial split").
- [X] Library: `reconstruct_learning_trial`, `prep_hitmiss_trials_learning_split`
      added to `scripts/ssl_timeresolved_decoding.py` (additive only).
- [X] Compute script `090_learning_trial_split_sensory_decode.py`; smoke
      test 3 sessions on haas OK (nulls ~0.50).
- [X] whole_brain run on haas (89 sessions, ~18 min): 68 decoded; skipped
      7 non-learners, 13 < 2 trials/class in an epoch (incl. all 4 R-
      fallback10 sessions -- 0 pre hits), MH038 length mismatch.
      `090_lt_split_results_whole_brain.parquet`.
- [ ] area_group -> area_acronym_custom runs on haas (queued sequentially,
      `090_log_<scheme>.txt`).
- [X] Figures `091_learning_trial_split_figures.py`, whole_brain done:
      `figures/whole_brain/learning/091_hitmiss_ltsplit_whole_brain_{sensory_paired_grid,
      sensory_overlaid,delta_vs_ntrials}_{all,learners}.png/.pdf`,
      `091_learning_trial_split_stats.csv`. Result (exploratory): pre
      accuracy equal across cohorts; post R+ up / R- down; delta R+ vs R-
      MW p=0.031/Welch p=0.017 (all), 0.006/0.003 (learners), size-matched.
      Within-cohort pre->post significant only for R+ learners (W p=0.010).
- [X] area_group run (~33 min) + figure
      `figures/area_group/learning/091_hitmiss_ltsplit_area_group_grid_condition_cohorts_{all,learners}.png/.pdf`.
      Matched above-null: epoch x area ANOVA R+ p=0.072 (all) / 0.020
      (learners), R- p=0.035 / 0.096; R+ rises mainly in Motor, Midbrain,
      Hippocampus; R- falls in Motor, Midbrain, Thalamus. No single area
      survives paired-Wilcoxon FDR (best raw p~0.10).
- [X] area_acronym_custom run (~38 min) + figure
      `figures/area_acronym_custom/learning/091_hitmiss_ltsplit_area_acronym_custom_grid_condition_cohorts_{all,learners}.png/.pdf`.
      Matched above-null epoch x area ANOVA: R+ p=0.0008 (all) / 0.0001
      (learners); R- p=0.042 / 0.13. No area survives per-area FDR (best
      raw: R+ PAG p=0.010 n=11, MRN p=0.030, MB p=0.034; R- RN p=0.047 n=7).
- [ ] CONSULT USER with figures; optional follow-up: placebo-split null.
- [X] Trial-aligned decoding `092_learning_trial_aligned_decoding.py` (whole_brain done,
      `092_lt_aligned_trials_whole_brain.parquet`; area_group running on haas),
      figure script `093_learning_trial_aligned_figures.py` (tested on smoke data).
- [X] Example sessions `094_learning_trial_example_sessions.py` ->
      `figures/whole_brain/learning/094_hitmiss_lt_example_sessions.png/.pdf`
      (per cohort: top-2 in group direction, median, counter-direction by 090 delta).
      Observations: post-LT R+ sessions often near-ceiling hits (e.g. AB123 76h/3m);
      R- sessions have very few hits (AB134 4h/4h) where per-trial p_correct-null is
      unreliable; late misses coincide with disengagement (slow-state confound).
- [ ] CONSULT USER on decoding improvements (listed in chat 2026-09-24).

## 2026-09-24 (continued): learning-trial split REDO (shift null, 3 windows, disengagement)
- [X] `detect_terminal_disengagement` added to `scripts/ssl_timeresolved_decoding.py`;
      check `095a_disengagement_check.py` -> `.csv`/`.png` (8/89 sessions cut: 7 R+, 1 R-).
- [X] `095_learning_trial_split_shiftnull.py` smoke-tested (fixed-50 shifts left 5/50 valid
      for 9-trial pre epochs -> switched to first-50-valid of up to 1000 draws; now 48-50).
- [X] whole_brain (26 min, 68/89 decoded) -> `095_lt_split_shiftnull_whole_brain.parquet`;
      figures `figures/whole_brain/learning/096_hitmiss_ltsplit_shiftnull_whole_brain_<window>_*`.
      Result: with the shift null the R+ pre->post RISE is gone (sensory matched 0.080 -> 0.087,
      p=0.72); the R- DROP remains (0.099 -> 0.042, paired t p=0.049; sensory-minus-baseline
      0.089 -> 0.026, W p=0.012). R+ vs R- delta: all p~0.09-0.11, learners p=0.03.
      Baseline window: above null in both epochs (~0.05), no change, no cohort difference.
- [X] area_group (52 min) + figures `figures/area_group/learning/096_hitmiss_ltsplit_shiftnull_area_group_<window>_*`.
      Epoch x area ANOVA (matched, acc - shift null): sensory R+ p=0.30 (all)/0.049 (learners),
      R- p=0.017/0.039; sensory_minus_base R+ p=0.56, R- p=4e-6 (all)/<1e-4 (learners);
      baseline n.s. both. R- drop broad across areas (Midbrain, Motor strongest, q=0.074).
- [ ] area_acronym_custom (running).
- [ ] Figures `096_learning_trial_split_shiftnull_figures.py` (reuses 091, parametrized),
      per window; stats `096_learning_trial_split_shiftnull_stats_<window>.csv`.
- [ ] CONSULT USER.

## 2026-09-24 (continued): placebo-split test of learning_trial
- [X] `097_learning_trial_placebo_split.py` smoke-tested (2 sessions; ~7 core-min/session;
      placebo delta SD 0.08-0.16 within session).
- [X] `098_learning_trial_placebo_figures.py` tested on smoke data.
- [ ] Full whole_brain run (queued on haas to start when 095 area_acronym_custom's PID exits;
      log `097_log_whole_brain.txt`).
- [ ] 098 figures + stats; 096 area_acronym_custom figures.
- [ ] CONSULT USER.

## 2026-09-24/25: follow-ups (disengagement, halves comparison, new LTs)
- [X] `095b_disengaged_sessions_detail.py` -> `figures/whole_brain/learning/095b_disengaged_sessions_detail.png`.
- [X] 095 options: SSL_SPLIT_MODE=half, SSL_NO_DISENGAGE=1, SSL_SESSION_IDS, SSL_LT_COLUMN (LT from
      ssl-learning-trial-identification/artifacts/006_learning_trials_final.csv).
- [X] No-drop re-run of the 8 flagged sessions -> `099_disengagement_sensitivity.csv`: |d change| mean 0.04,
      no systematic direction (5 decodable sessions).
- [X] Halves split, same pipeline (`095_lt_split_shiftnull_halfsplit_whole_brain.parquet`) + `099_lt_vs_halves_
      and_disengagement.py` -> `figures/whole_brain/learning/099_hitmiss_lt_vs_halves_whole_brain_*.png`,
      `099_lt_vs_halves_stats.csv`: R- sensory_minus_base change -0.063 at LT vs -0.013 at halves (t p=0.033).
- [X] Placebo split 097 (whole brain, 68 decodable sessions, median 21 (R+) / 49 (R-) placebo splits) -> 098
      (`figures/whole_brain/learning/098_hitmiss_lt_placebo_whole_brain_{summary_all,summary_learners,sessions_*}.png`,
      `098_learning_trial_placebo_stats.csv`, `_persession.csv`). Fixed an is_real object-dtype bug before
      final stats. Sensory: R+ pct 0.61 (W p=0.05; size-matched 0.65, p=0.015), R- 0.41 (p=0.11; size-
      matched 0.40, p=0.054), R+ vs R- MW p=0.005 (size-matched 0.002). R- delta profile has its minimum AT
      the LT; R+ pct driven by splits 20-40 trials before LT (composition caveat). Baseline: pct ~0.5 both.
- [X] 095 with new LTs (lt_recommended n=27, lt_recommended_broad n=37) -> `100_new_learning_trial_decoding.py`
      (see ssl-learning-trial-identification/SUMMARY.md).
- [ ] CONSULT USER.
- [ ] 096 stats for area_acronym_custom done (figures regenerated with shortened prefix `096_ltsplit_shiftnull`).

## 2026-09-25: new learning trials (lenient, clean-gated), disengagement KEPT, report
- [X] 095 runs with SSL_NO_DISENGAGE=1: lt_lenient_clean (007 table), halves (all sessions; halves mode fixed to
      not require a stored LT), stored LT. 097 placebo with lt_lenient_clean, disengagement kept.
- [X] `098 ... _lt-lt_lenient_clean_nodisengagedrop` -> `098_learning_trial_placebo_lt_lenient_clean_nodrop_*`:
      R+ sensory pct 0.68 (p=0.017; learners 0.70, p=0.007); R- 0.47 (n.s.).
- [X] `101_learners_vs_nonlearners.py` -> `figures/whole_brain/learning/101_*`, `101_*_stats.csv`: R+ sensory
      change learners@LT +0.060 vs learners@midpoint -0.003 (paired p=0.023); non-learners@midpoint +0.007.
      Lenient LT: R+ baseline +0.050 (p=0.011); R- s-b -0.078 (t p=0.045). Stored LT nodrop ~= with drop.
- [X] `102_build_learning_trial_report.py` -> `report_learning_trial_split.html` (self-contained, 23 figures, local).
- [ ] CONSULT USER: LT version to adopt; port fixes to behaviour_analysis; extend to areas / expert.

## Publication figure sets (added 2026-09-28)
- [x] Set 1, session-wide decoding (when / where, across cohorts): `113_publication_merged.py` (was 112) ->
      `figures/publication/set1_session_wide/` (whole-brain + area figures per analysis).
- [x] Set 2, within-session changes (halves; cross-half generalisation; per cohort and
      area): `112_publication_sets.py` -> `figures/publication/set2_within_session/`.
- [ ] Pooled decoding approach for Set 1: pool sessions per cohort x area into pseudo-populations with equalised
      trial counts per class and equalised neuron counts (subsample to a common n, repeat), so that cohort and area
      comparisons are not driven by different unit/trial numbers; add as extra Set 1 panels.
- [ ] Set 2: add the learning-trial split (095/103/105 results) and, once resolved, the coupling model (111).

## First-whisker-trial rule recompute (added 2026-09-29)
- [ ] Fine-area runs paused (user 2026-09-29): recompute `area_acronym_custom` (learning) for hitmiss_stim,
      modality_stim, modality_lick with the new trial prep (first whisker trial kept, 1 pre-whisker trial) and the
      min-3 rule: `SSL_FILLIN_TARGETS=exploratory-analyses/024_fillin_firstwhisker_<target>_area_acronym_custom.parquet
      python exploratory-analyses/024_master_sweep.py <target> <alignment> learning area_acronym_custom` (targets
      already built by 024c; old results in *.bak-prefirstwhisker-20260928). Then 032/034/042 for those tags. Until then the
      acronym results still LACK the first whisker trial (and use the old min-5 rule).
- [ ] Rerun other analyses using prep_session (listed in change-log 2026-09-28): LT-split family (089-111b), 031, 033,
      036/037, 039, 043-048, 073, behavior/001; BWM single-cell project (confirmatory 001/002/004, exploratory 002-006).
- [ ] Decide: learning-curve helper `_active_trials_from_whisker_onset_for_curve` keeps the first whisker trial but not
      the preceding trial (positional alignment with stored learning curves).
- [ ] Expert-stage first-whisker recompute PAUSED (user 2026-09-29, to focus on the pseudo-population project):
      done: hitmiss_stim_expert_whole_brain. INTERRUPTED at 5/28 sessions: hitmiss_stim_expert_area_group (its parquet
      now mixes 5 recomputed sessions with old rows -> rerun the whole fill-in, same target file). Not started:
      modality_stim / modality_lick expert whole_brain + area_group. Then figure scripts 025-030, 032, 034, 034b,
      041, 042. Queue script for reference: haas /tmp/firstwhisker_queue.sh.part4; log
      exploratory-analyses/024_firstwhisker_queue_log.txt.

## 2026-09-30 to 2026-10-04: split decoding (halves, LT variants, placebo, single trial, stimulus epochs)
- [X] 117/117b count-matched halves; 118-120 LT definitions (_A1v2); 122/122b, 126 placebo splits (step1_pl100); 124 / 125
      behaviour; 127, 130 single-trial margins; 129 mid / best split; 131 cohort permutation; 132-135b stimulus epochs.
- [X] Article report (skills/project-report): `report/build_report.py` (haas) + `report/render.sh` (local) -> `combined_results_ks4/ssl-whisker-hitmiss-timeresolved-decoding/report/` (report.pdf / .md / .html, numbers.json, 17 figures). Not yet published to the private reports repo.
- [ ] Rerun 117 and 123b with perf == 6 excluded (both ran before the rule).
- [ ] CONSULT USER (report section 9): pursue the R+ hit/miss increase at the whisker change point (L5)? lock an LT
      definition; 135b robustness; expert reruns incl. MH062_20260113 (on hold); pilot repetitions.

## Cleanup 2026-10-04 (see change-log for the per-block decisions)
- [X] Archive / perfquant split done (manifest in _archive/).
- [X] Results home: moved to combined_results_ks4/ssl-whisker-hitmiss-timeresolved-decoding/ (symlinks on haas; run sync_results.sh after each job).
- [ ] Deferred reruns (perf == 6): 024 sweep (13 sessions, whole brain + area_group), 117, 123b.
- [X] question.md rewritten (2026-10-05); report rebuilt with Parts I (session-wide), II (within-session), III (geometry).

## Hit-median split and population geometry (planned 2026-10-05, user-approved design)
Units for all of these: tracked "stable" units = not non-soma, nSpikes >= 300, spikes missing <= 20 %, coverage >= 0.9,
presence >= 0.5, drift joint test passed (fail = |r| > 0.5 and p < 0.01), NO refractory-violation criterion (MUA allowed),
>= 0.5 Hz in every analysed epoch. Units without drift results (614 learning units in 77 sessions): use quality_label good
until their drift test is run.
- [X] Run the missing drift tests for the units without results: done 2026-10-05 (142). Rechecked 2026-10-06: 893 somatic units (786 mua, 107 good, 103 sessions) remain untestable (no spikes in the central segment), presence <= 0.38, so none could pass presence >= 0.5 and none is stable either way; not rerun.
- [ ] 137 stable-unit table (learning + expert), from the full v2 cache (before the shared-area filter).
- [ ] Hit-median split (cumulative whisker hits reach half; after exclusions and A1 trim) + midpoint reference, whole brain,
      learning stage, windows only: hit/miss (baseline -200..-10, 5-50, 5-100 ms) and pre-lick modality (-100..0 ms);
      single whole-session decoder scored per half (116-style) and separate count-matched decoders + cross-half
      generalisation (117-style); one C per session; within-half linear-shift null; >= 3 trials per class per half.
      Pilot 4 + 4 sessions, time estimate, CONSULT USER before the full run.
- [ ] Coding direction (5-35 ms, hit - miss on whisker trials, trials with a lick before 35 ms excluded), per half (hit-median
      and midpoint): rotation (reliability-normalised cosine between halves, disjoint trials) vs gain (d' along own vs other
      half's axis); projection of passive pre / passive post whisker, auditory and whisker - auditory responses onto each
      active coding direction.
- [ ] Pilot (4 + 4 sessions): within-class noise covariance (top ~10 PCs, shrinkage) along the coding direction; bias-corrected
      linear Fisher information per half. CONSULT USER with the pilot.
- [x] Auditory misses counted (median 2 per learning session; 18 / 89 with >= 5): cross-modality generalisation dropped.
- [X] 2026-10-05 overnight: missing drift tests rerun (142), stable units rebuilt (137), 5-35 ms LT/placebo rerun, 139 hit-median
      split, 140 coding direction + noise (full runs), figures 143/144/145 (both scopes), report rebuilt (report_20261005.pdf).
- [ ] CONSULT USER on the overnight results; drift tests still missing for 893 somatic units (too few bins for the test).
- [ ] 2026-10-05 robustness of the choice-axis readout (146/147): session selection documented, >= 3 hits / misses (was 6);
      response variants (epoch baseline, per-trial baseline, baseline window alone); timing / reward / baseline-rate controls,
      covariate model, behaviour link, learners scope; 140/145 with whisker- and auditory-evoked alignment. Report update after.
