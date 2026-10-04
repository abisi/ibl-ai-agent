# Question

## Original question (from user, 2026-08-14)

Plot the PSTHs (10ms bins, 2ms stride, mindful of the artifact dead zone) for
each mouse, each area (using allen_utils), for whisker trial and auditory
trial, and the difference of the two. One row per cohort, one column per
plot type (whisker, auditory, difference), showing both passive_pre and
passive_post.

Statistical analysis of the interaction of stimulus type with reward group
(cohort), at three levels: baseline (1s before stimulus), stimulus-evoked
(5-45ms), and stimulus-evoked baseline-corrected (single-neuron baseline
correction). Parsimonious linear mixed-effects models (random effect: mouse)
predicting the change in firing rate post-minus-pre learning, in stages: (1)
all areas together, stim_type * reward_group; (2) + number of whisker/
auditory stimuli each mouse received during the active portion; (3) + area
group (allen_utils) -- resolve whether area needs a three-way interaction
with stim_type * reward_group, or just an intercept shift. Plot effects as
barplots + points per window; an interaction figure (stim_type x cohort) for
the global model; scatter plots (whisker delta vs auditory delta), global
and per area, at mouse level and single-neuron level (density maps).

Summarize with a suppression-selectivity modulation index: (delta_whisker -
delta_auditory) / (delta_whisker + delta_auditory), bounded in [-1, 1] --
resolve whether the deltas should be absolute-valued. Plot the global index
distribution by cohort, test the cohort difference with a non-parametric
unpaired test. Plot the index by area_group x cohort in one figure. Ask
whether the index is driven by high-firing neurons (correlate with overall
firing rate).

Restrict to units firing in both passive_pre and passive_post (e.g. high
coverage ratio = fraction of recording time spanned by first-to-last spike),
keeping both `good` and `mua` units. Summarize everything in a formal
report. Run autonomously through to the report, without further check-ins,
after the scheduled KS4 dataset rebuild (2026-08-13 23:00) completed.

## Judgment calls resolved during the analysis

The user explicitly delegated two design decisions; both were resolved
analytically rather than by assumption, and are reported here so the
resolution method is auditable:

- **Absolute value in the modulation index**: required. `(a-b)/(a+b)` is
  only guaranteed bounded in [-1,1] when `a` and `b` share a sign. Whisker
  and auditory deltas can have either sign (excitation or suppression), so
  a signed-delta version can have its denominator cross zero and become
  unbounded/undefined -- the opposite of the stated requirement. Used
  `(|delta_w| - |delta_a|) / (|delta_w| + |delta_a|)`, a magnitude-
  selectivity index (which modality drives the larger absolute change),
  not a direction index.
- **Three-way area interaction vs. area intercept**: resolved via a nested
  likelihood-ratio test (both models fit by ML) rather than assumed either
  way, per window. See Results.

Additional choices made without being explicitly specified, documented for
auditability:

- **reward_group / learning_category source**: `joint_mouse_reference_weight.xlsx`
  (per user's redirect mid-analysis), not the NWB `experiment_description`
  `wh_reward` field originally planned. `exclude` / `exclude_ephys` flags
  from the same sheet applied.
- **area_group source**: `allen_utils.process_allen_labels(subdivide_areas=True)`
  + `get_custom_area_groups()`, per user's explicit instruction, with
  `target_region` (a required input the compressed dataset does not carry)
  re-extracted from raw NWB `electrode_group.location['area']`.
- **Coverage-ratio threshold**: 0.9 (first-to-last-spike span over session
  duration). Chosen from the observed distribution (Q1=0.982, median=0.998)
  -- a fairly strict cut that only removes units clearly in the degraded/
  drop-out tail, cross-checked against a direct >=1-spike-in-each-passive-
  epoch criterion (97% agreement between the two).
- **"delta spikes" window for the modulation index**: the evoked,
  baseline-corrected delta (most direct isolation of a stimulus-specific
  response change).
- **Cohort-difference test for the modulation index**: implemented as a
  mouse-block permutation test (on per-mouse median MI) rather than a
  pooled-unit Mann-Whitney U, because reward_group is a mouse-level factor
  and unit-level rows would be pseudoreplicated -- this project's own
  `ssl-analyze/references/ssl_analysis_patterns.md` explicitly flags this
  failure mode for other cohort comparisons.

## Data-quality issues found and handled

- One raw NWB file, `MH006_20250123_131233.nwb`, failed to parse ("Missing
  NWB version in file") during the scheduled rebuild -- excluded from the
  build (857/858 sessions succeeded).
- 9,180 / 179,124 units were dropped by `allen_utils.process_allen_labels`
  itself (excluded-area list, or missing CCF registration for 57 mice) --
  not further investigated, treated as upstream data quality already
  handled by that established utility.
- The `area_group` "Pons and medulla" has zero R+ units (33 R- units only)
  -- a structurally empty cell for the stim_type x reward_group x area_group
  design. An initial fit produced a degenerate (~1e16) coefficient for it;
  excluded from the area-stratified models rather than left in.
- The LMM's default optimizer (L-BFGS) produced degenerate zero-variance
  boundary solutions (infinite likelihood) for every model in this dataset.
  Diagnosed as an optimizer artifact, not a data problem (no NaNs,
  duplicates, or zero-variance groups); fixed by trying BFGS/CG/Powell/
  Nelder-Mead per fit and keeping the best converged result (Powell was
  needed specifically for the 51-parameter three-way area models).
