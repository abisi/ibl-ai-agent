## Purpose
Established metric definitions and statistical designs from completed SSL
projects (`projects/ssl-*`), so a new analysis of a similar shape does not
have to re-derive definitions the user has already locked in and validated.
These are prior-project conventions, not fixed requirements — state clearly
when a new question deliberately deviates from one.

## Area grouping — two schemes, do not mix
1. **Allen custom large-groups** (`allen_utils.get_custom_area_groups()` /
   `process_allen_labels()` / `create_area_custom_column`, from the user's
   external `M:\analysis\Axel_Bisi\Github\allen_utils` — not part of this
   repo): groups such as Motor and frontal, Somatosensory, Auditory,
   Retrosplenial, Visual, Hippocampus, Striatum and pallidum, Thalamus,
   Midbrain, Pons and medulla, Olfactory, Amygdala and hypothalamus. An
   insertion is counted under every group at least one of its units maps to.
   "Unassigned" aggregates fiber tracts, ventricles, unregistered (`root`)
   units, and anything outside the curated list — it is a real, large bucket,
   not missing data, unless an excluded-area filter was explicitly applied
   upstream. ~1% of units lack CCF-atlas registration
   (`ccf_atlas_acronym` null); fall back to the always-populated
   `ccf_acronym`/`ccf_parent_acronym` fields for those rather than dropping them.
2. **IBL Beryl mapping** (`iblatlas.regions.BrainRegions.acronym2acronym`,
   this repo's own `ibl-anatomy` skill): used in `ssl-ks2-single-mouse-tca`
   for `ccf_acronym` → Beryl normalization before area-level aggregation,
   with `void`/`root` counted but excluded from area comparisons. This is the
   preferred scheme when the analysis otherwise reuses `ibl-anatomy`/IBL
   conventions and no dependency on the external `allen_utils` package is
   wanted.
Declare which scheme is in use before any area-level groupby; do not compare
area labels produced by the two schemes as if they were the same grouping.

## Passive sensory responsiveness and modality selectivity
From `ssl-passive-sensory-selectivity` (locked design, GLMM confirmatory —
not an exploration/confirmation split; see that project's
`projects/ssl-passive-sensory-selectivity/question.md` for full detail if
reproducing it exactly). Passive stimulation itself is 35 auditory + 35
whisker stimuli, randomly interleaved every 3 s, spout retracted — see
`ssl_behavioral_paradigm.md`'s Passive stimulation section for the protocol.
- Passive-only, dead-zone excluded: never compute firing rate from spikes
  within **10ms** of stimulus onset, in baseline or response window. This is
  an analysis-time choice, unrelated to the task's own 100ms acquisition-time
  artifact window (`ssl_behavioral_paradigm.md`) — the two are not meant to
  be reconciled.
- Response windows (compute both, in parallel): **10-50ms** and **10-30ms** post-stim-onset.
- Baseline window: **-60 to -10ms** pre-stim-onset.
- `response` (per unit, per modality, per pre/post-task period) = mean FR in
  response window minus mean FR in baseline window, averaged over that
  period's trials of that modality.
- `diff` (per unit, per modality) = `response(passive_post) - response(passive_pre)`.
- Selectivity index = `(whisker_diff - auditory_diff) / (whisker_diff + auditory_diff)`.
- Run separately per day-stage (`day==0` vs `day>0`) — do not pool across training stage.
  Passive data exists for every recorded day for AB116+ subjects (see
  `ssl_behavioral_paradigm.md`'s Passive stimulation section), so both arms
  should have data for that subset; subjects below AB116 contribute to
  neither arm's passive analysis.
- GLMM: `firing_rate_diff ~ modality * reward_group + whisker_active_count + auditory_active_count + (1|mouse_id)`,
  fit **pooled across all brain areas** (area comparison stays in descriptive
  figures only, not as a model term); fit separately per response window and per day-stage.
- Passive-trial selection: use the trial-level `context` column, not
  `epochs.parquet` boundaries (see `../../ssl-load/references/ssl_dataset_schema.md`'s epoch caveat).

## Reward-history modulation indices
From `ssl-reward-history-modulation`.
- Restrict to `context=='active'` and drop `perf==6` (association) trials first.
- Apply the **cohort-corrected** hit/non-hit definition from `ssl_task_semantics.md`
  — this is the whole point of the analysis; the naive `lick_flag` reading gives wrong signs for R-.
- Common index structure, per unit, in a **5-50ms post-stimulus-onset window**:
  ```
  index = (FR | t-1 hit − FR | t-1 non-hit) / mean(FR across all trials satisfying the t-condition)
  ```
  computed for: hit-stay (whisker t=hit, split by t-1), non-hit-stay (whisker
  t=non-hit, split by t-1), whisker-to-auditory transition (t=auditory_hit,
  t-1=whisker trial, split by that whisker trial's cohort-corrected outcome),
  auditory-to-whisker transition (t=cohort-corrected whisker-hit, t-1=auditory
  trial). A unit's index is NaN if either t-1 bucket has fewer than 5 qualifying trials.
- **The auditory-to-whisker transition index is structurally thin for R+ mice**
  (auditory performance near-ceiling in R+ means a whisker-hit trial is almost
  never immediately preceded by an `auditory_miss`) — checked and dropped in
  the source project after confirming it wasn't a threshold artifact. Check
  trial counts for this index before relying on it; do not assume loosening
  the trial floor fixes a structural absence.
- **PERMANOVA pseudoreplication trap (found and fixed in this project — do
  not repeat it):** `reward_group`/cohort is a **mouse-level** property but
  units are the row grain (~300-900 units per mouse). Permuting unit labels
  directly for a PERMANOVA null is invalid — it treats units as independent
  replicates of the cohort effect when they are not. The fix is to permute
  **whole mice** between cohorts (block permutation), keeping each mouse's
  units together. In the source project, invalid unit-level permutation gave
  p=0.0004 with several "significant" areas; the corrected mouse-block
  permutation on the *same data* gave p=0.57 with nothing significant — the
  original result was pseudoreplication, not signal. Apply mouse-block (or
  equivalent subject-block) permutation to **any** unit-level test of a
  mouse-level factor, not just PERMANOVA.

## Task-performance curves
From `ssl-task-performance`. Trial-by-trial hit-rate/false-alarm curves
during the active epoch: restrict trials to the active epoch (bound by
`epochs.parquet`'s active window, or `context=='active'` — cross-check both
per `../../ssl-load/references/ssl_dataset_schema.md`'s epoch-reliability caveat), then compute a
trailing rolling mean of `lick_flag` per `trial_type` (window=10 trials of
that type, `min_periods=3`) plotted against trial position within the active
sequence. This uses the naive (cohort-agnostic) outcome classification — fine
for visualizing raw performance, but re-derive with the cohort-corrected
definition before interpreting whisker-trial "hit rate" as a reward-seeking
measure for an R- cohort.

## Trial-factor / TCA (tensor component analysis) pipeline
From `ssl-ks2-single-mouse-tca`, extending the user's external "megamouse"
pipeline (`M:\analysis\Axel_Bisi\brain_wide_analysis\tca\tca_pipeline_bis.py`)
to single-mouse-level fits. Key design choices, if extending or reproducing:
- Tensor axes: trials × neurons × time bins, one tensor **per session/mouse**
  (not pooled across mice on the neuron axis, unlike the megamouse pipeline).
- Trial alignment: whisker trials aligned to first active-context hit (or a
  hit-rate inflection point), with `trial_window_pre`/`trial_window_post`
  around it.
- Decomposition: non-negative CP/TCA (`tensortools.Ensemble`, `ncp_hals`) —
  non-negative factors mean no CP sign ambiguity, only a permutation
  ambiguity when matching components across mice.
- "Neural activity tracks learning" test: correlate each component's
  trial-mode loading against a behavioral trace (smoothed P(lick), or d-prime
  derived per-mouse from `trials.parquet` `lick_flag`/`trial_type` — this
  dataset has no pre-computed `d_prime_w` column). **Null model: circular/
  block-shift permutation**, not independent-trial shuffling — a TCA trial
  factor and a behavioral learning curve can both drift monotonically over a
  session for unrelated reasons, which independent shuffling would not
  control for; circular shifting preserves each series' own autocorrelation
  and trend while destroying only their alignment.
- Cross-mouse component matching: correlate time-mode loading vectors (the
  only axis shared across mice, since there is no shared neuron axis at the
  single-mouse level) pairwise, solve linear-sum assignment (Hungarian
  algorithm) per mouse pair, then aggregate into cohort-wide consensus groups
  via iterative Hungarian-assignment consensus clustering (each mouse
  contributes exactly one component per consensus slot).
- Multiple-comparisons scale: at cohort scale (e.g. 68 mice × 4 components ×
  2 behavioral variables) this is an exploratory/hypothesis-generating sweep,
  not a single confirmatory test — state the uncorrected test count explicitly.

## Rare/skip exploration-confirmation split cases
Two prior SSL projects explicitly skipped or altered the standard
exploration/confirmation split from `exploration-confirmation/SKILL.md`, both
with explicit user approval recorded in each project's question.md file:
- `ssl-passive-sensory-selectivity`: fully locked design agreed before any
  computation (windows, cohort definition, model structure) — treated as a
  single planned confirmatory analysis, no split needed.
- `ssl-ks2-single-mouse-tca`: user explicitly approved skipping confirmation
  entirely for a hypothesis-generating cohort sweep; results were reported as
  exploratory/uncorrected, not as confirmed effects.
Do not skip or alter the split on your own judgment — get the same kind of
explicit, recorded user approval first, per `exploration-confirmation/SKILL.md`.
