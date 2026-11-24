# Reward/choice-history modulation of day-0 whisker-learning firing rates: R+ vs R- cohorts

## Question

Do R+ and R- mice (whisker-rewarded vs. whisker-unrewarded contingency
cohorts) differ in how strongly recent choice/reward history modulates
single-unit firing rates on trial *t*, on the single day each mouse first
encounters the whisker stimulus ("day 0")? And if so, in which brain areas?

## Approach

**Population**: 74 mice (38 R+, 36 R-) with day-0 whisker-learning ephys,
excluding bad/unannotated learners (`learning_category` in
`joint_mouse_reference_weight.xlsx`) and `exclude_ephys` sessions. Day 0 is
identified directly from `session_description == 'whisker_0'`.

**Metric**: for a chosen category of trial *t* (e.g. "the mouse just made a
rewarded whisker response"), split those trials by whether the *preceding*
trial (*t-1*) was itself a rewarded/hit trial or not, and compare mean
firing rate in a 5-50ms post-stimulus window:

```
index = (FR | t-1 hit − FR | t-1 non-hit) / mean(FR across all trials satisfying the t-condition)
```

"hit" is cohort-corrected: for R- mice, *withholding* a response to whisker
stimuli is the rewarded action, so the raw stimulus-response labels are
flipped relative to R+ before computing the index. Figure 1 shows this
worked through for one real unit.

![Figure 1: metric definition](figures/fig1_metric_definition_hitstay.png)

*Figure 1. hit_stay index for one example unit (session AB159_20250409_135813,
cluster 4000256). Blue = trials where the mouse's whisker-hit was preceded by
another hit ("win-stay"); red = preceded by a non-hit. The shaded band is the
5-50ms response window the index is computed from.*

Three indices were carried into the confirmatory analysis:
- **hit_stay**: *t* = whisker hit, split by *t-1* hit/non-hit.
- **nonhit_stay**: *t* = whisker non-hit (error), split by *t-1* hit/non-hit — the mirror of hit_stay.
- **w2a_transition**: *t* = auditory hit immediately after a whisker trial, split by that whisker trial's outcome.

A fourth index, **a2w_transition** (*t* = whisker hit immediately after an
auditory trial, split by that auditory trial's outcome), was dropped before
locking the confirmatory test: it requires an `auditory_miss` at *t-1*, and
auditory performance is near-ceiling (~88% hit rate) in R+ mice, so almost no
R+ mouse ever produces this trial sequence (2/38 mice do, vs. 9/36 R-; even
loosening the per-unit trial floor to 1 trial/bucket left R+ at 0 valid
units — this is a structural absence of the triggering event, not a
threshold artifact).

**Exploration/confirmation split**: since the design (indices, cohort/mouse
filters, area pipeline, statistics) was fully locked through discussion
before any code was written, an exploration subset of 15 mice (8 R+/7 R-)
was still held out and used only to sanity-check index coverage and
plausibility before the confirmatory test — see `../question.md` for the
full locked specification and the exploration-phase findings that led to
dropping a2w_transition.

**Areas**: `allen_utils.process_allen_labels` (default settings) applied to
each unit's re-extracted target region, then restricted to areas with ≥2
mice/cohort, ≥5 units/recording, ≥30 total units — 40 areas retained.

**Statistics**: PERMANOVA (Euclidean distance) on the standardized index
vector, factor = cohort, with **mouse-block permutation** — cohort is a
property of the mouse, not the unit, so the null distribution permutes which
whole mice get which cohort label rather than shuffling individual units.
Post-hoc: the same test repeated within each area, BH-FDR corrected at 5%.

## A bug caught before reporting any number

The first version of this test permuted *unit* labels for the null
distribution (7496 "samples"). That's invalid: `reward_group` is fixed per
mouse, so ~680 non-independent units per mouse were being treated as
independent draws. That version gave pseudo-F=5.73, **p=0.0004**, with
ORB/SSp-bfd/MOp "significant" post-hoc. Switching to mouse-block permutation
on the *identical data* gave p=0.573 with nothing significant — the original
result was pseudoreplication, not a real cohort effect. All results below
use the corrected mouse-block permutation.

## Results

| Run | Indices | n units | n mice (R+/R-) | pseudo-F | p | Areas surviving BH-FDR 5% |
|---|---|---|---|---|---|---|
| **Locked confirmatory** | hit_stay, nonhit_stay, w2a_transition | 7,496 | 11 | 5.73 | 0.573 | 0 / 10 tested |
| Full population (same 3 indices, exploration+confirmation merged for power) | hit_stay, nonhit_stay, w2a_transition | 9,442 | 14 | 9.85 | 0.242 | 0 / 11 tested |
| Supplementary, not locked (w2a_transition dropped for its own mouse-level bottleneck) | hit_stay, nonhit_stay | 31,153 | 41 | 2.08 | 0.809 | 0 / 32 tested |

The requirement that a unit have *all three* indices simultaneously defined
turned out to be the binding constraint on sample size, and it's structural:
merging the exploration and confirmation sets only recovered 14/74 mice with
any qualifying unit, because most day-0 sessions aren't modality-interleaved
enough to generate a whisker-then-auditory transition trial at all. Dropping
`w2a_transition` (supplementary run) recovered 41/74 mice — the
best-powered version of this test — and is also null.

![Figure 2: index distributions](figures/fig2_index_distributions_fullpop.png)

*Figure 2. Per-unit index distributions by cohort, full population. All
three show substantial overlap between R+ and R- with no clearly separated
mode — consistent with the PERMANOVA null result. Note `hit_stay`'s long
left tail (see Limitations).*

![Figure 3: post-hoc areas](figures/fig3_posthoc_areas_all_variants.png)

*Figure 3. Post-hoc per-area PERMANOVA, all three test variants. No area
crosses the BH-FDR line (would be plotted red) in any version; CA3 in the
locked-confirmatory run is the closest any area gets to even the
uncorrected p=0.05 line, and it doesn't survive correction.*

## Conclusion

No evidence that R+ and R- mice differ in reward/choice-history modulation
of day-0 whisker-learning firing rates, for any of the three well-covered
indices, in the locked confirmatory test or in either follow-up run with
more statistical power. No brain area shows a cohort difference surviving
multiple-comparisons correction in any version tested.

## Limitations

- **`hit_stay`'s denominator is unstable for low-firing-rate units.** The
  index divides by the *mean* firing rate across all qualifying trials
  (chosen for interpretability, not boundedness); for a near-silent unit
  with a stray high-count trial in one bucket, this ratio can blow up (one
  early candidate example unit produced an index of -8.93). This is visible
  as the long left tail in Figure 2's `hit_stay` panel. Indices were
  z-scored before PERMANOVA, which is sensitive to such outliers — a robust
  (median/IQR-based or firing-rate-floored) version of the index would be
  worth checking if this analysis is revisited, though it's unlikely to
  turn a p=0.8 result significant.
- **`w2a_transition` and especially `a2w_transition` are governed by
  session-design structure** (whether whisker/auditory trials are
  interleaved on a given day-0 session) more than by biology, which limits
  how much sample size can help them.
- **Units within a mouse are not independent**, and the mouse-block
  permutation correctly treats the mouse, not the unit, as the effective
  sample for the *cohort* test — but the per-area post-hoc still has modest
  mouse counts per area in several cases (see `n_mice` column in
  `../artifacts/posthoc_area_permanova*.csv`), so absence of a significant
  area should be read as "not detected at this power," not "definitively
  absent."

## Reproducing

Scripts, in order, under `../exploratory-analyses/`: `000` (mouse split) →
`001` (trial outcome/history) → `002` (area labels) → `003` (per-unit
indices) → `004` (exploration-set sanity checks) → `005` (locked
confirmatory PERMANOVA) → `006`/`007` (full-population and supplementary
re-runs) → `008` (this report's figures). All artifacts in
`../artifacts/`.
