# Report outline: ssl-whisker-hitmiss-timeresolved-decoding (draft 2026-10-15, for review)

Scope: decodability (time-resolved, whole session), single-session vs pseudo-population decoding, within-session change
(session splits), learning-trial definitions. Passive axis analyses moved to ssl-hitmiss-axis-passive.
Dimensions shown at every level (inside each figure, not as separate sections): cohort (R+ / R−), stage (day 0 / expert
where run), whole brain and area groups (allen_utils order), and a control decoding where it exists (modality decoding
as the non-choice control of hit / miss decoding).

## Methods (shared_methods: task, learners, trials, units, lick_alignment, shift_null, statistics, areas)
- Decoders (024): causal 50-ms bins, 5-ms stride; L2 logistic; stratified CV; balanced accuracy; label-shuffle and
  linear-shift nulls; group-level tests (026).
- Pseudo-population (033) and neuron-count control (073).
- Splits: halves (separate, single decoder 114/116, count-matched 117/117b), hit-median (139), learning-trial definitions
  (017, 118), placebo splits (122), cohort-label permutation (131).
- Single-trial held-out margin (123), behaviour scores (124/125), learning curves (024 HMM, σ = 1).

## Results (one main figure per question; panels taken from the latest versions)
1. Choice and stimulus information across the session: hit / miss, modality at stimulus, modality before the lick,
   performance state; time course, window value, latency; R+ vs R−; areas.
   Main: 113_hitmiss_set1_session_wide (+ modality / perfstate set1 as panels or Supp). Supp: 042 latency, 041 learning vs
   expert, 109 classifier comparison.
2. Single-session vs pseudo-population, and neuron count: 033 / 038, 073 (Supp unless a claim needs it in main).
3. Within-session change, session splits: 113_hitmiss_set2_within_session, 113_hitmiss_set2b_single_decoder, 117b
   count-matched, 144 hit-median split, 129 mid vs best split, 131 cohort permutation.
   Conference composite (Supp): 115_cosyne_halves_figure (panels from 113 set2 / 114).
4. Learning-trial splits: 119 comparison across definitions, 120 defined vs undefined, 122b placebo, 143 L5 windows and step
   vs gradual. Conference composite (Supp): 138_cosyne_lt_placebo (panels from 119 / 122b).
5. Single-trial readout and behaviour: 121 examples, 123b vs LT, 124 / 125 behaviour scores, 127 + 176 margin vs learning
   curve (lagged), 130 neural transition. Expected status: mostly n.s. / fragile (stated as such).

## Registry statuses to settle in review
- Which session-split effects hold beyond the linear-shift null and cohort permutation.
- Whether the LT split adds anything over the placebo distribution (122b).

## Removed from the previous report
Part III (moved to ssl-hitmiss-axis-passive); pilot figures (003-022 series, 035, 117 pilot); deck figures.
