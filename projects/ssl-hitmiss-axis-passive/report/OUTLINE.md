# Report outline: ssl-hitmiss-axis-passive (draft 2026-10-15, for review)

Question: does learning the whisker-reward contingency change the early (5–35 ms, pre-movement) passive whisker response
relative to the population direction that predicts licking?
Dimensions shown at every level (2–5), inside each figure: whisker vs auditory (control stimulus), R+ vs R−, whole brain and
area groups (allen_utils order, ≥ 3 sessions per group), real axis vs linear-shifted axes.

## Methods (shared_methods: task, learners, trials, units, shift_null, statistics, areas)
- Tracked stable units; 5–35 ms rate minus the epoch baseline; harmonised scaling (unit mean / SD from active trials, 135
  SSL_135_HARM=1); hit − miss axis from one half of the active whisker trials, held-out hits / misses and passive pre /
  post trials projected per neuron (/ √N); thresholds: ≥ 10 hits and misses within day, ≥ 4 across days (MIN_CLASS 2).
- Decoder readout variant (146): L2 logistic hit vs miss, passive trials scored by models not trained on them.
- Along vs orthogonal / gain vs rotation decomposition (169).

## Results
1. The axis: definition and what it separates. Schematic 161 (user's own schematic in figures), 167 definition example,
   168 / 165-psth time course of held-out hits, misses and passive trials (whisker and auditory).
2. Within day 0: passive pre → post on the axis, real vs shifted; whisker and auditory; areas.
   Panels: 158 (within-day, latest), 164 shift-null check, 162 per area; decoder readout 149 / 160 as one panel.
3. Beyond session time: the change against the linear-shift null (164, 149, 150), with exposure (153) and expert (156)
   controls; whisker and auditory; areas.
4. Rotation vs gain: 169–171 (165 angle / polar panels); whisker and auditory; areas.
5. Across days: day 0 vs expert passive projections, pre and post, R+ vs R−; whisker and auditory; areas (175 v2; area
   panel as in ssl-prelick-convergence 119 row 1, to be generated here).
Conference composites (Supp, panels present above): 154, 158, 160, 165_two_axes_compact_angle, 173.

## Registry statuses to settle
- Within-day R− move away from the hit side beyond the shift null (holds?); R+ change (n.s.?).
- Across-days cohort × stage interaction (n.s.), per-area post hoc (n.s.).
- Axis vs behaviour coupling (170 / 176): fragile, time-confounded.
