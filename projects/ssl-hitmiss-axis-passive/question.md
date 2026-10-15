# ssl-hitmiss-axis-passive

## Question
How does learning a whisker-reward contingency change the early sensory response to the whisker stimulus, relative to the
neural population direction that predicts licking? Each session's hit - miss axis is defined from active whisker trials
5-35 ms after stimulus onset (before any movement), and passive whisker and auditory responses (before and after the task)
are projected onto it; within day 0 (passive pre -> post) and across learning stages (day 0 vs expert), R+ vs R-.

## Scope
Moved from ssl-whisker-hitmiss-timeresolved-decoding (Part III, "stimulus-onset geometry across epochs") on 2026-10-15;
the decoding project keeps decodability, single-session and pseudo-population decoding, session splits and
learning-trial definitions.
- Axis and projections: 135 (alignment / projections, harmonised scaling), 140 (coding direction across halves), 146 (choice
  decoder readout of passive responses), 168 (time-resolved projections), 169-171 (along vs orthogonal, gain vs rotation).
- Controls: linear-shift null (shift_null.py), exposure (153), expert sessions (156), auditory responses, per area (162).
- Across days: 175; figures 158, 160, 165, 173 (two-axis figures shared with ssl-prelick-convergence), 174 (schematic).
- Modality decoding across epochs: 132-134 (data preparation used by the other scripts).

## Shared code
143_lt_split_windows_figures.py, tracked_units.py and shift_null.py are re-exports of the files in
ssl-whisker-hitmiss-timeresolved-decoding (kept in one place); that project's 132 is a re-export of this project's 132.

## Results
combined_results_ks4/ssl-hitmiss-axis-passive/{tables,cache,figures}; scripts write next to themselves through
symlinks (sync_results.sh after each job). Move manifest: move_manifest_20261015.tsv (old paths are symlinks to the new files).
