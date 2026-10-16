# ssl-within-day-two-axes

## Question
Within the learning session (day 0), which changes along the two coding directions are specific to learning, and which are
present (or stronger) whenever the task is run? Two axes per session:
- reward-lick coding direction (ssl-prelick-convergence): auditory hit − spontaneous lick in the 100 ms before the lick; the
  within-session drift of whisker hits along it (slope over the session, beyond a linear-shift null);
- hit − miss axis (ssl-hitmiss-axis-passive): active whisker hits − misses 5–35 ms after the stimulus; the passive pre → post change
  of the whisker (and auditory) response along it, beyond a linear-shift null.
Expert sessions are the control: the same within-session measures in sessions where the contingency is already learned.

## Classification (per cohort, whole brain and per area group)
- learning-specific: within-session change on day 0 (≠ 0), none at expert, day 0 vs expert different;
- stronger with learning: change at both stages, larger on day 0 (day 0 vs expert different);
- present at both stages: change at both stages, day 0 vs expert not different;
- day 0 only, stage contrast n.s.: change on day 0 only, but day 0 and expert not distinguishable (inconclusive);
- none.
Cohort contrasts (R+ vs R−) and the cohort × stage interaction are reported with each.

## Controls
- Reaction time: whisker-hit drift adjusted for reaction time and with reaction times matched early vs late (reward-lick axis).
- Jaw kinematics: pre-lick jaw opening drift of whisker hits, whisker-hit vs auditory-hit jaw similarity early vs late, and the drift
  adjusted for jaw opening (reward-lick axis); movement state around passive trials, pre vs post (hit − miss axis).
- Jaw-aligned versions are not used (user 2026-10-15).

## Data
Inputs (not recomputed): ssl-prelick-convergence within_day/sl_stable (002 trial scores, 026 shift-null slopes, whole brain and areas),
ssl-hitmiss-axis-passive (135 min2 harmonised tables, 175 sessions), 051 / 105 per-event files, NWB DLC series (113 loader).
Learners rule for day 0 (definition under review), all expert sessions; session = unit.

## Results
combined_results_ks4/ssl-within-day-two-axes/{tables,figures}; report/ (report_lib).
