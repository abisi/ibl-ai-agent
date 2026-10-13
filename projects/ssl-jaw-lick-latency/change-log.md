# change-log

- 2026-10-13: project created (user: "estimate the reliability of the timing between jaw onset and lick time ... to infer a jaw
  opening time for sessions without video data"). 001 jaw-lick latency reliability: example trajectories, distributions, variance
  components, dependence on reaction time / class / stage / time in session, leave-one-session-out inference.
- 2026-10-13: 001 SSL_JAW_POP=all (every session with usable jaw video, no learners rule; outputs in all_sessions/), per-session estimates (tables/session_estimates.csv) and a within-session leave-one-out lead ~ RT model.
- 2026-10-13: 002 detector sensitivity (K = 4 / 7 / 10, velocity), prediction models M0-M6 (LOSO, conformal 90 % intervals), lick-referenced windows (contamination, offsets for 80-95 % clean events, model + margin windows). 003 neural validation of lick-referenced windows against the jaw-aligned reward-lick CD (087 jaw).
