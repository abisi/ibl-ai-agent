# Change log

- **2026-11-19: project created (user: reproduce Ranjbar-Slamloo & Kamigaki 2026 with the same areas).**
  - Existing results checked first: ssl-cross-area-communication fine-area runs (2026-09, SSp-bfd x MO-wM1 9/8,
    x MO-wM2 6/7 sessions, day 0) used a single zero-lag CCA per session; they can neither confirm nor refute the paper.
  - Coverage: 38 sessions / 33 mice with >= 20 good+mua units (>= 0.5 Hz) in SSp-bfd and MO-wM1 or MO-wM2; probe
    sheet agrees (45 wS1 + wM1/wM2 co-insertions, 32 valid with units in the table, AB077 / AB105 excluded).
    Expert sessions are few (wM1: R+ 1, R- 4; wM2: R+ 4, R- 3).
  - Paper Zenodo data are CCV maps (no spikes) + figure scripts; copied to combined_results_ks4/<slug>/paper_zenodo.
  - 001 reproduces the paper's numbers from their maps with our readout code: Total All FF vs d' r = 0.63
    (p < 1e-5), Go r = 0.60, NoGo n.s. (paper: 0.63 / 0.60 / n.s.).
  - 003: canoncorr port, time-shifted maps (delays +-100 ms, 5-ms bins, 50-ms windows every 10 ms), one trial-shuffle
    map, units capped at 50 per area (random draw; paper ~50; in-sample CCV depends on units / samples). First uncapped
    test (170+ units) was too slow and not paper-matched; stopped.
- **2026-11-19: first full run (003: 38 sessions, 43 session-pairs, no failures; 004 stats + figures).**
  - Paper map (their data): total CCV transient 20-70 ms after the puff at delays 0..+10 ms; readout Total All FF
    +0.038 (n = 50, Wilcoxon p < 1e-4); noise readout n.s. with this 0-100 ms readout.
  - SSL, total CCV, pooled S1-wMO, learning: R- All FF +0.040 (n = 14, Wilcoxon p = 0.013, mouse sign-flip
    p = 0.006, FF > FB p = 0.011, FF > shuffle p = 0.013); R+ All +0.004 (n = 12, n.s.). Maps: R- learning shows the
    paper-like 0-50 ms, delay 0..+10 ms patch on W_all / W_miss / All in both MO-wM1 and MO-wM2; R+ learning does not.
    R+ vs R- not significant (pooled All MW p = 0.14, Welch p = 0.07). W_hit, A_hit: no rise (FF below baseline).
  - Expert: 12 sessions in total (R+ 5, R- 7 pooled), too few to test.
  - Caveat: the whisker-artefact correction replaces spikes in -10..+5 ms with independent Poisson spikes, diluting
    windows centred before ~+30 ms on whisker trials.
