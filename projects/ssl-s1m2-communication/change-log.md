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
- **2026-11-19: v3 = paper recipe on all stable units (user: "keep all units, stable, >= 10 per area"; pool wM1 + wM2).**
  - v2 (all units, plain in-sample CCA) stopped: CCV saturates at ~1 with ~150-270 units per area. v3: each area reduced
    to its top 10 PCs (one PCA per area and condition, all bins) before CCA; 10-fold CV control refits the PCA on
    training trials. Artefact: spikes -10..+5 ms around every trial start removed, no Poisson fill. Sessions_v2: 39
    sessions; pooled S1-wMO learning R+ 12 / R- 14, expert R+ 6 / R- 7. 232 jobs, no failures.
  - All trials, total CCV, FF 0-100 ms: paper +0.038 (n = 50, p = 3e-5); SSL learning R+ +0.052 (p = 0.005),
    R- +0.059 (p = 0.005); expert R+ +0.093 (p = 0.06), R- +0.024 (p = 0.08); CV versions p = 0.005 / 0.005 / 0.03 /
    0.05. Same delay (0..+10 ms) patch in the maps. FF > FB: paper p = 0.09, SSL R- learning p = 0.002, others n.s.
    SSL adds a broad 100-250 ms rise (movement / licks) absent in the paper. Noise CCV: null, as in the paper's data.
  - 005_replication_figure.py: one page per trial type (All, W_all, W_hit, W_miss, A_hit) in replication/.
- **2026-11-19: option a (artefact-free windows only; user).** s1m2_lib.gap_mask / readout_clean: a map cell is used only
  if both the S1 and the M2 50-ms window are clear of -10..+5 ms (baseline cells up to -40 ms, early cells +30..+100 ms);
  same mask on the paper's maps. replication/replication_total_clean.pdf.
  - Paper transient survives (it sits at 20-70 ms). In SSL most of the sharp delay-0 patch was in excluded cells:
    blanked rows are zero in both areas at the same samples and create shared variance at delays near 0, so the earlier
    patch is at least partly an artefact of the blanking.
  - All trials, clean: SSL learning R+ +0.045 (p = 0.03), R- +0.051 (p = 0.01), expert n.s.; driven by a broad rise over
    all delays (FF - FB ~ 0 in every SSL group; paper FF - FB p = 0.17 on the same cells). Whisker hits clean: in-sample
    n.s.; CV R+ / R- learning p = 0.01 / 0.04; a small delay 0..+15 ms patch at 30-50 ms in R- learning only.
  - Shorter windows are needed to see 5-30 ms: first clean centre = 5 ms + half window (50 ms: +30; 20 ms: +15;
    10 ms: +10).
- **2026-11-19: artefact check (006) + high-resolution run (profile "hires", maps_v4_hires; v3 kept).**
  - 006, raw spikes 1-ms bins, 39 sessions: on whisker trials every area (S1, M2, non-whisker areas) shows a dip at
    0-2 ms and a sharp spike at 3-4 ms; back at baseline by +5 ms in M2 and other areas, S1 sensory rise from +5 ms
    (peak ~13 ms). No extra S1-M2 zero-lag synchrony after +5 ms. The -10..+5 ms gap covers the artefact (~1 ms margin).
  - hires: 1-ms bins, 10-ms windows stepped 1 ms (labelled by start), delays -20..+20 ms (1 ms), PCA -> CCA, only cells
    with both windows clear of the gap; 233 jobs, no failures. 007_hires_figure.py -> replication/hires_total.pdf.
  - CV CCV, S1 leading 0..15 ms, windows starting 5-15 ms: whisker trials learning R+ +0.046 (p = 0.012), R- +0.062
    (p = 0.003), expert R+ +0.013 (p = 0.03, n = 6), R- n.s.; whisker misses learning p = 0.014 / 0.013; whisker hits
    p = 0.15 / 0.11; auditory hits and no-stim (negative control) n.s. in every group. Maps: S1 leads M2 by 0..+7 ms in
    the first windows at learning. S1-leading minus M2-leading (windows 25-95 ms) n.s. everywhere.
