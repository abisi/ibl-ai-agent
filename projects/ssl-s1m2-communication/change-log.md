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
- **2026-11-20: connected pairs, unit threshold 10 -> 5 per area (user).**
  - Coverage check (latest NWB_ks4; unit tables rebuilt 2026-11-19 after the last NWB change): min 10 -> 46 / 105
    pairs pass in >= 1 cohort (23 in both); min 5 -> 52 (27). Session gains are 0-1 per cohort x stage: coverage is
    limited by co-recording, mostly expert sessions. Newly passing: wS2-RSP, wS2-MRN, A1-MRN, A1-RSP (extracted),
    DLS-RSP, DMS-TS (not extracted, not run). wS1-wS2 / wS1-A1 / wS1-PPC / wS1-TS still fail.
  - PCA keeps min(10, n units) dims, so 5-9-unit areas enter CCA with fewer dims. MH019_20250306_121844 is in
    137_stable_units but not in unit_info_all_days_v2 (unexplained, to check).
  - 009 --skip-done now runs only the missing pairs of a session; running v3 / hires sweep stopped and relaunched.
- **2026-11-20: Jules Lebert's context dataset (user: same S1 -> M2 analyses, session = unit).** question.md "Dataset 2".
  - Code: dataset switch `S1M2_DATASET=jl` in s1m2_lib (NWB dir, OUT/jules_lebert, context-split conditions, context
    labels, electrode_group names); jl branches in 002 / 003 / 004 / 005 / 007; jl_001_units.py (unit table, trial
    summary); jl_pages.py (figure pages). SSL behaviour unchanged (default ssl).
  - Coverage: 33 NWBs with units, 30 with atlas labels; 17 sessions / 10 mice with >= 10 stable units in SSp-bfd and in
    MO-wM1 + MO-wM2 (median 93 / 164); 8 on Jules's expert list. Dropped (13 labelled): S1 but no wM1/wM2 units
    (JL002_0507, JL005_0520, JL006_0602), wM1/wM2 but no S1 (JL002_0523, JL011_1106, PB196), neither (JL007_0603,
    JL007_0605, JL012_1204-1206), < 10 stable S1 units (JL007_0523: 2, PB200: 4); unlabelled (PB191, PB192, RD076);
    no units in the NWB (RD072 x2, RD077 x2, JL002_0522, JL014_1209).
  - Artefact check (006 on jl): same whisker artefact as SSL (spikes absent 0..3 ms in all areas, synchronous S1-M2
    excess at -1 and +4 ms); baseline in M2 / other areas from +5 ms; -10..+5 ms gap kept.
  - Pilot 3 sessions (2.5 min, < 3 GB); full v3 192 jobs and hires 221 jobs, no failures. W_hit in the non-rewarded
    context reaches 20 trials in only 8 sessions, W_miss rewarded in 9.
  - v3, total, FF 0-100 ms: All +0.110 (n = 17, p < 1e-4; FF - FB p = 0.002; CV +0.13); whisker rewarded +0.080,
    non-rewarded +0.098 (both p = 2e-4; non-rewarded > rewarded, paired Wilcoxon p = 0.04, t p = 0.03); auditory hits
    ~0; no-stim +0.011. Paper All +0.038. Clean cells (30-100 ms): whisker +0.056 / +0.085 (p = 0.04 / 0.002), broad
    over delays, FF - FB ~ 0. Noise CCV: null for All / whisker. JL011_20251105 is an in-sample outlier on whisker trials
    (FF 0.6-0.8, CV ~0.03-0.24); conclusions hold without it. No correlation with context d' (|r| < 0.45, n.s.).
  - hires (CV, S1 leading 0..15 ms, windows 5-15 ms): whisker non-rewarded +0.017 (p = 0.003), rewarded +0.013
    (p = 0.06); 15-35 ms p = 0.008 / 0.02; All +0.07 (p < 1e-3); auditory hits, no-stim n.s.; S1-leading minus
    M2-leading n.s. everywhere; contexts not different for whisker trials (paired p > 0.5).
- **2026-11-20: JL RSP vs M2 (user: "For JL; now do RSP vs M2"). jl_rsp_m2.py, outputs jules_lebert/rsp_m2/.**
  - Same recipe as JL S1-M2 (v3 + hires, >= 10 stable units, same trials / conditions / artefact gap); RSP in the S1
    slot (delay > 0 = RSP leads). 17 sessions / 11 mice for RSP x wMO (RSP units all on RSP-targeted probes), 14 for
    wM1, 6 for wM2. 220 v3 + hires jobs, no failures. 002 / 003 / 004 / 005 / 007 reused unchanged via the wrapper;
    jl_pages relabelled at load time (paper S1-M2 maps / values kept as grey reference, labelled as such).
  - v3 total, wMO, all cells: All trials RSP-M2 +0.063 (p = 0.001; S1-M2 +0.110, p < 1e-3); option a +0.059
    (p = 0.004). Whisker trials: non-rewarded context +0.037 (p = 0.02, clean p = 0.035), rewarded n.s.; rew vs nonrew
    n.s. Whisker misses rew < nonrew (paired p = 0.012 / t 0.028; S1-M2 same direction p = 0.027). No-stim n.s.
    FF - FB n.s. everywhere (S1-M2: All p = 0.002): RSP-M2 coupling is not directional.
  - hires (windows from +5 ms, RSP leading 0..15 ms): no early RSP -> M2 coupling on whisker trials (CV p >= 0.2 in
    every context x outcome); S1-M2 shows it (W_all nonrew p = 0.003). All trials: small early rise (CV p = 0.001).
- **2026-11-20: connected-pairs extension (question.md "Connected pairs"; pairs_lib.py, 008-011; outputs in pairs/).**
  - Step 1 (008): Allen recorded-site projection density for sources wS1 / wS2 / A1 (AUDp) and Liu et al. (wS1 / wS2);
    Allen class by the 70 % / 90 % mass-contour thresholds; Liu percentile classes call nearly every pair strong (size-
    biased structure totals) and are only flagged. 39 pairs with a source end, 66 "connectivity not extracted".
    Unit threshold changed from 10 to 5 stable units per area (user, mid-run): 21 extracted pairs pass >= 3 sessions at
    both stages in >= 1 cohort (strong 5: wS1-DLS, wS2-DLS, wS1-MO-wM1, wS1-MO-wM2, wS2-A1; weak 4: wS1-SC, A1-DLS,
    A1-SC, wS2-MO-wM1; none 12 incl. wS1-DMS, wS1-RSP, wS1-MO-ALM, wS1-MO-tjM1, wS1-MRN, A1-DMS).
  - Step 2 (009): v3 + hires profiles, 6 conditions, total + noise; 91 sessions, 436 session-pairs, 853 jobs, no
    failures (maps from the first min-10 run reused: unit sets identical). 010 / 011: stats, scaling, figures.
  - Main readout (hires CV, source leads 0..15 ms, windows starting 5-15 ms), whisker trials, learning: coupling is
    present in strong and weak pairs (e.g. wS1-DLS R+ / R- p = 0.016 / 0.039, wS1-MO-wM1 R- p = 0.004, wS1-MO-wM2 R+
    p = 0.03, wS1-SC R- p = 0.008) and ~0 in most no-projection pairs (wS1-DMS, wS1-RSP, wS1-MO-ALM, A1-DMS / MRN /
    RSP), but also present in wS1-MRN and wS2-SC (class none in Allen; reachable via subcortical whisker routes).
  - Scaling with projection strength (whisker trials, learning, hires CV 5-15 ms): within-session rho R+ 0.40 (n = 29,
    p = 3e-4), R- 0.37 (n = 25, p = 9e-4); mixed model slopes p = 1e-4 / 3e-4, still p = 0.002 / 0.004 adjusted for
    unit counts; pair-level Spearman R+ 0.48 (p = 0.03), R- 0.58 (p = 0.007), also without shared-probe pairs. Absent on
    no-stim and auditory-hit trials, in late windows (35-95 ms) and at expert (few sessions). v3 50-ms readout: weaker.
  - R+ vs R-: mostly n.s.; exploratory exceptions wS1-MO-wM1 (R- > R+, MW p = 0.036, Welch p = 0.08).
  - Whisker dead-time probe exclusion: option in PAIR_CONFIG / 010 --flagged; no flagged table yet, not applied.
