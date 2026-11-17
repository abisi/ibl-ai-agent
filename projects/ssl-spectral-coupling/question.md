# Question

## Request (Axel Bisi, 2026-11-17, overnight, autonomous)
"You piloted an internal communication project. I want you to do the same types of analyses and hypothesis testing
using spectral decomposition (fft style) using population activity (stable units only) from ap band and binned spike
activity. Be autonomous and run overnight. My hypothesis is that in the quiet window or response window (early sensory
or execution), R+ and R- have different coupling between area group, area acronyms."

## Relation to earlier work
- Follows `ssl-communication-methods-pilot` (RRR, CCA, pCCA, partial RRR, Granger; time-domain). Same sessions, area
  groups (`area_group_coarse_v3`), stable unit set (137_stable_units), trial rules, draws/shuffle-null logic, statistics
  (session unit, MW + Welch, BH over pairs, learners + all mice). This project replaces the time-domain measures by their
  frequency-domain counterparts.

## Interpretation of "ap band and binned spike activity" (decision taken autonomously, to confirm with the user)
- **Binned spikes (primary):** spike counts of stable good + mua units in 5-ms bins (200 Hz, Nyquist 100 Hz).
- **AP band (secondary):** a continuous multi-unit signal from the raw AP band (`*_tcat_corrected.imecN.ap.bin`,
  artifact-corrected), restricted to the channels that carry stable units of each area: band-pass 300-3000 Hz, rectify,
  low-pass and resample to the same 5-ms grid (the "MUA envelope"). It contains unsorted and below-threshold spiking of
  the same channels. Probe clock to NWB time from KS4 `spike_times.npy` (samples) vs `*_spike_times_sec_sync.npy`.
- AP band is high-passed (> 300 Hz), so it carries no LFP; "spectral" here is the spectrum of population spiking
  fluctuations (1-100 Hz), not of field potentials.

## Signal
- Trial-to-trial fluctuations: per-unit counts minus the condition mean per bin, soft-normalised per unit (pilot "z"
  preprocessing), so PSTH-locked co-modulation does not count as coupling.
- Per area and draw: 15 random stable units (area group) or 8 (area acronym); population signal = mean of the unit
  residuals ("sum" signal) and the top 3 PCs of the unit residuals.

## Windows (stimulus-aligned, trial start t = 0)
- **quiet:** -1.0 to 0 s (inside the required >= 2 s no-lick quiet window; trials with early licks are excluded).
- **early (early sensory):** 0.005 to 0.155 s (after the -1/+4 ms artifact dead zone, before most jaw onsets, median ~0.2 s).
- **execution:** 0.2 to 0.5 s (licking / response on lick trials).
- **time-resolved:** 200-ms windows every 50 ms, centres -0.9 to 0.8 s (descriptive).
- Multitaper: NW = 3 for quiet (1 s, +-3 Hz), NW = 1.5 for early (+-10 Hz), NW = 3 for execution (+-10 Hz), NW = 2 for
  200-ms windows (+-10 Hz).

## Measures (frequency-domain analogue of each pilot method)
| pilot (time domain) | here (frequency domain) | what it captures |
|---|---|---|
| noise correlation of summed signals | **MSC**: magnitude-squared coherence of the two "sum" signals | linear coupling per frequency |
| CCA (PCA -> CCA) | **canonical coherence** (3 PCs per area; Brillinger) | best coupling over population combinations |
| pCCA / partial RRR | **partial coherence** of sum signals given the top 3 PCs of 200 other units | coupling not explained by the rest of the recording |
| (zero-lag common input check) | **imaginary coherency** | lagged coupling only |
| lagged-RRR direction index | **phase-slope index** (PSI, Nolte 2008) per band | which area leads |
| Granger | **spectral Granger** (Geweke 1982; bivariate VAR, order 4 = 20 ms, ensemble over trials) | directed coupling per frequency |
- Bands: theta 3-8, alpha 8-13, beta 13-30, low gamma 30-60, high gamma 60-100 Hz (theta/alpha only in the 1-s quiet
  window, where the resolution allows).
- Null: 20 trial shuffles of the source per draw; every quantity is reported as excess over the shuffle mean (removes
  the trial-count bias of coherence).
- Controls: matched units (15/8) and matched trials per draw (quiet 60, event windows 15), 3 draws; within-area ceiling
  (disjoint 15 units of the same group, groups >= 30 units); per-area power spectra and firing rates per cohort;
  simulations with known coupling (001).

## Conditions
- quiet: `ALL` = every included active trial (residualised per trial type x lick); also per condition.
- early / execution / time-resolved: W_lick, W_nolick, A_hit (hits; Go in both cohorts), NS_cr (control).
- Trials: context active (per-trial rule as the pilot loader), perf != 6, early_lick == 0, warm-up block cut (keep 1
  trial before the first whisker trial), A1 end-of-session trim.

## Pre-registered tests (written 2026-11-17 before any real-data result)
- **H1 (quiet):** R+ and R- differ in excess MSC between area groups in the quiet window (ALL trials).
- **H2 (early sensory):** R+ and R- differ in excess MSC in the early window, on W_lick and W_nolick trials.
- **H3 (execution):** R+ and R- differ in excess MSC in the execution window, on W_lick trials.
- **Specificity:** for H2/H3 the same comparison on A_hit (Go in both cohorts) and the whisker - auditory difference.
- **Levels:** area group (primary), area acronym (secondary).
- **Per test:** per pair and band, session-level excess (mean over draws), R+ vs R- with Mann-Whitney and Welch, both
  reported; BH-FDR over pairs within (level, window, condition, band, measure). Omnibus per session: mean excess over
  all its pairs (one value per session), MW + Welch. Frequency-cluster permutation (cohort labels permuted over
  sessions, cluster mass over frequencies) on the full MSC spectrum per pair.
- **Supported** only if the MSC effect survives BH and the canonical coherence or partial coherence shows the same sign.
- Secondary: imaginary coherency, partial coherence, canonical coherence, PSI, spectral Granger (directed).
- Population: learners (sheet learning_category good/moderate) primary, all mice sensitivity. Unit = session (one per
  mouse at day 0). A pair enters a cohort with >= 3 sessions; cohort tests need both cohorts.
- Results are exploratory/pre-specified; confirmation on a held-out set is not possible with one session per mouse.
