# Question

## Request (Axel Bisi, 2026-11-05)
"Overnight, I want you to do a small pilot at inter-areal interactions between area groups, comparing different
methods (granger causality, pca-to-cca, rrr, using established methods from the literature); do extensive controls,
and use several mice. Test the hypothesis that communication between pairs of area groups differs across cohorts,
following different trial types; show time-resolved start-time aligned data that show these communication metrics."

## Design (pilot, exploratory)
- **Relation to earlier work.** It follows on from `ssl-cross-area-communication`, a partial-CCA project from 2026-08 and 09. This pilot reuses that project's v3 area groups and its unit-quality thresholds.
- **Sessions.** Whisker day 0 (learning), one session per mouse.
  - 8 R+ learners: AB087, AB127, AB130, AB149, AB121, MH032, AB150, AB147.
  - 8 R- learners: AB085, AB093, AB116, AB122, AB126, AB128, AB140, MH034.
  - Chosen as the sessions with the most good units in all three area groups, at least 40 per area.
- **Area groups.** Motor and frontal areas, SSp-w and Striatum and pallidum: three pairs, analysed in both directions.
- **Trials and conditions.**
  - Active trials, perf != 6, early_lick == 0.
  - Four conditions: whisker hit, whisker miss, auditory hit, no-stimulus correct rejection.
  - 30 trials per condition and draw.
- **Signal.** 10-ms spike counts minus each unit's condition-mean PSTH, i.e. trial-to-trial co-fluctuations.
  - Spikes are loaded with ephys_utilities `process_single_nwb`, which applies the lab's whisker-artifact correction.
- **Time resolution.** 100-ms windows every 20 ms, from -0.4 to +1.0 s around trial start.
- **Methods.**
  - **RRR:** reduced-rank ridge regression, cross-validated by trial, with the optimal rank (Semedo et al. 2019).
  - **PCA→CCA:** held-out first canonical correlation.
  - **Ensemble Granger:** conditional Granger causality on the top 3 PCs, 150-ms windows, order 3 (Ding et al. 2000).
  - **Time-reversed Granger:** a directionality check (Haufe et al. 2013; Winkler et al. 2016).
- **Controls.**
  - Trial-shuffle null for every method.
  - Within-area RRR as a ceiling.
  - Matched numbers of units (20 source, 20 target) and trials, over 2 random draws.
  - Firing rates and trial counts per cohort.
  - Jaw and whisker movement (dlc_helpers) regressed out.
  - Synthetic positive and negative controls (001).
- **Statistics.**
  - R+ vs R- per window with Mann-Whitney and Welch, both reported, uncorrected.
  - Cluster permutation over time.
  - Within-cohort comparison against zero.
  - The unit is the mouse (one session each).

## Scripts
- `comm_lib.py`: loading, binning and the methods.
- `001_simulation_validation.py`: synthetic validation.
- `002_run_pilot.py`: per-session computation, run on haas.
- `003_stats_and_figures.py`: aggregation, tests and figures.
- `004_method_explainer_figures.py`: figures explaining each method step by step.
- `005_method_benchmark.py`: benchmark on synthetic data with known ground truth.
- `006_coverage_table.py`: sessions and trials available per area pair and design cell.
- `007_run_hypothesis.py`: full-dataset run for the hypothesis test below.

Results are in `combined_results_ks4/ssl-communication-methods-pilot/`.

## Hypothesis test v2 (pre-registered 2026-11-06, before any real-data result of 007)

### Request (Axel Bisi, 2026-11-06)
"An hypothesis to test: in R+, specific pairs of areas allow for information to go through areas (a go signal shared
across areas, possibly downstream of S1-w); in R-, other areas act as top-down modulation (NoGo) signal. How to test
this. How to contrast with auditory trial (hits only), where it is Go for both?"
Follow-up: "Autonomously proceed with this, benchmarking when possible methods. Include into the mix CCA, pCCA."

### Predictions
- **P1, R+ feedforward Go.**
  - On whisker lick trials, R+ sessions show a larger A→B lead than R- sessions from SSp-w to wMO, ALM-tjM1 and Striatum.
  - The lead is measured by the lagged-RRR direction index (A→B minus B→A) and by Granger with time reversal (net TRGC).
  - Expected in early windows (0–150 ms after the stimulus), before jaw onset.
- **P2, R- top-down NoGo.**
  - On whisker no-lick trials, R- sessions show a larger lead than R+ sessions from Frontal to SSp-w, Striatum, wMO and ALM-tjM1.
  - Expected after the stimulus, or already before it (anticipatory gating).
- **P3, modality specificity.**
  - The cohort difference in P1 is larger on whisker licks than on auditory hits, which are Go in both cohorts.
  - Tested per session as the whisker-lick value minus the auditory-hit value, then R+ vs R-.
  - This is the cohort × modality interaction.
- **Positive control.** VPM-PO → SSp-w should show a feedforward lead in both cohorts. If it does not, the directional measures are not trusted.

### Primary measure and windows
- **Main quantity:** the lagged-RRR direction index, in excess over its trial-shuffle null.
  - The benchmark (005 v3) chose this measure. It has the highest power: 93–97% for true feedforward or feedback at coupling 1, and it gets the direction right in 93–97% of those.
  - Its false-positive rate is 3%.
- **Required robustness check:** the same index from lagged RRR conditioned on the target's own past and the rest of the recording (lagRRR|rest).
  - In the benchmark it removes recorded common input (spurious leads 77% → 7%) and most shared gain (detection 100% → 23%).
  - The cost is power: 50–60% at coupling 1.
  - A cohort effect counts as supported only if the main index shows it and lagRRR|rest has the same sign.
- **Secondary quantities:**
  - net TRGC, from Granger and from Granger conditioned on all other recorded units ("rest");
  - lagged-RRR in each direction separately;
  - pCCA conditioned on rest, then CCA and same-bin RRR, which are undirected.
- **Primary windows:**
  - early, centres 0.05–0.15 s;
  - late, centres 0.2–0.4 s;
  - pre-stimulus, centres −0.3 to −0.1 s.
  - Each is the mean over its window centres.
- **Pre-movement check:** P1 is also tested on lick trials with jaw onset ≥ 150 ms, using windows that end by 150 ms.

### Design
- **Sessions:** whisker day 0, both cohorts, with a jaw-onset table and not `exclude_ephys`.
  - The primary analysis uses learners (sheet learning_category good or moderate); a sensitivity analysis uses all mice.
- **Units:** good units, ≥ 1 Hz, 20 per area per draw.
- **Trials:** 15 per condition per draw, 5 draws averaged.
- **Pairs:** the ten in `007_run_hypothesis.CONFIG2["pairs"]`.
  - A pair enters a cohort's analysis only with ≥ 3 sessions in that cohort. Cohort comparisons need both cohorts to pass.
- **Residuals:** the condition mean per unit and bin.
  - A version that also removed a per-area trial-wise gain was benchmarked and dropped: it cut power and did not remove shared gain for the lagged methods.
  - lagRRR|rest is the gain control instead.
- **Binning:** 20-ms bins, 200-ms windows, 3 lags (60 ms).
- **Unit of analysis:** the session, one per mouse at day 0.

### Statistics
- **Per session:** the excess over the null, averaged over draws.
- **R+ vs R-:** Mann-Whitney and Welch, both reported.
- **P3:** the same tests on the per-session whisker − auditory difference.
- **Corrections:** Benjamini-Hochberg FDR across pairs, within each prediction and window. Time-resolved curves are descriptive, with cluster permutation.

### Known limits (from the benchmark)
- **Unrecorded common input** produces a spurious lead in every method: 70% for lagged RRR, 37% for lagRRR|rest. Even Granger conditioned on the true driver leaves about 40%, because spiking is nonlinear.
- **Shared trial-wise gain** (arousal-like) looks like communication in every undirected method and in Granger.
- **pCCA:**
  - The partial_CCA 0.1.0 package works: it removes recorded common input, and plain CCA through the package matches our own CCA.
  - But CCA-family measures are the least sensitive at these unit counts.
  - The package scores in-sample and ignores Z in `transform`, so cross-validation and the held-out removal of Z are done in `comm_lib.pcca_cv`.
- **DLAG and mDLAG** are not benchmarked: they exist only in MATLAB, and haas has no MATLAB.
- **Interpretation:** results count as directed functional interactions, not causal routing.

## Time-resolved RRR across all area groups (2026-11-09)

### Request (Axel Bisi, 2026-11-09)
"Rerun the RRR communication pilot, use more area groups (e.g. hippocampus, midbrain). Use information in
https://arxiv.org/html/2512.12467v1; show time-resolved communication. Highlight R+ and R- differences."

### Method
- The method follows Wu & Pillow 2025, "Reduced rank regression for neural communication: a tutorial for neuroscientists", arXiv 2512.12467.
- **Model.** Ridge-RRR: rank-r projection of the ridge solution onto the eigenvectors of W'X'XW. This is not the truncated SVD of W.
- **Hyperparameters.** The ridge penalty and the rank are chosen by cross-validation, with folds split by trial. The rank is the smallest within one s.e.m. of the peak.
- **Centering.** Per time bin: the condition PSTH is removed per unit and bin.
- **Quantities per window:**
  - held-out R²;
  - the chosen rank;
  - the communication fraction CF = Tr(W'ΣxW)/Tr(Σy), computed in-sample as in the tutorial.
- **Null.** Each quantity is reported against a 10-permutation trial-shuffle null.
- **Areas.** All v3 coarse area groups with at least 20 good units (≥ 1 Hz), and every ordered pair. Same-bin RRR is not directional.
- **Trials.** W_lick, W_nolick and A_hit; 20 trials and 20 units per draw; 3 draws.
- **Time.** 20-ms bins in 200-ms windows, every 50 ms, from -0.4 to +0.8 s.
- **Statistics:**
  - unit: the session, one per mouse at day 0;
  - a pair needs at least 3 sessions per cohort;
  - cohort cluster permutation over time, with BH correction over pairs;
  - epoch tests (pre, early, late) with Mann-Whitney and Welch, BH over pairs;
  - a whisker-lick minus auditory-hit interaction.
- **Populations.** Learners are the primary analysis; all mice are the sensitivity analysis.
- **Scripts.** `010_rrr_timeresolved.py` (run) and `011_rrr_timeresolved_stats.py` (statistics and figures).
- **Outputs.** `rrr_timeresolved/{learners,all_mice}/`.

## Request (2026-11-09)
"Did you run these analyses on the entire dataset? Can you update the large scale, comparing several methods, including methods that allow to partialize other areas out. Also, run an analysis where you just have somatosensory cortex, auditory, midbrain, thalamus, hippocampus, frontal cortex, striatum. Gather all in one place, report, and make publication quality figure."
