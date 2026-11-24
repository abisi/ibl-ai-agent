# Question

## Request (Axel Bisi, 2026-11-19)
"Go back to my communication analysis with different methods. Check my results with that of
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3004095. Furthermore, try to replicate this type
of results, in particular using the same exact areas."
Follow-ups: "Compare with my wS1 and with MO-wM1 or MO-wM2. Separate my cohorts. Pull learning days and expert yes."
"Ok then you need to rerun, start with reproducing these results."

## Paper
Ranjbar-Slamloo & Kamigaki 2026, PLoS Biol 24(10): e3004095. Tactile delayed Go/NoGo, expert mice (d' > 1.5),
50 sessions / 18 mice, 32-channel probes in whisker S1 (AP -1.7..-2.0, ML 3-4) and M2 (AP 1.3..2.2, ML 0.5-1.0).
Time-shifted CCA: 5-ms bins, 50-ms windows every 10 ms, M2 shifted by delays (S1 leads > 0), first canonical
correlation (in-sample), total and noise (PSTH-subtracted). Main claims: transient S1 -> M2 coupling within 100 ms of
the stimulus (peak delay ~ +5 ms on Go), weak late feedback, coupling correlates with d' (r = 0.63).
Data (CCV maps only, no spikes) and figure scripts: Zenodo 10.5281/zenodo.21451128 and 10.5281/zenodo.23338822, copied
to combined_results_ks4/ssl-s1m2-communication/paper_zenodo.

## Design (reproduction stage)
- Areas: SSp-bfd vs MO-wM1 and SSp-bfd vs MO-wM2 (area_acronym_custom), separately; good + mua units >= 0.5 Hz,
  >= 20 per area. wM2 is closest to the paper's M2 coordinates.
- Sessions: all days; learning (day 0) and expert (day > 0); cohorts separate (reference sheet).
- Exact paper recipe and readout (s1m2_lib.CONFIG); paper values recomputed from their maps (001) for side-by-side.
- Conditions: All, W_all, W_hit, W_miss, A_hit, NS_all. Expert R+ whisker ~ paper Go; expert R- whisker ~ NoGo.
- Stats: session = unit; within cohort x stage rise vs 0 (Wilcoxon + mouse-level sign flip); R+ vs R- MW + Welch.

## Scripts
- s1m2_lib.py: loading, canoncorr port, time-shifted CCV maps, readout.
- 001_paper_reference.py: readouts from the paper's own maps.
- 002_sessions.py: session list.
- 003_ccv_maps.py: maps per session x pair x condition (haas).

## Dataset 2: Jules Lebert's context task (S1M2_DATASET=jl; request 2026-11-20)
Same three analyses (v3 paper recipe, option a clean cells, hires) on Jules Lebert's data, session = unit. Code: the
SSL scripts with `S1M2_DATASET=jl` (s1m2_lib switch; jl_001_units.py builds the unit table, jl_pages.py the figure
pages). Outputs: combined_results_ks4/ssl-s1m2-communication/jules_lebert/ (SSL outputs untouched).
- Data: /mnt/lsens-analysis/Jules_Lebert/NWB; 33 NWBs hold units (ephys); 30 have Allen-atlas labels (PB191, PB192:
  only ccf_* labels, no ccf_atlas_*; RD076: no labels; none of the three is in group.yaml ephys_context). RD072 / RD077
  (2 sessions each), JL002_20250522 and JL014_20251209 are listed in group.yaml but their NWBs have no units.
- Task: whisker_context, expert mice; whisker (3-ms pulse), auditory and no-stim trials in alternating context blocks
  (context 0 / 1, backgrounds brown / pink, ~20 blocks per session); context 1 = rewarded in every session (whisker
  reward_available 1 vs 0; auditory always rewarded). No passive trials, no perf 6, early_lick always 0.
- Areas: SSL definitions applied unchanged (allen_utils.process_allen_labels, split_merge_areas=True, on ccf_atlas_*
  labels): S1 = SSp-bfd; M2 = MO-wM1 + MO-wM2 pooled (MOs / MOp units on probes targeted at wM1 / wM2).
- Units: bombcell bc_label good / mua; stable = presence_ratio >= 0.5 (60-s bins) and coverage_ratio >= 0.9 (SSL 137
  rule; no DREDge drift test exists for this dataset, so that criterion is missing); >= 10 stable units per area.
- Sessions kept: 17 sessions / 10 mice (JL006 1, JL011 2, JL012 4, JL014 4, PB193, PB194, PB195, PB197, PB198, PB201);
  8 of them on Jules's ephys_context_expert list (secondary subset). PB198 is context day 0 (140 trials).
- Artefact: the whisker stimulus leaves a synchronous artefact in all areas (no spikes 0..3 ms, excess at -1 and +4 ms,
  S1-M2 zero-lag correlation peaks at -1 and +4 ms); M2 and non-whisker areas at baseline from +5 ms, S1 response from
  +5 ms. The SSL -10..+5 ms gap covers it with the same ~1-ms margin, so it is kept unchanged (jl artifact_check/).
- Conditions: All (both contexts) and All, W_all, W_hit, W_miss, A_hit, NS_all each per context (rew / nonrew);
  paper matching: rewarded whisker = Go, non-rewarded whisker = NoGo (lick: hit / FA, no lick: miss / CR). Noise CCV
  groups = trial type x lick x context.
- Stats: Wilcoxon vs 0 per context (session = unit); rewarded vs non-rewarded within session: paired Wilcoxon + paired
  t-test; uncorrected (exploratory). Context discrimination d' (whisker P(lick) rew vs nonrew) replaces the SSL d'.

## Connected pairs (extension, 2026-11-20)
Request (Axel Bisi): extend the time-shifted CCA to other area pairs known to be connected, among wS1, wS2, A1, PPC, DLS,
DMS, SC, MO-wM1, MO-wM2, TS, MO-ALM, MO-tjM1, MO-FC, MRN, RSP; control with weak / unconnected pairs and test whether
coupling scales with projection strength. Code: pairs_lib.py, 008-011 (s1m2_lib.py and 001-007 untouched); results in
combined_results_ks4/ssl-s1m2-communication/pairs/.

- Area labels (area_acronym_custom, unit_info_all_days_v2): wS1 = SSp-bfd, wS2 = SSs, A1 = AUD (no AUDp split in the
  table), PPC, DLS / DMS / TS (striatum split by bregma coordinates in allen_utils.create_areas_subdivisions), SC = SCs +
  SCm, MO-* = MOs / MOp by probe target; MO-FC = MOs / MOp of OFC-targeted probes (allen_utils cc65eac rule, applied in
  pairs_lib because the v2 table predates it); MRN; RSP.
- Units: as v3 (good + mua, 137 stable, no rate threshold); >= 5 units per area (PAIR_CONFIG min_units; user
  2026-11-20, v3 used 10; PCA keeps min(10, n units) dims).
- Connectivity (008), only for pairs with wS1 / wS2 / A1 at one end (the only extracted sources; the other 66 pairs are
  "connectivity not extracted"; next step would be Allen experiments for PPC, MO, striatum, SC, ... sources):
  - Allen: line-balanced projection density (ssl-sensory-spatial-maps 002 projection_zones.npz; A1 = AUDp injections)
    averaged over the unique 50-um CCF voxels holding the target's recorded units (all sessions, ML folded right,
    voxels inside the source excluded); enrichment = / mean candidate-voxel density. Class: strong if the mean density
    >= the 70 %-mass contour threshold of that source, weak if >= the 90 % threshold, none otherwise.
  - Liu et al. 2024 ipsilateral (wS1 / wS2 only), deepest Liu structure per unit; percentile among Liu structures:
    strong >= 90th, weak >= 50th. Structure totals are size-biased and coarse (DLS / DMS / TS share the CP value), so Liu
    calls almost every pair strong; the Allen class is used, Liu disagreement is flagged in the table.
  - Pairs of two sources: both directions computed, the stronger one sets the class. Orientation: the source (wS1 > wS2 >
    A1) is the leading area at delay > 0.
  - Coverage: a pair is analysed if >= 3 sessions at both stages in at least one cohort; statistics per cohort only
    where that cohort passes (ssl_area_inclusion_within_cohort).
- CCA (009): v3 profile (5-ms bins, 50-ms windows, PCA 10 -> CCA, in-sample + 10-fold CV + trial shuffle) and hires
  profile (1-ms bins, 10-ms windows, delays +-20 ms), conditions All, W_all, W_hit, W_miss, A_hit, NS_all, total and
  noise CCV. Readouts (010): v3 option a (readout_clean) FF / CV FF / FF - shuffle / FF - FB; hires CV and in-sample
  source-leading CCV for window starts 5-15 / 15-35 / 35-95 ms. Session = unit; Wilcoxon vs 0; R+ vs R- Mann-Whitney +
  Welch; uncorrected, exploratory.
- Projection strength vs coupling (010): within-session Spearman rho across a session's pairs (>= 3 pairs) with
  Wilcoxon of rho vs 0; mixed model readout ~ log10 enrichment + same-probe fraction + (1 | session); pair-level Spearman /
  Pearson of pair means (all pairs, and only pairs recorded on different probes).
- Probe exclusion: PAIR_CONFIG exclude_probes_csv (ssl-whisker-artefact-gap flagged probes; columns session_id,
  electrode_group, flagged); 010 writes *_noflag tables when a flagged table exists (none yet on 2026-11-20).
- Caveat: some pairs share a probe (wS2-A1, wS2-TS, A1-TS, wS1-DLS, wS2-DLS): shared reference / common-mode noise
  can add zero-lag coupling; same_probe_frac is stored per row and used as a covariate.
- Scripts: pairs_lib.py (areas, units, pair list, probe-exclusion option), 008_connectivity_pairs.py (step 1),
  009_pairs_ccv.py (maps, haas), 010_pairs_stats.py (readouts, stats, scaling tests), 011_pairs_figures.py.
