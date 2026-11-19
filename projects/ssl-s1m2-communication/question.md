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
