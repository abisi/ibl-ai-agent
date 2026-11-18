# Change log

- 2026-10-05: article-style report built (skills/project-report): combined_results_ks4/ssl-sensory-spatial-maps/report/ (report.tex, report.pdf, report.md, report.html, figures/); published (PDF, md, html) to the private repo abisi/ibl-ai-agent-reports, docs/ssl-sensory-spatial-maps/.

- 2026-10-08: 009 shared-input EDA (user): per-voxel mass rank per modality, shared level L = max(r_whisker, r_auditory) (strictest contour at which a voxel is in both zones); ROC enrichment (all analysis types, frac sig total / per sign and |sel|) in the shared core (L <= 0.5) vs rest, session-level Wilcoxon / t and area-stratified Mantel-Haenszel, BH; stable units. 002 also run at ZONE_PCT=50 / 30; 008 at 50 (stable units).
- 2026-10-08: 009 also tests bimodal neurons (003 classes) against the shared level.
- 2026-11-18: 010_fast_slow_peaks.py (user question: fast < 20 ms and slower < 100 ms transient peaks, from latency): peak-time distributions per modality and area group, from the 001 table, no new spike processing.
- 2026-11-18: 011_fast_slow_psths.py: PSTHs (recomputed as in 001, pre-trial baseline) of fast vs slow excited neurons per area group, Hz and peak-normalised, mean +- s.e.m. over sessions.
- 2026-11-18: 012_late_peaks.py: excited neurons peaking 50-95 ms (from 011 PSTHs): fraction per area group, sustained index, single-neuron heatmap and per-area PSTHs.
- 2026-11-18: 013_zeta_pilot.py: ZETA-test (zetapy 4.2, installed in ~/zeta_lib on haas) pilot on one session
  (MH069_20260122_111455), no-lick trials only: active whisker misses, passive whisker, passive auditory; all good + mua
  units; window 0-200 ms; IFR peak time; compared with the 001 ROC flag. Outputs in _sensory_spatial_maps/zeta_pilot/.
- 2026-11-18: 014_zeta_all.py: ZETA test on no-lick trials, all sessions (W_miss; passive whisker / auditory pre, post, pooled), all ROC-table units; excited = ZETA p < 0.01 + upward PSTH deviation (z >= 3); IFR-peak categories fast / slow / late; method figure, category PSTHs per area group, ZETA vs matched-context ROC. Outputs in _sensory_spatial_maps/zeta_all/.
