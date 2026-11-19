# SSL sensory spatial maps

Where whisker and auditory stimuli evoke responses across the brain (single-neuron ROC maps, latency, modality
preference, bimodal neurons, response timing), and where whisker- and auditory-cortex projections converge (Allen
anterograde tracing). Companion project: `ssl-stimulus-arrival-decoding` (when stimulus information arrives, population
decoding).
Question and scope: `question.md`. History: `change-log.md`.
Results and report (not in git): `combined_results_ks4/ssl-sensory-spatial-maps/` on the NAS (the old
`_sensory_spatial_maps` path is a symlink to it).

## Scripts (`exploratory-analyses/`, run on haas)

Maps and projection zones

| Script | What it does |
|---|---|
| 001_unit_latency.py | half-time-to-peak latency of responsive neurons (whisker searched after +5 ms, artefact window) |
| 002_projection_zones.py | projection zones of SSp-bfd, SSs, AUDp, AUDd/v: all projection-neuron Allen lines (interneuron lines excluded), averaged per line then across lines; 70 % / 90 % contours (`ZONE_PCT`); merged whisker / auditory zones and overlap |
| 003_spatial_maps.py | coronal / sagittal / target slabs and density maps (3-D Gaussian, sigma 150 um, normalised by recorded-neuron density) |
| 004_modality_contours.py | location of whisker- vs auditory-preferring neurons within target slabs (within-session permutation, Holm) |
| 005_bimodal_convergence.py | superseded by 006 (kept for reference) |
| 006_colocation_figure.py | bimodal neurons inside vs outside the overlap: hierarchical bootstrap (sessions, then neurons) + Fisher |
| 007_cortical_flatmaps.py | Allen butterfly flatmap of the isocortex, with and without zone contours |
| 008_prelick_convergence_colocation.py | pre-lick converging neurons (SL reference, ssl-prelick-convergence) vs the overlap; outputs in `ssl-prelick-convergence/across_days/sl/projection_colocation/` |
| 009_shared_zone_roc_enrichment.py | shared-input level L = max(r_whisker, r_auditory) per voxel (continuous; L <= 0.5 = shared core): which ROC selectivities and bimodal neurons are enriched in the core (session-level tests, Mantel-Haenszel area control); outputs in `shared_zone_roc<_UNITS>/` |
| figure2_maps/ | paper Figure 2 map explorations (day 0, pooled): explore_sparse (half-A4 grid, 0.4-mm slabs, dots coloured by \|ROC sel\|), explore_density_maps (sigma 100 um density / mean selectivity), explore_projection_selectivity (Chen et al. 2024 Fig. 4/S4 style: projection density vs responsive neurons, quartile trends); older variants kept for reference; outputs in `figure2_maps/` |

Response timing

| Script | What it does |
|---|---|
| 010_fast_slow_peaks.py | fast (peak < 20 ms) vs slow (peak 20-95 ms) transient responses from the 001 peak times: peak-time distributions (KDE peaks, dip test), fraction per area group, peak reliability z; outputs in `fast_slow_peaks/` |
| 011_fast_slow_psths.py | PSTHs of fast vs slow excited responses per area group (recomputed as in 001, pre-trial baseline); outputs in `fast_slow_peaks/` |
| 012_late_peaks.py | excited neurons peaking late (50-95 ms): fraction per area group, sustained index, single-neuron heatmap; outputs in `fast_slow_peaks/` |
| 013_zeta_pilot.py | ZETA-test pilot on one session, no-lick trials (superseded by 014; outputs deleted 2026-11-19) |
| 014_zeta_all.py | ZETA test on no-lick trials (whisker misses, passive whisker / auditory), all sessions: fast / slow / late categories from the IFR peak, category PSTHs per area group, ZETA vs matched-context ROC; outputs in `zeta_all/` |

Per-session ZETA / latenZy results for every ROC analysis type (one-condition responses incl. negative and early-dip
neurons, two-condition contrasts, latencies) are produced by the ZETA pipeline of unit_spikes_analysis
(`zeta_analysis/`, `--analyses-single zeta`) into `combined_results_ks4/<mouse>/whisker_<day>/zeta/`.

Run order after new ROC tables or zones: 002 (both `ZONE_PCT`) -> 003 -> 004 -> 006 -> 007 -> 008 -> 009;
response timing: 001 -> 010 -> 011 -> 012; 014 independently.

## Report

- `report/build_article.py`: the article report (skills/project-report) in `combined_results_ks4/ssl-sensory-spatial-maps/report/`
  (last built 2026-10-05: covers 001-008; not yet 009-014).
- Legacy (before the project-report skill): `report/build_report.py` + `render.sh` (Quarto report) and
  `build_deck_assets.py` + `deck/build_deck.py` + `deck/render_deck.sh` (PowerPoint deck). The deck still reads its
  numbers from `report_numbers.json`, written by `build_report.py`.

## Decisions (user, 2026-10-04 / 10-08)

- 90 % projection zones are the main result; 70 % is reported as the stricter version (report, deck, schematic).
- Target-slab figures are dropped from report and deck; the within-target offset test (004) is kept.
- No cohort / stage splits: pooled is the scope of this project.
- Report and deck numbers come from the result tables, never typed by hand.
- Pre-lick converging neurons x projection overlap (008): null; added to the pre-lick convergence article as a
  supplementary figure.
- Joint (shared) projection zones use the continuous shared level L (009), not intersections of two loose zones.

## TODO

- [ ] 007 cosmetics: crowded tick labels on the recorded-neuron density colour bar; legend fragment in the whisker /
      auditory flatmap crop.
- [ ] Article report: add 009-014 (ask before building).
