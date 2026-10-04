# SSL sensory spatial maps

Where whisker and auditory stimuli evoke responses across the brain (single-neuron ROC maps, latency, modality
preference, bimodal neurons), and where whisker- and auditory-cortex projections converge (Allen anterograde tracing).
Companion project: `ssl-stimulus-arrival-decoding` (when stimulus information arrives, population decoding).
Question and scope: `question.md`. Results (not in git): `combined_results_ks4/_sensory_spatial_maps/` on the NAS.

## Scripts (`exploratory-analyses/`, run on haas)

| Script | What it does |
|---|---|
| 001_unit_latency.py | half-time-to-peak latency of responsive neurons (whisker searched after +5 ms, artefact window) |
| 002_projection_zones.py | projection zones of SSp-bfd, SSs, AUDp, AUDd/v: all projection-neuron Allen lines (interneuron lines excluded), averaged per line then across lines; 70 % / 90 % contours (`ZONE_PCT`); merged whisker / auditory zones and overlap |
| 003_spatial_maps.py | coronal / sagittal / target slabs and density maps (3-D Gaussian, sigma 150 um, normalised by recorded-neuron density) |
| 004_modality_contours.py | location of whisker- vs auditory-preferring neurons within target slabs (within-session permutation, Holm) |
| 005_bimodal_convergence.py | superseded by 006 (kept for reference) |
| 006_colocation_figure.py | bimodal neurons inside vs outside the overlap: hierarchical bootstrap (sessions, then neurons) + Fisher |
| 007_cortical_flatmaps.py | Allen butterfly flatmap of the isocortex, with and without zone contours |
| 008_prelick_convergence_colocation.py | pre-lick converging neurons (SL reference, ssl-prelick-convergence) vs the overlap; outputs in `_roc_prelick_sl/projection_colocation/` |

`report/`: `build_report.py` (generates the Quarto report on haas), `render.sh` (renders the PDF locally),
`build_deck_assets.py` (panel crops + summary schematic), `deck/build_deck.py` + `deck/render_deck.sh` (PowerPoint).
Run order after new ROC tables or zones: 002 (both `ZONE_PCT`) -> 003 -> 004 -> 006 -> 007 -> 008 -> report / deck.

## Decisions (user, 2026-10-04)

- 90 % projection zones are the main result; 70 % is reported as the stricter version (report, deck, schematic).
- Target-slab figures are dropped from report and deck; the within-target offset test (004) is kept.
- No cohort / stage splits: pooled is the scope of this project.
- Deck and report numbers come from the result tables (`build_report.py` -> `report_numbers.json` -> `deck/build_deck.py`).
- Pre-lick converging neurons x projection overlap (008): null; added to the pre-lick convergence article as a
  supplementary figure after the overnight rerun.

## TODO

- [ ] Overnight: ROC rerun of the 3 context-fixed sessions (MH062, MH064, AB128) -> maps chain (045, 001, 003, 004, 006,
      007 at 70 % and 90 %) -> 008 at 70 % and 90 %. Compare with `combined_results_ks4/_snapshots/2026-10-04_2200/`.
- [ ] 2026-10-05, after all runs (incl. passive): render deck (`deck/render_deck.sh`) and PDF (`render.sh`).
- [ ] 007 cosmetics: crowded tick labels on the recorded-neuron density colour bar; legend fragment in the whisker /
      auditory flatmap crop.
