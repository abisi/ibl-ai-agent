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

## TODO (2026-10-04)

- [ ] Overnight: ROC rerun of the 3 context-fixed sessions (MH062, MH064, AB128) -> maps chain (045, 001, 003, 004, 006,
      007 at 70 % and 90 %) -> 008 at 70 % and 90 %. Then compare with the frozen statistics
      (`combined_results_ks4/_snapshots/2026-10-04_2200/`).
- [ ] Decide the main contour level for the co-location result: with the line-balanced zones the bimodal enrichment
      depends on it (70 % vs 90 %); report both, choose one as main.
- [ ] Re-check the overlap composition and sub-region names with the new zones (SCm share dropped) and the target-slab list
      (`003.TARGETS`, chosen from the old zones).
- [ ] Deck: numbers in `deck/build_deck.py` are hard-coded (injection counts, overlap volume, co-location, latency table)
      -> read them from the result tables, then re-render the deck; re-render the PDF report (`render.sh`) when wanted.
- [ ] Modality-preference slide: check the regional statement against the final maps.
- [ ] 007 cosmetics: crowded tick labels on the recorded-neuron density colour bar; legend fragment in the whisker /
      auditory flatmap crop.
- [ ] 008 (pre-lick converging neurons): null so far; possible controls before closing: area-matched comparison
      (inside vs outside within the same structures), lick-responsive baseline, learners population.
- [ ] Splits: cohort (R+ / R-) and stage (learning / expert) for the maps and the co-location test (all pooled now).
- [ ] Decide git home of the results figures / report (data are unpublished; this fork is public).
