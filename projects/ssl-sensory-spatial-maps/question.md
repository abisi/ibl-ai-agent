# SSL sensory spatial maps

## Request (Axel Bisi, 2026-10-04)
Publication-ready spatial maps of stimulus responsiveness, modality selectivity and latency in 500-um slabs, with the
projection zones of whisker (SSp-bfd, SSs) and auditory cortex, in the style of the slab figures of Chen et al. 2024
(Cell, "Brain-wide neural activity underlying memory-guided movement"). Slabs tile the whole AP axis (coronal) and the
ML axis (sagittal); plus slabs centred on projection zones / areas (SCm, frontal / caudal / tail of the striatum, wM2,
wM1, SSp-bfd / SSs, ORB, auditory targets). For modality preference, within given areas of given slabs, test whether
whisker-preferring and auditory-preferring neurons differ in location (80 % contours).

## Method (confirmed with the user, 2026-10-04)
| Item | Decision |
|---|---|
| Units | good + mua, all sessions pooled (both cohorts, both stages); CCF positions, ML folded onto the right hemisphere |
| Quantities | rate-based ROC (045 roc_long): whisker_active, auditory_active (responsiveness; + excited), wh_vs_aud_active (modality; + auditory-preferring); latency = half-time to peak of responsive units (001) |
| Latency | 1-ms PSTH (-100..+200 ms), artefact-corrected spikes, minus the local pre-stimulus mean (-100..-10 ms), Gaussian sigma 2 ms, signed by the ROC selectivity; peak in 5..100 ms; latency = last upward crossing of half the peak before the peak (whisker: searched after +5 ms) |
| Projection zones | Allen Mouse Connectivity anterograde tracing, wild type + Emx1-IRES-Cre injections (SSp-bfd 8, SSs 3, AUDp 6, AUDd+AUDv 3), projection density at 50 um, mirrored to the right hemisphere, normalised per experiment, averaged; candidate voxels exclude fibre tracts, ventricles and the source area; density smoothed (sigma 50 um); zone = 70 % contour (highest-density voxels holding 70 % of the projection; user 2026-10-04, replaces the first top-10 % zones); figures: 500-um slabs, coronal and sagittal, contour lines lightly smoothed, area composition per zone, whisker / auditory overlap row; colour maps white -> whisker #f7b519 / auditory #2c2cdb -> dark grey |
| Slabs | 500 um thick; coronal tiling of the recorded AP range, sagittal tiling of the ML range, 13 target slabs |
| Panels | schematic (Allen colours; slab as a band) / all neurons / neurons coloured by the quantity + projection-zone contours / density (mean in a 550-um window, 50-um grid, sigma 50 um, >= 5 neurons) / significant neurons |
| Contour test | per target slab x area with >= 15 significant neurons of each preference: 80 % highest-density contours (Gaussian KDE), centroid distance and axis shifts, Dice overlap; within-session label permutation (5000); Holm across tests also reported |

## Bimodal neurons and co-location with converging projections (user, 2026-10-04)
| Item | Decision |
|---|---|
| Responsive to a modality | any of the stimulus-vs-baseline ROC tests active / passive pre / passive post significant after Bonferroni over the number of those tests in the session (1-3); excited or inhibited; the active miss vs correct-rejection test is NOT used |
| Bimodal | responsive to both whisker and auditory stimuli; quantity = fraction of sensory-responsive neurons that are bimodal |
| Map | 003 quantity `bimodal` (categories: whisker only / auditory only / bimodal / not responsive; density = bimodal fraction) |
| Regions tested | the overlap of the merged whisker and auditory 70 % zones, cut by Allen structure into connected 3-D sub-regions (>= 0.04 mm^3) named by position (e.g. CP tail, CP caudal-lateral) -- not whole areas |
| Tests | global, no session pairing (sampling differs strongly between areas): pooled proportion inside vs outside; Fisher's exact (= unit-label permutation); spatial-shift null (region translated by random 3-D offsets 0.75-2 mm over the recorded tissue; main test); hierarchical bootstrap CI (sessions, then neurons); Holm across sub-regions; control: sub-region vs rest of the same structure |
| Interpretation | co-location only (tracing from other mice, axons of passage, CCF uncertainty) |
| Figure | 006 colocation_figure + colocation_figure_caption.md (005 session-based version superseded) |
