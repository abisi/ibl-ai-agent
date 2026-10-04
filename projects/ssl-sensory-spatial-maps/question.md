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
| Projection zones | Allen Mouse Connectivity anterograde tracing, wild type + Emx1-IRES-Cre injections (SSp-bfd 8, SSs 3, AUDp 6, AUDd+AUDv 3), projection density at 50 um, mirrored to the right hemisphere, normalised per experiment, averaged; candidate voxels exclude fibre tracts, ventricles and the source area; zone = top 10 % of candidate voxels |
| Slabs | 500 um thick; coronal tiling of the recorded AP range, sagittal tiling of the ML range, 13 target slabs |
| Panels | schematic (Allen colours; slab as a band) / all neurons / neurons coloured by the quantity + projection-zone contours / density (mean in a 550-um window, 50-um grid, sigma 50 um, >= 5 neurons) / significant neurons |
| Contour test | per target slab x area with >= 15 significant neurons of each preference: 80 % highest-density contours (Gaussian KDE), centroid distance and axis shifts, Dice overlap; within-session label permutation (5000); Holm across tests also reported |
