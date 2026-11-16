# ssl-dlc-utils-validation

Does the single DLC module `ephys_utilities/helpers/dlc_helpers.py` load and process the NWB_ks4 DLC data
correctly, and are the DLC data in the NWB files usable, before any dependent analysis is switched to the module?

Questions, by section (scripts in `exploratory-analyses/<section>/`, see README.md):
1. Units, reference point, scale: are the series in plausible physical units and per second (000)? What happens when
   the converter puts the jaw / nose reference point at the image origin, and is recomputing at loading needed (001)?
   Which rig scale (px/mm, setup 1 vs 2) applies to each session (004)?
2. Likelihood and cleaning: what each cleaning step does (002); is the jaw well placed at low likelihood (005); from
   which likelihood are pupil, whisker and nose points well placed (007)? Which defaults go into `CLEAN_DEFAULTS` /
   `LIKELIHOOD_CUTOFF`?
3. DLC lick times: video clock to session time, per-frame timestamps vs a single offset, against piezo licks (003).
4. Top view and video audit: top-camera orientation (008), what is missing or wrong per session and camera (018),
   how bad are the failed top-view sessions (019).
5. Pupil: tracking at pupil-area spikes (013), QC-based cleaning (014), outlier mice (015).
6. Overviews: event-triggered traces over all sessions (010), day-0 posture and movement per trial type and cohort
   (011).

Not in scope: changing NWB_converter (user: correct as is), updating dependent analyses (after validation).
Results: combined_results_ks4/ssl-dlc-utils-validation/{figures,tables,cache,report}.
