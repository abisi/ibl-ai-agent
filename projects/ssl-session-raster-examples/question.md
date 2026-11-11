# Question

Illustration only (no statistics): which sessions give the best-looking whole-brain raster after a whisker trial?

Definitions
- Valid probe: an imec probe with >= 1 unit of the "all" unit set (quality_label good/mua) in
  `137_stable_units.parquet`. Sessions with >= 4 such probes are included (74 sessions).
- Epoch: 1 s before to 3 s after the stimulus of one active whisker hit trial (perf != 6), the one with the
  largest population rate increase (sum over units, 0-300 ms after vs 500 ms before the stimulus).
- Default version: plot_session_raster output with all spikes black. Area version: spikes coloured by area_group
  (ephys_utilities `get_custom_area_groups_colors`).
- DLC traces from ephys_utilities `dlc_helpers` (clean_series defaults, pupil QC, nose_norm).
