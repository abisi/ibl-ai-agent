# change-log

- 2026-11-03: project created (validation of ephys_utilities helpers/dlc_helpers.py).
- 2026-11-03: 000-003 run; reference-at-origin scan of NWB_ks4 (tables/dlc_reference_at_origin.csv).
- 2026-11-03 to 2026-11-09: 004-019 run (rig scale, likelihood frames, default cleaning, top orientation, event
  overviews, day-0 posture / movement, pupil smoothing / QC / review, video-DLC audit, failed top-view frames).
- 2026-11-10: restructured into sections (scripts moved, not renumbered; result file names unchanged):
  `exploratory-analyses/1_units_reference_scale/` (000, 001, 004), `2_likelihood_and_cleaning/` (002, 005, 007),
  `3_lick_times/` (003), `4_top_view_and_video_audit/` (008, 018, 019), `5_pupil/` (013, 014, 015),
  `6_event_overviews/` (010, 011); `common.py` stays in `exploratory-analyses/` and every script appends that folder to
  `sys.path` (014 / 015 also add `6_event_overviews/` for 011). Archived (superseded defaults): `archive/006_default_cleaning.py`,
  `archive/012_pupil_smoothing.py`. README.md added; question.md and TODO.md updated. Haas copy mirrored (old flat
  copies left in place); all scripts parse and import common on haas.
- 2026-11-10: report built (report/build_report.py + render.sh; combined_results_ks4/ssl-dlc-utils-validation/report/),
  organised by the six sections, including the ssl-dlc-kuma 025 / 030 checks (NWB rebuild from the DLC h5, likelihood
  gating, whisker-angle direction, jaw-angle scale).
- 2026-11-10: 004 fix: the setup-2 marker matched any '2' in the probe-sheet rig (e.g. 'setup 1 AI3209') and circled
  nearly every ephys session; plotting moved to `plot()`, marker now 'setup 2' only, plus top view nose up from 008;
  `--plot-only` redraws from the existing table (table not rerun). Figure regenerated on haas; previous version kept as
  figures/004_rig_scale_markerbug_20261110.png.
