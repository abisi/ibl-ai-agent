# Question

## Original request (from user, 2026-08-14)

"Another project, more data visualization and diagnostics for whisker
trials." Narrowed via follow-up: **stimulus/artifact QC** specifically --
whisker stimulus amplitude/duration/timing jitter across trials and
sessions, and characterizing the artifact dead-zone itself (how the
-10ms/+5ms window in `ssl-analyze/references/ssl_artifact_dead_zone.md` was
chosen, and whether it holds across probes/sessions). New project, reusing
the existing `ssl_ephys` build and this session's derived unit tables
(`../ssl-whisker-auditory-cohort-modulation/`) rather than recomputing from
raw NWB where possible.

## Why this matters

`ssl_artifact_dead_zone.md` states the -10ms/+5ms window is a project
convention anchored to trials.parquet's `start_time`, and explicitly flags
that it is an *assumption*: "If a future analysis has reason to believe the
actual magnetic-stimulus delivery time differs from `start_time` for some
sessions, flag that explicitly rather than assuming." This project is that
check -- empirically, not by assumption.

## Planned analyses

1. **Stimulus parameter QC**: distributions of `whisker_stim_amplitude`,
   `whisker_stim_strength`, `whisker_stim_duration` (global, per session,
   per mouse); relationship between `start_time`, `whisker_stim_time`, and
   `stim_onset` (are they identical, constant-offset, or jittered?).
2. **Empirical artifact characterization**: fine-resolution (100us, the
   dataset's native spike-time quantization) population spike-density
   histograms around whisker `start_time`, spanning well outside the
   documented dead zone, contrasted against auditory and no-stim trials at
   the same resolution (auditory/no-stim should show no artifact signature
   -- the rule is whisker-specific by construction, so this is also a
   direct test of that assumption).
3. **Stability check**: does the artifact's timing/width hold constant
   across sessions and probes, or does it vary in a way that would break a
   single fixed -10ms/+5ms rule for some sessions?
4. **Example raster diagnostic**: a handful of single-session,
   many-unit rasters zoomed to the stimulus period, to visualize the
   artifact directly (a synchronous cross-unit spike band is the expected
   electrical/mechanical-artifact signature).
5. Compare findings against the documented rule and note agreement or
   discrepancy explicitly -- this project makes no changes to
   `ssl_artifact_dead_zone.md` itself, only reports what the data show.

## Scope notes

- Unit set: reuses `analysis_units.parquet` from the prior project (good +
  mua, coverage-ratio-filtered, 81 sessions) for consistency and to avoid
  recomputing from raw NWB; documented as a choice, not the only valid one
  -- an artifact affecting excluded/low-coverage units would not be caught
  here.
- No statistical modeling in this project; it is a visualization/diagnostic
  companion to the artifact-dead-zone rule already in use, not a
  re-derivation of it.
