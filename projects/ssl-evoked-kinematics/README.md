# ssl-evoked-kinematics

Stimulus-evoked orofacial movements (jaw, tongue, nose, whisker) in R+ and R- mice, whisker vs auditory trials, day 0
and expert sessions; tongue-reach and whisker-sweep trajectories in a head-centred frame (mm); day-0 whisker movement
after the stimulus; day-0 pre-stimulus whisking state before whisker hits vs misses. Question and methods:
`question.md`. Results and report are not in this repository (see Outputs).

## Scripts (`exploratory-analyses/`, run on haas, in this order)

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `000_evoked_kinematics.py` | Per-session cache of evoked time courses, first tongue reaches, whisker skeletons and per-trial metrics (onset latencies, reach latency / duration / extension / speed, whisker sweep amplitude); first-version figures; per-mouse metric and test tables per stage | NWB_ks4, reference sheet | `cache/000v1_*`, `tables/000_*`, `figures/000_*` |
| `001_trajectory_figures.py` | Publication-style figures: cohort-overlaid time courses, spout-centred tongue reaches, whisker skeleton sequence, example mice | 000 cache | `figures/001_*` |
| `002_day0_whisker_sensing.py` | Day 0: whisker movement 30-100 ms after the stimulus by trial type and cohort; whisker angle aligned to reach onset | 000 cache | `figures/002_*`, `tables/002_*` |
| `003_day0_prestim_state.py` | Day 0: single-trial pre-stimulus signals (whisker speed / angle, nose, jaw, tongue, pupil) for all active trials; lick vs no-lick windows; per-session logistic models | NWB_ks4 | `cache/003v1_*`, `tables/003_*`, `figures/003_*` (figures superseded by 004 / 005) |
| `004_day0_prestim_time_controls.py` | Day 0: time controls of the pre-stimulus hit-miss difference (stratified, detrended, matched, local shuffle) and its specificity across trial classes | 003 cache | `figures/004_*`, `tables/004_*` |
| `005_day0_prestim_whisking_figure.py` | Day 0: summary figures of pre-stimulus whisking, whisker posture and jaw posture (raw and detrended, SEM and SD versions) | 003 cache | `figures/005_*`, `tables/005_*` |
| `006_metrics_summary_figure.py` | Per-mouse metric summary, day 0 and expert side by side | `tables/000_*_metrics_mouse.csv` | `figures/006_*` |
| `report/build_report.py` | Report generator (skills/project-report); `report/render.sh` builds PDF / HTML locally | result tables and figures | `report/` |

Dependencies: 001, 002 and 006 import 000; 004 and 005 import 003 (which imports 000 and 001 for shared rules).
`000 --plot-only` and `003 --plot-only` redraw from the caches.

Python on haas: `~/code/ibl-ai-agent/.venv/bin/python` (with `nice -n 10`). DLC is read with
`ephys_utilities.helpers.dlc_helpers` from the NAS checkout.

## Outputs

`combined_results_ks4/ssl-evoked-kinematics/` on the NAS (`M:\analysis\Axel_Bisi\...` on Windows,
`/mnt/lsens-analysis/Axel_Bisi/...` on haas): `cache/`, `tables/`, `figures/`, `report/` (report.pdf, report.md,
report.html).

## Notes

- Whisker angle is recomputed from the whisker keypoints in a per-session head frame (negative = protraction) and only
  baseline-subtracted (or session-median-subtracted) values are compared; the NWB `whisker_angle` series is not used.
- Jaw opening is the NWB `jaw_distance` (mm); the jaw angle is not used.
- Failed top-view tracking: MH025_20250421 and MH027_20250422 are dropped explicitly; MH009_20250207 / 20250208 and
  MH025_20250422 yield no whisker data through the validity rules (see the 000 docstring).
- Kinematics come from the DLC models stored in the NWB files; rerun 000 -> 006 if the DLC is re-inferred.
