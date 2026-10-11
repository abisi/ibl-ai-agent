---
name: ssl-dlc-units
description: Use this skill whenever an SSL analysis loads, analyses or plots DeepLabCut (DLC) series from the NWB behaviour module (jaw, tongue, nose, pupil, whisker, spout, particle; `processing/behavior/BehavioralTimeSeries`). Gives the pixel-to-mm conversion per behaviour setup and camera, how to find a session's setup, the timestamp checks, and which series need no conversion.
---

# SSL DLC Units

## Use this skill when
- Any DLC-derived distance, position, speed or area is computed, compared across sessions, or plotted.
- Comparing DLC measures between mice or cohorts: sessions come from two behaviour setups with different camera magnifications.

## Rule
Report DLC distances in mm (areas in mm², speeds in mm/s), never in pixels. Convert per session with `scripts/ssl_dlc_calibration.py`:

```python
import ssl_dlc_calibration as cal                       # scripts/ on sys.path
jaw_mm = cal.to_mm(jaw_px, session_id, "jaw_distance")                 # px -> mm
pupil_mm2 = cal.to_mm(pupil_px2, session_id, "pupil_area", power=2)    # px^2 -> mm^2
cal.setup_of(session_id), cal.px_per_mm(session_id, "whisker_tip_x")   # setup 1 / 2, px per mm
```

## Why
- Calibration (grid images) differs between setups: setup 1 side 33.4 / top 30.4 px/mm; setup 2 side 28.8 / top 16.75 px/mm (`<Axel_Bisi>/mice_info/reference_setup{1,2}.xlsx`). Pooling pixels mixes magnifications (top camera nearly 2x).
- The NWB files carry no setup field. The setup is in `<share_internal>/Axel_Bisi_Share/dataset_info/joint_probe_insertion_info.xlsx` (`setup` per mouse_name + date; `M:\share_internal` locally, `/mnt/share_internal` on haas). It is per session, not per mouse: MH001-MH010 and some sessions of MH025 and MH027 are on setup 2.
- That sheet mixes Excel dates and `dd.mm.yyyy` text; text must be parsed day first (a month-first parse loses 18 AB sessions). The helper does this; checked 2026-10-11 on the 107 ssl-prelick-convergence sessions (all exact date matches).

## Camera and units of each series
- Top camera: `whisker_*`, `top_*`. Side camera: everything else (jaw, tongue, nose, pupil, spout, particle). Inferred from the series names, not checked against the videos.
- `*_angle` (degrees) and `*_likelihood`: no conversion.
- `*_velocity` as stored is per frame and scales with the frame rate: recompute `np.gradient(value, timestamps)` from the series and its own timestamps, then convert (the helper raises on `*_velocity`).
- Relative measures (pupil area as % of baseline, P(tongue detected)) need no conversion; say so in the axis label.
- Absolute `jaw_distance` differs between rigs by ~30x (different reference point); use baseline-subtracted changes.

## Timestamp checks (every series has its own timestamps)
- Equal lengths: use. Empty timestamps (some MH jaw series): use the whisker-camera timestamps only if the lengths match exactly.
- Length difference <= 0.5 %: truncate both to the shorter. Larger (e.g. data half the timestamps: 100 Hz camera, 200 Hz timestamps): exclude the series and list it.
- Frame rates differ (most 200 Hz, some 100 Hz): resample to a common grid before averaging. Reference implementation: `projects/ssl-prelick-convergence/exploratory-analyses/113_dlc_passive_pre.py`.
- DLC likelihoods are low in these models (jaw median ~0.75-0.8, whisker tip ~0.03 in MH sessions): a 0.9 cutoff masks most frames; 113 uses 0.5 on whisker base, jaw, nose tip and pupil.

## Quality gate
- Axis labels give real units (mm, mm², mm/s, °, °/s) or state that the measure is relative.
- Outputs record the setup and px/mm used per session, and the sessions without a setup (no conversion, excluded from mm results).
