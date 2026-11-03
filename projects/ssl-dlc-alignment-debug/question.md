# ssl-dlc-alignment-debug

## Question (user, 2026-11-02)
In some NWB_ks4 sessions the DLC series (video frames) and their camera timestamps (camera pulses on the ephys clock)
have different lengths. The user's hypothesis: the missing data is at the end of the session, i.e. DLC frame i belongs to
camera pulse i, counted from the start. If that is true, event-aligned DLC traces (PSTHs) per trial type should be sharp
and time-locked.

## Sessions
Mismatch sessions (DLC frames vs timestamps, from the 2026-11-02 NWB audit):

| session | side DLC / ts | top DLC / ts |
|---|---|---|
| MH065_20260114_154021 | 671,384 / 1,322,606 (-49.2 %) | same |
| MH020_20250304_161732 | equal | 192,598 / 1,359,735 (-85.8 %) |
| MH067_20260120_100233 | 321,195 / 360,043 (-10.8 %) | similar |
| MH067_20260121_101647 | 569,715 / 594,629 (-4.2 %) | similar |
| MH064_20260114_102210 | 2,132,790 / 2,060,604 (+3.5 %) | same |
| AB085_20231005_152636 | 1,636,739 / 1,605,452 (+1.9 %) | similar |

Controls (equal or near-equal lengths): MH065_20260116_174248 (+1 timestamp), MH064_20260116_133002 (-0.2 %).

## Definitions
- **Start alignment** (user hypothesis): frame i <-> timestamp i; the longer of the two is truncated at its end.
- **End alignment** (alternative): last frame <-> last timestamp; the longer is truncated at its start.
- **Half-rate alignment** (only when frames ~ half the timestamps): frame i <-> timestamp 2i (camera at half the pulse
  rate, cf. `skills/ssl-dlc-units/SKILL.md`).
- A hypothesis is supported if event-aligned traces are sharp (large, time-locked change after the event, as in the
  controls) and stay sharp across the session (first vs last third of trials). If frames were lost in the middle,
  start alignment is sharp early and degrades later.
- Events: trial start (stimulus onset) per trial type, and piezo lick times.
- Trial types: active whisker hit / miss, auditory hit / miss, false alarm, correct rejection; passive whisker /
  auditory. Active perf == 6 trials excluded (`skills/ssl-trial-exclusion`, rule 1); context labels as stored. No
  warm-up or disengagement trimming (not needed for an alignment check, more trials help).
- DLC series: every `*_angle`, `*_distance`, `*_velocity`, `pupil_area`, `tongue_likelihood`. Each z-scored over the
  session (relative units, no px -> mm conversion needed).
