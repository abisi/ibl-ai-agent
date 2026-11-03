# NWB_ks4 per-session exceptions (conversion review, started 2026-11-02)

Canonical list of sessions whose NWB_ks4 file deviates from the standard conversion, why, and where the deviation is
applied. Read before trusting a session-specific oddity (missing probe, missing epoch, video/DLC length, behaviour-only
file for a recorded day). Update in place when an exception is added or resolved.

Where exceptions live (user decision 2026-11-02): as commented clauses directly in the NWB_converter scripts, next to the
code they change. Raw files (session_config.json, results.csv, videos, probe sheet, NIDQ / log binaries) are never edited.

| clause | file | sessions |
|---|---|---|
| `SESSION_JSON_OVERRIDES` (session_config.json values) | `make_yaml_config.py` | MH068_20260125, MH069_20260125 (`ephys_session` 1); AB079_20230620, MH067_20260118 (`ephys_session` 0, behaviour-only whisker_0; MH067 also `camera_flag` 0); AB100_20240304 / 0305 / 0306_100728, AB111_20240421 / 0422 (`camera_flag` 0, videos not on the NAS) |
| `skip_probes` | `converters/ephys_to_nwb_test.py` | MH064_20260117 imec2 (no probe sheet row; user: ignore); MH069_20260125 imec0 (not aligned yet; remove once aligned) |
| `CAMERA_LAST_PULSE_RUN_SESSIONS` | `utils/ephys_converter_misc.py` | AB086_20231015, MH027_20250424, MH030_20250503, MH064_20260115, MH064_20260116, MH067_20260120, MH067_20260121 |
| `camera_from_log_sessions` (camera timestamps from the behaviour log, mapped to imec through the trial TTLs) | `utils/ephys_converter_misc.py` (`extract_ephys_timestamps`) | MH038_20250524 (NIDQ camera channels only 56,558 pulses; log 711,761 = DLC frames) |
| `session_config_corrected.json` (raw data folder, preferred by the converter) | `/data/<mouse>/Training/<session>/` | MH037_20250524 / 25, MH038_20250524 / 25, MH039_20250525: `camera_flag` 0 -> 1 (videos and DLC valid: frames = camera pulses +-2); NWBs rebuilt with DLC 2026-11-02 |
| `end_aligned_dlc_sessions` (last DLC frame <-> last camera timestamp; the camera pulsed before the first saved frame) | `converters/DLC_to_nwb.py` | MH013_20250218 (side 882, top 868 leading timestamps dropped), MH036_20250517 (side 510, top 515); rebuilt 2026-11-03, verified with 006 (NWB pairing = best offset - 3..8 frames in every third). AB153_20250209 top (47), AB127_20240823 top (video 2542 frames longer than the pulses: first 2542 frames dropped), AB153_20250209 side (47); rebuilt 2026-11-03 (after: AB127 top offset 6/7/6, AB153 side 5/6/4). AB153 top added 2026-11-03 (user) and rebuilt; after: the same metrics peak at 13-18, as in MH009. Evidence: 007 top metrics vs licks (37 licks) peak 43-49 frames later than the same metrics in control MH009 (whisker_angle, whisker_base_y, whisker_tip_y, top_particle_y, top_nose_base_y / tip_y), i.e. end-aligned (+47); whisker stimuli (38) inconclusive. Backups: <session>_backup_20261103.nwb (AB153 also _1126, side-only intermediate) |
| `TOP_VIDEO_NAMED_SIDE_SESSIONS` | `utils/ephys_converter_misc.py` | MH020_20250306, MH021_20250309 / 0311, MH022_20250309 / 0311, MH023_20250316 / 0318, MH025_20250421, MH026_20250425, MH027_20250422, MH028_20250501, MH030_20250501 |

## Camera / DLC rules (converter, 2026-11-02)
- Camera keys: for AB and MH sessions `cam1` = top camera (NIDQ `cam0_frame_times`, NIDQ channel 5), `cam2` = side camera
  (`cam1_frame_times`, channel 6), on both rigs (checked with the video file start times against the first / stray
  pulses). `behavior_to_nwb.py` (video links) already used this; `DLC_to_nwb.py` used the other experimenters'
  convention (`cam1` = side) and was fixed to take each view's own camera. Before the fix the swap only mattered where
  the two cameras' pulse trains differ (8 of 236 NWBs with both views); those were rebuilt. Verified on all 119 ephys NWBs
  with DLC: top-view timestamps = top-camera pulses, side-view = side-camera pulses (< 1 ms).
- Last camera pulse run: in the 7 sessions above the camera ran a test run, then one stray pulse when the video was
  armed, then the saved run; the video and DLC cover only the last run (pulses = video frames; video file name time =
  stray pulse). Evidence and figures: `projects/ssl-dlc-alignment-debug` (001 start / end of the saved run, 002 scan of
  all NWBs, 003 lick-locked tongue latency vs controls).
- Frames vs camera timestamps: a difference of less than 1 s of frames (200 frames at 200 Hz) is accepted (frames paired
  from the first timestamp; e.g. AB086 top camera, 7 frames); larger differences are printed as a warning by
  `DLC_to_nwb.py` and need a camera-run check.
- `tongue_dlc_licks` / `jaw_dlc_licks` are stored in video time (frame index / frame rate), not session time; the offset
  is the time of the first camera timestamp of the session (e.g. 275.2 s in MH067_20260120). No downstream code found
  (NWB_converter, Axel_Bisi/Github, ibl-ai-agent) that adds it: add `timestamps[0]` of the side-view series before use.

## DLC frame -> camera pulse checks (2026-11-02)
Per-session verdicts: `combined_results_ks4/ssl-dlc-alignment-debug/tables/session_verdicts.csv`; offset curves
(DLC frame -> pulse index over the session, side camera vs piezo licks, top camera vs licks + whisker stimuli):
`figures/<session>_offset_curve.png` (005). In register sessions show a constant offset of +3..+6 frames (tongue
before the piezo lick). Groups: in register (frame / pulse count differences at the end only); constant start offset
(missing frames at the video start: MH013_20250218 +882, MH036_20250517 +510, AB153_20250209 +47, AB127_20240823 top
-2542); progressive drift (frames dropped during the video: AB116_20240726, AB092_20231205 side, MH004_20241206,
MH013_20250219, MH017_20250302, AB085_20231005 small, AB095_20231212 top); DLC incomplete (MH020_20250304 top,
AB152_20250129 side, AB154_20250207 top, MH007_20250203 side) or mislabelled (MH028_20250503; user: being fixed).
Direct evidence per session (006, `figures/<session>_pairing_evidence.png`, `tables/pairing_evidence.csv`): lick-bout
heatmaps of tongue likelihood under the NWB pairing vs end-alignment, and the correlation vs assumed offset D per session
third. Constant offsets confirmed on the side camera (peak D per third): MH013_20250218 885/890/884 (end-align 882; NWB
tongue stripe at -4.4 s in every bout), MH036_20250517 516/516/515 (510), AB153_20250209 52/53/51 (47; 17 bouts only),
AB127_20240823 top -2557 in thirds 2-3 (-2542). Control MH009_20250206 (69 extra pulses) peaks at 5 in every third:
start-aligned, so the method separates start from end alignment. Top camera: whisker_angle around lick-bout onsets (it dips at
each lick; largest |deviation| taken): MH013 883/881/850 (868), MH036 523/498/522 (515), AB127 -2536 x3 (-2542), MH009
4/21/23 (start-aligned, not 68); AB153 undetermined (17 bouts). MH038_20250524: the 56,558 NIDQ side pulses match the log-derived
timestamps to -0.005 ms median (range +-0.23 ms, the 5 kHz log resolution).
Incomplete DLC (checked with ffmpeg 2026-11-02): MH020_20250304 top, AB152_20250129 side, AB154_20250207 top: the raw
video decodes past the last DLC row (DLC run stopped; rerun DLC). MH007_20250203 side: DLC was run on `MH007_side 25-02-03 14-03-30.avi`, which is all zero bytes after ~1.79 GB
(frame ~5,810); the folder has a second, complete side video `MH007_side 25-02-03 14-03-31.avi` (975,858 frames = top
video, decodes to the end): rerun side DLC on the 14-03-31 file. The NWB links all three videos.

## Unresolved DLC sessions
| session | issue |
|---|---|
| AB085_20231005_152636 | video 31,287 frames longer than the single pulse run (start pairing gives sharp traces; extra frames probably at the end) |
| MH064_20260114_102210 | video 72,191 frames longer than pulses; pairing breaks in the last third |
| MH065_20260114_154021 | video (671,385 frames) matches neither pulse run (891,327 + 431,279) |
| MH020_20250304_161732 (top), AB155_20250208_175103, MH006_20250123_131233 | DLC / video length != pulses, no run matches |
| single-run sessions with tens to thousands of missing frames (AB092, AB095, MH013 0218, MH036 0517, MH009 0206, AB151, ...) | location of the missing frames unknown; table `combined_results_ks4/ssl-dlc-alignment-debug/tables/camera_run_scan.csv` |
| MH028_20250503_112316 | both DLC files named `side`; the top-camera video was probably tracked with the side model |

## Behavioural epochs
- `get_passive_active_epoch_bounds` (`utils/behavior_converter_misc.py`) fixed 2026-11-02: passive blocks are runs of
  >= 10 passive trials; passive_pre = first block if <= 5 trials precede it, passive_post = last block if <= 5 trials
  follow it (extended backward over second-half blocks separated by <= 5 trials). Changed exactly AB122_20240804,
  MH019_20250306, MH037_20250527 (rebuilt); all 132 NWBs with epochs match the function. Epochs are written only for
  ephys sessions (`behavior_to_nwb.py`); behaviour-only NWBs have none. One-page overview of all 132:
  `combined_results_ks4/ssl-nwb-ks4-review/figures/epochs_all_nwb.pdf`.
- MH028_20250501_104058: context labels wrong in the raw results: trials 86-103 (526-1001 s; auditory hits, false
  alarms, licks, irregular ITIs) are labelled passive, so passive_pre runs to ~1007 s (104 trials) although the passive
  block ends at trial 79 (294 s); trials 80-85 (492-519 s, perf 6) unclear. Pending user decision.

## Deleted raw sessions
- MH002_20241127_115446 (8 trials) and MH002_20241128_101855 (14 trials): aborted auditory sessions, restarted minutes
  later; raw folders deleted 2026-11-02 (user; via haas, the Windows share is read-only). MH002 pretraining is now
  free_licking_-5 ... auditory_-1 (yamls regenerated, NWBs converted). Their yaml folders in
  `Axel_Bisi/data/MH002/` and `Myriam_Hamon/data/MH002/` were deleted too.

## Not converted / excluded
| session | reason |
|---|---|
| AB091, AB096 | not part of the SSL dataset (day labels also wrong) |
| AB155_20250208_185559 | second part of a split session ("sessions to concat"); concatenation decision pending |
| AB068_20230402_141857 imec1 | valid but not aligned; no NWB |
| AB077_20230531_143839 | NWB has no units table (user: leave) |

## Known converter facts checked in the review
- Piezo licks in every NWB come from the converter's own detection (`detect_piezo_lick_times` on `log_continuous.bin`,
  5 kHz), never from CatGT `sync_event_times/piezo_licks.txt` (unreliable: analog piezo, threshold-edge extraction; 0
  events in MH065_20260116). Ephys sessions: log licks mapped to the imec clock by trial-by-trial interpolation through
  the trial TTLs; the same detector on the NIDQ piezo channel (25 kHz), mapped through the NIDQ trial TTLs (TPrime
  equivalent), agrees to 0.02-0.04 ms median (IQR +-0.1 ms, no drift; AB105 0314, MH065 0116, MH067 0121).
- Raw `BehavioralTimeSeries/lick_trace` timestamps in ephys sessions assume exactly 25,000 Hz and ignore the NIDQ -> imec
  mapping: up to ~110 ms off at the session end. Use `piezo_lick_times`, not the raw trace, for timing.
- `make_yaml_config.py` reads `extracell_ephys_session` with a fallback to `ephys_session` (older session_config.json).
