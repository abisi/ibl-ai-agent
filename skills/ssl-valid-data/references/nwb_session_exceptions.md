# NWB_ks4 per-session exceptions (conversion review, started 2026-11-02)

Canonical list of sessions whose NWB_ks4 file deviates from the standard conversion, why, and where the deviation is
applied. Read before trusting a session-specific oddity (missing probe, missing epoch, video/DLC length, behaviour-only
file for a recorded day). Update in place when an exception is added or resolved.

Where exceptions live (user decision 2026-11-02): as commented clauses directly in the NWB_converter scripts, next to the
code they change. Raw files (session_config.json, results.csv, videos, probe sheet, NIDQ / log binaries) are never edited.

| clause | file | sessions |
|---|---|---|
| `SESSION_JSON_OVERRIDES` (session_config.json values) | `make_yaml_config.py` | MH068_20260125, MH069_20260125 (`ephys_session` 1); AB079_20230620, MH067_20260118 (`ephys_session` 0, behaviour-only whisker_0; MH067 also `camera_flag` 0); AB100_20240304 / 0305 / 0306_100728, AB111_20240421 / 0422 (`camera_flag` 0, videos not on the NAS) |
| `skip_probes` | `converters/ephys_to_nwb_test.py` | MH064_20260117 imec2 (no probe sheet row; user: ignore). MH069_20260125 imec0: aligned 2026-11-03 (channel_locations.json / prev_alignments.json moved to Myriam_Hamon/.../MH069_g0_imec0/ibl_format), clause removed, reconverted: 25 units (imec3 invalid recording) |
| `CAMERA_LAST_PULSE_RUN_SESSIONS` | `utils/ephys_converter_misc.py` | AB086_20231015, MH027_20250424, MH030_20250503, MH064_20260115, MH064_20260116, MH067_20260120, MH067_20260121 |
| `camera_from_log_sessions` (camera timestamps from the behaviour log, mapped to imec through the trial TTLs) | `utils/ephys_converter_misc.py` (`extract_ephys_timestamps`) | MH038_20250524 (NIDQ camera channels only 56,558 pulses; log 711,761 = DLC frames) |
| `session_config_corrected.json` (raw data folder, preferred by the converter) | `/data/<mouse>/Training/<session>/` | MH037_20250524 / 25, MH038_20250524 / 25, MH039_20250525: `camera_flag` 0 -> 1 (videos and DLC valid: frames = camera pulses +-2); NWBs rebuilt with DLC 2026-11-02 |
| `end_aligned_dlc_sessions` (last DLC frame <-> last camera timestamp; the camera pulsed before the first saved frame) | `converters/DLC_to_nwb.py` | MH013_20250218 (side 882, top 868 leading timestamps dropped), MH036_20250517 (side 510, top 515); rebuilt 2026-11-03, verified with 006 (NWB pairing = best offset - 3..8 frames in every third). AB153_20250209 top (47), AB127_20240823 top (video 2542 frames longer than the pulses: first 2542 frames dropped), AB153_20250209 side (47); rebuilt 2026-11-03 (after: AB127 top offset 6/7/6, AB153 side 5/6/4). AB153 top added 2026-11-03 (user) and rebuilt; after: the same metrics peak at 13-18, as in MH009. Evidence: 007 top metrics vs licks (37 licks) peak 43-49 frames later than the same metrics in control MH009 (whisker_angle, whisker_base_y, whisker_tip_y, top_particle_y, top_nose_base_y / tip_y), i.e. end-aligned (+47); whisker stimuli (38) inconclusive. Backups: <session>_backup_20261103.nwb (AB153 also _1126, side-only intermediate) |
| `piecewise_dlc_sessions` (reconverted 2026-11-03; lick-locked tongue latency per third after: MH013 -13/+23/+23 ms, MH017 -24/-18/+1, MH004 -17/-21/-33, AB116 (from 2,175 s) -31/-12/-12) (frames dropped during the video: per camera, (first DLC frame, offset pulse - frame) segments) | `converters/DLC_to_nwb.py` | MH013_20250219 (side 7 steps 0 -> 206 frames, top 0 -> 214), MH017_20250302 (side 38 -> 130, top 38 -> 127, 3 steps only), MH004_20241206 (one drop at frame 509,751 / 2,549 s: side +55, top +95), AB116_20240726 (end-aligned, DLC NaN before frame 434,963 / 2,175 s: continuous drops, not recoverable; clause `dlc_unusable_before`); ssl-dlc-alignment-debug 008 (lick-locked tongue, side lead 5 frames; top on the side step times). Not corrected: AB092_20231205 / AB095_20231212 (in register wherever there is licking; the earlier drift came from late windows without licks), AB085_20231005 (user: SSD full, the NIDQ stopped logging camera pulses before the video ended; start-aligned as stored, frames after the last pulse have no timestamp) |
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
| MH020_20250304_161732 (top), MH006_20250123_131233 | DLC / video length != pulses, no run matches |
| AB155_20250208_175103 | resolved 2026-11-06: the video spans both runs; joined DLC + camera on / off pulses removed per part (frames = DLC rows), see "AB155 day-0 merge" |
| single-run sessions with tens to thousands of missing frames (AB092, AB095, MH013 0218, MH036 0517, MH009 0206, AB151, ...) | location of the missing frames unknown; table `combined_results_ks4/ssl-dlc-alignment-debug/tables/camera_run_scan.csv` |
| MH028_20250503_112316 | both DLC files named `side`; the top-camera video was probably tracked with the side model |

## Top camera rotated 180 deg (setup 2; clause `top_rotated_180_sessions` in `converters/DLC_to_nwb.py`, confirmed by the user on video frames 2026-11-03)
Top view nose up / spout from the top (setup 1: nose down), identified from DLC (top_nose_tip_y - top_nose_base_y) and
one frame per session (ssl-dlc-utils-validation 008, figures/008_top_orientation_p*.png, tables/008_top_orientation.csv):
MH006_20250123, MH007_20250202 / 0203 / 0204, MH008_20250203 / 0205, MH009_20250206 / 0207 / 0208,
MH010_20250207 / 0208 / 0209, MH025_20250421 / 0422, MH027_20250422 (15 sessions). Setup 2 but not rotated:
MH004_20241205 / 1206. No DLC rerun on flipped videos is planned (user, 2026-11-09): orientation is handled only by the
converter check below.
Since 2026-11-04 the converter decides the rotation from the DLC data, not from this list (user: the DLC reruns may
be inferred on videos flipped before DLC, whose keypoints are already upright and must not be rotated again): top
keypoints are rotated when the median nose_tip_y - nose_base_y over frames with both nose likelihoods >= 0.6 is below
-20 px (upright sessions +61..+106 px; the existing DLC of the sessions above -49..-60 px, i.e. DLC was run on the raw
upside-down videos); |median| <= 20 px or < 1% confident frames: not rotated, warning printed.
Top DLC failed (check 2026-11-09 on the shuffle5 best-540 top filtered.h5): MH009_20250207 / 0208, MH025_20250421 /
0422, MH027_20250422 have 0% frames with both nose likelihoods >= 0.6 (median nose_tip 0.15-0.46, nose_base
0.08-0.54, whisker_tip 0.01-0.58), so the check returns "undetermined" and the keypoints stay unrotated. Their top
kinematics in the NWB are not usable; they are NOT in ephys_utilities VIDEO_EXCLUDED_SESSIONS (pending user decision).
Check of the NWB_ks4 files (2026-11-09; converted 2026-11-03 with the session list, MH007 0203 on 2026-11-04): the 10
well-tracked upside-down sessions are upright in the NWB (median top nose_tip_y - nose_base_y +49..+60 px over
confident frames, as in setup 1 sessions); the current data-based check gives the same rotation for them, so no
reconversion is needed. The 5 failed sessions have no confident nose frames: tracking failed, it is not an orientation
problem, and rotating (or not) cannot fix it.
Distances and whisker_angle are rotation invariant; raw top x / y are rotated back to the setup 1 orientation and
top_nose_angle changes sign. Setup 2 px/mm (reference_setup2: side 28.83, top 16.75): clause `setup2_sessions` in
`converters/DLC_to_nwb.py` for these 15 sessions and MH004_20241205 / 1206 (the yaml 'setup' is set per experimenter,
so they had been converted with setup 1 px/mm); evidence: probe sheet (ephys days) and head size in pixels
(ssl-dlc-utils-validation 004). All 17 reconverted 2026-11-03.

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
- Passive-labelled task trials inside the active epoch (review 2026-11-09): MH068_20260125 trials 88-102, MH026_20250422
  94-102, AB128_20240829 52 (variable ITIs, hits / false alarms after a burst of perf 6 trials); MH064_20260114 666-686
  are labelled active but at the 3.1 s passive pace before passive_post. User decision 2026-11-09: keep the labels as
  they are (no converter clause, no new passive-trial function).

## Deleted raw sessions
- MH002_20241127_115446 (8 trials) and MH002_20241128_101855 (14 trials): aborted auditory sessions, restarted minutes
  later; raw folders deleted 2026-11-02 (user; via haas, the Windows share is read-only). MH002 pretraining is now
  free_licking_-5 ... auditory_-1 (yamls regenerated, NWBs converted). Their yaml folders in
  `Axel_Bisi/data/MH002/` and `Myriam_Hamon/data/MH002/` were deleted too.

## Not converted / excluded
| session | reason |
|---|---|
| AB091, AB096 | not part of the SSL dataset (day labels also wrong) |
| AB155_20250208_185559 | second part of a split session; merged into AB155_20250208_175103 (passive post, see "AB155 day-0 merge"); never converted on its own |
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

## Raw video file names (lsens-data, fixed 2026-11-04)
Both camera videos of these MH sessions were saved as `<mouse>_side <date> <time>.avi`. The top video was identified from
a frame of each file (top = nose seen from above with the spout; side = eye + snout profile), cross-checked with the
existing DLC `*_top / *_side ... filtered.h5` names (always in agreement), and renamed `<mouse>_top ...` on
`/mnt/lsens-data/<mouse>/Recording/<session>/Video/` (user request). DLC outputs and NWBs were already correct (DLC
had been run with the right camera names); only tools that pick the raw video by name (pupil QC, frame figures) used
the top video before.
MH020 0306 (15-51-36), MH021 0309 (12-09-00), MH021 0311 (11-03-13), MH022 0310 (12-00-56), MH022 0311 (13-55-33),
MH023 0316 (10-49-14), MH023 0318 (10-45-49, the later file), MH025 0421 (11-45-30), MH025 0422 (14-11-13, the later
file), MH026 0425 (15-28-57), MH027 0422 (11-01-26), MH028 0501 (10-37-51), MH028 0502 (12-37-18), MH029 0503
(12-58-38), MH030 0501 (14-44-31).
Left as is (misplaced copies, user to decide): `MH019/Recording/MH019_20250306_121844/Video` holds the MH020 0306 files;
`MH022/Recording/MH021_20250310_110250/Video` holds the MH022 0310 files. Not duplicates: AB096 / AB105 (`-converted.mp4`
copies), AB131 (`...short.avi`), MH007 0203 (two side videos, the second is the complete one).

## Pupil: three-point area (2026-11-04)
MH029_20250502_110353 (day 0): the right pupil point is off the eye in 64% of frames with high likelihood (QC jumps 9%,
verdict warn). The pupil area is recomputed with the right point replaced by the mirror image of the left point across
the top-bottom axis (`pupil_qc.THREE_POINT` in projects/ssl-pupil-qc, `dlc_helpers.PUPIL_THREE_POINT` in
ephys_utilities; on a clean session the 3-point area matches the 4-point area at r = 0.988, -4%). QC rerun: valid
99.5%, jumps 0.2%, verdict pass. The NWB keypoints are unchanged. No other session in the QC batch has the problem
(next highest QC jump rate 2.3%).

## Video / DLC audit of whisker-day sessions (2026-11-04, ssl-dlc-utils-validation 018)
260 whisker-day sessions (AB082 onwards, MH), side and top camera; table `tables/018_video_dlc_audit_whisker.csv`.
- No DLC although the videos exist: AB135_20241111 (day 0, both cameras), AB105_20240314 top (day 0), AB104_20240314
  (+1), MH001_20241203 / 20241204 (+3 / +4), MH017_20250228 (+1), MH020_20250305 (+1), MH021_20250310 (+1),
  MH022_20250310 (+1), MH032_20250508 (+1); MH019 (excluded mouse) 0304 / 0305 / 0306.
- Video header blank (first bytes zero, frame data intact; unreadable by ffmpeg / OpenCV, likely why DLC never ran):
  AB104_20240314 side, AB105_20240314 top, MH032_20250508 side and top. Recoverable into a new file by rebuilding the
  AVI header (AB104: from the top video, same size and settings); raw files untouched.
- DLC exists but the NWB has no DLC series (reconvert): MH023_20250317, MH028_20250503.
- One camera never saved: AB082_20230630 top, MH064_20260117 side.
- File prefix typo: MH006_20250123 videos named `M006_side / M006_top` (DLC and NWB fine).
- Misplaced copy `MH022/Recording/MH021_20250310_110250` (MH021 behaviour files + MH022 0310 videos, all duplicates)
  moved to `MH021/misplaced_from_MH022/MH021_20250310_110250_with_MH022_videos`.
- Frames vs camera pulses mismatches in the NWB are all known cases (end-aligned / drift clauses, incomplete DLC).
- Reconverted 2026-11-04 with their existing DLC: MH023_20250317 (frames = pulses 579,636 on both cameras, lick-locked
  tongue offset +6 frames: in register) and MH028_20250503 (NOT aligned: camera pulses have 2 gaps of ~35 s, video has
  2,678 (side) / 2,219 (top) fewer frames than pulses, tongue offset +232 frames: needs the pulse <-> frame
  investigation of the drift sessions; do not use its DLC until then). AB107_20240318 top video also has a blank AVI
  header (DLC and NWB complete).
- DLC frames > camera pulses, not listed above (rebuild scan 2026-11-06, `ssl-table-rebuild-20261106/004`), checked
  2026-11-09 with the lick-triggered jaw opening (piezo bout onsets) per sixth of the session:
  - MH028_20250502_123728 (+1): 1,062,629 frames vs 972,622 pulses (+90,007, ~450 s). Frames are start-aligned (lick-
    triggered jaw opening as on MH028 0501, best lag -32 frames = jaw opens before piezo contact, same in both halves):
    the extra frames are after the last pulse and unused. DLC usable.
  - MH015_20250222_194156 (+1): +1,394 frames (~7 s), start-aligned (no drift across the session); jaw_distance has
    tracking outliers (1st percentile 56 vs median 465 px) that dlc_helpers cleaning must remove. DLC usable.
  - MH028_20250503_112316 (+2, known not aligned): side jaw lags the licks by a constant -2,712 frames (13.56 s) in both
    halves. Both views now in ephys_utilities `VIDEO_EXCLUDED_SESSIONS` (was top only).
    Cause (2026-11-09): the 2 "gaps" are at 13-52 s, before the video: a camera test run of 2,677 (side) / 2,218 (top)
    pulses and 1 stray pulse, then the last run from 51.97 s = 511,279 pulses = 511,279 frames on both cameras. Fix
    (user 2026-11-09): DLC end-aligned, side and top (NWB_converter converters/DLC_to_nwb.py `end_aligned_dlc_sessions`).
    Reconverted 2026-11-09 (ssl-dlc-alignment-debug 009; previous file kept as `..._backup_20261109.nwb`): frames =
    timestamps from 51.968 s, jaw opens 16 frames before piezo contact (was -2,692), trials / piezo licks unchanged (no
    units in this NWB). `VIDEO_EXCLUDED_SESSIONS`: side removed, top kept (top video probably tracked with the side model).
  - MH028 0502 / MH015 0222 need no fix: dlc_helpers pairs frames from the first timestamp and drops the extra frames.
- Pupil QC rule (2026-11-04): synchronous loss of the pupil points after full tracking = blink if <= 0.2 s (was 0.5 s),
  squint (eye closed) if 0.2-5 s, failed if longer. Labels only (valid frames unchanged). QC rerun on all 244 sessions
  (215 pass, 14 warn, 15 fail); MH023_20250317 warn (81.5% valid, 15.7% squint).

## AB155 day-0 merge (started 2026-11-04; projects/ssl-ab155-session-merge)
AB155_20250208_175103 (g0) + AB155_20250208_185559 (g1, passive post) are being merged into one session (id of part 1)
from the April 2025 supercat (checked: g0 byte-identical, g1 equal up to +-2 int16 units, all trial TTLs present).
Kuma: joined run copied to /work/lsens/bisi/data/AB155/AB155_20250208_175103/Ephys/catgt_AB155_g0; the per-run g0 work
folder renamed /work/lsens/bisi/data/AB155/AB155_20250208_175103_pergate. Existing NWBs / per-run outputs kept until
the merged NWB is validated.
- 2026-11-06: merged sorting (run 4819418) copied (009, rsync without preprocess/) to
  `Axel_Bisi/data/AB155/AB155_20250208_175103/Ephys/catgt_AB155_g0`; previous g0-only folder renamed
  `Ephys_g0only_backup_20261106`; `Video/*filtered.h5` = part 1 + part 2 DLC (part-1-only kept as
  `*_part1only_backup_20261106.h5`).
- Converter clause (uncommitted, NWB_converter on hold): `utils/joined_sessions.py` (`JOINED_SESSIONS`) with hooks in
  `server_paths.get_behavior_results_file` (results + results_2), `ephys_converter_misc.load_ephys_sync_timestamps`
  (part-1 end TTL; camera on / off pulse per part), `extract_ephys_timestamps` (part-1 log licks, part-2 NIDQ licks)
  and `NWB_conversion_parallel.convert_data_to_nwb` (recording_gap annotation; part 2 shifted by 903.667601 s after
  saving, epochs recomputed, BehavioralEpochs recording_gap). Also fixed: `t_ttl_imec[:len(t_ttl_imec)]` ->
  `[:len(t_ttl_log)]` in extract_ephys_timestamps.
- Test (018_converter_clause_test.py, output nwb_test/converter_clause): identical to the validated test NWB (010 + 012;
  units matched by probe and cluster; largest difference 0.18 us in useTheseTimesStop). Final conversion done 2026-11-06
  (019_final_conversion.py): NWB_ks4/AB155_20250208_175103.nwb, same comparison result; the previous part-1-only NWB
  moved to SessionToConcat/AB155_20250208_175103_joined/nwb_old/.

## DLC reruns by mhamon (arrived 2026-11-04 15:26-15:51) and reconversion
New `<mouse>_<cam>_<yy-mm-dd>_<hh-mm-ss>_filtered.h5` (Myriam_Hamon/data; rows = video frames for every file):
MH001 1203 / 1204, MH007 0203 side (14-03-31 video), MH017 0228, MH018 0301, MH019 0304 / 0305, MH020 0304 top /
0305, MH021 0310, MH022 0309 / 0310, MH023 0317, MH032 0508, AB104 0314, AB105 0315, AB152 0129 side, AB154 0207 top.
Superseded outputs renamed `*_backup_20261104.<ext>` (not read by the converter): MH020 0304 top, MH022 0309 side/top,
MH023 0317 side/top (shuffle5 best-540), AB152 0129 side, AB154 0207 top (shuffle3 090, truncated). The AB files were
copied (byte-identical) into Axel_Bisi/data/<mouse>/<session>/Video, where the converter reads AB DLC. Model of the
reruns not recorded in the file names (user: same keypoints, fine). All 18 NWBs reconverted 2026-11-04 (previous NWBs
kept as *_backup_20261104*.nwb); pupil QC to be rerun on them (QC frames are matched to the NWB keypoints).
No top reruns for the upside-down setup-2 sessions (user, 2026-11-09): the converter orientation check (section "Top
camera rotated 180 deg") rotates top keypoints when needed; the 5 sessions with failed top DLC keep their old files.
- camera_flag corrected 0 -> 1 (user, 2026-11-04) for MH018_20250301_190738 and AB105_20240315_124421 (both day +1,
  behaviour only): cameras were triggered normally (log_continuous cam1 / cam2: 228,836 / 433,428 pulses from 3.0 s =
  video frames = DLC rows). Edited in lsens-data `{Recording,Training}/<session>/session_config.json` (originals kept as
  `session_config_backup_20261104.json`) and in the converter yaml (Axel_Bisi/data, backup `*_backup_20261104.yaml`);
  NWBs reconverted (frames = pulses, side lick offset 5 / 3 frames), pupil QC pass (99.3 / 98.5% valid).
