# ssl-pupil-qc

Quality control of the DLC pupil tracking (side camera) for SSL sessions: one A4 page, metrics, per-trial QC and a
per-frame state table per session, written to `combined_results_ks4/<mouse>/<behaviour>_<day>/pupil_qc/`.

Design: the user's page specification (2026-11-03). The DLC network (`Tracking2D-Axel-2024-08-10`) has 4 pupil points
(`pupil_top/left/right/bottom`) and no eyelid points, so blinks are separated from tracking failures by the likelihood
timing signature (all points lost within <= 15 ms after full tracking, recovered within 0.5 s); the eye-ROI mean
intensity from the raw video validates it on pilot sessions (`--intensity`).

| File | Role |
|---|---|
| `pupil_qc.py` | load NWB_ks4 (keypoints, motion, active trials), match the side video (NWB keypoints vs each DLC filtered h5: handles top videos named `_side` and end-aligned sessions), frame states, cleaned diameter, session and trial metrics, verdict |
| `video.py` | example frames (OpenCV random access), eye-ROI intensity (ffmpeg crop pipe) |
| `page.py` | the A4 page (panels A-M) |
| `run.py` | CLI: `run.py <session_id> ... [--intensity] [--jobs N]` |

Data: NWB_ks4 pupil series and trial table (session time); raw side videos in
`/mnt/lsens-data/<mouse>/{Recording,Training}/<session>/Video/<mouse>_side <date> <time>.avi` (uncompressed 640 x 480
grey, 100 or 200 Hz; AB100 only as `.mp4` in `Axel_Bisi/data`). Run on haas with `~/code/ibl-ai-agent/.venv` (OpenCV,
pytables, openpyxl) and `/usr/bin/ffmpeg`.

Status: pilot on 5 sessions (AB131 0, AB164 0, MH022 +2, AB116 +1, AB102 0 at 100 Hz) before any batch.

006 (2026-11-09): the early post-stimulus dip (minimum 30-190 ms after trial start) is present in the 4-point area,
the vertical AND the horizontal diameter, the horizontal one 2-3x smaller (AB102 whisker -26 / -18 / -10 %); the
likelihood cutoff (0.1, 0.5, 0.8) barely changes it (0.8 only removes frames). Too fast for a pupil constriction and
larger vertically: most likely a startle (eyelid narrowing / eye movement), not a pupil signal; a higher cutoff does
not remove it.
