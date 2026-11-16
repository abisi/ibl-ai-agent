"""Facemap lick decoding, step 1 (haas): peri-trial clips of the side video around every active whisker trial.

Why clips: the raw side videos are uncompressed 640x480 AVIs (200-350 GB each, ~65 TB for the 242 whisker-day sessions)
and the NAS delivers ~0.5 GB/s to haas whatever the number of streams (measured 2026-11-03), i.e. ~36 h for full videos.
Upcoming-lick decoding only needs pre-stimulus frames, so only -2.5 to +0.5 s around each active whisker trial
(context != passive, perf != 6, early_lick == 0) is read (~6 TB, ~3 h).

Per session: side video + NWB-index -> frame offset from ssl-pupil-qc's matcher (video frame = NWB index + offset);
frames binned 2x (320x240, area) and encoded H.264 CRF 18 into one clips video; each clip starts with a duplicate of
its first frame so Facemap's frame difference is 0 at clip boundaries (drop frames with lead == 1 in the analysis).
Index npz: per output frame the clip number, NWB side-camera index, NWB timestamp and lead flag; per clip the trial
row (NWB trials index), start_time, lick_flag.

Output: combined_results_ks4/ssl-facemap-lick-decoding/clips/<session>_clips.{mp4,npz}, clips/summary.csv
Run on haas: nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python 001_extract_clips.py --list <sessions.txt> --jobs 12
"""

import argparse
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ssl-pupil-qc"))
import pupil_qc as pq  # noqa: E402

OUT = pq.RESULTS / "ssl-facemap-lick-decoding" / "clips"
WIN = (-2.5, 0.5)
SCALE = (320, 240)
CRF = 18


def clip_ranges(trials, ts, offset, n_frames):
    """Video frame ranges [f0, f1) of the active whisker trials whose window lies inside the video."""
    rows = []
    wh = trials[trials.trial_type == "whisker_trial"]
    for i, r in wh.iterrows():
        i0, i1 = np.searchsorted(ts, [r.start_time + WIN[0], r.start_time + WIN[1]])
        f0, f1 = i0 + offset, i1 + offset
        if i0 > 0 and i1 < len(ts) and f1 <= n_frames and i1 - i0 > 10:
            rows.append(dict(trial=i, start_time=r.start_time, lick_flag=int(r.lick_flag), nwb0=i0, nwb1=i1,
                             f0=f0, f1=f1))
    return pd.DataFrame(rows)


def encoder(path, fs):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{SCALE[0]}x{SCALE[1]}",
           "-r", str(int(round(fs))), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-threads", "4", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process(sid):
    t0 = time.time()
    dst = OUT / f"{sid}_clips.mp4"
    if dst.exists() and dst.with_suffix(".npz").exists():
        return dict(session_id=sid, status="exists")
    df, trials, meta = pq.load_session(sid)
    vid = pq.find_video(meta, df)
    if not vid or not vid["video"]:
        return dict(session_id=sid, status="no video")
    ts = df.t.to_numpy()
    cap = cv2.VideoCapture(vid["video"])
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clips = clip_ranges(trials, ts, vid["offset"], n_frames)
    if clips.empty:
        return dict(session_id=sid, status="no clips")
    seekable = vid["video"].lower().endswith(".avi")  # uncompressed AVI: exact frame seeking
    tmp = dst.with_suffix(".tmp.mp4")
    enc = encoder(tmp, meta["fs"])
    clip_id, nwb_idx, lead = [], [], []
    pos = 0
    for k, c in enumerate(clips.itertuples(index=False)):
        if seekable:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c.f0)
        else:  # compressed video: decode forward to the clip
            while pos < c.f0:
                cap.grab()
                pos += 1
        frames = []
        for _ in range(c.f1 - c.f0):
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(img[:, :, 0], SCALE, interpolation=cv2.INTER_AREA))
        pos = c.f1
        if len(frames) != c.f1 - c.f0:
            cap.release()
            enc.stdin.close()
            enc.wait()
            tmp.unlink(missing_ok=True)
            return dict(session_id=sid, status=f"short read at clip {k}")
        for fr in [frames[0]] + frames:
            enc.stdin.write(fr.tobytes())
        clip_id += [k] * (len(frames) + 1)
        nwb_idx += [c.nwb0] + list(range(c.nwb0, c.nwb1))
        lead += [1] + [0] * len(frames)
    cap.release()
    enc.stdin.close()
    enc.wait()
    nwb_idx = np.asarray(nwb_idx)
    np.savez(dst.with_suffix(".npz"), clip=np.asarray(clip_id, np.int32), nwb_index=nwb_idx.astype(np.int64),
             t=ts[nwb_idx], lead=np.asarray(lead, np.int8), clips=clips.to_records(index=False),
             offset=vid["offset"], fs=meta["fs"], video=vid["video"], dlc_h5=vid["h5"], win=WIN, scale=SCALE)
    tmp.rename(dst)
    return dict(session_id=sid, status="ok", n_clips=len(clips), n_out_frames=len(clip_id), fs=meta["fs"],
                video=vid["video"], offset=vid["offset"], seconds=round(time.time() - t0))


def safe(sid):
    try:
        return process(sid)
    except Exception as e:
        (OUT / f"{sid}_clips.tmp.mp4").unlink(missing_ok=True)
        return dict(session_id=sid, status=f"error: {type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    sessions = list(a.sessions)
    if a.list:
        sessions += [s.strip() for s in open(a.list) if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(safe, sessions):
            print(r["session_id"], r["status"], r.get("n_clips", ""), r.get("seconds", ""), flush=True)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / f"summary_{time.strftime('%Y%m%d_%H%M')}.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
