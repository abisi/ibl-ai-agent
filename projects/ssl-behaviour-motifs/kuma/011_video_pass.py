"""011 (Kuma, CPU in an l40s allocation) -- one sequential pass over a session's raw video (one view):
(1) motion energy (ME) per face region at 200 Hz (v2), (2) per-state ME maps, (3) short clips of the global top states for grid
movies, (4) the mean frame.

- Frames: 200 Hz uncompressed avi, frame i = DLC frame i; grey, downscaled x2. ME_t = mean |f_t - f_{t-1}| in a region;
  v2: kept at 200 Hz, frame i = keypoint / state frame i.
- Regions (boxes around median raw keypoint positions; d = median nose_tip-jaw (side) / nose_tip-whisker_base (top)
  distance): side snout (nose_tip, half-size 0.4 d), mouth (jaw, 0.4 d), cheek (mean of nose_base and jaw_ref, 0.4 d);
  top snout (nose_tip, 0.4 d), whisker_pad (whisker_base, 0.4 d), whiskers (whisker base-tip midpoint, 0.6 d); face =
  bounding box of all modelled keypoints + 0.5 d.
- States of interest per method (kpms seed 0, AR-HMM, lightning-action; this view's fits): global top 10 by usage
  (all 8 sessions) and this session's top 10. ME map of a state = mean |f_t - f_{t-1}| over the frames of up to 50
  randomly sampled bouts (>= 60 ms = 12 frames; at most the central 40 frames = 200 ms of each bout).
- Clips (global top 10 only): one bout per state in this session (>= 100 ms, length closest to the state's median bout),
  from 0.25 s before to 1.0 s after bout onset, every 2nd frame (100 Hz), face crop resized to 192 x 144.
Output: results/video/<sid>_<view>.npz, results/video/clips/<sid>_<view>.npz
Usage: python 011_video_pass.py <sid> <view>
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/work/lsens/bisi/behaviour_motifs")
INP, RES = BASE / "data" / "inputs", BASE / "results"
OUT = RES / "video"
METHODS = ("kpms", "arhmm", "lightning_action")
NTOP, NINST, MAXF = 10, 50, 40
CLIP = (-0.25, 1.0)
CLIP_WH = (192, 144)


def state_file(sid, method, view):
    """v2: chosen models from data/selected_models.json (same file as tables/selected_models.json on the NAS)"""
    if method == "lightning_action":
        return RES / "lightning_action" / view / f"{sid}_probs.npz"
    m = json.load(open(BASE / "data" / "selected_models.json"))[method][view]
    base = RES / "kpms" / view / m["norm"] if method == "kpms" else RES / "arhmm" / view
    return base / m["tags"]["0"] / f"{sid}.npz"


def load_z(sid, method, view):
    d = np.load(state_file(sid, method, view))
    return np.asarray(d["z"] if "z" in d else d["pred"])


def bouts(z):
    ch = np.flatnonzero(np.diff(z) != 0) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(z)]
    return st, en, z[st]


def main():
    sid, view = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(0)
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    sids = [p.name.split("_" + view)[0] for p in sorted(INP.glob(f"*_{view}_keypoints.npz"))]
    k = np.load(INP / f"{sid}_{view}_keypoints.npz")
    bps = [str(b) for b in k["bodyparts"]]
    raw = (k["xy"].astype(float) - k["shift"]) / float(k["mm_per_px"])  # v3: aligned mm -> raw pixels
    med = {b: np.nanmedian(raw[:, i], 0) for i, b in enumerate(bps)}
    rs = (k["xy_static"].astype(float) - k["shift"]) / float(k["mm_per_px"])  # nose_base (v3: not modelled)
    med.update({str(b): np.nanmedian(rs[:, i], 0) for i, b in enumerate(k["static_names"])})
    if view == "side":
        d = np.linalg.norm(med["nose_tip"] - med["jaw"])
        regions = {"snout": (med["nose_tip"], 0.4 * d), "mouth": (med["jaw"], 0.4 * d),
                   "cheek": ((med["nose_base"] + med["jaw_ref"]) / 2, 0.4 * d)}
    else:
        d = np.linalg.norm(med["nose_tip"] - med["whisker_base"])
        regions = {"snout": (med["nose_tip"], 0.4 * d), "whisker_pad": (med["whisker_base"], 0.4 * d),
                   "whiskers": ((med["whisker_base"] + med["whisker_tip"]) / 2, 0.6 * d)}
    allp = np.array(list(med.values()))
    lo, hi = allp.min(0) - 0.5 * d, allp.max(0) + 0.5 * d
    vid = glob.glob(str(BASE.parent / "data" / sid.split("_")[0] / sid / "Video" / f"*_{view} *.avi"))[0]
    cap = cv2.VideoCapture(vid)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    box = lambda c, h: (max(int((c[1] - h) / 2), 0), min(int((c[1] + h) / 2), H), max(int((c[0] - h) / 2), 0), min(int((c[0] + h) / 2), W))  # noqa: E731
    rbox = {r: box(c, h) for r, (c, h) in regions.items()}
    face = (max(int(lo[1] / 2), 0), min(int(hi[1] / 2), H), max(int(lo[0] / 2), 0), min(int(hi[0] / 2), W))
    rbox["face"] = face
    T50 = len(k["t"])  # v2: states, keypoints and video frames all at 200 Hz (frame i = DLC frame i)
    nfr = min(nvid, T50, int(os.environ.get("BM_MAXF", 10 ** 9)))  # BM_MAXF: quick test only
    # tasks: ME-map frames and clip frames
    me_task, clip_task, meta = {}, {}, {}
    clip_keys = []
    for m in METHODS:
        zs = {s: load_z(s, m, view) for s in sids}
        allz = np.concatenate([z[z >= 0] for z in zs.values()])
        gtop = [1, 2, 3, 4, 5] if m == "lightning_action" else list(np.argsort(-np.bincount(allz))[:NTOP])
        z = zs[sid][:T50]
        stop = [1, 2, 3, 4, 5] if m == "lightning_action" else list(np.argsort(-np.bincount(z[z >= 0]))[:NTOP])
        st, en, lab = bouts(z)
        for s in dict.fromkeys(gtop + stop):
            key = f"{m}|{int(s)}"
            sel = np.flatnonzero((lab == s) & (en - st >= 12))
            meta[key] = dict(global_top=s in gtop, session_top=s in stop, n_bouts=len(sel))
            if len(sel):
                for i in rng.choice(sel, min(NINST, len(sel)), replace=False):
                    a, b = st[i], en[i]
                    mid = (a + b) // 2
                    for f in range(max(a, mid - MAXF // 2), min(b, mid + MAXF // 2)):
                        me_task.setdefault(f, []).append(key)
            if s in gtop:
                lens = (en - st)[lab == s]
                ok = np.flatnonzero((lab == s) & (en - st >= 20))
                if len(ok):
                    target = np.median(lens)
                    i = ok[np.argmin(np.abs((en - st)[ok] - target))]
                    on = st[i]
                    frames = range(on + int(CLIP[0] * 200), on + int(CLIP[1] * 200), 2)
                    clip_keys.append(key)
                    meta[key]["clip_onset_frame"] = int(on)
                    meta[key]["clip_bout_ms"] = int((en[i] - st[i]) * 5)
                    for j, f in enumerate(frames):
                        clip_task.setdefault(f, []).append((key, j))
    nclip = len(range(int(CLIP[0] * 200), int(CLIP[1] * 200), 2))
    me_sum = {key: np.zeros((H, W), np.float32) for key in meta}
    me_n = {key: 0 for key in meta}
    clips = {key: np.zeros((nclip, CLIP_WH[1], CLIP_WH[0]), np.uint8) for key in clip_keys}
    me_reg = {r: np.zeros(nfr, np.float32) for r in rbox}
    mean_frame = np.zeros((H, W), np.float64)
    nmean = 0
    prev = None
    for f in range(nfr):
        ok, img = cap.read()
        if not ok:
            nfr = f
            break
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            dif = np.abs(g - prev)
            for r, (y0, y1, x0, x1) in rbox.items():
                me_reg[r][f] = dif[y0:y1, x0:x1].mean()
            for key in me_task.get(f, ()):
                me_sum[key] += dif
                me_n[key] += 1
        if f % 200 == 0:
            mean_frame += g
            nmean += 1
        for key, j in clip_task.get(f, ()):
            y0, y1, x0, x1 = face
            clips[key][j] = cv2.resize(g[y0:y1, x0:x1], CLIP_WH, interpolation=cv2.INTER_AREA).astype(np.uint8)
        prev = g
        if f % 100000 == 0:
            print(sid, view, f, nfr, flush=True)
    cap.release()
    me50 = {r: v[:nfr] for r, v in me_reg.items()}  # v2: 200 Hz
    np.savez_compressed(OUT / f"{sid}_{view}.npz", t=k["t"][:nfr], regions=np.array(list(rbox)),
                        boxes=np.array([rbox[r] for r in rbox]), mean_frame=(mean_frame / max(nmean, 1)).astype(np.float32),
                        me_keys=np.array(list(me_sum)), me_maps=np.array([me_sum[q] / max(me_n[q], 1) for q in me_sum], np.float16),
                        me_counts=np.array([me_n[q] for q in me_sum]), meta=np.array([str(meta)]),
                        **{f"me_{r}": v for r, v in me50.items()})
    np.savez_compressed(OUT / "clips" / f"{sid}_{view}.npz", keys=np.array(clip_keys),
                        clips=np.array([clips[q] for q in clip_keys]) if clip_keys else np.zeros((0,)),
                        meta=np.array([str({q: meta[q] for q in clip_keys})]))
    print("done", sid, view, nfr, len(meta), len(clip_keys), flush=True)


if __name__ == "__main__":
    main()
