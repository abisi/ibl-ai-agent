"""Facemap lick decoding, ROI step 1 (haas): automatic face ROIs from DLC keypoints, cropped from the clips videos.

ROIs (user 2026-11-04): rectangles centred on the median coordinate of DLC keypoints of the side video,
  eye      centroid of pupil_top / bottom / left / right
  jaw      jaw
  whisker  particle (side-view whisker pad marker)
  snout    nose_tip
Median over frames with likelihood >= MIN_LIK (fallback: the top-10 % likelihood frames when fewer than 5 % pass).
Sizes scale with the face: SCALE_PX = distance pupil centroid -> nose tip; width / height = SIZE[roi] x SCALE_PX
(fractions calibrated on the hand-drawn ROIs of the 6 ssl-facemap-pilot sessions). Whisker pad (user 2026-11-04):
the particle sits slightly low on the pad, so the box is 20 % larger than calibrated and moved up by WHISKER_UP_MM
after placement (also after the fallback below); mm -> px through the face scale, assuming FACE_MM = 10 mm from the
pupil centroid to the nose tip (no camera calibration available). Boxes are clipped to the frame;
the jaw box is cut on the side facing the spout when the spout keypoint falls inside it (+ SPOUT_CLEAR px).
Coordinates in clips-video pixels (raw 640x480 / 2).
Whisker fallback (user 2026-11-04): the particle keypoint is mis-tracked in a few sessions (box slides onto the
jaw / spout). The particle is suspicious when it lies at or below the jaw keypoint, within SPOUT_NEAR x SCALE_PX of
the spout, or > 3 MAD off the population median perpendicular to the pupil -> nose-tip axis (face frame, in
SCALE_PX). The whisker box is then re-centred midway between the jaw ROI and snout ROI centres
(rois.csv: whisker_fallback = True, whisker_reason).

Outputs: combined_results_ks4/ssl-facemap-lick-decoding/roi/{videos/<session>_<roi>.mp4, previews/<session>.png,
rois.csv}. Run on haas:
    nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python 004_roi_auto.py [--sessions ...] [--preview-only] --jobs 16
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-facemap-lick-decoding")
CLIPS = BASE / "clips"
OUT = BASE / "roi"
SCALE = 0.5  # raw -> clips pixels
W, H = 320, 240
MIN_LIK = 0.5
SIZE = {"eye": (0.32, 0.30), "jaw": (0.48, 0.28), "whisker": (0.384, 0.384), "snout": (0.30, 0.35)}  # (w, h) x SCALE_PX
FACE_MM = 10.0  # assumed pupil centroid -> nose tip distance (mm)
WHISKER_UP_MM = 1.0
SPOUT_CLEAR = 4
SPOUT_NEAR = 0.25
ANCHORS = {"eye": ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"], "jaw": ["jaw"],
           "whisker": ["particle"], "snout": ["nose_tip"]}
COLORS = {"eye": "#1f77b4", "snout": "#2ca02c", "jaw": "#d62728", "whisker": "#ff7f0e"}


def median_xy(d, bp):
    lik = d[(bp, "likelihood")]
    sel = lik >= MIN_LIK
    if sel.mean() < 0.05:
        sel = lik >= lik.quantile(0.9)
    return (float(d.loc[sel, (bp, "x")].median()) * SCALE, float(d.loc[sel, (bp, "y")].median()) * SCALE,
            float(sel.mean()))


def place(h5):
    d = pd.read_hdf(h5)
    d.columns = d.columns.droplevel(0)
    kp = {bp: median_xy(d, bp) for bp in sorted(set(c[0] for c in d.columns))}
    cent = {r: np.mean([kp[b][:2] for b in bps], 0) for r, bps in ANCHORS.items()}
    scale = float(np.hypot(*(cent["snout"] - cent["eye"])))
    rois = {}
    cent["whisker"] = cent["whisker"] - [0.0, WHISKER_UP_MM * scale / FACE_MM]  # up = smaller y
    for r, (fw, fh) in SIZE.items():
        w, h = fw * scale, fh * scale
        x0, y0, x1, y1 = cent[r][0] - w / 2, cent[r][1] - h / 2, cent[r][0] + w / 2, cent[r][1] + h / 2
        if r == "jaw" and "spout" in kp:
            sx, sy = kp["spout"][:2]
            if x0 < sx < x1 and y0 < sy < y1:
                if sx > cent[r][0]:
                    x1 = sx - SPOUT_CLEAR
                else:
                    x0 = sx + SPOUT_CLEAR
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
        rois[r] = [x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2]
    return rois, kp, scale


def mean_frame(video, n=200):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = []
    for f in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            acc.append(img[:, :, 0].astype(np.float32))
    cap.release()
    return np.mean(acc, 0)


def preview(sid, img, kp, rois):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.imshow(img, cmap="gray")
    for name, (x, y, w, h) in rois.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=COLORS[name], lw=1.5))
        ax.text(x, y - 2, name, color=COLORS[name], fontsize=7)
    for bp in sum(ANCHORS.values(), []) + ["spout"]:
        if bp in kp:
            ax.plot(*kp[bp][:2], "+", color="red" if bp == "spout" else "yellow", ms=5)
    ax.set_title(sid, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "previews" / f"{sid}.png", dpi=100)
    plt.close(fig)


def crop(sid, name, roi):
    x, y, w, h = roi
    dst = OUT / "videos" / f"{sid}_{name}.mp4"
    if dst.exists():
        return
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIPS / f"{sid}_clips.mp4"), "-vf",
                    f"crop={w}:{h}:{x}:{y}", "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(tmp)], check=True)
    tmp.rename(dst)


def process(args):
    sid, preview_only = args
    try:
        ix = np.load(CLIPS / f"{sid}_clips.npz", allow_pickle=True)
        rois, kp, scale = place(str(ix["dlc_h5"]))
        preview(sid, mean_frame(CLIPS / f"{sid}_clips.mp4"), kp, rois)
        if not preview_only:
            for name, roi in rois.items():
                if roi[2] >= 8 and roi[3] >= 8:
                    crop(sid, name, roi)
        row = dict(session_id=sid, scale_px=scale, status="ok")
        for bp in ("particle", "jaw", "spout"):
            if bp in kp:
                row[f"{bp}_kx"], row[f"{bp}_ky"] = kp[bp][:2]
        for name, roi in rois.items():
            row.update({f"{name}_{k}": v for k, v in zip("xywh", roi)})
        for bp in sum(ANCHORS.values(), []):
            row[f"{bp}_frac_lik"] = kp[bp][2]
        return row
    except Exception as e:
        return dict(session_id=sid, status=f"error: {type(e).__name__}: {e}")


def fix_whisker(df, preview_only):
    """Re-centre whisker boxes with a suspicious particle anchor midway between the jaw and snout ROIs."""
    c = lambda k: np.c_[df[f"{k}_x"] + df[f"{k}_w"] / 2, df[f"{k}_y"] + df[f"{k}_h"] / 2]  # noqa: E731
    eye, sn, jaw = c("eye"), c("snout"), c("jaw")
    sc = df.scale_px.to_numpy()
    part = df[["particle_kx", "particle_ky"]].to_numpy(float)
    u = (sn - eye) / np.linalg.norm(sn - eye, axis=1, keepdims=True)
    v = np.c_[-u[:, 1], u[:, 0]]
    b = ((part - eye) * v).sum(1) / sc
    mad = np.median(np.abs(b - np.median(b))) * 1.4826
    reasons = {"below_jaw": part[:, 1] >= df.jaw_ky.to_numpy(float),
               "near_spout": np.hypot(*(part - df[["spout_kx", "spout_ky"]].to_numpy(float)).T) < SPOUT_NEAR * sc,
               "face_frame_outlier": np.abs(b - np.median(b)) > 3 * mad}
    bad = np.any(list(reasons.values()), axis=0)
    df["whisker_fallback"] = bad
    df["whisker_reason"] = ["+".join(k for k, m in reasons.items() if m[i]) for i in range(len(df))]
    for i in np.flatnonzero(bad):
        sid = df.session_id.iloc[i]
        cen = (jaw[i] + sn[i]) / 2 - [0.0, WHISKER_UP_MM * sc[i] / FACE_MM]
        w, h = SIZE["whisker"][0] * sc[i], SIZE["whisker"][1] * sc[i]
        x0, y0 = max(0, int(round(cen[0] - w / 2))), max(0, int(round(cen[1] - h / 2)))
        x1, y1 = min(W, int(round(cen[0] + w / 2))), min(H, int(round(cen[1] + h / 2)))
        roi = [x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2]
        df.loc[df.index[i], ["whisker_x", "whisker_y", "whisker_w", "whisker_h"]] = roi
        if not preview_only:
            (OUT / "videos" / f"{sid}_whisker.mp4").unlink(missing_ok=True)
            crop(sid, "whisker", roi)
        ix = np.load(CLIPS / f"{sid}_clips.npz", allow_pickle=True)
        _, kp, _ = place(str(ix["dlc_h5"]))
        rois = {k: [int(df[f"{k}_{q}"].iloc[i]) for q in "xywh"] for k in SIZE}
        preview(sid, mean_frame(CLIPS / f"{sid}_clips.mp4"), kp, rois)
        print("whisker fallback:", sid, df.whisker_reason.iloc[i], roi, flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    for sub in ("previews", "videos"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    sids = a.sessions or sorted(p.name.replace("_clips.mp4", "") for p in CLIPS.glob("*_clips.mp4"))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(process, [(s, a.preview_only) for s in sids]))
    df = pd.DataFrame(rows)
    if not a.sessions:
        df = fix_whisker(df, a.preview_only)
    df.to_csv(OUT / ("rois_test.csv" if a.sessions else "rois.csv"), index=False)
    print(df.status.value_counts().to_string())
    print(df.drop(columns=[c for c in df.columns if c.endswith("frac_lik")]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
