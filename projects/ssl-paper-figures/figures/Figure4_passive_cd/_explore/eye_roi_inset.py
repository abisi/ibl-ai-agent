"""Eye ROI inset for the pupil panel (user 2026-11-25: "add as inset a little extracted eye ROI, high quality, with the diameter
shown"; then "find a better example where we see the pupil well; show the contour of the 4 area keypoints with white bars").

Example search over the sessions of 205 (passive state table) with a raw, uncompressed side video (lsens-data Recording /
Training) and the merged-model DLC file (*CspnextM*filtered_new.h5): per session, the middle 40 % of frames, the best-tracked
frame (highest minimum likelihood of the four pupil points among frames with a typical diameter), scored by
  quality  = that minimum likelihood
  contrast = (mean intensity of a ring around the pupil contour - mean inside the contour) / s.d. of the ring
and score = quality x contrast. The best session's frame is cropped around the pupil, upsampled (Lanczos), with the four-point
contour that defines pupil_area drawn as white bars, the points as small white dots, and a 0.5 mm scale bar (side camera
~0.030 mm / px, ephys_utilities dlc_helpers convention). The scores of every candidate are saved.
Outputs: Figures/_cache/eye_roi_inset.png (embedded by plot_pupil_traces), Figures/_explore/Figure4_passive_cd_<date>/
eye_roi_inset.{png,pdf,svg} and eye_roi_candidates.csv.
Run (haas, repo root; needs cv2): .venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/eye_roi_inset.py
"""
import datetime
import glob
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common import CACHE_DIR, FIGURES_ROOT  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

TAB = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-hitmiss-axis-passive/tables")
DATA = {"AB": Path("/mnt/lsens-analysis/Axel_Bisi/data"), "MH": Path("/mnt/lsens-analysis/Myriam_Hamon/data")}
RAW = Path("/mnt/lsens-data")
MM_PER_PX = 0.030
ORDER = ("pupil_top", "pupil_right", "pupil_bottom", "pupil_left")      # contour order (pupil_area polygon)
UPSAMPLE = 10
OUT = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"
CHOSEN = "AB157_20250412_154744"   # user 2026-11-25: picked by eye from eye_roi_contact_sheet.png (pupil and eye well visible)


def files(sid):
    mouse = sid.split("_")[0]
    h5 = sorted(glob.glob(str(DATA[mouse[:2]] / mouse / sid / "Video" / f"{mouse}_side*CspnextM*filtered_new.h5")))
    if not h5:
        return None
    stem = Path(h5[0]).name.split("DLC_")[0]
    avi = [p for r in ("Recording", "Training") for p in glob.glob(str(RAW / mouse / r / sid / "Video" / f"{stem}.avi"))]
    return (h5[0], avi[0]) if avi else None


def read_frame(video, i):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, im = cap.read()
    return im[..., 0].astype(float) if ok else None


def keypoints(h5):
    d = pd.read_hdf(h5)
    d.columns = d.columns.droplevel(0)
    xy = np.stack([d[p][["x", "y"]].to_numpy() for p in ORDER], 1)          # frames x 4 x 2
    lk = np.min(np.stack([d[p]["likelihood"].to_numpy() for p in ORDER], 1), 1)
    return xy, lk


def score(sid):
    f = files(sid)
    if f is None:
        return None
    h5, avi = f
    xy, lk = keypoints(h5)
    n = len(lk)
    mid = np.zeros(n, bool)
    mid[int(0.3 * n): int(0.7 * n)] = True
    diam = np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)
    typical = mid & (np.abs(diam - np.nanmedian(diam[mid])) < 0.15 * np.nanmedian(diam[mid]))
    if not typical.any():
        return None
    i = int(np.where(typical)[0][np.argmax(lk[typical])])
    im = read_frame(avi, i)
    if im is None:
        return None
    poly = xy[i]
    c = poly.mean(0)
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    pts = np.c_[xx.ravel(), yy.ravel()]
    inside = MPath(poly).contains_points(pts).reshape(im.shape)
    ring = MPath(c + 1.8 * (poly - c)).contains_points(pts).reshape(im.shape) & ~MPath(c + 1.2 * (poly - c)).contains_points(pts).reshape(im.shape)
    if inside.sum() < 10 or ring.sum() < 10:
        return None
    contrast = (im[ring].mean() - im[inside].mean()) / (im[ring].std() + 1e-6)
    return dict(session_id=sid, h5=h5, video=avi, frame=i, quality=float(lk[i]), contrast=float(contrast),
                diameter_px=float(diam[i]), score=float(lk[i] * contrast))


def render(r):
    xy, _ = keypoints(r["h5"])
    im = read_frame(r["video"], r["frame"])
    poly = xy[r["frame"]]
    c = poly.mean(0)
    half = int(round(2.2 * r["diameter_px"]))                     # wide enough to see the whole eye
    x0, y0 = max(int(round(c[0])) - half, 0), max(int(round(c[1])) - half, 0)
    crop = im[y0: y0 + 2 * half, x0: x0 + 2 * half]
    # contrast (user 2026-11-25: "emphasize contrast and better show pupil"): local contrast (CLAHE) on the 8-bit crop, then a
    # gamma < 1 that lifts the dark iris away from the darker pupil; display only (no measurement uses this image)
    c8 = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(c8)
    crop = 255.0 * (c8 / 255.0) ** 0.6
    big = cv2.resize(crop, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LANCZOS4)
    P = (poly - [x0, y0]) * UPSAMPLE
    set_nature_rc()
    fig = plt.figure(figsize=(22 * MM, 22 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    lo, hi = np.percentile(big, [1, 99.5])
    ax.imshow(big, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
    closed = np.vstack([P, P[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="white", lw=0.45, alpha=0.55, solid_joinstyle="miter")  # pupil_area contour
    ax.plot(P[:, 0], P[:, 1], "o", ms=1.0, color="white", alpha=0.55, mew=0)
    H, W = big.shape
    sb = 0.5 / MM_PER_PX * UPSAMPLE
    ax.plot([W * 0.94 - sb, W * 0.94], [H * 0.93] * 2, color="white", lw=1.0, solid_capstyle="butt")     # lower right
    ax.text(W * 0.94 - sb / 2, H * 0.89, "0.5 mm", color="white", fontsize=4.5, ha="center", va="bottom")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CACHE_DIR / "eye_roi_inset.png", dpi=1200)
    save_figure(fig, OUT, "eye_roi_inset")


def main():
    cand = OUT / "eye_roi_candidates.csv"
    if CHOSEN and cand.exists():                                       # re-render the chosen example, no new search
        s = pd.read_csv(cand)
        r = s[s.session_id == CHOSEN].iloc[0].to_dict()
        r["frame"] = int(r["frame"])
        render(r)
        print("rendered", CHOSEN, "frame", r["frame"], "diameter", round(r["diameter_px"] * MM_PER_PX, 2), "mm")
        return
    sids = pd.read_parquet(TAB / "205_passive_state_dlc.parquet").session_id.dropna().unique()
    rows = []
    for sid in sorted(sids):
        try:
            r = score(sid)
        except Exception as err:  # noqa: BLE001
            print("skip", sid, type(err).__name__, str(err)[:80], flush=True)
            continue
        if r:
            rows.append(r)
            print(sid, round(r["quality"], 3), round(r["contrast"], 2), flush=True)
    s = pd.DataFrame(rows).sort_values("score", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "eye_roi_candidates.csv", index=False)
    best = s.iloc[0].to_dict()
    print("best:", best["session_id"], "frame", best["frame"], "quality", round(best["quality"], 3), "contrast",
          round(best["contrast"], 2), "diameter", round(best["diameter_px"] * MM_PER_PX, 2), "mm", flush=True)
    render(best)


if __name__ == "__main__":
    main()
