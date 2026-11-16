"""Squint threshold figure (user, 2026-11-04: squints = extreme cases only, from the eye-ROI brightness).

Per pilot session (003_squint_intensity_pilot.py): eye-ROI mean intensity (video_cache/eye_intensity.npy; a closing lid
replaces the dark eye by brighter lid / fur), smoothed by a 1 s running median, as a robust z relative to the session's
open-eye frames (QC-valid frames with pupil area >= the session median: z = (I - median_open) / (1.4826 MAD_open)).
Columns: (a) z vs pupil area (% of the QC-valid median) over QC-valid frames, (b) 10 min around the longest run with
z > 5: z and pupil area, (c-g) eye crops at z ~ 0, 3, 5, 8, 12 (frames inside >= 1 s runs at that level; ROI box in
yellow, DLC pupil points). Table: % of QC-valid frames that a rule "z > thr for >= 1 s" would turn to squint.
Outputs: combined_results_ks4/ssl-pupil-qc/figures/004_squint_threshold.png, tables/004_squint_threshold.csv
Run (haas): python 004_squint_threshold_figure.py
"""
import glob
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pupil_qc as pq  # noqa: E402

PILOT = ["MH037_20250524_143522", "AB144_20241205_111557", "AB126_20240822_114405", "AB140_20241129_104154",
         "MH021_20250309_122552"]
LEVELS = [0, 3, 5, 8, 12]
THRESHOLDS = [3, 5, 8, 12]
MIN_RUN_S = 1.0
OUT = pq.RESULTS / "ssl-pupil-qc"


def qc_dir(sid):
    return Path(glob.glob(str(pq.RESULTS / sid[:5] / "*" / "pupil_qc" / f"{sid}_frames.parquet"))[0]).parent


def runs_above(z, thr, n_min):
    s, e = pq._runs(z > thr)
    keep = (e - s) >= n_min
    m = np.zeros(len(z), bool)
    for a, b in zip(s[keep], e[keep]):
        m[a:b] = True
    return m


def frame(path, i):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, im = cap.read()
    return im[..., 0] if ok else None


def main():
    rows = []
    fig, axs = plt.subplots(len(PILOT), 2 + len(LEVELS), figsize=(3.1 * (2 + len(LEVELS)), 3.3 * len(PILOT)),
                            squeeze=False)
    for r, sid in enumerate(PILOT):
        d = qc_dir(sid)
        fr = pd.read_parquet(d / f"{sid}_frames.parquet")
        cache = d / "video_cache"
        inten = np.load(cache / "eye_intensity.npy")
        meta = json.loads((cache / "roi.json").read_text())
        roi, video = meta["roi"], meta["video"]
        fs = float(np.round(1 / np.median(np.diff(fr.t.to_numpy()[:2000]))))
        n = len(fr)
        x = np.full(n, np.nan, np.float32)
        x[:min(n, len(inten))] = inten[:n]  # video offset 0 for the pilot sessions (003 log)
        xs = pd.Series(x).rolling(int(fs), center=True, min_periods=int(fs) // 2).median().to_numpy()
        valid = fr.state.to_numpy() == pq.STATE["valid"]
        area = fr.area_px2.to_numpy().astype(float)
        area_pct = 100 * area / np.nanmedian(area[valid])
        open_ = valid & (area_pct >= 100)
        ref = np.nanmedian(xs[open_])
        mad = 1.4826 * np.nanmedian(np.abs(xs[open_] - ref))
        z = (xs - ref) / mad
        n_min = int(MIN_RUN_S * fs)
        row = dict(session_id=sid, ref_intensity=ref, mad=mad, pct_valid=100 * valid.mean())
        for thr in THRESHOLDS:
            sq = runs_above(z, thr, n_min)
            row[f"pct_valid_frames_z>{thr}"] = 100 * (sq & valid).sum() / valid.sum()
            row[f"median_area_pct_z>{thr}"] = float(np.nanmedian(area_pct[sq & valid])) if (sq & valid).any() else np.nan
        rows.append(row)

        ax = axs[r, 0]
        ok = valid & np.isfinite(z)
        ax.hist2d(np.clip(area_pct[ok], 0, 250), np.clip(z[ok], -10, 40), bins=[100, 100], cmap="magma", cmin=1,
                  norm=matplotlib.colors.LogNorm())
        for thr in THRESHOLDS:
            ax.axhline(thr, color="c", lw=0.6, ls="--")
        ax.set_xlabel("pupil area (% of median, QC-valid)", fontsize=7)
        ax.set_ylabel("eye-ROI brightness (z vs open eye)", fontsize=7)
        ax.set_title(f"{sid}\nQC valid {row['pct_valid']:.1f}%", fontsize=7)
        ax.tick_params(labelsize=6)

        ax = axs[r, 1]
        sq5 = runs_above(z, 5, n_min)
        s, e = pq._runs(sq5)
        c = (s[np.argmax(e - s)] + e[np.argmax(e - s)]) // 2 if len(s) else n // 2
        a, b = max(0, c - int(300 * fs)), min(n, c + int(300 * fs))
        tt = (np.arange(a, b) - c) / fs
        step = max(1, (b - a) // 6000)
        ax.plot(tt[::step], z[a:b:step], color="k", lw=0.5)
        ax.set_ylabel("z", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(tt[::step], np.where(valid[a:b], area_pct[a:b], np.nan)[::step], color="tab:orange", lw=0.4)
        ax2.set_ylabel("pupil % (orange, QC-valid)", fontsize=6)
        for thr in THRESHOLDS:
            ax.axhline(thr, color="c", lw=0.5, ls="--")
        ax.set_xlabel(f"time from longest z>5 run (s)", fontsize=7)
        ax.set_title("10 min: brightness z (black), pupil (orange)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

        for k, lev in enumerate(LEVELS):
            ax = axs[r, 2 + k]
            ax.axis("off")
            # frame inside a >= 1 s run where z stays within lev +- 1 (lev 0: open eye, QC valid)
            m = runs_above(-np.abs(z - lev), -1.0, n_min) & np.isfinite(z)
            idx = np.flatnonzero(m)
            if not len(idx):
                ax.set_title(f"z~{lev}: none", fontsize=7)
                continue
            i = int(idx[len(idx) // 2])
            im = frame(video, i)
            if im is None:
                continue
            x0, y0, w, h = roi
            pad = w // 2
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="y", lw=0.8))
            with h5py.File(pq.NWB_DIR / f"{sid}.nwb", "r") as f:
                g = f["processing/behavior/BehavioralTimeSeries"]
                for p, col in (("top", "r"), ("right", "y"), ("bottom", "c"), ("left", "lime")):
                    ax.plot(g[f"pupil_{p}_x/data"][i], g[f"pupil_{p}_y/data"][i], "o", mfc="none", mec=col, ms=5,
                            mew=1)
            ax.set_xlim(max(0, x0 - pad), x0 + w + pad)
            ax.set_ylim(min(480, y0 + h + pad), max(0, y0 - pad))
            ax.set_title(f"z~{lev}: z={z[i]:.1f}, pupil {area_pct[i]:.0f}%\nQC {pq.STATE_NAMES[int(fr.state[i])]}, "
                         f"t={fr.t[i]:.0f} s", fontsize=7)
    fig.suptitle("Squint threshold pilot: eye-ROI brightness (1 s median) as z vs the session's open-eye frames; "
                 f"dashed = candidate thresholds {THRESHOLDS}; frames from >= {MIN_RUN_S:.0f} s runs at each level",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "004_squint_threshold.png", dpi=95)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tables" / "004_squint_threshold.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
