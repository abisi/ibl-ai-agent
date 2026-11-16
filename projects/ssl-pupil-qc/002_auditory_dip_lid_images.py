"""Direct evidence that the early pupil-area dip after stimuli is eyelid movement (user question, 2026-11-03).

Per pilot session (one column):
  row 1: eye-ROI mean intensity (robust z, lid closure positive; cached video pass) around trial start, per trial type
  row 2: mean eye image before the stimulus (-100 to 0 ms), auditory trials, with the mean pupil polygon
  row 3: mean image at the dip (+t_dip +- 15 ms, t_dip = auditory intensity peak) minus row 2, auditory trials
  row 4: same difference for no-stimulus trials (control, same time window)
  row 5: one auditory trial, frames from -50 to +250 ms (largest intensity response)
A lid movement appears in rows 3 / 5 as a band of change along the lid edge, absent in row 4; a pupil constriction
alone would only change a ring at the pupil border.
Output: combined_results_ks4/ssl-pupil-qc/figures/002_auditory_dip_lid_images.png (+ .pdf); mean images cached in
<session>/pupil_qc/video_cache/mean_<group>_<t0>_<t1>.npy, strip frames in frames.npz.
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 002_auditory_dip_lid_images.py
"""

import glob
import json

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pupil_qc as pq
import video

SESSIONS = ["AB131_20240905_123601", "AB164_20250422_115457", "MH022_20250311_140628", "AB116_20240725_094541",
            "AB102_20240309_114107"]
OUT = pq.RESULTS / "ssl-pupil-qc"
GROUPS = {"Auditory": ["A hit", "A miss"], "Whisker": ["W hit", "W miss"], "No stim.": ["FA", "CR"]}
COLORS = {"Auditory": "#2c2cdb", "Whisker": "#f7b519", "No stim.": "0.3"}
STRIP_MS = [-50, 0, 25, 50, 100, 150, 250]
MAX_TRIALS = 40


def load(sid):
    with h5py.File(pq.NWB_DIR / f"{sid}.nwb") as f:
        desc = f["session_description"][()].decode()
    d = pq.RESULTS / sid.split("_")[0] / desc.replace("+", "") / "pupil_qc"
    fr = pd.read_parquet(d / f"{sid}_frames.parquet")
    tq = pd.read_csv(d / f"{sid}_trials_qc.csv")
    meta = json.load(open(glob.glob(str(d / "*_metrics.json"))[0]))["meta"]
    off = meta["video"]["offset"]
    inten = np.load(d / "video_cache" / "eye_intensity.npy")[off:off + len(fr)].astype(float)
    t = fr.t.to_numpy()[:len(inten)]
    fs = round(1 / np.median(np.diff(t[:1000])))
    base = pd.Series(inten).rolling(int(5 * fs), center=True, min_periods=fs).median().to_numpy()
    z = (inten - base) / (np.nanmedian(np.abs(inten - base)) * 1.4826)
    # lid closure positive: orient with the blink frames
    bs, be = pq._runs(fr.state.to_numpy()[:len(z)] == pq.STATE["blink"])
    if len(bs) and np.nanmedian(z[((bs + be) // 2)]) < 0:
        z = -z
    return desc, d, fr, tq, meta, t, fs, z


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.2 * len(SESSIONS), 13))
    gs = fig.add_gridspec(5, len(SESSIONS), height_ratios=[1, 1, 1, 1, 0.55], hspace=0.35, wspace=0.25)
    for col, sid in enumerate(SESSIONS):
        desc, d, fr, tq, meta, t, fs, z = load(sid)
        cache = video.VideoCache(d, meta["video"]["video"])
        off = meta["video"]["offset"]
        nwb_df, _, _ = pq.load_session(sid)
        box = video.pupil_crop_box(nwb_df, fr.state.to_numpy() == pq.STATE["valid"])
        lags = np.arange(int(-0.3 * fs), int(0.6 * fs))
        ax = fig.add_subplot(gs[0, col])
        peak_ms = None
        for g, oc in GROUPS.items():
            idx = np.searchsorted(t, tq[tq.outcome.isin(oc)].start_time.to_numpy())
            idx = idx[(idx + lags[0] > 0) & (idx + lags[-1] < len(z))]
            m = np.nanmean(z[idx[:, None] + lags], 0)
            ax.plot(1000 * lags / fs, m - m[lags < 0].mean(), color=COLORS[g], label=f"{g} ({len(idx)})")
            if g == "Auditory":
                w = (lags >= 0) & (lags < 0.3 * fs)
                peak_ms = 1000 * lags[w][np.argmax(m[w])] / fs
        ax.axvline(0, color="k", lw=0.5)
        ax.axvline(peak_ms, color="#2c2cdb", lw=0.5, ls="--")
        ax.set_xlabel("Time from trial start (ms)")
        ax.set_ylabel("Eye-ROI intensity (Δz, lid closure +)")
        ax.set_title(f"{sid}\n{desc}", fontsize=8)
        ax.legend(fontsize=6, frameon=False)

        def mean_image(starts, t0_ms, t1_ms, tag):
            """Mean crop over frames t0..t1 ms around each start: one seek per trial, sequential read; cached."""
            f = cache.dir / f"mean_{tag}_{int(t0_ms)}_{int(t1_ms)}.npy"
            if f.exists():
                return np.load(f)
            cap = cv2.VideoCapture(meta["video"]["video"])
            acc, n = None, 0
            a0, a1 = int(t0_ms * fs / 1000), int(t1_ms * fs / 1000)
            for s_ in starts:
                i = np.searchsorted(t, s_) + off + a0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                for _ in range(a1 - a0 + 1):
                    ok, img = cap.read()
                    if not ok:
                        break
                    c_ = video.crop(img[:, :, 0], box).astype(np.float64)
                    acc = c_ if acc is None else acc + c_
                    n += 1
            cap.release()
            m = acc / max(n, 1)
            np.save(f, m)
            return m

        aud = tq[tq.outcome.isin(GROUPS["Auditory"])].start_time.to_numpy()[:MAX_TRIALS]
        nos = tq[tq.outcome.isin(GROUPS["No stim."])].start_time.to_numpy()[:MAX_TRIALS]
        win = (peak_ms - 15, peak_ms + 15)
        pre_a, dip_a = mean_image(aud, -100, 0, "aud"), mean_image(aud, *win, "aud")
        pre_n, dip_n = mean_image(nos, -100, 0, "nostim"), mean_image(nos, *win, "nostim")
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pre_a, cmap="gray")
        pts = nwb_df.iloc[np.searchsorted(t, aud)]
        ox, oy = box[0], box[1]
        px = [pts[f"{p}_x"].median() - ox for p in ("top", "right", "bottom", "left", "top")]
        py = [pts[f"{p}_y"].median() - oy for p in ("top", "right", "bottom", "left", "top")]
        ax.plot(px, py, color="#ff4081", lw=0.7)
        ax.set_title("Auditory trials, mean image -100 to 0 ms", fontsize=7)
        lim = max(np.nanpercentile(np.abs(dip_a - pre_a), 99.5), np.nanpercentile(np.abs(dip_n - pre_n), 99.5))
        for r, (dimg, pimg, name) in enumerate(((dip_a, pre_a, f"Auditory: {win[0]:.0f}-{win[1]:.0f} ms minus pre"),
                                                (dip_n, pre_n, f"No stim. (control): same window minus pre"))):
            ax = fig.add_subplot(gs[2 + r, col])
            im = ax.imshow(dimg - pimg, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.plot(px, py, color="k", lw=0.5, ls=":")
            ax.set_title(name, fontsize=7)
            if col == len(SESSIONS) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, label="Δ grey level")
        # single auditory trial with the largest intensity response
        ia = np.searchsorted(t, aud)
        resp = [np.nanmax(z[i:i + int(0.3 * fs)]) for i in ia]
        best = ia[int(np.nanargmax(resp))]
        sub = gs[4, col].subgridspec(1, len(STRIP_MS), wspace=0.03)
        imgs = cache.frames([best + int(ms * fs / 1000) + off for ms in STRIP_MS])
        for j, ms in enumerate(STRIP_MS):
            ax = fig.add_subplot(sub[0, j])
            img = imgs.get(best + int(ms * fs / 1000) + off)
            if img is not None:
                ax.imshow(video.crop(img, box), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{ms} ms", fontsize=5)
        print(sid, desc, f"auditory intensity peak at {peak_ms:.0f} ms; trials {len(aud)} aud / {len(nos)} no-stim")
        for a in fig.axes:
            if a.images and a.get_subplotspec() is not None and a.get_subplotspec().rowspan.start in (1, 2, 3):
                a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Early post-stimulus pupil-area dip = eyelid movement: eye-ROI intensity (row 1), mean images and "
                 "difference images at the dip (rows 2-4), example auditory trial (row 5)", fontsize=9, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"002_auditory_dip_lid_images.{ext}", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
