"""005 -- Is the jaw visible and well placed when its DLC likelihood is low? (see the project question.md)

Sessions with most jaw frames below likelihood 0.5 (scan: AB157, AB158, AB159, AB162, AB164, MH062) and a control.
Per session: side-video frames sampled in jaw_likelihood bins [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 1], cropped
around the jaw, with the DLC jaw (red), jaw_ref (cyan), nose_tip (yellow) and tongue (green, if likelihood > 0.5)
drawn on; and a 10 s trace of jaw_y and jaw_likelihood around licking.
Frames: NWB row index = side DLC row = video frame (no end-alignment clause in these sessions).
Outputs: figures/005_jaw_frames_<session>.png, tables/005_jaw_likelihood.csv
Run (haas): python 005_jaw_likelihood_frames.py
"""
import glob
import os
import subprocess

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import OUT, nwb  # noqa: E402

CASES = ['AB158_20250415_105500', 'AB157_20250412_154744', 'AB162_20250422_140344', 'MH062_20260110_113351',
         'AB164_20250422_115457', 'MH009_20250206_104829']
BINS = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
PER_BIN = 4
RAW = '/mnt/lsens-data'


def read_frame(video, idx, fps=200.0):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{idx / fps:.4f}', '-i', video, '-frames:v', '1', '-f', 'image2pipe',
                        '-vcodec', 'png', '-'], capture_output=True, timeout=120)
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    rows = []
    rng = np.random.default_rng(0)
    for s in CASES:
        vids = [p for p in glob.glob(f'{RAW}/{s[:5]}/Recording/{s}/Video/*.avi') if 'side' in os.path.basename(p)]
        if not vids:
            print(s, 'no side video')
            continue
        with h5py.File(nwb(s), 'r') as f:
            b = f['processing/behavior/BehavioralTimeSeries']
            g = lambda k: b[k]['data'][:].astype(float)  # noqa: E731
            jx, jy, jl = g('jaw_x'), g('jaw_y'), g('jaw_likelihood')
            rx, ry, rl = g('jaw_ref_x'), g('jaw_ref_y'), g('jaw_ref_likelihood')
            nx, ny = g('nose_tip_x'), g('nose_tip_y')
            tx, ty, tl = g('tongue_x'), g('tongue_y'), g('tongue_likelihood')
            ts = b['jaw_angle']['timestamps'][:len(jx)]
            licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        rows.append(dict(session_id=s, frac_below_0p5=float(np.mean(jl < 0.5)), median_likelihood=float(np.median(jl)),
                         jaw_ref_frac_below_0p5=float(np.mean(rl < 0.5)),
                         jaw_y_sd_high=float(np.std(jy[jl >= 0.5])) if (jl >= 0.5).any() else np.nan,
                         jaw_y_sd_low=float(np.std(jy[jl < 0.1])) if (jl < 0.1).any() else np.nan))
        cx, cy = np.median(jx), np.median(jy)
        fig = plt.figure(figsize=(3.0 * PER_BIN, 3.0 * len(BINS) + 3))
        gs = fig.add_gridspec(len(BINS) + 1, PER_BIN, height_ratios=[1] * len(BINS) + [1.1])
        for r, (lo, hi) in enumerate(BINS):
            idx = np.flatnonzero((jl >= lo) & (jl < hi))
            pick = np.sort(rng.choice(idx, min(PER_BIN, len(idx)), replace=False)) if len(idx) else []
            for c, i in enumerate(pick):
                ax = fig.add_subplot(gs[r, c])
                img = read_frame(sorted(vids)[0], int(i))
                if img is not None:
                    h = 90
                    x0, y0 = int(max(0, cx - h)), int(max(0, cy - h))
                    ax.imshow(img[y0:y0 + 2 * h, x0:x0 + 2 * h], cmap='gray', extent=[x0, x0 + 2 * h, y0 + 2 * h, y0])
                ax.plot(jx[i], jy[i], 'o', mfc='none', mec='r', ms=8, mew=1.5)
                ax.plot(rx[i], ry[i], 's', mfc='none', mec='c', ms=6)
                ax.plot(nx[i], ny[i], '^', mfc='none', mec='y', ms=6)
                if tl[i] > 0.5:
                    ax.plot(tx[i], ty[i], 'o', mfc='none', mec='g', ms=6)
                ax.set_title(f'frame {i}: jaw p={jl[i]:.2f}', fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            if len(pick):
                fig.axes[-len(pick)].set_ylabel(f'likelihood [{lo}, {min(hi, 1)})', fontsize=8)
        ax = fig.add_subplot(gs[len(BINS), :])
        t0 = licks[len(licks) // 2] - 3 if len(licks) else ts[len(ts) // 2]
        w = (ts >= t0) & (ts < t0 + 10)
        ax.plot(ts[w] - t0, jy[w], color='k', lw=0.7, label='jaw_y (px; up = smaller)')
        ax.scatter(ts[w][jl[w] < 0.5] - t0, jy[w][jl[w] < 0.5], s=3, color='r', label='frames with likelihood < 0.5')
        for lk in licks[(licks >= t0) & (licks < t0 + 10)]:
            ax.axvline(lk - t0, color='b', lw=0.4)
        ax.invert_yaxis()
        ax2 = ax.twinx()
        ax2.plot(ts[w] - t0, jl[w], color='tab:orange', lw=0.6)
        ax2.set_ylabel('jaw likelihood', fontsize=8, color='tab:orange')
        ax.set_xlabel('time (s); blue = piezo licks', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        fig.suptitle(f'{s}: jaw (red o), jaw_ref (cyan), nose tip (yellow), tongue (green); '
                     f'{100 * np.mean(jl < 0.5):.0f}% jaw frames < 0.5', fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'005_jaw_frames_{s}.png', dpi=110)
        plt.close(fig)
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '005_jaw_likelihood.csv', index=False)
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
