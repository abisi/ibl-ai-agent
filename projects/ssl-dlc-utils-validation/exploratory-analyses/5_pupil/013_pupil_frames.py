"""013 -- What the pupil tracking looks like in the 012 example sessions (largest pupil spike index + one typical
session): side-video frames of the eye with the four DLC pupil points (NWB keypoints, pixels) and the pupil polygon,
at a typical frame and at the largest spikes of the stored pupil_area, with the trace around each frame (stored,
running median 0.25 s, median 0.25 s + Gaussian 0.1 s).
Raw video: /mnt/lsens-data/<mouse>/{Recording,Training}/<session>/Video/<mouse>_side*.avi; video frame i = NWB DLC frame
i (none of these sessions has an alignment clause).
Output: figures/013_pupil_frames_<session>.png
Run (haas): python 013_pupil_frames.py [session ...]
"""
import glob
import sys

import cv2
import matplotlib
import numpy as np
from scipy.ndimage import median_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import OUT, dh, nwb  # noqa: E402

EXAMPLES = ['MH062_20260109_114135', 'AB162_20250421_140550', 'MH020_20250304_161732', 'MH038_20250524_105329',
            'MH037_20250524_143522']
POINTS = {'pupil_top': 'r', 'pupil_right': 'y', 'pupil_bottom': 'c', 'pupil_left': 'lime'}
CROP = 90  # half-width of the eye crop (px)
N_SPIKES = 4


def video_path(s):
    m = s[:5]
    hits = [p for p in glob.glob(f'/mnt/lsens-data/{m}/*/{s}/Video/{m}_side*.avi') if '_backup' not in p]
    return hits[0] if hits else None


def frame(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, im = cap.read()
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None


def one(s):
    v = dh.load_dlc(nwb(s), views=('side',), physical_units=False)['side']  # keypoints in px for the overlay
    vm = dh.load_dlc(nwb(s), views=('side',))['side']  # pupil_area in mm^2
    ts = vm.timestamps
    traces = {'stored': dh.clean_series(vm, 'pupil_area', median_s=None, smooth_sigma_s=None)[0],
              'median 0.25 s': dh.clean_series(vm, 'pupil_area', median_s=0.25, smooth_sigma_s=None)[0],
              'median 0.25 s + gauss 0.1 s': dh.clean_series(vm, 'pupil_area', median_s=0.25, smooth_sigma_s=0.1)[0]}
    raw = traces['stored']
    lik = np.min(np.vstack([v.likelihood[f'{p}_likelihood'].to_numpy() for p in POINTS]), 0)
    k = max(3, int(round(0.05 / np.median(np.diff(ts)))) | 1)
    dev = np.abs(raw - median_filter(np.nan_to_num(raw, nan=np.nanmedian(raw)), size=k, mode='nearest'))
    # typical frame: area near the session median, all four points confident, middle half of the session
    med = np.nanmedian(raw)
    n = len(raw)
    cand = np.flatnonzero((np.abs(raw - med) < 0.05 * med) & (lik > 0.9) & (np.arange(n) > n // 4)
                          & (np.arange(n) < 3 * n // 4))
    picks = [('typical', int(cand[len(cand) // 2]) if len(cand) else n // 2)]
    order = np.argsort(np.nan_to_num(dev, nan=-1))[::-1]
    for i in order:  # largest spikes, at least 10 s apart
        if all(abs(ts[i] - ts[j]) > 10 for _, j in picks):
            picks.append((f'spike {len(picks)}', int(i)))
        if len(picks) == N_SPIKES + 1:
            break
    # eye crop centre: session median of the four points
    cx = np.nanmedian(np.concatenate([v.data[f'{p}_x'].to_numpy() for p in POINTS]))
    cy = np.nanmedian(np.concatenate([v.data[f'{p}_y'].to_numpy() for p in POINTS]))
    vid = video_path(s)
    cap = cv2.VideoCapture(vid) if vid else None
    fig, axs = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.6), squeeze=False)
    for c, (label, i) in enumerate(picks):
        ax = axs[0, c]
        im = frame(cap, i) if cap is not None else None
        if im is not None:
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        xs = [v.data[f'{p}_x'].iloc[i] for p in POINTS]
        ys = [v.data[f'{p}_y'].iloc[i] for p in POINTS]
        ax.fill(xs, ys, fill=False, ec='w', lw=0.6, ls='--')
        for (p, col), x, y in zip(POINTS.items(), xs, ys):
            ax.plot(x, y, 'o', mfc='none', mec=col, ms=6, mew=1.2)
        ax.set_xlim(cx - CROP, cx + CROP)
        ax.set_ylim(cy + CROP, cy - CROP)
        ax.set_xticks([])
        ax.set_yticks([])
        liks = ', '.join(f'{v.likelihood[f"{p}_likelihood"].iloc[i]:.2f}' for p in POINTS)
        ax.set_title(f'{label}: frame {i} (t={ts[i]:.1f} s)\narea {raw[i]:.2f} -> median {traces["median 0.25 s"][i]:.2f}'
                     f' mm$^2$\nlik t/r/b/l {liks}', fontsize=6.5)
        ax = axs[1, c]
        a = (ts > ts[i] - 2) & (ts < ts[i] + 2)
        for (name, x), col, lw in zip(traces.items(), ('0.6', 'tab:orange', 'k'), (0.6, 1.2, 1.0)):
            ax.plot(ts[a] - ts[i], x[a], color=col, lw=lw, label=name)
        ax.axvline(0, color='r', lw=0.6, ls=':')
        ax.set_xlabel('time from frame (s)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect(1)
        if c == 0:
            ax.set_ylabel('pupil area (mm$^2$)', fontsize=7)
            ax.legend(fontsize=5.5)
    fig.suptitle(f'{s}: side video, DLC pupil points (top red, right yellow, bottom cyan, left green); '
                 f'frames with lik < 0.1 on a point: {np.mean(lik < 0.1) * 100:.1f}%', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'013_pupil_frames_{s}.png', dpi=120)
    plt.close(fig)
    print(s, 'video' if cap is not None else 'NO VIDEO', picks, flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:] or EXAMPLES:
        one(s)
