"""019 -- How bad is the top DLC of the 5 upside-down setup-2 sessions whose top tracking failed (MH009 0207 / 0208,
MH025 0421 / 0422, MH027 0422; shuffle5 best-540 on the raw upside-down video)? Reference: MH009_20250206 (upside-down
video, shuffle6 best-720, well tracked).

Per session: 4 top-video frames (300, 1200, 2400, 3600 s; DLC row = video frame) with the top keypoints drawn (filled
dot likelihood >= 0.6, open circle below), and the fraction of frames with likelihood >= 0.6 per top keypoint.
Outputs: figures/019_failed_top_dlc_frames.png, tables/019_failed_top_dlc_likelihood.csv
Run (haas): python 019_failed_top_dlc_frames.py
"""
import glob

import cv2
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
from common import OUT  # noqa: E402

RAW = '/mnt/lsens-data'
DLC = '/mnt/lsens-analysis/Myriam_Hamon/data'
SESSIONS = ['MH009_20250206_104829', 'MH009_20250207_131953', 'MH009_20250208_192403', 'MH025_20250421_115741',
            'MH025_20250422_141134', 'MH027_20250422_111013']
TIMES = [300, 1200, 2400, 3600]
POINTS = {'nose_tip': 'tab:red', 'nose_base': 'tab:orange', 'whisker_base': 'tab:blue', 'whisker_tip': 'tab:cyan',
          'spout': 'tab:green'}
THR = 0.6


def load(session_id):
    mouse = session_id[:5]
    h5 = [f for f in glob.glob(f'{DLC}/{mouse}/{session_id}/Video/*top*filtered.h5') if 'backup' not in f][0]
    d = pd.read_hdf(h5)
    d.columns = ['_'.join(c[1:]) for c in d.columns]
    video = glob.glob(f'{RAW}/{mouse}/Recording/{session_id}/Video/*top*.avi')[0]
    return d, video, h5.split('DLC_')[1].split('_filtered')[0]


def main():
    fig, axes = plt.subplots(len(SESSIONS), len(TIMES) + 1, figsize=(3.2 * (len(TIMES) + 1), 2.6 * len(SESSIONS)),
                             gridspec_kw={'width_ratios': [1] * len(TIMES) + [1.1]})
    rows = []
    for i, s in enumerate(SESSIONS):
        d, video, model = load(s)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        for j, t in enumerate(TIMES):
            ax = axes[i, j]
            k = min(int(t * fps), len(d) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if ok:
                ax.imshow(img[..., 0], cmap='gray')
            for p, c in POINTS.items():
                x, y, lk = d.at[k, f'{p}_x'], d.at[k, f'{p}_y'], d.at[k, f'{p}_likelihood']
                ax.scatter(x, y, s=40, facecolors=c if lk >= THR else 'none', edgecolors=c, linewidths=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{t} s', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{s[:14]}\n{model.replace("snapshot_", "")}', fontsize=8)
        cap.release()
        frac = {p: float((d[f'{p}_likelihood'] >= THR).mean()) for p in POINTS}
        rows.append({'session_id': s, 'model': model, 'n_frames': len(d), **{f'frac_{p}': v for p, v in frac.items()},
                     **{f'median_lik_{p}': float(d[f'{p}_likelihood'].median()) for p in POINTS}})
        ax = axes[i, -1]
        ax.barh(list(POINTS), list(frac.values()), color=list(POINTS.values()))
        ax.set_xlim(0, 1); ax.invert_yaxis(); ax.tick_params(labelsize=7)
        ax.set_title(f'frames with likelihood >= {THR}', fontsize=8)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=p) for p, c in POINTS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=len(POINTS), fontsize=8, frameon=False)
    fig.text(0.5, 0.005, 'Raw (upside-down) top video. Filled: likelihood >= 0.6; open: below. Row 1 = reference '
             '(well tracked, shuffle6).', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(OUT / 'figures' / '019_failed_top_dlc_frames.png', dpi=110)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '019_failed_top_dlc_likelihood.csv', index=False)
    print(t.round(2).to_string())


if __name__ == '__main__':
    main()
