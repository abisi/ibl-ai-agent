"""007 -- Are keypoints well placed when their DLC likelihood is low? Pupil, whisker, nose (see the project question.md, 005).

For sessions where the 0.5 cutoff rejects many frames (MH062 0110: top nose 70%, pupil 41%, whisker 26%; MH038
0524: pupil 51%) and a control: video frames sampled in likelihood bins [0, 0.1), [0.1, 0.3), [0.3, 0.5),
[0.5, 1] of the gating likelihood (minimum over the group's keypoints), cropped around the group, with the
keypoints drawn (likelihood written next to each). Groups: pupil (side: pupil_top/right/bottom/left), whisker (top:
whisker_base, top_particle), top nose (top_nose_tip, top_nose_base), side nose (nose_tip, nose_base).
Also: fraction of frames per likelihood bin per group (table).
Outputs: figures/007_<group>_<session>.png, tables/007_keypoint_likelihood.csv
Run (haas): python 007_keypoint_likelihood_frames.py
"""
import glob
import importlib.util
import os
import pathlib

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

spec = importlib.util.spec_from_file_location('m005', pathlib.Path(__file__).parent / '005_jaw_likelihood_frames.py')
m005 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m005)

CASES = ['MH062_20260110_113351', 'MH038_20250524_105329', 'AB158_20250415_105500', 'MH009_20250206_104829']
GROUPS = {
    'pupil': ('side', ['pupil_top', 'pupil_right', 'pupil_bottom', 'pupil_left'], 60),
    'whisker': ('top', ['whisker_base', 'top_particle'], 110),
    'top_nose': ('top', ['top_nose_tip', 'top_nose_base'], 80),
    'side_nose': ('side', ['nose_tip', 'nose_base'], 80),
}
COLORS = ['r', 'c', 'y', 'lime']
BINS = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
PER_BIN = 4


def main():
    rows = []
    rng = np.random.default_rng(1)
    for s in CASES:
        with h5py.File(nwb(s), 'r') as f:
            b = f['processing/behavior/BehavioralTimeSeries']
            data = {k: b[k]['data'][:].astype(float) for k in b if k.endswith(('_x', '_y', '_likelihood'))}
        for gname, (view, kps, half) in GROUPS.items():
            if any(f'{k}_likelihood' not in data for k in kps):
                continue
            vids = [p for p in glob.glob(f'{m005.RAW}/{s[:5]}/Recording/{s}/Video/*.avi') if view in os.path.basename(p)]
            if not vids:
                continue
            n = min(len(data[f'{k}_x']) for k in kps)
            lik = np.min(np.vstack([data[f'{k}_likelihood'][:n] for k in kps]), 0)
            for lo, hi in BINS:
                rows.append(dict(session_id=s, group=gname, bin=f'[{lo},{min(hi, 1)})', frac=float(np.mean((lik >= lo) & (lik < hi)))))
            # video frame index: top video frames of the label-swapped / end-aligned sessions are not used here
            cx = np.median(np.hstack([data[f'{k}_x'][:n] for k in kps]))
            cy = np.median(np.hstack([data[f'{k}_y'][:n] for k in kps]))
            fig, axs = plt.subplots(len(BINS), PER_BIN, figsize=(3.0 * PER_BIN, 3.1 * len(BINS)), squeeze=False)
            for r, (lo, hi) in enumerate(BINS):
                idx = np.flatnonzero((lik >= lo) & (lik < hi))
                pick = np.sort(rng.choice(idx, min(PER_BIN, len(idx)), replace=False)) if len(idx) else []
                for c in range(PER_BIN):
                    ax = axs[r, c]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if c >= len(pick):
                        ax.axis('off')
                        continue
                    i = int(pick[c])
                    img = m005.read_frame(sorted(vids)[0], i)
                    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                    if img is not None:
                        ax.imshow(img[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap='gray',
                                  extent=[x0, x0 + 2 * half, y0 + 2 * half, y0])
                    for k, col in zip(kps, COLORS):
                        ax.plot(data[f'{k}_x'][i], data[f'{k}_y'][i], 'o', mfc='none', mec=col, ms=7, mew=1.3)
                        ax.text(data[f'{k}_x'][i] + 3, data[f'{k}_y'][i], f"{data[f'{k}_likelihood'][i]:.2f}", color=col, fontsize=6)
                    ax.set_xlim(x0, x0 + 2 * half)
                    ax.set_ylim(y0 + 2 * half, y0)
                    ax.set_title(f'frame {i}: min p={lik[i]:.2f}', fontsize=7)
                    if c == 0:
                        ax.set_ylabel(f'likelihood [{lo}, {min(hi, 1)})', fontsize=8)
            fig.suptitle(f'{s} {gname} ({view} video): ' + ', '.join(f'{k} ({c})' for k, c in zip(kps, COLORS)) +
                         f'; {100 * np.mean(lik < 0.5):.0f}% frames < 0.5', fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / 'figures' / f'007_{gname}_{s}.png', dpi=110)
            plt.close(fig)
            print(s, gname, 'done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '007_keypoint_likelihood.csv', index=False)
    print(t.pivot_table(index=['session_id', 'group'], columns='bin', values='frac').round(3).to_string())


if __name__ == '__main__':
    main()
