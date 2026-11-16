"""008 -- Top-camera orientation per session: one top-video frame per session for review (see the project question.md).

Setup 2 top camera is mounted rotated 180 deg (nose up, spout from the top) relative to setup 1 (nose down). DLC
orientation per session from tables/top_orientation.csv (median top_nose_tip_y - top_nose_base_y over frames with
both likelihoods > 0.5: < -20 px nose up, > 20 nose down, else unclear), rig from the probe sheet (ephys days only).
Frames: all MH sessions with DLC + 6 AB references, one frame at 600 s (or mid-video), 5 x 5 per page.
Outputs: figures/008_top_orientation_p<k>.png, tables/008_top_orientation.csv
Run (haas): python 008_top_orientation_frames.py
"""
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

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
SHEET = '/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_probe_insertion_info.xlsx'
AB_REF = ['AB151_20250130_131124', 'AB127_20240823_094915', 'AB085_20231005_152636', 'AB153_20250209_153912',
          'AB164_20250424_132236', 'AB116_20240726_094206']


def grab(video):
    for t in (600, 120, 30):
        r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', video, '-frames:v', '1', '-f', 'image2pipe',
                            '-vcodec', 'png', '-'], capture_output=True, timeout=180)
        if r.stdout:
            return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None


def frame(session_id):
    """Frame of the top video; sessions whose top video is named 'side' (two 'side' files): both files side by side,
    in file-name order (left | right)."""
    allv = sorted(p for p in glob.glob(f'{RAW}/{session_id[:5]}/Recording/{session_id}/Video/*')
                  if p.lower().endswith(('.avi', '.mp4')))
    v = [p for p in allv if 'top' in os.path.basename(p)]
    if v:
        return grab(v[0]), os.path.basename(v[0])
    side = [p for p in allv if 'side' in os.path.basename(p)]
    if len(side) == 2:
        imgs = [grab(p) for p in side]
        if all(i is not None for i in imgs):
            return np.hstack(imgs), ' | '.join(os.path.basename(p) for p in side) + ' (two side files)'
    return None, ''


def main():
    t = pd.read_csv(OUT / 'tables' / 'top_orientation.csv')
    t['dlc_orientation'] = np.where(t.tip_minus_base_y < -20, 'nose up',
                                    np.where(t.tip_minus_base_y > 20, 'nose down', 'unclear'))
    t.loc[t.tip_minus_base_y.isna(), 'dlc_orientation'] = 'unclear'
    sh = pd.read_excel(SHEET, usecols=['mouse_name', 'date', 'setup'])
    sh['date'] = pd.to_datetime(sh.date, errors='coerce', dayfirst=True).dt.strftime('%Y%m%d')
    rig = {(r.mouse_name, r.date): r.setup for r in sh.drop_duplicates(['mouse_name', 'date']).itertuples()}
    t['probe_sheet_rig'] = [rig.get((s[:5], s[6:14]), '') for s in t.session_id]
    sel = t[t.session_id.str.startswith('MH') | t.session_id.isin(AB_REF)].sort_values('session_id').reset_index(drop=True)
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(frame, sel.session_id))
    sel['video'] = [r[1] for r in res]
    sel.to_csv(OUT / 'tables' / '008_top_orientation.csv', index=False)
    per = 25
    for k in range(int(np.ceil(len(sel) / per))):
        fig, axs = plt.subplots(5, 5, figsize=(17, 14))
        for ax, i in zip(axs.flat, range(k * per, (k + 1) * per)):
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= len(sel):
                ax.axis('off')
                continue
            img = res[i][0]
            if img is not None:
                ax.imshow(img, cmap='gray')
            r = sel.iloc[i]
            col = {'nose up': 'tab:red', 'nose down': 'k', 'unclear': 'tab:orange'}[r.dlc_orientation]
            two = ' (2 side files: L | R)' if 'two side' in r.video else ''
            ax.set_title(f'{r.session_id}{two}\nDLC: {r.dlc_orientation} | sheet: {str(r.probe_sheet_rig).replace(" AI3209", "") or "-"}',
                         fontsize=8, color=col)
        fig.suptitle(f'Top-camera frame per session (page {k + 1}); red = DLC nose up, orange = unclear', fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'008_top_orientation_p{k + 1}.png', dpi=80)
        plt.close(fig)
    print(sel[['session_id', 'dlc_orientation', 'probe_sheet_rig', 'video']].to_string(index=False))


if __name__ == '__main__':
    main()
