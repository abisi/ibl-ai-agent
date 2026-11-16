"""004 -- Which rig (camera scale) was each DLC session recorded on? (see the project question.md)

The yaml behaviour 'setup' is set per experimenter in make_yaml_config (AB -> setup 1, MH -> setup 2), not per
session, and session_config.json has no rig field. The two rigs differ in camera scale (reference_setup1 / 2:
side 33.4 / 28.8 px/mm, top 30.4 / 16.75 px/mm), so the pixel size of body features that do not change between
sessions identifies the rig. Per NWB with DLC (raw pixel values, likelihood > 0.5): median distance between
top_nose_tip and top_nose_base (top), whisker_base and top_nose_tip (top), nose_tip and nose_base (side), pupil
equivalent diameter (side); the scale stored in the NWB; the rig of the ephys probe sheet (joint_probe_insertion_info)
for ephys sessions; the video frame size (ffprobe).
Outputs: figures/004_rig_scale.png, tables/004_rig_scale.csv
Run (haas): python 004_rig_scale.py [--plot-only]  (--plot-only: figure from the existing table)
"""
import glob
import os
import subprocess

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
from common import NWB_DIR, OUT  # noqa: E402

PROBE_SHEET = '/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_probe_insertion_info.xlsx'
RAW = '/mnt/lsens-data'


def dist(b, p, q, cut=0.5):
    if any(f'{k}_{c}' not in b for k in (p, q) for c in ('x', 'y', 'likelihood')):
        return np.nan
    g = lambda k: b[k]['data'][:].astype(float)  # noqa: E731
    ok = (g(f'{p}_likelihood') > cut) & (g(f'{q}_likelihood') > cut)
    if ok.sum() < 1000:
        return np.nan
    d = np.hypot(g(f'{p}_x') - g(f'{q}_x'), g(f'{p}_y') - g(f'{q}_y'))
    return float(np.median(d[ok]))


def frame_size(session_id, view):
    v = [p for p in glob.glob(f'{RAW}/{session_id[:5]}/Recording/{session_id}/Video/*.avi') if view in os.path.basename(p)]
    if not v:
        return ''
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0', sorted(v)[0]], capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def probe_rigs():
    for p in [PROBE_SHEET]:
        if os.path.exists(p):
            df = pd.read_excel(p)
            col = 'Setup' if 'Setup' in df else 'setup' if 'setup' in df else None
            if col is None:
                return {}
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            return {(r.mouse_name, r.date.strftime('%Y%m%d')): str(r[col]) for _, r in df.iterrows() if pd.notna(r.date)}
    return {}


def plot(t):
    """Body-feature sizes per session. Marked: probe sheet 'setup 2' (red circle; the earlier version matched any '2',
    e.g. 'setup 1 AI3209', and circled nearly all ephys sessions) and top view nose up in 008 (setup 2 camera upside
    down; black x), when tables/008_top_orientation.csv exists."""
    rig = t.probe_sheet_rig.fillna('').astype(str)
    o = OUT / 'tables' / '008_top_orientation.csv'
    up = set(pd.read_csv(o).query("dlc_orientation == 'nose up'").session_id) if o.exists() else set()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (xc, yc) in zip(axs, [('top_nose_len_px', 'top_whiskerbase_nose_px'), ('side_nose_len_px', 'pupil_diam_px'),
                                  ('top_nose_len_px', 'side_nose_len_px')]):
        for m, d in t.groupby(t.mouse.str[:2]):
            ax.scatter(d[xc], d[yc], s=12, label=f'{m} mice', alpha=0.7)
        e = t[rig.str.startswith('setup 2')]
        ax.scatter(e[xc], e[yc], s=40, facecolors='none', edgecolors='r', label=f'probe sheet: setup 2 (n={len(e)})')
        u = t[t.session_id.isin(up)]
        ax.scatter(u[xc], u[yc], s=30, marker='x', c='k', label=f'top view nose up, 008 (n={len(u)})')
        ax.set_xlabel(xc, fontsize=8)
        ax.set_ylabel(yc, fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle('Body-feature sizes in pixels per session (two rigs = two camera scales)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '004_rig_scale.png', dpi=130)
    plt.close(fig)


def main():
    if '--plot-only' in sys.argv:
        plot(pd.read_csv(OUT / 'tables' / '004_rig_scale.csv'))
        return
    rigs = probe_rigs()
    rows = []
    for p in sorted(glob.glob(str(NWB_DIR / '*.nwb'))):
        s = os.path.basename(p)[:-4]
        if '_backup' in s:
            continue
        try:
            with h5py.File(p, 'r') as f:
                b = f.get('processing/behavior/BehavioralTimeSeries')
                if b is None or 'jaw_angle' not in b:
                    continue
                row = dict(session_id=s, mouse=s[:5],
                           nwb_side_px_per_mm=round(1 / b['jaw_distance']['data'].attrs['conversion'], 2),
                           nwb_top_px_per_mm=round(1 / b['top_nose_distance']['data'].attrs['conversion'], 2) if 'top_nose_distance' in b else np.nan,
                           top_nose_len_px=dist(b, 'top_nose_tip', 'top_nose_base'),
                           top_whiskerbase_nose_px=dist(b, 'whisker_base', 'top_nose_tip'),
                           side_nose_len_px=dist(b, 'nose_tip', 'nose_base'),
                           probe_sheet_rig=rigs.get((s[:5], s[6:14]), ''))
                if 'pupil_area' in b and 'pupil_likelihood' in b:
                    a = b['pupil_area']['data'][:]
                    ok = b['pupil_likelihood']['data'][:] > 0.5
                    row['pupil_diam_px'] = float(np.median(np.sqrt(4 * a[ok] / np.pi))) if ok.sum() > 1000 else np.nan
        except Exception as e:
            print(s, 'ERROR', e)
            continue
        row['top_frame'] = frame_size(s, 'top')
        row['side_frame'] = frame_size(s, 'side')
        rows.append(row)
        print(s, row['top_nose_len_px'], row['top_frame'], row['probe_sheet_rig'], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '004_rig_scale.csv', index=False)
    plot(t)
    print(t.groupby(['top_frame', 'side_frame']).size())


if __name__ == '__main__':
    main()
