"""018 -- Per-session audit of videos and DLC (user, 2026-11-04: "what's missing, what's wrong").

Sessions: NWB_ks4 (no backups) of AB / MH mice with camera_flag = 1 in the experiment description.
Per session and camera (side, top):
- raw video: /mnt/lsens-data/<mouse>/{Recording,Training}/<session>/Video/<mouse>_<cam>*.avi|mp4 (Axel_Bisi/data for
  AB mp4 copies): number of files, frame count (container header);
- DLC: <data>/<mouse>/<session>/Video/*<cam>*filtered.h5 (Axel_Bisi/data for AB, Myriam_Hamon/data for MH; no
  *_backup_*): number of files, rows;
- NWB: DLC series present, frames (data rows) and camera timestamps (pulses) of that camera.
Flags: no_video_folder (session folder without Video/), no_video (Video/ without this camera), multi_video (> 1 file named for the camera, excluding -converted / short copies), no_dlc,
dlc_incomplete (rows < 99% of video frames), dlc_vs_video (|rows - frames| > 10 otherwise), nwb_no_dlc (DLC file but
no NWB series), nwb_frames_vs_pulses (|frames - pulses| > 1 s of frames), wrong_mouse_in_name (video file of another
mouse). Joined with the alignment verdicts of ssl-dlc-alignment-debug (session_verdicts.csv) when present.
Outputs: tables/018_video_dlc_audit.csv (one row per session x camera), tables/018_video_dlc_issues.csv (flagged rows)
Run (haas): python 018_video_dlc_audit.py
"""
import ast
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5py
import numpy as np
import pandas as pd

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import NWB_DIR, OUT, dh  # noqa: E402

RAW = '/mnt/lsens-data'
DATA = {'AB': '/mnt/lsens-analysis/Axel_Bisi/data', 'MH': '/mnt/lsens-analysis/Myriam_Hamon/data'}
VERDICTS = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-alignment-debug/tables/session_verdicts.csv'


def experiment(f):
    try:
        return ast.literal_eval(f['general/experiment_description'][()].decode())
    except Exception:
        return {}


def video_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    cap.release()
    return n


def h5_rows(path):
    try:
        with h5py.File(path, 'r') as f:
            for k in ('df_with_missing/table', 'df_with_missing/_i_table/index'):
                if k in f:
                    return int(f[k].shape[0]) if k.endswith('table') else -1
            for k in f:
                g = f[k]
                if isinstance(g, h5py.Group) and 'table' in g:
                    return int(g['table'].shape[0])
    except Exception:
        return -1
    return -1


def one(path):
    s = os.path.basename(path)[:-4]
    mouse = s[:5]
    rows = []
    try:
        with h5py.File(path, 'r') as f:
            exp = experiment(f)
            if int(exp.get('camera_flag', 0)) != 1:
                return rows
            desc = f['session_description'][()].decode()
            grp = f[dh.TS_PATH] if dh.TS_PATH in f else None
            views = dh._camera_groups(grp) if grp is not None else {}
            nwb = {}
            for v, names in views.items():
                nwb[v] = (min(grp[n]['data'].shape[0] for n in names), grp[names[0]]['timestamps'].shape[0])
        vdirs = glob.glob(f'{RAW}/{mouse}/*/{s}/Video')
        vfiles = sorted({os.path.basename(p): p for d in vdirs for p in glob.glob(f'{d}/*')
                         if p.lower().endswith(('.avi', '.mp4'))}.values())
        ddir = f'{DATA[mouse[:2]]}/{mouse}/{s}/Video'
        h5s = [p for p in glob.glob(f'{ddir}/*filtered.h5') if '_backup' not in p]
        fps = float(exp.get('camera_freq', 200) or 200)
        for cam in ('side', 'top'):
            named = [p for p in vfiles if re.search(rf'_{cam}[ _-]', os.path.basename(p))]
            # the original .avi; old AB sessions only keep the -converted.mp4 copy (then that copy is the video)
            vids = [p for p in named if not re.search(r'converted|short', os.path.basename(p))] or \
                   [p for p in named if 'converted' in os.path.basename(p)][:1]
            other = [p for p in vfiles if not os.path.basename(p).startswith(mouse)]
            no_video_folder = not vdirs
            dl = [p for p in h5s if f'_{cam} ' in os.path.basename(p) or f'_{cam}_' in os.path.basename(p)]
            nfr = video_frames(vids[0]) if len(vids) == 1 else (max(video_frames(p) for p in vids) if vids else -1)
            nrow = h5_rows(dl[0]) if len(dl) == 1 else -1
            nwb_frames, nwb_pulses = nwb.get(cam, (np.nan, np.nan))
            flags = []
            if not vids:
                flags.append('no_video_folder' if no_video_folder else 'no_video')
            if len(vids) > 1:
                flags.append('multi_video')
            if other:
                flags.append('wrong_mouse_in_name')
            if not dl:
                flags.append('no_dlc')
            elif len(dl) > 1:
                flags.append('multi_dlc')
            if nrow > 0 and nfr > 0:
                if nrow < 0.99 * nfr:
                    flags.append('dlc_incomplete')
                elif abs(nrow - nfr) > 10:
                    flags.append('dlc_vs_video')
            if dl and cam not in nwb:
                flags.append('nwb_no_dlc')
            if cam in nwb and abs(nwb_frames - nwb_pulses) > fps:
                flags.append('nwb_frames_vs_pulses')
            rows.append(dict(session_id=s, mouse=mouse, day=desc, camera=cam, camera_freq=fps,
                             n_video_files=len(vids), video=os.path.basename(vids[0]) if vids else '',
                             video_frames=nfr, n_dlc_files=len(dl), dlc_rows=nrow, nwb_frames=nwb_frames,
                             nwb_pulses=nwb_pulses, flags=';'.join(flags)))
    except Exception as e:
        rows.append(dict(session_id=s, mouse=mouse, camera='', flags=f'ERROR {type(e).__name__}: {e}'))
    return rows


def main():
    files = sorted(p for p in glob.glob(str(NWB_DIR / '*.nwb')) if '_backup' not in p
                   and os.path.basename(p)[:2] in ('AB', 'MH'))
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(one, files) for r in rs]
    t = pd.DataFrame(rows)
    if os.path.exists(VERDICTS):
        v = pd.read_csv(VERDICTS)
        keep = [c for c in v.columns if c in ('session_id', 'camera', 'view', 'verdict')]
        if 'view' in keep:
            v = v.rename(columns={'view': 'camera'})
            keep = ['camera' if c == 'view' else c for c in keep]
        if {'session_id', 'camera', 'verdict'} <= set(keep):
            t = t.merge(v[['session_id', 'camera', 'verdict']].rename(columns={'verdict': 'alignment_verdict'}),
                        on=['session_id', 'camera'], how='left')
    t.to_csv(OUT / 'tables' / '018_video_dlc_audit.csv', index=False)
    issues = t[t['flags'].fillna('') != '']
    issues.to_csv(OUT / 'tables' / '018_video_dlc_issues.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    print('sessions with camera_flag 1:', t.session_id.nunique(), '| rows', len(t))
    flag_counts = issues['flags'].str.split(';').explode().value_counts()
    print(flag_counts.to_string())
    cols = ['session_id', 'day', 'camera', 'n_video_files', 'video_frames', 'n_dlc_files', 'dlc_rows', 'nwb_frames',
            'nwb_pulses', 'flags'] + (['alignment_verdict'] if 'alignment_verdict' in t else [])
    print(issues[cols].to_string(index=False))


if __name__ == '__main__':
    main()
