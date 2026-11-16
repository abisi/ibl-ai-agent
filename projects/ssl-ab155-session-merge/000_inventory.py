"""000 -- Read-only inventory of the two AB155 day-0 runs before the merge (see question.md).

Per run (current CatGT outputs catgt_AB155_g0 / catgt_AB155_g1, and the April 2025 supercat copy in lsens-data):
SpikeGLX meta (file creation time, duration, rates, saved channels) of the NIDQ and each probe; NIDQ edge files
(*.nidq.xa_<ch>_*.txt): number of edges, first / last time, per channel. Gap between the end of g0 and the start of g1
from fileCreateTime + fileTimeSecs. Behaviour: results.csv / results_corrected_bis.csv / results_2.csv trial counts by
context and perf; log_continuous.bin duration; video frames vs camera edges.
Run (haas): python 000_inventory.py
"""
import datetime
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

A = '/mnt/lsens-analysis/Axel_Bisi/data/AB155'
RAW = '/mnt/lsens-data/AB155/Recording/AB155_20250208_175103'
RUNS = {'g0': f'{A}/AB155_20250208_175103/Ephys/catgt_AB155_g0', 'g1': f'{A}/AB155_20250208_185559/Ephys/catgt_AB155_g1',
        'old_supercat': f'{RAW}/Ephys/AB155_g0'}
CHANNELS = {0: 'sync', 1: 'trial start', 2: 'auditory', 3: 'whisker', 4: 'valve', 5: 'camera 0', 6: 'camera 1', 7: 'piezo'}


def meta(path):
    d = {}
    for line in open(path):
        if '=' in line:
            k, v = line.rstrip('\n').split('=', 1)
            d[k.lstrip('~')] = v
    return d


def main():
    pd.set_option('display.width', 250)
    starts = {}
    for run, d in RUNS.items():
        print(f'\n===== {run}: {d}')
        nm = glob.glob(f'{d}/*nidq.meta')
        if nm:
            m = meta(nm[0])
            t0 = datetime.datetime.fromisoformat(m['fileCreateTime'])
            starts[run] = (t0, float(m['fileTimeSecs']))
            print(f"nidq: created {m['fileCreateTime']}, {float(m['fileTimeSecs']):.3f} s, rate {m.get('niSampRate')}, "
                  f"firstSample {m.get('firstSample')}, nSavedChans {m.get('nSavedChans')}")
        for pm in sorted(glob.glob(f'{d}/*imec*/*tcat.imec*.ap.meta')):
            m = meta(pm)
            print(f"  {os.path.basename(pm)}: {float(m['fileTimeSecs']):.3f} s, rate {m.get('imSampRate')}, "
                  f"created {m.get('fileCreateTime')}")
        rows = []
        for f in sorted(glob.glob(f'{d}/*nidq.xa_*.txt') + glob.glob(f'{d}/*nidq.xd_*.txt')):
            mm = re.search(r'\.(xa|xd)_(\d+)_([\d.]+)\.txt$', f)
            x = np.loadtxt(f, ndmin=1) if os.path.getsize(f) else np.array([])
            ch = int(mm.group(2))
            rows.append(dict(file=os.path.basename(f), channel=ch, meaning=CHANNELS.get(ch, '?'), n_edges=len(x),
                             first_s=x[0] if len(x) else np.nan, last_s=x[-1] if len(x) else np.nan))
        if rows:
            print(pd.DataFrame(rows).round(3).to_string(index=False))
    if 'g0' in starts and 'g1' in starts:
        (t0, d0), (t1, d1) = starts['g0'], starts['g1']
        gap = (t1 - t0).total_seconds() - d0
        print(f'\ng0 start {t0}, duration {d0:.1f} s; g1 start {t1}, duration {d1:.1f} s; gap between runs {gap:.1f} s')

    print('\n===== behaviour')
    for f in ['results.csv', 'results_corrected_bis.csv', 'results_2.csv']:
        r = pd.read_csv(f'{RAW}/{f}')
        ctx = r.context_block.astype(str) if 'context_block' in r else pd.Series(['?'] * len(r))
        print(f'{f}: {len(r)} trials, trial_time {r.trial_time.min():.1f} .. {r.trial_time.max():.1f} s; context '
              f'{ctx.value_counts().to_dict()}; perf {r.perf.value_counts().sort_index().to_dict()}')
    size = os.path.getsize(f'{RAW}/log_continuous.bin')
    for nch in (5, 6, 7, 8):
        print(f'log_continuous.bin: {size} bytes = {size / (nch * 8 * 5000):.1f} s if {nch} float64 channels at 5 kHz')
    for v in sorted(glob.glob(f'{RAW}/Video/*.avi')):
        c = cv2.VideoCapture(v)
        print(f'video {os.path.basename(v)}: {int(c.get(cv2.CAP_PROP_FRAME_COUNT))} frames')


if __name__ == '__main__':
    main()
