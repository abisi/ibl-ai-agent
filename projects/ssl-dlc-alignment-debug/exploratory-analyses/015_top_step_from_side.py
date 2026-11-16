"""015 -- Top-camera step clause from the top vs side nose-tip lag (014): sessions where the top camera drops a chunk of
frames once while the side camera stays aligned (user 2026-11-16: fix AB094_20231211 and AB087_20231017 top).

Per session: top and side nose-tip speed on the common 200 Hz grid of 014 (each camera's NWB timestamps); lag of each
segment pooled over all its bins (side bin = top bin + lag; lag > 0 = top frames stamped too early = top frames dropped);
the step bin maximising sum_{i<b} top[i] side[i + L0] + sum_{i>=b} top[i] side[i + L1] within the transition region from
the 014 windows; the bin converted to the top DLC frame stamped nearest to it. Current top pairing: stored top timestamps
-> raw top pulse index (010a cache), D0 = pulse index - frame. Clause: [(0, D0), (step frame, D0 + L1 - L0)].
Check plot: lag in 10 s windows around the step, before (NWB) and after (clause applied to the timestamps).
Outputs: drift_review/tables/top_step_<session>.csv, drift_review/figures/015_<session>.png; prints the clause.
Run (haas, ibl-ai-agent .venv, after 010a for the session): python 015_top_step_from_side.py SESSION:LO_BIN:HI_BIN ...
"""
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import correlate

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a014', HERE / '014_dataset_alignment_audit.py')
a014 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a014)
m006 = a014.m006
OUT = m006.OUT / 'drift_review'
CACHE = m006.OUT / 'cache'
FR = a014.FR


def grid_speeds(session_id):
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        n_s, n_t = len(b['jaw_angle']['data']), len(b['whisker_angle']['data'])
        ts = {'side': b['jaw_angle']['timestamps'][:n_s], 'top': b['whisker_angle']['timestamps'][:n_t]}
        v = {'side': a014.zs(a014.speed(b, 'nose_tip', len(ts['side']))),
             'top': a014.zs(a014.speed(b, 'top_nose_tip', len(ts['top'])))}
    t0 = min(np.nanmin(ts['side']), np.nanmin(ts['top']))
    ng = int((max(np.nanmax(ts['side']), np.nanmax(ts['top'])) - t0) * FR) + 2
    g = {}
    for view in ('side', 'top'):
        idx = np.round((ts[view] - t0) * FR)
        ok = np.isfinite(idx)
        arr = np.zeros(ng)
        arr[idx[ok].astype(int)] = v[view][ok]
        g[view] = arr
    return g, ts, t0


def pooled_lag(top, side, a, b, half=40, centre=0):
    lo = centre - half
    seg = top[a:b]
    c = correlate(side[a + lo:b + centre + half], seg, mode='valid', method='fft')
    return int(lo + np.argmax(c))


def lag_curve(top, side, centres, win=2000, half=60):
    out = []
    for c0 in centres:
        a, b = c0 - win // 2, c0 + win // 2
        if a - half < 0 or b + half > len(side):
            out.append(np.nan)
            continue
        c = correlate(side[a - half:b + half], top[a:b], mode='valid', method='fft')
        out.append(int(-half + np.argmax(c)))
    return np.array(out)


def main(arg):
    session_id, lo_bin, hi_bin = arg.split(':')
    lo_bin, hi_bin = int(lo_bin), int(hi_bin)
    g, ts, t0 = grid_speeds(session_id)
    top, side = g['top'], g['side']
    n = len(top)
    # pooled lags of the two segments (search +-40 bins around the 014 window lags)
    L0 = pooled_lag(top, side, 1000, lo_bin, centre=0)
    L1 = pooled_lag(top, side, hi_bin, n - 1000, centre=20)
    # step bin: maximise the two-segment score over [lo_bin, hi_bin)
    i = np.arange(lo_bin, hi_bin)
    pa = np.cumsum(top[i] * side[i + L0])
    pb = np.cumsum((top[i] * side[i + L1])[::-1])[::-1]
    score = np.r_[0, pa] + np.r_[pb, 0]
    k = int(np.argmax(score))
    step_bin = int(lo_bin + k)
    t_step = t0 + step_bin / FR
    f_step = int(np.nanargmin(np.abs(ts['top'] - t_step)))
    # current top pairing -> raw pulse index
    raw = np.load(CACHE / f'raw_pulses_{session_id}.npz')
    pulses = raw['top']
    j = np.searchsorted(pulses, ts['top'][:1000])
    j = np.clip(j, 1, len(pulses) - 1)
    j = np.where(np.abs(ts['top'][:1000] - pulses[j - 1]) <= np.abs(pulses[j] - ts['top'][:1000]), j - 1, j)
    D0 = int(np.median(j - np.arange(1000)))
    err = float(np.max(np.abs(pulses[j] - ts['top'][:1000])) * 1e3)
    clause = [(0, D0), (f_step, D0 + (L1 - L0))]
    # check: lag in 10 s windows around the step, NWB timestamps vs clause applied
    centres = np.arange(max(step_bin - 60000, 2000), min(step_bin + 60000, n - 2000), 1000)
    before = lag_curve(top, side, centres)
    t_new = ts['top'].copy()
    k1 = np.arange(len(t_new)) >= f_step
    pidx = np.arange(len(t_new)) + np.where(k1, clause[1][1], D0)
    okp = pidx < len(pulses)
    t_new = np.where(okp, pulses[np.minimum(pidx, len(pulses) - 1)], np.nan)
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b_ = f['processing/behavior/BehavioralTimeSeries']
        vt = a014.zs(a014.speed(b_, 'top_nose_tip', len(t_new)))
    idx = np.round((t_new - t0) * FR)
    ok = np.isfinite(idx) & (idx >= 0) & (idx < n)
    top2 = np.zeros(n)
    top2[idx[ok].astype(int)] = vt[ok]
    after = lag_curve(top2, side, centres)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot((centres - step_bin) / FR, before, '.', label='NWB now')
    ax.plot((centres - step_bin) / FR, after, '.', label='clause applied')
    ax.axvline(0, color='r', lw=0.6)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('time from step (s)')
    ax.set_ylabel('top vs side lag (frames, 10 s windows)')
    ax.set_title(f'{session_id} top: step at frame {f_step:,} (+{L1 - L0}); L0 {L0}, L1 {L1}, D0 {D0}', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'015_{session_id}.png', dpi=80)
    pd.DataFrame(dict(frame_start=[c[0] for c in clause], D=[c[1] for c in clause])).to_csv(
        OUT / 'tables' / f'top_step_{session_id}.csv', index=False)
    print(f"{session_id}: L0 {L0}, L1 {L1}, step bin {step_bin} (t {t_step:.2f} s), top frame {f_step:,}; "
          f"D0 {D0} (stored ts vs raw pulses max {err:.2f} ms); clause 'top': {clause}; "
          f"window lags before step now {np.nanmedian(before[centres < step_bin]):.0f}/after {np.nanmedian(before[centres > step_bin]):.0f}, "
          f"with clause {np.nanmedian(after[centres < step_bin]):.0f}/{np.nanmedian(after[centres > step_bin]):.0f}")


if __name__ == '__main__':
    for a in sys.argv[1:]:
        main(a)
