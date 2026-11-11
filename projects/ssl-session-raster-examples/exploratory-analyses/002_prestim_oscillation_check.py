"""Quick check of the rhythmic population bursts before the stimulus in the AB149_20241217 example (t = 1317.78 s):
dominant frequency, participating area groups, and state (whisking, snout, pupil) during the bursts.

Usage: python 002_prestim_oscillation_check.py [session_id t_stim]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('r001', HERE / '001_paper_style_whisker_trial_raster.py')
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)

SID = sys.argv[1] if len(sys.argv) > 1 else 'AB149_20241217_113458'
T_STIM = float(sys.argv[2]) if len(sys.argv) > 2 else 1317.780384
BIN = 0.005
PRE = (-3.0, 0.0)   # pre-stimulus segment analysed (s)

units_all, _ = r1.r0.valid_sessions()
d = r1.load_session(SID, units_all)
ut = d['unit_table']


def pop_rate(st_list, a, b):
    edges = np.arange(a, b + BIN, BIN)
    return np.sum([np.histogram(st, edges)[0] for st in st_list], 0) / BIN / max(len(st_list), 1)


def peak_freq(x):
    f, p = welch(x - x.mean(), fs=1 / BIN, nperseg=min(len(x), 256))
    m = (f >= 2) & (f <= 30)
    return f[m][np.argmax(p[m])], p[m].max() / np.median(p[m])


a, b = T_STIM + PRE[0], T_STIM + PRE[1]
rate = pop_rate(list(ut['spike_times']), a, b)
f0, snr = peak_freq(rate)
print(f'{SID} pre-stimulus {PRE}: population peak {f0:.1f} Hz (peak / median power {snr:.1f})')
print('per area group (n units, peak Hz, peak/median power, coefficient of variation of the group rate):')
for g, sub in ut.groupby('area_group'):
    if len(sub) < 20:
        continue
    r = pop_rate(list(sub['spike_times']), a, b)
    fg, sg = peak_freq(r)
    print(f'  {g:28s} {len(sub):4d}  {fg:5.1f}  {sg:5.1f}  {r.std() / r.mean():.2f}')

# Same measure over the whole session in 3 s windows before every trial, vs pupil and whisking
tt = d['trial_table']
tt = tt[tt['context'] == 'active']
allst = list(ut['spike_times'])
ts_p, pupil = d['series']['pupil_area']
ts_w, wh = d['series']['whisker_angle']
rows = []
for t0 in tt['start_time'].to_numpy():
    r = pop_rate(allst, t0 + PRE[0], t0 + PRE[1])
    f, s = peak_freq(r)
    mp = (ts_p >= t0 + PRE[0]) & (ts_p < t0)
    mw = (ts_w >= t0 + PRE[0]) & (ts_w < t0)
    rows.append((f, s, np.nanmedian(pupil[mp]) if mp.any() else np.nan,
                 np.nanstd(wh[mw]) if mw.any() else np.nan))
rows = np.array(rows)
osc = (rows[:, 1] > 8) & (rows[:, 0] >= 5) & (rows[:, 0] <= 12)
print(f'active trials: {len(rows)}; pre-stimulus 5-12 Hz peak with peak/median power > 8: {osc.sum()}'
      f' (frequency median {np.nanmedian(rows[osc, 0]):.1f} Hz)')
for lab, m in [('with oscillation', osc), ('without', ~osc)]:
    print(f'  {lab:17s} pupil median {np.nanmedian(rows[m, 2]):.3f} mm2, whisker angle SD median '
          f'{np.nanmedian(rows[m, 3]):.2f} deg')
