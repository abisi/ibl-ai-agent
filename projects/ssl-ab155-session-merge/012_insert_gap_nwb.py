"""012 -- Insert the real gap between the two runs into the full joined test NWB (010).
The joined clock (CatGT supercat, -supercat_trim_edges) puts part 2 (g1) straight after part 1 (g0). Here every time at
or after the junction is shifted by SHIFT = real start of g1 - joined start of g1, so times become physical (seconds from
the start of g0), and part 1 is unchanged.

SHIFT:
  - joined NIDQ time of part 1 = g0 NIDQ time (trial TTLs xa_1_0 identical, offset 0)
  - joined NIDQ time of part 2 = g1 NIDQ time + OFFSET_G1 (72/72 trial TTLs, spread 1 us)
  - real start of g1 relative to g0 = fileCreateTime_original(g1) - fileCreateTime_original(g0) (NIDQ meta, 1 s resolution)
  -> SHIFT = (t_g1 - t_g0) - OFFSET_G1 = real dead time between the runs + the trimmed g0 tail + the trimmed g1 head
  Cross-check: behaviour session folder names (start of each behaviour session) and their trial_time vs TTLs.
  Precision: +- 1 s (wall-clock timestamps are whole seconds). Applied as one constant to imec0-synced and NIDQ times
  (the clocks differ by < 1 ms at the junction).

Shifted (values >= TJ only): trials time columns, BehavioralEvents piezo_lick_times and the timestamps of every
TimeSeries / camera ImageSeries / BehavioralEvents, units spike_times, units useTheseTimesStart/Stop (Bombcell, strings).
BehavioralEpochs are recomputed with the converter's own get_passive_active_epoch_bounds on the shifted trials.
The gap is added as BehavioralEpochs IntervalSeries 'recording_gap' (excluded in presence / coverage ratio).
The supercat_junction annotation is replaced by 'recording_gap' (part 1 end, part 2 start).
Output: nwb_test/AB155_20250208_175103_joined_full_test_gap.nwb (the joined-clock file is kept).
Run (local, conda env nwb): python 012_insert_gap_nwb.py
"""
import datetime
import os
import shutil
import uuid
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\bisi\Github\NWB_converter')
from utils.behavior_converter_misc import get_passive_active_epoch_bounds  # noqa: E402

A = r'M:\analysis\Axel_Bisi'
J = rf'{A}\Misc\SessionToConcat\AB155_20250208_175103_joined'
SRC = rf'{J}\nwb_test\AB155_20250208_175103_joined_full_test.nwb'
DST = rf'{J}\nwb_test\AB155_20250208_175103_joined_full_test_gap.nwb'
G0 = rf'{A}\data\AB155\AB155_20250208_175103\Ephys\catgt_AB155_g0\AB155_g0_tcat.nidq'
G1 = rf'{A}\data\AB155\AB155_20250208_185559\Ephys\catgt_AB155_g1\AB155_g1_tcat.nidq'
JN = rf'{J}\Ephys\catgt_AB155_g0\AB155_g0_tcat.nidq'
TRAIN = r'M:\data\AB155\Training\AB155_20250208_175103'
TRIAL_TIME_COLS = ('start_time', 'stop_time', 'abort_window_start_time', 'abort_window_stop_time', 'auditory_stim_time',
                   'whisker_stim_time', 'no_stim_time', 'lick_time', 'response_window_start_time',
                   'response_window_stop_time', 'stim_onset', 'opto_stim_start_time', 'opto_stim_stop_time')


def meta(path):
    return dict(l.rstrip('\n').split('=', 1) for l in open(path) if '=' in l)


def compute_shift():
    m0, m1 = meta(G0 + '.meta'), meta(G1 + '.meta')
    t0 = datetime.datetime.fromisoformat(m0['fileCreateTime_original'])
    t1 = datetime.datetime.fromisoformat(m1['fileCreateTime_original'])
    dt_runs = (t1 - t0).total_seconds()
    # trial TTLs: the joined file holds all g0 TTLs, then all g1 TTLs
    c = np.loadtxt(JN + '.xa_1_0.txt')
    a0, a1 = np.loadtxt(G0 + '.xa_1_0.txt'), np.loadtxt(G1 + '.xa_1_0.txt')
    assert len(c) == len(a0) + len(a1)
    off = {}
    for name, d in (('g0', c[:len(a0)] - a0), ('g1', c[len(a0):] - a1)):
        assert np.ptp(d) < 1e-4, (name, np.ptp(d))
        off[name] = float(d.mean())
    assert abs(off['g0']) < 1e-6
    shift = dt_runs - off['g1']
    print(f'g0 start {t0}, g1 start {t1}: {dt_runs:.0f} s apart; g0 length {float(m0["fileTimeSecs"]):.3f} s '
          f'-> {dt_runs - float(m0["fileTimeSecs"]):.1f} s without recording')
    print(f'joined time of g1 t=0: {off["g1"]:.4f} s -> SHIFT {shift:.3f} s (+- 1 s)')
    # cross-check with the behaviour sessions (start = folder name time; trial_time relative to it)
    r1 = pd.read_csv(os.path.join(TRAIN, 'results_corrected_bis.csv'))
    r2 = pd.read_csv(os.path.join(TRAIN, 'results_2.csv'))
    b0 = np.loadtxt(G0 + '.xa_1_0.txt')[0] - r1.trial_time.iloc[0]
    b1 = np.loadtxt(G1 + '.xa_1_0.txt')[0] - r2.trial_time.iloc[0]
    beh_dt = (datetime.datetime(2025, 2, 8, 18, 55, 59) - datetime.datetime(2025, 2, 8, 17, 51, 3)).total_seconds()
    shift_beh = beh_dt + b0 - b1 - off['g1']
    print(f'cross-check from behaviour session start times: SHIFT {shift_beh:.3f} s (+- 1 s)')
    return shift, dt_runs, off['g1']


def shift_ds(ds, tj, shift):
    x = ds[()]
    if x.dtype.kind in 'SO':
        v = np.array([float(s.decode() if isinstance(s, bytes) else s) for s in x])
        m = v >= tj
        v[m] += shift
        ds[...] = np.array([repr(float(s)).encode() for s in v], dtype=object)
        return int(m.sum())
    m = np.isfinite(x) & (x >= tj)
    if m.any():
        x[m] += shift
        ds[...] = x
    return int(m.sum())


def main():
    shift, dt_runs, off_g1 = compute_shift()
    shutil.copyfile(SRC, DST)
    with h5py.File(DST, 'r+') as f:
        tj = float(f['acquisition/supercat_junction/timestamps'][0])
        log = []
        for c in TRIAL_TIME_COLS:
            if f'intervals/trials/{c}' in f:
                log.append((f'trials/{c}', shift_ds(f[f'intervals/trials/{c}'], tj, shift)))
        ts_paths = []
        f.visititems(lambda n, o: ts_paths.append(n) if isinstance(o, h5py.Dataset) and n.endswith('/timestamps')
                     and 'BehavioralEpochs' not in n and 'supercat_junction' not in n else None)
        for n in ts_paths:
            log.append((n, shift_ds(f[n], tj, shift)))
        log.append(('BehavioralEvents/piezo_lick_times/data',
                    shift_ds(f['processing/behavior/BehavioralEvents/piezo_lick_times/data'], tj, shift)))
        log.append(('units/spike_times', shift_ds(f['units/spike_times'], tj, shift)))
        for c in ('useTheseTimesStart', 'useTheseTimesStop'):
            log.append((f'units/{c}', shift_ds(f[f'units/{c}'], tj, shift)))
        for n, k in log:
            if k and 'BehavioralTimeSeries' not in n:
                print(f'  shifted {k:9d} values  {n}')
        nb = sum(k for n, k in log if 'BehavioralTimeSeries' in n)
        print(f'  shifted {nb:9d} values  BehavioralTimeSeries timestamps (all series)')
        # epochs from the shifted trials (converter rule)
        tr = f['intervals/trials']
        t = pd.DataFrame({'start_time': tr['start_time'][()], 'stop_time': tr['stop_time'][()],
                          'context': [x.decode() if isinstance(x, bytes) else x for x in tr['context'][()]]})
        bounds = get_passive_active_epoch_bounds(t)
        for name, (a, b) in bounds.items():
            ds = f[f'processing/behavior/BehavioralEpochs/{name}/timestamps']
            old = ds[()].copy()
            ds[...] = np.array([a, b])
            print(f'  epoch {name}: {old[0]:.3f}-{old[1]:.3f} -> {a:.3f}-{b:.3f}')
        # recording gap as a behavioural epoch (IntervalSeries like passive_pre / active / passive_post), to be
        # excluded from time-based metrics (e.g. unit presence / coverage ratio)
        ep = 'processing/behavior/BehavioralEpochs'
        f.copy(f'{ep}/passive_pre', f'{ep}/recording_gap')
        g = f[f'{ep}/recording_gap']
        g['timestamps'][...] = np.array([tj, tj + shift])
        g.attrs['object_id'] = str(uuid.uuid4())
        g.attrs['description'] = (
            'No recording: between the end of run g0 (part 1) and the start of run g1 (part 2, passive post). Exclude '
            'from time-based metrics (e.g. presence / coverage ratio, firing rates over the session).')
        print(f'  epoch recording_gap: {tj:.3f}-{tj + shift:.3f}')
        # junction annotation -> 'recording_gap' with two timestamps (h5py: pynwb append fails on the camera names)
        f.move('acquisition/supercat_junction', 'acquisition/recording_gap')
        g = f['acquisition/recording_gap']
        for name, val in (('data', np.array(['part 1 end (run g0, trimmed edge)',
                                             'part 2 start (run g1, passive post, trimmed edge)'], dtype=object)),
                          ('timestamps', np.array([tj, tj + shift]))):
            attrs = dict(g[name].attrs)
            del g[name]
            ds = g.create_dataset(name, data=val, dtype=h5py.string_dtype() if name == 'data' else 'f8')
            ds.attrs.update(attrs)
        g.attrs['description'] = (
            f'TEST joined session (CatGT supercat of runs g0 and g1, -supercat_trim_edges). Times are physical: seconds '
            f'from the start of g0. Sorting ran on the joined stream, where part 2 started at {tj:.6f} s (imec0); every '
            f'time from part 2 was shifted by {shift:.3f} s (+- 1 s, from the SpikeGLX file creation times: g1 started '
            f'{dt_runs:.0f} s after g0, g1 t=0 is at {off_g1:.4f} s in the joined NIDQ clock). No recording between '
            f'the two timestamps. Sorting: run 4819418 (DREDge + Kilosort 4 drift correction).')
    print(f'saved {DST}')


if __name__ == '__main__':
    main()
