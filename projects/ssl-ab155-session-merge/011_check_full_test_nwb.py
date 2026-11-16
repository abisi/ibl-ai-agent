"""011 -- Checks of the full joined test NWB (010) against the current (part-1-only) NWB_ks4 file.
  trials: count per part and type; part-1 trials identical to NWB_ks4 (start times, perf); TTL intervals vs results
    trial_time intervals in both parts; context / epochs
  licks: piezo lick times per part; trial lick_time present on hit / FA trials only
  cameras and DLC: frames vs timestamps per view, timestamps continuous across the junction, DLC rows of part 2
  units: count per probe; spike times span both parts; junction annotation
Run (haas): python 011_check_full_test_nwb.py
"""
import h5py
import numpy as np
import pandas as pd

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
NEW = f'{J}/nwb_test/AB155_20250208_175103_joined_full_test.nwb'
OLD = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4/AB155_20250208_175103.nwb'
TRAIN = '/mnt/lsens-data/AB155/Training/AB155_20250208_175103'


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def trials(f):
    t = f['intervals/trials']
    cols = [c for c in ('start_time', 'perf', 'trial_type', 'context', 'lick_flag', 'lick_time', 'whisker_stim',
                        'auditory_stim', 'reward_available') if c in t]
    d = {}
    for c in cols:
        v = t[c][:]
        d[c] = dec(v) if v.dtype.kind in 'SO' else v
    return pd.DataFrame(d)


def main():
    out = []
    with h5py.File(NEW, 'r') as fn, h5py.File(OLD, 'r') as fo:
        tj = float(fn['acquisition/supercat_junction/timestamps'][0])
        out.append(f'junction (imec0 time): {tj:.6f} s')
        tn, to = trials(fn), trials(fo)
        tn['part'] = np.where(tn.start_time < tj, 1, 2)
        out.append(f'trials: new {len(tn)} (part 1 {int((tn.part == 1).sum())}, part 2 {int((tn.part == 2).sum())}); '
                   f'old {len(to)}')
        p1 = tn[tn.part == 1].reset_index(drop=True)
        n = min(len(p1), len(to))
        out.append(f'part 1 vs old: start_time max |diff| {np.max(np.abs(p1.start_time[:n] - to.start_time[:n])):.6f} s; '
                   f'perf identical {bool(np.all(p1.perf[:n].to_numpy() == to.perf[:n].to_numpy()))}')
        r1 = pd.read_csv(f'{TRAIN}/results_corrected_bis.csv')
        r2 = pd.read_csv(f'{TRAIN}/results_2.csv')
        for part, r in ((1, r1), (2, r2)):
            s = tn[tn.part == part].start_time.to_numpy()
            k = min(len(s), len(r))
            err = np.max(np.abs(np.diff(s[:k]) - np.diff(r.trial_time.to_numpy()[:k])))
            out.append(f'part {part}: {len(s)} trials vs {len(r)} results rows; max |TTL - trial_time interval| '
                       f'{err * 1e3:.2f} ms; perf identical {bool(np.all(tn[tn.part == part].perf.to_numpy()[:k] == r.perf.to_numpy()[:k]))}')
        out.append('trial types per part:\n' + tn.groupby(['part', 'perf']).size().unstack(fill_value=0).to_string())
        if 'context' in tn:
            out.append('context per part:\n' + tn.groupby(['part', 'context']).size().unstack(fill_value=0).to_string())
        ep = fn['intervals']
        out.append('intervals tables: ' + ', '.join(ep.keys()))
        for name in ep:
            if name == 'trials':
                continue
            g = ep[name]
            if 'start_time' in g:
                st, sp_ = g['start_time'][:], g['stop_time'][:]
                tags = dec(g['tags'][:]) if 'tags' in g else ['']
                out.append(f'  {name}: ' + '; '.join(f'{a:.1f}-{b:.1f}' for a, b in zip(st, sp_))[:400])
        # licks
        be = fn['processing/behavior/BehavioralEvents']
        lk = be['piezo_lick_times/timestamps'][:].ravel()
        lk = lk[np.isfinite(lk)]
        out.append(f'piezo licks: part 1 {int((lk < tj).sum())}, part 2 {int((lk >= tj).sum())} '
                   f'(old NWB {len(fo["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:])})')
        hit = tn.perf.isin([2, 3, 5])
        out.append(f'lick_time finite: hits/FA {int(np.isfinite(tn.lick_time[hit]).sum())}/{int(hit.sum())}, '
                   f'other {int(np.isfinite(tn.lick_time[~hit]).sum())}/{int((~hit).sum())}')
        # cameras / DLC
        ts_grp = fn['processing/behavior/BehavioralTimeSeries']
        for marker in ('jaw_angle', 'whisker_angle'):
            if marker in ts_grp:
                g = ts_grp[marker]
                ts = g['timestamps'][:]
                d = np.diff(ts)
                gaps = np.flatnonzero(d > 0.05)
                out.append(f'{marker}: {g["data"].shape[0]} rows, {len(ts)} timestamps, {ts[0]:.3f}-{ts[-1]:.3f} s, '
                           f'median dt {np.median(d) * 1e3:.2f} ms, gaps > 50 ms at '
                           + ', '.join(f'{ts[i]:.2f} s ({d[i]:.3f} s)' for i in gaps[:5])
                           + f'; frames after junction {int((ts >= tj).sum())}')
        # units
        u = fn['units']
        st_idx = u['spike_times_index'][:]
        st = u['spike_times']
        n_units = len(st_idx)
        starts = np.r_[0, st_idx[:-1]]
        sample = np.linspace(0, n_units - 1, min(n_units, 200)).astype(int)
        span2 = np.mean([st[st_idx[i] - 1] > tj for i in sample if st_idx[i] > starts[i]])
        out.append(f'units: {n_units} (old {len(fo["units/spike_times_index"][:])}); fraction with spikes after the '
                   f'junction (200 sampled) {span2:.2f}')
        eg = fn['units/electrode_group'][:]
        names = pd.Series([fn[r].name.split('/')[-1] for r in eg]).value_counts().sort_index()
        out.append('units per probe: ' + ', '.join(f'{k}: {v}' for k, v in names.items()))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
