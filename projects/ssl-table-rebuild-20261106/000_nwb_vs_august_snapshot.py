"""000 -- Which sessions changed since the shared SSL tables were built? Compare the current NWB_ks4 session files with
the ssl_ephys 1.0.0 snapshot (built 2026-08-27 from NWB_ks4): trials (every common column), piezo / DLC lick events, unit
count, epochs, and session presence. Output: combined_results_ks4/ssl-table-rebuild-20261106/000_nwb_changes.csv
Run (haas): ~/code/ibl-ai-agent/.venv/bin/python 000_nwb_vs_august_snapshot.py
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
SNAP = os.path.expanduser('~/code/ibl-ai-agent/reports/datasets/ssl_ephys/1.0.0/metadata')
OUT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-table-rebuild-20261106'
RE = re.compile(r'^[A-Za-z]+\d+_\d{8}_\d{6}\.nwb$')
TR = pd.read_parquet(f'{SNAP}/trials.parquet')
EV = pd.read_parquet(f'{SNAP}/events.parquet')
SE = pd.read_parquet(f'{SNAP}/sessions.parquet').set_index('session_id')
EP = pd.read_parquet(f'{SNAP}/epochs.parquet')


def dec(x):
    return np.array([v.decode() if isinstance(v, bytes) else v for v in x], dtype=object) if x.dtype.kind in 'SO' else x


def one(sid):
    row = dict(session_id=sid)
    try:
        with h5py.File(f'{NWB}/{sid}.nwb', 'r') as f:
            old = TR[TR.session_id == sid].sort_values('start_time').reset_index(drop=True)
            if 'intervals/trials' in f:
                g = f['intervals/trials']
                new = pd.DataFrame({c: dec(g[c][()]) for c in old.columns if c in g and c != 'session_id'
                                    and g[c].ndim == 1 and g[c].shape[0] == g['start_time'].shape[0]})
                new = new.sort_values('start_time').reset_index(drop=True)
                row['n_trials_old'], row['n_trials_new'] = len(old), len(new)
                changed = []
                if len(old) == len(new):
                    for c in new.columns:
                        a, b = old[c].to_numpy(), new[c].to_numpy()
                        try:
                            a, b = a.astype(float), b.astype(float)
                            ok = np.allclose(a, b, atol=1e-4, equal_nan=True)
                        except (ValueError, TypeError):
                            ok = np.array_equal(a.astype(str), b.astype(str))
                        if not ok:
                            changed.append(c)
                row['trial_cols_changed'] = ';'.join(changed)
            for kind in ('piezo_lick_times', 'tongue_dlc_licks', 'jaw_dlc_licks'):
                p = f'processing/behavior/BehavioralEvents/{kind}/data'
                new_t = np.sort(f[p][()]) if p in f else np.array([])
                old_t = np.sort(EV[(EV.session_id == sid) & (EV.event_type == kind)].time.to_numpy())
                k = kind.split('_')[0]
                row[f'n_{k}_old'], row[f'n_{k}_new'] = len(old_t), len(new_t)
                if len(old_t) == len(new_t) and len(new_t):
                    row[f'{k}_max_diff_ms'] = float(np.abs(old_t - new_t).max() * 1e3)
            row['n_units_old'] = int(SE.n_units.get(sid, -1)) if sid in SE.index else -1
            row['n_units_new'] = int(f['units/id'].shape[0]) if 'units' in f else 0
            eo = EP[EP.session_id == sid].set_index('epoch_name')
            en = {}
            if 'processing/behavior/BehavioralEpochs' in f:
                for k, v in f['processing/behavior/BehavioralEpochs'].items():
                    if 'timestamps' in v and v['timestamps'].shape[0] >= 2:
                        en[k] = tuple(v['timestamps'][()][:2])
            row['epochs_old'] = ';'.join(sorted(eo.index))
            row['epochs_new'] = ';'.join(sorted(en))
            row['epochs_changed'] = (row['epochs_old'] != row['epochs_new']) or any(
                abs(eo.loc[k, 'start_time'] - en[k][0]) > 1e-3 or abs(eo.loc[k, 'stop_time'] - en[k][1]) > 1e-3
                for k in en if k in eo.index)
    except Exception as ex:  # noqa: BLE001
        row['error'] = f'{type(ex).__name__}: {ex}'[:200]
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    now = sorted(n[:-4] for n in os.listdir(NWB) if RE.match(n))
    old = set(SE.index)
    with ProcessPoolExecutor(24) as ex:
        rows = list(ex.map(one, now, chunksize=4))
    d = pd.DataFrame(rows)
    d['in_snapshot'] = d.session_id.isin(old)
    d.to_csv(f'{OUT}/000_nwb_changes.csv', index=False)
    print(f'{len(now)} session NWBs now, {len(old)} in the snapshot; new: {sorted(set(now) - old)}; '
          f'gone: {sorted(old - set(now))}')
    s = d[d.in_snapshot]
    print('errors:', s.error.notna().sum() if 'error' in s else 0)
    print('trial count changed:', s[s.n_trials_old != s.n_trials_new].session_id.tolist())
    tc = s[s.trial_cols_changed.fillna('') != '']
    print(f'trial columns changed in {len(tc)} sessions; by column:')
    print(tc.trial_cols_changed.str.split(';').explode().value_counts().to_string())
    for k in ('piezo', 'tongue', 'jaw'):
        ch = s[(s[f'n_{k}_old'] != s[f'n_{k}_new']) | (s[f'{k}_max_diff_ms'] > 1)]
        print(f'{k} licks changed (count, or times > 1 ms) in {len(ch)} sessions:')
        print(ch[['session_id', f'n_{k}_old', f'n_{k}_new', f'{k}_max_diff_ms']].to_string(index=False))
    print('unit count changed:', s[(s.n_units_old != s.n_units_new)][['session_id', 'n_units_old', 'n_units_new']].to_string(index=False))
    print('epochs changed:', int(s.epochs_changed.fillna(False).sum()))


if __name__ == '__main__':
    main()
