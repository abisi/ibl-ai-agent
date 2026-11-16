"""018 -- Convert the joined AB155 day-0 session with NWB_converter as is (session exception utils/joined_sessions.py and
its hooks; no runtime overrides), from the merged data copied into data/AB155/AB155_20250208_175103 (009), to a test
folder, and compare with the validated test NWB (010 + 012, ..._joined_full_test_gap.nwb).
Checks: trials (all time columns), units (spike times per unit), licks, camera / DLC timestamps and rows, epochs, the
recording_gap epoch and annotation.
Output: SessionToConcat/AB155_20250208_175103_joined/nwb_test/converter_clause/AB155_20250208_175103.nwb
Run (local, conda env nwb): python 018_converter_clause_test.py
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, r'C:\Users\bisi\Github\NWB_converter')
os.chdir(r'C:\Users\bisi\Github\NWB_converter')
import NWB_conversion_parallel as ncp  # noqa: E402

S = 'AB155_20250208_175103'
J = r'M:\analysis\Axel_Bisi\Misc\SessionToConcat\AB155_20250208_175103_joined'
OUT = os.path.join(J, 'nwb_test', 'converter_clause')
REF = os.path.join(J, 'nwb_test', f'{S}_joined_full_test_gap.nwb')
CONFIG = rf'M:\analysis\Axel_Bisi\data\AB155\{S}\config_{S}.yaml'


def unit_rows(f):
    """Units keyed by (probe, cluster_id): row index and spike train (the probe folder order may differ)."""
    u = f['units']
    key = [(f[r].name.split('/')[-1].split('_')[0], int(c)) for r, c in zip(u['electrode_group'][()], u['cluster_id'][()])]
    idx = u['spike_times_index'][()]
    st = u['spike_times'][()]
    return key, {k: st[(idx[i - 1] if i else 0):idx[i]] for i, k in enumerate(key)}


def compare_units(a, b):
    bad = []
    ka, sa = unit_rows(a)
    kb, sb = unit_rows(b)
    if sorted(ka) != sorted(kb):
        return [f'units: different (probe, cluster) sets ({len(ka)} vs {len(kb)})']
    order = [ka.index(k) for k in kb]
    n = sum(len(sa[k]) != len(sb[k]) or not np.allclose(sa[k], sb[k], atol=1e-6) for k in kb)
    if n:
        bad.append(f'units: {n} spike trains differ')
    for c, ds in b['units'].items():
        if not isinstance(ds, h5py.Dataset) or c.endswith('_index') or c in ('spike_times', 'electrode_group', 'electrodes', 'id'):
            continue
        x, y = a['units'][c][()], ds[()]
        if len(x) == len(ka):
            x = x[order]
        same = (np.allclose(x, y, atol=1e-6, equal_nan=True) if x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu'
                else np.array_equal(np.asarray(x).astype(str), np.asarray(y).astype(str)))
        if not same:
            bad.append(f'units/{c}')
    return bad


def compare(new_path, ref_path):
    bad = []
    with h5py.File(new_path, 'r') as a, h5py.File(ref_path, 'r') as b:
        def walk(name, o):
            if not isinstance(o, h5py.Dataset) or name.startswith(('general', 'specifications', 'file_create_date', 'units')):
                return
            if name not in b:
                bad.append(f'only in new: {name}')
                return
            x, y = o[()], b[name][()]
            if getattr(x, 'dtype', None) is not None and x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
                if x.shape != y.shape:
                    bad.append(f'shape {name}: {x.shape} vs {y.shape}')
                elif not np.allclose(x, y, atol=1e-6, equal_nan=True):
                    bad.append(f'values {name}: max |diff| {np.nanmax(np.abs(x - y)):.3g}')
            elif not np.array_equal(np.asarray(x).astype(str), np.asarray(y).astype(str)):
                if name.split('/')[-1] not in ('description', 'session_description', 'identifier'):
                    bad.append(f'values {name}')
        a.visititems(walk)
        b.visititems(lambda n, o: bad.append(f'only in reference: {n}') if isinstance(o, h5py.Dataset) and n not in a
                     and not n.startswith(('general', 'specifications')) else None)
        bad += compare_units(a, b)
        print(f'trials {len(a["intervals/trials/start_time"])}, units {len(a["units/id"])}, '
              f'spikes {len(a["units/spike_times"])}, licks '
              f'{len(a["processing/behavior/BehavioralEvents/piezo_lick_times/data"])}')
        for e in ('passive_pre', 'active', 'passive_post', 'recording_gap'):
            print(f'  epoch {e}: {a[f"processing/behavior/BehavioralEpochs/{e}/timestamps"][()].round(3)}')
        print(f'  annotation recording_gap: {a["acquisition/recording_gap/timestamps"][()].round(3)}')
    print(f'{len(bad)} differences with the validated test NWB')
    for x in bad[:60]:
        print('  ', x)


def main():
    os.makedirs(OUT, exist_ok=True)
    ncp.convert_data_to_nwb(config_file=CONFIG, output_folder=OUT, with_time_string=False, experimenter='Axel_Bisi')
    compare(os.path.join(OUT, f'{S}.nwb'), REF)


if __name__ == '__main__':
    main()
