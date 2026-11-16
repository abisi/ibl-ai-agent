"""Full NWB test reconversion with the new DLC files (user 2026-11-11: "make sure that all exceptions including AB155
will run smoothly"), before NWB_ks4 is reconverted.

Per session: NWB_conversion_parallel.convert_data_to_nwb (local NWB_converter repo with all session exceptions and the
*_filtered_new.h5 rule in utils/server_paths.get_dlc_file_path) into a TEST folder (NWB_ks4 is not touched), the
converter output saved per session, then the test NWB compared with the session's current NWB_ks4 file:
  unchanged by the DLC switch: trials table (every column), units (ids, spike counts), piezo licks, acquisition
  (video links: names, timestamps), epochs / intervals, recording-gap annotation (AB155)
  DLC: the series set, per series n data / n timestamps / first and last timestamp vs NWB_ks4, timestamps equal on the
  common frames, conversion; DLC events (tongue / jaw licks) counts; the converter's DLC file choice and exception
  messages.
Sessions: the 040 set (one or more per exception: end-aligned, piecewise, unusable start, rotated top / setup 2, camera
timestamps from the log, corrected config, joined runs AB155, pulses ending before the video, MH007 side rerun) minus
AB137 1116 (no Ephys folder, not convertible), plus two sessions without old DLC (AB096 0221 13-10, AB135 1111).
Outputs: M:/analysis/Axel_Bisi/NWB_ks4_dlc_test/ (test NWBs, logs/<session>.log, check_sessions.csv, check_series.csv)
Run (local, conda env nwb): python 044_test_reconversion.py [--sessions ...] [--compare-only]
"""
import argparse
import contextlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ON_WINDOWS = os.name == 'nt'  # local Windows (conda env nwb) or haas (~/code/nwb_converter_env)
CONVERTER = r'C:\Users\bisi\Github\NWB_converter' if ON_WINDOWS else os.path.expanduser('~/code/NWB_converter_local')
sys.path.insert(0, CONVERTER)
os.chdir(CONVERTER)
import NWB_conversion_parallel as ncp  # noqa: E402

AB = Path(r'M:\analysis\Axel_Bisi') if ON_WINDOWS else Path('/mnt/lsens-analysis/Axel_Bisi')
NWB_KS4 = AB / 'NWB_ks4'
OUT = AB / 'NWB_ks4_dlc_test'
spec = importlib.util.spec_from_file_location('c040', Path(__file__).with_name('040_nwb_conversion_check.py'))
c040 = importlib.util.module_from_spec(spec)
sys.argv, _argv = sys.argv[:1], sys.argv
spec.loader.exec_module(c040)
sys.argv = _argv
SESSIONS = {k: v for k, v in c040.SESSIONS.items() if k != 'AB137_20241116_134442'}
SESSIONS.update({'AB096_20240221_131022': 'no old DLC (new only)', 'AB135_20241111_172907': 'no old DLC (new only)'})
BTS, BEV = 'processing/behavior/BehavioralTimeSeries', 'processing/behavior/BehavioralEvents'


def convert(s):
    log = OUT / 'logs' / f'{s}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            ncp.convert_data_to_nwb(config_file=str(c040.config_of(s)), output_folder=str(OUT), with_time_string=False,
                                    experimenter='Axel_Bisi')
            print('CONVERSION OK')
        except Exception:  # noqa: BLE001  (logged, next session)
            traceback.print_exc()
            print('CONVERSION FAILED')
    return log.read_text(encoding='utf8', errors='replace')


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        return bool(np.allclose(a.astype(float), b.astype(float), equal_nan=True, rtol=0, atol=1e-9))
    return bool(np.array_equal(a, b))


def compare(s, log):
    new_p, old_p = OUT / f'{s}.nwb', NWB_KS4 / f'{s}.nwb'
    r = dict(session_id=s, case=SESSIONS.get(s, ''), converted='CONVERSION OK' in log, nwb_written=new_p.exists())
    lines = log.splitlines()
    r['dlc_files'] = ' / '.join(m for m in lines if 'new DLC files for' in m or 'No DLC file' in m)
    r['messages'] = ' / '.join(m for m in lines if (s in m and ('aligned' in m or 'piecewise' in m or 'setup 2' in m
                                                                  or 'rotated' in m or 'orientation' in m))
                               or m.startswith('WARNING'))
    series = []
    if not new_p.exists():
        return r, series
    with h5py.File(new_p, 'r') as n, h5py.File(old_p, 'r') as o:
        tr_n, tr_o = set(n['intervals/trials']), set(o['intervals/trials'])
        r['trials_columns_equal'] = tr_n == tr_o
        r['trials_equal'] = all(same(n[f'intervals/trials/{k}'][:], o[f'intervals/trials/{k}'][:])
                                for k in tr_n & tr_o if isinstance(n[f'intervals/trials/{k}'], h5py.Dataset)
                                and n[f'intervals/trials/{k}'].dtype.kind != 'O')
        r['n_units_new'] = n['units/id'].shape[0] if 'units' in n else 0
        r['n_units_old'] = o['units/id'].shape[0] if 'units' in o else 0
        if 'units' in n and 'units' in o:
            r['spike_times_equal'] = same(n['units/spike_times'][:], o['units/spike_times'][:])
        pz = f'{BEV}/piezo_lick_times/timestamps'
        r['piezo_equal'] = (pz in n and pz in o and same(n[pz][:], o[pz][:])) or (pz not in n and pz not in o)
        r['acquisition_equal'] = sorted(n['acquisition']) == sorted(o['acquisition']) and all(
            same(n[f'acquisition/{k}/timestamps'][:], o[f'acquisition/{k}/timestamps'][:])
            for k in n['acquisition'] if 'timestamps' in n[f'acquisition/{k}'])
        r['intervals_equal'] = sorted(n['intervals']) == sorted(o['intervals'])
        names_n = set(n[BTS]) if BTS in n else set()
        names_o = set(o[BTS]) if BTS in o else set()
        r['dlc_series_new'], r['dlc_series_old'] = len(names_n), len(names_o)
        r['dlc_series_set_equal'] = names_n == names_o
        for k in sorted(names_n | names_o):
            d = dict(session_id=s, series=k)
            for tag, f in (('new', n), ('old', o)):
                if k in (names_n if tag == 'new' else names_o):
                    d[f'n_data_{tag}'] = f[f'{BTS}/{k}/data'].shape[0]
                    t = f[f'{BTS}/{k}/timestamps'][:]
                    d[f'n_ts_{tag}'] = len(t)
                    d[f't0_{tag}'], d[f't1_{tag}'] = (float(t[0]), float(t[-1])) if len(t) else (np.nan, np.nan)
                    d[f'conversion_{tag}'] = float(f[f'{BTS}/{k}/data'].attrs['conversion'])
            if k in names_n and k in names_o:
                tn, to = n[f'{BTS}/{k}/timestamps'][:], o[f'{BTS}/{k}/timestamps'][:]
                m = min(len(tn), len(to))
                d['ts_common_max_abs_diff_ms'] = float(np.max(np.abs(tn[:m] - to[:m])) * 1000) if m else np.nan
            series.append(d)
        for ev in ('tongue_dlc_licks', 'jaw_dlc_licks'):
            for tag, f in (('new', n), ('old', o)):
                p = f'{BEV}/{ev}/timestamps'
                r[f'{ev}_{tag}'] = f[p].shape[0] if p in f else None
    return r, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=list(SESSIONS))
    ap.add_argument('--compare-only', action='store_true')
    ap.add_argument('--tag', default='', help='suffix of the check tables (parallel workers)')
    ap.add_argument('--out', default=None, help='output folder of the test NWBs (default NWB_ks4_dlc_test)')
    ap.add_argument('--jobs', type=int, default=1, help='parallel conversions (haas: ~100)')
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.jobs > 1 and not args.compare_only:  # conversions in parallel processes, then the comparisons
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend='loky')(delayed(convert)(s) for s in args.sessions)
        args.compare_only = True
    rows, series = [], []
    if args.compare_only and args.jobs > 1:  # comparisons in parallel processes (reads only)
        from joblib import Parallel, delayed

        def one(s):
            log_p = OUT / 'logs' / f'{s}.log'
            try:
                return compare(s, log_p.read_text(encoding='utf8', errors='replace') if log_p.exists() else '')
            except Exception as err:  # noqa: BLE001  (reported in the table, the other sessions go on)
                return dict(session_id=s, compare_error=repr(err)), []
        for a, b in Parallel(n_jobs=args.jobs, backend='loky')(delayed(one)(s) for s in args.sessions):
            rows.append(a)
            series += b
        pd.DataFrame(rows).to_csv(OUT / f'check_sessions{args.tag}.csv', index=False)
        pd.DataFrame(series).to_csv(OUT / f'check_series{args.tag}.csv', index=False)
        print('saved', OUT)
        return
    for s in args.sessions:
        print(f'--- {s} ({SESSIONS.get(s, "")})', flush=True)
        log_p = OUT / 'logs' / f'{s}.log'
        log = log_p.read_text(encoding='utf8', errors='replace') if args.compare_only and log_p.exists() else convert(s)
        a, b = compare(s, log)
        rows.append(a)
        series += b
        print({k: v for k, v in a.items() if k not in ('messages', 'dlc_files')}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / f'check_sessions{args.tag}.csv', index=False)
        pd.DataFrame(series).to_csv(OUT / f'check_series{args.tag}.csv', index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
