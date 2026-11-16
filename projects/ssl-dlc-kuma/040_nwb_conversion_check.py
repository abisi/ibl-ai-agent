"""NWB conversion with the new DLC files (CSPNeXt-M snapshot-330, 033): does the converter run exactly the same way?
(user 2026-11-11: "the number of frames never change and the alignment: start / end alignments, and other exceptions
should correctly be applied").

Per test session, the DLC step of the converter (NWB_converter converters/DLC_to_nwb.py convert_dlc_data, local repo
with all session exceptions) is run twice into an in-memory NWB file, with the same camera timestamps
(continuous_log_analysis.analyze_continuous_log, as in NWB_conversion_parallel; cached per session):
  old = the session's current *filtered.h5 (server_paths.get_dlc_file_path, the data folders),
  new = the staged new h5 (DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330/<session>/).
Only the DLC file list given to convert_dlc_data differs (get_dlc_file_path replaced in DLC_to_nwb for the run; the
data folders and NWB files are not touched). Checks per session:
  harness   old run == the NWB_ks4 file (every DLC series: data, timestamps, conversion; DLC events), i.e. the harness
            reproduces the real conversion (joined session AB155: timestamps shifted after saving, lengths only)
  frames    h5 rows old vs new per camera (equal unless the old DLC was incomplete)
  timing    per series: n timestamps and timestamps old == new (start / end alignment, piecewise mappings, dropped
            frames), conversion (px -> mm) old == new, converter messages (exception clauses applied) old == new
  orientation  top rotation decision (data-driven since 2026-11-04) old vs new, reported separately; NWBs of
            2026-11-03 rotated from a session list: top x / y also tested against the rotated harness output
  data      r(old, new) of the main series, DLC lick events old / new counts
Sessions: one or more per exception (nwb_session_exceptions.md): standard AB / MH, end-aligned, piecewise, unusable
start, top rotated 180 deg / setup 2 scale, camera timestamps from the log, corrected session config, joined runs,
pulses ending before the video, the MH007 0203 side video rerun.
Outputs: DLCProjects/merged_training_2026-11-09/nwb_conversion_check/ (series.csv, sessions.csv, messages/, cache/)
Run (local, conda env nwb): python 040_nwb_conversion_check.py [--sessions ...]
"""
import argparse
import contextlib
import datetime
import glob
import io
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ON_WINDOWS = os.name == 'nt'  # local Windows (conda env nwb) or haas (~/code/nwb_converter_env)
CONVERTER = r'C:\Users\bisi\Github\NWB_converter' if ON_WINDOWS else os.path.expanduser('~/code/NWB_converter_local')
sys.path.insert(0, CONVERTER)
os.chdir(CONVERTER)
from continuous_log_analysis import analyze_continuous_log  # noqa: E402
import converters.DLC_to_nwb as dlc_to_nwb  # noqa: E402
from utils.server_paths import get_dlc_file_path  # noqa: E402
from pynwb import NWBFile  # noqa: E402

AB = Path(r'M:\analysis\Axel_Bisi') if ON_WINDOWS else Path('/mnt/lsens-analysis/Axel_Bisi')
NWB_KS4 = AB / 'NWB_ks4'
STAGE = AB / 'DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330'
OUT = AB / 'DLCProjects/merged_training_2026-11-09/nwb_conversion_check'
SESSIONS = {
    'AB137_20241116_134442': 'standard AB',
    'AB150_20241220_082844': 'standard AB',
    'MH064_20260116_133002': 'standard MH',
    'MH070_20260121_140848': 'standard MH',
    'AB151_20250129_083713': 'AB151 (video / pulse check skipped in behaviour)',
    'MH013_20250218_110530': 'end-aligned side + top',
    'MH036_20250517_140456': 'end-aligned side + top',
    'AB127_20240823_094915': 'end-aligned top (video longer than pulses)',
    'AB153_20250209_153912': 'end-aligned side + top',
    'MH028_20250503_112316': 'end-aligned (camera test run)',
    'AB116_20240726_094206': 'end-aligned + unusable before frame 434,963',
    'MH013_20250219_170437': 'piecewise',
    'MH017_20250302_120918': 'piecewise',
    'MH004_20241206_093158': 'piecewise + setup 2 + top rotated',
    'MH004_20241205_115227': 'setup 2 + top rotated',
    'MH009_20250207_131953': 'setup 2 + top rotated (old top DLC failed)',
    'MH007_20250203_140412': 'setup 2 + top rotated + side video rerun (old side DLC on a broken file)',
    'MH038_20250524_105329': 'camera timestamps from the behaviour log',
    'MH037_20250524_143522': 'corrected session config (camera_flag)',
    'AB155_20250208_175103': 'joined runs',
    'AB085_20231005_152636': 'pulses end before the video (start-aligned)',
}
BTS, BEV = 'processing/behavior/BehavioralTimeSeries', 'processing/behavior/BehavioralEvents'
MAIN = ['jaw_angle', 'jaw_distance', 'pupil_area', 'nose_distance', 'tongue_distance', 'whisker_angle',
        'top_nose_distance']


def config_of(s):
    return AB / 'data' / s.split('_')[0] / s / f'config_{s}.yaml'


def camera_timestamps(s):
    f = OUT / 'cache' / f'{s}_camera_ts.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    ts, _, _ = analyze_continuous_log(config_file=str(config_of(s)), do_plot=False, plot_start=1, plot_stop=200,
                                      camera_filtering=False, experimenter='Axel_Bisi')
    ts = {k: ts[k] for k in ('cam1', 'cam2')}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(ts))
    return ts


def run_dlc_step(s, files, video_ts):
    """convert_dlc_data with the given DLC files -> {series: (data, timestamps, conversion)}, {event: timestamps}, log."""
    nwb = NWBFile(session_description='dlc check', identifier=s,
                  session_start_time=datetime.datetime.now(datetime.timezone.utc))
    original = dlc_to_nwb.get_dlc_file_path
    dlc_to_nwb.get_dlc_file_path = lambda config_file: [str(f) for f in files]
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            dlc_to_nwb.convert_dlc_data(nwb_file=nwb, config_file=str(config_of(s)), video_timestamps=video_ts,
                                        remove_extra_ts=False)
    finally:
        dlc_to_nwb.get_dlc_file_path = original
    beh = nwb.processing['behavior']
    series = {k: (np.asarray(v.data, dtype=float), np.asarray(v.timestamps), float(v.conversion))
              for k, v in beh['BehavioralTimeSeries'].time_series.items()}
    events = {k: np.asarray(v.timestamps) for k, v in beh['BehavioralEvents'].time_series.items()}
    return series, events, log.getvalue()


def nwb_dlc(s, names, event_names):
    with h5py.File(NWB_KS4 / f'{s}.nwb', 'r') as f:
        series = {k: (f[f'{BTS}/{k}/data'][:].astype(float), f[f'{BTS}/{k}/timestamps'][:],
                      float(f[f'{BTS}/{k}/data'].attrs['conversion'])) for k in names if k in f[BTS]}
        events = {k: f[f'{BEV}/{k}/timestamps'][:] for k in event_names if k in f[BEV]}
    return series, events


def same(a, b):
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9)


def h5_rows(f):
    return len(pd.read_hdf(f))


def check(s):
    old_files = get_dlc_file_path(str(config_of(s))) or []
    new_files = [str(f) for f in sorted((STAGE / s).glob('*_filtered.h5'))]
    # a camera without a new file (pilot folders: one camera) keeps its old file, reported in 'new_from_old'
    from_old = [f for f in old_files if not any(v in Path(f).name and any(v in Path(g).name for g in new_files)
                                                for v in ('side', 'top'))]
    new_files += from_old
    video_ts = camera_timestamps(s)
    o_ser, o_ev, o_log = run_dlc_step(s, old_files, video_ts)
    n_ser, n_ev, n_log = run_dlc_step(s, new_files, video_ts)
    w_ser, w_ev = nwb_dlc(s, o_ser, o_ev)
    (OUT / 'messages').mkdir(parents=True, exist_ok=True)
    (OUT / 'messages' / f'{s}.txt').write_text(f'== old\n{o_log}\n== new\n{n_log}', encoding='utf8')
    joined = s == 'AB155_20250208_175103'
    rows = []
    for k in sorted(set(o_ser) | set(n_ser)):
        o, n, w = o_ser.get(k), n_ser.get(k), w_ser.get(k)
        r = dict(session_id=s, series=k, view='top' if (k.startswith('top_') or 'whisker' in k) else 'side',
                 n_old=None if o is None else len(o[0]), n_new=None if n is None else len(n[0]),
                 n_ts_old=None if o is None else len(o[1]), n_ts_new=None if n is None else len(n[1]))
        if o is not None and n is not None:
            r['timestamps_equal'] = same(o[1], n[1])
            r['conversion_equal'] = o[2] == n[2]
            m = min(len(o[0]), len(n[0]))
            ok = np.isfinite(o[0][:m]) & np.isfinite(n[0][:m])
            r['r_old_new'] = np.corrcoef(o[0][:m][ok], n[0][:m][ok])[0, 1] if ok.sum() > 100 else np.nan
        if o is not None and w is not None:
            r['harness_data_equal_nwb'] = same(o[0], w[0])
            if not r['harness_data_equal_nwb'] and k.endswith(('_x', '_y')) and r['view'] == 'top':
                # NWBs of 2026-11-03 rotated the top keypoints from a session list (now decided from the data)
                r['harness_data_equal_nwb_rotated'] = same((639 if k.endswith('_x') else 479) - o[0], w[0])
            r['harness_ts_equal_nwb'] = len(o[1]) == len(w[1]) if joined else same(o[1], w[1])
            r['harness_conversion_equal_nwb'] = o[2] == w[2]
        rows.append(r)
    ev = {f'{k}_{src}': len(d.get(k, [])) for k in ('tongue_dlc_licks', 'jaw_dlc_licks')
          for src, d in (('old', o_ev), ('new', n_ev), ('nwb', w_ev))}
    ev.update({f'{k}_harness_equal_nwb': same(o_ev[k], w_ev[k]) for k in o_ev if k in w_ev})
    orient = lambda log: [m for m in log.splitlines() if 'orientation' in m or 'upside down' in m]  # noqa: E731
    msg = lambda log: [m for m in log.splitlines()  # noqa: E731
                       if (s in m or 'WARNING' in m) and m not in orient(log)]
    sess = dict(session_id=s, case=SESSIONS.get(s, ''), new_from_old=' | '.join(Path(f).name for f in from_old),
                old_files=' | '.join(Path(f).name for f in old_files),
                new_files=' | '.join(Path(f).name for f in new_files),
                **{f'rows_{v}_{src}': next((h5_rows(f) for f in fl if v in Path(f).name), None)
                   for v in ('side', 'top') for src, fl in (('old', old_files), ('new', new_files))},
                messages_equal=msg(o_log) == msg(n_log), messages_old=' / '.join(msg(o_log)),
                messages_new=' / '.join(msg(n_log)),
                orientation_old=' / '.join(orient(o_log)), orientation_new=' / '.join(orient(n_log)), **ev)
    return sess, rows


def main():
    global STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=list(SESSIONS))
    ap.add_argument('--stage', default=str(AB / 'DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330'), help='folder of the new h5 (<stage>/<session>/)')
    args = ap.parse_args()
    STAGE = Path(args.stage)
    OUT.mkdir(parents=True, exist_ok=True)
    sessions, series = [], []
    for s in args.sessions:
        print(f'--- {s} ({SESSIONS.get(s, "")})', flush=True)
        try:
            a, b = check(s)
        except Exception as e:  # noqa: BLE001  (report and continue with the next session)
            import traceback
            traceback.print_exc()
            a, b = dict(session_id=s, case=SESSIONS.get(s, ''), error=repr(e)), []
        sessions.append(a)
        series += b
        print({k: v for k, v in a.items() if not k.endswith('files')}, flush=True)
    pd.DataFrame(sessions).to_csv(OUT / 'sessions.csv', index=False)
    pd.DataFrame(series).to_csv(OUT / 'series.csv', index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
