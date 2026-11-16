"""009 -- Reconvert MH028_20250503_112316 with its DLC end-aligned (NWB_converter converters/DLC_to_nwb.py
end_aligned_dlc_sessions, side and top; user 2026-11-09), then check the new NWB against the previous one.

Cause: the camera sent a test run of 2,677 (side) / 2,218 (top) pulses plus 1 stray pulse (0-17.5 s) before the video;
the last pulse run (from 51.97 s) has 511,279 pulses = 511,279 frames on both cameras. Before the fix the jaw lagged the
piezo licks by ~2,690 frames (13.5 s).

Steps: previous NWB_ks4 file -> <session>_backup_20261109.nwb (globs keep only MOUSE_YYYYMMDD_HHMMSS.nwb); convert;
checks: DLC data length = timestamps, first DLC timestamp = start of the last pulse run, lick-locked jaw opening at a
normal lag (MH028 0501 / 0502: jaw opens ~30 frames before piezo contact), trials / units / piezo licks unchanged.
Run (local, conda env nwb): python 009_reconvert_mh028_0503.py
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, r'C:\Users\bisi\Github\NWB_converter')
os.chdir(r'C:\Users\bisi\Github\NWB_converter')
import NWB_conversion_parallel as ncp  # noqa: E402

S = 'MH028_20250503_112316'
NWB_KS4 = r'M:\analysis\Axel_Bisi\NWB_ks4'
CONFIG = rf'M:\analysis\Axel_Bisi\data\MH028\{S}\config_{S}.yaml'
OUT = os.path.join(NWB_KS4, f'{S}.nwb')
BACKUP = os.path.join(NWB_KS4, f'{S}_backup_20261109.nwb')
BTS = 'processing/behavior/BehavioralTimeSeries'


def lick_lag(f, sig='jaw_distance'):
    """Best lag (frames) of the median jaw opening around piezo bout onsets, searched over +-3200 frames."""
    x = f[f'{BTS}/{sig}/data'][:].astype(float)
    ts = f[f'{BTS}/{sig}/timestamps'][:]
    x = np.nan_to_num(x - np.nanmedian(x))
    n = min(len(x), len(ts))
    lk = f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:]
    lk = lk[np.r_[True, np.diff(lk) > 0.5]]
    i0 = np.searchsorted(ts[:n], lk)

    def amp(k):
        i = i0 + k
        i = i[(i > 40) & (i < n - 40)]
        return np.median([x[j:j + 20].mean() - x[j - 40:j - 20].mean() for j in i])
    return max(((amp(k), k) for k in range(-3200, 401, 4)))


def summary(path):
    with h5py.File(path, 'r') as f:
        out = {'trials': f['intervals/trials/start_time'].shape[0],
               'units': f['units/id'].shape[0] if 'units' in f else 0,  # MH028 0503: no units in NWB_ks4
               'piezo': f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'].shape[0]}
        for view, sig in (('side', 'jaw_distance'), ('top', 'whisker_angle')):
            out[f'{view}_frames'] = f[f'{BTS}/{sig}/data'].shape[0]
            out[f'{view}_ts'] = f[f'{BTS}/{sig}/timestamps'].shape[0]
            out[f'{view}_t0'] = round(float(f[f'{BTS}/{sig}/timestamps'][0]), 3)
        out['jaw_lick_lag'] = lick_lag(f)
    return out


def main():
    if not os.path.exists(BACKUP):
        os.rename(OUT, BACKUP)
    if not os.path.exists(OUT):  # rerun of the checks only once converted
        ncp.convert_data_to_nwb(config_file=CONFIG, output_folder=NWB_KS4, with_time_string=False,
                                experimenter='Axel_Bisi')
    old, new = summary(BACKUP), summary(OUT)
    for k in new:
        print(f'{k:14s} old {old[k]}  new {new[k]}')
    assert new['side_frames'] == new['side_ts'] and new['top_frames'] == new['top_ts'], 'DLC length != timestamps'
    assert all(new[k] == old[k] for k in ('trials', 'units', 'piezo')), 'trials / units / piezo changed'
    print('OK' if -60 <= new['jaw_lick_lag'][1] <= 20 else 'CHECK jaw lick lag')


if __name__ == '__main__':
    sys.exit(main())
