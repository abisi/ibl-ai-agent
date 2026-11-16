"""010a -- Raw camera pulse trains and all piezo licks of the drift sessions, as the NWB converter extracts them
(NWB_converter continuous_log_analysis.analyze_continuous_log: behaviour log, or NIDQ for ephys sessions; NWB clock).

The NWB DLC timestamps of fixed sessions are already remapped (piecewise / end-aligned), and the NWB piezo licks of the
2023 AB sessions keep only trial-window licks, so 010 needs the raw trains to show the naive pairing and to estimate the
frame -> pulse offset with every lick.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/cache/raw_pulses_<session>.npz (top, side, licks, trial_ttl)
Run (haas, converter env, from ~/code/NWB_converter_local):
    ~/code/nwb_converter_env/bin/python <this file> SESSION_ID [SESSION_ID ...]
"""
import os
import sys
from pathlib import Path

import numpy as np

CONVERTER = Path.home() / 'code' / 'NWB_converter_local'
sys.path.insert(0, str(CONVERTER))
os.chdir(CONVERTER)
from continuous_log_analysis import analyze_continuous_log  # noqa: E402

ROOT = Path('/mnt/lsens-analysis/Axel_Bisi')
OUT = ROOT / 'combined_results_ks4' / 'ssl-dlc-alignment-debug' / 'cache'
# AB and MH rigs: cam1 = top camera, cam2 = side camera (NWB_converter converters/DLC_to_nwb.py)
VIEW_CAM = {'top': 'cam1', 'side': 'cam2'}


def onsets(v):
    a = np.asarray(v, dtype=float)
    return a[:, 0] if a.ndim == 2 else a


def main(session_id):
    config = ROOT / 'data' / session_id[:5] / session_id / f'config_{session_id}.yaml'
    ts, n_frames, _ = analyze_continuous_log(config_file=str(config), do_plot=False, camera_filtering=False)
    out = {view: onsets(ts[cam]) for view, cam in VIEW_CAM.items() if cam in ts}
    out['licks'] = np.sort(onsets(ts['lick_trace'])) if 'lick_trace' in ts else np.array([])
    out['trial_ttl'] = onsets(ts['trial_TTL']) if 'trial_TTL' in ts else np.array([])
    np.savez(OUT / f'raw_pulses_{session_id}.npz', **out)
    print(session_id, {k: len(v) for k, v in out.items()})


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:]:
        try:
            main(s)
        except Exception as ex:
            print(s, 'ERROR', type(ex).__name__, ex)
