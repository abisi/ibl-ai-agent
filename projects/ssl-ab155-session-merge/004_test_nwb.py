"""004 -- Test NWB of the joined AB155 day-0 sorting, read from SessionToConcat (testing only; the real NWB is made later
from the original paths with the joined-session clause in the converter).

NWB_converter as in the repo (converters.ephys_to_nwb_test.convert_ephys_recording), with two path functions redirected
at runtime (repo unchanged): get_imec_probe_folder_list -> SessionToConcat imec folders, get_sync_event_times_folder ->
SessionToConcat sync_event_times. Probe alignment: the IBL alignment of the original g0 session (channel_locations.json,
channel-indexed, same insertion) is copied into <joined imec>/ibl_format/ if absent.
Content: subject/session metadata (original yaml), units + electrodes of all valid probes (noise and void units dropped
by the converter), epochs part1 / part2 in joined NIDQ time (junction from AB155_g0_sc_offsets.txt), and the trial-start
TTL times. No behaviour, no DLC: those need the joined-session clause (part-1 end TTL, logs, real-time gap).

Output: <joined>/nwb_test/AB155_20250208_175103_joined_test.nwb
Run (local, conda env nwb): python 004_test_nwb.py
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\bisi\Github\NWB_converter')
os.chdir(r'C:\Users\bisi\Github\NWB_converter')
import converters.ephys_to_nwb_test as ephys  # noqa: E402
from converters.nwb_saving import save_nwb_file  # noqa: E402
from converters.subject_to_nwb import create_nwb_file  # noqa: E402

S = 'AB155_20250208_175103'
J = r'M:\analysis\Axel_Bisi\Misc\SessionToConcat\AB155_20250208_175103_joined'
RUN = os.path.join(J, 'Ephys', 'catgt_AB155_g0')
ORIG = rf'M:\analysis\Axel_Bisi\data\AB155\{S}'
CONFIG = os.path.join(ORIG, f'config_{S}.yaml')
OUT = os.path.join(J, 'nwb_test')


def junction_nidq():
    off = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(os.path.join(RUN, 'AB155_g0_sc_offsets.txt'))}
    return float(off['sec_nidq'][1])


def main():
    imecs = sorted(os.path.join(RUN, f) for f in os.listdir(RUN) if 'imec' in f and os.path.isdir(os.path.join(RUN, f)))
    for d in imecs:
        dst = os.path.join(d, 'ibl_format', 'channel_locations.json')
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(ORIG, 'Ephys', 'catgt_AB155_g0', os.path.basename(d), 'ibl_format',
                                      'channel_locations.json'), dst)
            print(f'copied original alignment -> {dst}')
    ephys.get_imec_probe_folder_list = lambda config_file, experimenter=None: imecs
    ephys.get_sync_event_times_folder = lambda config_file, experimenter=None: os.path.join(RUN, 'sync_event_times')

    nwb = create_nwb_file(config_file=CONFIG)
    ephys.convert_ephys_recording(nwb_file=nwb, config_file=CONFIG, experimenter='Axel_Bisi')

    tj = junction_nidq()
    t_end = max(max(float(np.max(x)) for x in nwb.units['spike_times'][:] if len(x)), tj)
    nwb.add_epoch(start_time=0.0, stop_time=tj, tags=['part1'])
    nwb.add_epoch(start_time=tj, stop_time=float(t_end), tags=['part2'])
    ts = np.loadtxt(os.path.join(RUN, 'sync_event_times', 'trial_start_times.txt'))
    from pynwb.misc import AnnotationSeries
    nwb.add_acquisition(AnnotationSeries(name='trial_start_ttl', data=np.array(['trial_start'] * len(ts)),
                                         timestamps=ts, description=f'TEST joined sorting (supercat g0 + g1 passive post): trial-start TTLs in joined NIDQ time, junction at {tj:.6f} s; part-2 times not shifted by the real gap'))
    os.makedirs(OUT, exist_ok=True)
    save_nwb_file(nwb_file=nwb, output_folder=OUT, with_time_string=False, suffix=None)
    src = os.path.join(OUT, f'{nwb.identifier}.nwb')
    dst = os.path.join(OUT, f'{S}_joined_test.nwb')
    if os.path.exists(src) and src != dst:
        os.replace(src, dst)
    print(f'saved {dst}; units {len(nwb.units)}; junction {tj:.6f} s; trial-start TTLs {len(ts)}')


if __name__ == '__main__':
    main()
